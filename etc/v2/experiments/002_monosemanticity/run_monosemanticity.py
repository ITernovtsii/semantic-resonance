#!/usr/bin/env python3
"""Experiment 002: Compare monosemanticity of cosine vs linear routers.

Sub-A: Generate blinded human evaluation CSVs with PMI-ranked tokens per expert.
Sub-B: Word-level routing coherence across carrier sentences.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 experiments/002_monosemanticity/run_monosemanticity.py
    CUDA_VISIBLE_DEVICES=1 python3 experiments/002_monosemanticity/run_monosemanticity.py --sub-a-only
    CUDA_VISIBLE_DEVICES=1 python3 experiments/002_monosemanticity/run_monosemanticity.py --sub-b-only
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.analyze_extended import collect_routing_data
from scripts.analyze import get_embeddings, calc_internal_cohesion, calc_external_purity
from src.models.csr import ChamberOfSemanticResonance
from src.utils.config_utils import load_config, ConfigDict
from src.models.sra import SRA
from tokenizers import Tokenizer


def _convert_stacked_to_modulelist(state_dict):
    """Convert stacked expert params (expert_w1) back to nn.ModuleList format.

    The current csr.py uses nn.ModuleList[MicroExpert], but some checkpoints
    (e.g., StdMoE) were saved with stacked nn.Parameter format.
    """
    import re
    has_stacked = any('expert_w1' in k for k in state_dict)
    has_modulelist = any('experts.0.linear1.weight' in k for k in state_dict)
    if has_modulelist or not has_stacked:
        return state_dict  # Already in correct format

    logging.info("Converting stacked expert params → nn.ModuleList format")
    new_sd = {}
    name_map = {
        'expert_w1': ('linear1', 'weight'),
        'expert_b1': ('linear1', 'bias'),
        'expert_w2': ('linear2', 'weight'),
        'expert_b2': ('linear2', 'bias'),
    }
    pattern = re.compile(r'^(.+\.ff_or_moe)\.(expert_[wb][12])$')
    for k, v in state_dict.items():
        m = pattern.match(k)
        if m:
            prefix = m.group(1)
            param_name = m.group(2)
            layer, param = name_map[param_name]
            # v is (num_experts, ...) — split into individual expert params
            for i in range(v.size(0)):
                new_sd[f"{prefix}.experts.{i}.{layer}.{param}"] = v[i]
        else:
            new_sd[k] = v
    return new_sd


def load_model_from_checkpoint(checkpoint_path, config_path=None):
    """Load model, handling both old (nn.ModuleList) and new (stacked) checkpoint formats.

    The current csr.py uses nn.ModuleList[MicroExpert], so:
    - Old format checkpoints load directly
    - New stacked format checkpoints get converted back to nn.ModuleList keys
    """
    logging.info(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'config' in checkpoint:
        config = ConfigDict(checkpoint['config'])
    elif config_path:
        config = ConfigDict(load_config(config_path))
    else:
        raise RuntimeError("No config in checkpoint and no --config provided")

    model = SRA(config.model, config.data.vocab_size)
    sd = checkpoint['model_state_dict']
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    # Convert stacked→modulelist if needed (model expects nn.ModuleList)
    sd = _convert_stacked_to_modulelist(sd)
    # Fix temperature shape
    for k in list(sd.keys()):
        if k.endswith('.temperature') and sd[k].dim() == 0:
            sd[k] = sd[k].unsqueeze(0)
    model.load_state_dict(sd)
    for module in model.modules():
        if hasattr(module, '_needs_batch_seed'):
            module._needs_batch_seed = False
    model.eval()
    return model, config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─── Configuration ──────────────────────────────────────────────────────────

SEED_CHECKPOINTS = {
    19: {
        'SRA-B': 'outputs/1SRA/outputs_wt103_256exp_d256_s6m_k14/best_model.pt',
        'StdMoE-B': 'outputs/1SRA/outputs_wt103_256exp_d256_s6m_k14_stdmoe/best_model.pt',
    },
    42: {
        'SRA-B': 'experiments/001_multi_seed/outputs/SRA_B_seed42/best_model.pt',
        'StdMoE-B': 'experiments/001_multi_seed/outputs/StdMoE_B_seed42/best_model.pt',
    },
    137: {
        'SRA-B': 'experiments/001_multi_seed/outputs/SRA_B_seed137/best_model.pt',
        'StdMoE-B': 'experiments/001_multi_seed/outputs/StdMoE_B_seed137/best_model.pt',
    },
}

# Defaults — overridden by --seed in main()
CHECKPOINTS = SEED_CHECKPOINTS[19]

BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
OUTPUT_DIR = BASE_OUTPUT_DIR
SUBEXP_A_DIR = os.path.join(OUTPUT_DIR, 'subexp_a')
SUBEXP_B_DIR = os.path.join(OUTPUT_DIR, 'subexp_b')
SUMMARY_DIR = os.path.join(OUTPUT_DIR, 'summary')

NUM_EXPERTS_SAMPLE = 100  # per layer
TOP_TOKENS = 50
MAX_BATCHES = 200  # for routing data collection
SEED = 42

# ─── Carrier Sentence Templates (Sub-B) ────────────────────────────────────

CARRIER_TEMPLATES = [
    # Sentence-internal (varied positions)
    "The {word} was discussed at the conference last week.",
    "We carefully studied the {word} in our analysis.",
    "The concept of {word} is important for understanding the field.",
    "Researchers have recently investigated {word} in several studies.",
    "A detailed report on {word} was published yesterday.",
    "The {word} remained a topic of debate among experts.",
    "Students were asked to explain the meaning of {word} clearly.",
    "New evidence about {word} has emerged from the laboratory.",
    # Sentence-initial and sentence-final (position variation per Codex)
    "{word} has been a major research topic for decades.",
    "The researchers concluded their study of {word}.",
]

# Minimum token count for PMI ranking (filters out rare-token noise per Gemini)
PMI_MIN_COUNT = 3

# ─── Word List for Sub-B ──────────────────────────────────────────────────

def build_word_list():
    """Multi-token words across semantic categories for routing coherence analysis."""
    words = {
        'scientific': [
            'photosynthesis', 'electromagnetic', 'mitochondria', 'thermodynamics',
            'semiconductor', 'biochemistry', 'neuroscience', 'astrophysics',
            'paleontology', 'crystallography', 'pharmacology', 'epidemiology',
            'spectroscopy', 'nanotechnology', 'geochemistry', 'bioinformatics',
            'microorganism', 'chromatography', 'immunology', 'radiotherapy',
        ],
        'geographic': [
            'Mediterranean', 'Constantinople', 'Massachusetts', 'Philadelphia',
            'Mesopotamia', 'Scandinavian', 'Afghanistan', 'Mozambique',
            'Saskatchewan', 'Transylvania', 'Yellowstone', 'Kilimanjaro',
            'archipelago', 'subcontinent', 'hemisphere', 'cartography',
            'Newfoundland', 'Appalachian', 'Azerbaijan', 'Liechtenstein',
        ],
        'abstract': [
            'consciousness', 'sustainability', 'infrastructure', 'globalization',
            'accountability', 'democratization', 'entrepreneurship', 'individualism',
            'philosophical', 'unprecedented', 'disproportionate', 'characterization',
            'conceptualization', 'fundamentalism', 'multidimensional', 'systematically',
            'transformation', 'vulnerabilities', 'authoritarianism', 'decentralization',
        ],
        'compound': [
            'thunderstorm', 'butterscotch', 'nevertheless', 'extraordinary',
            'counterproductive', 'straightforward', 'overwhelming', 'breakthrough',
            'misunderstanding', 'troubleshooting', 'championship', 'approximately',
            'disadvantage', 'overwhelming', 'spokesperson', 'unemployment',
            'underestimate', 'uncomfortable', 'understanding', 'communication',
        ],
        'technical': [
            'implementation', 'functionality', 'authentication', 'configuration',
            'documentation', 'optimization', 'compatibility', 'vulnerability',
            'infrastructure', 'microprocessor', 'interoperability', 'virtualization',
            'parallelization', 'synchronization', 'serialization', 'encapsulation',
            'polymorphism', 'multithreading', 'containerization', 'decompression',
        ],
    }

    # Ambiguous words with contrastive contexts
    ambiguous = [
        ('bank', [
            "The bank of the river was covered with wildflowers.",
            "The bank approved the mortgage application today.",
        ]),
        ('spring', [
            "The spring rain brought new flowers to the garden.",
            "The spring in the mechanism was tightly coiled.",
        ]),
        ('bat', [
            "The bat flew silently through the dark cave.",
            "He picked up the bat and walked to home plate.",
        ]),
        ('crane', [
            "A crane stood motionless in the shallow water.",
            "The construction crane lifted the heavy steel beam.",
        ]),
        ('bark', [
            "The bark of the ancient tree was rough and thick.",
            "We could hear the bark of the dog from outside.",
        ]),
        ('match', [
            "She lit a match to start the campfire.",
            "The championship match was watched by millions.",
        ]),
        ('pitch', [
            "The pitch of the violin was perfectly tuned.",
            "The pitcher threw a fastball pitch over the plate.",
        ]),
        ('ring', [
            "He placed the ring on her finger carefully.",
            "The phone continued to ring in the empty room.",
        ]),
        ('scale', [
            "The fish had a beautiful iridescent scale.",
            "The scale indicated the package weighed five pounds.",
        ]),
        ('seal', [
            "The seal swam gracefully through the cold water.",
            "He pressed the seal into the warm red wax.",
        ]),
    ]

    # Nonce words (random letter combinations, BPE-valid)
    nonce = [
        'blorpington', 'snazzlewort', 'frimbulation', 'gloptersteen',
        'quazziferous', 'drimblethorpe', 'fluxomatic', 'gruntleberry',
        'splandiforium', 'wibblezatch',
    ]

    return words, ambiguous, nonce


def _encode_word_in_context(tokenizer, word):
    """Encode a word as it would appear in a sentence (with leading space).

    BPE tokenizers produce different token IDs for standalone 'word' vs
    in-context ' word' (Ġ prefix). We encode "the word" and strip the
    prefix token(s) for "the" to get the correct in-context token IDs.

    Returns (token_ids, tokens, n_tokens) for the word as it appears mid-sentence.
    """
    # Encode "the WORD" — "the" is a single token in all BPE vocabs
    probe = f"the {word}"
    encoded_probe = tokenizer.encode(probe)
    # Encode just "the" to find prefix length
    encoded_prefix = tokenizer.encode("the")
    prefix_len = len(encoded_prefix.ids)

    # The word's tokens are everything after the prefix
    word_ids = encoded_probe.ids[prefix_len:]
    word_tokens = encoded_probe.tokens[prefix_len:]

    return word_ids, word_tokens, len(word_ids)


def verify_word_list(tokenizer):
    """Verify words tokenize into 2-5 subtokens, filter those that don't."""
    words, ambiguous, nonce = build_word_list()
    verified = {}
    stats = defaultdict(int)

    for category, word_list in words.items():
        verified[category] = []
        for word in word_list:
            word_ids, word_tokens, n_tokens = _encode_word_in_context(tokenizer, word)
            if 2 <= n_tokens <= 7:  # Allow up to 7 for very long words
                verified[category].append({
                    'word': word,
                    'token_ids': word_ids,
                    'tokens': word_tokens,
                    'n_tokens': n_tokens,
                })
                stats[n_tokens] += 1
            else:
                logging.debug(f"Skipping '{word}': {n_tokens} tokens")

    # Verify nonce words
    verified_nonce = []
    for word in nonce:
        word_ids, word_tokens, n_tokens = _encode_word_in_context(tokenizer, word)
        if n_tokens >= 2:
            verified_nonce.append({
                'word': word,
                'token_ids': word_ids,
                'tokens': word_tokens,
                'n_tokens': n_tokens,
            })

    total = sum(len(v) for v in verified.values())
    logging.info(f"Verified {total} words + {len(verified_nonce)} nonce words")
    logging.info(f"Token count distribution: {dict(stats)}")

    return verified, ambiguous, verified_nonce


# ─── Sub-A: Expert Token Matrix + PMI ──────────────────────────────────────

def build_expert_token_matrix(topk_indices, token_ids, num_experts, vocab_size):
    """Build (E, V) count matrix of tokens routed to each expert."""
    E = num_experts
    K = topk_indices.size(1)
    counts = np.zeros((E, vocab_size), dtype=np.float64)

    indices_np = topk_indices.numpy()
    tids_np = token_ids.numpy()

    for ki in range(K):
        expert_ids = indices_np[:, ki]
        np.add.at(counts, (expert_ids, tids_np), 1)

    return counts


def compute_pmi(expert_token_counts):
    """Compute pointwise mutual information for each (expert, token) pair.

    PMI(e, t) = log2(P(e,t) / (P(e) * P(t)))
    High PMI = token is characteristic of expert (beyond frequency).
    """
    E, V = expert_token_counts.shape
    total = expert_token_counts.sum()
    if total == 0:
        return np.zeros_like(expert_token_counts)

    # P(e, t)
    p_joint = expert_token_counts / total
    # P(e) = sum over tokens
    p_expert = expert_token_counts.sum(axis=1, keepdims=True) / total  # (E, 1)
    # P(t) = sum over experts
    p_token = expert_token_counts.sum(axis=0, keepdims=True) / total   # (1, V)

    # Avoid division by zero
    denom = p_expert * p_token
    mask = (p_joint > 0) & (denom > 0)

    pmi = np.zeros_like(expert_token_counts)
    pmi[mask] = np.log2(p_joint[mask] / denom[mask])

    return pmi


def stratified_expert_sample(expert_token_counts, n_sample=100, seed=42):
    """Sample experts stratified by utilization quartile.

    Returns list of expert indices (sorted for reproducibility).
    """
    rng = np.random.RandomState(seed)
    total_per_expert = expert_token_counts.sum(axis=1)  # (E,)
    E = len(total_per_expert)

    # Only sample from active experts
    active = np.where(total_per_expert > 0)[0]
    if len(active) <= n_sample:
        return sorted(active.tolist())

    # Compute quartiles on active experts
    utilizations = total_per_expert[active]
    quartiles = np.percentile(utilizations, [25, 50, 75])

    q1 = active[utilizations <= quartiles[0]]
    q2 = active[(utilizations > quartiles[0]) & (utilizations <= quartiles[1])]
    q3 = active[(utilizations > quartiles[1]) & (utilizations <= quartiles[2])]
    q4 = active[utilizations > quartiles[2]]

    per_q = n_sample // 4
    remainder = n_sample - per_q * 4

    quartile_lists = [q1, q2, q3, q4]
    sampled = []
    shortfall = 0
    for i, q in enumerate(quartile_lists):
        target = per_q + (1 if i < remainder else 0) + shortfall
        actual = min(target, len(q))
        shortfall = target - actual  # accumulate total deficit
        sampled.extend(rng.choice(q, actual, replace=False).tolist())

    # Backfill any remaining shortfall from unsampled active experts
    if len(sampled) < n_sample:
        remaining = set(active.tolist()) - set(sampled)
        need = n_sample - len(sampled)
        if remaining:
            sampled.extend(rng.choice(list(remaining), min(need, len(remaining)),
                                       replace=False).tolist())

    return sorted(sampled)


# Anonymous model codes for blinded filenames (reviewer can't identify model)
_MODEL_BLIND_CODES = {}
_BLIND_CODE_COUNTER = [0]


def _get_blind_code(model_label):
    """Assign a stable anonymous code to each model label."""
    if model_label not in _MODEL_BLIND_CODES:
        _BLIND_CODE_COUNTER[0] += 1
        _MODEL_BLIND_CODES[model_label] = f'model_{chr(64 + _BLIND_CODE_COUNTER[0])}'
    return _MODEL_BLIND_CODES[model_label]


def generate_human_eval_csv(model_label, layer_name, layer_idx,
                            expert_token_counts, pmi_matrix, sampled_experts,
                            ic_scores, ep_scores, id_to_token, output_dir, seed=42):
    """Generate blinded and analysis CSV files for human evaluation."""
    rng = random.Random(seed + layer_idx)

    rows_blind = []
    rows_analysis = []

    for expert_id in sampled_experts:
        counts = expert_token_counts[expert_id]
        total = counts.sum()
        unique = int((counts > 0).sum())

        if total == 0:
            continue

        # Top-50 by PMI (with minimum frequency filter to avoid rare-token noise)
        pmi_scores = pmi_matrix[expert_id]
        # Only consider tokens routed at least PMI_MIN_COUNT times
        active_tokens = np.where(counts >= PMI_MIN_COUNT)[0]
        if len(active_tokens) == 0:
            # Fallback: use any token that was routed
            active_tokens = np.where(counts > 0)[0]
        if len(active_tokens) == 0:
            continue

        pmi_active = pmi_scores[active_tokens]
        top_indices = np.argsort(pmi_active)[::-1][:TOP_TOKENS]
        top_token_ids = active_tokens[top_indices]

        top_tokens_str = '; '.join(
            f"{id_to_token.get(int(tid), f'[{tid}]')}({pmi_scores[tid]:.2f})"
            for tid in top_token_ids
        )

        # Mass in top-50
        top50_mass = counts[top_token_ids].sum() / total * 100

        # Vocab entropy
        p = counts / total
        p_nz = p[p > 0]
        entropy = -(p_nz * np.log(p_nz)).sum()

        # IC/EP (may be None if not computed for this expert)
        ic = ic_scores.get(expert_id, '')
        ep = ep_scores.get(expert_id, '')

        rows_blind.append({
            'expert_id': expert_id,  # Will be anonymized below
            'total_tokens_routed': int(total),
            'unique_tokens_routed': unique,
            'top_50_tokens': top_tokens_str,
            'top50_mass_pct': f'{top50_mass:.1f}',
            'vocab_entropy': f'{entropy:.3f}',
            'human_purity_score': '',
            'human_category_label': '',
            'human_syntactic_or_semantic': '',
        })

        rows_analysis.append({
            'original_expert_id': expert_id,
            'model': model_label,
            'layer': layer_idx,
            'IC': f'{ic:.4f}' if isinstance(ic, float) else '',
            'EP': f'{ep:.4f}' if isinstance(ep, float) else '',
            'total_tokens_routed': int(total),
            'unique_tokens_routed': unique,
            'vocab_entropy': f'{entropy:.3f}',
            'top50_mass_pct': f'{top50_mass:.1f}',
        })

    # Randomize order for blind version
    indices = list(range(len(rows_blind)))
    rng.shuffle(indices)
    rows_blind = [rows_blind[i] for i in indices]
    rows_analysis = [rows_analysis[i] for i in indices]

    # Anonymize expert IDs in blind version
    for i, row in enumerate(rows_blind):
        row['expert_id'] = f'E{i:03d}'

    if not rows_blind:
        logging.warning(f"No expert rows for {model_label} layer {layer_idx}")
        return None, None

    # Save blind CSV (anonymous filename — no model identity)
    blind_code = _get_blind_code(model_label)
    blind_path = os.path.join(output_dir, f'{blind_code}_layer{layer_idx}_blind.csv')
    with open(blind_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows_blind[0].keys())
        writer.writeheader()
        writer.writerows(rows_blind)

    # Save analysis CSV (includes model identity — NOT for reviewer)
    analysis_path = os.path.join(output_dir, f'{model_label}_layer{layer_idx}_analysis.csv')
    with open(analysis_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows_analysis[0].keys())
        writer.writeheader()
        writer.writerows(rows_analysis)

    logging.info(f"Saved {len(rows_blind)} experts to {blind_path}")

    # Save blind-to-model mapping (for post-hoc unblinding)
    mapping_path = os.path.join(output_dir, '_blind_mapping.json')
    mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path) as f:
            mapping = json.load(f)
    mapping[blind_code] = model_label
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)

    return blind_path, analysis_path


# ─── Sub-A: IC/EP Computation ──────────────────────────────────────────────

def compute_ic_ep_for_experts(model, sampled_experts, layer_name,
                              expert_token_counts, word_embeds, id_to_token, k=20):
    """Compute IC and EP for sampled experts using embedding-space neighbors."""
    # Get anchors
    _, anchors_dict = get_embeddings(model)
    if layer_name not in anchors_dict:
        logging.warning(f"No anchors for {layer_name}")
        return {}, {}

    anchors = anchors_dict[layer_name]
    anchors_norm = F.normalize(anchors.float(), p=2, dim=-1)
    embeds_norm = F.normalize(word_embeds.float(), p=2, dim=-1)
    sim = torch.matmul(anchors_norm, embeds_norm.T)

    ic_scores = {}
    ep_scores = {}

    for expert_id in sampled_experts:
        # Top-K by embedding similarity to anchor
        top_ids = torch.topk(sim[expert_id], k).indices.tolist()
        ic = calc_internal_cohesion(top_ids, word_embeds)
        ic_scores[expert_id] = ic

        top_strs = [id_to_token.get(tid, '?') for tid in top_ids]
        ep = calc_external_purity(top_strs)
        ep_scores[expert_id] = ep

    return ic_scores, ep_scores


# ─── Sub-B: Routing Coherence ──────────────────────────────────────────────

def compute_routing_coherence(model, tokenizer, device, verified_words, nonce_words, ambiguous_words):
    """Run carrier sentences and measure subtoken routing coherence."""
    id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}

    # Set K=4 for analysis
    for module in model.modules():
        if isinstance(module, ChamberOfSemanticResonance):
            module.set_top_k(4)

    # Register hooks
    routing_data = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, output):
            if isinstance(output, tuple) and len(output) > 1:
                aux = output[1]
                routing_data[name] = {
                    'probs': aux['routing_probs'].detach().cpu(),
                    'indices': aux['topk_indices'].detach().cpu(),
                }
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, ChamberOfSemanticResonance):
            hooks.append(module.register_forward_hook(make_hook(name)))

    layer_names = sorted([n for n, m in model.named_modules()
                          if isinstance(m, ChamberOfSemanticResonance)])

    results = []

    # Process all word categories
    all_words = []
    for category, word_list in verified_words.items():
        for w in word_list:
            all_words.append({**w, 'category': category})
    for w in nonce_words:
        all_words.append({**w, 'category': 'nonce'})

    for word_info in tqdm(all_words, desc="Computing routing coherence"):
        word = word_info['word']
        n_tokens = word_info['n_tokens']
        category = word_info['category']

        for template_id, template in enumerate(CARRIER_TEMPLATES):
            sentence = template.format(word=word)
            encoded = tokenizer.encode(sentence)
            input_ids = encoded.ids

            # Find where the target word's subtokens appear in the sentence.
            # Use per-template local vars to avoid leaking across iterations.
            word_token_ids = word_info['token_ids']
            match_subtokens = word_info['tokens']  # local copy per template
            word_start = None
            n_tok = len(word_token_ids)
            for i in range(len(input_ids) - n_tok + 1):
                if input_ids[i:i + n_tok] == word_token_ids:
                    word_start = i
                    break

            # Fallback: try standalone encoding (sentence-initial, no Ġ prefix)
            if word_start is None:
                standalone = tokenizer.encode(word)
                alt_ids = standalone.ids
                alt_n = len(alt_ids)
                for i in range(len(input_ids) - alt_n + 1):
                    if input_ids[i:i + alt_n] == alt_ids:
                        word_start = i
                        n_tok = alt_n
                        match_subtokens = standalone.tokens
                        break

            if word_start is None:
                continue

            # Forward pass
            routing_data.clear()
            with torch.no_grad():
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
                model({'input_ids': input_tensor})

            # Extract routing for target subtokens per layer
            for li, layer_name in enumerate(layer_names):
                if layer_name not in routing_data:
                    continue
                indices = routing_data[layer_name]['indices']  # (S, K)
                n_seq = min(indices.size(0), len(input_ids))

                # Get expert assignments for target subtokens
                expert_ids = []
                topk_sets = []
                for offset in range(n_tok):
                    pos = word_start + offset
                    if pos >= n_seq:
                        break
                    primary = indices[pos, 0].item()
                    expert_ids.append(primary)
                    topk_sets.append(set(indices[pos].tolist()))

                if len(expert_ids) < 2:
                    continue

                # Primary coherence: fraction sharing most common top-1 expert
                from collections import Counter
                counts = Counter(expert_ids)
                most_common_count = counts.most_common(1)[0][1]
                primary_coherence = most_common_count / len(expert_ids)

                # Top-K overlap: average pairwise Jaccard
                jaccard_sum = 0
                n_pairs = 0
                for i in range(len(topk_sets)):
                    for j in range(i + 1, len(topk_sets)):
                        inter = len(topk_sets[i] & topk_sets[j])
                        union = len(topk_sets[i] | topk_sets[j])
                        jaccard_sum += inter / union if union > 0 else 0
                        n_pairs += 1
                topk_overlap = jaccard_sum / n_pairs if n_pairs > 0 else 0

                # Routing flicker: count expert switches between adjacent subtokens
                flicker = sum(1 for k in range(len(expert_ids) - 1)
                              if expert_ids[k] != expert_ids[k + 1])

                results.append({
                    'word': word,
                    'category': category,
                    'num_subtokens': n_tok,
                    'subtokens': '|'.join(match_subtokens),
                    'layer': li,
                    'template_id': template_id,
                    'expert_ids': ','.join(map(str, expert_ids)),
                    'primary_coherence': primary_coherence,
                    'topk_overlap': topk_overlap,
                    'routing_flicker': flicker,
                })

    for h in hooks:
        h.remove()

    # Process ambiguous words
    ambiguous_results = []
    hooks2 = []
    for name, module in model.named_modules():
        if isinstance(module, ChamberOfSemanticResonance):
            hooks2.append(module.register_forward_hook(make_hook(name)))

    for word, contexts in ambiguous_words:
        # Encode word in context (with leading space)
        word_ids, word_tokens, n_tok = _encode_word_in_context(tokenizer, word)

        for ctx_id, sentence in enumerate(contexts):
            encoded = tokenizer.encode(sentence)
            input_ids = encoded.ids

            # Find target word position (try in-context encoding, then standalone)
            word_start = None
            actual_n_tok = n_tok
            match_tokens = word_tokens  # local per-context
            for i in range(len(input_ids) - n_tok + 1):
                if input_ids[i:i + n_tok] == word_ids:
                    word_start = i
                    break
            if word_start is None:
                standalone = tokenizer.encode(word)
                alt_ids = standalone.ids
                alt_n = len(alt_ids)
                for i in range(len(input_ids) - alt_n + 1):
                    if input_ids[i:i + alt_n] == alt_ids:
                        word_start = i
                        actual_n_tok = alt_n
                        match_tokens = standalone.tokens
                        break
            if word_start is None:
                continue

            routing_data.clear()
            with torch.no_grad():
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
                model({'input_ids': input_tensor})

            for li, layer_name in enumerate(layer_names):
                if layer_name not in routing_data:
                    continue
                indices = routing_data[layer_name]['indices']
                n_seq = min(indices.size(0), len(input_ids))

                expert_ids = []
                topk_sets = []
                for offset in range(actual_n_tok):
                    pos = word_start + offset
                    if pos < n_seq:
                        expert_ids.append(indices[pos, 0].item())
                        topk_sets.append(set(indices[pos].tolist()))

                if len(expert_ids) < 1:
                    continue

                # Compute same metrics as main words (per Gemini feedback)
                primary_coherence = 0.0
                topk_overlap = 0.0
                flicker = 0
                if len(expert_ids) >= 2:
                    from collections import Counter
                    counts = Counter(expert_ids)
                    primary_coherence = counts.most_common(1)[0][1] / len(expert_ids)
                    jaccard_sum, n_pairs = 0, 0
                    for ii in range(len(topk_sets)):
                        for jj in range(ii + 1, len(topk_sets)):
                            inter = len(topk_sets[ii] & topk_sets[jj])
                            union = len(topk_sets[ii] | topk_sets[jj])
                            jaccard_sum += inter / union if union > 0 else 0
                            n_pairs += 1
                    topk_overlap = jaccard_sum / n_pairs if n_pairs > 0 else 0
                    flicker = sum(1 for k in range(len(expert_ids) - 1)
                                  if expert_ids[k] != expert_ids[k + 1])

                ambiguous_results.append({
                    'word': word,
                    'context_id': ctx_id,
                    'context': sentence,
                    'layer': li,
                    'expert_ids': ','.join(map(str, expert_ids)),
                    'n_tokens': actual_n_tok,
                    'tokens': '|'.join(match_tokens),
                    'primary_coherence': primary_coherence,
                    'topk_overlap': topk_overlap,
                    'routing_flicker': flicker,
                })

    for h in hooks2:
        h.remove()

    return results, ambiguous_results


# ─── Automated Metrics ──────────────────────────────────────────────────────

def compute_nmi(expert_token_counts):
    """Normalized mutual information I(Token; Expert) / sqrt(H(Token) * H(Expert))."""
    total = expert_token_counts.sum()
    if total == 0:
        return 0.0

    p_joint = expert_token_counts / total  # (E, V)
    p_expert = p_joint.sum(axis=1)  # (E,)
    p_token = p_joint.sum(axis=0)   # (V,)

    # H(Expert)
    pe_nz = p_expert[p_expert > 0]
    h_expert = -(pe_nz * np.log(pe_nz)).sum()

    # H(Token)
    pt_nz = p_token[p_token > 0]
    h_token = -(pt_nz * np.log(pt_nz)).sum()

    # I(Token; Expert) = sum p(e,t) log(p(e,t) / p(e)p(t))
    mi = 0.0
    nz = p_joint > 0
    if nz.any():
        denom = np.outer(p_expert, p_token)
        safe = nz & (denom > 0)
        mi = (p_joint[safe] * np.log(p_joint[safe] / denom[safe])).sum()

    norm = math.sqrt(h_expert * h_token) if h_expert > 0 and h_token > 0 else 1.0
    return float(mi / norm)


def compute_js_divergence_matrix(expert_token_counts, sampled_experts):
    """Pairwise JS divergence between sampled expert token distributions."""
    from scipy.spatial.distance import jensenshannon

    n = len(sampled_experts)
    js_matrix = np.zeros((n, n))

    # Normalize to distributions
    dists = []
    for e in sampled_experts:
        c = expert_token_counts[e]
        total = c.sum()
        if total > 0:
            dists.append(c / total)
        else:
            dists.append(np.ones(c.shape) / c.shape[0])

    for i in range(n):
        for j in range(i + 1, n):
            jsd = jensenshannon(dists[i], dists[j])
            js_matrix[i, j] = jsd
            js_matrix[j, i] = jsd

    mean_jsd = js_matrix[np.triu_indices(n, k=1)].mean()
    return mean_jsd, js_matrix


# ─── Main Orchestrator ──────────────────────────────────────────────────────

def run_sub_a(device):
    """Sub-Experiment A: Human evaluation CSVs."""
    os.makedirs(SUBEXP_A_DIR, exist_ok=True)

    for model_label, ckpt_path in CHECKPOINTS.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"Sub-A: Processing {model_label}")
        logging.info(f"{'='*60}")

        model, config = load_model_from_checkpoint(ckpt_path)

        # Set K=4
        for module in model.modules():
            if isinstance(module, ChamberOfSemanticResonance):
                module.set_top_k(4)

        model.to(device)

        # Load tokenizer
        tokenizer = Tokenizer.from_file(config.data.tokenizer_path)
        vocab = tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}
        vocab_size = config.data.vocab_size

        # Collect routing data
        routing = collect_routing_data(model, config, device, max_batches=MAX_BATCHES)

        # Get embeddings for IC/EP
        word_embeds, anchors_dict = get_embeddings(model)

        # Process each layer
        for layer_name in sorted(routing.keys()):
            layer_idx = int(layer_name.split('blocks.')[1].split('.')[0])
            logging.info(f"\n--- Layer {layer_idx} ({layer_name}) ---")

            data = routing[layer_name]
            topk_indices = data['topk_indices']
            token_ids = data['token_ids']
            num_experts = config.model.csr.num_experts

            # Build token matrix
            expert_token_counts = build_expert_token_matrix(
                topk_indices, token_ids, num_experts, vocab_size
            )

            # Compute PMI
            pmi_matrix = compute_pmi(expert_token_counts)

            # Stratified sample
            sampled = stratified_expert_sample(expert_token_counts, NUM_EXPERTS_SAMPLE, SEED)
            logging.info(f"Sampled {len(sampled)} experts (stratified by utilization)")

            # Compute IC/EP
            ic_scores, ep_scores = compute_ic_ep_for_experts(
                model, sampled, layer_name, expert_token_counts,
                word_embeds, id_to_token
            )

            # Generate CSVs
            generate_human_eval_csv(
                model_label, layer_name, layer_idx,
                expert_token_counts, pmi_matrix, sampled,
                ic_scores, ep_scores, id_to_token,
                SUBEXP_A_DIR, seed=SEED
            )

            # Automated metrics
            nmi = compute_nmi(expert_token_counts)
            mean_jsd, _ = compute_js_divergence_matrix(expert_token_counts, sampled)
            logging.info(f"  NMI(Token;Expert) = {nmi:.4f}")
            logging.info(f"  Mean JS Divergence = {mean_jsd:.4f}")

        # Free GPU memory
        model.cpu()
        del model
        torch.cuda.empty_cache()


def run_sub_b(device):
    """Sub-Experiment B: Word-level routing coherence."""
    os.makedirs(SUBEXP_B_DIR, exist_ok=True)

    # Load tokenizer from first checkpoint to verify words
    first_ckpt = list(CHECKPOINTS.values())[0]
    model, config = load_model_from_checkpoint(first_ckpt)
    tokenizer = Tokenizer.from_file(config.data.tokenizer_path)
    model.cpu()
    del model
    torch.cuda.empty_cache()

    # Verify word list
    verified_words, ambiguous_words, nonce_words = verify_word_list(tokenizer)

    # Save word list
    word_list_path = os.path.join(SUBEXP_B_DIR, 'word_list.csv')
    with open(word_list_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'category', 'n_tokens', 'tokens'])
        for cat, words in verified_words.items():
            for w in words:
                writer.writerow([w['word'], cat, w['n_tokens'], '|'.join(w['tokens'])])
        for w in nonce_words:
            writer.writerow([w['word'], 'nonce', w['n_tokens'], '|'.join(w['tokens'])])
    logging.info(f"Saved word list to {word_list_path}")

    all_coherence = []
    all_ambiguous = []

    for model_label, ckpt_path in CHECKPOINTS.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"Sub-B: Processing {model_label}")
        logging.info(f"{'='*60}")

        model, config = load_model_from_checkpoint(ckpt_path)
        model.to(device)

        results, ambig_results = compute_routing_coherence(
            model, tokenizer, device, verified_words, nonce_words, ambiguous_words
        )

        # Tag with model label
        for r in results:
            r['model'] = model_label
        for r in ambig_results:
            r['model'] = model_label

        all_coherence.extend(results)
        all_ambiguous.extend(ambig_results)

        model.cpu()
        del model
        torch.cuda.empty_cache()

    # Save coherence CSV
    coherence_path = os.path.join(SUBEXP_B_DIR, 'routing_coherence.csv')
    if all_coherence:
        with open(coherence_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_coherence[0].keys())
            writer.writeheader()
            writer.writerows(all_coherence)
        logging.info(f"Saved {len(all_coherence)} coherence rows to {coherence_path}")

    # Save ambiguous words CSV
    ambig_path = os.path.join(SUBEXP_B_DIR, 'ambiguous_routing.csv')
    if all_ambiguous:
        with open(ambig_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_ambiguous[0].keys())
            writer.writeheader()
            writer.writerows(all_ambiguous)
        logging.info(f"Saved {len(all_ambiguous)} ambiguous rows to {ambig_path}")

    # Print summary statistics
    print_coherence_summary(all_coherence)


def print_coherence_summary(results):
    """Print summary of routing coherence results."""
    if not results:
        return

    logging.info("\n" + "="*60)
    logging.info("ROUTING COHERENCE SUMMARY")
    logging.info("="*60)

    # Group by model and layer
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        key = (r['model'], r['layer'])
        grouped[key].append(r)

    for (model, layer), rows in sorted(grouped.items()):
        pc = [r['primary_coherence'] for r in rows]
        tk = [r['topk_overlap'] for r in rows]
        fl = [r['routing_flicker'] for r in rows]

        logging.info(f"\n{model} Layer {layer}:")
        logging.info(f"  Primary Coherence: {np.mean(pc):.4f} ± {np.std(pc):.4f}")
        logging.info(f"  Top-K Overlap:     {np.mean(tk):.4f} ± {np.std(tk):.4f}")
        logging.info(f"  Routing Flicker:   {np.mean(fl):.2f} ± {np.std(fl):.2f}")

        # By category
        cat_data = defaultdict(list)
        for r in rows:
            cat_data[r['category']].append(r['primary_coherence'])
        for cat, vals in sorted(cat_data.items()):
            logging.info(f"    {cat:12s}: {np.mean(vals):.4f} (n={len(vals)})")


def main():
    global CHECKPOINTS, OUTPUT_DIR, SUBEXP_A_DIR, SUBEXP_B_DIR, SUMMARY_DIR

    parser = argparse.ArgumentParser(description='Experiment 002: Monosemanticity comparison')
    parser.add_argument('--sub-a-only', action='store_true', help='Run only Sub-Experiment A')
    parser.add_argument('--sub-b-only', action='store_true', help='Run only Sub-Experiment B')
    parser.add_argument('--gpu', type=int, default=None, help='GPU index (default: auto)')
    parser.add_argument('--seed', type=int, default=19,
                        help='Training seed to analyze (selects checkpoints). Available: 19, 42, 137')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logging.error("CUDA not available. This experiment requires a GPU.")
        sys.exit(1)

    # Select checkpoints for requested seed
    if args.seed not in SEED_CHECKPOINTS:
        logging.error(f"No checkpoints defined for seed={args.seed}. Available: {list(SEED_CHECKPOINTS.keys())}")
        sys.exit(1)
    CHECKPOINTS = SEED_CHECKPOINTS[args.seed]

    # Verify checkpoints exist
    for label, path in CHECKPOINTS.items():
        if not os.path.exists(path):
            logging.error(f"Checkpoint not found for {label} seed={args.seed}: {path}")
            sys.exit(1)

    # Set output dirs — seed>19 goes into seed-specific subdirectory
    if args.seed != 19:
        OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f'seed{args.seed}')
    else:
        OUTPUT_DIR = BASE_OUTPUT_DIR
    SUBEXP_A_DIR = os.path.join(OUTPUT_DIR, 'subexp_a')
    SUBEXP_B_DIR = os.path.join(OUTPUT_DIR, 'subexp_b')
    SUMMARY_DIR = os.path.join(OUTPUT_DIR, 'summary')

    device = torch.device('cuda')
    logging.info(f"Using device: {device} ({torch.cuda.get_device_name()})")
    logging.info(f"Analyzing seed={args.seed} checkpoints")
    for label, path in CHECKPOINTS.items():
        logging.info(f"  {label}: {path}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    if args.sub_b_only:
        run_sub_b(device)
    elif args.sub_a_only:
        run_sub_a(device)
    else:
        run_sub_a(device)
        run_sub_b(device)

    logging.info(f"\nExperiment 002 (seed={args.seed}) complete!")


if __name__ == '__main__':
    main()
