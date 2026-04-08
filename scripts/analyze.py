# NUMBA_CACHE_DIR=/tmp/numba_cache python3 scripts/analyze.py --config configs/sra_wikitext103-ablation1-full-sre-base.yaml --checkpoint outputs_wt103-ablation/1-full/best_model.pt
# python3 scripts/analyze.py --config configs/sra_wikitext103-ablation2-noHSE.yaml --checkpoint outputs_wt103-ablation/2-noHSE/best_model.pt
# python3 scripts/analyze.py --config configs/sra_wikitext103-ablation3-no-dispersion.yaml --checkpoint outputs_wt103-ablation/3-noDispersion/best_model.pt
# python3 scripts/analyze.py --config configs/sra_wikitext103-ablation4-standard-MoE.yaml --checkpoint outputs_wt103-ablation/4-standard-MoE/best_model.pt

import argparse
import logging
import os
import re
import sys
import csv

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
try:
    import umap
except ImportError:
    umap = None
from sklearn.manifold import TSNE
from tokenizers import Tokenizer
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    EXTERNAL_JUDGE = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("External Judge (SentenceTransformer) loaded for Objective Purity Metric.")
except Exception as e:
    EXTERNAL_JUDGE = None
    logging.warning(f"External Purity metric unavailable: {e}. Install 'sentence-transformers'.")

# Ensure the src directory is in PYTHONPATH for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import create_dataloaders
from src.utils.config_utils import load_config, ConfigDict
from src.models.sra import SRA
from src.models.csr import ChamberOfSemanticResonance

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def save_figure_publication(fig, output_dir, filename_base):
    """Save figure in publication-quality formats (PDF and SVG)."""
    for fmt in ['pdf', 'svg']:
        save_path = os.path.join(output_dir, f"{filename_base}.{fmt}")
        fig.savefig(save_path, format=fmt, dpi=300, bbox_inches='tight')
        logging.info(f"Saved: {save_path}")


def _convert_old_expert_format(state_dict, num_experts):
    """Convert old per-expert nn.Module format to stacked parameter format."""
    import re as _re
    has_old = any('experts.0.linear1.weight' in k for k in state_dict)
    has_new = any('expert_w1' in k for k in state_dict)
    if has_new or not has_old:
        return state_dict
    logging.info("Converting old per-expert format to stacked parameter format...")
    new_sd = {}
    expert_groups = {}
    pattern = _re.compile(r'^(.+\.ff_or_moe)\.experts\.(\d+)\.(linear[12])\.(weight|bias)$')
    for k, v in state_dict.items():
        m = pattern.match(k)
        if m:
            prefix, idx, layer, param = m.group(1), int(m.group(2)), m.group(3), m.group(4)
            key = (prefix, layer, param)
            if key not in expert_groups:
                expert_groups[key] = {}
            expert_groups[key][idx] = v
        else:
            new_sd[k] = v
    name_map = {
        ('linear1', 'weight'): 'expert_w1', ('linear1', 'bias'): 'expert_b1',
        ('linear2', 'weight'): 'expert_w2', ('linear2', 'bias'): 'expert_b2',
    }
    for (prefix, layer, param), experts in expert_groups.items():
        stacked_name = name_map[(layer, param)]
        tensors = [experts[i] for i in range(len(experts))]
        new_sd[f"{prefix}.{stacked_name}"] = torch.stack(tensors)
    logging.info(f"Converted {len(expert_groups)} expert parameter groups")
    return new_sd


def load_model_from_checkpoint(checkpoint_path, config_path=None):
    """Loads the trained SRA model state and configuration."""
    logging.info(f"Loading model from checkpoint: {checkpoint_path}")
    try:
        # Load checkpoint onto CPU
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        logging.error(f"Checkpoint not found at {checkpoint_path}")
        return None, None

    # Load config from checkpoint if available, otherwise from file
    if 'config' in checkpoint:
        config = ConfigDict(checkpoint['config'])
    elif config_path:
        config_dict = load_config(config_path)
        if not config_dict: return None, None
        config = ConfigDict(config_dict)
    else:
        logging.error("Configuration not found in checkpoint and no config path provided.")
        return None, None

    # Initialize model structure
    model = SRA(config.model, config.data.vocab_size)

    # Load state dict (handles potential DDP prefix mismatch)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix if present
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Convert old per-expert format to stacked parameter format if needed
    # Only convert if the model expects stacked format (expert_w1) but checkpoint has old format
    model_expects_stacked = any('expert_w1' in k for k in model.state_dict())
    if model_expects_stacked:
        new_state_dict = _convert_old_expert_format(new_state_dict, config.model.csr.num_experts)

    # Fix temperature shape: old checkpoint uses scalar [], current code uses [1]
    for k in list(new_state_dict.keys()):
        if k.endswith('.temperature') and new_state_dict[k].dim() == 0:
            new_state_dict[k] = new_state_dict[k].unsqueeze(0)

    model.load_state_dict(new_state_dict)
    # Disable batch seeding — anchors already loaded from checkpoint
    for module in model.modules():
        if hasattr(module, '_needs_batch_seed'):
            module._needs_batch_seed = False
    model.eval()
    logging.info("Model loaded successfully.")
    return model, config


def get_embeddings(model):
    """Extracts vocabulary embeddings and semantic anchors from the model."""
    # 1. Vocabulary Embeddings (reference space)
    word_embeds = model.token_embed.weight.data

    # 2. Semantic Anchors (from CSR layers) or Gate weights (for linear routing)
    anchors = {}
    for name, module in model.named_modules():
        if isinstance(module, ChamberOfSemanticResonance):
            if hasattr(module, 'semantic_anchors'):
                anchor_data = module.semantic_anchors.data
                # Multi-head anchors: (E, H, D) -> average across heads -> (E, D)
                if anchor_data.dim() == 3:
                    anchor_data = anchor_data.mean(dim=1)
                anchors[name] = anchor_data
            elif hasattr(module, 'gate'):
                # Linear routing: use gate weight rows as pseudo-anchors
                # gate.weight shape: (num_experts, d_model)
                anchors[name] = module.gate.weight.data
                logging.info(f"Using gate weights as pseudo-anchors for {name}")

    return word_embeds, anchors


def find_nearest_neighbors(query_vectors, reference_vectors, tokenizer, k=10):
    """
    Finds the nearest neighbor words (from reference) for each query vector
    using Cosine Similarity.
    """
    logging.info("Calculating nearest neighbors (Cosine Similarity)...")

    # Normalize vectors (crucial for Cosine Similarity)
    query_norm = F.normalize(query_vectors, p=2, dim=-1)
    ref_norm = F.normalize(reference_vectors, p=2, dim=-1)

    # Calculate similarity: (N_query, D) @ (D, N_vocab) -> (N_query, N_vocab)
    similarity = torch.matmul(query_norm, ref_norm.T)

    # Find top K
    topk_scores, topk_indices = torch.topk(similarity, k, dim=-1)

    results = []
    # Get vocabulary mapping (ID -> Token)
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}

    for i in range(query_vectors.size(0)):
        neighbors = []
        for j in range(k):
            token_id = topk_indices[i, j].item()
            score = topk_scores[i, j].item()
            # Use mapping instead of decode for speed and accuracy
            token_str = id_to_token.get(token_id, "[UNK]")
            neighbors.append((token_str, score))
        results.append(neighbors)
    return results


def calc_internal_cohesion(top_token_ids, word_embeds):
    """Average pairwise cosine similarity of an expert's top-K tokens in model embedding space."""
    if len(top_token_ids) < 2:
        return 0.0
    vecs = word_embeds[top_token_ids]  # (K, D)
    vecs = F.normalize(vecs.float(), p=2, dim=-1)
    sim = torch.matmul(vecs, vecs.T)  # (K, K)
    # Upper triangle excluding diagonal
    mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
    return sim[mask].mean().item()


def calc_external_purity(tokens_str_list):
    """External purity: pairwise cosine similarity via pretrained judge (all-MiniLM-L6-v2).

    Aggressively cleans BPE artifacts and filters out subword fragments / punctuation.
    Returns 0.0 for purely syntactic experts (suffixes, punctuation) — by design,
    creating the Internal-high / External-low signature for syntactic experts.
    """
    if EXTERNAL_JUDGE is None:
        return 0.0

    clean_tokens = []
    for t in tokens_str_list:
        # Aggressive BPE cleaning: GPT-2 (Ġ), SentencePiece (▁), fastBPE (</w>), WordPiece (##)
        t_clean = t.replace('Ġ', '').replace('\u2581', '').replace('</w>', '').replace('##', '').strip()
        # Keep only tokens with length > 1 that contain at least one letter.
        # Punctuation-only and single-char subwords are intentionally excluded —
        # they have no "semantic purity" in isolation for the external judge.
        if len(t_clean) > 1 and re.search(r'[a-zA-Z]', t_clean):
            clean_tokens.append(t_clean)

    if len(clean_tokens) < 2:
        return 0.0

    with torch.no_grad():
        embs = EXTERNAL_JUDGE.encode(clean_tokens, convert_to_tensor=True)
        embs = F.normalize(embs.float(), p=2, dim=-1)
        sims = torch.matmul(embs, embs.T)
        k = sims.size(0)
        triu = torch.triu_indices(k, k, offset=1)
        return sims[triu[0], triu[1]].mean().item()


def analyze_anchors(word_embeds, anchors_dict, tokenizer, output_dir):
    """Analyzes expert specialization with monosemanticity metrics and saves to CSV.

    Returns:
        dict: {layer_name: {'internal': [float], 'external': [float]}} for plotting.
    """
    K = 20
    logging.info(f"\n--- Anchor Specialization Analysis (Top {K} Neighbors + Monosemanticity Metrics) ---")

    # Build id-to-token mapping
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "experts.csv")

    # Pre-normalize word embeddings once (constant across layers)
    embeds_norm = F.normalize(word_embeds.float(), p=2, dim=-1)

    all_scores = {}

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "expert_id", "Internal_Cohesion", "External_Purity", "tokens"])

        for layer_name, anchors in anchors_dict.items():
            num_experts = anchors.size(0)
            logging.info(f"\nLayer: {layer_name} (Total Experts: {num_experts})")

            # Compute cosine similarity: anchors vs word embeddings
            anchors_norm = F.normalize(anchors.float(), p=2, dim=-1)
            similarity = torch.matmul(anchors_norm, embeds_norm.T)  # (E, V)
            topk_scores, topk_indices = torch.topk(similarity, K, dim=-1)  # (E, K)

            layer_int_scores = []
            layer_ext_scores = []

            for i in range(num_experts):
                token_ids = topk_indices[i].tolist()
                token_strs = [id_to_token.get(tid, "[UNK]") for tid in token_ids]
                scores = topk_scores[i].tolist()

                # Monosemanticity metrics
                int_coh = calc_internal_cohesion(topk_indices[i], word_embeds)
                ext_pur = calc_external_purity(token_strs)

                layer_int_scores.append(int_coh)
                layer_ext_scores.append(ext_pur)

                # Tag
                if int_coh > 0.35:
                    tag = "PURE"
                elif int_coh < 0.15:
                    tag = "JUNK"
                else:
                    tag = "MIX"

                # Log top 10 for readability, but metrics use all K=20
                top10_desc = ", ".join([f"'{w}' ({s:.2f})" for w, s in zip(token_strs[:10], scores[:10])])
                logging.info(f"  Expert {i:03d} [{tag}] Int={int_coh:.3f} Ext={ext_pur:.3f}: {top10_desc}")

                # CSV row with all K tokens
                writer.writerow([layer_name, i, f"{int_coh:.4f}", f"{ext_pur:.4f}", ", ".join(token_strs)])

            # Layer summary
            avg_int = np.mean(layer_int_scores)
            avg_ext = np.mean(layer_ext_scores)
            logging.info(f"  >> AVERAGE INTERNAL COHESION: {avg_int:.4f}")
            logging.info(f"  >> AVERAGE EXTERNAL PURITY:   {avg_ext:.4f}")

            all_scores[layer_name] = {'internal': layer_int_scores, 'external': layer_ext_scores}

    logging.info(f"Anchor specialization results saved to {csv_path}")
    return all_scores


def visualize_monosemanticity_scatter(all_scores, output_dir):
    """2D Interpretability Matrix: Internal Cohesion (x) vs External Purity (y).

    Reveals three expert types:
      - Semantic (high Int, high Ext): learned real-world concepts
      - Syntactic (high Int, low Ext): learned grammar/subword patterns
      - Dead/Junk (low Int, low Ext): gradient-starved, random tokens
    """
    logging.info("\n--- Monosemanticity Scatter Plot ---")

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = sns.color_palette("tab10", len(all_scores))
    for idx, (layer_name, scores) in enumerate(all_scores.items()):
        int_arr = np.array(scores['internal'])
        ext_arr = np.array(scores['external'])
        # Short layer label for legend
        short = layer_name.split('.')[-2] if '.' in layer_name else layer_name
        ax.scatter(int_arr, ext_arr, s=25, alpha=0.6, color=colors[idx], label=short)

    # Quadrant boundaries
    ax.axvline(x=0.35, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.10, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0.15, color='grey', linestyle=':', linewidth=0.8, alpha=0.3)

    # Quadrant labels
    ax.text(0.70, 0.85, 'Semantic', transform=ax.transAxes, fontsize=11,
            fontweight='bold', color='#2ca02c', ha='center', alpha=0.7)
    ax.text(0.70, 0.08, 'Syntactic', transform=ax.transAxes, fontsize=11,
            fontweight='bold', color='#1f77b4', ha='center', alpha=0.7)
    ax.text(0.10, 0.08, 'Dead / Junk', transform=ax.transAxes, fontsize=11,
            fontweight='bold', color='#d62728', ha='center', alpha=0.7)

    ax.set_xlabel('Internal Cohesion (model embedding space)', fontsize=12)
    ax.set_ylabel('External Purity (all-MiniLM-L6-v2)', fontsize=12)
    ax.set_title('Expert Monosemanticity: Semantic vs Syntactic Specialization', fontsize=13)
    ax.legend(title='Layer', fontsize=9, title_fontsize=10)
    ax.set_xlim(-0.05, max(0.8, ax.get_xlim()[1]))
    ax.set_ylim(-0.05, max(0.8, ax.get_ylim()[1]))

    plt.tight_layout()
    save_figure_publication(fig, output_dir, "monosemanticity_scatter")
    plt.close(fig)


def visualize_space(word_embeds, anchors_dict, output_dir, seed=42):
    """Visualizes the embedding space using UMAP or t-SNE."""
    logging.info("\n--- Semantic Space Visualization (t-SNE/UMAP) ---")

    # Prepare data: Combine a subset of words and all anchors
    num_words_to_visualize = 500  # Use a subset for clarity
    combined_vectors = [word_embeds[:num_words_to_visualize]]

    anchor_vectors = []
    for anchors in anchors_dict.values():
        anchor_vectors.append(anchors)

    if not anchor_vectors:
        logging.warning("No anchors found to visualize.")
        return

    combined_vectors.extend(anchor_vectors)
    all_vectors = torch.cat(combined_vectors, dim=0).cpu().numpy()

    logging.info(f"Reducing dimensionality of {all_vectors.shape[0]} vectors...")

    # Select and run dimensionality reduction
    if umap is not None:
        logging.info("Using UMAP (Metric: Cosine).")
        # Use cosine metric, matching the SRA routing mechanism
        reducer = umap.UMAP(n_components=2, metric='cosine', n_neighbors=15, min_dist=0.1, random_state=seed)
    else:
        logging.info("Using t-SNE (Metric: Cosine). Install 'umap-learn' for better visualization.")
        perplexity = min(30.0, float(all_vectors.shape[0] - 1))
        reducer = TSNE(n_components=2, metric='cosine', perplexity=perplexity, init='pca', learning_rate='auto',
                       random_state=seed)

    embeddings_2d = reducer.fit_transform(all_vectors)

    # Plotting
    plt.figure(figsize=(16, 16))
    sns.set_style("whitegrid")

    # 1. Plot words (small, grey dots)
    plt.scatter(embeddings_2d[:num_words_to_visualize, 0],
                embeddings_2d[:num_words_to_visualize, 1],
                s=15, alpha=0.3, color='grey', label="Vocabulary (Subset)")

    # 2. Plot anchors (colored stars, different color per layer)
    start_idx = num_words_to_visualize
    colors = sns.color_palette("hsv", len(anchors_dict))

    for i, (layer_name, anchors) in enumerate(anchors_dict.items()):
        end_idx = start_idx + anchors.size(0)
        plt.scatter(embeddings_2d[start_idx:end_idx, 0],
                    embeddings_2d[start_idx:end_idx, 1],
                    marker='*', s=120, alpha=0.8, label=layer_name, color=colors[i])
        start_idx = end_idx

    plt.legend(loc='best')
    plt.title("SRA Semantic Space: Vocabulary and Learned Anchors")

    save_figure_publication(plt.gcf(), output_dir, "semantic_space_visualization")


def _clean_layer_name(layer_name):
    """Convert raw module name like 'blocks.0.ff_or_moe' to 'Layer 0'."""
    import re
    m = re.search(r'blocks\.(\d+)', layer_name)
    return f"Layer {m.group(1)}" if m else layer_name


def visualize_dispersion_heatmap(anchors_dict, output_dir):
    """Visualizes the anchor similarity matrix as a heatmap."""
    logging.info("\n--- Anchor Dispersion Visualization (Heatmaps) ---")

    # Check routing method (dispersion only applies to cosine routing)
    if not anchors_dict:
        logging.info("No anchors found (Standard MoE or Dense model?). Skipping heatmap.")
        return

    for layer_name, anchors in anchors_dict.items():
        # Use FP32 for precision
        anchors_norm = F.normalize(anchors.float(), p=2, dim=-1).cpu().numpy()
        similarity_matrix = np.dot(anchors_norm, anchors_norm.T)

        n_experts = similarity_matrix.shape[0]
        plt.figure(figsize=(12, 10))
        # Fixed range for cosine similarity [-1, 1]
        ax = sns.heatmap(similarity_matrix, cmap="viridis", vmin=-1, vmax=1, square=True,
                         xticklabels=False, yticklabels=False)
        # Show sparse tick labels for readability
        tick_step = max(1, n_experts // 8)
        tick_positions = list(range(0, n_experts, tick_step))
        ax.set_xticks([p + 0.5 for p in tick_positions])
        ax.set_xticklabels(tick_positions, fontsize=9)
        ax.set_yticks([p + 0.5 for p in tick_positions])
        ax.set_yticklabels(tick_positions, fontsize=9)
        ax.set_xlabel("Expert ID", fontsize=12)
        ax.set_ylabel("Expert ID", fontsize=12)
        clean_name = _clean_layer_name(layer_name)
        plt.title(f"Anchor Cosine Similarity ({clean_name})", fontsize=14)

        save_figure_publication(plt.gcf(), output_dir, f"dispersion_heatmap_{layer_name.replace('.', '_')}")
        plt.close()


@torch.no_grad()
def analyze_expert_utilization(model, config, device='cuda'):
    """Analyzes expert utilization on the validation set using forward hooks."""
    logging.info("\n--- Expert Utilization Analysis (Validation Set) ---")

    if not config.model.csr.enabled:
        logging.info("CSR disabled. Skipping utilization analysis.")
        return

    # 1. Setup DataLoader
    try:
        bs = 32  # Fixed batch size for analysis
        _, val_loader, _ = create_dataloaders(config.data, batch_size=bs, num_workers=2)
    except Exception as e:
        logging.error(f"Could not create validation loader: {e}")
        return

    # 2. Setup Hooks
    hooks = []
    utilization_stats = {}

    def get_activation_hook(name):
        # This hook reads topk_indices added to aux_data in CSR.forward
        def hook(module, input, output):
            # output[1] is aux_data
            if isinstance(output, tuple) and len(output) > 1 and 'topk_indices' in output[1]:
                topk_indices = output[1]['topk_indices']
                indices_flat = topk_indices.view(-1)
                # Count activations
                counts = torch.bincount(indices_flat, minlength=module.num_experts)

                if name not in utilization_stats:
                    utilization_stats[name] = counts.detach().cpu()
                else:
                    utilization_stats[name] += counts.detach().cpu()

        return hook

    # Attach hooks
    model.to(device)
    for name, module in model.named_modules():
        if isinstance(module, ChamberOfSemanticResonance):
            hooks.append(module.register_forward_hook(get_activation_hook(name)))

    # 3. Forward Pass
    model.eval()
    logging.info("Running forward pass on validation set...")
    for batch in tqdm(val_loader, desc="Analyzing Utilization"):
        # Move data to device, skipping 'labels'
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels' and isinstance(v, torch.Tensor)}
        model(inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # 4. Analysis and Visualization
    if not utilization_stats: return

    plt.figure(figsize=(15, 5 * len(utilization_stats)))
    sorted_layers = sorted(utilization_stats.items())

    for i, (layer_name, counts) in enumerate(sorted_layers):
        total_activations = counts.sum().item()
        if total_activations == 0: continue

        utilization_percent = (counts.float() / total_activations) * 100

        # Balance metrics (CV = Coefficient of Variation, lower = better)
        mean_util = utilization_percent.mean().item()
        std_util = utilization_percent.std().item()
        cv = (std_util / mean_util) if mean_util > 0 else 0
        dead_experts = (utilization_percent < 0.01).sum().item()

        logging.info(f"\nLayer: {layer_name} | CV: {cv:.4f} | Dead Experts: {dead_experts}")

        # Visualization
        ax = plt.subplot(len(utilization_stats), 1, i + 1)
        sns.barplot(x=np.arange(len(counts)), y=utilization_percent.numpy(), color='lightblue', ax=ax)
        clean_name = _clean_layer_name(layer_name)
        ax.set_title(f"{clean_name} (CV={cv:.4f}, Dead={dead_experts})", fontsize=13)
        ax.set_xlabel("Expert ID", fontsize=11)
        ax.set_ylabel("Utilization (%)", fontsize=11)
        ax.axhline(mean_util, color='r', linestyle='--', label=f'Mean: {mean_util:.2f}%')
        ax.legend(fontsize=10)
        # Sparse x-ticks for readability
        n_exp = len(counts)
        tick_step = max(1, n_exp // 16)
        ax.set_xticks(range(0, n_exp, tick_step))
        ax.set_xticklabels(range(0, n_exp, tick_step), fontsize=8)

    plt.tight_layout()
    save_figure_publication(plt.gcf(), config.project.output_dir, "expert_utilization_analysis")


def main(checkpoint_path, config_path=None):
    # 1. Load Model and Config
    # Try loading config from checkpoint, but allow manual override
    model, config = load_model_from_checkpoint(checkpoint_path, config_path)
    if not model: return

    # 2. Load Tokenizer
    try:
        tokenizer = Tokenizer.from_file(config.data.tokenizer_path)
        logging.info(f"Tokenizer loaded. Vocab size: {tokenizer.get_vocab_size()}")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        return

    # 3. Extract Embeddings and Anchors
    word_embeds, anchors_dict = get_embeddings(model)

    if not anchors_dict:
        logging.warning("No Semantic Anchors found. The model might be using standard FFN or dense layers.")
        if not config.model.csr.enabled:
            logging.info("CSR is disabled in config. Analysis will be limited to basic model info.")
            logging.info(f"Model has {sum(p.numel() for p in model.parameters())} total parameters")
            return
    else:
        logging.info(f"Found {len(anchors_dict)} CSR layers with anchors")

    # 4. Analyze Anchor Specialization + Monosemanticity Scatter
    if anchors_dict:
        mono_scores = analyze_anchors(word_embeds, anchors_dict, tokenizer, config.project.output_dir)
        if mono_scores:
            visualize_monosemanticity_scatter(mono_scores, config.project.output_dir)

    # 5. Visualize Semantic Space
    if anchors_dict:
        os.makedirs(config.project.output_dir, exist_ok=True)
        visualize_space(word_embeds, anchors_dict, config.project.output_dir, config.project.seed)

    # 6. Dispersion Heatmap
    if anchors_dict:
        visualize_dispersion_heatmap(anchors_dict, config.project.output_dir)

    # 7. Utilization Analysis
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyze_expert_utilization(model, config, device)

    logging.info("\n--- Analysis Complete ---")


if __name__ == '__main__':
    # python scripts/analyze.py --checkpoint outputs/best_model.pt [--config configs/sra_wikitext103.yaml]

    parser = argparse.ArgumentParser(description="Analyze the trained SRA model's interpretability.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the configuration YAML file (optional, if stored in checkpoint).")
    args = parser.parse_args()
    main(args.checkpoint, args.config)
