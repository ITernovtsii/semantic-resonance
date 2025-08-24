import os
import sys
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_utils import load_config, ConfigDict
from src.models.sra import SRA


@torch.no_grad()
def load_model_and_config(checkpoint_path, config_path=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'config' in checkpoint:
        config = ConfigDict(checkpoint['config'])
    elif config_path:
        config = ConfigDict(load_config(config_path))
    else:
        raise RuntimeError("Configuration not found in checkpoint and no config path provided.")

    model = SRA(config.model, config.data.vocab_size)

    state_dict = checkpoint['model_state_dict']
    new_state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return model, config


@torch.no_grad()
def compute_top2_from_aux(aux_data, seq_len, top_k):
    # resonance_scores: (B*S, N_experts), topk_indices: (B*S, top_k)
    resonance_scores = aux_data["resonance_scores"]  # dtype == model dtype
    topk_indices = aux_data["topk_indices"]  # int64

    bs_tokens = topk_indices.size(0)
    assert bs_tokens == seq_len, "Expected B=1. If B>1, adapt indexing per batch item."

    # Gather scores for those top_k indices
    rows = torch.arange(bs_tokens).unsqueeze(-1)
    topk_scores = resonance_scores[rows, topk_indices]  # (S, top_k)

    # Convert to weights with softmax across the K experts
    weights = F.softmax(topk_scores.float(), dim=-1)  # (S, top_k)
    # Choose top-2 from these K per token
    top2_weights, top2_pos = torch.topk(weights, k=min(2, top_k), dim=-1)  # (S, 2)
    # Map back to expert ids
    top2_expert_ids = topk_indices.gather(1, top2_pos)  # (S, 2)

    return top2_expert_ids, top2_weights


@torch.no_grad()
def nearest_tokens_for_anchor(anchor_vec, word_embeds, tokenizer, k=5):
    # Normalize
    a = F.normalize(anchor_vec.unsqueeze(0).float(), p=2, dim=-1)  # (1, D)
    W = F.normalize(word_embeds.float(), p=2, dim=-1)  # (V, D)
    sims = torch.matmul(a, W.T).squeeze(0)  # (V,)
    topk_scores, topk_indices = torch.topk(sims, k)
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    return [(id_to_token.get(idx.item(), "[UNK]"), topk_scores[i].item()) for i, idx in enumerate(topk_indices)]


def format_interpretation(neighbors, max_terms=3):
    # Turn nearest neighbors into a compact hint
    return ", ".join([f"{tok}" for tok, _ in neighbors[:max_terms]])


def print_markdown_table(rows):
    # rows: list of dict with keys:
    # Token, Expert1, Weight1, Expert2, Weight2, Interpretation
    headers = ["Token", "Expert 1", "Weight 1", "Expert 2", "Weight 2", "Interpretation"]
    # Markdown
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        print("| " + " | ".join([
            str(r["Token"]),
            str(r["Expert1"]),
            f"{r['Weight1']:.2f}",
            str(r["Expert2"]),
            f"{r['Weight2']:.2f}",
            r["Interpretation"]
        ]) + " |")


def main():
    checkpoint_path = "notes/final/results/sra-D512-1024-1_2@128-4-a4-b6-z0-ppl-13.4/best_model.pt"
    config_path = "notes/final/results/sra-D512-1024-1_2@128-4-a4-b6-z0-ppl-13.4/sra_wikitext103.yaml"
    sentence = "The film was released in December 1995 and received positive reviews."

    # 1) Load model + config
    model, config = load_model_and_config(checkpoint_path, config_path)

    # 2) Load tokenizer
    tokenizer = Tokenizer.from_file(config.data.tokenizer_path)

    # 3) Prepare inputs
    enc = tokenizer.encode(sentence)
    input_ids = torch.tensor([enc.ids], dtype=torch.long)  # (1, S)
    tokens = enc.tokens  # list[str] length S
    seq_len = input_ids.size(1)

    # 4) Prepare hook for the first CSR layer (Layer 0)
    captured = {}

    def hook(name):
        def fn(module, inputs, output):
            # output is (ff_output, aux_data)
            if isinstance(output, tuple) and len(output) > 1:
                aux = output[1]
                if ("resonance_scores" in aux) and ("topk_indices" in aux):
                    # Only capture first CSR layer data and module references once
                    if "done" not in captured:
                        captured["aux"] = {
                            "resonance_scores": aux["resonance_scores"].detach().cpu()[:seq_len, :],
                            "topk_indices": aux["topk_indices"].detach().cpu()[:seq_len, :]
                        }
                        captured["anchors"] = module.semantic_anchors.detach().cpu()
                        captured["layer_name"] = name
                        captured["top_k"] = aux["topk_indices"].size(-1)
                        captured["done"] = True

        return fn

    hooks = []
    for name, module in model.named_modules():
        if "ChamberOfSemanticResonance" in str(type(module)):
            hooks.append(module.register_forward_hook(hook(name)))
            break  # only first CSR layer (Layer 0)

    # 5) Forward pass
    with torch.no_grad():
        model({"input_ids": input_ids})

    # Remove hooks
    for h in hooks:
        h.remove()

    if "aux" not in captured:
        raise RuntimeError("No CSR routing was captured. Ensure CSR is enabled and the hook attached to a CSR layer.")

    # 6) Compute top-2 experts and weights
    aux = captured["aux"]
    top_k = captured["top_k"]
    top2_expert_ids, top2_weights = compute_top2_from_aux(aux, seq_len, top_k)

    # 7) Build interpretations (nearest tokens for each chosen expert)
    # Get word embeddings from model
    word_embeds = model.token_embed.weight.data.cpu()
    anchors = captured["anchors"]

    rows = []
    for i in range(seq_len):
        e1 = int(top2_expert_ids[i, 0].item())
        w1 = float(top2_weights[i, 0].item())
        if top2_expert_ids.size(1) > 1:
            e2 = int(top2_expert_ids[i, 1].item())
            w2 = float(top2_weights[i, 1].item())
        else:
            e2, w2 = None, 0.0

        # Interpret experts via nearest tokens to their anchors
        neighbors_e1 = nearest_tokens_for_anchor(anchors[e1], word_embeds, tokenizer, k=5)
        neighbors_e2 = nearest_tokens_for_anchor(anchors[e2], word_embeds, tokenizer, k=5) if e2 is not None else []

        interp = f"E{e1}: {format_interpretation(neighbors_e1)}"
        if e2 is not None:
            interp += f" | E{e2}: {format_interpretation(neighbors_e2)}"

        rows.append({
            "Token": tokens[i],
            "Expert1": f"E_{e1}",
            "Weight1": w1,
            "Expert2": f"E_{e2}" if e2 is not None else "",
            "Weight2": w2,
            "Interpretation": interp
        })

    # 8) Print Markdown table (Layer 0)
    print(f"\nTable: Routing decisions for the example sentence (Layer 0: {captured['layer_name']})\n")
    print_markdown_table(rows)


if __name__ == "__main__":
    main()
