import argparse
import logging
import os
import sys
import csv

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import umap
from sklearn.manifold import TSNE
from tokenizers import Tokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import create_dataloaders
from src.utils.config_utils import load_config, ConfigDict
from src.models.sra import SRA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    model.load_state_dict(new_state_dict)
    model.eval()
    logging.info("Model loaded successfully.")
    return model, config


def get_embeddings(model):
    """Extracts vocabulary embeddings and semantic anchors from the model."""
    # 1. Vocabulary Embeddings (reference space)
    word_embeds = model.token_embed.weight.data

    # 2. Semantic Anchors (from CSR layers)
    anchors = {}
    # We use model.named_modules() to iterate over all components
    for name, module in model.named_modules():
        # Identify CSR layers by class type
        if 'ChamberOfSemanticResonance' in str(type(module)):
            if hasattr(module, 'semantic_anchors'):
                anchors[name] = module.semantic_anchors.data

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
    # Get dictionary for ID -> Token mapping
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


def analyze_anchors(word_embeds, anchors_dict, tokenizer, output_dir):
    """Analyzes and prints the specialization (nearest words) of each anchor and saves to CSV."""
    logging.info("\n--- Anchor Specialization Analysis (Top 10 Neighbors) ---")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "experts.csv")

    # Prepare CSV writer
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(["layer", "expert_id", "tokens"])

        for layer_name, anchors in anchors_dict.items():
            logging.info(f"\nLayer: {layer_name} (Total Experts: {anchors.size(0)})")

            # Find nearest words for the anchors in this layer
            neighbors = find_nearest_neighbors(anchors, word_embeds, tokenizer, k=10)

            for i, neighbor_list in enumerate(neighbors):
                # Log formatted output
                expert_desc = ", ".join([f"'{word}' ({score:.2f})" for word, score in neighbor_list])
                logging.info(f"  Expert {i:03d}: {expert_desc}")

                # Write to CSV: one row per expert with comma-separated tokens
                tokens = [word for word, _ in neighbor_list]
                writer.writerow([layer_name, i, ", ".join(tokens)])

    logging.info(f"Anchor specialization results saved to {csv_path}")


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
        # Use cosine metric matching SRA mechanism
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

    save_path = os.path.join(output_dir, "semantic_space_visualization.png")
    plt.savefig(save_path, dpi=300)
    logging.info(f"Visualization saved to {save_path}")


def visualize_dispersion_heatmap(anchors_dict, output_dir):
    """Visualizes anchor similarity matrix using heatmap."""
    logging.info("\n--- Anchor Dispersion Visualization (Heatmaps) ---")

    # Check routing method (Dispersion only makes sense for Cosine)
    if not anchors_dict:
        logging.info("No anchors found (Standard MoE or Dense model?). Skipping heatmap.")
        return

    for layer_name, anchors in anchors_dict.items():
        # Use FP32 for precision
        anchors_norm = F.normalize(anchors.float(), p=2, dim=-1).cpu().numpy()
        similarity_matrix = np.dot(anchors_norm, anchors_norm.T)

        plt.figure(figsize=(12, 10))
        # Fixed range for cosine similarity [-1, 1]
        sns.heatmap(similarity_matrix, cmap="viridis", vmin=-1, vmax=1, square=True)
        plt.title(f"Anchor Cosine Similarity Heatmap - {layer_name}")

        save_path = os.path.join(output_dir, f"dispersion_heatmap_{layer_name.replace('.', '_')}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logging.info(f"Heatmap for {layer_name} saved to {save_path}")


@torch.no_grad()
def analyze_expert_utilization(model, config, device='cuda'):
    """Analyzes expert utilization on validation set using Forward Hooks."""
    logging.info("\n--- Expert Utilization Analysis (Validation Set) ---")

    if not config.model.csr.enabled:
        logging.info("CSR disabled. Skipping utilization analysis.")
        return

    # 1. Setup DataLoader
    try:
        bs = 32  # Use fixed BS for analysis
        _, val_loader, _ = create_dataloaders(config.data, batch_size=bs, num_workers=2)
    except Exception as e:
        logging.error(f"Could not create validation loader: {e}")
        return

    # 2. Setup Hooks
    hooks = []
    utilization_stats = {}

    def get_activation_hook(name):
        # This hook uses topk_indices which we added to aux_data in CSR.forward
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
        if 'ChamberOfSemanticResonance' in str(type(module)):
            hooks.append(module.register_forward_hook(get_activation_hook(name)))

    # 3. Forward Pass
    model.eval()
    logging.info("Running forward pass on validation set...")
    for batch in tqdm(val_loader, desc="Analyzing Utilization"):
        # Move data to device, ignoring 'labels'
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

        # Balancing metrics (CV - Coefficient of Variation, lower = better)
        mean_util = utilization_percent.mean().item()
        std_util = utilization_percent.std().item()
        cv = (std_util / mean_util) if mean_util > 0 else 0
        dead_experts = (utilization_percent < 0.01).sum().item()

        logging.info(f"\nLayer: {layer_name} | CV: {cv:.4f} | Dead Experts: {dead_experts}")

        # Visualization
        plt.subplot(len(utilization_stats), 1, i + 1)
        sns.barplot(x=np.arange(len(counts)), y=utilization_percent.numpy(), color='lightblue')
        plt.title(f"{layer_name} (CV={cv:.4f}, Dead={dead_experts})")
        plt.xlabel("Expert ID")
        plt.ylabel("Utilization (%)")
        plt.axhline(mean_util, color='r', linestyle='--', label=f'Mean: {mean_util:.2f}%')
        plt.legend()

    plt.tight_layout()
    save_path = os.path.join(config.project.output_dir, "expert_utilization_analysis.png")
    plt.savefig(save_path, dpi=300)
    logging.info(f"Utilization analysis saved to {save_path}")


def main(checkpoint_path, config_path=None):
    # 1. Load Model and Config
    # We try to load config from checkpoint, but allow manual specification
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

    # 4. Analyze Anchor Specialization
    if anchors_dict:
        analyze_anchors(word_embeds, anchors_dict, tokenizer, config.project.output_dir)

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
    parser = argparse.ArgumentParser(description="Analyze the trained SRA model's interpretability.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the configuration YAML file (optional, if stored in checkpoint).")
    args = parser.parse_args()
    main(args.checkpoint, args.config)
