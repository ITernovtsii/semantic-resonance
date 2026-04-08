# SRA: Semantic Resonance Architecture

Official code for the paper:

**"Cosine-Similarity Routing with Semantic Anchors for Interpretable Mixture-of-Experts Language Models"**
Ivan Ternovtsii, Yurii Bilak (Uzhhorod National University)

## Overview

SRA is a Transformer-based language model with sparse Mixture of Experts (MoE). The core contribution is **cosine-similarity routing** through learnable Semantic Anchors in the CSR (Chamber of Semantic Resonance) layer. Unlike standard learned-gate MoE routers, CSR computes L2-normalized cosine similarity between token representations and per-expert anchor vectors, making routing decisions directly interpretable.

### Architecture

```
Input Tokens
    |
Token Embedding + Dropout
    |
  [SRABlock x N_layers]
    |--- LayerNorm -> CustomMHA (with RoPE) -> Residual
    |--- LayerNorm -> CSR (Sparse MoE) -> Residual
    |
Final LayerNorm -> LM Head
```

**CSR Layer**: Each expert has a learnable Semantic Anchor vector. Routing weights are computed as `softmax(cos_sim(token, anchors) * tau)`, then top-k experts are selected. This enables direct inspection of what each expert "specializes in" by examining anchor-vocabulary similarity.

### Key Results (WikiText-103)

| Model | Routing | Experts | PPL | Dead Experts |
|-------|---------|---------|-----|-------------|
| SRA (best single run) | Cosine | 256 x d_ff=512 | **12.20** | 0% |
| SRA (3-seed mean) | Cosine | 256 x d_ff=512 | 12.57 +/- 0.03 | 0-6% |
| StdMoE (3-seed mean) | Linear | 256 x d_ff=512 | 12.45 +/- 0.03 | 0-3% |

Key findings:
- Cosine and linear routing achieve comparable perplexity with the same training recipe
- **Bandpass routing loss** eliminates dead experts (from 30-45% down to 0-6%) for both routing types
- **Progressive K-scheduling** (K=1 -> K=4 at epoch 3) is essential for expert health
- Cosine routing provides inherent interpretability via anchor-token similarity

## Checkpoints

Pre-trained checkpoints (multi-seed, OpenWebText cross-validation) are available on HuggingFace:
[https://huggingface.co/iternovtsii/sra-checkpoints](https://huggingface.co/iternovtsii/sra-checkpoints)

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and PyTorch 2.0+. A GPU with bf16 support (e.g., RTX 3090, A100) is recommended.

## Data Preparation

Download and tokenize WikiText-103:

```bash
python scripts/prepare_data.py --config configs/sra_wikitext103.yaml
```

This creates a BPE tokenizer (32k vocab) and preprocesses the dataset into Arrow format under `./data/`.

## Training

### Reproduce Paper's Best Result (PPL 12.20)

```bash
# Single GPU with Accelerate
accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    scripts/train.py --config configs/sra_wikitext103.yaml
```

The default config (`configs/sra_wikitext103.yaml`) matches the paper's best setup:
- 256 experts, d_ff=512 per expert, 4 layers, d_model=512
- Cosine routing with tau=10.0, progressive K=1->4 at epoch 3
- Bandpass loss (alpha=0.4), 10 epochs, lr=3e-4

### Train Linear Routing Baseline (for comparison)

```bash
accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    scripts/train.py --config configs/sra_wikitext103_stdmoe.yaml
```

### Resume from Checkpoint

```bash
accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    scripts/train.py --config configs/sra_wikitext103.yaml --resume path/to/checkpoint.pt
```

## Analysis

After training, analyze the trained model:

```bash
python scripts/analyze.py \
    --checkpoint outputs/best_model.pt \
    --config configs/sra_wikitext103.yaml
```

This generates:
- Expert utilization plots per layer
- Semantic anchor similarity heatmaps
- Internal Cohesion (IC) and External Purity (EP) metrics
- t-SNE/UMAP projections of vocabulary embeddings with anchor positions

## Text Generation

```bash
python scripts/generate.py \
    --checkpoint outputs/best_model.pt \
    --config configs/sra_wikitext103.yaml
```

## Project Structure

```
.
├── configs/
│   ├── sra_wikitext103.yaml          # Cosine routing (paper's best)
│   ├── sra_wikitext103_stdmoe.yaml   # Linear routing baseline
│   └── accelerate/
│       └── 1gpu.yaml                 # Single-GPU accelerate config
├── src/
│   ├── models/
│   │   ├── sra.py                    # Main model (SRABlock, SRA)
│   │   ├── csr.py                    # Chamber of Semantic Resonance (MoE layer)
│   │   └── components.py             # RoPE, CustomMHA, MicroExpert
│   ├── training/
│   │   ├── trainer.py                # SRATrainer (Accelerate, AMP, W&B)
│   │   └── losses.py                 # Bandpass, CV-squared, elastic ceiling
│   ├── data/
│   │   └── dataset.py                # WikiText data loading
│   └── utils/
│       └── config_utils.py           # YAML config with dot-notation access
├── scripts/
│   ├── train.py                      # Training entry point
│   ├── prepare_data.py               # Data download and tokenization
│   ├── analyze.py                    # Model analysis and visualization
│   └── generate.py                   # Text generation
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

```bibtex
@article{ternovtsii2026sra,
  title={Cosine-Similarity Routing with Semantic Anchors for Interpretable Mixture-of-Experts Language Models},
  author={Ternovtsii, Ivan and Bilak, Yurii},
  journal={arXiv preprint arXiv:2509.14255},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
