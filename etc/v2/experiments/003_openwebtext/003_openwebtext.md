# Experiment 003: Cross-Dataset Validation on OpenWebText

**Status**: RUNNING
**GPU**: 1, 2
**Script**: `experiments/003_openwebtext/run_openwebtext.py`
**Code version**: `last-cosine-bandpass` branch
**Training time**: ~8h estimated (2 runs in parallel on GPU 1+2)

## Intent
Validate that SRA's routing properties (comparable PPL, interpretability metrics, expert utilization) generalize beyond WikiText-103 to a different domain (web text). This addresses the primary remaining reviewer feedback: all experiments use a single dataset.

## Hypothesis
- Cosine and linear routing achieve comparable perplexity on OpenWebText (gap < 0.5 PPL)
- Bandpass loss maintains healthy expert utilization (dead experts < 10%)
- IC/EP metrics show similar patterns to WikiText-103 (comparable across routing types)
- Progressive K=2->4 training dynamics transfer to web-domain text

## Design

| Parameter | Value |
|-----------|-------|
| Dataset | OpenWebText (first 104K documents, ~125M tokens, target ~405K training chunks matching WT103's 405K) |
| Models | SRA E (cosine K=2->4), StdMoE E (linear K=2->4) |
| Experts | 256, d_ff=256, d_model=512, 4 layers |
| Seed | 19 |
| Epochs | 8 |
| Loss | Bandpass S6 masked (0.0005-0.0040), alpha=0.4 |
| Tokenizer | New BPE 32K trained on OpenWebText subset |
| What changes vs Exp 001 | Only the dataset; all other hyperparams identical |

## Baselines (WikiText-103, Exp 001 Round 2, 3-seed mean)

| Model | Routing | K Schedule | Mean PPL | Std | IC | EP |
|-------|---------|------------|----------|-----|-----|-----|
| SRA E | Cosine | K=2->4 | 12.52 | ±0.02 | 0.219 | 0.326 |
| StdMoE E | Linear | K=2->4 | 12.57 | ±0.02 | 0.240 | 0.325 |

## Run Schedule

| GPU | Model | Est. Time |
|-----|-------|-----------|
| 2 | SRA_E seed=19 | ~8h |
| 1 | StdMoE_E seed=19 | ~8h |

Runs in parallel — total wall time ~8h.

## Steps

1. Prepare data: `python3 experiments/003_openwebtext/run_openwebtext.py --prepare-data`
2. Train both models: `python3 experiments/003_openwebtext/run_openwebtext.py --train`
3. Analyze: `python3 experiments/003_openwebtext/run_openwebtext.py --analyze`

## Results

### PPL Trajectory

| Epoch | SRA_E (cosine) | StdMoE_E (linear) |
|-------|---------------|-------------------|
| 1 | 97.27 | 96.31 |
| 2 | 61.45 | 61.56 |
| 3 | 52.57 | 52.88 |
| 4 | 48.60 | 48.99 |
| 5 | 46.22 | 46.75 |
| 6 | 45.10 | 45.67 |
| 7 | **44.88** | **45.44** |
| 8 | 45.02 | 45.65 |

### Router Diagnostics (final)

| Metric | SRA_E (cosine) | StdMoE_E (linear) |
|--------|---------------|-------------------|
| Best Val PPL | 44.88 | 45.44 |
| Dead experts | 0 (0%) | 0 (0%) |
| IC (mean) | 0.212 | 0.238 |
| EP (mean) | 0.327 | 0.332 |

### Training Details

| Metric | SRA_E | StdMoE_E |
|--------|-------|----------|
| GPU | 2 | 1 |
| Training time | ~12h | ~12h |
| Steps/epoch | 3187 | 3187 |
| Speed (K=2) | ~1.3 s/it | ~1.3 s/it |
| Speed (K=4) | ~2.0 s/it | ~2.0 s/it |

## Analysis

### Key Findings
1. Cosine routing wins by 0.56 PPL on OWT (44.88 vs 45.44) — same E-schedule pattern as WT103
2. Zero dead experts in both models — bandpass + K=2→4 recipe transfers perfectly
3. IC/EP relationship preserved: linear slightly higher IC, EP comparable
4. Higher absolute PPL (44-45 vs 12-13) reflects web text diversity, not architecture degradation
