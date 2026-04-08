# Experiment 001: Multi-Seed Matched Routing Comparison

**Status**: COMPLETED — all 12 runs done, analysis complete, paper updated (2026-03-27).
**GPU**: 0, 1, 2
**Script**: `experiments/001_multi_seed/run_multi_seed.py`
**Code version**: `last-cosine-bandpass` branch
**Training time**: ~5h per run, 3 GPUs

## Intent
Add variance estimates (mean +/- std across 3 seeds) to the matched cosine vs linear routing comparison (Tables 2 and 7 in the paper), strengthening the claim that IC/EP differences between routing types are small in magnitude.

## Hypothesis
- PPL gap between cosine and linear routing (0.07) is within seed variance
- IC/EP differences (within 0.03 IC, 0.06 EP) remain small across seeds
- Dead expert counts remain consistent (0-4% for cosine, 0% for linear)

## Design

| Parameter | Value |
|-----------|-------|
| Models | SRA B (cosine K=1->4), SRA E (cosine K=2->4), StdMoE B (linear K=1->4), StdMoE E (linear K=2->4) |
| Experts | 256, d_ff=256 |
| Seeds | 19 (existing), 42, 137 |
| Epochs | 8 |
| Loss | Bandpass S6 masked (0.0005-0.0040), alpha=0.4 |
| What changes | Only the seed; all other hyperparams identical |

## Baselines (existing seed=19 results)

| Model | Routing | PPL | Dead | IC | EP |
|-------|---------|-----|------|-----|-----|
| StdMoE B (K=1->4) | Linear | 12.48 | 0% | 0.242 | 0.360 |
| SRA E (K=2->4) | Cosine | 12.52 | 0% | 0.221 | 0.328 |
| SRA B (K=1->4) | Cosine | 12.55 | 4% | 0.227 | 0.363 |
| StdMoE E (K=2->4) | Linear | 12.59 | 0% | 0.232 | 0.319 |

## Run Schedule

Run in pairs (GPU 1 + GPU 2 simultaneously):

| Batch | GPU 1 | GPU 2 | Est. Time |
|-------|-------|-------|-----------|
| 1 | SRA_B seed=42 | StdMoE_B seed=42 | ~3.5h |
| 2 | SRA_E seed=42 | StdMoE_E seed=42 | ~3.5h |
| 3 | SRA_B seed=137 | StdMoE_B seed=137 | ~3.5h |
| 4 | SRA_E seed=137 | StdMoE_E seed=137 | ~3.5h |

Total wall time: ~14h

## Results

### Round 1 (config mismatch — batch=32/accum=4, no gradient checkpointing, anchor_init=orthogonal)

**CONFIG ISSUE**: Round 1 used `batch_size=32, grad_accum=4, anchor_init=orthogonal, no gradient_checkpointing`.
Original seed=19 runs used `batch_size=128, grad_accum=1, anchor_init=batch, gradient_checkpointing=true`.
Effective batch size is the same (128), but gradient checkpointing + anchor_init differ.
Result: PPL ~30 vs original ~12.5. Not directly comparable to paper Table 2.

**These results may still be useful as a separate ablation** (e.g., showing cosine vs linear gap is consistent even in a lower-performing config, or as an anchor_init=orthogonal datapoint).

| Model | Seed | Best Val PPL | Epochs | Training Time | Status |
|-------|------|-------------|--------|---------------|--------|
| SRA_B (cosine K=1→4) | 42 | **30.11** | 8/8 | 4h57m | DONE |
| StdMoE_B (linear K=1→4) | 42 | **29.94** | 8/8 | 5h21m | DONE |
| SRA_E (cosine K=2→4) | 137 | **30.02** | 8/8 | ~5h | DONE (pending confirm) |
| SRA_B (cosine K=1→4) | 137 | — | running | — | GPU 0, started 07:59 |
| StdMoE_B (linear K=1→4) | 137 | — | running | — | GPU 1, started 08:23 |
| StdMoE_E (linear K=2→4) | 137 | — | queued | — | GPU 2, after SRA_E/137 |
| SRA_E (cosine K=2→4) | 42 | — | queued | — | GPU 0, 3rd in queue |
| StdMoE_E (linear K=2→4) | 42 | — | queued | — | GPU 1, 3rd in queue |

#### Validation PPL trajectory (completed runs)

| Epoch | SRA_B/42 | StdMoE_B/42 | SRA_E/137 |
|-------|----------|-------------|-----------|
| 1 | 95.11 | 90.58 | 76.39 |
| 2 | 50.54 | 49.15 | 44.52 |
| 3 (K↑) | 39.48 | 38.30 | 36.77 |
| 4 | 34.73 | 34.17 | 33.43 |
| 5 | 32.09 | 31.71 | 31.27 |
| 6 | 30.75 | 30.35 | 30.32 |
| 7 | 30.12 | 29.94 | 30.02 |
| 8 | 30.11 | 29.97 | 30.02 |

**Observations from Round 1:**
- Linear routing (StdMoE_B) beats cosine (SRA_B) by 0.17 PPL — same direction as seed=19 (0.07 gap)
- SRA_E (K=2→4) converges to nearly same PPL as SRA_B (K=1→4): 30.02 vs 30.11
- PPL plateau after epoch 7 — similar convergence dynamics to original runs
- K=4 transition at epoch 3 shows clear inflection (same pattern as paper Fig 2)

### Diagnostic Tests (code regression investigation)

All new runs get PPL ~90-106 at epoch 1 vs original's ~25-30. Root cause: batched GEMM refactor.

| Test | Code | Seed | DeepSpeed | Epoch 1 PPL | Expected |
|------|------|------|-----------|-------------|----------|
| Original SRA_B | MicroExpert (old) | 19 | No | **29.73** | baseline |
| Round 2 SRA_B | Batched GEMM (new) | 42 | No | **96.15** | ~30 |
| Round 2 StdMoE_B | Batched GEMM (new) | 42 | No | **89.82** | ~28 |
| TEST seed=19 | Batched GEMM (new) | 19 | ZeRO-2 | **106.18** | ~30 |
| TEST old code seed=19 | MicroExpert (old) | 19 | No | **29.73** | ~30 ✅ |
| TEST init-fix seed=19 | Batched GEMM + normal(0.02) init | 19 | No | **95.36** | ~30 ❌ |
| TEST for-loop seed=19 | For-loop + stacked nn.Parameter + init fix | 19 | No | **93.86** | ~30 ❌ |
| TEST OLD csr + NEW sra/trainer | MicroExpert + new sra+trainer | 19 | No | **29.40** | ~30 ✅ |
| TEST NEW csr + OLD sra/trainer | Stacked GEMM + old sra+trainer | 19 | No | **97.38** | ~30 ❌ |
| TEST GEMM + full init fix | Stacked GEMM + init normal(0.02) + GPT2 w2 | 19 | No | **95.24** | ~30 ❌ |
| TEST ParameterList GEMM | nn.ParameterList + torch.stack + init fix | 19 | No | **95.26** | ~30 ❌ |

**Root cause CONFIRMED**: Bug is in `csr.py` forward pass logic (not init, not optimizer state).
Old MicroExpert + new sra/trainer works (29.40 ✅). New csr.py forward breaks regardless of:
- Init strategy (kaiming, normal 0.02, GPT2-scaled)
- Parameter storage (stacked nn.Parameter, nn.ParameterList)
- Dispatch method (batched GEMM, for-loop)
**Remaining suspects**: capacity factor truncation, token dispatch differences, temperature [1] vs [].
**Fix**: Use old MicroExpert csr.py for multi-seed runs. Investigate forward path later.

### Round 2 (matching original config — using OLD MicroExpert code)

Fix: Reverted to old MicroExpert csr.py (for-loop dispatch with nn.ModuleList).
See `researches/001_gemm_refactor_regression.md` for full root cause analysis.

Started 2026-03-25 00:49. Using all 3 GPUs.

### Per-Run Results (Round 2)

| Model | Seed | PPL | Dead | IC | EP | Status | Train Time |
|-------|------|-----|------|-----|-----|--------|------------|
| SRA B | 19 | 12.55 | 4% | 0.227 | 0.363 | ✅ DONE (baseline) | — |
| SRA B | 42 | **12.55** | 4% | 0.235 | 0.363 | ✅ DONE | 12.0h |
| SRA B | 137 | **12.60** | 6% | 0.234 | 0.361 | ✅ DONE | 12.2h |
| StdMoE B | 19 | 12.48 | 0% | 0.242 | 0.360 | ✅ DONE (baseline) | — |
| StdMoE B | 42 | **12.43** | 3% | 0.245 | 0.370 | ✅ DONE | 14.2h |
| StdMoE B | 137 | **12.44** | 5% | 0.263 | 0.353 | ✅ DONE | 14.6h |
| SRA E | 19 | 12.52 | 0% | 0.221 | 0.328 | ✅ DONE (baseline) | — |
| SRA E | 42 | **12.54** | 0% | 0.216 | 0.326 | ✅ DONE | 12.7h |
| SRA E | 137 | **12.50** | 0% | 0.219 | 0.324 | ✅ DONE | 12.7h |
| StdMoE E | 19 | 12.59 | 0% | 0.232 | 0.319 | ✅ DONE (baseline) | — |
| StdMoE E | 42 | **12.54** | 0% | 0.246 | 0.329 | ✅ DONE (resumed from ep 6) | 6.5h |
| StdMoE E | 137 | **12.58** | 0% | 0.243 | 0.327 | ✅ DONE (resumed, best from ep 7) | 4.2h |

**IC/EP note**: All values from completed full-training checkpoints analyzed via `analyze.py`.
All 12 runs complete as of 2026-03-27.

#### Validation PPL Trajectories (Round 2 completed runs)

| Epoch | SRA_B/42 | SRA_B/137 | StdMoE_B/42 | StdMoE_B/137 |
|-------|----------|-----------|-------------|--------------|
| 1 | 29.84 | 29.86 | 28.86 | 28.62 |
| 2 | 18.57 | 18.52 | 18.00 | 17.88 |
| 3 (K↑4) | 15.19 | 15.25 | 14.92 | 14.87 |
| 4 | 13.89 | 13.93 | 13.67 | 13.66 |
| 5 | 13.09 | 13.19 | 12.93 | 12.94 |
| 6 | 12.70 | 12.77 | 12.54 | 12.55 |
| 7 | **12.55** | **12.60** | 12.43 | 12.44 |
| 8 | 12.56 | 12.61 | **12.43** | **12.44** |

#### Dead Expert Trajectories (Round 2)

| Epoch | SRA_B/42 | SRA_B/137 | StdMoE_B/42 | StdMoE_B/137 |
|-------|----------|-----------|-------------|--------------|
| 1 | 19% | 19% | 9% | 8% |
| 2 | 16% | 18% | 9% | 8% |
| 3 (K↑4) | 4% | 5% | 2% | 4% |
| 4 | 5% | 7% | 2% | 5% |
| 5 | 5% | 7% | 2% | 5% |
| 6 | 5% | 6% | 3% | 5% |
| 7 | 4% | 6% | 3% | 5% |
| 8 | 4% | 6% | 3% | 5% |

### Aggregated Multi-Seed Results (all 12 runs complete, 2026-03-27)

| Model | Routing | K Schedule | Mean PPL | Std | Seeds (PPL) | Mean IC | IC Std | Mean EP | EP Std |
|-------|---------|------------|----------|-----|-------------|---------|--------|---------|--------|
| StdMoE B | Linear | K=1→4 | **12.45** | ±0.03 | 12.48, 12.43, 12.44 | 0.250 | ±0.011 | 0.361 | ±0.009 |
| SRA E | Cosine | K=2→4 | **12.52** | ±0.02 | 12.52, 12.54, 12.50 | 0.219 | ±0.003 | 0.326 | ±0.002 |
| SRA B | Cosine | K=1→4 | **12.57** | ±0.03 | 12.55, 12.55, 12.60 | 0.232 | ±0.004 | 0.362 | ±0.001 |
| StdMoE E | Linear | K=2→4 | **12.57** | ±0.02 | 12.59, 12.54, 12.58 | 0.240 | ±0.007 | 0.325 | ±0.005 |

**Observations (corrected 2026-03-27, using BEST PPL from training logs):**
- B schedule: StdMoE_B 12.45±0.03 vs SRA_B 12.57±0.03 → 0.12 gap, consistent across seeds
- E schedule: SRA_E 12.52±0.02 vs StdMoE_E 12.57±0.02 → reversed ordering (cosine wins by 0.05)
- StdMoE_E (12.57) now ties with SRA_B (12.57) — both second-tier after StdMoE_B
- Dead experts higher for cosine routing (4-6%) vs linear (0-5%)
- IC/EP gap is small: IC 0.018-0.021 (StdMoE slightly higher), EP 0.001 (essentially identical)
- SRA more stable across seeds: IC ±0.003-0.004 vs ±0.007-0.011, EP ±0.001-0.002 vs ±0.005-0.009
- **PPL audit fix (2026-03-27)**: SRA_E seed42 was 12.56 (final epoch), corrected to 12.54 (best); StdMoE_E seed42 was 12.62 (final epoch), corrected to 12.54 (best). Root cause: used final-epoch PPL instead of best-checkpoint PPL.

## Paper Impact

### If Round 1 results usable (as separate ablation)
- Could add a row/note showing cosine vs linear gap is consistent across configs
- Strengthens "direction of trend" argument even without matching PPL magnitude

### Round 2 (primary goal)
- Updates Tables 2 and 7 with mean ± std
- Strengthens "comparable IC/EP" claim with variance bars
- May change wording in Robustness section if variance is low
- Replaces "trend-based robustness" with actual multi-seed evidence
