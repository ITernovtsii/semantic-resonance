#!/usr/bin/env python3
"""Statistical analysis for seed-137 all-256-expert AI evaluation.

Mirrors run_statistical_analysis.py but uses the full-census all-256 data
from ai_scoring_s137_all256/ instead of seeds 19/42 sampled data.

Tests:
1. Mann-Whitney U on AI purity scores (SRA vs StdMoE per layer) with Holm correction
2. Cliff's delta effect sizes + bootstrap 95% CIs
3. Spearman/Kendall correlation: IC vs AI purity score
4. Correlation: vocab_entropy vs AI purity score
5. Chi-squared test on syntactic vs semantic type distribution
6. Overall summary with aggregated statistics

Usage:
    python3 experiments/002_monosemanticity/run_statistical_analysis_all256.py
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats as scipy_stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED137_DIR = os.path.join(BASE_DIR, 'outputs', 'seed137', 'subexp_a')
SCORES_DIR = os.path.join(BASE_DIR, 'outputs', 'ai_scoring_s137_all256')
LAYERS = [0, 1, 2, 3]
EVALUATORS = ['gemini', 'codex', 'opus']
MODELS_BLIND = ['model_A', 'model_B']


def load_blind_mapping():
    with open(os.path.join(SEED137_DIR, '_blind_mapping.json')) as f:
        return json.load(f)


def load_blind_csv(model_code, layer):
    """Load blind CSV, returning list of dicts keyed by expert_id."""
    path = os.path.join(SEED137_DIR, f'{model_code}_layer{layer}_blind.csv')
    entries = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                'expert_id': row['expert_id'],
                'vocab_entropy': float(row['vocab_entropy']) if row['vocab_entropy'] else None,
            })
    return entries


def load_analysis_csv(model_name, layer):
    """Load analysis CSV with IC/EP, returning list of dicts (same row order as blind)."""
    path = os.path.join(SEED137_DIR, f'{model_name}_layer{layer}_analysis.csv')
    entries = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                'original_expert_id': int(row['original_expert_id']),
                'IC': float(row['IC']),
                'EP': float(row['EP']),
                'vocab_entropy': float(row['vocab_entropy']),
            })
    return entries


def load_scores(evaluator, key, base_dir=None):
    """Load scores from a score file. Returns list of dicts or None."""
    if base_dir is None:
        base_dir = SCORES_DIR
    path = os.path.join(base_dir, f"{evaluator}_{key}_scores.csv")
    if not os.path.exists(path):
        return None
    results = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                results.append({
                    'expert_id': row['expert_id'],
                    'score': int(float(row['score'])),
                    'category': row.get('category', ''),
                    'type': row.get('type', 'mixed').lower().strip(),
                })
            except (ValueError, KeyError):
                continue
    return results if results else None


def load_all_scores_for_model_layer(evaluator, model_code, layer, base_dir=None):
    """Load scores from all chunks for a given model-layer."""
    all_scores = {}  # expert_id -> {score, category, type}

    # Try single file first
    key = f"s137_{model_code}_layer{layer}"
    scores = load_scores(evaluator, key, base_dir)
    if scores:
        for s in scores:
            all_scores[s['expert_id']] = s
        return all_scores

    # Try chunked files
    for chunk_idx in range(10):
        key = f"s137_{model_code}_layer{layer}_c{chunk_idx}"
        scores = load_scores(evaluator, key, base_dir)
        if scores is None:
            break
        for s in scores:
            all_scores[s['expert_id']] = s

    return all_scores if all_scores else None


def cliffs_delta(x, y):
    """Compute Cliff's delta effect size (vectorized)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    diffs = x[:, None] - y[None, :]
    more = np.sum(diffs > 0)
    less = np.sum(diffs < 0)
    return (more - less) / (len(x) * len(y))


def bootstrap_ci(x, y, stat_func, n_boot=2000, ci=0.95):
    """Bootstrap confidence interval for a statistic comparing two groups."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rng = np.random.RandomState(42)
    observed = stat_func(x, y)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        bx = x[rng.randint(0, len(x), len(x))]
        by = y[rng.randint(0, len(y), len(y))]
        boot_stats[i] = stat_func(bx, by)
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_stats, alpha * 100)
    hi = np.percentile(boot_stats, (1 - alpha) * 100)
    return observed, lo, hi


def holm_correction(pvalues):
    """Apply Holm-Bonferroni correction to a list of p-values."""
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    corrected = [None] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected[orig_idx] = min(1.0, p * (n - rank))
    # Enforce monotonicity
    for i in range(1, n):
        idx = indexed[i][0]
        prev_idx = indexed[i - 1][0]
        corrected[idx] = max(corrected[idx], corrected[prev_idx])
    return corrected


def main():
    mapping = load_blind_mapping()

    print("=" * 80)
    print("SEED 137 ALL-256-EXPERT STATISTICAL ANALYSIS")
    print("=" * 80)

    # ── Build merged dataset: scores + IC/EP + vocab_entropy ──
    # Structure: {model_name: [{expert_id, layer, IC, EP, vocab_entropy, {ev}_score, {ev}_type}]}
    merged = {'SRA-B': [], 'StdMoE-B': []}

    for model_code in MODELS_BLIND:
        model_name = mapping[model_code]
        for layer in LAYERS:
            blind_entries = load_blind_csv(model_code, layer)
            analysis_entries = load_analysis_csv(model_name, layer)

            # Load scores for all evaluators
            ev_scores = {}
            for ev in EVALUATORS:
                ev_scores[ev] = load_all_scores_for_model_layer(ev, model_code, layer)

            for i, (blind, analysis) in enumerate(zip(blind_entries, analysis_entries)):
                eid = blind['expert_id']
                entry = {
                    'expert_id': eid,
                    'layer': layer,
                    'IC': analysis['IC'],
                    'EP': analysis['EP'],
                    'vocab_entropy': analysis['vocab_entropy'],
                }
                for ev in EVALUATORS:
                    if ev_scores[ev] and eid in ev_scores[ev]:
                        entry[f'{ev}_score'] = ev_scores[ev][eid]['score']
                        entry[f'{ev}_type'] = ev_scores[ev][eid]['type']
                        entry[f'{ev}_category'] = ev_scores[ev][eid]['category']

                merged[model_name].append(entry)

    # Report coverage
    for model in ['SRA-B', 'StdMoE-B']:
        total = len(merged[model])
        for ev in EVALUATORS:
            scored = sum(1 for e in merged[model] if f'{ev}_score' in e)
            print(f"  {model} {ev}: {scored}/{total} experts scored")

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Mann-Whitney U on AI purity scores (per layer, Holm corrected)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("1. MANN-WHITNEY U: AI PURITY SCORES (SRA vs StdMoE per layer)")
    print("=" * 70)

    for ev in EVALUATORS:
        print(f"\n--- {ev.upper()} ---")
        pvalues = []
        results = []

        for layer in LAYERS:
            sra_scores = [e[f'{ev}_score'] for e in merged['SRA-B']
                          if e['layer'] == layer and f'{ev}_score' in e]
            std_scores = [e[f'{ev}_score'] for e in merged['StdMoE-B']
                          if e['layer'] == layer and f'{ev}_score' in e]

            if len(sra_scores) > 0 and len(std_scores) > 0:
                stat, p = scipy_stats.mannwhitneyu(sra_scores, std_scores, alternative='two-sided')
                d = cliffs_delta(sra_scores, std_scores)
                _, ci_lo, ci_hi = bootstrap_ci(
                    np.array(sra_scores), np.array(std_scores),
                    lambda x, y: cliffs_delta(x, y), n_boot=5000
                )
                pvalues.append(p)
                results.append({
                    'layer': layer,
                    'sra_mean': np.mean(sra_scores),
                    'std_mean': np.mean(std_scores),
                    'delta': d,
                    'ci_lo': ci_lo,
                    'ci_hi': ci_hi,
                    'p': p,
                    'n_sra': len(sra_scores),
                    'n_std': len(std_scores),
                })

        if not pvalues:
            print(f"  No data for {ev}")
            continue

        corrected = holm_correction(pvalues)

        print(f"  {'Layer':<8} {'SRA':>6} {'StdMoE':>8} {'n_sra':>6} {'n_std':>6} "
              f"{'Cliff d':>8} {'95% CI':>16} {'p':>10} {'p_holm':>10} {'Sig':>5}")
        for i, r in enumerate(results):
            sig = '*' if corrected[i] < 0.05 else ''
            print(f"  {r['layer']:<8} {r['sra_mean']:>6.2f} {r['std_mean']:>8.2f} "
                  f"{r['n_sra']:>6} {r['n_std']:>6} "
                  f"{r['delta']:>8.3f} [{r['ci_lo']:>6.3f}, {r['ci_hi']:>6.3f}] "
                  f"{r['p']:>10.4f} {corrected[i]:>10.4f} {sig:>5}")

        # Overall (pooled across layers)
        sra_all = [e[f'{ev}_score'] for e in merged['SRA-B'] if f'{ev}_score' in e]
        std_all = [e[f'{ev}_score'] for e in merged['StdMoE-B'] if f'{ev}_score' in e]
        if sra_all and std_all:
            stat, p = scipy_stats.mannwhitneyu(sra_all, std_all, alternative='two-sided')
            d = cliffs_delta(sra_all, std_all)
            _, ci_lo, ci_hi = bootstrap_ci(
                np.array(sra_all), np.array(std_all),
                lambda x, y: cliffs_delta(x, y), n_boot=5000
            )
            print(f"  {'POOLED':<8} {np.mean(sra_all):>6.2f} {np.mean(std_all):>8.2f} "
                  f"{len(sra_all):>6} {len(std_all):>6} "
                  f"{d:>8.3f} [{ci_lo:>6.3f}, {ci_hi:>6.3f}] "
                  f"{p:>10.4f} {'':>10} {'*' if p < 0.05 else '':>5}")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Correlation: IC vs AI purity score
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("2. CORRELATION: IC vs AI PURITY SCORE")
    print("=" * 70)

    for ev in EVALUATORS:
        ic_vals = []
        score_vals = []
        for model in ['SRA-B', 'StdMoE-B']:
            for entry in merged[model]:
                if f'{ev}_score' in entry:
                    ic_vals.append(entry['IC'])
                    score_vals.append(entry[f'{ev}_score'])

        if len(ic_vals) > 10:
            spearman_r, spearman_p = scipy_stats.spearmanr(ic_vals, score_vals)
            kendall_t, kendall_p = scipy_stats.kendalltau(ic_vals, score_vals)
            print(f"  {ev:>8}: Spearman r={spearman_r:.4f} (p={spearman_p:.6f}), "
                  f"Kendall τ={kendall_t:.4f} (p={kendall_p:.6f}), n={len(ic_vals)}")
        else:
            print(f"  {ev:>8}: insufficient paired data (n={len(ic_vals)})")

    # Per-model breakdown
    print("\n  Per-model:")
    for ev in EVALUATORS:
        for model in ['SRA-B', 'StdMoE-B']:
            ic_vals = [e['IC'] for e in merged[model] if f'{ev}_score' in e]
            score_vals = [e[f'{ev}_score'] for e in merged[model] if f'{ev}_score' in e]
            if len(ic_vals) > 10:
                r, p = scipy_stats.spearmanr(ic_vals, score_vals)
                print(f"    {ev:>8} {model:>10}: Spearman r={r:.4f} (p={p:.6f}), n={len(ic_vals)}")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Correlation: vocab_entropy vs AI purity score
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("3. CORRELATION: VOCAB ENTROPY vs AI PURITY SCORE")
    print("=" * 70)

    for ev in EVALUATORS:
        ve_vals = []
        score_vals = []
        for model in ['SRA-B', 'StdMoE-B']:
            for entry in merged[model]:
                if f'{ev}_score' in entry and entry.get('vocab_entropy') is not None:
                    ve_vals.append(entry['vocab_entropy'])
                    score_vals.append(entry[f'{ev}_score'])

        if len(ve_vals) > 10:
            spearman_r, spearman_p = scipy_stats.spearmanr(ve_vals, score_vals)
            print(f"  {ev:>8}: Spearman r={spearman_r:.4f} (p={spearman_p:.6f}), n={len(ve_vals)}")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Correlation: EP vs AI purity score
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("4. CORRELATION: EP vs AI PURITY SCORE")
    print("=" * 70)

    for ev in EVALUATORS:
        ep_vals = []
        score_vals = []
        for model in ['SRA-B', 'StdMoE-B']:
            for entry in merged[model]:
                if f'{ev}_score' in entry:
                    ep_vals.append(entry['EP'])
                    score_vals.append(entry[f'{ev}_score'])

        if len(ep_vals) > 10:
            spearman_r, spearman_p = scipy_stats.spearmanr(ep_vals, score_vals)
            print(f"  {ev:>8}: Spearman r={spearman_r:.4f} (p={spearman_p:.6f}), n={len(ep_vals)}")

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Chi-squared: Syntactic vs Semantic distribution
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("5. SYNTACTIC vs SEMANTIC DISTRIBUTION (chi-squared)")
    print("=" * 70)

    for ev in EVALUATORS:
        types = {'SRA-B': defaultdict(int), 'StdMoE-B': defaultdict(int)}
        for model in ['SRA-B', 'StdMoE-B']:
            for entry in merged[model]:
                t = entry.get(f'{ev}_type', '')
                if t in ('syntactic', 'semantic', 'mixed'):
                    types[model][t] += 1

        cats = ['syntactic', 'semantic', 'mixed']
        observed = np.array([[types['SRA-B'][c] for c in cats],
                             [types['StdMoE-B'][c] for c in cats]])
        if observed.sum() > 0:
            chi2, p, dof, expected = scipy_stats.chi2_contingency(observed)
            print(f"  {ev:>8}: χ²={chi2:.3f}, p={p:.4f}, dof={dof}")
            for model in ['SRA-B', 'StdMoE-B']:
                total = sum(types[model].values())
                if total > 0:
                    parts = ', '.join(f"{c}={types[model][c]} ({types[model][c]/total*100:.1f}%)"
                                      for c in cats)
                    print(f"           {model}: {parts} (n={total})")

    # ═══════════════════════════════════════════════════════════════════════
    # 6. Per-layer means and overall summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("6. SUMMARY: PER-EVALUATOR OVERALL AND PER-LAYER MEANS")
    print("=" * 70)

    print(f"\n  {'Evaluator':<12} {'SRA-B':>8} {'StdMoE-B':>10} {'Δ':>8} {'n(SRA)':>8} {'n(Std)':>8}")
    print("  " + "-" * 58)
    for ev in EVALUATORS:
        sra = [e[f'{ev}_score'] for e in merged['SRA-B'] if f'{ev}_score' in e]
        std = [e[f'{ev}_score'] for e in merged['StdMoE-B'] if f'{ev}_score' in e]
        if sra and std:
            print(f"  {ev:<12} {np.mean(sra):>8.2f} {np.mean(std):>10.2f} "
                  f"{np.mean(sra)-np.mean(std):>+8.2f} {len(sra):>8} {len(std):>8}")

    for ev in EVALUATORS:
        print(f"\n  {ev.upper()} per-layer:")
        print(f"    {'Layer':>5} {'SRA':>8} {'StdMoE':>8} {'Δ':>8} {'n':>5}")
        for layer in LAYERS:
            sra = [e[f'{ev}_score'] for e in merged['SRA-B']
                   if e['layer'] == layer and f'{ev}_score' in e]
            std = [e[f'{ev}_score'] for e in merged['StdMoE-B']
                   if e['layer'] == layer and f'{ev}_score' in e]
            if sra and std:
                print(f"    {layer:>5} {np.mean(sra):>8.2f} {np.mean(std):>8.2f} "
                      f"{np.mean(sra)-np.mean(std):>+8.2f} {len(sra):>5}")

    # ═══════════════════════════════════════════════════════════════════════
    # 7. Score distribution statistics
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("7. SCORE DISTRIBUTION (median, std, IQR)")
    print("=" * 70)

    for ev in EVALUATORS:
        print(f"\n  {ev.upper()}:")
        for model in ['SRA-B', 'StdMoE-B']:
            scores = [e[f'{ev}_score'] for e in merged[model] if f'{ev}_score' in e]
            if scores:
                arr = np.array(scores)
                q1, q3 = np.percentile(arr, [25, 75])
                print(f"    {model:>10}: mean={arr.mean():.2f}, median={np.median(arr):.1f}, "
                      f"std={arr.std():.2f}, IQR=[{q1:.0f}, {q3:.0f}], n={len(scores)}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
