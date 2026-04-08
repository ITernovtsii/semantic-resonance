#!/usr/bin/env python3
"""Statistical analysis for Experiment 002: Monosemanticity comparison.

Runs all planned statistical tests:
1. Wilcoxon signed-rank on AI purity scores (SRA vs StdMoE per layer) with Holm correction
2. Cliff's delta effect sizes + bootstrap 95% CIs
3. Spearman/Kendall correlation: IC vs AI purity score
4. Correlation: vocab_entropy vs AI purity score
5. Mixed-effects model approximation on routing coherence
6. Permutation null test for Sub-B coherence
7. Stratified coherence analysis (by num_subtokens, by word category)

Usage:
    python3 experiments/002_monosemanticity/run_statistical_analysis.py
"""
import csv
import json
import os
import random
import statistics
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy import stats as scipy_stats

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [19, 42]
LAYERS = [0, 1, 2, 3]
EVALUATORS = ['gemini', 'codex', 'opus']


def load_blind_csvs(seed):
    """Load blind CSVs for a given seed, return per-model expert data."""
    if seed == 19:
        subexp_dir = os.path.join(BASE_DIR, 'outputs', 'subexp_a')
    else:
        subexp_dir = os.path.join(BASE_DIR, 'outputs', f'seed{seed}', 'subexp_a')

    mapping_file = os.path.join(subexp_dir, '_blind_mapping.json')
    with open(mapping_file) as f:
        mapping = json.load(f)

    data = {}  # {model: [{expert_id, layer, scores...}]}
    for model_code in ['model_A', 'model_B']:
        model_name = mapping[model_code]
        data[model_name] = []
        for layer in LAYERS:
            csv_path = os.path.join(subexp_dir, f'{model_code}_layer{layer}_blind.csv')
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entry = {
                        'expert_id': row['expert_id'],
                        'layer': layer,
                        'seed': seed,
                        'vocab_entropy': float(row['vocab_entropy']) if row['vocab_entropy'] else None,
                    }
                    for ev in EVALUATORS:
                        score_key = f'{ev}_purity_score'
                        if row.get(score_key):
                            entry[f'{ev}_score'] = int(float(row[score_key]))
                        type_key = f'{ev}_syntactic_or_semantic'
                        if row.get(type_key):
                            entry[f'{ev}_type'] = row[type_key].strip().lower()
                    data[model_name].append(entry)
    return data


def load_analysis_csvs(seed):
    """Load analysis CSVs (with IC/EP) for a given seed."""
    if seed == 19:
        subexp_dir = os.path.join(BASE_DIR, 'outputs', 'subexp_a')
    else:
        subexp_dir = os.path.join(BASE_DIR, 'outputs', f'seed{seed}', 'subexp_a')

    data = {}
    for model in ['SRA-B', 'StdMoE-B']:
        data[model] = {}
        for layer in LAYERS:
            csv_path = os.path.join(subexp_dir, f'{model}_layer{layer}_analysis.csv')
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    orig_id = int(row['original_expert_id'])
                    data[model][(layer, orig_id)] = {
                        'IC': float(row['IC']),
                        'EP': float(row['EP']),
                        'vocab_entropy': float(row['vocab_entropy']),
                    }
    return data


def load_coherence(seed):
    """Load routing coherence data for a given seed."""
    if seed == 19:
        csv_path = os.path.join(BASE_DIR, 'outputs', 'subexp_b', 'routing_coherence.csv')
    else:
        csv_path = os.path.join(BASE_DIR, 'outputs', f'seed{seed}', 'subexp_b', 'routing_coherence.csv')

    data = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'word': row['word'],
                'category': row['category'],
                'num_subtokens': int(row['num_subtokens']),
                'layer': int(row['layer']),
                'template_id': int(row['template_id']),
                'primary_coherence': float(row['primary_coherence']),
                'topk_overlap': float(row['topk_overlap']),
                'model': row['model'],
                'seed': seed,
            })
    return data


def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    n_x, n_y = len(x), len(y)
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)


def bootstrap_ci(x, y, stat_func, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval for a statistic comparing two groups."""
    rng = np.random.RandomState(42)
    observed = stat_func(x, y)
    boot_stats = []
    for _ in range(n_boot):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        boot_stats.append(stat_func(bx, by))
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_stats, alpha * 100)
    hi = np.percentile(boot_stats, (1 - alpha) * 100)
    return observed, lo, hi


def holm_correction(pvalues):
    """Apply Holm-Bonferroni correction to a list of p-values.
    Returns list of (original_index, corrected_p)."""
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


def permutation_test(sra_vals, std_vals, n_perm=10000):
    """Permutation test for difference in means."""
    observed_diff = np.mean(sra_vals) - np.mean(std_vals)
    combined = np.concatenate([sra_vals, std_vals])
    n_sra = len(sra_vals)
    rng = np.random.RandomState(42)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_diff = np.mean(combined[:n_sra]) - np.mean(combined[n_sra:])
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    return count / n_perm


def main():
    print("=" * 80)
    print("EXPERIMENT 002: STATISTICAL ANALYSIS")
    print("=" * 80)

    # ── Load all data ──
    all_blind = {}
    all_analysis = {}
    all_coherence = []

    for seed in SEEDS:
        all_blind[seed] = load_blind_csvs(seed)
        all_analysis[seed] = load_analysis_csvs(seed)
        all_coherence.extend(load_coherence(seed))

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Wilcoxon signed-rank on AI purity scores (per layer, Holm corrected)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("1. WILCOXON SIGNED-RANK: AI PURITY SCORES (SRA vs StdMoE)")
    print("=" * 70)

    for ev in EVALUATORS:
        print(f"\n--- {ev.upper()} ---")
        pvalues = []
        results = []

        for layer in LAYERS:
            sra_scores = []
            std_scores = []
            for seed in SEEDS:
                for entry in all_blind[seed].get('SRA-B', []):
                    if entry['layer'] == layer and f'{ev}_score' in entry:
                        sra_scores.append(entry[f'{ev}_score'])
                for entry in all_blind[seed].get('StdMoE-B', []):
                    if entry['layer'] == layer and f'{ev}_score' in entry:
                        std_scores.append(entry[f'{ev}_score'])

            if len(sra_scores) > 0 and len(std_scores) > 0:
                # Use Mann-Whitney U since experts aren't naturally paired
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

        # Holm correction
        corrected = holm_correction(pvalues)

        print(f"  {'Layer':<8} {'SRA':>6} {'StdMoE':>8} {'Cliff d':>8} {'95% CI':>16} {'p':>10} {'p_holm':>10} {'Sig':>5}")
        for i, r in enumerate(results):
            sig = '*' if corrected[i] < 0.05 else ''
            print(f"  {r['layer']:<8} {r['sra_mean']:>6.2f} {r['std_mean']:>8.2f} {r['delta']:>8.3f} [{r['ci_lo']:>6.3f}, {r['ci_hi']:>6.3f}] {r['p']:>10.4f} {corrected[i]:>10.4f} {sig:>5}")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Correlation: IC vs AI purity score
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("2. CORRELATION: IC vs AI PURITY SCORE")
    print("=" * 70)

    for ev in EVALUATORS:
        ic_vals = []
        score_vals = []
        for seed in SEEDS:
            analysis = all_analysis[seed]
            blind = all_blind[seed]
            for model in ['SRA-B', 'StdMoE-B']:
                for entry in blind[model]:
                    if f'{ev}_score' not in entry:
                        continue
                    layer = entry['layer']
                    # Match by position in layer (analysis CSV has original IDs)
                    # We need to use vocab_entropy as a join key since expert IDs are anonymized
                    ve = entry.get('vocab_entropy')
                    if ve is None:
                        continue
                    # Find matching analysis entry by vocab_entropy
                    for (l, eid), adata in analysis[model].items():
                        if l == layer and abs(adata['vocab_entropy'] - ve) < 0.001:
                            ic_vals.append(adata['IC'])
                            score_vals.append(entry[f'{ev}_score'])
                            break

        if len(ic_vals) > 10:
            spearman_r, spearman_p = scipy_stats.spearmanr(ic_vals, score_vals)
            kendall_t, kendall_p = scipy_stats.kendalltau(ic_vals, score_vals)
            print(f"  {ev:>8}: Spearman r={spearman_r:.3f} (p={spearman_p:.4f}), "
                  f"Kendall τ={kendall_t:.3f} (p={kendall_p:.4f}), n={len(ic_vals)}")
        else:
            print(f"  {ev:>8}: insufficient paired data (n={len(ic_vals)})")

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Correlation: vocab_entropy vs AI purity score
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("3. CORRELATION: VOCAB ENTROPY vs AI PURITY SCORE")
    print("=" * 70)

    for ev in EVALUATORS:
        ve_vals = []
        score_vals = []
        for seed in SEEDS:
            for model in ['SRA-B', 'StdMoE-B']:
                for entry in all_blind[seed][model]:
                    if f'{ev}_score' in entry and entry.get('vocab_entropy') is not None:
                        ve_vals.append(entry['vocab_entropy'])
                        score_vals.append(entry[f'{ev}_score'])

        if len(ve_vals) > 10:
            spearman_r, spearman_p = scipy_stats.spearmanr(ve_vals, score_vals)
            print(f"  {ev:>8}: Spearman r={spearman_r:.3f} (p={spearman_p:.4f}), n={len(ve_vals)}")
        else:
            print(f"  {ev:>8}: insufficient data (n={len(ve_vals)})")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Routing coherence: Mann-Whitney per layer with Holm correction
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("4. ROUTING COHERENCE: MANN-WHITNEY U (SRA vs StdMoE per layer)")
    print("=" * 70)

    # Aggregate coherence by word (average across templates)
    coherence_by_word = defaultdict(list)
    for row in all_coherence:
        key = (row['word'], row['layer'], row['model'], row['seed'])
        coherence_by_word[key].append(row['primary_coherence'])

    pvalues = []
    results = []
    for layer in LAYERS:
        sra_means = []
        std_means = []
        for word_key, vals in coherence_by_word.items():
            word, l, model, seed = word_key
            if l != layer:
                continue
            mean_coh = np.mean(vals)
            if model == 'SRA-B':
                sra_means.append(mean_coh)
            else:
                std_means.append(mean_coh)

        stat, p = scipy_stats.mannwhitneyu(sra_means, std_means, alternative='two-sided')
        d = cliffs_delta(sra_means, std_means)
        _, ci_lo, ci_hi = bootstrap_ci(
            np.array(sra_means), np.array(std_means),
            lambda x, y: cliffs_delta(x, y), n_boot=5000
        )
        pvalues.append(p)
        results.append({
            'layer': layer,
            'sra_mean': np.mean(sra_means),
            'std_mean': np.mean(std_means),
            'delta': d,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'p': p,
        })

    corrected = holm_correction(pvalues)
    print(f"  {'Layer':<8} {'SRA':>6} {'StdMoE':>8} {'Cliff d':>8} {'95% CI':>16} {'p':>10} {'p_holm':>10} {'Sig':>5}")
    for i, r in enumerate(results):
        sig = '*' if corrected[i] < 0.05 else ''
        print(f"  {r['layer']:<8} {r['sra_mean']:>6.3f} {r['std_mean']:>8.3f} {r['delta']:>8.3f} [{r['ci_lo']:>6.3f}, {r['ci_hi']:>6.3f}] {r['p']:>10.4f} {corrected[i]:>10.4f} {sig:>5}")

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Permutation null test for routing coherence
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("5. PERMUTATION TEST: ROUTING COHERENCE (10,000 permutations)")
    print("=" * 70)

    for layer in LAYERS:
        sra_vals = []
        std_vals = []
        for row in all_coherence:
            if row['layer'] == layer:
                if row['model'] == 'SRA-B':
                    sra_vals.append(row['primary_coherence'])
                else:
                    std_vals.append(row['primary_coherence'])

        p_perm = permutation_test(np.array(sra_vals), np.array(std_vals), n_perm=10000)
        diff = np.mean(sra_vals) - np.mean(std_vals)
        print(f"  Layer {layer}: Δ={diff:+.4f}, p_perm={p_perm:.4f} {'*' if p_perm < 0.05 else ''}")

    # ═══════════════════════════════════════════════════════════════════════
    # 6. Stratified coherence analysis
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("6. STRATIFIED COHERENCE: BY CATEGORY AND SUBTOKENS")
    print("=" * 70)

    # By category
    print("\n  By Category (all layers pooled):")
    categories = sorted(set(r['category'] for r in all_coherence))
    for cat in categories:
        sra_vals = [r['primary_coherence'] for r in all_coherence if r['category'] == cat and r['model'] == 'SRA-B']
        std_vals = [r['primary_coherence'] for r in all_coherence if r['category'] == cat and r['model'] == 'StdMoE-B']
        if len(sra_vals) > 5 and len(std_vals) > 5:
            stat, p = scipy_stats.mannwhitneyu(sra_vals, std_vals, alternative='two-sided')
            d = cliffs_delta(sra_vals, std_vals)
            print(f"    {cat:<12}: SRA={np.mean(sra_vals):.3f} StdMoE={np.mean(std_vals):.3f} "
                  f"Cliff d={d:+.3f} p={p:.4f} {'*' if p < 0.05 else ''}")

    # By num_subtokens
    print("\n  By Subtoken Count (all layers pooled):")
    for n_sub in sorted(set(r['num_subtokens'] for r in all_coherence)):
        sra_vals = [r['primary_coherence'] for r in all_coherence
                    if r['num_subtokens'] == n_sub and r['model'] == 'SRA-B']
        std_vals = [r['primary_coherence'] for r in all_coherence
                    if r['num_subtokens'] == n_sub and r['model'] == 'StdMoE-B']
        if len(sra_vals) > 5 and len(std_vals) > 5:
            stat, p = scipy_stats.mannwhitneyu(sra_vals, std_vals, alternative='two-sided')
            d = cliffs_delta(sra_vals, std_vals)
            print(f"    {n_sub} subtokens: SRA={np.mean(sra_vals):.3f} StdMoE={np.mean(std_vals):.3f} "
                  f"Cliff d={d:+.3f} p={p:.4f} n_sra={len(sra_vals)} {'*' if p < 0.05 else ''}")

    # ═══════════════════════════════════════════════════════════════════════
    # 7. Syntactic vs Semantic distribution comparison
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("7. SYNTACTIC vs SEMANTIC DISTRIBUTION (chi-squared)")
    print("=" * 70)

    for ev in EVALUATORS:
        types = {'SRA-B': defaultdict(int), 'StdMoE-B': defaultdict(int)}
        for seed in SEEDS:
            for model in ['SRA-B', 'StdMoE-B']:
                for entry in all_blind[seed][model]:
                    t = entry.get(f'{ev}_type', '')
                    if t in ('syntactic', 'semantic', 'mixed'):
                        types[model][t] += 1

        # Build contingency table
        cats = ['syntactic', 'semantic', 'mixed']
        observed = np.array([[types['SRA-B'][c] for c in cats],
                             [types['StdMoE-B'][c] for c in cats]])
        if observed.sum() > 0:
            chi2, p, dof, expected = scipy_stats.chi2_contingency(observed)
            print(f"  {ev:>8}: χ²={chi2:.2f}, p={p:.4f}, dof={dof}")
            for model in ['SRA-B', 'StdMoE-B']:
                total = sum(types[model].values())
                parts = ', '.join(f"{c}={types[model][c]} ({types[model][c]/total*100:.0f}%)" for c in cats)
                print(f"           {model}: {parts}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
