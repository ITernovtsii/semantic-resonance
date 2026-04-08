#!/usr/bin/env python3
"""Independent statistical verification for subexp_a seed137 all256."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parent
SCORES_DIR = ROOT / "outputs" / "ai_scoring_s137_all256"
ANALYSIS_DIR = ROOT / "outputs" / "seed137" / "subexp_a"
LAYERS = (0, 1, 2, 3)
EVALUATORS = ("gemini", "codex", "opus")
MODEL_CODES = ("model_A", "model_B")


CLAIMED = {
    "means": {
        "gemini": {"SRA-B": 7.32, "StdMoE-B": 7.20, "delta": 0.12},
        "codex": {"SRA-B": 6.86, "StdMoE-B": 6.72, "delta": 0.14},
        "opus": {"SRA-B": 5.95, "StdMoE-B": 5.76, "delta": 0.18},
    },
    "mw": {
        "gemini": {"delta": 0.034, "p": 0.18},
        "codex": {"delta": 0.041, "p": 0.11},
        "opus": {"delta": 0.032, "p": 0.21},
    },
    "ic_spearman": {
        "gemini": {"r": -0.053, "p": 0.018},
        "codex": {"r": -0.097, "p": 0.000011},
        "opus": {"r": -0.139, "p": 0.000001},
    },
    "ve_spearman": {
        "gemini": {"r": -0.127},
        "codex": {"r": -0.347},
        "opus": {"r": -0.227},
    },
    "chi2": {
        "gemini": {"chi2": 2.45, "p": 0.29},
        "codex": {"chi2": 5.59, "p": 0.06},
        "opus": {"chi2": 0.27, "p": 0.87},
    },
}


@dataclass
class MetricCheck:
    metric: str
    evaluator: str
    field: str
    claimed: float
    computed: float
    diff: float
    matched: bool


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    diffs = a[:, None] - b[None, :]
    more = np.sum(diffs > 0)
    less = np.sum(diffs < 0)
    return float((more - less) / (len(a) * len(b)))


def load_mapping() -> dict[str, str]:
    with (ANALYSIS_DIR / "_blind_mapping.json").open() as f:
        return json.load(f)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def load_scores(evaluator: str, model_code: str, layer: int) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for chunk in range(10):
        fp = SCORES_DIR / f"{evaluator}_s137_{model_code}_layer{layer}_c{chunk}_scores.csv"
        if not fp.exists():
            break
        for row in read_tsv(fp):
            out[row["expert_id"]] = row
    return out


def build_merged() -> dict[str, list[dict[str, object]]]:
    mapping = load_mapping()
    merged = {"SRA-B": [], "StdMoE-B": []}
    for model_code in MODEL_CODES:
        model_name = mapping[model_code]
        for layer in LAYERS:
            blind_rows = read_csv(ANALYSIS_DIR / f"{model_code}_layer{layer}_blind.csv")
            analysis_rows = read_csv(ANALYSIS_DIR / f"{model_name}_layer{layer}_analysis.csv")
            if len(blind_rows) != len(analysis_rows):
                raise ValueError(
                    f"Row count mismatch: {model_code} layer{layer}, "
                    f"blind={len(blind_rows)} analysis={len(analysis_rows)}"
                )

            scores_by_ev = {
                ev: load_scores(ev, model_code, layer)
                for ev in EVALUATORS
            }

            for b, a in zip(blind_rows, analysis_rows):
                entry: dict[str, object] = {
                    "expert_id": b["expert_id"],
                    "layer": layer,
                    "IC": float(a["IC"]),
                    "EP": float(a["EP"]),
                    "vocab_entropy": float(a["vocab_entropy"]),
                }
                for ev in EVALUATORS:
                    row = scores_by_ev[ev].get(b["expert_id"])
                    if row is None:
                        continue
                    entry[f"{ev}_score"] = int(float(row["score"]))
                    entry[f"{ev}_type"] = row.get("type", "").strip().lower()
                merged[model_name].append(entry)
    return merged


def approx_match(computed: float, claimed: float, tol: float) -> bool:
    return math.isfinite(computed) and abs(computed - claimed) <= tol


def main() -> None:
    merged = build_merged()
    checks: list[MetricCheck] = []

    def add(metric: str, evaluator: str, field: str, claimed: float, computed: float, tol: float) -> None:
        checks.append(
            MetricCheck(
                metric=metric,
                evaluator=evaluator,
                field=field,
                claimed=claimed,
                computed=computed,
                diff=computed - claimed,
                matched=approx_match(computed, claimed, tol),
            )
        )

    # 1) Overall means
    for ev in EVALUATORS:
        sra = [x[f"{ev}_score"] for x in merged["SRA-B"] if f"{ev}_score" in x]
        std = [x[f"{ev}_score"] for x in merged["StdMoE-B"] if f"{ev}_score" in x]
        sra_m = float(np.mean(sra))
        std_m = float(np.mean(std))
        d = sra_m - std_m
        add("Overall mean", ev, "SRA-B", CLAIMED["means"][ev]["SRA-B"], sra_m, 0.0051)
        add("Overall mean", ev, "StdMoE-B", CLAIMED["means"][ev]["StdMoE-B"], std_m, 0.0051)
        add("Overall mean", ev, "delta", CLAIMED["means"][ev]["delta"], d, 0.0051)

    # 2) Mann-Whitney pooled
    for ev in EVALUATORS:
        sra = np.array([x[f"{ev}_score"] for x in merged["SRA-B"] if f"{ev}_score" in x], dtype=float)
        std = np.array([x[f"{ev}_score"] for x in merged["StdMoE-B"] if f"{ev}_score" in x], dtype=float)
        _, p = stats.mannwhitneyu(sra, std, alternative="two-sided")
        d = cliffs_delta(sra, std)
        add("Mann-Whitney pooled", ev, "cliffs_delta", CLAIMED["mw"][ev]["delta"], float(d), 0.0006)
        add("Mann-Whitney pooled", ev, "p_value", CLAIMED["mw"][ev]["p"], float(p), 0.0051)

    # 3) IC vs purity Spearman
    for ev in EVALUATORS:
        ic_vals, score_vals = [], []
        for model in ("SRA-B", "StdMoE-B"):
            for row in merged[model]:
                if f"{ev}_score" in row:
                    ic_vals.append(float(row["IC"]))
                    score_vals.append(float(row[f"{ev}_score"]))
        r, p = stats.spearmanr(ic_vals, score_vals)
        add("IC vs purity", ev, "spearman_r", CLAIMED["ic_spearman"][ev]["r"], float(r), 0.0006)
        # For claimed p<0.000001, compare against that threshold value.
        p_claim = CLAIMED["ic_spearman"][ev]["p"]
        add("IC vs purity", ev, "p_value", p_claim, float(p), 0.000006 if p_claim <= 1e-6 else 0.0011)

    # 4) Vocab entropy vs purity Spearman
    for ev in EVALUATORS:
        ve_vals, score_vals = [], []
        for model in ("SRA-B", "StdMoE-B"):
            for row in merged[model]:
                if f"{ev}_score" in row:
                    ve_vals.append(float(row["vocab_entropy"]))
                    score_vals.append(float(row[f"{ev}_score"]))
        r, _ = stats.spearmanr(ve_vals, score_vals)
        add("Vocab entropy vs purity", ev, "spearman_r", CLAIMED["ve_spearman"][ev]["r"], float(r), 0.0006)

    # 5) Chi-squared type distribution (syntactic, semantic, mixed)
    for ev in EVALUATORS:
        cats = ("syntactic", "semantic", "mixed")
        obs = []
        for model in ("SRA-B", "StdMoE-B"):
            counts = {c: 0 for c in cats}
            for row in merged[model]:
                t = row.get(f"{ev}_type", "")
                if t in counts:
                    counts[t] += 1
            obs.append([counts[c] for c in cats])
        chi2, p, _, _ = stats.chi2_contingency(np.array(obs))
        add("Chi-squared types", ev, "chi2", CLAIMED["chi2"][ev]["chi2"], float(chi2), 0.015)
        add("Chi-squared types", ev, "p_value", CLAIMED["chi2"][ev]["p"], float(p), 0.0051)

    # Print report
    print("Independent verification report")
    print("=" * 80)
    for c in checks:
        status = "MATCH" if c.matched else "DIFFERS"
        print(
            f"{status:7} | {c.metric:24} | {c.evaluator:6} | {c.field:11} | "
            f"claimed={c.claimed:.9g} computed={c.computed:.9g} diff={c.diff:+.9g}"
        )

    print("=" * 80)
    mismatches = [c for c in checks if not c.matched]
    print(f"Summary: {len(checks) - len(mismatches)}/{len(checks)} fields matched tolerance.")
    if mismatches:
        print("Mismatches:")
        for c in mismatches:
            print(
                f"  - {c.metric} / {c.evaluator} / {c.field}: "
                f"claimed={c.claimed:.9g}, computed={c.computed:.9g}, diff={c.diff:+.9g}"
            )

    # Additional consistency flags
    for ev in EVALUATORS:
        n_sra = sum(1 for r in merged["SRA-B"] if f"{ev}_score" in r)
        n_std = sum(1 for r in merged["StdMoE-B"] if f"{ev}_score" in r)
        print(f"Coverage {ev:6}: n(SRA-B)={n_sra}, n(StdMoE-B)={n_std}")


if __name__ == "__main__":
    main()
