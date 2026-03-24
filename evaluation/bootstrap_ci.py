"""
evaluation/bootstrap_ci.py
============================
Bootstrap confidence interval computation for experiment results.

Computes 95% CI for all metrics using bootstrap resampling (10000 iterations).
Also computes paired bootstrap test for comparing two systems.

Usage:
    PYTHONPATH=. python evaluation/bootstrap_ci.py \
        --results_file results/ablation/ablation_summary.json \
        --output_dir results/bootstrap
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval.

    Returns:
        {"mean": ..., "std": ..., "ci_lower": ..., "ci_upper": ..., "n": ...}
    """
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    n = len(arr)

    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means)
    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n": n,
    }


def paired_bootstrap_test(
    values_a: List[float],
    values_b: List[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired bootstrap significance test.

    Tests H0: mean(A) == mean(B) via bootstrap.
    Returns p-value and effect size (Cohen's d).
    """
    rng = np.random.RandomState(seed)
    a = np.array(values_a)
    b = np.array(values_b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    observed_diff = float(a.mean() - b.mean())
    diffs = a - b

    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs.append(diffs[idx].mean())

    boot_diffs = np.array(boot_diffs)

    # Two-sided p-value
    p_value = float(np.mean(np.abs(boot_diffs - boot_diffs.mean()) >= abs(observed_diff)))

    # Cohen's d
    pooled_std = float(np.sqrt((a.std()**2 + b.std()**2) / 2))
    cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0.0

    return {
        "observed_diff": round(observed_diff, 4),
        "p_value": round(p_value, 4),
        "cohens_d": round(cohens_d, 4),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def compute_all_cis(
    per_sample_results: List[dict],
    metrics: List[str] = None,
) -> Dict[str, dict]:
    """Compute bootstrap CIs for all metrics in per-sample results."""
    if metrics is None:
        metrics = ["u_f1_strict", "u_f1_relaxed", "conflict_f1", "temporal_f1", "iou"]

    cis = {}
    for metric in metrics:
        values = [r.get(metric, 0.0) for r in per_sample_results if metric in r]
        if values:
            cis[metric] = bootstrap_ci(values)
    return cis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True, help="JSON file with per-sample results")
    parser.add_argument("--output_dir", default="results/bootstrap")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.results_file) as f:
        data = json.load(f)

    # Handle different result formats
    if isinstance(data, dict) and "per_sample" in data:
        # Single system results
        cis = compute_all_cis(data["per_sample"])
        print(f"\nBootstrap 95% CIs (n={len(data['per_sample'])}):")
        for metric, ci in cis.items():
            print(f"  {metric:<20s}: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    elif isinstance(data, dict):
        # Multi-variant results (ablation)
        all_cis = {}
        for variant, vdata in data.items():
            if isinstance(vdata, dict) and "per_sample" in vdata:
                cis = compute_all_cis(vdata["per_sample"])
                all_cis[variant] = cis
                print(f"\n{variant} (n={len(vdata['per_sample'])}):")
                for metric, ci in cis.items():
                    print(f"  {metric:<20s}: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

        # Paired tests between full and ablations
        if "full" in data and "per_sample" in data["full"]:
            full_samples = data["full"]["per_sample"]
            print(f"\nPaired bootstrap tests (vs full):")
            for variant in data:
                if variant == "full" or "per_sample" not in data[variant]:
                    continue
                var_samples = data[variant]["per_sample"]
                for metric in ["u_f1_strict", "conflict_f1"]:
                    full_vals = [r.get(metric, 0) for r in full_samples]
                    var_vals = [r.get(metric, 0) for r in var_samples]
                    test = paired_bootstrap_test(full_vals, var_vals)
                    sig = "***" if test["significant_001"] else ("*" if test["significant_005"] else "ns")
                    print(f"  full vs {variant} [{metric}]: "
                          f"Δ={test['observed_diff']:+.4f} p={test['p_value']:.4f} {sig}")

        data_out = all_cis
    else:
        print("Unrecognized format")
        return

    output_file = os.path.join(args.output_dir, "bootstrap_results.json")
    final_data = data_out if 'data_out' in dir() else cis
    with open(output_file, "w") as f:
        json.dump(final_data, f, indent=2)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
