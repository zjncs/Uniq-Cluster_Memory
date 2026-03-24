"""
experiments/run_tcp_ablation.py
================================
TCP (Temporal Constraint Propagation) ablation experiment.

Compares full pipeline (with TCP) vs pipeline without TCP.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.uniq_cluster_memory.pipeline import UniqueClusterMemoryPipeline
from src.uniq_cluster_memory.schema import CanonicalMemory
from evaluation.temporal_eval import compute_temporal_metrics, TemporalMetrics

import re


def load_sample(sample_dir: Path) -> dict:
    meta = json.load(open(sample_dir / "metadata.json"))
    dialogue = [json.loads(l) for l in open(sample_dir / "dialogue.jsonl")]
    gt_canonical = [json.loads(l) for l in open(sample_dir / "canonical_gt.jsonl")]
    gt_conflicts = []
    if (sample_dir / "conflict_gt.jsonl").exists():
        gt_conflicts = [json.loads(l) for l in open(sample_dir / "conflict_gt.jsonl")]
    return {
        "meta": meta,
        "dialogue": dialogue,
        "gt_canonical": gt_canonical,
        "gt_conflicts": gt_conflicts,
        "dialogue_id": meta["dialogue_id"],
    }


def gt_to_memories(gt_list: list, patient_id: str) -> List[CanonicalMemory]:
    mems = []
    for rec in gt_list:
        mems.append(CanonicalMemory(
            patient_id=patient_id,
            attribute=rec.get("attribute", ""),
            value=rec.get("value", ""),
            unit=rec.get("unit", ""),
            time_scope=rec.get("time_scope", "global"),
            confidence=rec.get("confidence", 1.0),
            conflict_flag=rec.get("conflict_flag", False),
            start_time=rec.get("start_time", ""),
            end_time=rec.get("end_time", ""),
        ))
    return mems


def norm(s):
    return re.sub(r"[^\w]", "", s.strip().lower())


def compute_metrics(pred: List[CanonicalMemory], gt_mems: List[CanonicalMemory], gt_conflicts: list):
    """Compute U-F1 strict/relaxed and C-F1."""
    # U-F1 strict: exact (attribute, value, time_scope) match
    pred_keys_s = {(norm(m.attribute), norm(m.value), norm(m.time_scope)) for m in pred}
    gt_keys_s = {(norm(m.attribute), norm(m.value), norm(m.time_scope)) for m in gt_mems}

    # U-F1 relaxed: (attribute, value) match
    pred_keys_r = {(norm(m.attribute), norm(m.value)) for m in pred}
    gt_keys_r = {(norm(m.attribute), norm(m.value)) for m in gt_mems}

    def f1(pred_set, gt_set):
        if not pred_set and not gt_set:
            return 1.0
        if not pred_set or not gt_set:
            return 0.0
        tp = len(pred_set & gt_set)
        p = tp / len(pred_set) if pred_set else 0
        r = tp / len(gt_set) if gt_set else 0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    u_f1_s = f1(pred_keys_s, gt_keys_s)
    u_f1_r = f1(pred_keys_r, gt_keys_r)

    # C-F1: conflict detection
    pred_conflicts = {(norm(m.attribute), norm(m.time_scope)) for m in pred if m.conflict_flag}
    gt_conflict_keys = set()
    for c in gt_conflicts:
        attr = norm(c.get("attribute", ""))
        ts = norm(c.get("time_scope", ""))
        gt_conflict_keys.add((attr, ts))

    c_f1 = f1(pred_conflicts, gt_conflict_keys)

    # T-F1: temporal F1 (strict with time)
    t_metrics = compute_temporal_metrics(pred, gt_mems)
    t_f1 = t_metrics.temporal_f1

    # IoU
    all_pred = pred_keys_s | pred_keys_r
    all_gt = gt_keys_s | gt_keys_r
    iou = len(all_pred & all_gt) / len(all_pred | all_gt) if (all_pred | all_gt) else 1.0

    return {
        "u_f1_strict": round(u_f1_s, 4),
        "u_f1_relaxed": round(u_f1_r, 4),
        "conflict_f1": round(c_f1, 4),
        "temporal_f1": round(t_f1, 4),
        "iou": round(iou, 4),
    }


def run_variant(name, data_path, max_samples, use_tcp):
    """Run one variant (with or without TCP)."""
    import src.uniq_cluster_memory.pipeline as pipe_mod
    from src.uniq_cluster_memory.temporal_reasoning import constraint_propagation as tcp_mod
    from src.uniq_cluster_memory.temporal_reasoning.constraint_propagation import TCPResult, run_tcp as real_run_tcp

    if not use_tcp:
        def noop_tcp(memories, max_iterations=50):
            return memories, TCPResult(n_nodes=len(memories))
        pipe_mod.run_tcp = noop_tcp
    else:
        pipe_mod.run_tcp = real_run_tcp

    pipeline = UniqueClusterMemoryPipeline(use_embedding=True)
    data_dir = Path(data_path)
    sample_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])[:max_samples]

    results = []
    tcp_stats = {"total_inconsistencies": 0, "total_tightened": 0}

    for i, sd in enumerate(sample_dirs):
        sample = load_sample(sd)
        did = sample["dialogue_id"]
        t0 = time.time()

        memories = pipeline.build_memory(sample["dialogue"], did)
        elapsed = time.time() - t0

        if use_tcp and pipeline._last_tcp_result:
            tcp_stats["total_inconsistencies"] += pipeline._last_tcp_result.n_inconsistencies
            tcp_stats["total_tightened"] += pipeline._last_tcp_result.n_tightened

        gt_mems = gt_to_memories(sample["gt_canonical"], did)
        metrics = compute_metrics(memories, gt_mems, sample.get("gt_conflicts", []))
        metrics["sample_id"] = did
        metrics["latency"] = round(elapsed, 1)
        results.append(metrics)

        print(f"  [{i+1}/{len(sample_dirs)}] {did}: "
              f"U-F1(S)={metrics['u_f1_strict']:.3f} C-F1={metrics['conflict_f1']:.3f} "
              f"T-F1={metrics['temporal_f1']:.3f} ({elapsed:.1f}s)")

    # Aggregate
    n = len(results)
    avg = {}
    for key in ["u_f1_strict", "u_f1_relaxed", "conflict_f1", "temporal_f1", "iou", "latency"]:
        vals = [r[key] for r in results]
        avg[key] = round(sum(vals) / n, 4) if n else 0

    # Restore
    pipe_mod.run_tcp = real_run_tcp

    return {"avg": avg, "tcp_stats": tcp_stats, "n": n, "per_sample": results}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/raw/med_longmem")
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--output_dir", default="results/tcp_ablation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    variants = [
        ("full_with_tcp", True),
        ("w/o_tcp", False),
    ]

    all_results = {}
    for vname, use_tcp in variants:
        print(f"\n{'=' * 60}")
        print(f"  [{vname}] TCP={'ON' if use_tcp else 'OFF'}")
        print(f"{'=' * 60}")
        all_results[vname] = run_variant(vname, args.data_path, args.max_samples, use_tcp)
        a = all_results[vname]["avg"]
        print(f"\n  Avg: U-F1(S)={a['u_f1_strict']:.4f} C-F1={a['conflict_f1']:.4f} T-F1={a['temporal_f1']:.4f}")

    # Comparison
    print(f"\n{'=' * 60}")
    print(f"  TCP Ablation Comparison")
    print(f"{'=' * 60}")
    print(f"  {'Variant':<20s} {'U-F1(S)':>8s} {'C-F1':>8s} {'T-F1':>8s} {'IoU':>8s}")
    print(f"  {'─' * 50}")
    for vname, vdata in all_results.items():
        a = vdata["avg"]
        print(f"  {vname:<20s} {a['u_f1_strict']:>8.4f} {a['conflict_f1']:>8.4f} "
              f"{a['temporal_f1']:>8.4f} {a['iou']:>8.4f}")

    delta = all_results["full_with_tcp"]["avg"]["u_f1_strict"] - all_results["w/o_tcp"]["avg"]["u_f1_strict"]
    print(f"\n  Δ U-F1(S) (TCP - no TCP): {delta:+.4f}")

    with open(os.path.join(args.output_dir, "tcp_ablation_summary.json"), "w") as f:
        json.dump({v: {"avg": d["avg"], "tcp_stats": d["tcp_stats"]} for v, d in all_results.items()}, f, indent=2)
    print(f"  Saved: {args.output_dir}/")


if __name__ == "__main__":
    main()
