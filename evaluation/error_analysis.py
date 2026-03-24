"""
evaluation/error_analysis.py
==============================
Error Analysis 框架：系统性分析 UCM pipeline 的错误模式。

分析维度：
    1. 按难度分层（Easy/Medium/Hard）的性能差异
    2. 按属性分层（measurement/medication/symptom/diagnosis）
    3. M1 提取错误的下游传播
    4. 时间 grounding 失败模式分类
    5. 冲突检测的 FP/FN 分析
    6. 信息团粒度对冲突检测的影响

用法：
    PYTHONPATH=. python evaluation/error_analysis.py \\
        --data_path data/raw/med_longmem \\
        --output_path results/error_analysis/analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.uniq_cluster_memory.schema import CanonicalMemory
from evaluation.uniqueness_eval import compute_unique_f1
from evaluation.conflict_eval import compute_conflict_f1
from evaluation.temporal_eval import compute_temporal_metrics


def analyze_errors(
    samples: list,
    pipeline_fn,
) -> Dict:
    """
    运行 pipeline 并分析错误模式。

    Args:
        samples: UnifiedSample 列表
        pipeline_fn: 接受 (dialogue, dialogue_id, dialogue_date) 返回 List[CanonicalMemory] 的函数

    Returns:
        完整的 error analysis 报告。
    """
    # 初始化收集器
    by_difficulty = defaultdict(list)
    by_attribute = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    time_grounding_errors = []
    conflict_fp = []  # False positive conflicts
    conflict_fn = []  # False negative conflicts
    extraction_propagation = []
    all_results = []

    for i, sample in enumerate(samples, 1):
        dialogue = [
            {
                "turn_id": j,
                "speaker": "patient" if turn.role == "user" else "doctor",
                "text": turn.content,
            }
            for j, turn in enumerate(sample.dialog_history)
        ]

        predicted = pipeline_fn(dialogue, sample.sample_id, sample.question_date)
        gt = sample.metadata.get("canonical_gt", [])
        difficulty = sample.metadata.get("difficulty", "unknown")

        u = compute_unique_f1(predicted, gt)
        c = compute_conflict_f1(predicted, gt)
        t = compute_temporal_metrics(predicted, gt)

        result = {
            "sample_id": sample.sample_id,
            "difficulty": difficulty,
            "n_gt": len(gt),
            "n_predicted": len(predicted),
            "unique_f1_strict": round(u.f1, 4),
            "conflict_f1": round(c.f1, 4),
            "temporal_f1": round(t.temporal_f1, 4),
        }
        all_results.append(result)
        by_difficulty[difficulty].append(result)

        # 按属性统计
        gt_keys = {(m.attribute, m.time_scope, m.value.strip().lower()) for m in gt}
        pred_keys = {(m.attribute, m.time_scope, m.value.strip().lower()) for m in predicted}
        for key in pred_keys & gt_keys:
            by_attribute[key[0]]["tp"] += 1
        for key in pred_keys - gt_keys:
            by_attribute[key[0]]["fp"] += 1
        for key in gt_keys - pred_keys:
            by_attribute[key[0]]["fn"] += 1

        # 时间 grounding 错误分析
        gt_scopes = {(m.attribute, m.value.strip().lower()): m.time_scope for m in gt}
        for mem in predicted:
            key = (mem.attribute, mem.value.strip().lower())
            if key in gt_scopes and gt_scopes[key] != mem.time_scope:
                time_grounding_errors.append({
                    "sample_id": sample.sample_id,
                    "attribute": mem.attribute,
                    "value": mem.value,
                    "predicted_scope": mem.time_scope,
                    "gt_scope": gt_scopes[key],
                })

        # 冲突 FP/FN 分析
        gt_conflict_keys = {
            (m.attribute, m.time_scope)
            for m in gt if m.conflict_flag
        }
        pred_conflict_keys = {
            (m.attribute, m.time_scope)
            for m in predicted if m.conflict_flag
        }
        for key in pred_conflict_keys - gt_conflict_keys:
            conflict_fp.append({
                "sample_id": sample.sample_id,
                "attribute": key[0],
                "time_scope": key[1],
            })
        for key in gt_conflict_keys - pred_conflict_keys:
            conflict_fn.append({
                "sample_id": sample.sample_id,
                "attribute": key[0],
                "time_scope": key[1],
            })

    # 汇总统计
    report = {
        "n_samples": len(samples),
        "overall": _aggregate_results(all_results),
    }

    # 按难度分层
    report["by_difficulty"] = {}
    for diff, results in sorted(by_difficulty.items()):
        report["by_difficulty"][diff] = _aggregate_results(results)

    # 按属性分层
    report["by_attribute"] = {}
    for attr, counts in sorted(by_attribute.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        report["by_attribute"][attr] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    # 时间 grounding 错误
    report["time_grounding_errors"] = {
        "total": len(time_grounding_errors),
        "error_rate": round(
            len(time_grounding_errors) / max(sum(r["n_predicted"] for r in all_results), 1), 4
        ),
        "by_type": _classify_time_errors(time_grounding_errors),
        "examples": time_grounding_errors[:10],
    }

    # 冲突检测错误
    report["conflict_errors"] = {
        "false_positives": len(conflict_fp),
        "false_negatives": len(conflict_fn),
        "fp_examples": conflict_fp[:5],
        "fn_examples": conflict_fn[:5],
    }

    return report


def _aggregate_results(results: list) -> dict:
    """汇总一组结果的平均指标。"""
    n = len(results)
    if n == 0:
        return {"n_samples": 0}
    return {
        "n_samples": n,
        "mean_unique_f1_strict": round(sum(r["unique_f1_strict"] for r in results) / n, 4),
        "mean_conflict_f1": round(sum(r["conflict_f1"] for r in results) / n, 4),
        "mean_temporal_f1": round(sum(r["temporal_f1"] for r in results) / n, 4),
        "std_unique_f1_strict": round(
            (sum((r["unique_f1_strict"] - sum(r2["unique_f1_strict"] for r2 in results) / n) ** 2
                 for r in results) / n) ** 0.5, 4
        ),
    }


def _classify_time_errors(errors: list) -> dict:
    """分类时间 grounding 错误的类型。"""
    categories = defaultdict(int)
    for err in errors:
        pred = err["predicted_scope"]
        gt = err["gt_scope"]
        if pred == "global" and gt != "global":
            categories["missed_grounding"] += 1
        elif pred != "global" and gt == "global":
            categories["over_grounding"] += 1
        elif pred != "global" and gt != "global":
            categories["wrong_date"] += 1
        else:
            categories["other"] += 1
    return dict(categories)


def main():
    parser = argparse.ArgumentParser(description="Error Analysis")
    parser.add_argument("--data_path", default="data/raw/med_longmem")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_path", default="results/error_analysis/analysis.json")
    args = parser.parse_args()

    from benchmarks.med_longmem_task import MedLongMemTask
    from src.uniq_cluster_memory.pipeline import UniqueClusterMemoryPipeline
    from src.uniq_cluster_memory.defaults import recommended_pipeline_options

    task = MedLongMemTask(data_path=args.data_path, max_samples=args.max_samples)
    samples = task.get_samples()
    print(f"Loaded {len(samples)} samples for error analysis\n")

    defaults = recommended_pipeline_options("med_longmem")
    pipeline = UniqueClusterMemoryPipeline(**defaults)

    def pipeline_fn(dialogue, dialogue_id, dialogue_date):
        return pipeline.build_memory(dialogue, dialogue_id, dialogue_date)

    report = analyze_errors(samples, pipeline_fn)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Error Analysis Report")
    print(f"{'='*60}")
    print(f"  Overall U-F1(S): {report['overall'].get('mean_unique_f1_strict', 0):.4f}")
    print(f"\n  By Difficulty:")
    for diff, stats in report.get("by_difficulty", {}).items():
        print(f"    {diff:8s}: U-F1(S)={stats.get('mean_unique_f1_strict', 0):.4f} (n={stats['n_samples']})")
    print(f"\n  By Attribute:")
    for attr, stats in report.get("by_attribute", {}).items():
        print(f"    {attr:20s}: P={stats['precision']:.3f} R={stats['recall']:.3f} F1={stats['f1']:.3f}")
    print(f"\n  Time Grounding Errors: {report['time_grounding_errors']['total']}")
    print(f"    Error rate: {report['time_grounding_errors']['error_rate']:.4f}")
    errs = report['time_grounding_errors'].get('by_type', {})
    for etype, count in errs.items():
        print(f"    {etype}: {count}")
    print(f"\n  Conflict Detection Errors:")
    print(f"    False Positives: {report['conflict_errors']['false_positives']}")
    print(f"    False Negatives: {report['conflict_errors']['false_negatives']}")
    print(f"\n  Saved: {args.output_path}")


if __name__ == "__main__":
    main()
