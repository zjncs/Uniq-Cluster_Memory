"""
temporal_eval.py
================
时间推理评测模块。

指标：
1. Temporal-Exact-F1
   - 匹配维度：patient_id + relation_type + target_value + start_time + end_time
2. Interval-IoU
   - 在同 patient_id + relation_type + target_value 条件下，
     对 GT 逐条寻找最佳预测区间 IoU 并取平均。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from src.uniq_cluster_memory.schema import CanonicalMemory


@dataclass
class TemporalMetrics:
    temporal_precision: float
    temporal_recall: float
    temporal_f1: float
    mean_interval_iou: float
    tp: int
    fp: int
    fn: int
    n_predicted: int
    n_gt: int


@dataclass
class TemporalAggMetrics:
    mean_temporal_precision: float
    mean_temporal_recall: float
    mean_temporal_f1: float
    mean_interval_iou: float
    n_samples: int


def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _norm_relation(mem: CanonicalMemory) -> str:
    rel = (mem.relation_type or "").strip().upper()
    if rel:
        return rel
    attr = (mem.attribute or "").strip().lower()
    if attr == "medication":
        return "TAKES_DRUG"
    if attr == "symptom":
        return "HAS_SYMPTOM"
    if attr == "primary_diagnosis":
        return "HAS_DIAGNOSIS"
    return "HAS_MEASUREMENT"


def _norm_target(mem: CanonicalMemory) -> str:
    return _norm_text(mem.target_value or mem.value)


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None


def _interval(mem: CanonicalMemory) -> Optional[Tuple[datetime, datetime]]:
    start = _parse_date(mem.start_time)
    end = _parse_date(mem.end_time)
    if start and end:
        if start <= end:
            return start, end
        return end, start
    if start and mem.is_ongoing:
        return start, datetime.strptime("2100-12-31", "%Y-%m-%d")
    if start:
        return start, start
    if mem.time_scope == "global":
        return datetime.strptime("1900-01-01", "%Y-%m-%d"), datetime.strptime("2100-12-31", "%Y-%m-%d")
    return None


def _exact_key(mem: CanonicalMemory) -> tuple:
    return (
        _norm_text(mem.patient_id),
        _norm_relation(mem),
        _norm_target(mem),
        mem.start_time or "",
        mem.end_time or "",
    )


def _relation_key(mem: CanonicalMemory) -> tuple:
    return (
        _norm_text(mem.patient_id),
        _norm_relation(mem),
        _norm_target(mem),
    )


def interval_iou(a: CanonicalMemory, b: CanonicalMemory) -> float:
    ia = _interval(a)
    ib = _interval(b)
    if ia is None or ib is None:
        return 0.0

    inter_start = max(ia[0], ib[0])
    inter_end = min(ia[1], ib[1])
    if inter_start > inter_end:
        return 0.0

    inter_days = (inter_end - inter_start).days + 1
    union_start = min(ia[0], ib[0])
    union_end = max(ia[1], ib[1])
    union_days = (union_end - union_start).days + 1
    if union_days <= 0:
        return 0.0
    return inter_days / union_days


def compute_temporal_metrics(
    predicted: List[CanonicalMemory],
    ground_truth: List[CanonicalMemory],
) -> TemporalMetrics:
    # Temporal-Exact-F1
    pred_keys = [_exact_key(m) for m in predicted]
    gt_keys = [_exact_key(m) for m in ground_truth]

    gt_key_set = set(gt_keys)
    matched_gt = set()
    tp = 0
    fp = 0
    for key in pred_keys:
        if key in gt_key_set and key not in matched_gt:
            tp += 1
            matched_gt.add(key)
        else:
            fp += 1
    fn = len(gt_key_set - matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Interval-IoU（对每条 GT 找最佳匹配）
    ious = []
    for gt in ground_truth:
        gt_key = _relation_key(gt)
        cands = [p for p in predicted if _relation_key(p) == gt_key]
        if not cands:
            ious.append(0.0)
            continue
        best = max(interval_iou(gt, p) for p in cands)
        ious.append(best)
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    return TemporalMetrics(
        temporal_precision=round(precision, 4),
        temporal_recall=round(recall, 4),
        temporal_f1=round(f1, 4),
        mean_interval_iou=round(mean_iou, 4),
        tp=tp,
        fp=fp,
        fn=fn,
        n_predicted=len(predicted),
        n_gt=len(ground_truth),
    )


def aggregate_temporal_metrics(metrics_list: List[TemporalMetrics]) -> TemporalAggMetrics:
    if not metrics_list:
        return TemporalAggMetrics(
            mean_temporal_precision=0.0,
            mean_temporal_recall=0.0,
            mean_temporal_f1=0.0,
            mean_interval_iou=0.0,
            n_samples=0,
        )

    n = len(metrics_list)
    return TemporalAggMetrics(
        mean_temporal_precision=round(sum(m.temporal_precision for m in metrics_list) / n, 4),
        mean_temporal_recall=round(sum(m.temporal_recall for m in metrics_list) / n, 4),
        mean_temporal_f1=round(sum(m.temporal_f1 for m in metrics_list) / n, 4),
        mean_interval_iou=round(sum(m.mean_interval_iou for m in metrics_list) / n, 4),
        n_samples=n,
    )

