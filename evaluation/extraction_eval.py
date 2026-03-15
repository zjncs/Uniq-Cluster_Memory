"""
extraction_eval.py
==================
M1 事件抽取评测。

对比对象：
- predicted: M1 输出的 ExtractedEvent 列表（保留原始 time_expr）
- gt: Med-LongMem 的 RawEvent 列表（使用标准化 time_scope）

评测策略：
1. 先对 attribute/value/unit 做同口径规范化。
2. 将 predicted 的 time_expr 通过 TimeGrounder 归一化为 time_scope。
3. 计算：
   - Event-level F1（严格）：attribute + value + unit + time_scope
   - Event-level F1（宽松）：attribute + value + unit（忽略 time_scope）
   - Field-level F1：attribute / value / unit / time_scope / speaker
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from src.uniq_cluster_memory.m1_event_extraction import ExtractedEvent, MedicalEventExtractor
from src.uniq_cluster_memory.m3_uniqueness import TimeGrounder


@dataclass
class ExtractionMetrics:
    """单样本事件抽取评测结果。"""
    event_precision: float
    event_recall: float
    event_f1: float
    relaxed_event_precision: float
    relaxed_event_recall: float
    relaxed_event_f1: float
    field_f1: dict
    tp: int
    fp: int
    fn: int
    tp_relaxed: int
    fp_relaxed: int
    fn_relaxed: int
    n_predicted: int
    n_gt: int


@dataclass
class ExtractionAggMetrics:
    """多样本抽取评测聚合结果。"""
    mean_event_f1: float
    mean_event_precision: float
    mean_event_recall: float
    mean_relaxed_event_f1: float
    mean_relaxed_event_precision: float
    mean_relaxed_event_recall: float
    mean_field_f1: dict
    n_samples: int


def _event_to_dict(event: Any) -> dict:
    if isinstance(event, ExtractedEvent):
        return event.to_dict()
    if hasattr(event, "to_dict"):
        payload = event.to_dict()
        if isinstance(payload, dict):
            return payload
    if isinstance(event, dict):
        return dict(event)
    raise TypeError(f"Unsupported event type: {type(event).__name__}")


def _normalize_event(
    event: Any,
    *,
    time_grounder: Optional[TimeGrounder],
    treat_as_prediction: bool,
) -> Optional[dict]:
    payload = _event_to_dict(event)
    attribute = str(payload.get("attribute", "")).strip()
    value = str(payload.get("value", "")).strip()
    unit = str(payload.get("unit", "")).strip()
    speaker = str(payload.get("speaker", "unknown")).strip().lower()
    confidence = float(payload.get("confidence", 1.0) or 1.0)
    snippet = str(payload.get("raw_text_snippet", value)).strip()

    normalized = MedicalEventExtractor._normalize_record(
        attribute=attribute,
        value=value,
        unit=unit,
        confidence=confidence,
        speaker=speaker,
        snippet=snippet,
    )
    if normalized is None:
        return None
    attribute, value, unit, _ = normalized

    if treat_as_prediction:
        raw_scope = str(payload.get("time_expr", payload.get("time_scope", "global"))).strip()
        time_scope = time_grounder.ground(raw_scope) if time_grounder is not None else raw_scope or "global"
    else:
        time_scope = str(payload.get("time_scope", "global")).strip() or "global"

    return {
        "attribute": attribute,
        "value": value,
        "unit": unit,
        "time_scope": time_scope,
        "speaker": speaker or "unknown",
    }


def _make_key(record: dict, fields: Sequence[str]) -> tuple:
    return tuple(str(record.get(field, "")).strip().lower() for field in fields)


def _compute_f1_from_keys(
    predicted_records: Sequence[dict],
    gt_records: Sequence[dict],
    fields: Sequence[str],
) -> tuple[float, float, float, int, int, int]:
    pred_keys = [_make_key(record, fields) for record in predicted_records]
    gt_keys = {_make_key(record, fields) for record in gt_records}

    matched_gt: set[tuple] = set()
    tp = 0
    fp = 0
    for key in pred_keys:
        if key in gt_keys and key not in matched_gt:
            tp += 1
            matched_gt.add(key)
        else:
            fp += 1

    fn = len(gt_keys - matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4), tp, fp, fn


def _field_level_f1(
    predicted_records: Sequence[dict],
    gt_records: Sequence[dict],
    fields: Iterable[str],
) -> dict:
    metrics: dict[str, float] = {}
    for field in fields:
        pred_vals = {str(record.get(field, "")).strip().lower() for record in predicted_records}
        gt_vals = {str(record.get(field, "")).strip().lower() for record in gt_records}
        tp = len(pred_vals & gt_vals)
        precision = tp / len(pred_vals) if pred_vals else 0.0
        recall = tp / len(gt_vals) if gt_vals else 0.0
        metrics[field] = round(
            2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0,
            4,
        )
    return metrics


def compute_extraction_f1(
    predicted_events: Sequence[Any],
    gt_events: Sequence[Any],
    *,
    dialogue_date: Optional[str] = None,
    missing_time_scope: str = "global",
) -> ExtractionMetrics:
    """
    计算单样本 M1 事件抽取评测指标。
    """
    time_grounder = TimeGrounder(
        dialogue_date=dialogue_date,
        missing_time_scope=missing_time_scope,
    )
    predicted_records = [
        record
        for record in (
            _normalize_event(
                event,
                time_grounder=time_grounder,
                treat_as_prediction=True,
            )
            for event in predicted_events
        )
        if record is not None
    ]
    gt_records = [
        record
        for record in (
            _normalize_event(
                event,
                time_grounder=None,
                treat_as_prediction=False,
            )
            for event in gt_events
        )
        if record is not None
    ]

    event_precision, event_recall, event_f1, tp, fp, fn = _compute_f1_from_keys(
        predicted_records,
        gt_records,
        fields=("attribute", "value", "unit", "time_scope"),
    )
    relaxed_precision, relaxed_recall, relaxed_f1, tp_relaxed, fp_relaxed, fn_relaxed = _compute_f1_from_keys(
        predicted_records,
        gt_records,
        fields=("attribute", "value", "unit"),
    )

    field_f1 = _field_level_f1(
        predicted_records,
        gt_records,
        fields=("attribute", "value", "unit", "time_scope", "speaker"),
    )

    return ExtractionMetrics(
        event_precision=event_precision,
        event_recall=event_recall,
        event_f1=event_f1,
        relaxed_event_precision=relaxed_precision,
        relaxed_event_recall=relaxed_recall,
        relaxed_event_f1=relaxed_f1,
        field_f1=field_f1,
        tp=tp,
        fp=fp,
        fn=fn,
        tp_relaxed=tp_relaxed,
        fp_relaxed=fp_relaxed,
        fn_relaxed=fn_relaxed,
        n_predicted=len(predicted_records),
        n_gt=len(gt_records),
    )


def aggregate_extraction_f1(metrics_list: Sequence[ExtractionMetrics]) -> ExtractionAggMetrics:
    """
    聚合多个样本的抽取评测结果（宏平均）。
    """
    if not metrics_list:
        return ExtractionAggMetrics(
            mean_event_f1=0.0,
            mean_event_precision=0.0,
            mean_event_recall=0.0,
            mean_relaxed_event_f1=0.0,
            mean_relaxed_event_precision=0.0,
            mean_relaxed_event_recall=0.0,
            mean_field_f1={field: 0.0 for field in ("attribute", "value", "unit", "time_scope", "speaker")},
            n_samples=0,
        )

    n = len(metrics_list)
    fields = ("attribute", "value", "unit", "time_scope", "speaker")
    mean_field_f1 = {
        field: round(sum(metric.field_f1.get(field, 0.0) for metric in metrics_list) / n, 4)
        for field in fields
    }
    return ExtractionAggMetrics(
        mean_event_f1=round(sum(metric.event_f1 for metric in metrics_list) / n, 4),
        mean_event_precision=round(sum(metric.event_precision for metric in metrics_list) / n, 4),
        mean_event_recall=round(sum(metric.event_recall for metric in metrics_list) / n, 4),
        mean_relaxed_event_f1=round(sum(metric.relaxed_event_f1 for metric in metrics_list) / n, 4),
        mean_relaxed_event_precision=round(sum(metric.relaxed_event_precision for metric in metrics_list) / n, 4),
        mean_relaxed_event_recall=round(sum(metric.relaxed_event_recall for metric in metrics_list) / n, 4),
        mean_field_f1=mean_field_f1,
        n_samples=n,
    )


def explain_extraction_result(metrics: ExtractionMetrics) -> str:
    """
    生成可读的抽取评测摘要。
    """
    lines = [
        "Extraction Evaluation Summary",
        f"  Event F1 (strict)  : {metrics.event_f1:.4f} "
        f"(P={metrics.event_precision:.4f}, R={metrics.event_recall:.4f})",
        f"  Event F1 (relaxed) : {metrics.relaxed_event_f1:.4f} "
        f"(P={metrics.relaxed_event_precision:.4f}, R={metrics.relaxed_event_recall:.4f})",
        f"  Counts             : TP={metrics.tp}, FP={metrics.fp}, FN={metrics.fn}",
        "  Field F1:",
    ]
    for field, score in metrics.field_f1.items():
        lines.append(f"    - {field}: {score:.4f}")
    return "\n".join(lines)
