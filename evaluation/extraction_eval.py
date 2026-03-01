"""
extraction_eval.py
==================
事件抽取评测模块（M1 模块评测）。

评测 M1 事件抽取模块从对话文本中抽取结构化事件的质量。

评测维度：
    - Field-level F1: 对每个字段（entity, attribute, value, unit, time）分别计算 F1。
    - Event-level F1: 将整个事件元组作为一个整体进行匹配，计算 F1。

注意：
    此模块依赖于 Med-LongMem 数据集的 GT 格式（event_table_gt 字段）。
    在 Step 3（Med-LongMem 生成）完成后，将根据最终 GT 格式完整实现此模块。
"""

from dataclasses import dataclass


@dataclass
class ExtractionMetrics:
    """事件抽取评测结果。"""
    event_precision: float
    event_recall: float
    event_f1: float
    field_f1: dict  # 每个字段的 F1 分数


def compute_extraction_f1(
    predicted_events: list[dict],
    gt_events: list[dict],
    key_fields: list[str] = None,
) -> ExtractionMetrics:
    """
    计算事件抽取 F1 指标。

    Args:
        predicted_events: 系统抽取的事件列表。
        gt_events: GT 事件列表。
        key_fields: 用于匹配事件的字段列表。默认为所有字段。

    Returns:
        ExtractionMetrics 对象。
    """
    if key_fields is None:
        key_fields = ["entity", "attribute", "value", "unit", "time"]

    def event_to_key(event: dict) -> tuple:
        return tuple(str(event.get(f, "")).lower().strip() for f in key_fields)

    predicted_keys = {event_to_key(e) for e in predicted_events}
    gt_keys = {event_to_key(e) for e in gt_events}

    tp = len(predicted_keys & gt_keys)
    fp = len(predicted_keys - gt_keys)
    fn = len(gt_keys - predicted_keys)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # 字段级别 F1（简化版：逐字段匹配）
    field_f1 = {}
    for field in key_fields:
        pred_vals = {str(e.get(field, "")).lower().strip() for e in predicted_events}
        gt_vals = {str(e.get(field, "")).lower().strip() for e in gt_events}
        field_tp = len(pred_vals & gt_vals)
        field_p = field_tp / len(pred_vals) if pred_vals else 0.0
        field_r = field_tp / len(gt_vals) if gt_vals else 0.0
        field_f1[field] = round(
            2 * field_p * field_r / (field_p + field_r) if (field_p + field_r) > 0 else 0.0,
            4,
        )

    return ExtractionMetrics(
        event_precision=round(precision, 4),
        event_recall=round(recall, 4),
        event_f1=round(f1, 4),
        field_f1=field_f1,
    )
