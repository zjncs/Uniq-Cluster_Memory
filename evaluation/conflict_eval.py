"""
conflict_eval.py
================
冲突检测评测模块（Conflict Detection F1）。

这是论文中"第二优先指标"，直接衡量系统识别和记录矛盾信息的能力。

核心概念：
    一个"冲突事件"（Conflict Event）是指：
    在相同的 (patient_id, attribute, time_scope) 下，
    系统检测到一个新值与历史记录中的旧值不一致。

    系统通过将 CanonicalMemory.conflict_flag = True 来标记冲突，
    并在 CanonicalMemory.conflict_history 中记录冲突的详细信息。

评测逻辑：
    我们将"冲突检测"视为一个二分类问题：
    - 正例（Positive）：GT 中 conflict_flag = True 的 CanonicalMemory 记录。
    - 负例（Negative）：GT 中 conflict_flag = False 的 CanonicalMemory 记录。

    匹配标准（TP 的判定）：
    系统输出的 CanonicalMemory 中，conflict_flag = True 的记录，
    与 GT 中 conflict_flag = True 的记录，在 unique_key 上匹配。
    即：(patient_id, attribute, time_scope) 三元组相同。

    注意：我们不要求 conflict_history 中的 old_value 完全匹配，
    因为系统可能以不同的方式表达旧值。
    但在更严格的评测模式下（strict=True），我们要求 old_value 也匹配。

评测指标：
    TP = 系统正确检测到的冲突数（predicted conflict_flag=True & GT conflict_flag=True & key匹配）
    FP = 系统误报的冲突数（predicted conflict_flag=True & GT conflict_flag=False 或 key不匹配）
    FN = 系统漏报的冲突数（GT conflict_flag=True & 系统未检测到）
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * P * R / (P + R)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.uniq_cluster_memory.schema import CanonicalMemory


# ─── 评测结果数据结构 ─────────────────────────────────────────────────────────

@dataclass
class ConflictMetrics:
    """单个样本的冲突检测评测结果。"""
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    n_predicted_conflicts: int  # 系统报告的冲突总数
    n_gt_conflicts: int         # GT 中的冲突总数


@dataclass
class ConflictAggMetrics:
    """多个样本的聚合冲突检测评测结果（宏平均）。"""
    mean_f1: float
    mean_precision: float
    mean_recall: float
    n_samples: int
    total_gt_conflicts: int     # 所有样本中 GT 冲突的总数


# ─── 核心匹配逻辑 ────────────────────────────────────────────────────────────

def _make_conflict_key(mem: CanonicalMemory) -> tuple:
    """
    生成冲突匹配键：(patient_id, attribute, time_scope)。

    我们不将 value 纳入匹配键，因为冲突的核心是"同一属性在同一时间范围内
    出现了矛盾"，而不是具体的值。
    """
    return (
        mem.patient_id.strip().lower(),
        mem.attribute.strip().lower(),
        mem.time_scope.strip().lower(),
    )


def _make_strict_conflict_key(mem: CanonicalMemory) -> Optional[tuple]:
    """
    严格模式下的冲突匹配键：(patient_id, attribute, time_scope, old_value, new_value)。

    要求 conflict_history 中至少有一条记录，且 old_value 可以被提取。
    """
    if not mem.conflict_history:
        return None
    # 取最近一条冲突记录的 old_value
    latest_conflict = mem.conflict_history[-1]
    return (
        mem.patient_id.strip().lower(),
        mem.attribute.strip().lower(),
        mem.time_scope.strip().lower(),
        latest_conflict.old_value.strip().lower(),
        latest_conflict.new_value.strip().lower(),
    )


# ─── 主评测函数 ──────────────────────────────────────────────────────────────

def compute_conflict_f1(
    predicted: List[CanonicalMemory],
    ground_truth: List[CanonicalMemory],
    strict: bool = False,
) -> ConflictMetrics:
    """
    计算单个样本的 Conflict Detection F1。

    Args:
        predicted:    系统输出的 CanonicalMemory 列表（M3 模块的输出）。
        ground_truth: GT 的 CanonicalMemory 列表。
        strict:       是否使用严格匹配模式（要求 old_value 也匹配）。

    Returns:
        ConflictMetrics 对象。
    """
    # 提取 GT 中的冲突集合
    if strict:
        gt_conflict_keys: set = {
            _make_strict_conflict_key(m)
            for m in ground_truth
            if m.conflict_flag and _make_strict_conflict_key(m) is not None
        }
    else:
        gt_conflict_keys: set = {
            _make_conflict_key(m)
            for m in ground_truth
            if m.conflict_flag
        }

    # 提取 predicted 中的冲突集合
    if strict:
        pred_conflict_keys: set = {
            _make_strict_conflict_key(m)
            for m in predicted
            if m.conflict_flag and _make_strict_conflict_key(m) is not None
        }
    else:
        pred_conflict_keys: set = {
            _make_conflict_key(m)
            for m in predicted
            if m.conflict_flag
        }

    # 计算 TP / FP / FN
    tp = len(pred_conflict_keys & gt_conflict_keys)
    fp = len(pred_conflict_keys - gt_conflict_keys)
    fn = len(gt_conflict_keys - pred_conflict_keys)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return ConflictMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        tp=tp,
        fp=fp,
        fn=fn,
        n_predicted_conflicts=len(pred_conflict_keys),
        n_gt_conflicts=len(gt_conflict_keys),
    )


def aggregate_conflict_f1(
    metrics_list: List[ConflictMetrics],
) -> ConflictAggMetrics:
    """
    聚合多个样本的 Conflict Detection F1 结果（宏平均）。

    Args:
        metrics_list: 多个样本的 ConflictMetrics 列表。

    Returns:
        ConflictAggMetrics 对象。
    """
    n = len(metrics_list)
    if n == 0:
        return ConflictAggMetrics(
            mean_f1=0.0, mean_precision=0.0, mean_recall=0.0,
            n_samples=0, total_gt_conflicts=0,
        )
    return ConflictAggMetrics(
        mean_f1=round(sum(m.f1 for m in metrics_list) / n, 4),
        mean_precision=round(sum(m.precision for m in metrics_list) / n, 4),
        mean_recall=round(sum(m.recall for m in metrics_list) / n, 4),
        n_samples=n,
        total_gt_conflicts=sum(m.n_gt_conflicts for m in metrics_list),
    )


# ─── 诊断工具 ────────────────────────────────────────────────────────────────

def explain_conflict_result(
    predicted: List[CanonicalMemory],
    ground_truth: List[CanonicalMemory],
    metrics: ConflictMetrics,
) -> str:
    """
    生成可读的冲突检测诊断报告。

    Args:
        predicted:    系统输出的 CanonicalMemory 列表。
        ground_truth: GT 的 CanonicalMemory 列表。
        metrics:      compute_conflict_f1 的输出。

    Returns:
        格式化的诊断报告字符串。
    """
    gt_conflict_keys = {
        _make_conflict_key(m) for m in ground_truth if m.conflict_flag
    }
    pred_conflict_keys = {
        _make_conflict_key(m) for m in predicted if m.conflict_flag
    }

    missed = gt_conflict_keys - pred_conflict_keys   # FN
    false_alarms = pred_conflict_keys - gt_conflict_keys  # FP

    lines = [
        "─" * 60,
        "Conflict Detection F1 Diagnostic Report",
        "─" * 60,
        f"  Precision : {metrics.precision:.4f}",
        f"  Recall    : {metrics.recall:.4f}",
        f"  F1        : {metrics.f1:.4f}",
        f"  TP={metrics.tp}, FP={metrics.fp}, FN={metrics.fn}",
        f"  Predicted conflicts: {metrics.n_predicted_conflicts}",
        f"  GT conflicts       : {metrics.n_gt_conflicts}",
        "",
    ]

    if missed:
        lines.append("  ❌ Missed conflicts (FN):")
        for k in sorted(missed):
            lines.append(f"     patient={k[0]}, attr={k[1]}, scope={k[2]}")
    if false_alarms:
        lines.append("  ⚠️  False alarm conflicts (FP):")
        for k in sorted(false_alarms):
            lines.append(f"     patient={k[0]}, attr={k[1]}, scope={k[2]}")
    if not missed and not false_alarms:
        lines.append("  ✅ Perfect conflict detection!")

    lines.append("─" * 60)
    return "\n".join(lines)
