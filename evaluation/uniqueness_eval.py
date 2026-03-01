"""
uniqueness_eval.py
==================
唯一性评测模块（Unique-F1）。

这是论文中"第一优先指标"，直接衡量记忆库的规范化和去重效果。

核心概念：
    给定一段对话历史，系统应该输出一个 CanonicalMemory 列表（predicted）。
    GT 同样是一个 CanonicalMemory 列表（ground truth）。
    我们通过比较两者来计算 Unique-F1。

匹配策略（两级匹配）：
    Level 1 — Strict Match（严格匹配）：
        匹配键为 (patient_id, attribute, time_scope, value)。
        用于计算 Unique-F1 的主要指标。

    Level 2 — Scope-Relaxed Match（宽松匹配）：
        匹配键为 (patient_id, attribute, value)，忽略 time_scope。
        用于诊断和辅助分析，反映"内容正确但时间标注缺失"的情况。
        这是一个合理的宽松设定，因为：
        (a) LLM 在没有显式日期的对话中倾向于使用 "global" 作为 scope；
        (b) 对于 "latest" 策略的属性（如 medication），scope 本身就是 "global"；
        (c) 对于 "unique" 策略的属性，scope 是区分不同时间点的关键，
            但如果对话中日期信息不够显式，LLM 可能无法准确提取。

    Level 3 — Attribute-Only Match（仅属性匹配）：
        匹配键为 (patient_id, attribute)，忽略 time_scope 和 value。
        用于诊断"属性识别率"，即系统是否至少识别出了正确的属性类型。

评测指标：
    Unique-F1（严格）：
        TP = predicted 中与 GT 精确匹配的记录数（attribute + scope + value 均匹配）
        FP = predicted 中无法与 GT 匹配的记录数（包括冗余记录）
        FN = GT 中无法被 predicted 覆盖的记录数

    Unique-F1（宽松，scope-relaxed）：
        同上，但匹配时忽略 time_scope。

    Redundancy Reduction（冗余消除率）：
        Redundancy = 1 - (len(unique predicted keys) / len(all predicted records))
        越低越好（0 表示完全无冗余）。

    Coverage（覆盖率）：
        Coverage = len(GT attributes covered by predicted) / len(GT attributes)
        衡量系统是否"记住了"所有重要信息（宽松版，只要属性匹配即算覆盖）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple

from src.uniq_cluster_memory.schema import CanonicalMemory


# ─── 评测结果数据结构 ─────────────────────────────────────────────────────────

@dataclass
class UniquenessMetrics:
    """单个样本的唯一性评测结果。"""
    # 严格匹配指标（attribute + scope + value）
    precision: float
    recall: float
    f1: float
    # 宽松匹配指标（attribute + value，忽略 scope）
    relaxed_precision: float
    relaxed_recall: float
    relaxed_f1: float
    # 属性覆盖率（attribute only）
    attribute_coverage: float
    # 辅助指标
    redundancy: float       # 冗余率（越低越好）
    coverage: float         # 宽松覆盖率（越高越好）
    # 原始计数（用于调试）
    tp: int
    fp: int
    fn: int
    tp_relaxed: int
    fp_relaxed: int
    fn_relaxed: int
    n_predicted: int
    n_unique_predicted: int
    n_gt: int


@dataclass
class UniquenessAggMetrics:
    """多个样本的聚合唯一性评测结果（宏平均）。"""
    mean_f1: float
    mean_precision: float
    mean_recall: float
    mean_relaxed_f1: float
    mean_relaxed_precision: float
    mean_relaxed_recall: float
    mean_attribute_coverage: float
    mean_redundancy: float
    mean_coverage: float
    n_samples: int


# ─── 核心匹配逻辑 ────────────────────────────────────────────────────────────

def _normalize_value(value: str) -> str:
    """
    对 value 进行规范化，提高匹配容忍度。
    - 去除首尾空格
    - 转小写
    - 去除多余空格
    """
    return " ".join(value.strip().lower().split())


def _normalize_attribute(attr: str) -> str:
    """
    对 attribute 进行规范化。
    - 去除首尾空格
    - 转小写
    - 将常见别名统一（如 systolic_blood_pressure -> blood_pressure_sys）
    """
    attr = attr.strip().lower().replace(" ", "_")
    # 常见别名映射
    alias_map = {
        "systolic_blood_pressure": "blood_pressure_sys",
        "diastolic_blood_pressure": "blood_pressure_dia",
        "blood_pressure_systolic": "blood_pressure_sys",
        "blood_pressure_diastolic": "blood_pressure_dia",
        "bp_systolic": "blood_pressure_sys",
        "bp_diastolic": "blood_pressure_dia",
        "blood_pressure_dias": "blood_pressure_dia",
        "blood_pressure_sys": "blood_pressure_sys",
        "blood_pressure_dia": "blood_pressure_dia",
        "hb": "hemoglobin",
        "hgb": "hemoglobin",
        "temp": "body_temperature",
        "temperature": "body_temperature",
        "hr": "heart_rate",
        "pulse": "heart_rate",
        "bg": "blood_glucose",
        "glucose": "blood_glucose",
        "blood_sugar": "blood_glucose",
        "diagnosis": "primary_diagnosis",
        "chief_complaint": "symptom",
    }
    return alias_map.get(attr, attr)


def _make_strict_key(mem: CanonicalMemory) -> tuple:
    """严格匹配键：(patient_id, attribute, time_scope, value)"""
    return (
        mem.patient_id.strip().lower(),
        _normalize_attribute(mem.attribute),
        mem.time_scope.strip().lower(),
        _normalize_value(mem.value),
    )


def _make_relaxed_key(mem: CanonicalMemory) -> tuple:
    """宽松匹配键：(patient_id, attribute, value)，忽略 time_scope"""
    return (
        mem.patient_id.strip().lower(),
        _normalize_attribute(mem.attribute),
        _normalize_value(mem.value),
    )


def _make_attribute_key(mem: CanonicalMemory) -> tuple:
    """属性匹配键：(patient_id, attribute)"""
    return (
        mem.patient_id.strip().lower(),
        _normalize_attribute(mem.attribute),
    )


def _make_unique_key(mem: CanonicalMemory) -> tuple:
    """
    冗余检测键（strategy-aware）。

    - latest:  以 (patient, attribute) 判重（全局只应保留最新值）
    - unique:  以 (patient, attribute, time_scope) 判重
    - append:  以 (patient, attribute, time_scope, value) 判重

    这样可避免把“同属性不同日期”的合法记录错误算作冗余。
    """
    patient = mem.patient_id.strip().lower()
    attr = _normalize_attribute(mem.attribute)
    scope = mem.time_scope.strip().lower()
    value = _normalize_value(mem.value)
    policy = (mem.update_policy or "unique").strip().lower()

    if policy == "latest":
        return patient, attr
    if policy == "append":
        return patient, attr, scope, value
    return patient, attr, scope


# ─── 主评测函数 ──────────────────────────────────────────────────────────────

def _compute_f1_from_sets(
    predicted: List[CanonicalMemory],
    ground_truth: List[CanonicalMemory],
    key_fn,
) -> Tuple[float, float, float, int, int, int]:
    """
    通用 F1 计算函数，根据给定的 key_fn 进行匹配。

    Returns:
        (precision, recall, f1, tp, fp, fn)
    """
    gt_keys: Set[tuple] = {key_fn(m) for m in ground_truth}
    matched_gt: Set[tuple] = set()

    tp = 0
    fp = 0

    for pred_mem in predicted:
        key = key_fn(pred_mem)
        # 所有策略统一按 key 级别匹配：
        # - 同 key 首次命中计 TP
        # - 重复预测或错误预测计 FP
        if key in gt_keys and key not in matched_gt:
            tp += 1
            matched_gt.add(key)
        else:
            fp += 1

    fn = len(gt_keys - matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return round(precision, 4), round(recall, 4), round(f1, 4), tp, fp, fn


def compute_unique_f1(
    predicted: List[CanonicalMemory],
    ground_truth: List[CanonicalMemory],
) -> UniquenessMetrics:
    """
    计算单个样本的 Unique-F1（严格 + 宽松）及辅助指标。

    Args:
        predicted:     系统输出的 CanonicalMemory 列表。
        ground_truth:  GT 的 CanonicalMemory 列表。

    Returns:
        UniquenessMetrics 对象。
    """
    # 严格匹配（attribute + scope + value）
    p_s, r_s, f1_s, tp_s, fp_s, fn_s = _compute_f1_from_sets(
        predicted, ground_truth, _make_strict_key
    )

    # 宽松匹配（attribute + value，忽略 scope）
    p_r, r_r, f1_r, tp_r, fp_r, fn_r = _compute_f1_from_sets(
        predicted, ground_truth, _make_relaxed_key
    )

    # 属性覆盖率（attribute only）
    gt_attr_keys = {_make_attribute_key(m) for m in ground_truth}
    pred_attr_keys = {_make_attribute_key(m) for m in predicted}
    attr_coverage = (
        len(gt_attr_keys & pred_attr_keys) / len(gt_attr_keys)
        if gt_attr_keys
        else 1.0
    )

    # 冗余率（基于宽松 unique_key）
    n_predicted = len(predicted)
    n_unique_predicted = len({_make_unique_key(m) for m in predicted})
    redundancy = (
        1.0 - n_unique_predicted / n_predicted
        if n_predicted > 0
        else 0.0
    )

    # 宽松覆盖率（基于属性匹配）
    coverage = round(attr_coverage, 4)

    return UniquenessMetrics(
        precision=p_s,
        recall=r_s,
        f1=f1_s,
        relaxed_precision=p_r,
        relaxed_recall=r_r,
        relaxed_f1=f1_r,
        attribute_coverage=round(attr_coverage, 4),
        redundancy=round(redundancy, 4),
        coverage=coverage,
        tp=tp_s,
        fp=fp_s,
        fn=fn_s,
        tp_relaxed=tp_r,
        fp_relaxed=fp_r,
        fn_relaxed=fn_r,
        n_predicted=n_predicted,
        n_unique_predicted=n_unique_predicted,
        n_gt=len(ground_truth),
    )


def aggregate_unique_f1(
    metrics_list: List[UniquenessMetrics],
) -> UniquenessAggMetrics:
    """聚合多个样本的 Unique-F1 结果（宏平均）。"""
    n = len(metrics_list)
    if n == 0:
        return UniquenessAggMetrics(
            mean_f1=0.0, mean_precision=0.0, mean_recall=0.0,
            mean_relaxed_f1=0.0, mean_relaxed_precision=0.0, mean_relaxed_recall=0.0,
            mean_attribute_coverage=0.0,
            mean_redundancy=0.0, mean_coverage=0.0, n_samples=0,
        )
    return UniquenessAggMetrics(
        mean_f1=round(sum(m.f1 for m in metrics_list) / n, 4),
        mean_precision=round(sum(m.precision for m in metrics_list) / n, 4),
        mean_recall=round(sum(m.recall for m in metrics_list) / n, 4),
        mean_relaxed_f1=round(sum(m.relaxed_f1 for m in metrics_list) / n, 4),
        mean_relaxed_precision=round(sum(m.relaxed_precision for m in metrics_list) / n, 4),
        mean_relaxed_recall=round(sum(m.relaxed_recall for m in metrics_list) / n, 4),
        mean_attribute_coverage=round(sum(m.attribute_coverage for m in metrics_list) / n, 4),
        mean_redundancy=round(sum(m.redundancy for m in metrics_list) / n, 4),
        mean_coverage=round(sum(m.coverage for m in metrics_list) / n, 4),
        n_samples=n,
    )


# ─── 诊断工具 ────────────────────────────────────────────────────────────────

def explain_uniqueness_result(
    predicted: List[CanonicalMemory],
    ground_truth: List[CanonicalMemory],
    metrics: UniquenessMetrics,
) -> str:
    """生成可读的评测结果诊断报告。"""
    gt_strict_keys = {_make_strict_key(m) for m in ground_truth}
    gt_relaxed_keys = {_make_relaxed_key(m) for m in ground_truth}
    pred_strict_keys = {_make_strict_key(m) for m in predicted}
    pred_relaxed_keys = {_make_relaxed_key(m) for m in predicted}

    strict_missing = gt_strict_keys - pred_strict_keys
    relaxed_missing = gt_relaxed_keys - pred_relaxed_keys

    lines = [
        "─" * 65,
        "Unique-F1 Diagnostic Report",
        "─" * 65,
        f"  [Strict]  P={metrics.precision:.4f} R={metrics.recall:.4f} F1={metrics.f1:.4f}",
        f"  [Relaxed] P={metrics.relaxed_precision:.4f} R={metrics.relaxed_recall:.4f} F1={metrics.relaxed_f1:.4f}",
        f"  [Attr Coverage] {metrics.attribute_coverage:.4f}",
        f"  Redundancy: {metrics.redundancy:.4f}  |  Coverage: {metrics.coverage:.4f}",
        f"  TP={metrics.tp} FP={metrics.fp} FN={metrics.fn} (strict)",
        f"  TP={metrics.tp_relaxed} FP={metrics.fp_relaxed} FN={metrics.fn_relaxed} (relaxed)",
        f"  Predicted: {metrics.n_predicted} records ({metrics.n_unique_predicted} unique keys)",
        f"  GT       : {metrics.n_gt} records",
        "",
    ]

    if strict_missing and not relaxed_missing:
        lines.append("  ⚠️  Scope mismatch (content correct, scope wrong):")
        for k in sorted(strict_missing):
            lines.append(f"     {k}")
    elif relaxed_missing:
        lines.append("  ❌ Missing from predicted (even relaxed):")
        for k in sorted(relaxed_missing):
            lines.append(f"     {k}")
    else:
        lines.append("  ✅ Perfect match (strict)!")

    lines.append("─" * 65)
    return "\n".join(lines)
