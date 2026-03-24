"""
m3_uniqueness/formal_constraints.py
====================================
医疗领域形式化时序约束（Medical Domain Formal Constraints）。

使用形式逻辑规则对候选值进行一致性验证，自动惩罚违反临床逻辑的候选值的置信度。
这是纯规则逻辑，不调用 LLM。

核心思想：
    形式约束作为先验知识注入冲突解决过程，与 LLM 的语义判断互补。
    形式约束处理结构性/逻辑性矛盾（如时间顺序、药物状态），
    LLM 处理语义矛盾（如 "血压正常" vs "高血压"）。

参考：ALICE (ASE 2024): formal logic + LLM hybrid for contradiction detection。

约束类型：
    - 硬约束（hard）：违反 → confidence = 0.0（逻辑上不可能的状态）
    - 软约束（soft）：违反 → confidence *= penalty（临床上不典型但可能的状态）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from src.uniq_cluster_memory.schema import CandidateValue, CanonicalMemory, scope_to_interval


@dataclass
class ConstraintResult:
    """约束检查结果。"""
    violated: bool
    constraint_name: str
    severity: str = "soft"     # "hard" or "soft"
    penalty: float = 0.5       # soft 约束的置信度乘数
    reason: str = ""


class MedicalTemporalConstraint:
    """医疗时序约束基类。"""
    name: str = "base_constraint"

    def check(
        self,
        candidate: CandidateValue,
        existing_memories: List[CanonicalMemory],
        attribute: str,
    ) -> Optional[ConstraintResult]:
        """检查候选值是否违反约束。返回 None 表示通过。"""
        return None


class MedicationStartStopConstraint(MedicalTemporalConstraint):
    """
    硬约束：同一药物不能在同一日期同时处于"开始"和"停止"状态。

    例：如果记忆中已有 "Metformin started on 2024-01-15"，
    则候选 "Metformin stopped on 2024-01-15" 应被惩罚。
    """
    name = "medication_start_stop"

    START_PATTERNS = re.compile(
        r"\b(start(?:ed)?|prescrib(?:ed)?|begin|initiat(?:ed)?|开始|开|启用)\b", re.IGNORECASE
    )
    STOP_PATTERNS = re.compile(
        r"\b(stop(?:ped)?|discontinu(?:ed)?|ceas(?:ed)?|terminat(?:ed)?|停|停用|停药)\b", re.IGNORECASE
    )

    def check(
        self,
        candidate: CandidateValue,
        existing_memories: List[CanonicalMemory],
        attribute: str,
    ) -> Optional[ConstraintResult]:
        if attribute != "medication":
            return None

        cand_text = candidate.value.lower()
        cand_is_start = bool(self.START_PATTERNS.search(cand_text))
        cand_is_stop = bool(self.STOP_PATTERNS.search(cand_text))

        if not cand_is_start and not cand_is_stop:
            return None

        cand_date = candidate.t_event
        if not cand_date:
            return None

        for mem in existing_memories:
            if mem.attribute != "medication":
                continue
            mem_date = mem.t_event or mem.start_time
            if mem_date != cand_date:
                continue
            mem_text = mem.value.lower()
            mem_is_start = bool(self.START_PATTERNS.search(mem_text))
            mem_is_stop = bool(self.STOP_PATTERNS.search(mem_text))

            if (cand_is_start and mem_is_stop) or (cand_is_stop and mem_is_start):
                return ConstraintResult(
                    violated=True,
                    constraint_name=self.name,
                    severity="hard",
                    penalty=0.0,
                    reason=f"Medication cannot be both started and stopped on {cand_date}",
                )
        return None


class DiagnosisPrecedesTestConstraint(MedicalTemporalConstraint):
    """
    软约束：确认性诊断不应早于相关检查结果。

    例：如果血糖测量在 2024-01-20，而 "Type 2 Diabetes" 诊断标注为 2024-01-15，
    则诊断候选的置信度应被降低（但不为零，因为可能是基于其他证据）。
    """
    name = "diagnosis_precedes_test"

    MEASUREMENT_ATTRS = {"blood_glucose", "blood_pressure_sys", "blood_pressure_dia",
                         "heart_rate", "body_temperature", "hemoglobin"}

    def check(
        self,
        candidate: CandidateValue,
        existing_memories: List[CanonicalMemory],
        attribute: str,
    ) -> Optional[ConstraintResult]:
        if attribute != "primary_diagnosis":
            return None

        cand_date = candidate.t_event
        if not cand_date:
            return None

        try:
            cand_dt = datetime.strptime(cand_date, "%Y-%m-%d")
        except ValueError:
            return None

        for mem in existing_memories:
            if mem.attribute not in self.MEASUREMENT_ATTRS:
                continue
            mem_date = mem.t_event or mem.start_time
            if not mem_date:
                continue
            try:
                mem_dt = datetime.strptime(mem_date, "%Y-%m-%d")
            except ValueError:
                continue
            if cand_dt < mem_dt:
                return ConstraintResult(
                    violated=True,
                    constraint_name=self.name,
                    severity="soft",
                    penalty=0.5,
                    reason=f"Diagnosis date {cand_date} precedes test date {mem_date}",
                )
        return None


class TemporalOrderConstraint(MedicalTemporalConstraint):
    """
    软约束：事件时间不应早于对话中的首次记录时间。

    如果候选的 t_event 早于所有已有记忆的最早 t_event 超过合理范围，
    可能是时间解析错误。
    """
    name = "temporal_order"

    # 允许的最大回溯天数（超过则视为异常）
    MAX_LOOKBACK_DAYS = 365 * 5  # 5 年

    def check(
        self,
        candidate: CandidateValue,
        existing_memories: List[CanonicalMemory],
        attribute: str,
    ) -> Optional[ConstraintResult]:
        cand_date = candidate.t_event
        if not cand_date:
            return None

        try:
            cand_dt = datetime.strptime(cand_date, "%Y-%m-%d")
        except ValueError:
            return None

        earliest = None
        for mem in existing_memories:
            mem_date = mem.t_event or mem.start_time
            if not mem_date:
                continue
            try:
                mem_dt = datetime.strptime(mem_date, "%Y-%m-%d")
                if earliest is None or mem_dt < earliest:
                    earliest = mem_dt
            except ValueError:
                continue

        if earliest is None:
            return None

        diff_days = (earliest - cand_dt).days
        if diff_days > self.MAX_LOOKBACK_DAYS:
            return ConstraintResult(
                violated=True,
                constraint_name=self.name,
                severity="soft",
                penalty=0.3,
                reason=f"Event date {cand_date} is {diff_days} days before earliest known record",
            )
        return None


class DoseMonotonicityConstraint(MedicalTemporalConstraint):
    """
    软约束：药物剂量变化应遵循临床模式。

    大幅度的剂量跳跃（如从 5mg 直接到 500mg）在临床上不典型，
    可能是提取错误。
    """
    name = "dose_monotonicity"

    MAX_DOSE_RATIO = 10.0  # 单次变化不超过 10 倍

    DOSE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*mg\b", re.IGNORECASE)

    def check(
        self,
        candidate: CandidateValue,
        existing_memories: List[CanonicalMemory],
        attribute: str,
    ) -> Optional[ConstraintResult]:
        if attribute != "medication":
            return None

        cand_dose = self._extract_dose(candidate.value)
        if cand_dose is None:
            return None

        for mem in existing_memories:
            if mem.attribute != "medication":
                continue
            mem_dose = self._extract_dose(mem.value)
            if mem_dose is None or mem_dose == 0:
                continue
            ratio = max(cand_dose, mem_dose) / max(min(cand_dose, mem_dose), 0.01)
            if ratio >= self.MAX_DOSE_RATIO:
                return ConstraintResult(
                    violated=True,
                    constraint_name=self.name,
                    severity="soft",
                    penalty=0.4,
                    reason=f"Dose change ratio {ratio:.1f}x exceeds threshold ({cand_dose}mg vs {mem_dose}mg)",
                )
        return None

    def _extract_dose(self, text: str) -> Optional[float]:
        m = self.DOSE_PATTERN.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None


# ── 约束检查器 ────────────────────────────────────────────────────────────────

DEFAULT_CONSTRAINTS = [
    MedicationStartStopConstraint(),
    DiagnosisPrecedesTestConstraint(),
    TemporalOrderConstraint(),
    DoseMonotonicityConstraint(),
]


class ConstraintChecker:
    """
    运行所有形式约束并调整候选置信度。

    集成点：在 UniquenessManager 的 _merge_unique/_merge_latest 中，
    score_candidates() 之后、最终选择之前调用。
    """

    def __init__(
        self,
        constraints: Optional[List[MedicalTemporalConstraint]] = None,
        enabled: bool = True,
    ):
        self.constraints = constraints if constraints is not None else DEFAULT_CONSTRAINTS
        self.enabled = enabled

    def check_and_adjust(
        self,
        candidates: List[CandidateValue],
        existing_memories: List[CanonicalMemory],
        attribute: str,
    ) -> List[CandidateValue]:
        """
        对每个候选检查所有约束，违反者降低 confidence。

        Returns:
            调整后的候选列表（按 confidence 降序）。
        """
        if not self.enabled or not candidates:
            return candidates

        for cand in candidates:
            violations = []
            for constraint in self.constraints:
                result = constraint.check(cand, existing_memories, attribute)
                if result is not None and result.violated:
                    violations.append(result)
                    if result.severity == "hard":
                        cand.confidence = 0.0
                        break
                    else:
                        cand.confidence *= result.penalty

        # 重新排序
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates
