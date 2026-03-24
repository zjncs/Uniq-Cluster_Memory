"""
tests/test_formal_constraints.py
=================================
测试医疗领域形式约束。
"""

from src.uniq_cluster_memory.schema import CanonicalMemory, CandidateValue
from src.uniq_cluster_memory.m3_uniqueness.formal_constraints import (
    MedicationStartStopConstraint,
    DiagnosisPrecedesTestConstraint,
    DoseMonotonicityConstraint,
    TemporalOrderConstraint,
    ConstraintChecker,
)


def test_medication_start_stop_same_day():
    """同日开始和停止同一药物应触发硬约束。"""
    constraint = MedicationStartStopConstraint()
    candidate = CandidateValue(
        value="Metformin stopped",
        t_event="2024-01-15",
    )
    existing = [
        CanonicalMemory(
            patient_id="p1",
            attribute="medication",
            value="Metformin started",
            time_scope="2024-01-15",
            t_event="2024-01-15",
        ),
    ]
    result = constraint.check(candidate, existing, "medication")
    assert result is not None
    assert result.violated is True
    assert result.severity == "hard"
    assert result.penalty == 0.0


def test_medication_start_stop_different_day():
    """不同日期开始和停止不应触发约束。"""
    constraint = MedicationStartStopConstraint()
    candidate = CandidateValue(
        value="Metformin stopped",
        t_event="2024-01-20",
    )
    existing = [
        CanonicalMemory(
            patient_id="p1",
            attribute="medication",
            value="Metformin started",
            time_scope="2024-01-15",
            t_event="2024-01-15",
        ),
    ]
    result = constraint.check(candidate, existing, "medication")
    assert result is None


def test_diagnosis_precedes_test():
    """诊断早于检查结果应触发软约束。"""
    constraint = DiagnosisPrecedesTestConstraint()
    candidate = CandidateValue(
        value="Type 2 Diabetes",
        t_event="2024-01-10",
    )
    existing = [
        CanonicalMemory(
            patient_id="p1",
            attribute="blood_glucose",
            value="11.2",
            time_scope="2024-01-15",
            t_event="2024-01-15",
        ),
    ]
    result = constraint.check(candidate, existing, "primary_diagnosis")
    assert result is not None
    assert result.violated is True
    assert result.severity == "soft"
    assert result.penalty == 0.5


def test_dose_monotonicity_extreme_jump():
    """极端剂量跳跃应触发软约束。"""
    constraint = DoseMonotonicityConstraint()
    candidate = CandidateValue(value="Metformin 5000mg daily")
    existing = [
        CanonicalMemory(
            patient_id="p1",
            attribute="medication",
            value="Metformin 500mg daily",
            time_scope="2024-01-15",
        ),
    ]
    result = constraint.check(candidate, existing, "medication")
    assert result is not None
    assert result.violated is True


def test_constraint_checker_adjusts_confidence():
    """ConstraintChecker 应降低违反约束的候选的置信度。"""
    checker = ConstraintChecker()
    candidates = [
        CandidateValue(
            value="Metformin stopped",
            confidence=0.8,
            t_event="2024-01-15",
        ),
        CandidateValue(
            value="Metformin 500mg daily",
            confidence=0.7,
            t_event="2024-01-15",
        ),
    ]
    existing = [
        CanonicalMemory(
            patient_id="p1",
            attribute="medication",
            value="Metformin started",
            time_scope="2024-01-15",
            t_event="2024-01-15",
        ),
    ]
    adjusted = checker.check_and_adjust(candidates, existing, "medication")
    # "Metformin stopped" 违反硬约束 → confidence = 0.0
    stopped = [c for c in adjusted if "stopped" in c.value]
    assert stopped[0].confidence == 0.0
    # "Metformin 500mg daily" 应排在前面
    assert adjusted[0].value == "Metformin 500mg daily"


def test_constraint_checker_disabled():
    """禁用时不应修改候选。"""
    checker = ConstraintChecker(enabled=False)
    candidates = [
        CandidateValue(value="Metformin stopped", confidence=0.8, t_event="2024-01-15"),
    ]
    adjusted = checker.check_and_adjust(candidates, [], "medication")
    assert adjusted[0].confidence == 0.8
