"""
tests/test_confidence_scoring.py
=================================
测试多候选置信度评分和冲突判定。
"""

from src.uniq_cluster_memory.schema import CandidateValue
from src.uniq_cluster_memory.m3_uniqueness.conflict_detector import ConflictDetector


def test_score_candidates_normalizes():
    """评分后所有候选的 confidence 应归一化到 1.0。"""
    detector = ConflictDetector()
    candidates = [
        CandidateValue(value="7.2", confidence=0.9, speaker="doctor",
                       source_authority=1.0, temporal_recency=0.8, evidence_count=3),
        CandidateValue(value="8.5", confidence=0.7, speaker="patient",
                       source_authority=0.5, temporal_recency=0.6, evidence_count=1),
    ]
    scored = detector.score_candidates(candidates, "blood_glucose")
    total = sum(c.confidence for c in scored)
    assert abs(total - 1.0) < 0.01, f"Total confidence should be ~1.0, got {total}"
    assert scored[0].confidence > scored[1].confidence


def test_score_candidates_merges_equivalent():
    """等价候选应合并，证据计数累加。"""
    detector = ConflictDetector()
    candidates = [
        CandidateValue(value="Metformin 500mg daily", confidence=0.8,
                       speaker="doctor", source_authority=1.0,
                       temporal_recency=0.9, evidence_count=1),
        CandidateValue(value="Metformin 500mg qd", confidence=0.7,
                       speaker="doctor", source_authority=1.0,
                       temporal_recency=0.7, evidence_count=1),
    ]
    scored = detector.score_candidates(candidates, "medication")
    # daily ≡ qd → 应合并为 1 个候选
    assert len(scored) == 1
    assert scored[0].evidence_count == 2


def test_should_flag_conflict_single_candidate():
    """单个候选不应标记冲突。"""
    scored = [CandidateValue(value="7.2", confidence=0.9)]
    assert ConflictDetector.should_flag_conflict(scored) is False


def test_should_flag_conflict_close_scores():
    """前两名差距小于阈值应标记冲突。"""
    scored = [
        CandidateValue(value="7.2", confidence=0.52),
        CandidateValue(value="8.5", confidence=0.48),
    ]
    assert ConflictDetector.should_flag_conflict(scored) is True


def test_should_flag_conflict_low_top():
    """Top 候选低于阈值应标记冲突。"""
    scored = [
        CandidateValue(value="7.2", confidence=0.6),
        CandidateValue(value="8.5", confidence=0.4),
    ]
    assert ConflictDetector.should_flag_conflict(scored) is True
