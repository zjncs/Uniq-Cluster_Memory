"""
tests/test_causal_scorer.py
============================
测试因果去混淆共指评分。
"""

from src.uniq_cluster_memory.m1_event_extraction import ExtractedEvent
from src.uniq_cluster_memory.m2_clustering.causal_scorer import CausalCoreferenceScorer


def _make_event(
    event_id: str = "e1",
    attribute: str = "medication",
    value: str = "Metformin 500mg",
    time_expr: str = "today",
    speaker: str = "doctor",
    provenance: list = None,
) -> ExtractedEvent:
    return ExtractedEvent(
        event_id=event_id,
        dialogue_id="dlg1",
        attribute=attribute,
        value=value,
        unit="",
        time_expr=time_expr,
        update_policy="latest",
        confidence=0.9,
        provenance=provenance or [1],
        speaker=speaker,
        raw_text_snippet=value,
    )


def test_same_medication_different_context():
    """同药名但不同语境：去混淆分数应低于原始分数。"""
    scorer = CausalCoreferenceScorer()
    evt_a = _make_event(event_id="e1", value="Metformin 500mg", time_expr="today")
    evt_b = _make_event(event_id="e2", value="Metformin 1000mg", time_expr="today")

    raw = scorer.compute_raw_similarity(evt_a, evt_b)
    deconf = scorer.deconfounded_score(evt_a, evt_b)
    assert deconf < raw, f"Deconfounded ({deconf}) should be < raw ({raw})"


def test_different_medications():
    """完全不同的药物应有低分数。"""
    scorer = CausalCoreferenceScorer()
    evt_a = _make_event(event_id="e1", value="Metformin 500mg", attribute="medication")
    evt_b = _make_event(event_id="e2", value="Lisinopril 10mg", attribute="medication")

    score = scorer.deconfounded_score(evt_a, evt_b)
    assert score < 0.5, f"Different medications should score < 0.5, got {score}"


def test_identical_events_high_score():
    """完全相同的事件应有高分数。"""
    scorer = CausalCoreferenceScorer()
    evt_a = _make_event(event_id="e1", value="Metformin 500mg daily", time_expr="today")
    evt_b = _make_event(event_id="e2", value="Metformin 500mg daily", time_expr="today")

    score = scorer.deconfounded_score(evt_a, evt_b)
    assert score > 0.0, f"Identical events should score > 0, got {score}"


def test_confounders_identified():
    """应正确识别混淆因子。"""
    scorer = CausalCoreferenceScorer()
    evt_a = _make_event(event_id="e1", value="Metformin 500mg", time_expr="today", speaker="doctor")
    evt_b = _make_event(event_id="e2", value="Metformin 1000mg", time_expr="today", speaker="doctor")

    confounders = scorer.identify_confounders(evt_a, evt_b)
    assert "same_name" in confounders
    assert "temporal_proximity" in confounders
    assert "same_speaker" in confounders


def test_analyze_returns_full_details():
    """analyze() 应返回完整分析结果。"""
    scorer = CausalCoreferenceScorer()
    evt_a = _make_event(event_id="e1", value="Metformin 500mg")
    evt_b = _make_event(event_id="e2", value="Metformin 1000mg")

    analysis = scorer.analyze(evt_a, evt_b)
    assert analysis.raw_similarity >= 0
    assert analysis.deconfounded_score >= 0
    assert isinstance(analysis.active_confounders, list)
    assert isinstance(analysis.confounder_adjustments, dict)


def test_cross_attribute_no_merge():
    """不同属性的事件不应被合并。"""
    scorer = CausalCoreferenceScorer()
    evt_a = _make_event(event_id="e1", attribute="medication", value="Metformin 500mg")
    evt_b = _make_event(event_id="e2", attribute="blood_glucose", value="7.2 mmol/L")

    assert scorer.should_merge(evt_a, evt_b) is False
