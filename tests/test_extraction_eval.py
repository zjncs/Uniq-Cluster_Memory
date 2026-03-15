from evaluation.extraction_eval import aggregate_extraction_f1, compute_extraction_f1
from src.uniq_cluster_memory.m1_event_extraction import ExtractedEvent


def _pred(
    *,
    attribute: str,
    value: str,
    unit: str = "",
    time_expr: str = "global",
    speaker: str = "patient",
) -> ExtractedEvent:
    return ExtractedEvent(
        event_id="e1",
        dialogue_id="d1",
        attribute=attribute,
        value=value,
        unit=unit,
        time_expr=time_expr,
        update_policy="unique",
        confidence=1.0,
        provenance=[1],
        speaker=speaker,
        raw_text_snippet=value,
    )


def test_extraction_eval_normalizes_measurement_units():
    predicted = [
        _pred(
            attribute="blood_glucose",
            value="180 mg/dL",
            time_expr="2024-01-15",
            speaker="patient",
        )
    ]
    gt = [
        {
            "attribute": "blood_glucose",
            "value": "10",
            "unit": "mmol/L",
            "time_scope": "2024-01-15",
            "speaker": "patient",
        }
    ]

    metrics = compute_extraction_f1(predicted, gt, dialogue_date="2024-01-15")

    assert metrics.event_f1 == 1.0
    assert metrics.field_f1["unit"] == 1.0


def test_extraction_eval_grounds_relative_time():
    predicted = [
        _pred(
            attribute="symptom",
            value="dizziness",
            time_expr="yesterday",
            speaker="patient",
        )
    ]
    gt = [
        {
            "attribute": "symptom",
            "value": "dizziness",
            "unit": "",
            "time_scope": "2024-01-15",
            "speaker": "patient",
        }
    ]

    metrics = compute_extraction_f1(predicted, gt, dialogue_date="2024-01-16")

    assert metrics.event_f1 == 1.0
    assert metrics.field_f1["time_scope"] == 1.0


def test_extraction_eval_relaxed_match_survives_time_mismatch():
    predicted = [
        _pred(
            attribute="medication",
            value="Metformin 500 mg twice daily",
            time_expr="global",
            speaker="doctor",
        )
    ]
    gt = [
        {
            "attribute": "medication",
            "value": "Metformin 500mg bid",
            "unit": "",
            "time_scope": "2024-01-15",
            "speaker": "doctor",
        }
    ]

    metrics = compute_extraction_f1(predicted, gt, dialogue_date="2024-01-16")

    assert metrics.event_f1 == 0.0
    assert metrics.relaxed_event_f1 == 1.0


def test_aggregate_extraction_f1_macro_means():
    m1 = compute_extraction_f1(
        [_pred(attribute="symptom", value="fatigue", time_expr="2024-01-15")],
        [{"attribute": "symptom", "value": "fatigue", "unit": "", "time_scope": "2024-01-15", "speaker": "patient"}],
        dialogue_date="2024-01-15",
    )
    m2 = compute_extraction_f1(
        [_pred(attribute="symptom", value="fatigue", time_expr="global")],
        [{"attribute": "symptom", "value": "fatigue", "unit": "", "time_scope": "2024-01-15", "speaker": "patient"}],
        dialogue_date="2024-01-15",
    )

    agg = aggregate_extraction_f1([m1, m2])

    assert agg.n_samples == 2
    assert agg.mean_event_f1 == 0.5
    assert agg.mean_relaxed_event_f1 == 1.0
