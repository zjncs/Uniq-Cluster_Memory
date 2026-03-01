from evaluation.temporal_eval import compute_temporal_metrics, interval_iou
from src.uniq_cluster_memory.schema import CanonicalMemory


def _mem(
    patient_id: str,
    relation_type: str,
    value: str,
    start: str,
    end: str | None,
) -> CanonicalMemory:
    return CanonicalMemory(
        patient_id=patient_id,
        attribute="medication" if relation_type == "TAKES_DRUG" else "symptom",
        relation_type=relation_type,
        value=value,
        target_value=value,
        start_time=start,
        end_time=end,
        time_scope=f"{start}..{end}" if end else "global",
        update_policy="latest" if relation_type == "TAKES_DRUG" else "append",
        is_ongoing=end is None,
    )


def test_interval_iou_partial_overlap():
    a = _mem("p1", "TAKES_DRUG", "A", "2026-01-01", "2026-01-10")
    b = _mem("p1", "TAKES_DRUG", "A", "2026-01-05", "2026-01-15")
    # overlap: 6 days, union: 15 days
    assert round(interval_iou(a, b), 4) == 0.4


def test_compute_temporal_metrics_exact_and_iou():
    gt = [
        _mem("p1", "TAKES_DRUG", "DrugA 10mg qd", "2026-01-01", "2026-01-14"),
        _mem("p1", "TAKES_DRUG", "DrugB 5mg qd", "2026-01-15", None),
    ]
    pred = [
        _mem("p1", "TAKES_DRUG", "DrugA 10mg qd", "2026-01-01", "2026-01-14"),  # exact TP
        _mem("p1", "TAKES_DRUG", "DrugB 5mg qd", "2026-01-16", None),  # non-exact, but high IoU
    ]

    m = compute_temporal_metrics(pred, gt)
    assert m.temporal_precision == 0.5
    assert m.temporal_recall == 0.5
    assert m.temporal_f1 == 0.5
    assert 0.9 <= m.mean_interval_iou <= 1.0
