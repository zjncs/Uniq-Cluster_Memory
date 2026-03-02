from evaluation.uniqueness_eval import compute_unique_f1
from src.uniq_cluster_memory.m1_event_extraction import ExtractedEvent
from src.uniq_cluster_memory.m2_clustering import (
    AttributeCluster,
    BundleGraph,
    EventBundle,
    EntityBundle,
    BundleLink,
)
from src.uniq_cluster_memory.m3_uniqueness import UniquenessManager
from src.uniq_cluster_memory.m5_retrieval import HybridMemoryRetriever
from src.uniq_cluster_memory.schema import CanonicalMemory


def _event(
    value: str,
    time_expr: str,
    provenance: list[int],
    update_policy: str = "latest",
    attribute: str = "medication",
    confidence: float = 1.0,
    speaker: str = "doctor",
) -> ExtractedEvent:
    return ExtractedEvent(
        event_id=f"e_{value}_{time_expr}",
        dialogue_id="d1",
        attribute=attribute,
        value=value,
        unit="",
        time_expr=time_expr,
        update_policy=update_policy,
        confidence=confidence,
        provenance=provenance,
        speaker=speaker,
        raw_text_snippet=value,
    )


def test_m3_latest_is_global_not_per_scope():
    cluster = AttributeCluster(
        cluster_id="c1",
        canonical_attribute="medication",
        update_policy="latest",
        events=[
            _event("metformin 500mg", "2025-01-01", [2]),
            _event("insulin 10u", "2025-01-10", [9]),
        ],
    )
    manager = UniquenessManager(dialogue_date="2025-01-15")
    memories = manager.process([cluster], patient_id="p1")

    assert len(memories) == 1
    assert memories[0].attribute == "medication"
    assert memories[0].value == "insulin 10u"
    assert memories[0].time_scope == "global"
    assert memories[0].update_policy == "latest"


def test_m5_can_infer_attribute_without_explicit_query_attribute():
    memories = [
        CanonicalMemory(
            patient_id="p1",
            attribute="blood_glucose",
            value="9.0",
            time_scope="2024-01-01",
            provenance=[2],
            update_policy="unique",
        ),
        CanonicalMemory(
            patient_id="p1",
            attribute="blood_glucose",
            value="7.2",
            time_scope="2024-02-01",
            provenance=[10],
            update_policy="unique",
        ),
        CanonicalMemory(
            patient_id="p1",
            attribute="heart_rate",
            value="88",
            time_scope="2024-02-02",
            provenance=[11],
            update_policy="unique",
        ),
    ]
    retriever = HybridMemoryRetriever(use_embedding=False, top_k=3)
    results = retriever.retrieve("what is my latest blood sugar?", memories)

    assert results
    assert results[0].memory.attribute == "blood_glucose"
    assert results[0].memory.time_scope == "2024-02-01"


def test_uniqueness_redundancy_not_penalize_unique_different_dates():
    gt = [
        CanonicalMemory(
            patient_id="p1",
            attribute="blood_glucose",
            value="8.9",
            time_scope="2024-01-01",
            update_policy="unique",
        ),
        CanonicalMemory(
            patient_id="p1",
            attribute="blood_glucose",
            value="7.1",
            time_scope="2024-01-10",
            update_policy="unique",
        ),
    ]
    predicted = [
        CanonicalMemory(
            patient_id="p1",
            attribute="blood_glucose",
            value="8.9",
            time_scope="2024-01-01",
            update_policy="unique",
        ),
        CanonicalMemory(
            patient_id="p1",
            attribute="blood_glucose",
            value="7.1",
            time_scope="2024-01-10",
            update_policy="unique",
        ),
    ]
    metrics = compute_unique_f1(predicted, gt)
    assert metrics.f1 == 1.0
    assert metrics.redundancy == 0.0


def test_m3_unique_prefers_recent_source_over_confidence():
    cluster = AttributeCluster(
        cluster_id="c_u1",
        canonical_attribute="blood_pressure_sys",
        update_policy="unique",
        events=[
            _event(
                value="110",
                time_expr="global",
                provenance=[1],
                update_policy="unique",
                attribute="blood_pressure_sys",
                confidence=0.99,
                speaker="patient",
            ),
            _event(
                value="129",
                time_expr="global",
                provenance=[10],
                update_policy="unique",
                attribute="blood_pressure_sys",
                confidence=0.8,
                speaker="doctor",
            ),
        ],
    )
    manager = UniquenessManager(dialogue_date="2025-01-15")
    memories = manager.process([cluster], patient_id="p1")

    assert len(memories) == 1
    assert memories[0].value == "129"


def test_m3_latest_medication_equivalent_not_marked_conflict():
    cluster = AttributeCluster(
        cluster_id="c_l1",
        canonical_attribute="medication",
        update_policy="latest",
        events=[
            _event("Metformin 500mg bid", "global", [5]),
            _event("Metformin 500mg twice daily", "global", [7]),
        ],
    )
    manager = UniquenessManager(dialogue_date="2025-01-15")
    memories = manager.process([cluster], patient_id="p1")

    assert len(memories) == 1
    assert memories[0].conflict_flag is False
    assert len(memories[0].conflict_history) == 0


def test_m3_unique_small_numeric_change_is_conflict():
    cluster = AttributeCluster(
        cluster_id="c_u2",
        canonical_attribute="hemoglobin",
        update_policy="unique",
        events=[
            _event("178", "global", [2], update_policy="unique", attribute="hemoglobin"),
            _event("180", "global", [7], update_policy="unique", attribute="hemoglobin"),
        ],
    )
    manager = UniquenessManager(dialogue_date="2025-01-15")
    memories = manager.process([cluster], patient_id="p1")

    assert len(memories) == 1
    assert memories[0].value == "180"
    assert memories[0].conflict_flag is True
    assert len(memories[0].conflict_history) == 1


def test_m3_append_symptom_cap_keeps_most_frequent():
    cluster = AttributeCluster(
        cluster_id="c_a1",
        canonical_attribute="symptom",
        update_policy="append",
        events=[
            _event("dizziness", "global", [1], update_policy="append", attribute="symptom"),
            _event("dizziness", "global", [3], update_policy="append", attribute="symptom"),
            _event("fatigue", "global", [5], update_policy="append", attribute="symptom"),
        ],
    )
    manager = UniquenessManager(dialogue_date="2025-01-15", max_symptoms_per_scope=1)
    memories = manager.process([cluster], patient_id="p1")

    assert len(memories) == 1
    assert memories[0].attribute == "symptom"
    assert memories[0].value == "dizziness"


def test_m3_bundle_qualifiers_and_versions_present():
    cluster = AttributeCluster(
        cluster_id="c_l2",
        canonical_attribute="medication",
        update_policy="latest",
        events=[
            _event("Metformin 500mg bid", "2025-01-01", [3]),
            _event("Metformin 500mg qd", "2025-01-15", [9]),
        ],
    )
    bundle_graph = BundleGraph(
        dialogue_id="d1",
        entity_bundles=[
            EntityBundle(
                bundle_id="entity_d1_001",
                entity_type="medication",
                canonical_name="metformin",
                aliases=["Metformin 500mg bid", "Metformin 500mg qd"],
                provenance_turns=[3, 9],
                event_ids=["e_Metformin 500mg bid_2025-01-01", "e_Metformin 500mg qd_2025-01-15"],
            )
        ],
        event_bundles=[
            EventBundle(
                bundle_id="event_d1_001",
                time_anchor="2025-01-01",
                attributes={"medication": ["Metformin 500mg bid"]},
                provenance_turns=[3],
                event_ids=["e_Metformin 500mg bid_2025-01-01"],
            ),
            EventBundle(
                bundle_id="event_d1_002",
                time_anchor="2025-01-15",
                attributes={"medication": ["Metformin 500mg qd"]},
                provenance_turns=[9],
                event_ids=["e_Metformin 500mg qd_2025-01-15"],
            ),
        ],
        links=[
            BundleLink("event_d1_001", "entity_d1_001", "MENTIONS_MEDICATION"),
            BundleLink("event_d1_002", "entity_d1_001", "MENTIONS_MEDICATION"),
        ],
    )
    manager = UniquenessManager(dialogue_date="2025-01-20")
    memories = manager.process([cluster], patient_id="p1", bundle_graph=bundle_graph)

    assert len(memories) == 1
    q = memories[0].qualifiers
    assert q["medication_entity_name"] == "metformin"
    assert len(q["event_bundle_ids"]) == 2
    assert len(q["value_versions"]) == 2
    assert any(v["is_selected"] for v in q["value_versions"])
    assert q["decision_level"] == "bundle"
    assert q["selected_event_bundle_id"] == "event_d1_002"
    assert q["selected_event_bundle_time_anchor"] == "2025-01-15"


def test_m3_latest_conflict_reduced_by_bundle_competition():
    cluster = AttributeCluster(
        cluster_id="c_l3",
        canonical_attribute="medication",
        update_policy="latest",
        events=[
            _event("Lisinopril 10mg once daily", "2025-01-01", [1]),
            _event("Lisinopril 20mg once daily", "2025-01-01", [2]),
            _event("Amlodipine 5mg once daily", "2025-01-20", [8]),
        ],
    )
    bundle_graph = BundleGraph(
        dialogue_id="d1",
        event_bundles=[
            EventBundle(
                bundle_id="event_d1_old",
                time_anchor="2025-01-01",
                attributes={"medication": ["Lisinopril 10mg once daily", "Lisinopril 20mg once daily"]},
                provenance_turns=[1, 2],
                event_ids=[
                    "e_Lisinopril 10mg once daily_2025-01-01",
                    "e_Lisinopril 20mg once daily_2025-01-01",
                ],
            ),
            EventBundle(
                bundle_id="event_d1_new",
                time_anchor="2025-01-20",
                attributes={"medication": ["Amlodipine 5mg once daily"]},
                provenance_turns=[8],
                event_ids=["e_Amlodipine 5mg once daily_2025-01-20"],
            ),
        ],
    )
    manager = UniquenessManager(dialogue_date="2025-01-25")
    memories = manager.process([cluster], patient_id="p1", bundle_graph=bundle_graph)

    assert len(memories) == 1
    m = memories[0]
    assert m.value == "Amlodipine 5mg once daily"
    # 旧 bundle 内部的多版本不会重复计冲突，只按 bundle 级记录一次
    assert m.conflict_flag is True
    assert len(m.conflict_history) == 1
    assert m.qualifiers["selected_event_bundle_id"] == "event_d1_new"
    assert m.qualifiers["competing_event_bundle_ids"] == ["event_d1_old"]


def test_m3_unique_bundle_first_selection_and_conflict():
    cluster = AttributeCluster(
        cluster_id="c_u3",
        canonical_attribute="blood_pressure_sys",
        update_policy="unique",
        events=[
            _event("110", "global", [1], update_policy="unique", attribute="blood_pressure_sys"),
            _event("110", "global", [2], update_policy="unique", attribute="blood_pressure_sys"),
            _event("129", "global", [10], update_policy="unique", attribute="blood_pressure_sys"),
        ],
    )
    bundle_graph = BundleGraph(
        dialogue_id="d1",
        event_bundles=[
            EventBundle(
                bundle_id="event_d1_old_bp",
                time_anchor="turn:2",
                attributes={"blood_pressure_sys": ["110"]},
                provenance_turns=[1, 2],
                event_ids=["e_110_global"],
            ),
            EventBundle(
                bundle_id="event_d1_new_bp",
                time_anchor="turn:10",
                attributes={"blood_pressure_sys": ["129"]},
                provenance_turns=[10],
                event_ids=["e_129_global"],
            ),
        ],
    )
    manager = UniquenessManager(dialogue_date="2025-01-25")
    memories = manager.process([cluster], patient_id="p1", bundle_graph=bundle_graph)

    assert len(memories) == 1
    m = memories[0]
    assert m.value == "129"
    assert m.conflict_flag is True
    assert len(m.conflict_history) == 1
    assert m.qualifiers["decision_level"] == "bundle"
    assert m.qualifiers["selected_event_bundle_id"] == "event_d1_new_bp"
