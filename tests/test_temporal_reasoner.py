from src.uniq_cluster_memory.m1_event_extraction import ExtractedEvent
from src.uniq_cluster_memory.m2_clustering import AttributeCluster
from src.uniq_cluster_memory.m3_uniqueness import UniquenessManager
from src.uniq_cluster_memory.schema import CanonicalMemory, scope_to_interval
from src.uniq_cluster_memory.temporal_reasoning import TemporalReasoner


def _event(value: str, time_expr: str, provenance: list[int]) -> ExtractedEvent:
    return ExtractedEvent(
        event_id=f"e_{value}_{time_expr}",
        dialogue_id="d_temporal",
        attribute="medication",
        value=value,
        unit="",
        time_expr=time_expr,
        update_policy="latest",
        confidence=1.0,
        provenance=provenance,
        speaker="doctor",
        raw_text_snippet=value,
    )


def test_scope_to_interval_parsing():
    assert scope_to_interval("2026-01-10") == ("2026-01-10", "2026-01-10", "day")
    assert scope_to_interval("2026-W03") == ("2026-01-12", "2026-01-18", "week")
    assert scope_to_interval("2026-02") == ("2026-02-01", "2026-02-28", "month")
    assert scope_to_interval("2026") == ("2026-01-01", "2026-12-31", "year")
    assert scope_to_interval("2026-01-01..2026-01-10") == (
        "2026-01-01",
        "2026-01-10",
        "range",
    )


def test_medication_qualifiers_are_auto_extracted():
    mem = CanonicalMemory(
        patient_id="p1",
        attribute="medication",
        value="Metformin 500mg bid after breakfast po",
        time_scope="global",
        update_policy="latest",
    )
    assert mem.relation_type == "TAKES_DRUG"
    assert mem.dosage in {"500 mg", "500mg"}
    assert mem.frequency == "bid"
    assert mem.time_of_day == "after_breakfast"
    assert mem.route == "oral"


def test_query_meds_on_uses_med_timeline():
    cluster = AttributeCluster(
        cluster_id="c_med",
        canonical_attribute="medication",
        update_policy="latest",
        events=[
            _event("DrugA 10mg qd", "2026-01-01", [2]),
            _event("DrugB 5mg qd", "2026-01-15", [9]),
        ],
    )
    manager = UniquenessManager(dialogue_date="2026-01-20")
    memories = manager.process([cluster], patient_id="p1")
    assert len(memories) == 1

    reasoner = TemporalReasoner()
    meds_0110 = reasoner.query_meds_on("2026-01-10", memories, patient_id="p1")
    meds_0120 = reasoner.query_meds_on("2026-01-20", memories, patient_id="p1")

    assert len(meds_0110) == 1
    assert meds_0110[0].value.lower().startswith("druga")
    assert len(meds_0120) == 1
    assert meds_0120[0].value.lower().startswith("drugb")


def test_query_symptoms_between_interval_overlap():
    reasoner = TemporalReasoner()
    memories = [
        CanonicalMemory(
            patient_id="p1",
            attribute="symptom",
            value="cough",
            start_time="2026-01-01",
            end_time="2026-01-05",
            time_scope="2026-01-01..2026-01-05",
            update_policy="append",
        ),
        CanonicalMemory(
            patient_id="p1",
            attribute="symptom",
            value="fever",
            start_time="2026-01-10",
            end_time="2026-01-12",
            time_scope="2026-01-10..2026-01-12",
            update_policy="append",
        ),
    ]
    out = reasoner.query_symptoms_between("2026-01-04", "2026-01-11", memories, patient_id="p1")
    values = {m.value for m in out}
    assert values == {"cough", "fever"}


def test_query_relations_between_generic():
    reasoner = TemporalReasoner()
    memories = [
        CanonicalMemory(
            patient_id="p1",
            attribute="medication",
            value="DrugA 10mg qd",
            start_time="2026-01-01",
            end_time="2026-01-20",
            relation_type="TAKES_DRUG",
            update_policy="latest",
        ),
        CanonicalMemory(
            patient_id="p1",
            attribute="symptom",
            value="cough",
            start_time="2026-01-02",
            end_time="2026-01-03",
            relation_type="HAS_SYMPTOM",
            update_policy="append",
        ),
    ]
    out = reasoner.query_relations_between(
        "2026-01-02",
        "2026-01-10",
        memories,
        patient_id="p1",
        relation_type="TAKES_DRUG",
    )
    assert len(out) == 1
    assert out[0].relation_type == "TAKES_DRUG"
