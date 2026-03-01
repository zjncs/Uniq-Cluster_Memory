from src.uniq_cluster_memory.m1_event_extraction import ExtractedEvent
from src.uniq_cluster_memory.m2_clustering import InformationBundleBuilder


def _evt(
    event_id: str,
    attribute: str,
    value: str,
    time_expr: str,
    provenance: list[int],
    speaker: str = "doctor",
) -> ExtractedEvent:
    return ExtractedEvent(
        event_id=event_id,
        dialogue_id="d1",
        attribute=attribute,
        value=value,
        unit="",
        time_expr=time_expr,
        update_policy="unique",
        confidence=1.0,
        provenance=provenance,
        speaker=speaker,
        raw_text_snippet=value,
    )


def test_medication_entity_bundle_merges_alias_like_mentions():
    builder = InformationBundleBuilder()
    events = [
        _evt("e1", "medication", "Metformin 500mg bid", "global", [2]),
        _evt("e2", "medication", "Metformin 500 mg twice daily", "global", [6]),
    ]
    graph = builder.build(events, dialogue_id="dlg1")

    assert len(graph.entity_bundles) == 1
    med = graph.entity_bundles[0]
    assert med.entity_type == "medication"
    assert med.canonical_name == "metformin"
    assert len(med.aliases) == 2
    assert "500 mg" in med.dosage_values
    assert "bid" in med.frequency_values


def test_event_bundle_groups_multi_attribute_same_time_anchor():
    builder = InformationBundleBuilder()
    events = [
        _evt("e1", "blood_glucose", "7.2", "this morning", [1], speaker="patient"),
        _evt("e2", "symptom", "fatigue", "this morning", [1], speaker="patient"),
        _evt("e3", "medication", "Metformin 500mg bid", "this morning", [1], speaker="doctor"),
    ]
    graph = builder.build(events, dialogue_id="dlg2")

    assert len(graph.event_bundles) == 1
    ev = graph.event_bundles[0]
    assert ev.time_anchor == "day:today"
    assert set(ev.attributes.keys()) == {"blood_glucose", "symptom", "medication"}
    assert len(graph.links) == 1
    assert graph.links[0].relation == "MENTIONS_MEDICATION"
