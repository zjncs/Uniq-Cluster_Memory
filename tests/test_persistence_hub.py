from src.uniq_cluster_memory.pipeline import UniqueClusterMemoryPipeline
from src.uniq_cluster_memory.schema import CanonicalMemory
from src.uniq_cluster_memory.stores import MemoryPersistenceHub


def test_persistence_hub_disabled_behaves_noop():
    hub = MemoryPersistenceHub(enable_qdrant=False, enable_neo4j=False)
    assert hub.enabled is False
    hub.upsert_memories([])
    assert hub.query_meds_on("2026-01-01", patient_id="p1") == []
    assert hub.vector_search("metformin", patient_id="p1", limit=3) == []
    hub.close()


def test_pipeline_vector_search_without_qdrant_returns_empty():
    p = UniqueClusterMemoryPipeline(
        enable_qdrant=False,
        enable_neo4j=False,
        persist_to_stores=False,
    )
    res = p.vector_search_memories("blood sugar", patient_id="p1")
    assert res == []
    p.close()


def test_pipeline_query_meds_on_fallback_local_when_store_disabled():
    p = UniqueClusterMemoryPipeline(
        enable_qdrant=False,
        enable_neo4j=False,
        persist_to_stores=False,
    )
    memories = [
        CanonicalMemory(
            patient_id="p1",
            attribute="medication",
            value="DrugA 10mg qd",
            start_time="2026-01-01",
            end_time="2026-01-10",
            relation_type="TAKES_DRUG",
            update_policy="latest",
        )
    ]
    out = p.query_meds_on("2026-01-05", memories, patient_id="p1")
    assert len(out) == 1
    assert out[0].value.startswith("DrugA")
    p.close()

