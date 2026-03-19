from types import SimpleNamespace

from benchmarks.base_task import DialogTurn, UnifiedSample
from experiments import eval_our_method, run_ablation
from src.uniq_cluster_memory.m1_event_extraction import ExtractedEvent
from src.uniq_cluster_memory.schema import CanonicalMemory


def test_run_ablation_parser_defaults_to_embedding_enabled():
    parser = run_ablation.build_arg_parser()

    assert parser.parse_args([]).use_embedding is True
    assert parser.parse_args(["--no_embedding"]).use_embedding is False
    assert parser.parse_args(["--use_embedding"]).use_embedding is True


def test_ablation_pipeline_passes_bundle_graph_to_uniqueness_manager(monkeypatch):
    captured = {}

    class FakeUniquenessManager:
        def __init__(self, *args, **kwargs):
            captured["manager_kwargs"] = kwargs

        def process(self, clusters, patient_id, bundle_graph=None):
            captured["clusters"] = clusters
            captured["patient_id"] = patient_id
            captured["bundle_graph"] = bundle_graph
            return [
                CanonicalMemory(
                    patient_id=patient_id,
                    attribute="medication",
                    value="Metformin 500mg bid",
                    time_scope="global",
                    update_policy="latest",
                )
            ]

    monkeypatch.setattr(run_ablation, "UniquenessManager", FakeUniquenessManager)

    pipeline = run_ablation.AblationPipeline(
        run_ablation.ABLATION_CONFIGS["full"],
        use_embedding=False,
    )
    event = ExtractedEvent(
        event_id="e1",
        dialogue_id="dlg1",
        attribute="medication",
        value="Metformin 500mg bid",
        unit="",
        time_expr="today",
        update_policy="latest",
        confidence=1.0,
        provenance=[1],
        speaker="doctor",
        raw_text_snippet="Continue metformin 500mg bid.",
    )

    pipeline.m1 = SimpleNamespace(extract=lambda dialogue, dialogue_id: [event])
    pipeline.m2 = SimpleNamespace(cluster=lambda events, dialogue_id: ["cluster-1"])
    pipeline.m25 = SimpleNamespace(
        build=lambda events, dialogue_id: {"dialogue_id": dialogue_id, "n_events": len(events)}
    )
    pipeline.m4 = SimpleNamespace(compress=lambda memories, patient_id: memories)

    memories = pipeline.build_memory(
        dialogue=[{"turn_id": 0, "speaker": "doctor", "text": "Continue metformin 500mg bid."}],
        dialogue_id="dlg1",
        dialogue_date="2025-01-01",
    )

    assert memories[0].patient_id == "dlg1"
    assert captured["clusters"] == ["cluster-1"]
    assert captured["bundle_graph"] == {"dialogue_id": "dlg1", "n_events": 1}
    assert captured["manager_kwargs"]["missing_time_scope"] == "reference_day"


def test_eval_our_method_records_runtime_config(monkeypatch):
    captured = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured["pipeline_kwargs"] = kwargs

        def build_memory(self, dialogue, dialogue_id, dialogue_date=None):
            return [
                CanonicalMemory(
                    patient_id=dialogue_id,
                    attribute="blood_glucose",
                    value="7.1",
                    unit="mmol/L",
                    time_scope="2025-01-01",
                    update_policy="unique",
                )
            ]

    monkeypatch.setattr(eval_our_method, "UniqueClusterMemoryPipeline", FakePipeline)
    monkeypatch.setattr(
        eval_our_method,
        "compute_unique_f1",
        lambda predicted, gt: SimpleNamespace(
            f1=1.0,
            relaxed_f1=1.0,
            attribute_coverage=1.0,
            redundancy=0.0,
            n_predicted=len(predicted),
            n_gt=len(gt),
        ),
    )
    monkeypatch.setattr(
        eval_our_method,
        "compute_conflict_f1",
        lambda predicted, gt: SimpleNamespace(f1=0.0),
    )
    monkeypatch.setattr(
        eval_our_method,
        "aggregate_unique_f1",
        lambda metrics: SimpleNamespace(
            mean_f1=1.0,
            mean_relaxed_f1=1.0,
            mean_attribute_coverage=1.0,
            mean_precision=1.0,
            mean_recall=1.0,
            mean_redundancy=0.0,
            mean_coverage=1.0,
            n_samples=len(metrics),
        ),
    )
    monkeypatch.setattr(
        eval_our_method,
        "aggregate_conflict_f1",
        lambda metrics: SimpleNamespace(
            mean_f1=0.0,
            mean_precision=0.0,
            mean_recall=0.0,
        ),
    )

    sample = UnifiedSample(
        sample_id="dlg1",
        source="med_longmem",
        question="Summarize the patient memory.",
        answer="[]",
        dialog_history=[DialogTurn(role="user", content="My blood glucose was 7.1 today.")],
        question_date="2025-01-01",
        metadata={"canonical_gt": []},
    )

    result = eval_our_method.evaluate_our_method(
        samples=[sample],
        w_struct=0.7,
        use_embedding=False,
    )

    assert captured["pipeline_kwargs"]["use_embedding"] is False
    assert result["config"]["use_embedding"] is False
    assert result["config"]["missing_time_scope"] == "reference_day"
    assert result["config"]["bundle_graph_enabled"] is True
