import json
from pathlib import Path

from experiments.run_core_comparison import (
    aggregate_pipeline_jsonl,
    build_report,
    choose_largest_result_file,
    load_summary,
    load_system_from_eval,
)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_aggregate_pipeline_jsonl_computes_macro_means(tmp_path: Path):
    result_path = tmp_path / "pipeline_med_longmem_w7.jsonl"
    result_path.write_text(
        "\n".join(
            [
                json.dumps({"unique_f1_strict": 0.8, "unique_f1_relaxed": 0.9, "conflict_f1": 1.0, "attribute_coverage": 1.0, "latency": 10.0}),
                json.dumps({"unique_f1_strict": 0.6, "unique_f1_relaxed": 0.7, "conflict_f1": 0.5, "attribute_coverage": 0.8, "latency": 20.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = aggregate_pipeline_jsonl(result_path)

    assert metrics["n_samples"] == 2
    assert metrics["unique_f1"] == 0.7
    assert metrics["unique_relaxed_f1"] == 0.8
    assert metrics["conflict_f1"] == 0.75
    assert metrics["mean_attribute_coverage"] == 0.9
    assert metrics["avg_latency"] == 15.0


def test_choose_largest_result_file_prefers_more_samples(tmp_path: Path):
    small = tmp_path / "pipeline_med_longmem_w7.jsonl"
    large = tmp_path / "pipeline_med_longmem_w8.jsonl"
    small.write_text("{}\n", encoding="utf-8")
    large.write_text("{}\n{}\n", encoding="utf-8")

    selected = choose_largest_result_file(tmp_path, ("pipeline_med_longmem_w*.jsonl",))

    assert selected == large


def test_build_report_uses_f1_and_recall_deltas(tmp_path: Path):
    eval_path = tmp_path / "med_longmem_v01_eval.json"
    with_memory_summary = tmp_path / "longmemeval_hybrid_rag_summary.json"
    no_memory_summary = tmp_path / "longmemeval_no_memory_summary.json"

    _write_json(
        eval_path,
        [
            {
                "system": "No_Memory",
                "unique_f1": 0.1,
                "n_samples": 20,
            }
        ],
    )
    _write_json(
        with_memory_summary,
        {
            "recall_at_5": 0.63,
            "n_samples": 50,
            "accuracy": 0.4,
            "mean_quality_score": 2.66,
        },
    )
    _write_json(
        no_memory_summary,
        {
            "n_samples": 50,
            "accuracy": 0.02,
            "mean_quality_score": 1.84,
        },
    )

    med_no_memory = load_system_from_eval(eval_path, "No_Memory")
    lme_with = load_summary(with_memory_summary, "Hybrid_RAG")
    lme_without = load_summary(no_memory_summary, "No_Memory")
    lme_without["recall_at_5"] = 0.0

    report = build_report(
        medlongmem_with_memory={
            "system": "UCM",
            "unique_f1": 0.75,
            "n_samples": 20,
            "source_file": "ucm.jsonl",
        },
        medlongmem_no_memory=med_no_memory,
        longmemeval_with_memory=lme_with,
        longmemeval_no_memory=lme_without,
    )

    assert report["med_longmem"]["delta"]["absolute"] == 0.65
    assert report["longmemeval"]["delta"]["absolute"] == 0.63
    assert report["executive_summary"]["defense_ready_metrics"]["f1"] == 0.75
