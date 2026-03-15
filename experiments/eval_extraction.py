"""
eval_extraction.py
==================
在 Med-LongMem 上评测 M1 事件抽取质量。

输出：
- per-sample 详细结果
- 宏平均 Event-F1（strict / relaxed）
- field-level F1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.med_longmem_task import MedLongMemTask
from evaluation.extraction_eval import (
    aggregate_extraction_f1,
    compute_extraction_f1,
)
from src.uniq_cluster_memory.defaults import recommended_pipeline_options
from src.uniq_cluster_memory.m1_event_extraction import MedicalEventExtractor
from src.uniq_cluster_memory.utils.llm_client import ensure_llm_api_key


DEFAULT_OUTPUT_PATH = Path("results/main_results/extraction_eval_med_longmem.json")


def _sample_to_dialogue(sample) -> list[dict]:
    return [
        {
            "turn_id": i,
            "speaker": "patient" if turn.role == "user" else "doctor",
            "text": turn.content,
        }
        for i, turn in enumerate(sample.dialog_history)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate M1 extraction on Med-LongMem.")
    parser.add_argument("--data_path", type=str, default="data/raw/med_longmem")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()
    ensure_llm_api_key()

    task = MedLongMemTask(data_path=args.data_path, max_samples=args.max_samples)
    samples = task.get_samples()
    defaults = recommended_pipeline_options("med_longmem")
    extractor = MedicalEventExtractor()

    metrics_list = []
    per_sample = []

    print(f"\n{'='*70}")
    print(f"  M1 Extraction Evaluation on Med-LongMem")
    print(f"  Samples: {len(samples)}")
    print(f"{'='*70}\n")

    for idx, sample in enumerate(samples, start=1):
        print(f"[{idx}/{len(samples)}] {sample.sample_id}...", end=" ", flush=True)
        dialogue = _sample_to_dialogue(sample)
        t0 = time.time()
        predicted = extractor.extract(dialogue, sample.sample_id)
        latency = time.time() - t0

        metrics = compute_extraction_f1(
            predicted,
            sample.metadata.get("raw_events", []),
            dialogue_date=sample.question_date,
            missing_time_scope=defaults["missing_time_scope"],
        )
        metrics_list.append(metrics)
        per_sample.append(
            {
                "sample_id": sample.sample_id,
                "event_f1": metrics.event_f1,
                "relaxed_event_f1": metrics.relaxed_event_f1,
                "field_f1": metrics.field_f1,
                "n_predicted": metrics.n_predicted,
                "n_gt": metrics.n_gt,
                "latency": round(latency, 2),
            }
        )
        print(
            f"F1={metrics.event_f1:.3f} "
            f"Relaxed={metrics.relaxed_event_f1:.3f} "
            f"({latency:.1f}s)"
        )

    agg = aggregate_extraction_f1(metrics_list)
    summary = {
        "system": "M1_Extractor",
        "dataset": "med_longmem",
        "n_samples": agg.n_samples,
        "event_f1": agg.mean_event_f1,
        "event_precision": agg.mean_event_precision,
        "event_recall": agg.mean_event_recall,
        "relaxed_event_f1": agg.mean_relaxed_event_f1,
        "relaxed_event_precision": agg.mean_relaxed_event_precision,
        "relaxed_event_recall": agg.mean_relaxed_event_recall,
        "field_f1": agg.mean_field_f1,
        "per_sample": per_sample,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nMean strict Event-F1 : {agg.mean_event_f1:.4f}")
    print(f"Mean relaxed Event-F1: {agg.mean_relaxed_event_f1:.4f}")
    print(f"Field F1             : {agg.mean_field_f1}")
    print(f"Saved                : {output_path}")


if __name__ == "__main__":
    main()
