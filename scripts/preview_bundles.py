"""
preview_bundles.py
==================
快速预览 M2.5 信息团（Entity/Event Bundles）。

示例：
    source .venv/bin/activate
    python scripts/preview_bundles.py --dataset med_longmem --max_samples 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.med_longmem_task import MedLongMemTask
from benchmarks.meddialog_task import MedDialogTask
from src.uniq_cluster_memory.pipeline import UniqueClusterMemoryPipeline


def _load_task(dataset: str, max_samples: int):
    if dataset == "med_longmem":
        return MedLongMemTask(
            data_path="data/raw/med_longmem",
            max_samples=max_samples,
        )
    if dataset == "meddialog":
        return MedDialogTask(
            data_path="data/raw/meddialog/meddialog_zh_sample50.json",
            max_samples=max_samples,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["med_longmem", "meddialog"], default="med_longmem")
    parser.add_argument("--max_samples", type=int, default=1)
    args = parser.parse_args()

    task = _load_task(args.dataset, args.max_samples)
    samples = task.get_samples()
    if not samples:
        print("No samples found.")
        return

    pipe = UniqueClusterMemoryPipeline(use_embedding=False)
    for sample in samples:
        dialogue = [
            {
                "turn_id": i,
                "speaker": "patient" if t.role == "user" else "doctor",
                "text": t.content,
            }
            for i, t in enumerate(sample.dialog_history)
        ]
        _, graph = pipe.build_memory_with_bundles(
            dialogue=dialogue,
            dialogue_id=sample.sample_id,
            dialogue_date=sample.question_date,
        )
        payload = graph.to_dict()
        print(f"\n=== {sample.sample_id} ===")
        print(
            f"event_bundles={len(payload['event_bundles'])}, "
            f"entity_bundles={len(payload['entity_bundles'])}, links={len(payload['links'])}"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2)[:3000])

    pipe.close()


if __name__ == "__main__":
    main()

