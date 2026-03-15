"""
run_manual_eval_meddialog.py
============================
真实世界人工评测流水线（MedDialog-CN）。

功能：
1) 从公开中文医疗对话数据中随机抽样（默认 100 条）。
2) 运行 Uniq-Cluster Memory pipeline，生成记忆抽取结果。
3) 导出人工标注模板（CSV）用于 Accuracy/Completeness/Noise 评分。

示例：
    source .venv/bin/activate
    python scripts/run_manual_eval_meddialog.py \
        --data_path data/raw/meddialog/meddialog_zh_0000.parquet \
        --n_samples 100 \
        --seed 42 \
        --output_dir results/manual_eval/meddialog_cn_r100_s42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.base_task import UnifiedSample
from benchmarks.meddialog_task import MedDialogTask
from src.uniq_cluster_memory.defaults import recommended_pipeline_options
from src.uniq_cluster_memory.pipeline import UniqueClusterMemoryPipeline
from src.uniq_cluster_memory.schema import CanonicalMemory
from src.uniq_cluster_memory.utils.llm_client import ensure_llm_api_key

try:
    import pyarrow.parquet as pq

    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


DEFAULT_REFERENCE_DATE = "2024-01-01"


def _iter_raw_records(data_path: Path) -> Iterable[Tuple[int, Dict]]:
    suffix = data_path.suffix.lower()
    if suffix == ".parquet":
        if not PARQUET_AVAILABLE:
            raise ImportError("pyarrow is required for parquet input. Please install pyarrow.")
        pf = pq.ParquetFile(str(data_path))
        columns = ["instruction", "input", "output", "history"]
        row_index = 0
        for batch in pf.iter_batches(batch_size=2048, columns=columns):
            payload = batch.to_pydict()
            if not payload:
                continue
            n = len(next(iter(payload.values())))
            for i in range(n):
                item = {k: payload[k][i] for k in payload}
                yield row_index, item
                row_index += 1
        return
    if suffix == ".json":
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for row_index, item in enumerate(data):
            yield row_index, item
        return
    raise ValueError(f"Unsupported file format: {suffix}. Expected .parquet or .json")


def _reservoir_sample_records(
    data_path: Path,
    n_samples: int,
    seed: int,
) -> List[Tuple[int, Dict]]:
    rng = random.Random(seed)
    reservoir: List[Tuple[int, Dict]] = []
    seen = 0
    for seen, record in enumerate(_iter_raw_records(data_path), start=1):
        if len(reservoir) < n_samples:
            reservoir.append(record)
            continue
        j = rng.randint(1, seen)
        if j <= n_samples:
            reservoir[j - 1] = record
    if not reservoir:
        return []
    reservoir.sort(key=lambda x: x[0])
    return reservoir


def _to_unified_sample(raw_index: int, item: Dict) -> Optional[UnifiedSample]:
    sample = MedDialogTask._convert_to_unified(item, raw_index)
    if sample is None:
        return None
    sample.sample_id = f"meddialog_rw_{raw_index:08d}"
    sample.metadata["raw_index"] = raw_index
    return sample


def _dialogue_for_pipeline(sample: UnifiedSample) -> List[Dict]:
    turns: List[Dict] = []
    for i, t in enumerate(sample.dialog_history):
        speaker = "patient" if t.role == "user" else "doctor"
        turns.append({"turn_id": i, "speaker": speaker, "text": t.content})
    return turns


def _format_dialogue_text(dialogue: List[Dict]) -> str:
    lines = []
    for t in dialogue:
        speaker = "患者" if t["speaker"] == "patient" else "医生"
        lines.append(f"{speaker}: {t['text']}")
    return "\n".join(lines)


def _format_memory_bundle(memories: List[CanonicalMemory]) -> str:
    rows = []
    sorted_mems = sorted(
        memories,
        key=lambda m: (m.attribute, m.time_scope, m.value.lower()),
    )
    for i, m in enumerate(sorted_mems, start=1):
        unit = f" {m.unit}" if m.unit else ""
        conflict = "冲突" if m.conflict_flag else "无冲突"
        rows.append(f"{i}. [{m.attribute} | {m.time_scope}] {m.value}{unit} ({conflict})")
    return "\n".join(rows)


def _write_annotation_guide(path: Path) -> None:
    text = """# MedDialog-CN 人工评测说明

请针对每条样本，对模型抽取的“记忆信息团”进行人工评估。

评分维度（1-5分）：
- score_accuracy: 抽取内容是否准确（属性/值/冲突是否符合原对话）
- score_completeness: 关键信息覆盖是否完整（是否漏掉核心医疗信息）
- score_noise: 去噪效果是否好（口语噪声、泛化词、错误项是否少）
- score_clinical_safety: 临床安全性（是否出现可能误导临床判断的严重错误）

推荐判定：
- pass_flag = 1：总体可用，错误不影响主要结论
- pass_flag = 0：总体不可用，存在明显漏提/误提/误冲突

备注：
- annotator: 标注人姓名或编号
- notes: 记录典型错误（如时间错位、药物频次归一化错误、冲突误报）
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedDialog-CN real-world manual evaluation package generation.")
    parser.add_argument(
        "--data_path",
        default="data/raw/meddialog/meddialog_zh_0000.parquet",
        help="Path to MedDialog-CN parquet/json.",
    )
    parser.add_argument("--n_samples", type=int, default=100, help="Number of random samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--output_dir",
        default="results/manual_eval/meddialog_cn_r100_s42",
        help="Output directory for manual evaluation package.",
    )
    parser.add_argument(
        "--reference_date",
        default=DEFAULT_REFERENCE_DATE,
        help="Fixed ISO reference date for grounding relative times when MedDialog lacks dialogue dates.",
    )
    parser.add_argument("--w_struct", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--use_embedding", action="store_true", help="Enable embedding in retriever.")
    parser.add_argument(
        "--max_symptoms_per_scope",
        type=int,
        default=-1,
        help="Symptom cap per time scope. -1 => unlimited for meddialog.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_path = output_dir / "sampled_records.jsonl"
    prediction_path = output_dir / "predictions.jsonl"
    annotation_csv = output_dir / "annotation_template.csv"
    guide_md = output_dir / "annotation_guide.md"

    print("=" * 68)
    print("  MedDialog-CN Real-World Manual Evaluation")
    print(f"  Data     : {data_path}")
    print(f"  Samples  : {args.n_samples}")
    print(f"  Seed     : {args.seed}")
    print(f"  Ref date : {args.reference_date}")
    print(f"  Output   : {output_dir}")
    print(f"  Embedding: {'enabled' if args.use_embedding else 'disabled'}")
    print("=" * 68)

    ensure_llm_api_key()

    sampled_raw = _reservoir_sample_records(data_path, args.n_samples, args.seed)
    print(f"\nSampled raw records: {len(sampled_raw)}")

    samples: List[UnifiedSample] = []
    for raw_index, item in sampled_raw:
        s = _to_unified_sample(raw_index, item)
        if s is not None:
            samples.append(s)
    print(f"Valid unified samples: {len(samples)}")
    if not samples:
        raise RuntimeError("No valid samples after conversion.")

    with open(sampled_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(
                json.dumps(
                    {
                        "sample_id": s.sample_id,
                        "raw_index": s.metadata.get("raw_index"),
                        "question": s.question,
                        "dialog_history": [{"role": t.role, "content": t.content} for t in s.dialog_history],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    recommended = recommended_pipeline_options("meddialog")
    symptom_cap = recommended["max_symptoms_per_scope"]
    if args.max_symptoms_per_scope >= 0:
        symptom_cap = args.max_symptoms_per_scope
    pipeline = UniqueClusterMemoryPipeline(
        w_struct=args.w_struct,
        top_k=args.top_k,
        use_embedding=args.use_embedding,
        missing_time_scope=recommended["missing_time_scope"],
        max_symptoms_per_scope=symptom_cap,
    )

    total_latency = 0.0
    n_pred_sum = 0
    with open(prediction_path, "w", encoding="utf-8") as fpred, open(
        annotation_csv, "w", encoding="utf-8", newline=""
    ) as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "sample_id",
                "raw_index",
                "n_turns",
                "n_predicted",
                "latency_sec",
                "question",
                "dialogue_text",
                "predicted_memory_bundle",
                "score_accuracy",
                "score_completeness",
                "score_noise",
                "score_clinical_safety",
                "pass_flag",
                "annotator",
                "notes",
            ],
        )
        writer.writeheader()

        for i, sample in enumerate(samples, start=1):
            dialogue = _dialogue_for_pipeline(sample)
            t0 = time.time()
            memories = pipeline.build_memory(
                dialogue=dialogue,
                dialogue_id=sample.sample_id,
                dialogue_date=sample.question_date or args.reference_date,
            )
            latency = time.time() - t0
            total_latency += latency
            n_pred_sum += len(memories)

            record = {
                "sample_id": sample.sample_id,
                "raw_index": sample.metadata.get("raw_index"),
                "question": sample.question,
                "n_turns": len(dialogue),
                "n_predicted": len(memories),
                "latency_sec": round(latency, 2),
                "dialogue": dialogue,
                "predicted_memories": [m.to_dict() for m in memories],
            }
            fpred.write(json.dumps(record, ensure_ascii=False) + "\n")

            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "raw_index": sample.metadata.get("raw_index"),
                    "n_turns": len(dialogue),
                    "n_predicted": len(memories),
                    "latency_sec": round(latency, 2),
                    "question": sample.question,
                    "dialogue_text": _format_dialogue_text(dialogue),
                    "predicted_memory_bundle": _format_memory_bundle(memories),
                    "score_accuracy": "",
                    "score_completeness": "",
                    "score_noise": "",
                    "score_clinical_safety": "",
                    "pass_flag": "",
                    "annotator": "",
                    "notes": "",
                }
            )

            print(
                f"[{i:03d}/{len(samples)}] {sample.sample_id}  "
                f"n_pred={len(memories)}  latency={latency:.1f}s"
            )

    pipeline.close()
    _write_annotation_guide(guide_md)

    avg_pred = n_pred_sum / len(samples)
    avg_latency = total_latency / len(samples)
    print("\n" + "=" * 68)
    print(f"  Done: {len(samples)} samples")
    print(f"  Avg predicted memories: {avg_pred:.2f}")
    print(f"  Avg latency          : {avg_latency:.2f}s/sample")
    print("=" * 68)
    print(f"  Sample records : {sampled_path}")
    print(f"  Predictions    : {prediction_path}")
    print(f"  Annotation CSV : {annotation_csv}")
    print(f"  Guide          : {guide_md}")


if __name__ == "__main__":
    main()
