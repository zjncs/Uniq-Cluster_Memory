"""
build_realworld_validation.py
=============================
从官方 MedDialog 多轮中文数据中筛选真实长对话，构建真实世界验证包：

1. 筛选轮次足够长的真实对话
2. 运行 UCM pipeline 生成 Silver GT
3. 导出人工抽查包（10-20%）
4. 导出 Case Study 结果

注意：
- Silver GT 适合作为 bootstrap / 质检 / case study 依据
- 不应用同一版 UCM 对自身生成的 Silver GT 做 headline 定量自证
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.uniq_cluster_memory.pipeline import UniqueClusterMemoryPipeline
from src.uniq_cluster_memory.schema import CanonicalMemory


DEFAULT_DATA_PATH = "data/raw/meddialog_official/processed_zh_test.json"
DEFAULT_OUTPUT_DIR = "results/real_world_validation/meddialog_official_zh_test_long_r50_s42"
SELECTED_FILENAME = "selected_dialogues.jsonl"
SILVER_FILENAME = "silver_gt.jsonl"
ERRORS_FILENAME = "errors.jsonl"

PATIENT_PREFIXES = (
    "病人",
    "患者",
    "咨询者",
    "家长",
    "患儿家长",
    "孕妇",
    "本人",
)
DOCTOR_PREFIXES = (
    "医生",
    "医师",
    "大夫",
    "主任",
    "专家",
)


def load_processed_dialogues(path: Path) -> list[list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {path}, got {type(data).__name__}")
    dialogues: list[list[str]] = []
    for item in data:
        if isinstance(item, list):
            utterances = [str(x).strip() for x in item if str(x).strip()]
            if utterances:
                dialogues.append(utterances)
    return dialogues


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def append_jsonl(path: Path, payload: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _infer_role_from_prefix(prefix: str, fallback_role: str) -> str:
    normalized = prefix.strip()
    if any(token in normalized for token in DOCTOR_PREFIXES):
        return "doctor"
    if any(token in normalized for token in PATIENT_PREFIXES):
        return "patient"
    return fallback_role


def parse_utterance(utterance: str, fallback_role: str) -> tuple[str, str]:
    raw = (utterance or "").strip()
    if not raw:
        return fallback_role, ""

    for sep in ("：", ":"):
        if sep in raw:
            prefix, rest = raw.split(sep, 1)
            role = _infer_role_from_prefix(prefix, fallback_role)
            if role != fallback_role or prefix.strip():
                return role, rest.strip() or raw
    return fallback_role, raw


def utterances_to_dialogue(utterances: Sequence[str]) -> list[dict]:
    dialogue: list[dict] = []
    for idx, utterance in enumerate(utterances):
        fallback_role = "patient" if idx % 2 == 0 else "doctor"
        role, text = parse_utterance(utterance, fallback_role)
        dialogue.append(
            {
                "turn_id": idx,
                "speaker": role,
                "text": text,
            }
        )
    return dialogue


def select_long_dialogues(
    dialogues: Sequence[Sequence[str]],
    min_turns: int,
    n_samples: int,
    seed: int,
) -> list[dict]:
    candidates = [
        {"raw_index": idx, "utterances": list(utterances), "n_turns": len(utterances)}
        for idx, utterances in enumerate(dialogues)
        if len(utterances) >= min_turns
    ]
    if not candidates:
        raise RuntimeError(f"No dialogues with >= {min_turns} turns were found.")

    rng = random.Random(seed)
    k = min(n_samples, len(candidates))
    chosen = rng.sample(candidates, k=k)
    chosen.sort(key=lambda item: item["raw_index"])
    for sample_idx, item in enumerate(chosen):
        item["sample_id"] = f"meddialog_real_{item['raw_index']:07d}"
        item["source_split"] = "processed.zh.test"
        item["sample_order"] = sample_idx
    return chosen


def format_memory_bundle(memories: Sequence[CanonicalMemory]) -> str:
    if not memories:
        return "(empty)"
    rows: list[str] = []
    sorted_mems = sorted(memories, key=lambda m: (m.attribute, m.time_scope, m.value.lower()))
    for idx, mem in enumerate(sorted_mems, start=1):
        unit = f" {mem.unit}" if mem.unit else ""
        conflict = "冲突" if mem.conflict_flag else "无冲突"
        rows.append(f"{idx}. [{mem.attribute} | {mem.time_scope}] {mem.value}{unit} ({conflict})")
    return "\n".join(rows)


def format_dialogue_text(dialogue: Sequence[dict]) -> str:
    lines: list[str] = []
    for turn in dialogue:
        speaker = "患者" if turn["speaker"] == "patient" else "医生"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


def build_audit_subset(records: Sequence[dict], audit_ratio: float, seed: int) -> list[dict]:
    if not records:
        return []
    audit_n = max(1, round(len(records) * audit_ratio))
    rng = random.Random(seed + 1000)
    selected = rng.sample(list(records), k=min(audit_n, len(records)))
    selected.sort(key=lambda item: item["sample_id"])
    return selected


def write_audit_guide(path: Path) -> None:
    text = """# Real-World Validation Audit Guide

本包用于抽查 Silver GT 的可信度，而不是用同一版 UCM 对自身做定量自证。

建议抽查维度：
- gt_accept: Silver GT 是否整体可接受（1/0）
- factual_accuracy: 事实是否与原对话一致（1-5）
- completeness: 关键医疗信息是否漏提（1-5）
- conflict_quality: 冲突检测是否合理（1-5）
- correction_needed: 是否需要人工修正后进入小规模 Gold 集（1/0）

重点关注：
- 数值型指标是否被错误归一化
- 药物更新是否被误判为冲突
- 同一医生列举多个建议时，是否被错误压成 latest
- 时间表达缺失时，scope 是否过度具体
"""
    path.write_text(text, encoding="utf-8")


def write_case_studies(path: Path, records: Sequence[dict], case_count: int) -> None:
    ranked = sorted(
        records,
        key=lambda item: (
            item["n_conflicts"],
            item["n_predicted"],
            item["n_turns"],
        ),
        reverse=True,
    )
    chosen = ranked[: min(case_count, len(ranked))]
    lines = [
        "# MedDialog Real-World Case Studies",
        "",
        "以下 Case Study 用于论文的真实世界定性分析，不作为同版 UCM 的自证测试集。",
        "",
    ]
    for idx, item in enumerate(chosen, start=1):
        lines.extend(
            [
                f"## Case {idx}: {item['sample_id']}",
                "",
                f"- Raw index: `{item['raw_index']}`",
                f"- Turns: `{item['n_turns']}`",
                f"- Predicted memories: `{item['n_predicted']}`",
                f"- Conflict memories: `{item['n_conflicts']}`",
                "",
                "### Dialogue",
                "",
                "```text",
                format_dialogue_text(item["dialogue"]),
                "```",
                "",
                "### Silver GT / Predicted Memory Bundle",
                "",
                "```text",
                format_memory_bundle(item["memories"]),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_records(records: Sequence[dict], config: dict) -> dict:
    n_turns = [item["n_turns"] for item in records]
    n_pred = [item["n_predicted"] for item in records]
    n_conflicts = [item["n_conflicts"] for item in records]
    return {
        "methodological_note": (
            "Silver GT is intended for bootstrap labeling, audit, and case studies. "
            "Do not use the same UCM configuration to claim headline quantitative superiority on this set."
        ),
        "config": config,
        "n_selected": len(records),
        "mean_turns": round(statistics.mean(n_turns), 2) if n_turns else 0.0,
        "median_turns": statistics.median(n_turns) if n_turns else 0,
        "mean_predicted_memories": round(statistics.mean(n_pred), 2) if n_pred else 0.0,
        "mean_conflicts": round(statistics.mean(n_conflicts), 2) if n_conflicts else 0.0,
        "n_with_conflicts": sum(1 for x in n_conflicts if x > 0),
        "total_conflicts": sum(n_conflicts),
    }


def silver_payload_to_record(payload: dict) -> dict:
    memories = [CanonicalMemory.from_dict(item) for item in payload.get("silver_memories", [])]
    return {
        "sample_id": payload["sample_id"],
        "raw_index": payload["raw_index"],
        "source_split": payload["source_split"],
        "n_turns": payload["n_turns"],
        "utterances": payload.get("utterances", []),
        "dialogue": payload["dialogue"],
        "memories": memories,
        "n_predicted": payload["n_predicted"],
        "n_conflicts": payload["n_conflicts"],
        "latency_sec": payload["latency_sec"],
    }


def record_to_silver_payload(item: dict) -> dict:
    return {
        "sample_id": item["sample_id"],
        "raw_index": item["raw_index"],
        "source_split": item["source_split"],
        "n_turns": item["n_turns"],
        "utterances": item["utterances"],
        "n_predicted": item["n_predicted"],
        "n_conflicts": item["n_conflicts"],
        "latency_sec": item["latency_sec"],
        "dialogue": item["dialogue"],
        "silver_memories": [mem.to_dict() for mem in item["memories"]],
    }


def load_existing_silver_records(path: Path) -> dict[str, dict]:
    rows = load_jsonl(path)
    return {row["sample_id"]: silver_payload_to_record(row) for row in rows}


def load_existing_error_records(path: Path) -> list[dict]:
    return load_jsonl(path)


def unresolved_error_records(records: Sequence[dict], completed_ids: set[str]) -> list[dict]:
    latest_by_id: dict[str, dict] = {}
    for item in records:
        sample_id = item.get("sample_id")
        if sample_id and sample_id not in completed_ids:
            latest_by_id[sample_id] = item
    return [latest_by_id[key] for key in sorted(latest_by_id)]


def validate_resume_selection(path: Path, selected: Sequence[dict]) -> None:
    existing = load_jsonl(path)
    if not existing:
        return
    existing_ids = [item["sample_id"] for item in existing]
    current_ids = [item["sample_id"] for item in selected]
    if existing_ids != current_ids:
        raise RuntimeError(
            "Resume target does not match the current sampled dialogues. "
            "Use a fresh output_dir or rerun without --resume."
        )


def iter_batches(records: Sequence[dict], batch_size: int) -> Iterable[list[dict]]:
    batch_size = max(1, batch_size)
    for offset in range(0, len(records), batch_size):
        yield list(records[offset: offset + batch_size])


def run_pipeline_on_records(
    records: list[dict],
    *,
    output_dir: Path,
    pipeline_kwargs: dict,
    resume: bool,
    batch_size: int,
    sleep_seconds: float,
    fail_fast: bool,
) -> tuple[list[dict], list[dict]]:
    silver_path = output_dir / SILVER_FILENAME
    error_path = output_dir / ERRORS_FILENAME
    existing = load_existing_silver_records(silver_path) if resume else {}
    pending = [record for record in records if record["sample_id"] not in existing]

    if existing:
        print(
            f"Resuming from existing Silver GT: completed={len(existing)}  pending={len(pending)}"
        )
    if not pending:
        print("All selected dialogues already have Silver GT. Skipping pipeline execution.")
        return [existing[item["sample_id"]] for item in records if item["sample_id"] in existing], load_existing_error_records(error_path)

    total_batches = max(1, (len(pending) + max(1, batch_size) - 1) // max(1, batch_size))
    for batch_idx, batch in enumerate(iter_batches(pending, batch_size), start=1):
        print(f"\nBatch {batch_idx}/{total_batches}  size={len(batch)}")
        pipeline = UniqueClusterMemoryPipeline(**pipeline_kwargs)
        try:
            for record in batch:
                record_idx = record["sample_order"] + 1
                dialogue = utterances_to_dialogue(record["utterances"])
                t0 = time.time()
                try:
                    memories = pipeline.build_memory(
                        dialogue=dialogue,
                        dialogue_id=record["sample_id"],
                        dialogue_date=None,
                    )
                except Exception as exc:
                    latency = time.time() - t0
                    error_payload = {
                        "sample_id": record["sample_id"],
                        "raw_index": record["raw_index"],
                        "source_split": record["source_split"],
                        "n_turns": record["n_turns"],
                        "utterances": record["utterances"],
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "latency_sec": round(latency, 2),
                    }
                    append_jsonl(error_path, error_payload)
                    print(
                        f"[{record_idx:03d}/{len(records)}] {record['sample_id']}  "
                        f"turns={record['n_turns']}  ERROR={type(exc).__name__}  latency={latency:.1f}s"
                    )
                    if fail_fast:
                        raise
                    continue

                latency = time.time() - t0
                n_conflicts = sum(1 for mem in memories if mem.conflict_flag)
                enriched = {
                    **record,
                    "dialogue": dialogue,
                    "memories": memories,
                    "n_predicted": len(memories),
                    "n_conflicts": n_conflicts,
                    "latency_sec": round(latency, 2),
                }
                append_jsonl(silver_path, record_to_silver_payload(enriched))
                existing[record["sample_id"]] = enriched
                print(
                    f"[{record_idx:03d}/{len(records)}] {record['sample_id']}  "
                    f"turns={record['n_turns']}  pred={len(memories)}  "
                    f"conflicts={n_conflicts}  latency={latency:.1f}s"
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
        finally:
            pipeline.close()

    enriched_records = [existing[item["sample_id"]] for item in records if item["sample_id"] in existing]
    return enriched_records, load_existing_error_records(error_path)


def write_selected_dialogues(path: Path, records: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            payload = {
                "sample_id": item["sample_id"],
                "raw_index": item["raw_index"],
                "source_split": item["source_split"],
                "n_turns": item["n_turns"],
                "utterances": item["utterances"],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_silver_gt(path: Path, records: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(record_to_silver_payload(item), ensure_ascii=False) + "\n")


def write_audit_csv(path: Path, audit_records: Sequence[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "raw_index",
                "n_turns",
                "n_predicted",
                "n_conflicts",
                "latency_sec",
                "dialogue_text",
                "silver_memory_bundle",
                "gt_accept",
                "factual_accuracy",
                "completeness",
                "conflict_quality",
                "correction_needed",
                "annotator",
                "notes",
            ],
        )
        writer.writeheader()
        for item in audit_records:
            writer.writerow(
                {
                    "sample_id": item["sample_id"],
                    "raw_index": item["raw_index"],
                    "n_turns": item["n_turns"],
                    "n_predicted": item["n_predicted"],
                    "n_conflicts": item["n_conflicts"],
                    "latency_sec": item["latency_sec"],
                    "dialogue_text": format_dialogue_text(item["dialogue"]),
                    "silver_memory_bundle": format_memory_bundle(item["memories"]),
                    "gt_accept": "",
                    "factual_accuracy": "",
                    "completeness": "",
                    "conflict_quality": "",
                    "correction_needed": "",
                    "annotator": "",
                    "notes": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real-world validation package from official MedDialog multi-turn data.")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH, help="Path to official processed.zh MedDialog JSON.")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of long dialogues to sample.")
    parser.add_argument("--min_turns", type=int, default=10, help="Minimum number of utterances to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dialogue selection.")
    parser.add_argument("--audit_ratio", type=float, default=0.2, help="Fraction of sampled dialogues to manually audit.")
    parser.add_argument("--case_count", type=int, default=5, help="Number of case studies to export.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--w_struct", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--use_embedding", action="store_true", help="Enable embedding in M5.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing silver_gt.jsonl in output_dir.")
    parser.add_argument("--batch_size", type=int, default=5, help="How many dialogues to process before recreating the pipeline.")
    parser.add_argument("--sleep_seconds", type=float, default=1.0, help="Delay between dialogues to avoid overloading the API.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop immediately on the first pipeline error.")
    parser.add_argument(
        "--max_symptoms_per_scope",
        type=int,
        default=-1,
        help="Symptom cap per scope. -1 means unlimited for real-world validation.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / SELECTED_FILENAME
    silver_path = output_dir / SILVER_FILENAME
    summary_path = output_dir / "summary.json"
    audit_csv_path = output_dir / "audit_subset.csv"
    audit_guide_path = output_dir / "audit_guide.md"
    case_studies_path = output_dir / "case_studies.md"
    error_path = output_dir / ERRORS_FILENAME

    print("=" * 72)
    print("  MedDialog Real-World Validation Builder")
    print(f"  Data      : {data_path}")
    print(f"  Min turns : {args.min_turns}")
    print(f"  Samples   : {args.n_samples}")
    print(f"  Audit %   : {args.audit_ratio:.0%}")
    print(f"  Output    : {output_dir}")
    print(f"  Embedding : {'enabled' if args.use_embedding else 'disabled'}")
    print("=" * 72)

    dialogues = load_processed_dialogues(data_path)
    selected = select_long_dialogues(
        dialogues=dialogues,
        min_turns=args.min_turns,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    print(f"\nLoaded dialogues: {len(dialogues)}")
    print(f"Selected long dialogues: {len(selected)}")
    if args.resume:
        validate_resume_selection(selected_path, selected)
    else:
        for path in (silver_path, error_path):
            if path.exists():
                path.unlink()
    write_selected_dialogues(selected_path, selected)

    symptom_cap = None if args.max_symptoms_per_scope < 0 else args.max_symptoms_per_scope
    enriched, error_records = run_pipeline_on_records(
        selected,
        output_dir=output_dir,
        pipeline_kwargs={
            "w_struct": args.w_struct,
            "top_k": args.top_k,
            "use_embedding": args.use_embedding,
            "missing_time_scope": "global",
            "max_symptoms_per_scope": symptom_cap,
        },
        resume=args.resume,
        batch_size=args.batch_size,
        sleep_seconds=max(0.0, args.sleep_seconds),
        fail_fast=args.fail_fast,
    )

    audit_records = build_audit_subset(enriched, audit_ratio=args.audit_ratio, seed=args.seed)

    write_silver_gt(silver_path, enriched)
    write_audit_csv(audit_csv_path, audit_records)
    write_audit_guide(audit_guide_path)
    write_case_studies(case_studies_path, enriched, case_count=args.case_count)

    summary = summarize_records(
        enriched,
        config={
            "data_path": str(data_path),
            "n_samples": len(enriched),
            "min_turns": args.min_turns,
            "seed": args.seed,
            "audit_ratio": args.audit_ratio,
            "case_count": args.case_count,
            "use_embedding": args.use_embedding,
            "w_struct": args.w_struct,
            "top_k": args.top_k,
            "resume": args.resume,
            "batch_size": args.batch_size,
            "sleep_seconds": max(0.0, args.sleep_seconds),
            "max_symptoms_per_scope": symptom_cap,
        },
    )
    remaining_errors = unresolved_error_records(
        error_records,
        completed_ids={item["sample_id"] for item in enriched},
    )
    summary["audit_sample_ids"] = [item["sample_id"] for item in audit_records]
    summary["n_completed"] = len(enriched)
    summary["n_failed"] = len(remaining_errors)
    summary["failed_sample_ids"] = [item["sample_id"] for item in remaining_errors]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print(f"  Done: {len(enriched)} selected dialogues")
    print(f"  Mean turns            : {summary['mean_turns']}")
    print(f"  Mean predicted memories: {summary['mean_predicted_memories']}")
    print(f"  Mean conflicts        : {summary['mean_conflicts']}")
    print(f"  Failed dialogues      : {summary['n_failed']}")
    print("=" * 72)
    print(f"  Selected dialogues : {selected_path}")
    print(f"  Silver GT          : {silver_path}")
    print(f"  Errors             : {error_path}")
    print(f"  Audit CSV          : {audit_csv_path}")
    print(f"  Case studies       : {case_studies_path}")
    print(f"  Summary            : {summary_path}")


if __name__ == "__main__":
    main()
