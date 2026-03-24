"""
baselines/long_context_llm.py
==============================
Long-Context LLM Baseline：直接用大上下文 LLM 处理全部对话并输出结构化记忆。

这是最重要的 baseline：如果 128K 上下文的 LLM 直接处理全部对话就能做到
和 UCM 差不多的结果，那 UCM 的整个 pipeline 就没有存在价值。

用法：
    PYTHONPATH=. python baselines/long_context_llm.py \\
        --data_path data/raw/med_longmem \\
        --output_path results/baselines/long_context_llm.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.med_longmem_task import MedLongMemTask
from benchmarks.base_task import UnifiedSample
from src.uniq_cluster_memory.schema import CanonicalMemory, ConflictRecord
from src.uniq_cluster_memory.utils.llm_client import get_llm_client, LLM_MODEL
from evaluation.uniqueness_eval import compute_unique_f1
from evaluation.conflict_eval import compute_conflict_f1
from evaluation.temporal_eval import compute_temporal_metrics


# ── 系统提示：要求 LLM 直接输出结构化记忆 ──────────────────────────────────

SYSTEM_PROMPT = """You are a medical record analyst. Given a multi-turn patient-doctor dialogue,
extract ALL unique medical facts into structured canonical memory records.

For each medical fact, output a JSON object with these fields:
- "attribute": one of ["blood_glucose", "blood_pressure_sys", "blood_pressure_dia",
  "heart_rate", "body_temperature", "hemoglobin", "primary_diagnosis", "medication", "symptom"]
- "value": the actual value (e.g., "7.2", "Metformin 500mg bid", "dizziness")
- "unit": measurement unit if applicable (e.g., "mmol/L", "mmHg", "bpm", "°C", "g/L"), empty string otherwise
- "time_scope": when this fact applies. Use ISO date "YYYY-MM-DD" if specific, or "global" if unspecified.
  For medications, always use "global" (latest prescription).
- "update_policy": "unique" for measurements/diagnosis, "latest" for medications, "append" for symptoms
- "conflict_flag": true if you detect contradictory values for the same attribute at the same time
- "conflict_history": if conflict_flag is true, list objects with {"old_value", "new_value", "conflict_type": "value_change"}

Rules:
1. UNIQUENESS: For measurements ("unique" policy), keep ONE record per (attribute, time_scope).
   If multiple values exist for the same attribute and date, keep the latest mentioned and flag conflict.
2. MEDICATIONS ("latest" policy): Keep only the MOST RECENT prescription. time_scope = "global".
   If the medication changed during the dialogue, flag conflict with old/new values.
3. SYMPTOMS ("append" policy): Keep ALL distinct symptoms mentioned. No deduplication needed.
4. Do NOT include speculative or questioned diagnoses (e.g., "could it be diabetes?").
5. Resolve coreferences (e.g., "that reading" → the actual value it refers to).

Output ONLY a JSON array of memory objects. No markdown, no explanation."""


def run_long_context_baseline(
    dialogue_turns: List[dict],
    dialogue_id: str,
    dialogue_date: Optional[str] = None,
) -> List[CanonicalMemory]:
    """用 long-context LLM 直接处理全部对话。"""
    client = get_llm_client()

    # 构建完整对话文本
    dialogue_text = ""
    for turn in dialogue_turns:
        speaker = turn.get("speaker", turn.get("role", "unknown"))
        text = turn.get("text", turn.get("content", ""))
        turn_id = turn.get("turn_id", "")
        dialogue_text += f"[Turn {turn_id}] {speaker}: {text}\n"

    date_hint = f"\nDialogue date: {dialogue_date}" if dialogue_date else ""

    user_prompt = (
        f"Patient ID: {dialogue_id}{date_hint}\n\n"
        f"DIALOGUE:\n{dialogue_text}\n\n"
        f"Extract all unique medical facts as structured memory records."
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=4000,
            )
            raw_text = (response.choices[0].message.content or "").strip()

            # 清理 markdown
            if raw_text.startswith("```"):
                parts = raw_text.split("```")
                raw_text = parts[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            records = json.loads(raw_text)
            if not isinstance(records, list):
                records = [records]

            memories = []
            for rec in records:
                conflict_history = []
                for ch in rec.get("conflict_history", []):
                    if isinstance(ch, dict):
                        conflict_history.append(ConflictRecord(
                            old_value=ch.get("old_value", ""),
                            new_value=ch.get("new_value", ""),
                            old_provenance=[],
                            new_provenance=[],
                            conflict_type=ch.get("conflict_type", "value_change"),
                            detected_at="",
                        ))
                memories.append(CanonicalMemory(
                    patient_id=dialogue_id,
                    attribute=str(rec.get("attribute", "")),
                    value=str(rec.get("value", "")),
                    unit=str(rec.get("unit", "")),
                    time_scope=str(rec.get("time_scope", "global")),
                    confidence=1.0,
                    provenance=[],
                    conflict_flag=bool(rec.get("conflict_flag", False)),
                    conflict_history=conflict_history,
                    update_policy=str(rec.get("update_policy", "unique")),
                ))
            return memories

        except Exception as e:
            print(f"    [Attempt {attempt+1}/3] Failed: {e}")
            if attempt < 2:
                time.sleep(3)

    return []


def main():
    parser = argparse.ArgumentParser(description="Long-Context LLM Baseline")
    parser.add_argument("--data_path", default="data/raw/med_longmem")
    parser.add_argument("--output_path", default="results/baselines/long_context_llm.json")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    task = MedLongMemTask(data_path=args.data_path, max_samples=args.max_samples)
    samples = task.get_samples()
    print(f"\nLoaded {len(samples)} samples\n")

    results = []
    for i, sample in enumerate(samples, 1):
        print(f"[{i:03d}/{len(samples)}] {sample.sample_id}...", end=" ", flush=True)
        dialogue = [
            {
                "turn_id": j,
                "speaker": "patient" if turn.role == "user" else "doctor",
                "text": turn.content,
            }
            for j, turn in enumerate(sample.dialog_history)
        ]

        t0 = time.time()
        predicted = run_long_context_baseline(
            dialogue, sample.sample_id, sample.question_date
        )
        latency = time.time() - t0

        gt = sample.metadata.get("canonical_gt", [])
        u = compute_unique_f1(predicted, gt)
        c = compute_conflict_f1(predicted, gt)
        t = compute_temporal_metrics(predicted, gt)

        result = {
            "sample_id": sample.sample_id,
            "n_gt": len(gt),
            "n_predicted": len(predicted),
            "unique_f1_strict": round(u.f1, 4),
            "unique_f1_relaxed": round(u.relaxed_f1, 4),
            "attribute_coverage": round(u.attribute_coverage, 4),
            "conflict_f1": round(c.f1, 4),
            "temporal_exact_f1": round(t.temporal_f1, 4),
            "interval_iou": round(t.mean_interval_iou, 4),
            "latency": round(latency, 2),
        }
        results.append(result)
        print(
            f"U-F1(S)={result['unique_f1_strict']:.3f}  "
            f"U-F1(R)={result['unique_f1_relaxed']:.3f}  "
            f"C-F1={result['conflict_f1']:.3f}  "
            f"({latency:.1f}s)"
        )

    # Aggregate
    n = len(results)
    if n > 0:
        summary = {
            "baseline": "long_context_llm",
            "model": LLM_MODEL,
            "n_samples": n,
            "unique_f1_strict": round(sum(r["unique_f1_strict"] for r in results) / n, 4),
            "unique_f1_relaxed": round(sum(r["unique_f1_relaxed"] for r in results) / n, 4),
            "attribute_coverage": round(sum(r["attribute_coverage"] for r in results) / n, 4),
            "conflict_f1": round(sum(r["conflict_f1"] for r in results) / n, 4),
            "temporal_exact_f1": round(sum(r["temporal_exact_f1"] for r in results) / n, 4),
            "interval_iou": round(sum(r["interval_iou"] for r in results) / n, 4),
            "avg_latency": round(sum(r["latency"] for r in results) / n, 2),
            "per_sample": results,
        }
    else:
        summary = {"baseline": "long_context_llm", "n_samples": 0}

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Long-Context LLM Baseline Results")
    print(f"{'='*60}")
    if n > 0:
        print(f"  U-F1(S)  : {summary['unique_f1_strict']:.4f}")
        print(f"  U-F1(R)  : {summary['unique_f1_relaxed']:.4f}")
        print(f"  AttrCov  : {summary['attribute_coverage']:.4f}")
        print(f"  C-F1     : {summary['conflict_f1']:.4f}")
        print(f"  T-F1     : {summary['temporal_exact_f1']:.4f}")
        print(f"  IoU      : {summary['interval_iou']:.4f}")
        print(f"  Latency  : {summary['avg_latency']:.1f}s/sample")
    print(f"\n  Saved: {args.output_path}")


if __name__ == "__main__":
    main()
