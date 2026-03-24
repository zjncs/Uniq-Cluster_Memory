"""
experiments/run_meditod_eval.py
================================
MediTOD 外部 benchmark 评估。

使用 MediTOD raw_data（231 对话 + slot 标注）评估 UCM 在外部数据集上的
时序槽位（temporal slot）提取能力。

评估指标：
  - Slot F1：提取的 (symptom, onset, progression, severity) 与 GT 的 F1
  - Temporal Slot Accuracy：onset/duration 等时间属性的匹配率
  - Overall Memory Quality：记忆覆盖率

Usage:
    DASHSCOPE_API_KEY=xxx PYTHONPATH=. python experiments/run_meditod_eval.py \
        --max_samples 50 --output_dir results/meditod
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.uniq_cluster_memory.pipeline import UniqueClusterMemoryPipeline
from src.uniq_cluster_memory.schema import CanonicalMemory


# ── MediTOD slot → UCM attribute mapping ──────────────────────────────────────
SLOT_TO_ATTR = {
    "positive_symptom": "symptom",
    "negative_symptom": "symptom",
    "past_medical_history": "primary_diagnosis",
    "family_history": "primary_diagnosis",
    "current_medication": "medication",
    "allergy": "symptom",
}

TEMPORAL_KEYS = {"onset", "duration", "frequency", "progression", "severity"}


def load_meditod_raw(data_path: str) -> List[dict]:
    """Load MediTOD raw dialogues + annotations."""
    base = Path(data_path)
    dialogs = json.load(open(base / "raw_data" / "dialogs.json"))
    annotations = json.load(open(base / "raw_data" / "annotations.json"))

    samples = []
    for did, dlg_data in dialogs.items():
        annot = annotations.get(did, {})
        turns = []
        if isinstance(dlg_data, dict):
            raw_turns = dlg_data.get("dialogue", dlg_data.get("utterances", []))
        elif isinstance(dlg_data, list):
            raw_turns = dlg_data
        else:
            continue

        for i, turn in enumerate(raw_turns):
            if isinstance(turn, dict):
                speaker = turn.get("speaker", turn.get("role", "patient"))
                text = turn.get("text", turn.get("utterance", turn.get("content", "")))
            elif isinstance(turn, str):
                speaker = "patient" if i % 2 == 0 else "doctor"
                text = turn
            else:
                continue

            sp = speaker.strip().lower()
            if sp in ("doctor", "assistant", "agent"):
                sp = "doctor"
            else:
                sp = "patient"

            turns.append({
                "turn_id": i,
                "speaker": sp,
                "text": text,
            })

        # Extract GT slots from the final dialog_state in canonicalized data
        # (raw annotations are per-utterance NLU, not slot-level GT)
        gt_slots = extract_gt_from_turns(turns, did)

        samples.append({
            "dialogue_id": f"meditod_{did}",
            "dialogue": turns,
            "gt_slots": gt_slots,
            "raw_dialogue_id": did,
            "raw_annotation": annot,
        })

    return samples


def extract_gt_from_turns(turns: List[dict], dialogue_id: str) -> List[dict]:
    """Extract GT slots from MediTOD annotations.

    Strategy:
    1. Try canonicalized data (3 dialogues with full dialog_state)
    2. Fall back to raw annotations (231 dialogues with per-utterance NLU)
    """
    gt_slots = []

    # Strategy 1: canonicalized data
    canon_path = Path("data/raw/meditod/temp_clone/data/dialogs.json")
    if canon_path.exists():
        canon = json.load(open(canon_path))
        if dialogue_id in canon:
            uttrs = canon[dialogue_id].get("utterances", [])
            final_state = {}
            for u in uttrs:
                ds = u.get("dialog_state", {})
                if ds:
                    final_state = ds
            for slot_type, items in final_state.items():
                ucm_attr = SLOT_TO_ATTR.get(slot_type)
                if not ucm_attr:
                    continue
                if not isinstance(items, list):
                    items = [items]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    value = item.get("value", "")
                    if not value:
                        continue
                    temporal_info = {}
                    for tk in TEMPORAL_KEYS:
                        tv = item.get(tk, "")
                        if tv and str(tv).lower() not in ("none", "n/a", ""):
                            temporal_info[tk] = str(tv)
                    gt_slots.append({
                        "attribute": ucm_attr,
                        "value": value,
                        "temporal_info": temporal_info,
                        "slot_type": slot_type,
                    })
            if gt_slots:
                return gt_slots

    # Strategy 2: raw annotations (per-utterance NLU)
    raw_annot_path = Path("data/raw/meditod/temp_clone/raw_data/annotations.json")
    if raw_annot_path.exists():
        all_annot = json.load(open(raw_annot_path))
        annot_list = all_annot.get(dialogue_id, [])

        raw_slot_map = {
            "symptom": "symptom",
            "medication": "medication",
            "medical history": "primary_diagnosis",
            "family history": "primary_diagnosis",
        }

        seen = set()
        for uttr_annots in annot_list:
            if not isinstance(uttr_annots, list):
                continue
            for a in uttr_annots:
                if not isinstance(a, dict):
                    continue
                if a.get("intent") != "inform":
                    continue
                slot = a.get("slot", "")
                value = a.get("value", "")
                status = a.get("status", "")
                when = a.get("when", "")

                ucm_attr = raw_slot_map.get(slot)
                if not ucm_attr or not value:
                    continue
                # Only positive findings
                if status and status.lower() in ("no", "false", "negative"):
                    continue

                key = (ucm_attr, value.strip().lower())
                if key in seen:
                    continue
                seen.add(key)

                temporal_info = {}
                if when:
                    temporal_info["onset"] = when

                gt_slots.append({
                    "attribute": ucm_attr,
                    "value": value.strip(),
                    "temporal_info": temporal_info,
                    "slot_type": slot,
                })

    return gt_slots


def extract_gt_slots(annotation: dict, dialogue_id: str) -> List[dict]:
    """Extract structured GT slots from MediTOD annotation."""
    gt_slots = []

    if not isinstance(annotation, dict):
        return gt_slots

    for slot_type, ucm_attr in SLOT_TO_ATTR.items():
        items = annotation.get(slot_type, [])
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            value = item.get("value", "")
            if not value or value.lower() in ("none", "n/a", ""):
                continue

            temporal_info = {}
            for tk in TEMPORAL_KEYS:
                tv = item.get(tk, "")
                if tv and str(tv).lower() not in ("none", "n/a", ""):
                    temporal_info[tk] = str(tv)

            gt_slots.append({
                "attribute": ucm_attr,
                "value": value,
                "temporal_info": temporal_info,
                "slot_type": slot_type,
            })

    return gt_slots


def normalize_for_match(s: str) -> str:
    """Normalize string for fuzzy matching."""
    import re
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def slot_matches(pred_value: str, gt_value: str) -> bool:
    """Check if predicted value matches GT value (fuzzy)."""
    pn = normalize_for_match(pred_value)
    gn = normalize_for_match(gt_value)
    if pn == gn:
        return True
    # Substring containment
    if len(pn) > 3 and len(gn) > 3:
        if pn in gn or gn in pn:
            return True
    # Token overlap
    pt = set(pn.split())
    gt = set(gn.split())
    if pt and gt:
        overlap = len(pt & gt) / max(len(pt), len(gt))
        if overlap >= 0.5:
            return True
    return False


def evaluate_sample(
    pred_memories: List[CanonicalMemory],
    gt_slots: List[dict],
) -> dict:
    """Evaluate one sample: slot F1 + temporal accuracy."""
    # Slot-level matching
    matched_gt = set()
    matched_pred = set()

    for pi, mem in enumerate(pred_memories):
        for gi, gt in enumerate(gt_slots):
            if gi in matched_gt:
                continue
            # Attribute must be compatible
            if mem.attribute != gt["attribute"]:
                # Allow cross-matching for symptom/diagnosis
                if not (mem.attribute in ("symptom", "primary_diagnosis") and
                        gt["attribute"] in ("symptom", "primary_diagnosis")):
                    continue

            if slot_matches(mem.value, gt["value"]):
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    n_pred = len(pred_memories)
    n_gt = len(gt_slots)
    n_matched = len(matched_gt)

    precision = n_matched / n_pred if n_pred > 0 else 0.0
    recall = n_matched / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Temporal accuracy: among matched slots, how many have correct onset?
    temporal_correct = 0
    temporal_total = 0

    for gi in matched_gt:
        gt = gt_slots[gi]
        if not gt["temporal_info"]:
            continue
        temporal_total += 1

        # Find matching prediction
        for pi, mem in enumerate(pred_memories):
            if slot_matches(mem.value, gt["value"]):
                # Check if time_scope captures the temporal info
                if mem.time_scope and mem.time_scope != "global":
                    # Any overlap in temporal expression is a partial match
                    gt_onset = gt["temporal_info"].get("onset", "")
                    if gt_onset:
                        pred_time = normalize_for_match(mem.time_scope)
                        gt_time = normalize_for_match(gt_onset)
                        if pred_time in gt_time or gt_time in pred_time:
                            temporal_correct += 1
                break

    temporal_acc = temporal_correct / temporal_total if temporal_total > 0 else 0.0

    return {
        "slot_precision": round(precision, 4),
        "slot_recall": round(recall, 4),
        "slot_f1": round(f1, 4),
        "n_pred": n_pred,
        "n_gt": n_gt,
        "n_matched": n_matched,
        "temporal_accuracy": round(temporal_acc, 4),
        "temporal_total": temporal_total,
        "temporal_correct": temporal_correct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/raw/meditod/temp_clone")
    parser.add_argument("--output_dir", default="results/meditod")
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    samples = load_meditod_raw(args.data_path)
    print(f"Loaded {len(samples)} MediTOD dialogues")

    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"Evaluating {len(samples)} samples\n")

    # Init pipeline
    pipeline = UniqueClusterMemoryPipeline(use_embedding=True)

    all_results = []
    t0 = time.time()

    MAX_TURNS = 30  # Truncate long dialogues to avoid API timeouts

    for i, sample in enumerate(samples):
        did = sample["dialogue_id"]
        dialogue = sample["dialogue"][:MAX_TURNS]  # Truncate
        gt_slots = sample["gt_slots"]

        if not dialogue or not gt_slots:
            print(f"  [{i+1}/{len(samples)}] {did}: skipped (empty)")
            continue

        try:
            memories = pipeline.build_memory(dialogue, did)
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] {did}: pipeline error ({e})")
            continue

        metrics = evaluate_sample(memories, gt_slots)
        metrics["sample_id"] = did
        metrics["n_turns"] = len(dialogue)
        all_results.append(metrics)

        elapsed = time.time() - t0
        avg_time = elapsed / (i + 1)

        print(f"  [{i+1}/{len(samples)}] {did}: "
              f"Slot-F1={metrics['slot_f1']:.3f}  "
              f"TempAcc={metrics['temporal_accuracy']:.3f}  "
              f"({avg_time:.1f}s/sample)")

    # Aggregate
    if not all_results:
        print("\nNo results to aggregate.")
        return

    n = len(all_results)
    avg_f1 = sum(r["slot_f1"] for r in all_results) / n
    avg_prec = sum(r["slot_precision"] for r in all_results) / n
    avg_rec = sum(r["slot_recall"] for r in all_results) / n
    avg_temp = sum(r["temporal_accuracy"] for r in all_results) / n
    total_temp_correct = sum(r["temporal_correct"] for r in all_results)
    total_temp_total = sum(r["temporal_total"] for r in all_results)
    micro_temp_acc = total_temp_correct / total_temp_total if total_temp_total > 0 else 0.0

    summary = {
        "n_samples": n,
        "slot_precision": round(avg_prec, 4),
        "slot_recall": round(avg_rec, 4),
        "slot_f1": round(avg_f1, 4),
        "temporal_accuracy_macro": round(avg_temp, 4),
        "temporal_accuracy_micro": round(micro_temp_acc, 4),
        "temporal_correct": total_temp_correct,
        "temporal_total": total_temp_total,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    print(f"\n{'=' * 60}")
    print(f"  MediTOD Evaluation Results (n={n})")
    print(f"{'=' * 60}")
    print(f"  Slot Precision : {avg_prec:.4f}")
    print(f"  Slot Recall    : {avg_rec:.4f}")
    print(f"  Slot F1        : {avg_f1:.4f}")
    print(f"  Temporal Acc   : {micro_temp_acc:.4f} ({total_temp_correct}/{total_temp_total})")
    print(f"  Elapsed        : {summary['elapsed_seconds']:.1f}s")

    with open(os.path.join(args.output_dir, "meditod_results.json"), "w") as f:
        json.dump({"summary": summary, "per_sample": all_results}, f, indent=2)
    print(f"\n  Saved: {args.output_dir}/meditod_results.json")


if __name__ == "__main__":
    main()
