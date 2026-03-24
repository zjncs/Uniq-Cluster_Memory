"""
experiments/run_meditod_eval_v2.py
===================================
MediTOD external validation (v2): truncated dialogues + raw NLU GT.

Uses all 231 raw MediTOD dialogues, truncated to first 20 turns each
to avoid M1 timeout on long conversations.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.uniq_cluster_memory.pipeline import UniqueClusterMemoryPipeline
from src.uniq_cluster_memory.schema import CanonicalMemory


SLOT_MAP = {
    "symptom": "symptom",
    "medication": "medication",
    "medical history": "primary_diagnosis",
    "family history": "primary_diagnosis",
}

MAX_TURNS = 20  # Truncate to avoid M1 timeout


def load_meditod_all(base_path: str) -> List[dict]:
    """Load all 231 MediTOD raw dialogues."""
    raw_dlg = json.load(open(Path(base_path) / "raw_data" / "dialogs.json"))
    raw_ann = json.load(open(Path(base_path) / "raw_data" / "annotations.json"))

    samples = []
    for did, dlg_data in raw_dlg.items():
        # Parse dialogue
        raw_turns = dlg_data if isinstance(dlg_data, list) else dlg_data.get("dialogue", dlg_data.get("utterances", []))
        turns = []
        for i, turn in enumerate(raw_turns[:MAX_TURNS]):
            if isinstance(turn, dict):
                sp = turn.get("speaker", turn.get("role", "patient")).strip().lower()
                text = turn.get("text", turn.get("utterance", turn.get("content", "")))
            elif isinstance(turn, str):
                sp = "patient" if i % 2 == 0 else "doctor"
                text = turn
            else:
                continue
            turns.append({
                "turn_id": i,
                "speaker": "doctor" if sp in ("doctor", "assistant") else "patient",
                "text": text,
            })

        # Extract GT from raw annotations
        annot_list = raw_ann.get(did, [])
        gt_slots = extract_gt(annot_list)

        if turns and gt_slots:
            samples.append({
                "dialogue_id": f"meditod_{did}",
                "dialogue": turns,
                "gt_slots": gt_slots,
                "n_total_turns": len(raw_turns),
            })

    return samples


def extract_gt(annot_list) -> List[dict]:
    """Extract positive inform slots from raw annotations."""
    gt = []
    seen = set()

    for uttr_annots in annot_list:
        if not isinstance(uttr_annots, list):
            continue
        for a in uttr_annots:
            if not isinstance(a, dict) or a.get("intent") != "inform":
                continue
            slot = a.get("slot", "")
            value = a.get("value", "").strip()
            status = a.get("status", "")
            when = a.get("when", "")

            ucm_attr = SLOT_MAP.get(slot)
            if not ucm_attr or not value:
                continue
            if status and status.lower() in ("no", "false"):
                continue

            key = (ucm_attr, value.lower())
            if key in seen:
                continue
            seen.add(key)

            gt.append({
                "attribute": ucm_attr,
                "value": value,
                "temporal": when if when else "",
            })

    return gt


def norm(s):
    return re.sub(r"[^\w]", "", s.strip().lower())


def slot_match(pred_val, gt_val):
    pn, gn = norm(pred_val), norm(gt_val)
    if pn == gn:
        return True
    if len(pn) > 3 and len(gn) > 3 and (pn in gn or gn in pn):
        return True
    pt, gt = set(pn.split()), set(gn.split())
    if pt and gt and len(pt & gt) / max(len(pt), len(gt)) >= 0.5:
        return True
    return False


def evaluate(pred: List[CanonicalMemory], gt_slots: List[dict]) -> dict:
    matched_gt = set()
    matched_pred = set()

    for pi, mem in enumerate(pred):
        for gi, gt in enumerate(gt_slots):
            if gi in matched_gt:
                continue
            if mem.attribute != gt["attribute"]:
                if not (mem.attribute in ("symptom", "primary_diagnosis") and
                        gt["attribute"] in ("symptom", "primary_diagnosis")):
                    continue
            if slot_match(mem.value, gt["value"]):
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    n_pred, n_gt, n_match = len(pred), len(gt_slots), len(matched_gt)
    p = n_match / n_pred if n_pred else 0
    r = n_match / n_gt if n_gt else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0

    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "n_pred": n_pred, "n_gt": n_gt, "n_match": n_match}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/raw/meditod/temp_clone")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--output_dir", default="results/meditod_v2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    samples = load_meditod_all(args.data_path)[:args.max_samples]
    print(f"MediTOD v2: {len(samples)} samples (truncated to {MAX_TURNS} turns each)\n")

    pipeline = UniqueClusterMemoryPipeline(use_embedding=True)
    results = []
    t0 = time.time()

    for i, s in enumerate(samples):
        did = s["dialogue_id"]
        try:
            memories = pipeline.build_memory(s["dialogue"], did)
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] {did}: error ({e})")
            continue

        metrics = evaluate(memories, s["gt_slots"])
        metrics["sample_id"] = did
        metrics["n_turns_used"] = len(s["dialogue"])
        metrics["n_turns_total"] = s["n_total_turns"]
        results.append(metrics)

        print(f"  [{i+1}/{len(samples)}] {did}: "
              f"F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
              f"({len(s['dialogue'])}/{s['n_total_turns']} turns)")

    if not results:
        print("No results.")
        return

    n = len(results)
    avg_f1 = sum(r["f1"] for r in results) / n
    avg_p = sum(r["precision"] for r in results) / n
    avg_r = sum(r["recall"] for r in results) / n

    print(f"\n{'=' * 60}")
    print(f"  MediTOD v2 Results (n={n})")
    print(f"{'=' * 60}")
    print(f"  Slot Precision : {avg_p:.4f}")
    print(f"  Slot Recall    : {avg_r:.4f}")
    print(f"  Slot F1        : {avg_f1:.4f}")
    print(f"  Elapsed        : {time.time()-t0:.1f}s")

    with open(os.path.join(args.output_dir, "meditod_v2_results.json"), "w") as f:
        json.dump({"summary": {"n": n, "precision": round(avg_p, 4), "recall": round(avg_r, 4),
                                "f1": round(avg_f1, 4)}, "per_sample": results}, f, indent=2)
    print(f"  Saved: {args.output_dir}/")


if __name__ == "__main__":
    main()
