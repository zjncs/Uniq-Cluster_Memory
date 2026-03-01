"""
diagnose_u_f1.py
================
诊断 U-F1 全 0 的根本原因。

分析维度：
    1. LLM 实际抽取了什么 attribute 名称？GT 用的是什么名称？
    2. LLM 抽取的 value 与 GT 的 value 是否接近？
    3. LLM 抽取的 time_scope 与 GT 是否一致？
    4. patient_id 是否一致？
    5. 匹配失败的主要原因分布。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.uniq_cluster_memory.schema import CanonicalMemory, ConflictRecord
from evaluation.uniqueness_eval import _make_strict_key, _make_unique_key
from src.uniq_cluster_memory.utils.llm_client import (
    get_llm_client,
    LLM_MODEL,
)

client = get_llm_client()
DATA_DIR = Path("data/raw/med_longmem")


def _make_match_key(mem: CanonicalMemory) -> tuple:
    """诊断脚本沿用旧命名，实际使用 strict key。"""
    return _make_strict_key(mem)

EXTRACTION_SYSTEM_PROMPT = """You are a medical information extraction system. 
Given a patient-doctor dialogue, extract all unique medical facts as structured records.

For each unique medical attribute mentioned, create ONE canonical record with:
- patient_id: use the dialogue_id provided
- attribute: standardized attribute name (e.g., blood_glucose, blood_pressure_sys, medication, symptom, primary_diagnosis)
- value: the FINAL/MOST RECENT value mentioned for this attribute in this time scope
- unit: measurement unit if applicable (e.g., mmol/L, mmHg, bpm, °C, g/L), empty string if none
- time_scope: the date or time period this value refers to (use ISO date format if mentioned, otherwise "global")
- conflict_flag: true if you detect that the same attribute had DIFFERENT values mentioned for the same time scope
- update_policy: "unique" for measurements/diagnoses, "latest" for medications, "append" for symptoms
- conflict_history: list of {old_value, new_value, conflict_type} if conflict_flag is true, else empty list

Return ONLY a JSON array of canonical memory objects. No explanation."""


def extract_for_diagnosis(dialogue_text: str, dialogue_id: str) -> list[CanonicalMemory]:
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Dialogue ID: {dialogue_id}\n\n{dialogue_text}"},
            ],
            temperature=0.0,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        records = json.loads(raw)
        memories = []
        for r in records:
            conflict_history = []
            for ch in r.get("conflict_history", []):
                conflict_history.append(ConflictRecord(
                    old_value=str(ch.get("old_value", "")),
                    new_value=str(ch.get("new_value", "")),
                    old_provenance=[], new_provenance=[],
                    conflict_type=ch.get("conflict_type", "value_change"),
                    detected_at=r.get("time_scope", ""),
                ))
            memories.append(CanonicalMemory(
                patient_id=str(r.get("patient_id", dialogue_id)),
                attribute=str(r.get("attribute", "")),
                value=str(r.get("value", "")),
                unit=str(r.get("unit", "")),
                time_scope=str(r.get("time_scope", "global")),
                confidence=float(r.get("confidence", 1.0)),
                provenance=[],
                conflict_flag=bool(r.get("conflict_flag", False)),
                conflict_history=conflict_history,
                update_policy=str(r.get("update_policy", "unique")),
            ))
        return memories
    except Exception as e:
        print(f"  Extraction error: {e}")
        return []


def load_sample(dialogue_id: str) -> dict:
    sample_dir = DATA_DIR / dialogue_id
    dialogue = []
    with open(sample_dir / "dialogue.jsonl") as f:
        for line in f:
            dialogue.append(json.loads(line))
    canonical_gt = []
    with open(sample_dir / "canonical_gt.jsonl") as f:
        for line in f:
            canonical_gt.append(CanonicalMemory.from_dict(json.loads(line)))
    return {"dialogue_id": dialogue_id, "dialogue": dialogue, "canonical_gt": canonical_gt}


def dialogue_to_text(dialogue: list[dict]) -> str:
    lines = []
    for turn in dialogue:
        speaker = "Doctor" if turn["speaker"] == "doctor" else "Patient"
        lines.append(f"{speaker} (Turn {turn['turn_id']}): {turn['text']}")
    return "\n".join(lines)


def diagnose_sample(sample: dict) -> dict:
    """对单条样本进行深度诊断。"""
    did = sample["dialogue_id"]
    gt = sample["canonical_gt"]
    dialogue_text = dialogue_to_text(sample["dialogue"])
    predicted = extract_for_diagnosis(dialogue_text, did)

    gt_match_keys = {_make_match_key(m) for m in gt}
    gt_unique_keys = {_make_unique_key(m) for m in gt}
    pred_match_keys = {_make_match_key(m) for m in predicted}
    pred_unique_keys = {_make_unique_key(m) for m in predicted}

    # 分析匹配失败原因
    failures = []
    for gt_mem in gt:
        gt_key = _make_match_key(gt_mem)
        gt_ukey = _make_unique_key(gt_mem)

        # 检查 patient_id 是否匹配
        pred_same_pid = [p for p in predicted if p.patient_id.strip().lower() == gt_mem.patient_id.strip().lower()]
        # 检查 attribute 是否匹配（宽松：只要包含关键词）
        pred_same_attr = [p for p in predicted if gt_mem.attribute.lower() in p.attribute.lower() or p.attribute.lower() in gt_mem.attribute.lower()]
        # 检查 unique_key 是否匹配（pid + attr + scope）
        pred_same_ukey = [p for p in predicted if _make_unique_key(p) == gt_ukey]
        # 检查完整 match_key（pid + attr + scope + value）
        exact_match = gt_key in pred_match_keys

        if exact_match:
            continue  # 完全匹配，跳过

        reason = []
        if not pred_same_pid:
            reason.append(f"patient_id_mismatch: GT={gt_mem.patient_id!r} vs PRED={[p.patient_id for p in predicted[:3]]}")
        elif not pred_same_attr:
            reason.append(f"attribute_mismatch: GT={gt_mem.attribute!r} vs PRED_attrs={[p.attribute for p in predicted]}")
        elif not pred_same_ukey:
            reason.append(f"scope_mismatch: GT_scope={gt_mem.time_scope!r} vs PRED_scopes={[p.time_scope for p in pred_same_attr]}")
        else:
            # unique_key 匹配但 value 不匹配
            pred_values = [p.value for p in pred_same_ukey]
            reason.append(f"value_mismatch: GT={gt_mem.value!r} vs PRED={pred_values}")

        failures.append({
            "gt_attribute": gt_mem.attribute,
            "gt_value": gt_mem.value,
            "gt_scope": gt_mem.time_scope,
            "gt_patient_id": gt_mem.patient_id,
            "reason": " | ".join(reason),
        })

    return {
        "dialogue_id": did,
        "n_gt": len(gt),
        "n_predicted": len(predicted),
        "n_exact_match": len(gt_match_keys & pred_match_keys),
        "n_ukey_match": len(gt_unique_keys & pred_unique_keys),
        "failures": failures,
        "gt_attributes": [m.attribute for m in gt],
        "pred_attributes": [m.attribute for m in predicted],
        "gt_scopes": list({m.time_scope for m in gt}),
        "pred_scopes": list({m.time_scope for m in predicted}),
        "gt_patient_ids": list({m.patient_id for m in gt}),
        "pred_patient_ids": list({m.patient_id for m in predicted}),
    }


def main():
    # 只诊断前 3 条样本（足够找出规律）
    summary_path = DATA_DIR / "dataset_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    dialogue_ids = [s["dialogue_id"] for s in summary["samples"][:3]]

    print(f"\n{'='*70}")
    print("  U-F1 = 0 Root Cause Diagnosis (3 samples)")
    print(f"{'='*70}\n")

    all_failure_reasons = []
    for did in dialogue_ids:
        print(f"Diagnosing {did}...")
        sample = load_sample(did)
        result = diagnose_sample(sample)

        print(f"\n  [{did}]")
        print(f"  GT attributes   : {result['gt_attributes']}")
        print(f"  Pred attributes : {result['pred_attributes']}")
        print(f"  GT scopes       : {result['gt_scopes']}")
        print(f"  Pred scopes     : {result['pred_scopes']}")
        print(f"  GT patient_ids  : {result['gt_patient_ids']}")
        print(f"  Pred patient_ids: {result['pred_patient_ids']}")
        print(f"  Exact match     : {result['n_exact_match']} / {result['n_gt']}")
        print(f"  UniqueKey match : {result['n_ukey_match']} / {result['n_gt']}")
        if result["failures"]:
            print(f"  Failure reasons:")
            for f in result["failures"]:
                print(f"    - {f['gt_attribute']:20s} (GT={f['gt_value']!r:10s}, scope={f['gt_scope']!r}): {f['reason']}")
        all_failure_reasons.extend(result["failures"])

    # 统计失败原因分布
    print(f"\n{'='*70}")
    print("  Failure Reason Distribution")
    print(f"{'='*70}")
    reason_counts = {}
    for f in all_failure_reasons:
        reason_type = f["reason"].split(":")[0]
        reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:30s}: {count}")

    print(f"\n{'='*70}")
    print("  Diagnosis Complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
