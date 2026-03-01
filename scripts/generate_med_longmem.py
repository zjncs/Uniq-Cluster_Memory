"""
generate_med_longmem.py
=======================
Med-LongMem v0.1 主生成脚本。

流程（GT-First 原则）：
    1. 调用 EventGenerator 生成结构化事件时间线（纯确定性逻辑，无 LLM）。
    2. 调用 derive_canonical_gt / derive_conflict_gt 派生 GT 三层。
    3. 调用 LLM 将事件时间线渲染为 20 轮英文患者-医生对话（LLM 仅作渲染引擎）。
    4. 将所有文件写入 data/raw/med_longmem/{dialogue_id}/ 目录。

用法：
    python3 scripts/generate_med_longmem.py --n_samples 20 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.event_generator import (
    EventGenerator,
    RawEvent,
    derive_canonical_gt,
    derive_conflict_gt,
)
from src.uniq_cluster_memory.utils.llm_client import (
    get_llm_client,
    LLM_MODEL,
)

# ─── LLM 客户端初始化 ────────────────────────────────────────────────────────

client = get_llm_client()


# ─── 对话渲染提示词 ──────────────────────────────────────────────────────────

SURFACING_SYSTEM_PROMPT = """You are a medical dialogue writer. Your task is to write a realistic, 
multi-turn patient-doctor conversation in English. The conversation must naturally incorporate 
all the medical events provided in the event timeline, while following these strict rules:

1. TURN ASSIGNMENT: Each event has a `turn_id` (0-indexed). The dialogue must have exactly 20 turns 
   (turn 0 to turn 19). Each turn is either a patient statement or a doctor statement.
   
2. EVENT EMBEDDING: Each event must appear in its assigned turn. The information must be naturally 
   embedded in the dialogue, not listed mechanically.

3. ADVERSARIAL TAGS — CRITICAL:
   - "conflict": The same attribute appears with a DIFFERENT value at a later turn. 
     The speaker should mention this naturally (e.g., "Actually, when I checked again this morning, 
     my blood sugar was X"). Do NOT explicitly say "this conflicts with".
   - "coref": Use a natural coreference expression (e.g., "that reading", "the result from last time", 
     "that number") to refer back to the event specified in coref_target_event_id.
   - "update": The medication/treatment has been changed. The doctor should naturally prescribe 
     the new medication (e.g., "I am going to switch you to X").

4. MEDICAL REALISM: Use appropriate medical terminology. The doctor should ask follow-up questions, 
   provide explanations, and give clinical advice. The patient should describe symptoms naturally.

5. OUTPUT FORMAT: Return ONLY a JSON array of 20 objects, each with:
   {"turn_id": <int>, "speaker": "patient" or "doctor", "text": "<dialogue text>"}
   
Do not include any explanation or markdown formatting outside the JSON array."""


def render_dialogue(events: list[RawEvent], dialogue_id: str) -> list[dict]:
    """
    调用 LLM 将事件时间线渲染为 20 轮英文对话。
    """
    timeline_lines = []
    for evt in sorted(events, key=lambda e: e.turn_id):
        tag_info = ""
        if evt.adversarial_tag:
            tag_info = f" [TAG: {evt.adversarial_tag}"
            if evt.coref_target_event_id:
                tag_info += f", refers_to={evt.coref_target_event_id}"
            tag_info += "]"
        unit_str = f" {evt.unit}" if evt.unit else ""
        timeline_lines.append(
            f"Turn {evt.turn_id:02d} | {evt.speaker:7s} | "
            f"{evt.attribute}: {evt.value}{unit_str} (scope: {evt.time_scope}){tag_info}"
        )

    user_prompt = (
        f"Please write a 20-turn patient-doctor dialogue for dialogue ID: {dialogue_id}\n\n"
        f"EVENT TIMELINE:\n"
        + "\n".join(timeline_lines)
        + "\n\nRemember: Return ONLY the JSON array of 20 dialogue turns."
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SURFACING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=3000,
            )
            raw_text = response.choices[0].message.content.strip()
            # 清理 markdown 代码块
            if raw_text.startswith("```"):
                parts = raw_text.split("```")
                raw_text = parts[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            raw_text = raw_text.strip()
            turns = json.loads(raw_text)
            assert isinstance(turns, list) and len(turns) == 20
            for t in turns:
                assert "turn_id" in t and "speaker" in t and "text" in t
            return turns
        except Exception as e:
            print(f"    [Attempt {attempt+1}/3] Rendering failed: {e}")
            if attempt < 2:
                time.sleep(3)

    print(f"    [WARNING] Using fallback dialogue for {dialogue_id}")
    return _fallback_dialogue(events)


def _fallback_dialogue(events: list[RawEvent]) -> list[dict]:
    """LLM 渲染失败时的占位对话。"""
    event_map = {e.turn_id: e for e in events}
    turns = []
    for i in range(20):
        if i in event_map:
            evt = event_map[i]
            unit_str = f" {evt.unit}" if evt.unit else ""
            text = f"The {evt.attribute} reading is {evt.value}{unit_str}."
            speaker = evt.speaker
        else:
            speaker = "doctor" if i % 2 == 0 else "patient"
            text = "Please continue with the consultation."
        turns.append({"turn_id": i, "speaker": speaker, "text": text})
    return turns


# ─── 主生成函数 ──────────────────────────────────────────────────────────────

def generate_sample(dialogue_id: str, output_dir: Path, seed: int) -> dict:
    """生成单条样本，写入 GT 三层文件，返回 metadata。"""
    gen = EventGenerator(seed=seed)
    events = gen.generate(dialogue_id)

    canonical_gt = derive_canonical_gt(events)
    conflict_gt = derive_conflict_gt(canonical_gt)

    print(f"  Rendering dialogue {dialogue_id} ({len(events)} events)...")
    dialogue_turns = render_dialogue(events, dialogue_id)

    sample_dir = output_dir / dialogue_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    with open(sample_dir / "raw_events.jsonl", "w", encoding="utf-8") as f:
        for evt in sorted(events, key=lambda e: e.turn_id):
            f.write(json.dumps(evt.to_dict(), ensure_ascii=False) + "\n")

    with open(sample_dir / "canonical_gt.jsonl", "w", encoding="utf-8") as f:
        for mem in canonical_gt:
            f.write(json.dumps(mem.to_dict(), ensure_ascii=False) + "\n")

    with open(sample_dir / "conflict_gt.jsonl", "w", encoding="utf-8") as f:
        for mem in conflict_gt:
            f.write(json.dumps(mem.to_dict(), ensure_ascii=False) + "\n")

    with open(sample_dir / "dialogue.jsonl", "w", encoding="utf-8") as f:
        for turn in dialogue_turns:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")

    metadata = {
        "dialogue_id": dialogue_id,
        "difficulty": "hard",
        "n_turns": 20,
        "n_raw_events": len(events),
        "n_canonical_gt": len(canonical_gt),
        "n_conflict_gt": len(conflict_gt),
        "n_conflicts": len(conflict_gt),
        "n_corefs": sum(1 for e in events if e.adversarial_tag == "coref"),
        "n_updates": sum(1 for e in events if e.adversarial_tag == "update"),
        "seed": seed,
    }
    with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Generate Med-LongMem v0.1 dataset")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="data/raw/med_longmem")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Med-LongMem v0.1 Generator")
    print(f"  Generating {args.n_samples} Hard-level samples (English)")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    all_metadata = []
    for i in range(args.n_samples):
        dialogue_id = f"medlm_{i:04d}"
        sample_seed = args.seed + i
        print(f"[{i+1:02d}/{args.n_samples}] {dialogue_id} (seed={sample_seed})")
        try:
            meta = generate_sample(dialogue_id, output_dir, seed=sample_seed)
            all_metadata.append(meta)
            print(
                f"  OK: {meta['n_raw_events']} events | "
                f"{meta['n_canonical_gt']} canonical | "
                f"{meta['n_conflict_gt']} conflicts | "
                f"{meta['n_corefs']} corefs | "
                f"{meta['n_updates']} updates"
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    summary = {
        "version": "v0.1",
        "n_samples": len(all_metadata),
        "difficulty": "hard",
        "total_raw_events": sum(m["n_raw_events"] for m in all_metadata),
        "total_canonical_gt": sum(m["n_canonical_gt"] for m in all_metadata),
        "total_conflict_gt": sum(m["n_conflict_gt"] for m in all_metadata),
        "avg_conflicts_per_sample": round(
            sum(m["n_conflicts"] for m in all_metadata) / max(len(all_metadata), 1), 2
        ),
        "avg_corefs_per_sample": round(
            sum(m["n_corefs"] for m in all_metadata) / max(len(all_metadata), 1), 2
        ),
        "samples": all_metadata,
    }
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Generation Complete!")
    print(f"  Samples generated: {len(all_metadata)}")
    print(f"  Avg conflicts/sample: {summary['avg_conflicts_per_sample']}")
    print(f"  Avg corefs/sample:    {summary['avg_corefs_per_sample']}")
    print(f"  Summary saved: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
