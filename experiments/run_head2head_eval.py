"""
experiments/run_head2head_eval.py
=================================
Head-to-head comparison of UCM vs Long-Context LLM on real MediTOD dialogues.

Uses LLM-as-Judge to blindly evaluate which system produces better structured memories.
No ground-truth needed — the judge directly compares two memory sets.

Usage:
    PYTHONPATH=. python experiments/run_head2head_eval.py \
        --n_samples 20 --output_dir results/head2head
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.uniq_cluster_memory.pipeline import UniqueClusterMemoryPipeline
from src.uniq_cluster_memory.utils.llm_client import get_llm_client, LLM_MODEL

# ---------------------------------------------------------------------------
# Long-Context LLM baseline: directly ask LLM to extract structured memories
# ---------------------------------------------------------------------------

BASELINE_PROMPT = """You are a medical record assistant. Given the following patient-doctor dialogue,
extract ALL important medical facts as structured memories.

For each fact, output a JSON object with:
- "attribute": the medical attribute (e.g., "blood_pressure_sys", "medication", "symptom", "diagnosis")
- "value": the specific value mentioned
- "time_scope": when this was recorded/mentioned (ISO date if possible, or "unknown")
- "confidence": how confident you are (0.0-1.0)

Output a JSON array of these objects. Be thorough — capture every clinically relevant fact.

DIALOGUE:
{dialogue}

OUTPUT (JSON array):"""

# ---------------------------------------------------------------------------
# LLM-as-Judge: pairwise blind comparison
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are a senior medical AI evaluator. You will compare TWO sets of structured
medical memories extracted from the SAME patient-doctor dialogue. The memories are labeled
"System A" and "System B" (order is randomized).

Evaluate each system on these 5 dimensions (score 1-5 each):

1. COMPLETENESS: Does the memory capture all clinically important facts from the dialogue?
2. ACCURACY: Are the extracted values factually correct based on the dialogue?
3. TEMPORAL_GROUNDING: Are time references properly resolved (not just "yesterday" but actual dates)?
4. CONFLICT_HANDLING: When the dialogue contains contradictory information, does the memory handle it properly?
5. CONCISENESS: Is the memory free of redundant or trivial entries?

Then give an OVERALL_WINNER: "A", "B", or "tie".

DIALOGUE:
{dialogue}

--- SYSTEM A MEMORIES ---
{memories_a}

--- SYSTEM B MEMORIES ---
{memories_b}

Respond in this exact JSON format:
{{
  "system_a": {{"completeness": X, "accuracy": X, "temporal_grounding": X, "conflict_handling": X, "conciseness": X}},
  "system_b": {{"completeness": X, "accuracy": X, "temporal_grounding": X, "conflict_handling": X, "conciseness": X}},
  "overall_winner": "A" or "B" or "tie",
  "reasoning": "brief explanation"
}}"""


def load_meditod_dialogues(data_path: str, n_samples: int = 20, max_turns: int = 30) -> list:
    """Load MediTOD dialogues and format as turn-based text."""
    dialogs_path = os.path.join(data_path, "raw_data", "dialogs.json")
    dialogs = json.load(open(dialogs_path))

    samples = []
    keys = sorted(dialogs.keys())
    selected = keys[:n_samples] if n_samples <= len(keys) else keys

    for k in selected:
        d = dialogs[k]
        utterances = d.get("utterances", [])
        turns = []
        for i, utt in enumerate(utterances[:max_turns]):
            if isinstance(utt, dict):
                speaker = utt.get("speaker", "doctor" if i % 2 == 0 else "patient")
                text = utt.get("text", utt.get("utterance", str(utt)))
            else:
                speaker = "doctor" if i % 2 == 0 else "patient"
                text = str(utt)
            turns.append(f"{speaker}: {text}")

        samples.append({
            "id": d.get("dlg_id", k),
            "dialogue_text": "\n".join(turns),
            "n_turns": len(turns),
        })

    return samples


def run_baseline(dialogue_text: str, client, max_retries: int = 3) -> str:
    """Run Long-Context LLM baseline."""
    prompt = BASELINE_PROMPT.format(dialogue=dialogue_text)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return f"[ERROR: {e}]"


def run_ucm(dialogue_text: str, sample_id: str) -> str:
    """Run UCM pipeline on dialogue."""
    try:
        pipeline = UniqueClusterMemoryPipeline()
        # Convert text to turn dicts
        turns = []
        for line in dialogue_text.split("\n"):
            if ": " in line:
                speaker, text = line.split(": ", 1)
                turns.append({"speaker": speaker.strip(), "text": text.strip()})

        memories = pipeline.build_memory(turns, dialogue_id=sample_id)
        if not memories:
            return "[]"

        mem_list = []
        for m in memories:
            entry = {
                "attribute": getattr(m, "attribute", "unknown"),
                "value": getattr(m, "value", ""),
                "time_scope": getattr(m, "time_scope", "unknown"),
                "confidence": getattr(m, "confidence", 1.0),
                "conflict_flag": getattr(m, "conflict_flag", False),
            }
            mem_list.append(entry)
        return json.dumps(mem_list, indent=2, ensure_ascii=False)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"[ERROR: {e}]"


def run_judge(dialogue_text: str, memories_a: str, memories_b: str,
              client, max_retries: int = 3) -> dict:
    """Run LLM-as-Judge pairwise comparison."""
    prompt = JUDGE_PROMPT.format(
        dialogue=dialogue_text,
        memories_a=memories_a,
        memories_b=memories_b,
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0,
            )
            text = resp.choices[0].message.content.strip()
            # Extract JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return {"error": "JSON parse failed", "raw": text}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/raw/meditod/temp_clone")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--max_turns", type=int, default=30)
    parser.add_argument("--output_dir", default="results/head2head")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    client = get_llm_client()

    print(f"Loading MediTOD dialogues...")
    samples = load_meditod_dialogues(args.data_path, args.n_samples, args.max_turns)
    print(f"Loaded {len(samples)} dialogues")

    results = []
    ucm_wins = 0
    llm_wins = 0
    ties = 0
    ucm_scores = {k: [] for k in ["completeness", "accuracy", "temporal_grounding", "conflict_handling", "conciseness"]}
    llm_scores = {k: [] for k in ["completeness", "accuracy", "temporal_grounding", "conflict_handling", "conciseness"]}

    for i, sample in enumerate(samples):
        t0 = time.time()
        print(f"\n  [{i+1:02d}/{len(samples)}] {sample['id']} ({sample['n_turns']} turns)...")

        # Run both systems
        print(f"    Running LLM baseline...", end="", flush=True)
        llm_memories = run_baseline(sample["dialogue_text"], client)
        print(f" done ({time.time()-t0:.1f}s)")

        t1 = time.time()
        print(f"    Running UCM...", end="", flush=True)
        ucm_memories = run_ucm(sample["dialogue_text"], sample["id"])
        print(f" done ({time.time()-t1:.1f}s)")

        # Randomize order to avoid position bias
        ucm_is_a = random.random() < 0.5
        if ucm_is_a:
            mem_a, mem_b = ucm_memories, llm_memories
        else:
            mem_a, mem_b = llm_memories, ucm_memories

        # Judge
        t2 = time.time()
        print(f"    Running Judge...", end="", flush=True)
        judge_result = run_judge(sample["dialogue_text"], mem_a, mem_b, client)
        print(f" done ({time.time()-t2:.1f}s)")

        # Map back to real systems
        if "error" not in judge_result:
            winner_label = judge_result.get("overall_winner", "tie")
            if ucm_is_a:
                ucm_label, llm_label = "system_a", "system_b"
                if winner_label == "A":
                    winner = "UCM"
                elif winner_label == "B":
                    winner = "LLM"
                else:
                    winner = "tie"
            else:
                ucm_label, llm_label = "system_b", "system_a"
                if winner_label == "A":
                    winner = "LLM"
                elif winner_label == "B":
                    winner = "UCM"
                else:
                    winner = "tie"

            if winner == "UCM":
                ucm_wins += 1
            elif winner == "LLM":
                llm_wins += 1
            else:
                ties += 1

            # Collect dimension scores
            for dim in ucm_scores:
                ucm_s = judge_result.get(ucm_label, {}).get(dim, 0)
                llm_s = judge_result.get(llm_label, {}).get(dim, 0)
                if isinstance(ucm_s, (int, float)):
                    ucm_scores[dim].append(ucm_s)
                if isinstance(llm_s, (int, float)):
                    llm_scores[dim].append(llm_s)

            print(f"    Winner: {winner} | UCM={'A' if ucm_is_a else 'B'}")
        else:
            print(f"    Judge error: {judge_result.get('error')}")
            winner = "error"

        results.append({
            "sample_id": sample["id"],
            "ucm_is_a": ucm_is_a,
            "ucm_memories": ucm_memories[:500],
            "llm_memories": llm_memories[:500],
            "judge": judge_result,
            "winner": winner,
        })

    # Summary
    total = ucm_wins + llm_wins + ties
    print(f"\n{'='*60}")
    print(f"  Head-to-Head Results (n={total})")
    print(f"{'='*60}")
    print(f"  UCM wins : {ucm_wins} ({ucm_wins/max(total,1)*100:.1f}%)")
    print(f"  LLM wins : {llm_wins} ({llm_wins/max(total,1)*100:.1f}%)")
    print(f"  Ties     : {ties} ({ties/max(total,1)*100:.1f}%)")
    print()
    print(f"  Dimension Scores (avg):")
    print(f"  {'Dimension':<25s} {'UCM':>6s} {'LLM':>6s} {'Δ':>7s}")
    print(f"  {'─'*48}")

    summary = {"ucm_wins": ucm_wins, "llm_wins": llm_wins, "ties": ties, "dimensions": {}}
    for dim in ucm_scores:
        ucm_avg = sum(ucm_scores[dim]) / max(len(ucm_scores[dim]), 1)
        llm_avg = sum(llm_scores[dim]) / max(len(llm_scores[dim]), 1)
        delta = ucm_avg - llm_avg
        print(f"  {dim:<25s} {ucm_avg:>6.2f} {llm_avg:>6.2f} {delta:>+7.2f}")
        summary["dimensions"][dim] = {"ucm": ucm_avg, "llm": llm_avg, "delta": delta}

    # Save
    output_file = os.path.join(args.output_dir, "head2head_results.json")
    json.dump({"summary": summary, "details": results}, open(output_file, "w"), indent=2, ensure_ascii=False)
    print(f"\n  Saved: {output_file}")


if __name__ == "__main__":
    main()
