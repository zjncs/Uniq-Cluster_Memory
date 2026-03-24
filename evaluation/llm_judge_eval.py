"""
evaluation/llm_judge_eval.py
==============================
GPT-4-as-Judge 评估模块。

使用 LLM 作为评审员对系统输出进行多维度评分，
替代人工评估（MedThink-Bench 显示 LLM 评估和人工评估相关性 r=0.87）。

评估维度：
    1. Factual Correctness（事实正确性）：记忆中的值是否与对话内容一致
    2. Completeness（完整性）：是否遗漏了对话中提到的重要医学信息
    3. Temporal Accuracy（时间准确性）：时间标注是否正确
    4. Conflict Detection（冲突检测）：是否正确识别了对话中的信息矛盾
    5. Redundancy（冗余度）：是否有不必要的重复记录

评分标准：1-5 分 Likert 量表。

用法：
    PYTHONPATH=. python evaluation/llm_judge_eval.py \\
        --data_path data/raw/med_longmem \\
        --predictions_path results/main_results/our_method_eval.json \\
        --output_path results/llm_judge/judge_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.uniq_cluster_memory.schema import CanonicalMemory
from src.uniq_cluster_memory.utils.llm_client import get_llm_client, LLM_MODEL


JUDGE_SYSTEM_PROMPT = """You are a medical AI evaluation expert. You will evaluate the quality of
structured medical memories extracted from a patient-doctor dialogue.

You must evaluate along 5 dimensions, each scored 1-5:

1. FACTUAL_CORRECTNESS (1-5): Are the extracted values accurate?
   5 = All values exactly match the dialogue
   3 = Most values correct, minor errors
   1 = Major factual errors

2. COMPLETENESS (1-5): Are all important medical facts captured?
   5 = All medical facts from dialogue captured
   3 = Major facts present, some omissions
   1 = Most facts missing

3. TEMPORAL_ACCURACY (1-5): Are time scopes correctly assigned?
   5 = All timestamps/scopes perfectly match
   3 = Most correct, some misaligned
   1 = Systematic temporal errors

4. CONFLICT_DETECTION (1-5): Are contradictory values correctly flagged?
   5 = All conflicts detected with correct old/new values
   3 = Most conflicts detected, some missed
   1 = Conflicts not detected

5. REDUNDANCY (1-5, higher is better): Is the output concise without duplicates?
   5 = No redundancy, every record is unique and necessary
   3 = Some minor redundancy
   1 = Heavily redundant

Output JSON ONLY with this exact schema:
{
  "factual_correctness": <1-5>,
  "completeness": <1-5>,
  "temporal_accuracy": <1-5>,
  "conflict_detection": <1-5>,
  "redundancy": <1-5>,
  "overall": <1-5>,
  "reasoning": "<brief explanation>"
}

No markdown. No extra text."""


def judge_single_sample(
    dialogue_turns: List[dict],
    predicted_memories: List[CanonicalMemory],
    gt_memories: Optional[List[CanonicalMemory]] = None,
    dialogue_id: str = "",
) -> Dict:
    """对单个样本执行 LLM-as-Judge 评估。"""
    client = get_llm_client()

    # 构建对话文本
    dialogue_text = ""
    for turn in dialogue_turns:
        speaker = turn.get("speaker", turn.get("role", "unknown"))
        text = turn.get("text", turn.get("content", ""))
        turn_id = turn.get("turn_id", "")
        dialogue_text += f"[Turn {turn_id}] {speaker}: {text}\n"

    # 构建预测记忆文本
    pred_text = ""
    for i, mem in enumerate(predicted_memories, 1):
        conflict_info = ""
        if mem.conflict_flag and mem.conflict_history:
            old_vals = [cr.old_value for cr in mem.conflict_history]
            conflict_info = f" [CONFLICT: was {', '.join(old_vals)}]"
        pred_text += (
            f"  {i}. {mem.attribute}: {mem.value}"
            f" (unit={mem.unit}, scope={mem.time_scope}, policy={mem.update_policy})"
            f"{conflict_info}\n"
        )

    # 可选：加入 GT 供参考
    gt_text = ""
    if gt_memories:
        gt_text = "\n\nGROUND TRUTH MEMORIES (for reference):\n"
        for i, mem in enumerate(gt_memories, 1):
            conflict_info = ""
            if mem.conflict_flag:
                conflict_info = " [HAS_CONFLICT]"
            gt_text += (
                f"  {i}. {mem.attribute}: {mem.value}"
                f" (scope={mem.time_scope}){conflict_info}\n"
            )

    user_prompt = (
        f"DIALOGUE (ID: {dialogue_id}):\n{dialogue_text}\n\n"
        f"EXTRACTED MEMORIES ({len(predicted_memories)} records):\n{pred_text}"
        f"{gt_text}\n"
        f"Evaluate the quality of the extracted memories."
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            result = json.loads(raw)
            result["sample_id"] = dialogue_id
            return result
        except Exception as e:
            if attempt < 2:
                time.sleep(3)

    return {
        "sample_id": dialogue_id,
        "factual_correctness": 0,
        "completeness": 0,
        "temporal_accuracy": 0,
        "conflict_detection": 0,
        "redundancy": 0,
        "overall": 0,
        "reasoning": "Judge failed",
        "error": True,
    }


def run_judge_evaluation(
    data_path: str,
    max_samples: Optional[int] = None,
    output_path: str = "results/llm_judge/judge_results.json",
) -> Dict:
    """对 Med-LongMem 数据集运行 LLM-as-Judge 评估。"""
    from benchmarks.med_longmem_task import MedLongMemTask
    from src.uniq_cluster_memory.pipeline import UniqueClusterMemoryPipeline
    from src.uniq_cluster_memory.defaults import recommended_pipeline_options

    task = MedLongMemTask(data_path=data_path, max_samples=max_samples)
    samples = task.get_samples()
    print(f"Loaded {len(samples)} samples for LLM-Judge evaluation\n")

    defaults = recommended_pipeline_options("med_longmem")
    pipeline = UniqueClusterMemoryPipeline(**defaults)

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
        predicted = pipeline.build_memory(
            dialogue, sample.sample_id, sample.question_date
        )
        build_time = time.time() - t0

        gt = sample.metadata.get("canonical_gt", [])

        t1 = time.time()
        judge_result = judge_single_sample(
            dialogue_turns=dialogue,
            predicted_memories=predicted,
            gt_memories=gt,
            dialogue_id=sample.sample_id,
        )
        judge_time = time.time() - t1

        judge_result["build_latency"] = round(build_time, 2)
        judge_result["judge_latency"] = round(judge_time, 2)
        judge_result["n_predicted"] = len(predicted)
        judge_result["n_gt"] = len(gt)
        results.append(judge_result)

        print(
            f"FC={judge_result.get('factual_correctness', 0)} "
            f"CM={judge_result.get('completeness', 0)} "
            f"TA={judge_result.get('temporal_accuracy', 0)} "
            f"CD={judge_result.get('conflict_detection', 0)} "
            f"RD={judge_result.get('redundancy', 0)} "
            f"OV={judge_result.get('overall', 0)} "
            f"({build_time:.1f}s+{judge_time:.1f}s)"
        )

    # Aggregate
    n = len([r for r in results if not r.get("error")])
    dimensions = [
        "factual_correctness", "completeness", "temporal_accuracy",
        "conflict_detection", "redundancy", "overall",
    ]

    summary = {
        "evaluation_type": "llm_judge",
        "judge_model": LLM_MODEL,
        "n_samples": len(results),
        "n_valid": n,
    }
    for dim in dimensions:
        vals = [r[dim] for r in results if not r.get("error") and dim in r]
        if vals:
            summary[f"mean_{dim}"] = round(sum(vals) / len(vals), 2)
            summary[f"std_{dim}"] = round(
                (sum((v - summary[f"mean_{dim}"]) ** 2 for v in vals) / len(vals)) ** 0.5, 2
            )

    summary["per_sample"] = results

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  LLM-as-Judge Evaluation Results")
    print(f"{'='*60}")
    for dim in dimensions:
        mean_key = f"mean_{dim}"
        if mean_key in summary:
            print(f"  {dim:25s}: {summary[mean_key]:.2f} (±{summary.get(f'std_{dim}', 0):.2f})")
    print(f"\n  Saved: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Evaluation")
    parser.add_argument("--data_path", default="data/raw/med_longmem")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_path", default="results/llm_judge/judge_results.json")
    args = parser.parse_args()

    run_judge_evaluation(
        data_path=args.data_path,
        max_samples=args.max_samples,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
