"""
run_main_exp.py
===============
主实验运行脚本。

在 LongMemEval 和 MedDialog 数据集上，对所有 baseline 进行评测，
并将结果保存到 results/main_results/ 目录。

用法：
    cd /path/to/uniq_cluster_memory
    PYTHONPATH=. python3 experiments/run_main_exp.py \\
        --dataset longmemeval \\
        --baseline raw_rag \\
        --max_samples 10 \\
        --output_dir results/main_results

参数说明：
    --dataset: 数据集名称，'longmemeval' 或 'meddialog'。
    --baseline: baseline 名称，'no_memory', 'raw_rag', 'hybrid_rag', 'recursive_summary'。
    --max_samples: 最大样本数量（用于快速调试，None 表示使用全部数据）。
    --output_dir: 结果保存目录。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.longmemeval_task import LongMemEvalTask
from benchmarks.meddialog_task import MedDialogTask
from baselines.no_memory import run_no_memory
from baselines.raw_rag import RawRAGBaseline
from baselines.hybrid_rag import HybridRAGBaseline
from baselines.recursive_summary import RecursiveSummaryBaseline
from evaluation.retrieval_eval import compute_recall_at_k, aggregate_retrieval_metrics
from evaluation.qa_judge_eval import judge_single, aggregate_qa_metrics


# ─── 数据集注册表 ────────────────────────────────────────────────────────────────
DATASET_REGISTRY = {
    "longmemeval": {
        "class": LongMemEvalTask,
        "path": "data/raw/longmemeval/longmemeval_oracle.json",
    },
    "meddialog": {
        "class": MedDialogTask,
        "path": "data/raw/meddialog/meddialog_zh_sample50.json",
    },
}

# ─── Baseline 注册表 ─────────────────────────────────────────────────────────────
BASELINE_REGISTRY = {
    "no_memory": None,          # 无状态函数，特殊处理
    "raw_rag": RawRAGBaseline,
    "hybrid_rag": HybridRAGBaseline,
    "recursive_summary": RecursiveSummaryBaseline,
}


def run_experiment(
    dataset_name: str,
    baseline_name: str,
    max_samples: int | None,
    output_dir: str,
) -> dict:
    """
    在指定数据集上运行指定 baseline 并评测。

    Args:
        dataset_name: 数据集名称。
        baseline_name: baseline 名称。
        max_samples: 最大样本数量。
        output_dir: 结果保存目录。

    Returns:
        包含评测结果的字典。
    """
    print(f"\n{'='*60}")
    print(f"Dataset  : {dataset_name}")
    print(f"Baseline : {baseline_name}")
    print(f"Samples  : {max_samples or 'all'}")
    print(f"{'='*60}")

    # 1. 加载数据集
    dataset_cfg = DATASET_REGISTRY[dataset_name]
    task = dataset_cfg["class"](
        data_path=dataset_cfg["path"],
        max_samples=max_samples,
    )
    samples = task.get_samples()
    print(f"Loaded {len(samples)} samples from {dataset_name}")

    # 2. 初始化 baseline
    if baseline_name == "no_memory":
        baseline = None
    else:
        baseline = BASELINE_REGISTRY[baseline_name]()

    # 3. 运行 baseline 并收集结果
    all_outputs = []
    retrieval_metrics_list = []
    qa_results = []

    start_time = time.time()

    for sample in tqdm(samples, desc=f"Running {baseline_name}"):
        try:
            if baseline_name == "no_memory":
                answer = run_no_memory(sample)
                result = {"answer": answer, "retrieved_has_answer": []}
            else:
                result = baseline.run(sample)

            # 收集检索评测数据（仅对 RAG 类 baseline）
            if baseline_name != "no_memory" and "retrieved_has_answer" in result:
                try:
                    rm = compute_recall_at_k(
                        result["retrieved_has_answer"],
                        total_answer_chunks=result.get("total_answer_chunks"),
                    )
                    retrieval_metrics_list.append(rm)
                except ValueError as e:
                    print(f"\n[WARNING] Skip retrieval metric for {sample.sample_id}: {e}")

            # 收集 QA 评测数据
            qa_result = judge_single(
                sample_id=sample.sample_id,
                question=sample.question,
                gt_answer=sample.answer,
                hypothesis=result["answer"],
            )
            qa_results.append(qa_result)

            all_outputs.append({
                "sample_id": sample.sample_id,
                "question": sample.question,
                "gt_answer": sample.answer,
                "hypothesis": result["answer"],
                "correctness": qa_result.correctness,
                "quality_score": qa_result.quality_score,
                "reasoning": qa_result.reasoning,
            })

        except Exception as e:
            print(f"\n[WARNING] Failed on sample {sample.sample_id}: {e}")
            continue

    elapsed = time.time() - start_time

    # 4. 聚合评测指标
    qa_agg = aggregate_qa_metrics(qa_results)
    retrieval_agg = aggregate_retrieval_metrics(retrieval_metrics_list) if retrieval_metrics_list else {}

    summary = {
        "dataset": dataset_name,
        "baseline": baseline_name,
        "n_samples": len(samples),
        "n_evaluated": len(qa_results),
        "elapsed_seconds": round(elapsed, 2),
        "latency_per_sample": round(elapsed / max(len(qa_results), 1), 2),
        **qa_agg,
        **retrieval_agg,
    }

    # 5. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = f"{output_dir}/{dataset_name}_{baseline_name}"

    with open(f"{output_prefix}_outputs.jsonl", "w", encoding="utf-8") as f:
        for item in all_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(f"{output_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n📊 Results Summary:")
    print(f"   Accuracy      : {summary.get('accuracy', 'N/A')}")
    print(f"   Mean Quality  : {summary.get('mean_quality_score', 'N/A')}")
    print(f"   Recall@K      : {summary.get('mean_recall_at_k', 'N/A')}")
    print(f"   Latency/sample: {summary['latency_per_sample']}s")
    print(f"   Saved to      : {output_prefix}_*.json(l)")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run main experiment for uniq_cluster_memory")
    parser.add_argument("--dataset", choices=list(DATASET_REGISTRY.keys()), default="longmemeval")
    parser.add_argument("--baseline", choices=list(BASELINE_REGISTRY.keys()), default="raw_rag")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="results/main_results")
    args = parser.parse_args()

    summary = run_experiment(
        dataset_name=args.dataset,
        baseline_name=args.baseline,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )
    return summary


if __name__ == "__main__":
    main()
