"""
run_pipeline.py
================
Uniq-Cluster Memory 完整 pipeline 的命令行入口。

用法：
    # 在 Med-LongMem v0.1 上运行（默认 5 条）
    cd /path/to/uniq_cluster_memory
    PYTHONPATH=. python3 scripts/run_pipeline.py --dataset med_longmem --max_samples 5

    # 在 LongMemEval 上运行
    PYTHONPATH=. python3 scripts/run_pipeline.py --dataset longmemeval --max_samples 10

    # 指定 w_struct 权重
    PYTHONPATH=. python3 scripts/run_pipeline.py --dataset med_longmem --w_struct 0.8

    # 禁用 Embedding（更快，精度略低）
    PYTHONPATH=. python3 scripts/run_pipeline.py --dataset med_longmem --no_embedding

输出：
    results/main_results/pipeline_{dataset}_w{w}.jsonl
    每行一个 JSON 对象，包含：
        - sample_id, source, question, n_turns
        - n_predicted, predicted_memories
        - unique_f1_strict, unique_f1_relaxed, conflict_f1（仅 med_longmem）
        - latency
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.longmemeval_task import LongMemEvalTask
from benchmarks.meddialog_task import MedDialogTask
from benchmarks.med_longmem_task import MedLongMemTask
from benchmarks.base_task import UnifiedSample
from src.uniq_cluster_memory.pipeline import UniqueClusterMemoryPipeline
from src.uniq_cluster_memory.schema import CanonicalMemory
from evaluation.uniqueness_eval import compute_unique_f1
from evaluation.conflict_eval import compute_conflict_f1


# ─── 数据集注册表 ─────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "med_longmem": {
        "class": MedLongMemTask,
        "data_path": "data/raw/med_longmem",
        "has_gt": True,   # 有 CanonicalMemory GT，可计算 U-F1、C-F1
    },
    "longmemeval": {
        "class": LongMemEvalTask,
        "data_path": "data/raw/longmemeval/longmemeval_oracle.json",
        "has_gt": False,  # LongMemEval 使用 QA 评测
    },
    "meddialog": {
        "class": MedDialogTask,
        "data_path": "data/raw/meddialog/meddialog_zh_sample50.json",
        "has_gt": False,
    },
}


# ─── 单样本 pipeline 运行 ─────────────────────────────────────────────────────

def run_single_sample(
    sample: UnifiedSample,
    pipeline: UniqueClusterMemoryPipeline,
    has_gt: bool,
) -> dict:
    """
    对单个样本运行完整 M1-M4 pipeline，并计算评测指标（如有 GT）。

    Args:
        sample:   UnifiedSample 对象。
        pipeline: 已初始化的 UniqueClusterMemoryPipeline。
        has_gt:   是否有 CanonicalMemory GT（决定是否计算 U-F1/C-F1）。

    Returns:
        包含预测结果和评测指标的字典。
    """
    # 将 UnifiedSample 的 dialog_history 转换为 pipeline 所需格式
    dialogue = [
        {
            "turn_id": i,
            "speaker": "patient" if turn.role == "user" else "doctor",
            "text": turn.content,
        }
        for i, turn in enumerate(sample.dialog_history)
    ]

    t0 = time.time()
    predicted: list[CanonicalMemory] = pipeline.build_memory(
        dialogue=dialogue,
        dialogue_id=sample.sample_id,
        dialogue_date=sample.question_date,
    )
    latency = time.time() - t0

    result = {
        "sample_id":          sample.sample_id,
        "source":             sample.source,
        "question":           sample.question,
        "question_date":      sample.question_date,
        "n_turns":            len(dialogue),
        "n_predicted":        len(predicted),
        "predicted_memories": [m.to_dict() for m in predicted],
        "latency":            round(latency, 2),
    }

    # 如果有 GT，计算评测指标
    if has_gt and "canonical_gt" in sample.metadata:
        gt: list[CanonicalMemory] = sample.metadata["canonical_gt"]
        result["n_gt"] = len(gt)

        u = compute_unique_f1(predicted, gt)
        c = compute_conflict_f1(predicted, gt)

        result.update({
            "unique_f1_strict":   round(u.f1, 4),
            "unique_f1_relaxed":  round(u.relaxed_f1, 4),
            "unique_precision":   round(u.precision, 4),
            "unique_recall":      round(u.recall, 4),
            "attribute_coverage": round(u.attribute_coverage, 4),
            "redundancy":         round(u.redundancy, 4),
            "conflict_f1":        round(c.f1, 4),
            "conflict_precision": round(c.precision, 4),
            "conflict_recall":    round(c.recall, 4),
        })

    return result


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run Uniq-Cluster Memory Pipeline (M1-M5)"
    )
    parser.add_argument(
        "--dataset", default="med_longmem",
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to run on (default: med_longmem)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=5,
        help="Maximum number of samples to process (default: 5)",
    )
    parser.add_argument(
        "--w_struct", type=float, default=0.7,
        help="Structural retrieval weight for M5 (default: 0.7)",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of memories to retrieve in M5 (default: 5)",
    )
    parser.add_argument(
        "--output_dir", default="results/main_results",
        help="Output directory for results (default: results/main_results)",
    )
    parser.add_argument(
        "--no_embedding", action="store_true",
        help="Disable semantic embedding in M5 (faster, lower quality)",
    )
    parser.add_argument(
        "--missing_time_scope",
        choices=["auto", "global", "reference_day"],
        default="auto",
        help=(
            "How to handle missing/implicit time expressions in M3 "
            "(default: auto; med_longmem=reference_day, others=global)"
        ),
    )
    parser.add_argument(
        "--max_symptoms_per_scope",
        type=int,
        default=-1,
        help=(
            "Cap number of symptom memories per time scope. "
            "Default -1 means auto (med_longmem=1, others=unlimited)."
        ),
    )
    parser.add_argument(
        "--enable_qdrant", action="store_true",
        help="Enable persistent vector storage in Qdrant",
    )
    parser.add_argument(
        "--enable_neo4j", action="store_true",
        help="Enable persistent graph storage in Neo4j",
    )
    parser.add_argument(
        "--persist_to_stores", action="store_true",
        help="Persist M4 canonical memories to enabled stores",
    )
    parser.add_argument(
        "--qdrant_url", type=str, default=None,
        help="Qdrant URL (default: env QDRANT_URL or http://localhost:6333)",
    )
    parser.add_argument(
        "--qdrant_api_key", type=str, default=None,
        help="Qdrant API key (default: env QDRANT_API_KEY)",
    )
    parser.add_argument(
        "--qdrant_collection", type=str, default="medical_memory",
        help="Qdrant collection name (default: medical_memory)",
    )
    parser.add_argument(
        "--neo4j_uri", type=str, default=None,
        help="Neo4j URI (default: env NEO4J_URI or bolt://localhost:7687)",
    )
    parser.add_argument(
        "--neo4j_user", type=str, default=None,
        help="Neo4j username (default: env NEO4J_USER or neo4j)",
    )
    parser.add_argument(
        "--neo4j_password", type=str, default=None,
        help="Neo4j password (default: env NEO4J_PASSWORD)",
    )
    parser.add_argument(
        "--neo4j_database", type=str, default="neo4j",
        help="Neo4j database (default: neo4j)",
    )
    args = parser.parse_args()

    dataset_cfg = DATASET_REGISTRY[args.dataset]
    has_gt = dataset_cfg["has_gt"]
    if args.missing_time_scope == "auto":
        missing_time_scope = "reference_day" if args.dataset == "med_longmem" else "global"
    else:
        missing_time_scope = args.missing_time_scope
    if args.max_symptoms_per_scope < 0:
        max_symptoms_per_scope = 1 if args.dataset == "med_longmem" else None
    else:
        max_symptoms_per_scope = args.max_symptoms_per_scope

    print(f"\n{'='*65}")
    print(f"  Uniq-Cluster Memory Pipeline")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Samples  : {args.max_samples}")
    print(f"  w_struct : {args.w_struct}")
    print(f"  Embedding: {'disabled' if args.no_embedding else 'enabled (MiniLM)'}")
    print(f"  MissingTS: {missing_time_scope}")
    print(f"  SymptomCap: {max_symptoms_per_scope if max_symptoms_per_scope is not None else 'none'}")
    print(f"  Qdrant   : {'enabled' if args.enable_qdrant else 'disabled'}")
    print(f"  Neo4j    : {'enabled' if args.enable_neo4j else 'disabled'}")
    print(f"  Persist  : {'enabled' if args.persist_to_stores else 'disabled'}")
    print(f"{'='*65}\n")

    # ── 加载数据集 ────────────────────────────────────────────────────────────
    TaskClass = dataset_cfg["class"]
    task = TaskClass(
        data_path=dataset_cfg["data_path"],
        max_samples=args.max_samples,
    )
    samples = task.get_samples()
    print(f"  Loaded {len(samples)} samples from {args.dataset}\n")

    # ── 初始化 pipeline ───────────────────────────────────────────────────────
    pipeline = UniqueClusterMemoryPipeline(
        w_struct=args.w_struct,
        top_k=args.top_k,
        use_embedding=not args.no_embedding,
        missing_time_scope=missing_time_scope,
        max_symptoms_per_scope=max_symptoms_per_scope,
        enable_qdrant=args.enable_qdrant,
        enable_neo4j=args.enable_neo4j,
        persist_to_stores=args.persist_to_stores,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_collection=args.qdrant_collection,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
    )

    # ── 逐样本运行 ────────────────────────────────────────────────────────────
    all_results = []
    total_latency = 0.0

    for i, sample in enumerate(samples, 1):
        print(f"  [{i}/{len(samples)}] {sample.sample_id}...", end=" ", flush=True)
        try:
            result = run_single_sample(sample, pipeline, has_gt)
            all_results.append(result)
            total_latency += result["latency"]

            if has_gt:
                print(
                    f"U-F1(S)={result.get('unique_f1_strict', 0):.3f}  "
                    f"U-F1(R)={result.get('unique_f1_relaxed', 0):.3f}  "
                    f"C-F1={result.get('conflict_f1', 0):.3f}  "
                    f"({result['latency']:.1f}s)"
                )
            else:
                print(
                    f"n_predicted={result['n_predicted']}  "
                    f"({result['latency']:.1f}s)"
                )
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ── 汇总指标 ──────────────────────────────────────────────────────────────
    n = len(all_results)
    avg_latency = total_latency / n if n > 0 else 0.0

    print(f"\n{'='*65}")
    if has_gt and n > 0:
        mean_uf1s = sum(r.get("unique_f1_strict", 0) for r in all_results) / n
        mean_uf1r = sum(r.get("unique_f1_relaxed", 0) for r in all_results) / n
        mean_cf1  = sum(r.get("conflict_f1", 0) for r in all_results) / n
        mean_cov  = sum(r.get("attribute_coverage", 0) for r in all_results) / n
        print(f"  Summary ({n} samples):")
        print(f"    U-F1(S)  : {mean_uf1s:.4f}")
        print(f"    U-F1(R)  : {mean_uf1r:.4f}")
        print(f"    AttrCov  : {mean_cov:.4f}")
        print(f"    C-F1     : {mean_cf1:.4f}")
        print(f"    Latency  : {avg_latency:.1f}s/sample")
    else:
        mean_pred = sum(r.get("n_predicted", 0) for r in all_results) / n if n > 0 else 0
        print(f"  Summary ({n} samples):")
        print(f"    Avg predicted memories : {mean_pred:.1f}")
        print(f"    Avg latency            : {avg_latency:.1f}s/sample")
    print(f"{'='*65}\n")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"pipeline_{args.dataset}_w{int(args.w_struct * 10)}.jsonl",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pipeline.close()

    print(f"  Results saved: {output_path}")


if __name__ == "__main__":
    main()
