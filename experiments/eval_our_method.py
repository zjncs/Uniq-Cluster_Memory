"""
eval_our_method.py
==================
在 Med-LongMem v0.1 上评测 Uniq-Cluster Memory 方法，
并与 baseline 进行对比。

评测指标：
    - U-F1(S): 严格 Unique-F1（attribute + time_scope + value 全匹配）
    - U-F1(R): 宽松 Unique-F1（attribute + value 匹配，忽略 time_scope）
    - AttrCov: 属性覆盖率
    - C-F1:    Conflict-F1
    - Latency: 平均处理时间（秒/样本）
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.base_task import UnifiedSample
from benchmarks.med_longmem_task import MedLongMemTask
from src.uniq_cluster_memory.defaults import recommended_pipeline_options
from src.uniq_cluster_memory.pipeline import UniqueClusterMemoryPipeline
from src.uniq_cluster_memory.schema import CanonicalMemory
from src.uniq_cluster_memory.utils.llm_client import ensure_llm_api_key
from evaluation.uniqueness_eval import compute_unique_f1, aggregate_unique_f1
from evaluation.conflict_eval import compute_conflict_f1, aggregate_conflict_f1

DATA_DIR = Path("data/raw/med_longmem")
RESULTS_DIR = Path("results/main_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _sample_to_dialogue(sample: UnifiedSample) -> list[dict]:
    return [
        {
            "turn_id": i,
            "speaker": "patient" if turn.role == "user" else "doctor",
            "text": turn.content,
        }
        for i, turn in enumerate(sample.dialog_history)
    ]


def evaluate_our_method(
    samples: list[UnifiedSample],
    w_struct: float = 0.7,
    use_embedding: bool = True,
) -> dict:
    """在所有样本上运行 Uniq-Cluster Memory，计算聚合指标。"""
    defaults = recommended_pipeline_options("med_longmem")
    top_k = 5
    pipeline = UniqueClusterMemoryPipeline(
        w_struct=w_struct,
        top_k=top_k,
        use_embedding=use_embedding,
        missing_time_scope=defaults["missing_time_scope"],
        max_symptoms_per_scope=defaults["max_symptoms_per_scope"],
    )

    u_metrics_list = []
    c_metrics_list = []
    per_sample_results = []
    total_latency = 0.0

    for sample in samples:
        did = sample.sample_id
        print(f"  [UCM w={w_struct}] {did}...", end=" ", flush=True)
        try:
            dialogue = _sample_to_dialogue(sample)
            t0 = time.time()
            predicted = pipeline.build_memory(
                dialogue=dialogue,
                dialogue_id=did,
                dialogue_date=sample.question_date,
            )
            latency = time.time() - t0
            total_latency += latency

            gt: list[CanonicalMemory] = sample.metadata.get("canonical_gt", [])
            u = compute_unique_f1(predicted, gt)
            c = compute_conflict_f1(predicted, gt)
            u_metrics_list.append(u)
            c_metrics_list.append(c)

            per_sample_results.append({
                "dialogue_id": did,
                "unique_f1_strict": u.f1,
                "unique_f1_relaxed": u.relaxed_f1,
                "attribute_coverage": u.attribute_coverage,
                "redundancy": u.redundancy,
                "conflict_f1": c.f1,
                "n_predicted": u.n_predicted,
                "n_gt": u.n_gt,
                "latency": round(latency, 2),
            })
            print(
                f"U-F1(S)={u.f1:.3f} U-F1(R)={u.relaxed_f1:.3f} "
                f"AttrCov={u.attribute_coverage:.3f} C-F1={c.f1:.3f} "
                f"({latency:.1f}s)"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()

    u_agg = aggregate_unique_f1(u_metrics_list)
    c_agg = aggregate_conflict_f1(c_metrics_list)
    avg_latency = total_latency / len(samples) if samples else 0.0

    return {
        "system": f"UCM_w{int(w_struct*10)}",
        "w_struct": w_struct,
        "unique_f1": u_agg.mean_f1,
        "unique_relaxed_f1": u_agg.mean_relaxed_f1,
        "mean_attribute_coverage": u_agg.mean_attribute_coverage,
        "unique_precision": u_agg.mean_precision,
        "unique_recall": u_agg.mean_recall,
        "mean_redundancy": u_agg.mean_redundancy,
        "mean_coverage": u_agg.mean_coverage,
        "conflict_f1": c_agg.mean_f1,
        "conflict_precision": c_agg.mean_precision,
        "conflict_recall": c_agg.mean_recall,
        "avg_latency": round(avg_latency, 2),
        "n_samples": u_agg.n_samples,
        "config": {
            "dataset": "med_longmem",
            "pipeline_class": "UniqueClusterMemoryPipeline",
            "w_struct": w_struct,
            "top_k": top_k,
            "use_embedding": use_embedding,
            "missing_time_scope": defaults["missing_time_scope"],
            "max_symptoms_per_scope": defaults["max_symptoms_per_scope"],
            "bundle_graph_enabled": True,
        },
        "per_sample": per_sample_results,
    }


def select_comparison_baselines(baseline_results: list[dict]) -> list[dict]:
    """
    过滤掉 oracle / upper-bound 结果，只保留真实可比 baseline。
    """
    return [
        item
        for item in baseline_results
        if "upper_bound" not in item["system"].lower() and "oracle" not in item["system"].lower()
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate UCM on Med-LongMem.")
    parser.add_argument("--data_path", type=str, default=str(DATA_DIR))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--w_struct", type=float, default=0.7)
    parser.add_argument("--no_embedding", action="store_true")
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(RESULTS_DIR / "our_method_eval.json"),
    )
    args = parser.parse_args()
    ensure_llm_api_key()

    task = MedLongMemTask(data_path=args.data_path, max_samples=args.max_samples)
    samples = task.get_samples()

    print(f"\n{'='*75}")
    print(f"  Uniq-Cluster Memory Evaluation on Med-LongMem v0.1")
    print(f"  n={len(samples)} Hard-level samples")
    print(f"{'='*75}\n")

    # 评测我们的方法（默认权重 w_struct=0.7）
    print(f"[1/1] Uniq-Cluster Memory (w_struct={args.w_struct})")
    our_result = evaluate_our_method(
        samples,
        w_struct=args.w_struct,
        use_embedding=not args.no_embedding,
    )

    # 加载之前的 baseline 结果
    baseline_path = RESULTS_DIR / "med_longmem_v01_eval.json"
    baseline_results = []
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        # baseline_data 可能是 list 或 dict
        if isinstance(baseline_data, list):
            items = baseline_data
        elif isinstance(baseline_data, dict):
            items = [v for k, v in baseline_data.items() if k != "metadata"]
        else:
            items = []
        for sys_data in items:
            if isinstance(sys_data, dict) and "system" in sys_data:
                baseline_results.append({
                    "system": sys_data.get("system", "unknown"),
                    "unique_f1": sys_data.get("unique_f1", 0.0),
                    "unique_relaxed_f1": sys_data.get("unique_relaxed_f1", 0.0),
                    "mean_attribute_coverage": sys_data.get("mean_attribute_coverage", 0.0),
                    "mean_redundancy": sys_data.get("mean_redundancy", 0.0),
                    "conflict_f1": sys_data.get("conflict_f1", 0.0),
                    "avg_latency": sys_data.get("avg_latency", 0.0),
                })

    # 打印汇总表
    all_results = baseline_results + [our_result]
    print(f"\n{'='*85}")
    print(f"  Comparative Results on Med-LongMem v0.1 (n={len(samples)}, Hard)")
    print(f"{'='*85}")
    print(f"  [Strict]  attribute + time_scope + value must all match")
    print(f"  [Relaxed] attribute + value match (time_scope ignored)")
    print(f"{'='*85}")
    header = f"{'System':<22} {'U-F1(S)':>9} {'U-F1(R)':>9} {'AttrCov':>9} {'Redund':>8} {'C-F1':>8} {'Latency':>9}"
    print(header)
    print("-" * 85)
    for r in all_results:
        marker = " ◀ OUR METHOD" if "UCM" in r["system"] else ""
        row = (
            f"{r['system']:<22} "
            f"{r['unique_f1']:>9.4f} "
            f"{r.get('unique_relaxed_f1', 0.0):>9.4f} "
            f"{r.get('mean_attribute_coverage', 0.0):>9.4f} "
            f"{r['mean_redundancy']:>8.4f} "
            f"{r['conflict_f1']:>8.4f} "
            f"{r.get('avg_latency', 0.0):>8.1f}s"
            f"{marker}"
        )
        print(row)
    print("=" * 85)

    # 计算相对提升
    comparison_baselines = select_comparison_baselines(baseline_results)
    if comparison_baselines:
        best_baseline_uf1r = max(r.get("unique_relaxed_f1", 0.0) for r in comparison_baselines)
        best_baseline_cf1 = max(r.get("conflict_f1", 0.0) for r in comparison_baselines)
        our_uf1r = our_result.get("unique_relaxed_f1", 0.0)
        our_cf1 = our_result.get("conflict_f1", 0.0)
        our_uf1s = our_result.get("unique_f1", 0.0)

        print(f"\n  Improvement over best baseline:")
        print(f"    U-F1(S): {0.0:.4f} -> {our_uf1s:.4f}  (+{our_uf1s:.4f})")
        if best_baseline_uf1r > 0:
            delta_r = (our_uf1r - best_baseline_uf1r) / best_baseline_uf1r * 100
            print(f"    U-F1(R): {best_baseline_uf1r:.4f} -> {our_uf1r:.4f}  ({delta_r:+.1f}%)")
        if best_baseline_cf1 > 0:
            delta_c = (our_cf1 - best_baseline_cf1) / best_baseline_cf1 * 100
            print(f"    C-F1:    {best_baseline_cf1:.4f} -> {our_cf1:.4f}  ({delta_c:+.1f}%)")
        print()

    # 保存结果
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(our_result, f, indent=2, ensure_ascii=False)
    print(f"  Results saved: {output_path}")


if __name__ == "__main__":
    main()
