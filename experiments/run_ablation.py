"""
run_ablation.py
================
消融实验脚本：在 Med-LongMem v0.1 上系统地评测各模块的贡献。

消融变体设计（共 5 个）：
    full          : 完整 UCM（M1+M2+M3+M4），基准线
    w/o_time      : 去掉 M3 的 Time Grounding（time_scope 保持 "global"）
    w/o_conflict  : 去掉 M3 的 Conflict Detector（conflict_flag 全为 False）
    w/o_m4        : 去掉 M4 压缩（直接输出 M3 结果，不合并冗余）
    w/o_m2        : 去掉 M2 聚类（每个 RawEvent 直接转为 CanonicalMemory）

评测指标：
    U-F1(S)  : Unique-F1 Strict（主指标）
    U-F1(R)  : Unique-F1 Relaxed
    AttrCov  : Attribute Coverage
    C-F1     : Conflict Detection F1
    Latency  : 平均每样本耗时

用法：
    cd /path/to/uniq_cluster_memory
    PYTHONPATH=. python3 experiments/run_ablation.py --max_samples 5
    PYTHONPATH=. python3 experiments/run_ablation.py --max_samples 20 --ablation all
    PYTHONPATH=. python3 experiments/run_ablation.py --ablation w/o_time --max_samples 10

输出：
    results/ablation/ablation_{name}.jsonl   每条样本的详细结果
    results/ablation/ablation_summary.json   所有变体的汇总对比表
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
from src.uniq_cluster_memory.defaults import recommended_pipeline_options
from src.uniq_cluster_memory.schema import CanonicalMemory
from src.uniq_cluster_memory.m1_event_extraction import MedicalEventExtractor, ExtractedEvent
from src.uniq_cluster_memory.m2_clustering import (
    EventClusterer,
    AttributeCluster,
    InformationBundleBuilder,
)
from src.uniq_cluster_memory.m3_uniqueness import UniquenessManager
from src.uniq_cluster_memory.m4_compression import MemoryCompressor
from evaluation.uniqueness_eval import compute_unique_f1
from evaluation.conflict_eval import compute_conflict_f1
from evaluation.temporal_eval import compute_temporal_metrics


# ─── 消融变体定义 ─────────────────────────────────────────────────────────────

ABLATION_CONFIGS = {
    "full": {
        "description": "Complete UCM: M1 + M2 + M3(Time+Conflict) + M4",
        "use_m2":       True,
        "use_time":     True,
        "use_conflict": True,
        "use_m4":       True,
    },
    "w/o_time": {
        "description": "Ablation: Remove Time Grounding from M3 (time_scope stays 'global')",
        "use_m2":       True,
        "use_time":     False,
        "use_conflict": True,
        "use_m4":       True,
    },
    "w/o_conflict": {
        "description": "Ablation: Remove Conflict Detector from M3 (conflict_flag always False)",
        "use_m2":       True,
        "use_time":     True,
        "use_conflict": False,
        "use_m4":       True,
    },
    "w/o_m4": {
        "description": "Ablation: Remove M4 Compression (output M3 result directly)",
        "use_m2":       True,
        "use_time":     True,
        "use_conflict": True,
        "use_m4":       False,
    },
    "w/o_m2": {
        "description": "Ablation: Remove M2 Clustering (each RawEvent -> CanonicalMemory directly)",
        "use_m2":       False,
        "use_time":     True,
        "use_conflict": True,
        "use_m4":       True,
    },
}


# ─── 消融 pipeline 实现 ───────────────────────────────────────────────────────

class AblationPipeline:
    """
    支持消融配置的灵活 pipeline。
    通过 use_m2 / use_time / use_conflict / use_m4 四个开关控制各模块的启用状态。
    """

    def __init__(self, config: dict, use_embedding: bool = True):
        self.config = config
        defaults = recommended_pipeline_options("med_longmem")
        self.missing_time_scope = defaults["missing_time_scope"]
        self.max_symptoms_per_scope = defaults["max_symptoms_per_scope"]
        self.use_embedding = use_embedding
        self.m1 = MedicalEventExtractor()
        self.m2 = EventClusterer(use_embedding=use_embedding) if config["use_m2"] else None
        self.m25 = InformationBundleBuilder()
        self.m4 = MemoryCompressor() if config["use_m4"] else None

    def build_memory(
        self,
        dialogue: List[dict],
        dialogue_id: str,
        dialogue_date: Optional[str] = None,
    ) -> List[CanonicalMemory]:
        """运行消融配置下的 pipeline，返回 CanonicalMemory 列表。"""

        # M1: 事件抽取（始终运行）
        events: List[ExtractedEvent] = self.m1.extract(dialogue, dialogue_id)
        if not events:
            return []

        # M2: 聚类规范化（可消融）
        if self.config["use_m2"] and self.m2 is not None:
            clusters = self.m2.cluster(events, dialogue_id)
        else:
            # w/o_m2：跳过语义聚类，但仍构造 AttributeCluster 契约
            clusters = self._build_identity_clusters(events, dialogue_id)

        # M2.5: 信息团构建，供 M3 做 bundle-aware 决策
        bundle_graph = self.m25.build(events, dialogue_id)

        # M3: Time Grounding + Uniqueness Manager（可部分消融）
        m3 = UniquenessManager(
            dialogue_date=dialogue_date,
            enable_time_grounding=self.config["use_time"],
            enable_conflict_detection=self.config["use_conflict"],
            missing_time_scope=self.missing_time_scope,
            max_symptoms_per_scope=self.max_symptoms_per_scope,
        )
        memories: List[CanonicalMemory] = m3.process(
            clusters,
            patient_id=dialogue_id,
            bundle_graph=bundle_graph,
        )

        # M4: 压缩（可消融）
        if self.config["use_m4"] and self.m4 is not None:
            memories = self.m4.compress(memories, patient_id=dialogue_id)

        return memories

    @staticmethod
    def _build_identity_clusters(
        events: List[ExtractedEvent],
        dialogue_id: str,
    ) -> List[AttributeCluster]:
        """
        构造“恒等聚类”：
        - 不做别名归一化与跨事件合并（即移除 M2）
        - 但保持 M3 需要的 AttributeCluster 输入契约
        """
        clusters: List[AttributeCluster] = []
        for idx, evt in enumerate(events, 1):
            policy = (evt.update_policy or "unique").strip().lower()
            if policy not in {"unique", "latest", "append"}:
                policy = "unique"
            clusters.append(
                AttributeCluster(
                    cluster_id=f"identity_{dialogue_id}_{idx:04d}",
                    canonical_attribute=evt.attribute,
                    update_policy=policy,
                    events=[evt],
                )
            )
        return clusters


# ─── 单样本运行 ───────────────────────────────────────────────────────────────

def run_single(
    sample: UnifiedSample,
    pipeline: AblationPipeline,
) -> dict:
    """对单个样本运行消融 pipeline 并计算指标。"""
    dialogue = [
        {
            "turn_id": i,
            "speaker": "patient" if turn.role == "user" else "doctor",
            "text": turn.content,
        }
        for i, turn in enumerate(sample.dialog_history)
    ]

    t0 = time.time()
    predicted: List[CanonicalMemory] = pipeline.build_memory(
        dialogue=dialogue,
        dialogue_id=sample.sample_id,
        dialogue_date=sample.question_date,
    )
    latency = time.time() - t0

    gt: List[CanonicalMemory] = sample.metadata.get("canonical_gt", [])
    u = compute_unique_f1(predicted, gt)
    c = compute_conflict_f1(predicted, gt)
    t = compute_temporal_metrics(predicted, gt)

    return {
        "sample_id":          sample.sample_id,
        "n_gt":               len(gt),
        "n_predicted":        len(predicted),
        "unique_f1_strict":   round(u.f1, 4),
        "unique_f1_relaxed":  round(u.relaxed_f1, 4),
        "attribute_coverage": round(u.attribute_coverage, 4),
        "redundancy":         round(u.redundancy, 4),
        "conflict_f1":        round(c.f1, 4),
        "temporal_exact_f1":  round(t.temporal_f1, 4),
        "interval_iou":       round(t.mean_interval_iou, 4),
        "latency":            round(latency, 2),
    }


# ─── 单个消融变体的完整运行 ───────────────────────────────────────────────────

def run_ablation_variant(
    ablation_name: str,
    samples: List[UnifiedSample],
    output_dir: str,
    use_embedding: bool = True,
) -> dict:
    """
    运行单个消融变体，保存详细结果，返回汇总指标。

    Args:
        ablation_name: 消融变体名称（ABLATION_CONFIGS 的键）。
        samples:       UnifiedSample 列表。
        output_dir:    结果保存目录。
        use_embedding: 是否使用 Embedding。

    Returns:
        包含平均指标的汇总字典。
    """
    config = ABLATION_CONFIGS[ablation_name]
    pipeline = AblationPipeline(config, use_embedding=use_embedding)

    print(f"\n  [{ablation_name}] {config['description']}")
    print(f"  {'─'*55}")

    per_sample_results = []
    for i, sample in enumerate(samples, 1):
        print(f"    [{i}/{len(samples)}] {sample.sample_id}...", end=" ", flush=True)
        try:
            result = run_single(sample, pipeline)
            per_sample_results.append(result)
            print(
                f"U-F1(S)={result['unique_f1_strict']:.3f}  "
                f"U-F1(R)={result['unique_f1_relaxed']:.3f}  "
                f"C-F1={result['conflict_f1']:.3f}  "
                f"T-F1={result['temporal_exact_f1']:.3f}  "
                f"IoU={result['interval_iou']:.3f}  "
                f"({result['latency']:.1f}s)"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    # 保存每条样本的详细结果
    os.makedirs(output_dir, exist_ok=True)
    safe_name = ablation_name.replace("/", "_")
    detail_path = os.path.join(output_dir, f"ablation_{safe_name}.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in per_sample_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 计算平均指标
    n = len(per_sample_results)
    if n == 0:
        return {"ablation": ablation_name, "n_samples": 0}

    summary = {
        "ablation":           ablation_name,
        "description":        config["description"],
        "n_samples":          n,
        "unique_f1_strict":   round(sum(r["unique_f1_strict"] for r in per_sample_results) / n, 4),
        "unique_f1_relaxed":  round(sum(r["unique_f1_relaxed"] for r in per_sample_results) / n, 4),
        "attribute_coverage": round(sum(r["attribute_coverage"] for r in per_sample_results) / n, 4),
        "redundancy":         round(sum(r["redundancy"] for r in per_sample_results) / n, 4),
        "conflict_f1":        round(sum(r["conflict_f1"] for r in per_sample_results) / n, 4),
        "temporal_exact_f1":  round(sum(r["temporal_exact_f1"] for r in per_sample_results) / n, 4),
        "interval_iou":       round(sum(r["interval_iou"] for r in per_sample_results) / n, 4),
        "avg_latency":        round(sum(r["latency"] for r in per_sample_results) / n, 2),
        "detail_file":        detail_path,
        "config": {
            "dataset": "med_longmem",
            "use_embedding": use_embedding,
            "missing_time_scope": pipeline.missing_time_scope,
            "max_symptoms_per_scope": pipeline.max_symptoms_per_scope,
            "bundle_graph_enabled": True,
            "use_m2": config["use_m2"],
            "use_time": config["use_time"],
            "use_conflict": config["use_conflict"],
            "use_m4": config["use_m4"],
        },
    }

    print(f"\n  Summary [{ablation_name}]:")
    print(f"    U-F1(S)  : {summary['unique_f1_strict']:.4f}")
    print(f"    U-F1(R)  : {summary['unique_f1_relaxed']:.4f}")
    print(f"    AttrCov  : {summary['attribute_coverage']:.4f}")
    print(f"    C-F1     : {summary['conflict_f1']:.4f}")
    print(f"    T-F1     : {summary['temporal_exact_f1']:.4f}")
    print(f"    IoU      : {summary['interval_iou']:.4f}")
    print(f"    Latency  : {summary['avg_latency']:.1f}s/sample")

    return summary


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ablation experiments on Med-LongMem v0.1"
    )
    parser.add_argument(
        "--ablation",
        default="all",
        choices=list(ABLATION_CONFIGS.keys()) + ["all"],
        help="Ablation variant to run (default: all)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=5,
        help="Maximum number of samples to process (default: 5)",
    )
    parser.add_argument(
        "--data_path", default="data/raw/med_longmem",
        help="Path to Med-LongMem dataset (default: data/raw/med_longmem)",
    )
    parser.add_argument(
        "--output_dir", default="results/ablation",
        help="Output directory for results (default: results/ablation)",
    )
    embedding_group = parser.add_mutually_exclusive_group()
    embedding_group.add_argument(
        "--use_embedding",
        dest="use_embedding",
        action="store_true",
        help="Enable semantic embedding in M2/M5.",
    )
    embedding_group.add_argument(
        "--no_embedding",
        dest="use_embedding",
        action="store_false",
        help="Disable semantic embedding in M2/M5.",
    )
    parser.set_defaults(use_embedding=True)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # 确定要运行的消融变体
    if args.ablation == "all":
        ablation_names = list(ABLATION_CONFIGS.keys())
    else:
        ablation_names = [args.ablation]

    print(f"\n{'='*65}")
    print(f"  Ablation Experiments on Med-LongMem")
    print(f"  Variants : {', '.join(ablation_names)}")
    print(f"  Samples  : {args.max_samples}")
    print(f"  Embedding: {'enabled' if args.use_embedding else 'disabled'}")
    print(f"{'='*65}")

    # 加载数据集（只加载一次，所有变体共用）
    task = MedLongMemTask(
        data_path=args.data_path,
        max_samples=args.max_samples,
    )
    samples = task.get_samples()
    print(f"\n  Loaded {len(samples)} samples from {args.data_path}\n")

    # 逐个运行消融变体
    all_summaries = []
    for name in ablation_names:
        summary = run_ablation_variant(
            ablation_name=name,
            samples=samples,
            output_dir=args.output_dir,
            use_embedding=args.use_embedding,
        )
        all_summaries.append(summary)

    # 保存汇总对比表
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    # 打印最终对比表
    print(f"\n{'='*65}")
    print(f"  Final Ablation Comparison Table")
    print(f"{'='*65}")
    header = (
        f"  {'Variant':<18} {'U-F1(S)':>8} {'U-F1(R)':>8} "
        f"{'AttrCov':>8} {'C-F1':>8} {'T-F1':>8} {'IoU':>8} {'Latency':>8}"
    )
    print(header)
    print(f"  {'─'*63}")
    for s in all_summaries:
        marker = " ◀ full" if s["ablation"] == "full" else ""
        print(
            f"  {s['ablation']:<18} "
            f"{s.get('unique_f1_strict', 0):>8.4f} "
            f"{s.get('unique_f1_relaxed', 0):>8.4f} "
            f"{s.get('attribute_coverage', 0):>8.4f} "
            f"{s.get('conflict_f1', 0):>8.4f} "
            f"{s.get('temporal_exact_f1', 0):>8.4f} "
            f"{s.get('interval_iou', 0):>8.4f} "
            f"{s.get('avg_latency', 0):>7.1f}s"
            f"{marker}"
        )
    print(f"{'='*65}")
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
