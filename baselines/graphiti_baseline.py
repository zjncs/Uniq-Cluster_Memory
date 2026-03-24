"""
baselines/graphiti_baseline.py
===============================
Graphiti (Zep) Baseline 对比。

Graphiti 是目前唯一有双时态冲突管理的通用 memory 系统（arXiv 2025.01）。
本 baseline 使用 Graphiti 的核心思路（双时态 KG + 语义冲突检测）
在 Med-LongMem 上做对比评估。

由于 Graphiti 的完整部署需要 Neo4j，这里实现其核心算法的简化版：
    1. 每条对话事实作为 KG edge
    2. 双时态时间戳 (t_event, t_ingest)
    3. 语义相似度检测冲突
    4. 保留冲突历史而非覆盖

用法：
    PYTHONPATH=. python baselines/graphiti_baseline.py \\
        --data_path data/raw/med_longmem \\
        --output_path results/baselines/graphiti_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.med_longmem_task import MedLongMemTask
from src.uniq_cluster_memory.schema import CanonicalMemory, ConflictRecord
from src.uniq_cluster_memory.m1_event_extraction import MedicalEventExtractor, ExtractedEvent
from evaluation.uniqueness_eval import compute_unique_f1
from evaluation.conflict_eval import compute_conflict_f1
from evaluation.temporal_eval import compute_temporal_metrics


@dataclass
class TemporalEdge:
    """双时态 KG 边（Graphiti 风格）。"""
    subject: str          # patient_id
    predicate: str        # attribute (relation type)
    object_value: str     # value
    unit: str = ""
    t_event: str = ""     # 事件发生时间
    t_ingest: int = 0     # 系统获知时间（turn number）
    t_valid_start: str = ""  # 有效开始
    t_valid_end: str = ""    # 有效结束（空=ongoing）
    confidence: float = 1.0
    provenance: List[int] = field(default_factory=list)
    speaker: str = ""
    is_invalidated: bool = False  # 被更新的旧边标为 invalidated


class GraphitiSimulator:
    """
    Graphiti 核心算法的简化实现。

    简化点：
        - 不需要 Neo4j，用内存字典模拟 KG
        - 语义相似度用简单字符串匹配（Graphiti 用 embedding）
        - 保留双时态模型和冲突检测的核心逻辑
    """

    def __init__(self):
        self.m1 = MedicalEventExtractor()
        self.edges: Dict[str, List[TemporalEdge]] = {}  # key: (subject, predicate)

    def build_memory(
        self,
        dialogue: List[dict],
        dialogue_id: str,
        dialogue_date: Optional[str] = None,
    ) -> List[CanonicalMemory]:
        """运行 Graphiti-style pipeline。"""
        self.edges = {}

        # Step 1: 用 M1 提取事件（与 UCM 共享，保证公平对比）
        events = self.m1.extract(dialogue, dialogue_id)
        if not events:
            return []

        # Step 2: 逐个事件插入双时态 KG
        for evt in sorted(events, key=lambda e: max(e.provenance) if e.provenance else 0):
            self._ingest_event(evt, dialogue_id)

        # Step 3: 从 KG 导出 CanonicalMemory
        return self._export_memories(dialogue_id)

    def _ingest_event(self, evt: ExtractedEvent, dialogue_id: str) -> None:
        """将事件插入双时态 KG，检测冲突。"""
        key = (dialogue_id, evt.attribute)
        t_ingest = max(evt.provenance) if evt.provenance else 0

        new_edge = TemporalEdge(
            subject=dialogue_id,
            predicate=evt.attribute,
            object_value=evt.value,
            unit=evt.unit,
            t_event=evt.time_expr,
            t_ingest=t_ingest,
            t_valid_start=evt.time_expr,
            confidence=evt.confidence,
            provenance=evt.provenance,
            speaker=evt.speaker,
        )

        existing = self.edges.get(key, [])

        # 冲突检测（Graphiti 的核心：语义+时间双重检查）
        for old_edge in existing:
            if old_edge.is_invalidated:
                continue
            if self._values_conflict(old_edge.object_value, new_edge.object_value, evt.attribute):
                # Graphiti 策略：不覆盖，而是将旧边标为 invalidated 并保留
                if evt.update_policy == "latest":
                    old_edge.is_invalidated = True
                    old_edge.t_valid_end = evt.time_expr

        existing.append(new_edge)
        self.edges[key] = existing

    def _values_conflict(self, old_value: str, new_value: str, attribute: str) -> bool:
        """简单的值冲突检测。"""
        old_norm = re.sub(r"\s+", " ", old_value.strip().lower())
        new_norm = re.sub(r"\s+", " ", new_value.strip().lower())
        if old_norm == new_norm:
            return False
        # 数值容差
        try:
            old_num = float(re.sub(r"[^\d.\-]", "", old_value))
            new_num = float(re.sub(r"[^\d.\-]", "", new_value))
            if abs(old_num - new_num) / max(abs(old_num), 0.01) < 0.001:
                return False
        except (ValueError, ZeroDivisionError):
            pass
        return True

    def _export_memories(self, dialogue_id: str) -> List[CanonicalMemory]:
        """从 KG 导出 CanonicalMemory。"""
        memories = []
        for key, edges in self.edges.items():
            active_edges = [e for e in edges if not e.is_invalidated]
            invalidated_edges = [e for e in edges if e.is_invalidated]

            if not active_edges:
                continue

            attribute = key[1]
            policy = "latest" if attribute == "medication" else (
                "append" if attribute == "symptom" else "unique"
            )

            if policy == "append":
                # 每个不同值一条
                seen = {}
                for edge in active_edges:
                    vkey = edge.object_value.strip().lower()
                    if vkey not in seen:
                        seen[vkey] = edge
                for edge in seen.values():
                    memories.append(self._edge_to_memory(edge, dialogue_id, policy, []))
            elif policy == "latest":
                # 取最新的 active edge
                best = max(active_edges, key=lambda e: e.t_ingest)
                conflicts = self._build_conflicts(invalidated_edges, best)
                memories.append(self._edge_to_memory(best, dialogue_id, policy, conflicts))
            else:
                # unique: 按 time_scope 分组
                by_scope = {}
                for edge in active_edges:
                    scope = edge.t_event or "global"
                    if scope not in by_scope or edge.t_ingest > by_scope[scope].t_ingest:
                        by_scope[scope] = edge
                for scope, edge in by_scope.items():
                    scope_invalids = [
                        e for e in invalidated_edges
                        if (e.t_event or "global") == scope
                    ]
                    conflicts = self._build_conflicts(scope_invalids, edge)
                    memories.append(self._edge_to_memory(
                        edge, dialogue_id, policy, conflicts, time_scope=scope
                    ))

        return memories

    def _edge_to_memory(
        self,
        edge: TemporalEdge,
        dialogue_id: str,
        policy: str,
        conflicts: List[ConflictRecord],
        time_scope: Optional[str] = None,
    ) -> CanonicalMemory:
        scope = time_scope or edge.t_event or "global"
        return CanonicalMemory(
            patient_id=dialogue_id,
            attribute=edge.predicate,
            value=edge.object_value,
            unit=edge.unit,
            time_scope=scope,
            confidence=edge.confidence,
            provenance=sorted(set(edge.provenance)),
            conflict_flag=len(conflicts) > 0,
            conflict_history=conflicts,
            update_policy=policy,
        )

    @staticmethod
    def _build_conflicts(
        invalidated: List[TemporalEdge],
        current: TemporalEdge,
    ) -> List[ConflictRecord]:
        conflicts = []
        for old in invalidated:
            conflicts.append(ConflictRecord(
                old_value=old.object_value,
                new_value=current.object_value,
                old_provenance=sorted(set(old.provenance)),
                new_provenance=sorted(set(current.provenance)),
                conflict_type="value_change",
                detected_at=current.t_event,
            ))
        return conflicts


def main():
    parser = argparse.ArgumentParser(description="Graphiti Baseline")
    parser.add_argument("--data_path", default="data/raw/med_longmem")
    parser.add_argument("--output_path", default="results/baselines/graphiti_baseline.json")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    task = MedLongMemTask(data_path=args.data_path, max_samples=args.max_samples)
    samples = task.get_samples()
    print(f"\nLoaded {len(samples)} samples\n")

    sim = GraphitiSimulator()
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
        predicted = sim.build_memory(dialogue, sample.sample_id, sample.question_date)
        latency = time.time() - t0

        gt = sample.metadata.get("canonical_gt", [])
        u = compute_unique_f1(predicted, gt)
        c = compute_conflict_f1(predicted, gt)
        t = compute_temporal_metrics(predicted, gt)

        result = {
            "sample_id": sample.sample_id,
            "n_gt": len(gt),
            "n_predicted": len(predicted),
            "unique_f1_strict": round(u.f1, 4),
            "unique_f1_relaxed": round(u.relaxed_f1, 4),
            "conflict_f1": round(c.f1, 4),
            "temporal_exact_f1": round(t.temporal_f1, 4),
            "interval_iou": round(t.mean_interval_iou, 4),
            "latency": round(latency, 2),
        }
        results.append(result)
        print(
            f"U-F1(S)={result['unique_f1_strict']:.3f}  "
            f"C-F1={result['conflict_f1']:.3f}  "
            f"({latency:.1f}s)"
        )

    n = len(results)
    summary = {
        "baseline": "graphiti_simulated",
        "n_samples": n,
    }
    if n > 0:
        for key in ["unique_f1_strict", "unique_f1_relaxed", "conflict_f1",
                     "temporal_exact_f1", "interval_iou", "latency"]:
            summary[key] = round(sum(r[key] for r in results) / n, 4)
        summary["per_sample"] = results

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Graphiti Baseline Results")
    print(f"{'='*60}")
    if n > 0:
        print(f"  U-F1(S)  : {summary['unique_f1_strict']:.4f}")
        print(f"  U-F1(R)  : {summary['unique_f1_relaxed']:.4f}")
        print(f"  C-F1     : {summary['conflict_f1']:.4f}")
        print(f"  T-F1     : {summary['temporal_exact_f1']:.4f}")
    print(f"\n  Saved: {args.output_path}")


if __name__ == "__main__":
    main()
