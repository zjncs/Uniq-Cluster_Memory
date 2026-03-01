"""
med_longmem_task.py
====================
Med-LongMem 对抗性基准数据集的加载器。

数据目录结构（每个样本一个子目录）：
    data/raw/med_longmem/
    ├── dataset_summary.json          # 数据集元信息
    └── medlm_XXXX/
        ├── dialogue.jsonl            # 对话轮次（turn_id, speaker, text）
        ├── raw_events.jsonl          # 原始事件（M1 抽取目标）
        ├── canonical_gt.jsonl        # 规范化记忆 GT（U-F1 评测目标）
        ├── conflict_gt.jsonl         # 冲突记录 GT（C-F1 评测目标）
        └── metadata.json             # 样本元信息

输出格式：
    UnifiedSample，与 LongMemEvalTask / MedDialogTask 完全兼容。
    由于 Med-LongMem 是记忆构建 benchmark（而非 QA benchmark），
    question 字段使用"请总结该患者的完整医疗记录"作为通用问题，
    answer 字段序列化 canonical_gt 列表，供评测脚本直接使用。

GT 三层数据通过 metadata 字段传递：
    sample.metadata["canonical_gt"]  -> List[CanonicalMemory]
    sample.metadata["conflict_gt"]   -> List[CanonicalMemory]（conflict_flag=True）
    sample.metadata["raw_events"]    -> List[dict]
    sample.metadata["difficulty"]    -> str（"easy"/"medium"/"hard"）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import re

from benchmarks.base_task import BaseTask, DialogTurn, UnifiedSample
from src.uniq_cluster_memory.schema import CanonicalMemory


class MedLongMemTask(BaseTask):
    """
    Med-LongMem 对抗性基准数据集加载器。

    Args:
        data_path:   Med-LongMem 数据根目录（包含 dataset_summary.json）。
        max_samples: 最大加载样本数（None 表示加载全部）。
        difficulty:  过滤难度级别（"easy"/"medium"/"hard"/None 表示全部）。
    """

    DEFAULT_QUESTION = (
        "Please summarize this patient's complete medical history, "
        "including all recorded attributes, values, and any detected conflicts."
    )
    _ISO_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    _ISO_RANGE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})$")

    def __init__(
        self,
        data_path: str = "data/raw/med_longmem",
        max_samples: Optional[int] = None,
        difficulty: Optional[str] = None,
    ):
        super().__init__(data_path=data_path, max_samples=max_samples)
        self.difficulty_filter = difficulty

    def load(self) -> list[UnifiedSample]:
        """
        加载 Med-LongMem 数据集，返回 UnifiedSample 列表。

        Returns:
            UnifiedSample 列表，每个样本的 metadata 包含完整 GT 三层数据。
        """
        root = Path(self.data_path)
        summary_path = root / "dataset_summary.json"

        if not summary_path.exists():
            raise FileNotFoundError(
                f"dataset_summary.json not found at {summary_path}. "
                f"Please run scripts/generate_med_longmem.py first."
            )

        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        all_meta = summary["samples"]

        # 难度过滤
        if self.difficulty_filter:
            all_meta = [
                s for s in all_meta
                if s.get("difficulty", "hard") == self.difficulty_filter
            ]

        # 数量限制
        if self.max_samples is not None:
            all_meta = all_meta[: self.max_samples]

        samples = []
        for sample_meta in all_meta:
            sample = self._load_single(root, sample_meta["dialogue_id"])
            if sample is not None:
                samples.append(sample)

        return samples

    def _load_single(self, root: Path, dialogue_id: str) -> Optional[UnifiedSample]:
        """加载单个样本，返回 UnifiedSample。"""
        sample_dir = root / dialogue_id

        if not sample_dir.exists():
            print(f"  [WARN] Sample directory not found: {sample_dir}, skipping.")
            return None

        # ── 加载对话轮次 ──────────────────────────────────────────────────────
        dialogue_turns: list[DialogTurn] = []
        dialogue_path = sample_dir / "dialogue.jsonl"
        if dialogue_path.exists():
            with open(dialogue_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    turn_data = json.loads(line)
                    role = "user" if turn_data.get("speaker") == "patient" else "assistant"
                    dialogue_turns.append(DialogTurn(
                        role=role,
                        content=turn_data.get("text", ""),
                        timestamp=turn_data.get("date"),
                    ))

        # ── 加载 GT 三层 ──────────────────────────────────────────────────────
        canonical_gt = self._load_canonical_gt(sample_dir / "canonical_gt.jsonl")
        conflict_gt  = [m for m in canonical_gt if m.conflict_flag]
        raw_events   = self._load_jsonl(sample_dir / "raw_events.jsonl")

        # ── 加载元信息 ────────────────────────────────────────────────────────
        metadata_path = sample_dir / "metadata.json"
        meta = {}
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                meta = json.load(f)
        dialogue_date = meta.get("dialogue_date") or self._infer_dialogue_date_from_gt(canonical_gt)

        # ── 构造 UnifiedSample ────────────────────────────────────────────────
        # answer 字段：序列化 canonical_gt 列表（供评测脚本直接使用）
        answer_str = json.dumps(
            [m.to_dict() for m in canonical_gt],
            ensure_ascii=False,
        )

        return UnifiedSample(
            sample_id=dialogue_id,
            source="med_longmem",
            question=self.DEFAULT_QUESTION,
            answer=answer_str,
            dialog_history=dialogue_turns,
            question_date=dialogue_date,
            question_type="memory_construction",
            metadata={
                "canonical_gt": canonical_gt,
                "conflict_gt": conflict_gt,
                "raw_events": raw_events,
                "dialogue_date": dialogue_date,
                "difficulty": meta.get("difficulty", "hard"),
                "n_turns": meta.get("n_turns", len(dialogue_turns)),
                "n_canonical_gt": meta.get("n_canonical_gt", len(canonical_gt)),
                "n_conflict_gt": meta.get("n_conflict_gt", len(conflict_gt)),
                "n_conflicts": meta.get("n_conflicts", 0),
                "n_corefs": meta.get("n_corefs", 0),
            },
        )

    @staticmethod
    def _load_canonical_gt(path: Path) -> list[CanonicalMemory]:
        """从 jsonl 文件加载 CanonicalMemory 列表。"""
        if not path.exists():
            return []
        memories = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    memories.append(CanonicalMemory.from_dict(json.loads(line)))
        return memories

    def _infer_dialogue_date_from_gt(self, canonical_gt: list[CanonicalMemory]) -> Optional[str]:
        """
        在 metadata 未提供 dialogue_date 时，从 canonical_gt 的 time_scope 推断参考日期。

        说明：
        - Med-LongMem 的严格评测依赖时间归一化；如果 reference date 缺失，M3 会退化到“今天”，
          导致系统 time_scope 与 GT 日期系统性错位。
        - 该回退逻辑仅用于 benchmark 加载阶段，优先取 day scope，其次取 range 的结束日期。
        """
        day_scopes: list[str] = []
        range_ends: list[str] = []

        for mem in canonical_gt:
            scope = (mem.time_scope or "").strip()
            if self._ISO_DAY_RE.match(scope):
                day_scopes.append(scope)
                continue
            m = self._ISO_RANGE_RE.match(scope)
            if m:
                range_ends.append(m.group(2))

        if day_scopes:
            return max(day_scopes)
        if range_ends:
            return max(range_ends)
        return None

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        """从 jsonl 文件加载字典列表。"""
        if not path.exists():
            return []
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
