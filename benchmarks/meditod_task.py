"""
benchmarks/meditod_task.py
===========================
MediTOD 数据集适配器（EMNLP 2024）。

MediTOD 是目前唯一专门针对医疗对话做时序标注的数据集：
- 症状起始（onset）、进展（progression）、严重度（severity）
- 结构化 slot 标注
- 多轮医患对话

数据集链接：https://github.com/UCSC-NLP/MediTOD

适配策略：
    将 MediTOD 的 slot 标注转为 CanonicalMemory 格式进行评估，
    重点关注 temporal slot（onset, duration, frequency）的准确性。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from benchmarks.base_task import UnifiedSample, DialogTurn
from src.uniq_cluster_memory.schema import CanonicalMemory


# MediTOD slot 到 CanonicalMemory attribute 的映射
MEDITOD_SLOT_MAP = {
    "symptom": "symptom",
    "symptom_name": "symptom",
    "medication": "medication",
    "medication_name": "medication",
    "diagnosis": "primary_diagnosis",
    "condition": "primary_diagnosis",
    "test": "measurement",
    "test_name": "measurement",
}

# MediTOD temporal slot 名
TEMPORAL_SLOTS = {"onset", "duration", "frequency", "severity", "progression"}


class MediTODTask:
    """MediTOD 数据集加载器。"""

    def __init__(
        self,
        data_path: str = "data/raw/meditod",
        split: str = "test",
        max_samples: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.max_samples = max_samples

    def get_samples(self) -> List[UnifiedSample]:
        """加载 MediTOD 样本并转为统一格式。"""
        data_file = self.data_path / f"{self.split}.json"
        if not data_file.exists():
            # 尝试其他命名
            for alt in [f"{self.split}_data.json", f"meditod_{self.split}.json", "data.json"]:
                alt_path = self.data_path / alt
                if alt_path.exists():
                    data_file = alt_path
                    break
            else:
                print(f"WARNING: MediTOD data not found at {self.data_path}")
                print(f"Please download from https://github.com/UCSC-NLP/MediTOD")
                return []

        with open(data_file, encoding="utf-8") as f:
            raw_data = json.load(f)

        samples = []
        items = raw_data if isinstance(raw_data, list) else raw_data.get("dialogues", raw_data.get("data", []))

        for idx, item in enumerate(items):
            if self.max_samples and idx >= self.max_samples:
                break

            dialogue_id = item.get("dialogue_id", item.get("id", f"meditod_{idx:04d}"))

            # 解析对话轮次
            turns = []
            raw_turns = item.get("turns", item.get("dialogue", item.get("utterances", [])))
            for t_idx, turn in enumerate(raw_turns):
                if isinstance(turn, dict):
                    role = turn.get("speaker", turn.get("role", "patient"))
                    text = turn.get("text", turn.get("utterance", turn.get("content", "")))
                elif isinstance(turn, str):
                    role = "patient" if t_idx % 2 == 0 else "doctor"
                    text = turn
                else:
                    continue
                turns.append(DialogTurn(role=role, content=text))

            # 解析 slot 标注为 GT
            canonical_gt = self._extract_gt(item, dialogue_id)

            samples.append(UnifiedSample(
                sample_id=dialogue_id,
                source="meditod",
                question="Extract structured medical memories from this dialogue.",
                answer=json.dumps([m.to_dict() for m in canonical_gt]),
                dialog_history=turns,
                question_date=None,
                question_type="memory_construction",
                metadata={
                    "canonical_gt": canonical_gt,
                    "conflict_gt": [],
                    "raw_slots": item.get("slots", item.get("annotations", {})),
                    "difficulty": "medium",
                    "n_turns": len(turns),
                    "n_canonical_gt": len(canonical_gt),
                },
            ))

        return samples

    def _extract_gt(self, item: dict, dialogue_id: str) -> List[CanonicalMemory]:
        """从 MediTOD slot 标注提取 CanonicalMemory GT。"""
        memories = []
        slots = item.get("slots", item.get("annotations", item.get("belief_state", {})))

        if isinstance(slots, dict):
            for slot_name, slot_value in slots.items():
                if not slot_value or slot_value in ("none", "None", "N/A"):
                    continue

                attribute = self._map_attribute(slot_name)
                if not attribute:
                    continue

                # 解析 temporal slots
                time_scope = "global"
                onset = slots.get("onset", slots.get(f"{slot_name}_onset", ""))
                if onset and onset not in ("none", "None", "N/A"):
                    time_scope = onset

                value = str(slot_value) if not isinstance(slot_value, str) else slot_value
                policy = "append" if attribute == "symptom" else "unique"

                memories.append(CanonicalMemory(
                    patient_id=dialogue_id,
                    attribute=attribute,
                    value=value,
                    time_scope=time_scope,
                    update_policy=policy,
                ))

        elif isinstance(slots, list):
            for slot in slots:
                if isinstance(slot, dict):
                    slot_name = slot.get("slot", slot.get("name", ""))
                    slot_value = slot.get("value", "")
                    if not slot_value:
                        continue
                    attribute = self._map_attribute(slot_name)
                    if attribute:
                        memories.append(CanonicalMemory(
                            patient_id=dialogue_id,
                            attribute=attribute,
                            value=str(slot_value),
                            time_scope="global",
                            update_policy="append" if attribute == "symptom" else "unique",
                        ))

        return memories

    @staticmethod
    def _map_attribute(slot_name: str) -> Optional[str]:
        """将 MediTOD slot 名映射到 CanonicalMemory attribute。"""
        normalized = slot_name.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in MEDITOD_SLOT_MAP:
            return MEDITOD_SLOT_MAP[normalized]
        # 模糊匹配
        for key, val in MEDITOD_SLOT_MAP.items():
            if key in normalized or normalized in key:
                return val
        return None
