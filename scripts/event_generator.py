"""
event_generator.py
==================
Med-LongMem v0.1 事件生成器与对抗注入逻辑。

职责：
    1. 生成一条样本的 RawEvent 时间线（结构化事件序列）。
    2. 向时间线中注入受控的对抗挑战：conflict / coref / update。
    3. 从最终时间线派生 GT 三层：raw_events / canonical_gt / conflict_gt。

设计原则（GT-First）：
    - 所有 GT 完全由此模块的确定性逻辑派生，LLM 不参与 GT 生成。
    - LLM 仅用于第四步：将结构化时间线渲染成自然语言对话。
"""

from __future__ import annotations

import random
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.uniq_cluster_memory.schema import CanonicalMemory, ConflictRecord


# ─── 医学合理区间定义 ────────────────────────────────────────────────────────

MEDICAL_RANGES: Dict[str, Dict] = {
    "blood_glucose": {
        "unit": "mmol/L", "min": 3.9, "max": 16.0,
        "normal_min": 3.9, "normal_max": 6.1,
        "conflict_delta_min": 1.5, "conflict_delta_max": 4.0,
        "policy": "unique", "event_type": "measurement",
    },
    "blood_pressure_sys": {
        "unit": "mmHg", "min": 90, "max": 180,
        "normal_min": 90, "normal_max": 140,
        "conflict_delta_min": 10, "conflict_delta_max": 30,
        "policy": "unique", "event_type": "measurement",
    },
    "blood_pressure_dia": {
        "unit": "mmHg", "min": 60, "max": 110,
        "normal_min": 60, "normal_max": 90,
        "conflict_delta_min": 5, "conflict_delta_max": 20,
        "policy": "unique", "event_type": "measurement",
    },
    "heart_rate": {
        "unit": "bpm", "min": 50, "max": 130,
        "normal_min": 60, "normal_max": 100,
        "conflict_delta_min": 10, "conflict_delta_max": 30,
        "policy": "unique", "event_type": "measurement",
    },
    "body_temperature": {
        "unit": "°C", "min": 36.0, "max": 40.0,
        "normal_min": 36.1, "normal_max": 37.2,
        "conflict_delta_min": 0.5, "conflict_delta_max": 1.5,
        "policy": "unique", "event_type": "measurement",
    },
    "hemoglobin": {
        "unit": "g/L", "min": 80, "max": 180,
        "normal_min": 120, "normal_max": 160,
        "conflict_delta_min": 10, "conflict_delta_max": 30,
        "policy": "unique", "event_type": "measurement",
    },
    "primary_diagnosis": {
        "unit": "", "policy": "unique", "event_type": "diagnosis",
        "values": [
            "Type 2 Diabetes Mellitus",
            "Essential Hypertension",
            "Coronary Artery Disease",
            "Chronic Kidney Disease Stage 3",
            "Hypothyroidism",
        ],
    },
    "medication": {
        "unit": "", "policy": "latest", "event_type": "medication",
        "values": [
            "Metformin 500mg bid",
            "Metformin 1000mg bid",
            "Amlodipine 5mg qd",
            "Amlodipine 10mg qd",
            "Atorvastatin 20mg qn",
            "Atorvastatin 40mg qn",
            "Lisinopril 10mg qd",
            "Lisinopril 20mg qd",
            "Insulin glargine 10U qn",
            "Insulin glargine 16U qn",
        ],
    },
    "symptom": {
        "unit": "", "policy": "append", "event_type": "symptom",
        "values": [
            "dizziness", "polyuria", "blurred vision",
            "lower limb edema", "fatigue", "chest tightness",
            "headache", "palpitations", "nausea",
        ],
    },
}

NUMERIC_ATTRIBUTES = [
    "blood_glucose", "blood_pressure_sys", "blood_pressure_dia",
    "heart_rate", "body_temperature", "hemoglobin",
]
CATEGORICAL_ATTRIBUTES = ["primary_diagnosis", "medication", "symptom"]


# ─── 原始事件数据结构 ────────────────────────────────────────────────────────

@dataclass
class RawEvent:
    """
    原始事件对象。代表对话中某一轮次提及的一个医疗信息单元。
    """
    event_id: str
    dialogue_id: str
    turn_id: int                        # 在对话中的轮次位置（0-indexed）
    speaker: str                        # "patient" | "doctor"
    attribute: str
    value: str
    unit: str
    time_scope: str                     # e.g., "2024-01-15"
    event_type: str                     # "measurement" | "diagnosis" | "medication" | "symptom"
    update_policy: str                  # "unique" | "append" | "latest"
    adversarial_tag: Optional[str] = None   # None | "duplicate" | "conflict" | "coref" | "update"
    coref_target_event_id: Optional[str] = None  # 若为 coref，指向被指代的 event_id

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "dialogue_id": self.dialogue_id,
            "turn_id": self.turn_id,
            "speaker": self.speaker,
            "attribute": self.attribute,
            "value": self.value,
            "unit": self.unit,
            "time_scope": self.time_scope,
            "event_type": self.event_type,
            "update_policy": self.update_policy,
            "adversarial_tag": self.adversarial_tag,
            "coref_target_event_id": self.coref_target_event_id,
        }


# ─── 辅助函数 ────────────────────────────────────────────────────────────────

def _random_date(start_year: int = 2023, span_days: int = 365, rng: random.Random = None) -> str:
    """生成一个随机日期字符串（ISO 格式）。"""
    r = rng if rng is not None else random
    base = date(start_year, 1, 1)
    return (base + timedelta(days=r.randint(0, span_days))).isoformat()


def _sample_numeric_value(attr: str, rng: random.Random) -> str:
    """在医学合理范围内采样一个数值。"""
    meta = MEDICAL_RANGES[attr]
    val = rng.uniform(meta["min"], meta["max"])
    # 保留合适精度
    if meta["unit"] in ("mmol/L", "°C"):
        return f"{val:.1f}"
    return str(int(round(val)))


def _conflict_value(attr: str, original_value: str, rng: random.Random) -> str:
    """
    在原始值的基础上，生成一个在医学合理范围内的冲突值。
    冲突值与原始值的差值在 [conflict_delta_min, conflict_delta_max] 之间。
    """
    meta = MEDICAL_RANGES[attr]
    orig = float(original_value)
    delta = rng.uniform(meta["conflict_delta_min"], meta["conflict_delta_max"])
    direction = rng.choice([-1, 1])
    new_val = orig + direction * delta
    new_val = max(meta["min"], min(meta["max"], new_val))
    if meta["unit"] in ("mmol/L", "°C"):
        return f"{new_val:.1f}"
    return str(int(round(new_val)))


def _geom_span(p: float, min_span: int, max_span: int, rng: random.Random) -> int:
    """
    从几何分布中采样一个跨度値，截断在 [min_span, max_span] 之间。
    Geom(p=0.08) 会产生重尾分布，适合模拟长跨度指代。
    Python 的 random.Random 没有 geometric 方法，用平均分布模拟几何分布采样。
    """
    # 用逆变换法实现几何分布：X = ceil(log(U) / log(1-p))
    import math
    for _ in range(1000):
        u = rng.random()
        if u == 0:
            continue
        span = math.ceil(math.log(u) / math.log(1 - p))
        if min_span <= span <= max_span:
            return span
    # 如果采样失败，返回区间内的随机均匀分布値
    return rng.randint(min_span, max_span)


# ─── 事件生成器主类 ──────────────────────────────────────────────────────────

class EventGenerator:
    """
    Med-LongMem 事件生成器。

    支持三个难度等级：
        Easy:   6-8 事件，0 冲突，0-1 共指，0 更新
        Medium: 8-10 事件，1-2 冲突，1 共指，0-1 更新
        Hard:   10-12 事件，2-3 冲突，1-2 共指，1-2 更新
    """

    DIFFICULTY_CONFIGS = {
        "easy": {
            "n_turns": 20,
            "events_per_dialogue_min": 6,
            "events_per_dialogue_max": 8,
            "n_conflicts_min": 0,
            "n_conflicts_max": 0,
            "n_updates_min": 0,
            "n_updates_max": 0,
            "coref_span_p": 0.08,
            "coref_span_min": 10,
            "coref_span_max": 15,
            "n_corefs_min": 0,
            "n_corefs_max": 1,
        },
        "medium": {
            "n_turns": 20,
            "events_per_dialogue_min": 8,
            "events_per_dialogue_max": 10,
            "n_conflicts_min": 1,
            "n_conflicts_max": 2,
            "n_updates_min": 0,
            "n_updates_max": 1,
            "coref_span_p": 0.08,
            "coref_span_min": 12,
            "coref_span_max": 17,
            "n_corefs_min": 1,
            "n_corefs_max": 1,
        },
        "hard": {
            "n_turns": 20,
            "events_per_dialogue_min": 10,
            "events_per_dialogue_max": 12,
            "n_conflicts_min": 2,
            "n_conflicts_max": 3,
            "n_updates_min": 1,
            "n_updates_max": 2,
            "coref_span_p": 0.08,
            "coref_span_min": 15,
            "coref_span_max": 19,
            "n_corefs_min": 1,
            "n_corefs_max": 2,
        },
    }

    # 向后兼容
    HARD_CONFIG = DIFFICULTY_CONFIGS["hard"]

    def __init__(self, seed: Optional[int] = None, difficulty: str = "hard"):
        self.rng = random.Random(seed)
        self.difficulty = difficulty.lower()
        if self.difficulty not in self.DIFFICULTY_CONFIGS:
            self.difficulty = "hard"

    def generate(self, dialogue_id: str) -> List[RawEvent]:
        """
        生成一条完整的事件时间线。

        Returns:
            按 turn_id 排序的 RawEvent 列表。
        """
        cfg = self.DIFFICULTY_CONFIGS[self.difficulty]
        n_turns = cfg["n_turns"]

        # Step 1: 选择本条样本涉及的属性集合
        n_numeric = self.rng.randint(3, 4)
        n_categorical = self.rng.randint(2, 3)
        chosen_numeric = self.rng.sample(NUMERIC_ATTRIBUTES, n_numeric)
        chosen_categorical = self.rng.sample(CATEGORICAL_ATTRIBUTES, n_categorical)
        chosen_attrs = chosen_numeric + chosen_categorical

        # Step 2: 为每个属性生成一个"基准事件"（分配到前半段对话）
        base_events: List[RawEvent] = []
        used_turns: set = set()
        base_date = _random_date(rng=self.rng)

        for attr in chosen_attrs:
            meta = MEDICAL_RANGES[attr]
            turn_id = self._pick_turn(used_turns, 0, n_turns // 2 - 1)
            used_turns.add(turn_id)

            if attr in NUMERIC_ATTRIBUTES:
                value = _sample_numeric_value(attr, self.rng)
            else:
                value = self.rng.choice(meta["values"])

            evt = RawEvent(
                event_id=f"evt_{dialogue_id}_{len(base_events):03d}",
                dialogue_id=dialogue_id,
                turn_id=turn_id,
                speaker="patient" if meta["event_type"] == "symptom" else "doctor",
                attribute=attr,
                value=value,
                unit=meta.get("unit", ""),
                time_scope=base_date,
                event_type=meta["event_type"],
                update_policy=meta["policy"],
                adversarial_tag=None,
            )
            base_events.append(evt)

        # Step 3: 注入对抗挑战
        adversarial_events: List[RawEvent] = []

        # 3a: 注入 conflict（针对 numeric 属性）
        n_conflicts = self.rng.randint(cfg["n_conflicts_min"], cfg["n_conflicts_max"])
        conflict_candidates = [e for e in base_events if e.attribute in NUMERIC_ATTRIBUTES]
        conflict_targets = self.rng.sample(
            conflict_candidates, min(n_conflicts, len(conflict_candidates))
        )
        for target in conflict_targets:
            conflict_turn = self._pick_turn(used_turns, n_turns // 2, n_turns - 1)
            used_turns.add(conflict_turn)
            conflict_val = _conflict_value(target.attribute, target.value, self.rng)
            adversarial_events.append(RawEvent(
                event_id=f"evt_{dialogue_id}_{len(base_events) + len(adversarial_events):03d}",
                dialogue_id=dialogue_id,
                turn_id=conflict_turn,
                speaker=target.speaker,
                attribute=target.attribute,
                value=conflict_val,
                unit=target.unit,
                time_scope=target.time_scope,   # 同一 time_scope，才构成冲突
                event_type=target.event_type,
                update_policy=target.update_policy,
                adversarial_tag="conflict",
            ))

        # 3b: 注入 update（针对 latest 策略属性，如 medication）
        n_updates = self.rng.randint(cfg["n_updates_min"], cfg["n_updates_max"])
        update_candidates = [e for e in base_events if e.update_policy == "latest"]
        update_targets = self.rng.sample(
            update_candidates, min(n_updates, len(update_candidates))
        )
        for target in update_targets:
            update_turn = self._pick_turn(used_turns, n_turns // 2, n_turns - 1)
            used_turns.add(update_turn)
            meta = MEDICAL_RANGES[target.attribute]
            new_val = self.rng.choice([v for v in meta["values"] if v != target.value])
            adversarial_events.append(RawEvent(
                event_id=f"evt_{dialogue_id}_{len(base_events) + len(adversarial_events):03d}",
                dialogue_id=dialogue_id,
                turn_id=update_turn,
                speaker="doctor",
                attribute=target.attribute,
                value=new_val,
                unit=target.unit,
                time_scope=target.time_scope,
                event_type=target.event_type,
                update_policy=target.update_policy,
                adversarial_tag="update",
            ))

        # 3c: 注入 coref（长跨度指代）
        n_corefs = self.rng.randint(cfg["n_corefs_min"], cfg["n_corefs_max"])
        coref_candidates = [e for e in base_events if e.attribute in NUMERIC_ATTRIBUTES]
        coref_targets = self.rng.sample(
            coref_candidates, min(n_corefs, len(coref_candidates))
        )
        for target in coref_targets:
            span = _geom_span(
                cfg["coref_span_p"],
                cfg["coref_span_min"],
                cfg["coref_span_max"],
                self.rng,
            )
            coref_turn = min(target.turn_id + span, n_turns - 1)
            coref_turn = self._pick_turn(used_turns, coref_turn, n_turns - 1)
            used_turns.add(coref_turn)
            # coref 事件的值与原始值相同（只是用指代表达重新提及）
            adversarial_events.append(RawEvent(
                event_id=f"evt_{dialogue_id}_{len(base_events) + len(adversarial_events):03d}",
                dialogue_id=dialogue_id,
                turn_id=coref_turn,
                speaker="patient",
                attribute=target.attribute,
                value=target.value,
                unit=target.unit,
                time_scope=target.time_scope,
                event_type=target.event_type,
                update_policy=target.update_policy,
                adversarial_tag="coref",
                coref_target_event_id=target.event_id,
            ))

        # Step 4: 合并并排序
        all_events = base_events + adversarial_events
        all_events.sort(key=lambda e: e.turn_id)

        return all_events

    def _pick_turn(self, used: set, lo: int, hi: int) -> int:
        """在 [lo, hi] 范围内随机选一个未被使用的轮次。若全满则允许复用。"""
        candidates = [t for t in range(lo, hi + 1) if t not in used]
        if not candidates:
            candidates = list(range(lo, hi + 1))
        return self.rng.choice(candidates)


# ─── GT 派生逻辑 ─────────────────────────────────────────────────────────────

def derive_canonical_gt(events: List[RawEvent]) -> List[CanonicalMemory]:
    """
    从原始事件时间线派生 canonical_gt。

    规则（与 uniqueness_policies.yaml 对齐）：
    - unique: 同一 (attribute, time_scope) 下，保留最后出现的值；若出现过不同值，标记 conflict_flag。
    - latest: 全局只保留最后出现的值；若出现过不同值，标记 conflict_flag。
    - append: 每个不同的 (attribute, time_scope, value) 都保留一条记录。
    """
    # 按 (attribute, time_scope) 分组，按 turn_id 排序
    from collections import defaultdict
    groups: Dict[Tuple, List[RawEvent]] = defaultdict(list)

    for evt in sorted(events, key=lambda e: e.turn_id):
        if evt.update_policy == "latest":
            key = (evt.attribute, "global")
        else:
            key = (evt.attribute, evt.time_scope)
        groups[key].append(evt)

    canonical: List[CanonicalMemory] = []

    for (attr, scope), evts in groups.items():
        meta = MEDICAL_RANGES.get(attr, {})
        policy = meta.get("policy", "unique")

        if policy == "append":
            # append: 每个不同 value 保留一条
            seen_values = {}
            for evt in evts:
                if evt.value not in seen_values:
                    seen_values[evt.value] = evt
            for val, evt in seen_values.items():
                canonical.append(CanonicalMemory(
                    patient_id=evt.dialogue_id,
                    attribute=attr,
                    value=val,
                    unit=evt.unit,
                    time_scope=scope,
                    confidence=1.0,
                    provenance=[evt.event_id],
                    conflict_flag=False,
                    conflict_history=[],
                    update_policy=policy,
                ))
        else:
            # unique / latest: 保留最后一个值，若有冲突则记录
            conflict_history = []
            seen_values_ordered = []  # [(value, event_id)]
            for evt in evts:
                if not seen_values_ordered or evt.value != seen_values_ordered[-1][0]:
                    seen_values_ordered.append((evt.value, evt.event_id))

            final_value, final_evt_id = seen_values_ordered[-1]
            conflict_flag = len({v for v, _ in seen_values_ordered}) > 1

            if conflict_flag:
                # 记录所有历史冲突
                for i in range(len(seen_values_ordered) - 1):
                    old_val, old_id = seen_values_ordered[i]
                    new_val, new_id = seen_values_ordered[i + 1]
                    conflict_history.append(ConflictRecord(
                        old_value=old_val,
                        new_value=new_val,
                        old_provenance=[old_id],
                        new_provenance=[new_id],
                        conflict_type="value_change",
                        detected_at=evts[-1].time_scope,
                    ))

            canonical.append(CanonicalMemory(
                patient_id=evts[0].dialogue_id,
                attribute=attr,
                value=final_value,
                unit=evts[0].unit,
                time_scope=scope,
                confidence=1.0,
                provenance=[e.event_id for e in evts],
                conflict_flag=conflict_flag,
                conflict_history=conflict_history,
                update_policy=policy,
            ))

    return canonical


def derive_conflict_gt(canonical: List[CanonicalMemory]) -> List[CanonicalMemory]:
    """从 canonical_gt 中提取所有 conflict_flag=True 的记录。"""
    return [m for m in canonical if m.conflict_flag]
