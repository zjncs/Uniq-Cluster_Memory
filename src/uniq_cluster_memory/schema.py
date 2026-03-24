"""
schema.py
=========
Canonical Memory 核心数据结构定义（关系 + 时序增强版）。

兼容目标：
1. 保持原有 attribute/value/time_scope 接口不变，避免现有评测链路断裂。
2. 新增关系与时间结构化字段，支持时间推理模块按时间轴查询。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple


ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_WEEK_RE = re.compile(r"^(\d{4})-W(\d{2})$", re.IGNORECASE)
ISO_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")
ISO_YEAR_RE = re.compile(r"^(19\d{2}|20\d{2})$")
ISO_RANGE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})$")

DOSAGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g|ml|u|iu|units?|片|粒|袋|滴|丸)",
    re.IGNORECASE,
)
FREQUENCY_PATTERNS = [
    (re.compile(r"\b(qd|daily|once\s+(a|per)\s+day|once\s+daily)\b", re.IGNORECASE), "daily"),
    (re.compile(r"\b(bid|twice\s+(a|per)\s+day|two\s+times\s+daily)\b", re.IGNORECASE), "bid"),
    (re.compile(r"\b(tid|three\s+times\s+daily)\b", re.IGNORECASE), "tid"),
    (re.compile(r"\b(qid|four\s+times\s+daily)\b", re.IGNORECASE), "qid"),
    (re.compile(r"\bq(\d+)h\b", re.IGNORECASE), "q{n}h"),
    (re.compile(r"每天\s*(\d+)\s*次"), "q{n}/day"),
    (re.compile(r"每周\s*(\d+)\s*次"), "q{n}/week"),
]
TIME_OF_DAY_CUES = {
    "before breakfast": "before_breakfast",
    "after breakfast": "after_breakfast",
    "before lunch": "before_lunch",
    "after lunch": "after_lunch",
    "before dinner": "before_dinner",
    "after dinner": "after_dinner",
    "morning": "morning",
    "noon": "noon",
    "afternoon": "afternoon",
    "evening": "evening",
    "night": "night",
    "bedtime": "night",
    "早餐前": "before_breakfast",
    "早餐后": "after_breakfast",
    "午餐前": "before_lunch",
    "午餐后": "after_lunch",
    "晚餐前": "before_dinner",
    "晚餐后": "after_dinner",
    "睡前": "night",
    "早上": "morning",
    "中午": "noon",
    "晚上": "evening",
}
ROUTE_CUES = {
    "po": "oral",
    "oral": "oral",
    "iv": "intravenous",
    "intravenous": "intravenous",
    "im": "intramuscular",
    "subcutaneous": "subcutaneous",
    "sc": "subcutaneous",
    "口服": "oral",
    "静脉": "intravenous",
    "肌注": "intramuscular",
    "皮下": "subcutaneous",
}


def _safe_parse_date(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None


def _end_of_month(year: int, month: int) -> datetime:
    if month == 12:
        return datetime(year + 1, 1, 1) - timedelta(days=1)
    return datetime(year, month + 1, 1) - timedelta(days=1)


def scope_to_interval(scope: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    将 time_scope 解析为 (start_time, end_time, precision)。
    precision: day|week|month|year|range|global|unknown
    """
    s = (scope or "").strip()
    if not s or s.lower() == "global":
        return None, None, "global"

    m = ISO_RANGE_RE.match(s)
    if m:
        start, end = m.group(1), m.group(2)
        ds, de = _safe_parse_date(start), _safe_parse_date(end)
        if ds and de:
            if ds <= de:
                return start, end, "range"
            return end, start, "range"
        return None, None, "unknown"

    if ISO_DATE_RE.match(s):
        return s, s, "day"

    m = ISO_WEEK_RE.match(s)
    if m:
        year, week = int(m.group(1)), int(m.group(2))
        try:
            start = datetime.fromisocalendar(year, week, 1).strftime("%Y-%m-%d")
            end = datetime.fromisocalendar(year, week, 7).strftime("%Y-%m-%d")
            return start, end, "week"
        except ValueError:
            return None, None, "unknown"

    m = ISO_MONTH_RE.match(s)
    if m and not ISO_DATE_RE.match(s):
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            start = datetime(year, month, 1).strftime("%Y-%m-%d")
            end = _end_of_month(year, month).strftime("%Y-%m-%d")
            return start, end, "month"
        return None, None, "unknown"

    if ISO_YEAR_RE.match(s):
        year = int(s)
        return f"{year:04d}-01-01", f"{year:04d}-12-31", "year"

    return None, None, "unknown"


def infer_relation_type(attribute: str) -> str:
    attr = (attribute or "").strip().lower()
    if attr == "medication":
        return "TAKES_DRUG"
    if attr == "symptom":
        return "HAS_SYMPTOM"
    if attr == "primary_diagnosis":
        return "HAS_DIAGNOSIS"
    return "HAS_MEASUREMENT"


def extract_medication_qualifiers(text: str) -> Tuple[str, str, str, str]:
    raw = (text or "").strip()
    lowered = raw.lower()
    dosage = ""
    frequency = ""
    time_of_day = ""
    route = ""

    m = DOSAGE_RE.search(lowered)
    if m:
        dosage = f"{m.group(1)} {m.group(2)}"

    for pattern, canonical in FREQUENCY_PATTERNS:
        fm = pattern.search(lowered)
        if not fm:
            continue
        if "{n}" in canonical and fm.groups():
            n = fm.group(1)
            frequency = canonical.replace("{n}", n)
        else:
            frequency = canonical
        break

    for cue, normalized in TIME_OF_DAY_CUES.items():
        if cue in lowered or cue in raw:
            time_of_day = normalized
            break

    for cue, normalized in ROUTE_CUES.items():
        if cue in lowered or cue in raw:
            route = normalized
            break

    return dosage, frequency, time_of_day, route


@dataclass
class CandidateValue:
    """
    候选值：置信度加权的多候选冲突解决中的单个候选。

    在双时态冲突图（Bi-Temporal Conflict Graph）中，同一 (patient_id, attribute, time_scope)
    可能存在多个候选值，每个候选携带 soft truth confidence 而非二元选择。
    这使得冲突解决从 "pick winner" 升级为 "rank by confidence"。

    参考：EvoKG (arXiv 2025.09) 的 confidence-based contradiction resolution。
    """
    value: str
    unit: str = ""
    confidence: float = 1.0         # soft truth in [0, 1]
    provenance: List[int] = field(default_factory=list)
    speaker: str = ""
    t_event: Optional[str] = None   # 医学事件发生时间
    t_ingest: Optional[int] = None  # 系统获知时间（turn number）
    source_authority: float = 0.0   # doctor=1.0, patient=0.5, unknown=0.0
    temporal_recency: float = 0.0   # 归一化时间新近度 [0, 1]
    evidence_count: int = 1         # 支持该候选的独立证据条数

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "speaker": self.speaker,
            "t_event": self.t_event,
            "t_ingest": self.t_ingest,
            "source_authority": self.source_authority,
            "temporal_recency": self.temporal_recency,
            "evidence_count": self.evidence_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateValue":
        return cls(
            value=d["value"],
            unit=d.get("unit", ""),
            confidence=d.get("confidence", 1.0),
            provenance=d.get("provenance", []),
            speaker=d.get("speaker", ""),
            t_event=d.get("t_event"),
            t_ingest=d.get("t_ingest"),
            source_authority=d.get("source_authority", 0.0),
            temporal_recency=d.get("temporal_recency", 0.0),
            evidence_count=d.get("evidence_count", 1),
        )


@dataclass
class ConflictRecord:
    """
    单条冲突历史记录。

    当 M3 唯一性管理模块检测到冲突时，旧值将被封存为 ConflictRecord，
    而非直接丢弃，以支持 Conflict Detection F1 的评测。
    """
    old_value: str              # 被覆盖的旧值
    new_value: str              # 触发冲突的新值（即当前 CanonicalMemory.value）
    old_provenance: List[str]   # 旧值的来源（对话轮次 ID 列表）
    new_provenance: List[str]   # 新值的来源（对话轮次 ID 列表）
    conflict_type: str          # 冲突类型："value_change" | "unit_change" | "implicit"
    detected_at: str            # 检测到冲突的时间戳（ISO 格式）


@dataclass
class CanonicalMemory:
    """
    Canonical Memory 核心对象。

    代表系统对某个实体某个属性在特定时间范围内的"最权威认知"。
    这是 M3 唯一性管理模块的输出，也是 M5 双通道检索的结构化输入。
    """
    patient_id: str
    attribute: str
    value: str
    unit: str                           = ""
    time_scope: str                     = "global"
    confidence: float                   = 1.0
    provenance: List[str]               = field(default_factory=list)
    conflict_flag: bool                 = False
    conflict_history: List[ConflictRecord] = field(default_factory=list)
    update_policy: str                  = "unique"
    # 关系语义（新）
    subject_id: str                     = ""
    relation_type: str                  = ""
    target_value: str                   = ""
    # 时间结构（新）
    start_time: Optional[str]           = None
    end_time: Optional[str]             = None
    duration_days: Optional[int]        = None
    time_precision: str                 = ""
    time_source: str                    = "time_scope"
    is_ongoing: Optional[bool]          = None
    # 双时态字段（Bi-Temporal Conflict Graph）
    # 参考：Zep/Graphiti (arXiv 2025.01) bi-temporal model
    t_event: Optional[str]              = None   # 医学事件发生时间（TimeGrounder 输出）
    t_ingest: Optional[int]             = None   # 系统获知时间（turn number）
    t_valid_start: Optional[str]        = None   # 事实生效时间
    t_valid_end: Optional[str]          = None   # 事实失效时间（None = 持续有效）
    # 医疗关系限定词（新）
    dosage: str                         = ""
    frequency: str                      = ""
    time_of_day: str                    = ""
    route: str                          = ""
    qualifiers: dict                    = field(default_factory=dict)

    def __post_init__(self):
        # 关系默认值
        if not self.subject_id:
            self.subject_id = self.patient_id
        if not self.relation_type:
            self.relation_type = infer_relation_type(self.attribute)
        if not self.target_value:
            self.target_value = self.value

        # 时间结构默认值（由 time_scope 推导）
        if not self.start_time and not self.end_time:
            start, end, precision = scope_to_interval(self.time_scope)
            self.start_time = start
            self.end_time = end
            if not self.time_precision:
                self.time_precision = precision
        elif not self.time_precision:
            self.time_precision = "day" if self.start_time and self.end_time and self.start_time == self.end_time else "range"

        if self.duration_days is None and self.start_time and self.end_time:
            ds = _safe_parse_date(self.start_time)
            de = _safe_parse_date(self.end_time)
            if ds and de:
                self.duration_days = (de - ds).days + 1

        if self.is_ongoing is None:
            self.is_ongoing = bool(
                self.attribute == "medication"
                and self.update_policy == "latest"
                and self.time_scope == "global"
            )

        # medication 结构化字段默认值
        if self.attribute == "medication":
            d, f, tod, r = extract_medication_qualifiers(self.value)
            if not self.dosage:
                self.dosage = d
            if not self.frequency:
                self.frequency = f
            if not self.time_of_day:
                self.time_of_day = tod
            if not self.route:
                self.route = r

        if self.dosage and "dosage" not in self.qualifiers:
            self.qualifiers["dosage"] = self.dosage
        if self.frequency and "frequency" not in self.qualifiers:
            self.qualifiers["frequency"] = self.frequency
        if self.time_of_day and "time_of_day" not in self.qualifiers:
            self.qualifiers["time_of_day"] = self.time_of_day
        if self.route and "route" not in self.qualifiers:
            self.qualifiers["route"] = self.route

        # 双时态字段自动派生
        if self.t_event is None and self.start_time:
            self.t_event = self.start_time
        if self.t_ingest is None and self.provenance:
            int_provs = [p for p in self.provenance if isinstance(p, int)]
            if int_provs:
                self.t_ingest = max(int_provs)
        if self.t_valid_start is None and self.start_time:
            self.t_valid_start = self.start_time
        if self.t_valid_end is None and self.end_time:
            self.t_valid_end = self.end_time

    # ─── 唯一性键（用于 Unique-F1 评测的匹配） ──────────────────────────────
    @property
    def unique_key(self) -> tuple[str, str, str]:
        """
        唯一性键：(patient_id, attribute, time_scope)。

        对于 unique 和 latest 策略的属性，此三元组在记忆库中应唯一。
        对于 append 策略的属性，此三元组可以对应多条记录。
        """
        return (self.patient_id, self.attribute, self.time_scope)

    # ─── 序列化 ─────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """将 CanonicalMemory 对象序列化为字典（用于保存 GT 文件）。"""
        return {
            "patient_id": self.patient_id,
            "attribute": self.attribute,
            "value": self.value,
            "unit": self.unit,
            "time_scope": self.time_scope,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "conflict_flag": self.conflict_flag,
            "conflict_history": [
                {
                    "old_value": cr.old_value,
                    "new_value": cr.new_value,
                    "old_provenance": cr.old_provenance,
                    "new_provenance": cr.new_provenance,
                    "conflict_type": cr.conflict_type,
                    "detected_at": cr.detected_at,
                }
                for cr in self.conflict_history
            ],
            "update_policy": self.update_policy,
            "subject_id": self.subject_id,
            "relation_type": self.relation_type,
            "target_value": self.target_value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_days": self.duration_days,
            "time_precision": self.time_precision,
            "time_source": self.time_source,
            "is_ongoing": self.is_ongoing,
            "t_event": self.t_event,
            "t_ingest": self.t_ingest,
            "t_valid_start": self.t_valid_start,
            "t_valid_end": self.t_valid_end,
            "dosage": self.dosage,
            "frequency": self.frequency,
            "time_of_day": self.time_of_day,
            "route": self.route,
            "qualifiers": self.qualifiers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CanonicalMemory":
        """从字典反序列化为 CanonicalMemory 对象（用于加载 GT 文件）。"""
        conflict_history = [
            ConflictRecord(
                old_value=cr["old_value"],
                new_value=cr["new_value"],
                old_provenance=cr["old_provenance"],
                new_provenance=cr["new_provenance"],
                conflict_type=cr["conflict_type"],
                detected_at=cr["detected_at"],
            )
            for cr in d.get("conflict_history", [])
        ]
        return cls(
            patient_id=d["patient_id"],
            attribute=d["attribute"],
            value=d["value"],
            unit=d.get("unit", ""),
            time_scope=d.get("time_scope", "global"),
            confidence=d.get("confidence", 1.0),
            provenance=d.get("provenance", []),
            conflict_flag=d.get("conflict_flag", False),
            conflict_history=conflict_history,
            update_policy=d.get("update_policy", "unique"),
            subject_id=d.get("subject_id", ""),
            relation_type=d.get("relation_type", ""),
            target_value=d.get("target_value", ""),
            start_time=d.get("start_time"),
            end_time=d.get("end_time"),
            duration_days=d.get("duration_days"),
            time_precision=d.get("time_precision", ""),
            time_source=d.get("time_source", "time_scope"),
            is_ongoing=d.get("is_ongoing"),
            t_event=d.get("t_event"),
            t_ingest=d.get("t_ingest"),
            t_valid_start=d.get("t_valid_start"),
            t_valid_end=d.get("t_valid_end"),
            dosage=d.get("dosage", ""),
            frequency=d.get("frequency", ""),
            time_of_day=d.get("time_of_day", ""),
            route=d.get("route", ""),
            qualifiers=d.get("qualifiers", {}),
        )

    def __repr__(self) -> str:
        conflict_marker = " ⚠️ CONFLICT" if self.conflict_flag else ""
        return (
            f"CanonicalMemory({self.patient_id!r}, {self.attribute!r}, "
            f"value={self.value!r}, unit={self.unit!r}, "
            f"scope={self.time_scope!r}, relation={self.relation_type!r}, "
            f"start={self.start_time!r}, end={self.end_time!r}, "
            f"policy={self.update_policy!r}"
            f"{conflict_marker})"
        )
