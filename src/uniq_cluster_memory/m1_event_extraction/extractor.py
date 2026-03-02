"""
m1_event_extraction/extractor.py
=================================
M1 模块：医疗事件抽取器。

核心职责：
    将对话历史（List[DialogueTurn]）转换为结构化的 ExtractedEvent 列表。

设计要点：
    1. Schema 完整性：每个事件包含 attribute / value / unit / time_expr / provenance。
    2. Provenance 追踪：记录每个事件来自哪一轮对话（turn_id），支持后续溯源。
    3. Time Expression 保留：保留原始时间表达（如"yesterday"、"last week"），
       由 M3 的 Time Grounder 负责将其解析为标准日期。
    4. 滑动窗口处理：将对话按 WINDOW_SIZE 轮分块，每块独立抽取，
       避免超长对话超出 LLM 上下文窗口。
    5. 去重合并：对同一 (attribute, time_expr) 的重复抽取结果进行合并。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.uniq_cluster_memory.utils.llm_client import get_llm_client, LLM_MODEL

# ─── 数据结构定义 ────────────────────────────────────────────────────────────

@dataclass
class ExtractedEvent:
    """
    M1 模块的输出单元：从对话中抽取的单个医疗事件。

    与 RawEvent（生成器内部结构）不同，ExtractedEvent 是系统从自然语言中
    抽取的结果，包含不确定性（confidence）和原始时间表达（time_expr）。
    """
    event_id: str                       # 唯一标识符（由 M1 生成）
    dialogue_id: str                    # 来源对话 ID
    attribute: str                      # 标准化属性名（e.g., blood_glucose）
    value: str                          # 属性值（e.g., "7.2"）
    unit: str                           # 单位（e.g., "mmol/L"），无则为 ""
    time_expr: str                      # 原始时间表达（e.g., "yesterday", "2024-01-15", "global"）
    update_policy: str                  # "unique" | "latest" | "append"
    confidence: float                   # 抽取置信度（0.0-1.0）
    provenance: List[int]               # 来源轮次 ID 列表（turn_id）
    speaker: str                        # 信息来源说话人（"patient" | "doctor"）
    raw_text_snippet: str               # 原始文本片段（用于调试）

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "dialogue_id": self.dialogue_id,
            "attribute": self.attribute,
            "value": self.value,
            "unit": self.unit,
            "time_expr": self.time_expr,
            "update_policy": self.update_policy,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "speaker": self.speaker,
            "raw_text_snippet": self.raw_text_snippet,
        }


# ─── 提示词 ──────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are a precise medical information extraction system for clinical dialogues.

Your task: Extract ALL medical facts from the given dialogue turns and return them as structured JSON.

For each medical fact, output ONE JSON object with these fields:
- "attribute": standardized snake_case name. Use ONLY these canonical names:
    Measurements: blood_glucose, blood_pressure_sys, blood_pressure_dia, heart_rate, body_temperature, hemoglobin
    Clinical: primary_diagnosis, medication, symptom
  If the fact doesn't fit, use a descriptive snake_case name.
- "value": the exact value mentioned (string). For medications, include dosage and frequency.
- "unit": measurement unit (e.g., "mmol/L", "mmHg", "bpm", "°C", "g/L"). Empty string if none.
- "time_expr": the time expression as spoken (e.g., "this morning", "2023-02-22", "yesterday", "global" if no time mentioned).
  IMPORTANT: preserve the ORIGINAL time expression, do NOT normalize it.
- "update_policy": "unique" for measurements/diagnoses, "latest" for medications, "append" for symptoms
- "confidence": 0.0-1.0 (1.0 = explicitly stated, 0.7 = implied, 0.5 = uncertain)
- "provenance_turns": list of turn_ids where this fact is mentioned (e.g., [3, 7])
- "speaker": "patient" or "doctor" (who provided this information)
- "raw_text_snippet": the exact quote from the dialogue (max 100 chars)

CRITICAL RULES:
1. Extract EVERY medical fact, including repeated mentions and apparent contradictions.
2. If the same attribute is mentioned with DIFFERENT values, create SEPARATE records for each.
3. For coreference ("that reading", "that number", "the result"), extract the value being referred to.
4. Do NOT merge or deduplicate - that is done by later modules.
5. Return ONLY a JSON array. No explanation, no markdown.

Example output:
[
  {"attribute": "blood_glucose", "value": "7.2", "unit": "mmol/L", "time_expr": "this morning",
   "update_policy": "unique", "confidence": 1.0, "provenance_turns": [2], "speaker": "patient",
   "raw_text_snippet": "my blood sugar was 7.2 this morning"},
  {"attribute": "medication", "value": "Metformin 500mg bid", "unit": "", "time_expr": "global",
   "update_policy": "latest", "confidence": 1.0, "provenance_turns": [5], "speaker": "doctor",
   "raw_text_snippet": "I am prescribing Metformin 500mg twice daily"}
]"""


# ─── 主抽取器类 ──────────────────────────────────────────────────────────────

class MedicalEventExtractor:
    """
    M1 医疗事件抽取器。

    使用 Qwen-Plus 对对话进行滑动窗口抽取，输出 ExtractedEvent 列表。
    """

    WINDOW_SIZE = 8       # 每次处理的对话轮数
    STRIDE = 6            # 滑动步长（重叠 2 轮，避免边界遗漏）
    MAX_RETRIES = 3
    MEASUREMENT_ATTRIBUTES = {
        "blood_glucose",
        "blood_pressure_sys",
        "blood_pressure_dia",
        "heart_rate",
        "body_temperature",
        "hemoglobin",
    }
    SYMPTOM_ATTRIBUTES = {"symptom"}
    DIAGNOSIS_ATTRIBUTES = {"primary_diagnosis"}
    PLACEHOLDER_VALUES = {
        "",
        "none",
        "null",
        "n/a",
        "na",
        "unknown",
        "unspecified",
        "not mentioned",
        "not_mentioned",
        "not provided",
        "that",
        "this",
        "it",
        "same",
        "same as above",
        "change",
    }
    GENERIC_SYMPTOM_VALUES = {
        "no other new symptoms",
        "no new symptoms",
        "other symptoms",
        "unusual symptoms",
        "feeling generally unwell",
        "general discomfort",
        "blood pressure issues",
        "heart rate issues",
    }
    MIN_CONFIDENCE_BY_ATTRIBUTE = {
        "symptom": 0.9,
        "primary_diagnosis": 0.8,
    }
    SPECULATIVE_Q_PATTERNS = [
        re.compile(r"\bcould\b", re.IGNORECASE),
        re.compile(r"\bmaybe\b", re.IGNORECASE),
        re.compile(r"\bis it\b", re.IGNORECASE),
        re.compile(r"\b可能\b|\b会不会\b|\b是不是\b"),
    ]
    _NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")
    _MEAS_VALUE_UNIT_RE = re.compile(
        r"(-?\d+(?:\.\d+)?)\s*(mmhg|bpm|g/l|gdl|g/dl|mmol/l|mg/dl|°c|c|°f|f)\b",
        re.IGNORECASE,
    )
    MEAS_EXPECTED_UNIT = {
        "blood_pressure_sys": "mmHg",
        "blood_pressure_dia": "mmHg",
        "heart_rate": "bpm",
        "body_temperature": "°C",
        "hemoglobin": "g/L",
        "blood_glucose": "mmol/L",
    }
    UNIT_ALIASES = {
        "mmhg": "mmHg",
        "bpm": "bpm",
        "g/l": "g/L",
        "gdl": "g/dL",
        "g/dl": "g/dL",
        "mmol/l": "mmol/L",
        "mg/dl": "mg/dL",
        "°c": "°C",
        "c": "°C",
        "°f": "°F",
        "f": "°F",
    }
    SYMPTOM_CANONICAL_PATTERNS = [
        (re.compile(r"\bnauseous\b|\bnausea\b", re.IGNORECASE), "nausea"),
        (re.compile(r"\bfrequent urination\b|\bincreased urination\b|\bpolyuria\b", re.IGNORECASE), "polyuria"),
        (re.compile(r"\btiredness\b|\bfatigue\b|\bfeeling weak\b|\bweakness\b|\bweak\b", re.IGNORECASE), "fatigue"),
        (re.compile(r"\bdizzy\b|\bdizziness\b|\blightheadedness\b|\blight-headedness\b", re.IGNORECASE), "dizziness"),
        (re.compile(r"\bpalpitations?\b|\birregular heartbeats?\b|\bheart racing\b|\bheart palpitations?\b", re.IGNORECASE), "palpitations"),
        (re.compile(r"\bheadaches?\b|\bfrequent headaches?\b", re.IGNORECASE), "headache"),
        (re.compile(r"\blower limb edema\b|\bleg swelling\b|\bankle swelling\b|\bswelling in legs\b|\bedema\b", re.IGNORECASE), "lower limb edema"),
        (re.compile(r"\bblurred vision\b", re.IGNORECASE), "blurred vision"),
        (re.compile(r"\bchest tightness\b|\bchest discomfort\b", re.IGNORECASE), "chest tightness"),
    ]
    DIAGNOSIS_CANONICAL_PATTERNS = [
        (re.compile(r"\bcoronary artery disease\b|\bcad\b", re.IGNORECASE), "Coronary Artery Disease"),
        (re.compile(r"\btype\s*2\b.*\bdiabetes\b|\btype ii\b.*\bdiabetes\b|\bt2dm\b|\bdiabetes mellitus\b", re.IGNORECASE), "Type 2 Diabetes Mellitus"),
        (re.compile(r"\bdiabetes\b", re.IGNORECASE), "Type 2 Diabetes Mellitus"),
        (re.compile(r"\bessential hypertension\b|\bhypertension\b", re.IGNORECASE), "Essential Hypertension"),
        (re.compile(r"\bhypothyroidism\b", re.IGNORECASE), "Hypothyroidism"),
        (re.compile(r"\bchronic kidney disease\b.*\bstage\s*3\b|\bckd\b.*\b3\b", re.IGNORECASE), "Chronic Kidney Disease Stage 3"),
    ]
    MED_FREQUENCY_PATTERNS = [
        (re.compile(r"\b(qn|qhs|hs|at\s+night|every\s+night|nightly|at\s+bedtime|before\s+bed)\b", re.IGNORECASE), "qn"),
        (re.compile(r"\b(bid|twice\s+(a|per)\s+day|twice\s+daily|two\s+times\s+daily)\b", re.IGNORECASE), "bid"),
        (re.compile(r"\b(tid|three\s+times\s+daily)\b", re.IGNORECASE), "tid"),
        (re.compile(r"\b(qid|four\s+times\s+daily)\b", re.IGNORECASE), "qid"),
        (re.compile(r"\b(qd|once\s+(a|per)\s+day|once\s+daily|daily|once\s+a\s+day)\b", re.IGNORECASE), "qd"),
    ]
    _MED_DOSE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g|ml|iu|u|units?)\b", re.IGNORECASE)
    MED_DOSE_UNIT_MAP = {
        "mg": "mg",
        "mcg": "mcg",
        "ug": "mcg",
        "g": "g",
        "ml": "ml",
        "iu": "U",
        "u": "U",
        "unit": "U",
        "units": "U",
    }

    def __init__(self):
        self.client = get_llm_client()
        self._event_counter = 0

    def extract(
        self,
        dialogue: List[dict],
        dialogue_id: str,
    ) -> List[ExtractedEvent]:
        """
        从完整对话中抽取所有医疗事件。

        Args:
            dialogue:     对话轮次列表，每条为 {"turn_id": int, "speaker": str, "text": str}。
            dialogue_id:  对话 ID。

        Returns:
            ExtractedEvent 列表（未去重，保留所有原始抽取结果）。
        """
        all_events: List[ExtractedEvent] = []
        seen_snippets: set = set()  # 用于避免滑动窗口重复抽取同一片段

        # 滑动窗口处理
        n = len(dialogue)
        windows = []
        start = 0
        while start < n:
            end = min(start + self.WINDOW_SIZE, n)
            windows.append(dialogue[start:end])
            if end == n:
                break
            start += self.STRIDE

        for window in windows:
            window_events = self._extract_window(window, dialogue_id, seen_snippets)
            all_events.extend(window_events)

        return all_events

    def _extract_window(
        self,
        turns: List[dict],
        dialogue_id: str,
        seen_snippets: set,
    ) -> List[ExtractedEvent]:
        """对单个窗口进行抽取。"""
        # 构造对话文本
        lines = []
        for t in turns:
            speaker = "Doctor" if t["speaker"] == "doctor" else "Patient"
            lines.append(f"[Turn {t['turn_id']}] {speaker}: {t['text']}")
        dialogue_text = "\n".join(lines)

        user_prompt = f"Extract all medical facts from these dialogue turns:\n\n{dialogue_text}"

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                )
                raw = response.choices[0].message.content.strip()
                # 清理 markdown 包装
                if raw.startswith("```"):
                    parts = raw.split("```")
                    raw = parts[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()
                records = json.loads(raw)
                if not isinstance(records, list):
                    raise ValueError("Expected JSON array")

                events = []
                for r in records:
                    attribute = str(r.get("attribute", "")).strip().lower().replace(" ", "_")
                    value = str(r.get("value", "")).strip()
                    unit = str(r.get("unit", "")).strip()
                    confidence = float(r.get("confidence", 1.0))
                    normalized = self._normalize_record(
                        attribute=attribute,
                        value=value,
                        unit=unit,
                        confidence=confidence,
                        speaker=str(r.get("speaker", "unknown")),
                        snippet=str(r.get("raw_text_snippet", ""))[:100],
                    )
                    if normalized is None:
                        continue
                    attribute, value, unit, confidence = normalized

                    snippet = str(r.get("raw_text_snippet", ""))[:100]
                    # 跳过已抽取过的相同片段（滑动窗口重叠去重）
                    snippet_key = (
                        attribute,
                        value,
                        str(r.get("time_expr", "")),
                    )
                    if snippet_key in seen_snippets:
                        continue
                    seen_snippets.add(snippet_key)

                    self._event_counter += 1
                    events.append(ExtractedEvent(
                        event_id=f"m1_{dialogue_id}_{self._event_counter:04d}",
                        dialogue_id=dialogue_id,
                        attribute=attribute,
                        value=value,
                        unit=unit,
                        time_expr=str(r.get("time_expr", "global")).strip(),
                        update_policy=str(r.get("update_policy", "unique")),
                        confidence=confidence,
                        provenance=list(r.get("provenance_turns", [])),
                        speaker=str(r.get("speaker", "unknown")),
                        raw_text_snippet=snippet,
                    ))
                return events

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2)
                else:
                    print(f"    [M1 WARN] Window extraction failed after {self.MAX_RETRIES} attempts: {e}")
                    return []

        return []

    @classmethod
    def _normalize_record(
        cls,
        attribute: str,
        value: str,
        unit: str,
        confidence: float,
        speaker: str = "unknown",
        snippet: str = "",
    ) -> Optional[tuple[str, str, str, float]]:
        """
        记录级规范化与过滤。

        返回:
            (attribute, value, unit, confidence) or None
        """
        attr = (attribute or "").strip().lower()
        val = (value or "").strip()
        unt = (unit or "").strip()
        conf = float(confidence)

        if not cls._is_valid_record(attr, val, conf, speaker=speaker, snippet=snippet):
            return None

        if attr in cls.MEASUREMENT_ATTRIBUTES:
            normalized = cls._normalize_measurement(attr, val, unt)
            if normalized is None:
                return None
            val, unt = normalized
        elif attr == "medication":
            val = cls._normalize_medication_value(val)
        elif attr in cls.SYMPTOM_ATTRIBUTES:
            val = cls._normalize_symptom_value(val)
        elif attr in cls.DIAGNOSIS_ATTRIBUTES:
            val = cls._normalize_diagnosis_value(val)

        return attr, val, unt, conf

    @classmethod
    def _is_valid_record(
        cls,
        attribute: str,
        value: str,
        confidence: float = 1.0,
        speaker: str = "unknown",
        snippet: str = "",
    ) -> bool:
        """
        过滤明显无效的抽取结果，降低 M3 冲突噪声。

        规则：
        - attribute/value 不能为空
        - value 不能是明显占位词（that/unknown/空值等）
        - 对 measurement 属性，value 必须包含数字
        """
        attr = (attribute or "").strip().lower()
        v = (value or "").strip()
        if not attr or not v:
            return False

        lowered = v.lower()
        if lowered in cls.PLACEHOLDER_VALUES:
            return False

        min_conf = cls.MIN_CONFIDENCE_BY_ATTRIBUTE.get(attr, 0.7)
        if confidence < min_conf:
            return False

        if attr in cls.SYMPTOM_ATTRIBUTES and lowered in cls.GENERIC_SYMPTOM_VALUES:
            return False

        if attr in cls.MEASUREMENT_ATTRIBUTES and cls._NUMERIC_RE.search(v) is None:
            return False

        # 过滤患者问句中的“推测诊断”，避免把猜测当成最终诊断
        if attr in cls.DIAGNOSIS_ATTRIBUTES and (speaker or "").strip().lower() == "patient":
            s = (snippet or value or "").strip()
            if "?" in s or "？" in s:
                if any(p.search(s) for p in cls.SPECULATIVE_Q_PATTERNS):
                    return False

        return True

    @classmethod
    def _normalize_symptom_value(cls, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            return raw
        for pattern, canonical in cls.SYMPTOM_CANONICAL_PATTERNS:
            if pattern.search(raw):
                return canonical
        normalized = re.sub(r"\s+", " ", raw.lower()).strip()
        return normalized

    @classmethod
    def _normalize_diagnosis_value(cls, value: str) -> str:
        raw = " ".join((value or "").strip().split())
        if not raw:
            return raw
        for pattern, canonical in cls.DIAGNOSIS_CANONICAL_PATTERNS:
            if pattern.search(raw):
                return canonical
        return raw

    @classmethod
    def _normalize_medication_value(cls, value: str) -> str:
        raw = " ".join((value or "").strip().split())
        if not raw:
            return raw

        normalized = raw.lower()
        normalized = normalized.replace("milligrams", "mg").replace("milligram", "mg")
        normalized = normalized.replace("micrograms", "mcg").replace("microgram", "mcg")
        normalized = normalized.replace("international units", "u")
        normalized = normalized.replace(" units", " u").replace(" unit", " u")

        frequency = ""
        for pattern, canonical in cls.MED_FREQUENCY_PATTERNS:
            if pattern.search(normalized):
                frequency = canonical
                break

        cleaned = normalized
        for pattern, _ in cls.MED_FREQUENCY_PATTERNS:
            cleaned = pattern.sub(" ", cleaned)
        cleaned = re.sub(r"\b(once|twice|daily|nightly|at|every|per|a|day|night|bedtime)\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        dose = ""
        med_name = cleaned
        dose_match = cls._MED_DOSE_RE.search(cleaned)
        if dose_match:
            num = cls._format_number(float(dose_match.group(1)))
            unit_raw = dose_match.group(2).lower()
            unit = cls.MED_DOSE_UNIT_MAP.get(unit_raw, unit_raw)
            dose = f"{num}{unit}"
            med_name = cleaned[:dose_match.start()].strip()

        med_name = re.sub(r"[^a-z0-9\s-]", " ", med_name)
        med_name = re.sub(r"\s+", " ", med_name).strip()
        if not med_name:
            med_name = re.sub(r"[^a-z0-9\s-]", " ", normalized)
            med_name = re.sub(r"\s+", " ", med_name).strip()

        med_name = " ".join(token.capitalize() for token in med_name.split())

        parts = [med_name] if med_name else []
        if dose:
            parts.append(dose)
        if frequency:
            parts.append(frequency)
        if not parts:
            return raw
        return " ".join(parts).strip()

    @classmethod
    def _normalize_measurement(
        cls,
        attribute: str,
        value: str,
        unit: str,
    ) -> Optional[tuple[str, str]]:
        """
        规范化 measurement 的 value/unit，并执行单位一致性校验。
        """
        expected = cls.MEAS_EXPECTED_UNIT.get(attribute, "")
        raw_value = (value or "").strip()
        raw_unit = (unit or "").strip()

        num_match = cls._NUMERIC_RE.search(raw_value)
        if num_match is None:
            return None
        numeric = float(num_match.group(0))

        inferred_unit = ""
        vu_match = cls._MEAS_VALUE_UNIT_RE.search(raw_value)
        if vu_match:
            inferred_unit = vu_match.group(2)

        candidate_unit = raw_unit or inferred_unit
        canonical_unit = cls._canonicalize_unit(candidate_unit)

        # 如果仍缺失单位，采用属性默认单位。
        if not canonical_unit:
            canonical_unit = expected

        # 单位转换与一致性校验
        converted = cls._convert_to_expected_unit(attribute, numeric, canonical_unit)
        if converted is None:
            return None
        norm_value, norm_unit = converted
        return norm_value, norm_unit

    @classmethod
    def _canonicalize_unit(cls, unit: str) -> str:
        if not unit:
            return ""
        key = unit.strip().lower().replace(" ", "")
        return cls.UNIT_ALIASES.get(key, unit.strip())

    @classmethod
    def _convert_to_expected_unit(
        cls,
        attribute: str,
        value: float,
        unit: str,
    ) -> Optional[tuple[str, str]]:
        expected = cls.MEAS_EXPECTED_UNIT.get(attribute, "")
        if not expected:
            return cls._format_number(value), unit

        if unit == expected:
            return cls._format_number(value), expected

        # 常见可转换单位
        if attribute == "blood_glucose" and unit == "mg/dL":
            mmol = value / 18.0
            return cls._format_number(mmol), expected
        if attribute == "hemoglobin" and unit == "g/dL":
            gl = value * 10.0
            return cls._format_number(gl), expected
        if attribute == "body_temperature" and unit == "°F":
            celsius = (value - 32.0) * 5.0 / 9.0
            return cls._format_number(celsius), expected

        # 无法转换，视为单位不一致
        return None

    @staticmethod
    def _format_number(value: float) -> str:
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        s = f"{value:.2f}"
        return s.rstrip("0").rstrip(".")
