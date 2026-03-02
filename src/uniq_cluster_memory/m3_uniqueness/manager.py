"""
m3_uniqueness/manager.py
=========================
M3 模块：唯一性管理器（Time Grounder + Uniqueness Manager）。

核心职责：
    1. Time Grounder：将 M1 抽取的原始时间表达（time_expr）解析为标准日期（ISO 8601）。
       - 显式日期（"2023-02-22"）直接使用。
       - 相对时间（"yesterday", "last week"）基于对话日期推算。
       - 无法解析的时间表达保留为 "global"。
    2. Uniqueness Manager：基于 scope_policies.yaml 的策略，
       对同一 (patient_id, attribute, time_scope) 的多条事件进行去重和合并，
       输出最终的 CanonicalMemory 列表。
    3. Conflict Detector：在合并过程中，检测同一 scope 下的值冲突，
       设置 conflict_flag 并记录 conflict_history。

这是论文 A 方向的核心模块，也是 Unique-F1(S) 指标的直接体现。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from src.uniq_cluster_memory.m1_event_extraction import ExtractedEvent
from src.uniq_cluster_memory.m2_clustering import AttributeCluster, BundleGraph
from src.uniq_cluster_memory.schema import CanonicalMemory, ConflictRecord, scope_to_interval
from src.uniq_cluster_memory.utils.llm_client import get_llm_client, LLM_MODEL_FAST


# ─── Time Grounder ───────────────────────────────────────────────────────────

class TimeGrounder:
    """
    将原始时间表达解析为标准 time_scope（day/week/month/year/range/global）。

    策略优先级：
        1. 规则匹配：识别显式日期/区间/周/月/年表达（最快，无 LLM 调用）。
        2. 规则推算：处理中英文相对时间表达（"yesterday", "上周一" 等）。
        3. LLM 归一化：对复杂表达使用 Qwen-Turbo 输出结构化规范化结果。
        4. 回退：无法解析时返回 "global"。
    """

    # 相对日期（天粒度）
    RELATIVE_DAY_RULES: Dict[str, int] = {
        "today": 0,
        "now": 0,
        "right now": 0,
        "recently": 0,
        "latest": 0,
        "most recent": 0,
        "just now": 0,
        "this noon": 0,
        "this morning": 0,
        "this afternoon": 0,
        "this evening": 0,
        "tonight": 0,
        "今早": 0,
        "今天": 0,
        "yesterday": -1,
        "day before yesterday": -2,
        "前天": -2,
        "昨天": -1,
        "two days ago": -2,
        "3 days ago": -3,
        "three days ago": -3,
    }

    # 相对周/月/年粒度
    RELATIVE_WEEK_RULES: Dict[str, int] = {
        "this week": 0,
        "last week": -1,
        "next week": 1,
        "本周": 0,
        "这周": 0,
        "上周": -1,
        "下周": 1,
    }

    RELATIVE_MONTH_RULES: Dict[str, int] = {
        "this month": 0,
        "last month": -1,
        "next month": 1,
        "a month ago": -1,
        "本月": 0,
        "这个月": 0,
        "上个月": -1,
        "下个月": 1,
    }

    RELATIVE_YEAR_RULES: Dict[str, int] = {
        "this year": 0,
        "last year": -1,
        "next year": 1,
        "今年": 0,
        "去年": -1,
        "明年": 1,
    }

    # 绝对时间表达
    ISO_DATE_PATTERN = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
    ISO_WEEK_PATTERN = re.compile(r"\b(\d{4})-?W(\d{1,2})\b", re.IGNORECASE)
    ISO_MONTH_PATTERN = re.compile(r"\b(\d{4})[-/](\d{1,2})\b")
    YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
    RANGE_DATE_PATTERN = re.compile(
        r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s*(?:to|~|至|到|-)\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
        re.IGNORECASE,
    )

    # 相对时间数值模式
    EN_DAYS_AGO_PATTERN = re.compile(r"\b(\d+)\s+days?\s+ago\b", re.IGNORECASE)
    EN_WEEKS_AGO_PATTERN = re.compile(r"\b(\d+)\s+weeks?\s+ago\b", re.IGNORECASE)
    EN_MONTHS_AGO_PATTERN = re.compile(r"\b(\d+)\s+months?\s+ago\b", re.IGNORECASE)
    EN_YEARS_AGO_PATTERN = re.compile(r"\b(\d+)\s+years?\s+ago\b", re.IGNORECASE)
    CN_DAYS_AGO_PATTERN = re.compile(r"(\d+)天前")
    CN_WEEKS_AGO_PATTERN = re.compile(r"(\d+)(?:周|星期)前")
    CN_MONTHS_AGO_PATTERN = re.compile(r"(\d+)个月前")
    CN_YEARS_AGO_PATTERN = re.compile(r"(\d+)年前")

    # 工作日映射
    WEEKDAY_MAP_EN = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    WEEKDAY_MAP_CN = {
        "一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6,
    }
    WEEKDAY_EN_PATTERN = re.compile(
        r"\b(last|this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        re.IGNORECASE,
    )
    WEEKDAY_CN_PATTERN = re.compile(r"(上|这|本|下)?(?:周|星期)(一|二|三|四|五|六|日|天)")

    def __init__(
        self,
        dialogue_date: Optional[str] = None,
        missing_time_scope: str = "global",
    ):
        """
        Args:
            dialogue_date: 对话发生日期（ISO 格式，如 "2023-02-22"）。
                           用于解析相对时间表达。如果为 None，则使用今天的日期。
        """
        if dialogue_date:
            try:
                self.reference_date = datetime.strptime(dialogue_date, "%Y-%m-%d")
            except ValueError:
                self.reference_date = datetime.today()
        else:
            self.reference_date = datetime.today()
        mode = (missing_time_scope or "global").strip().lower()
        if mode not in {"global", "reference_day"}:
            mode = "global"
        self.missing_time_scope = mode
        self._llm_client = None
        self._cache: Dict[str, str] = {}

    def _get_llm_client(self):
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    def ground(self, time_expr: str) -> str:
        """
        将时间表达解析为标准日期字符串。

        Args:
            time_expr: 原始时间表达（如 "yesterday", "2023-02-22", "global"）。

        Returns:
            标准 time_scope 字符串（如 "2023-02-21", "2023-W08", "2023-02", "global"）。
        """
        raw_expr = (time_expr or "").strip()
        if not raw_expr or raw_expr.lower() in ("global", "unknown", "n/a", ""):
            if self.missing_time_scope == "reference_day":
                return self.reference_date.strftime("%Y-%m-%d")
            return "global"
        if raw_expr in self._cache:
            return self._cache[raw_expr]

        # 策略 1：显式时间表达解析
        scope = self._parse_absolute(raw_expr)
        if scope is not None:
            self._cache[raw_expr] = scope
            return scope

        # 策略 2：规则推算（中英文）
        scope = self._rule_ground(raw_expr)
        if scope is not None:
            self._cache[raw_expr] = scope
            return scope

        # 策略 3：LLM 结构化归一化（处理复杂表达）
        scope = self._llm_ground(raw_expr)
        if scope == "global" and self.missing_time_scope == "reference_day":
            scope = self.reference_date.strftime("%Y-%m-%d")
        self._cache[raw_expr] = scope
        return scope

    def _parse_absolute(self, expression: str) -> Optional[str]:
        """解析显式日期/区间/周/月/年表达。"""
        range_match = self.RANGE_DATE_PATTERN.search(expression)
        if range_match:
            start = self._normalize_date_str(range_match.group(1))
            end = self._normalize_date_str(range_match.group(2))
            if start and end:
                if start <= end:
                    return f"{start}..{end}"
                return f"{end}..{start}"

        date_match = self.ISO_DATE_PATTERN.search(expression)
        if date_match:
            iso = self._safe_date_from_parts(
                int(date_match.group(1)),
                int(date_match.group(2)),
                int(date_match.group(3)),
            )
            if iso:
                return iso

        week_match = self.ISO_WEEK_PATTERN.search(expression)
        if week_match:
            year = int(week_match.group(1))
            week = int(week_match.group(2))
            if 1 <= week <= 53:
                return f"{year:04d}-W{week:02d}"

        # 月表达（避免与完整日期重复匹配）
        if not self.ISO_DATE_PATTERN.search(expression):
            month_match = self.ISO_MONTH_PATTERN.search(expression)
            if month_match:
                year = int(month_match.group(1))
                month = int(month_match.group(2))
                if 1 <= month <= 12:
                    return f"{year:04d}-{month:02d}"

        year_match = self.YEAR_PATTERN.search(expression)
        if year_match and any(k in expression for k in ("年", "year")):
            return year_match.group(1)

        return None

    def _rule_ground(self, expression: str) -> Optional[str]:
        """规则归一化：处理中英文相对时间、工作日与模糊粒度表达。"""
        expr = expression.strip().lower()

        for pattern, offset_days in self.RELATIVE_DAY_RULES.items():
            if pattern in expr:
                return (self.reference_date + timedelta(days=offset_days)).strftime("%Y-%m-%d")

        # 英文工作日表达
        match = self.WEEKDAY_EN_PATTERN.search(expr)
        if match:
            prefix, weekday_text = match.group(1).lower(), match.group(2).lower()
            week_offset = {"last": -1, "this": 0, "next": 1}[prefix]
            return self._resolve_weekday(self.WEEKDAY_MAP_EN[weekday_text], week_offset)

        # 中文工作日表达（如 上周三 / 周五 / 星期天）
        match = self.WEEKDAY_CN_PATTERN.search(expression)
        if match:
            prefix, weekday_text = match.group(1), match.group(2)
            week_offset = {"上": -1, "这": 0, "本": 0, "下": 1}.get(prefix or "这", 0)
            return self._resolve_weekday(self.WEEKDAY_MAP_CN[weekday_text], week_offset)

        for pattern, offset_weeks in self.RELATIVE_WEEK_RULES.items():
            if pattern in expr:
                target = self.reference_date + timedelta(days=7 * offset_weeks)
                return self._format_iso_week(target)
        for pattern, offset_months in self.RELATIVE_MONTH_RULES.items():
            if pattern in expr:
                return self._shift_months(self.reference_date, offset_months)
        for pattern, offset_years in self.RELATIVE_YEAR_RULES.items():
            if pattern in expr:
                return f"{self.reference_date.year + offset_years:04d}"

        # 英文数值相对表达
        match = self.EN_DAYS_AGO_PATTERN.search(expr)
        if match:
            days = int(match.group(1))
            return (self.reference_date - timedelta(days=days)).strftime("%Y-%m-%d")
        match = self.EN_WEEKS_AGO_PATTERN.search(expr)
        if match:
            weeks = int(match.group(1))
            return self._format_iso_week(self.reference_date - timedelta(days=7 * weeks))
        match = self.EN_MONTHS_AGO_PATTERN.search(expr)
        if match:
            months = int(match.group(1))
            return self._shift_months(self.reference_date, -months)
        match = self.EN_YEARS_AGO_PATTERN.search(expr)
        if match:
            years = int(match.group(1))
            return f"{self.reference_date.year - years:04d}"

        # 中文数值相对表达
        match = self.CN_DAYS_AGO_PATTERN.search(expression)
        if match:
            days = int(match.group(1))
            return (self.reference_date - timedelta(days=days)).strftime("%Y-%m-%d")
        match = self.CN_WEEKS_AGO_PATTERN.search(expression)
        if match:
            weeks = int(match.group(1))
            return self._format_iso_week(self.reference_date - timedelta(days=7 * weeks))
        match = self.CN_MONTHS_AGO_PATTERN.search(expression)
        if match:
            months = int(match.group(1))
            return self._shift_months(self.reference_date, -months)
        match = self.CN_YEARS_AGO_PATTERN.search(expression)
        if match:
            years = int(match.group(1))
            return f"{self.reference_date.year - years:04d}"

        # 粒度兜底：出现 week/month/year 但无法映射到具体日期
        if any(k in expr for k in ("week", "周", "星期")):
            return self._format_iso_week(self.reference_date)
        if any(k in expr for k in ("month", "月")):
            return self.reference_date.strftime("%Y-%m")
        if any(k in expr for k in ("year", "年")):
            return f"{self.reference_date.year:04d}"

        return None

    def _llm_ground(self, time_expr: str) -> str:
        """使用 LLM 结构化归一化复杂时间表达。"""
        try:
            client = self._get_llm_client()
            ref_str = self.reference_date.strftime("%Y-%m-%d")
            prompt = (
                f"Reference date: {ref_str}\n"
                f"Time expression: \"{time_expr}\"\n\n"
                "Normalize the expression into a medical memory time scope.\n"
                "Output JSON ONLY with schema:\n"
                "{"
                "\"type\":\"day|week|month|year|range|global\","
                "\"value\":\"...\","
                "\"start\":\"YYYY-MM-DD\","
                "\"end\":\"YYYY-MM-DD\""
                "}\n"
                "Rules:\n"
                "- day => value is YYYY-MM-DD\n"
                "- week => value is YYYY-Www\n"
                "- month => value is YYYY-MM\n"
                "- year => value is YYYY\n"
                "- range => set start/end (YYYY-MM-DD)\n"
                "- unresolved => type=global\n"
                "No markdown. No extra keys."
            )
            response = client.chat.completions.create(
                model=LLM_MODEL_FAST,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120,
            )
            result = (response.choices[0].message.content or "").strip()
            parsed = self._parse_llm_output(result)
            if parsed is not None:
                return parsed
        except Exception:
            pass
        return "global"

    def _parse_llm_output(self, raw: str) -> Optional[str]:
        """解析 LLM 结构化输出。"""
        candidate = raw
        if candidate.startswith("```"):
            parts = candidate.split("```")
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if part.startswith("json"):
                    part = part[4:].strip()
                candidate = part
                break

        payload = None
        try:
            payload = json.loads(candidate)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", candidate)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except Exception:
                    payload = None

        if not isinstance(payload, dict):
            return None

        t = str(payload.get("type", "")).strip().lower()
        value = str(payload.get("value", "")).strip()
        start = str(payload.get("start", "")).strip()
        end = str(payload.get("end", "")).strip()

        if t == "global":
            return "global"
        if t == "day":
            normalized = self._normalize_date_str(value)
            if normalized:
                return normalized
        elif t == "week":
            week_match = self.ISO_WEEK_PATTERN.search(value)
            if week_match:
                year = int(week_match.group(1))
                week = int(week_match.group(2))
                if 1 <= week <= 53:
                    return f"{year:04d}-W{week:02d}"
        elif t == "month":
            month_match = self.ISO_MONTH_PATTERN.search(value)
            if month_match:
                year = int(month_match.group(1))
                month = int(month_match.group(2))
                if 1 <= month <= 12:
                    return f"{year:04d}-{month:02d}"
        elif t == "year":
            year_match = self.YEAR_PATTERN.search(value)
            if year_match:
                return year_match.group(1)
        elif t == "range":
            start_iso = self._normalize_date_str(start)
            end_iso = self._normalize_date_str(end)
            if start_iso and end_iso:
                if start_iso <= end_iso:
                    return f"{start_iso}..{end_iso}"
                return f"{end_iso}..{start_iso}"

        return None

    def _resolve_weekday(self, weekday: int, week_offset: int) -> str:
        """根据参考日期解析目标工作日（返回 YYYY-MM-DD）。"""
        anchor = self.reference_date + timedelta(days=7 * week_offset)
        delta = weekday - anchor.weekday()
        return (anchor + timedelta(days=delta)).strftime("%Y-%m-%d")

    @staticmethod
    def _format_iso_week(dt: datetime) -> str:
        iso_year, iso_week, _ = dt.isocalendar()
        return f"{iso_year:04d}-W{iso_week:02d}"

    @staticmethod
    def _safe_date_from_parts(year: int, month: int, day: int) -> Optional[str]:
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    def _normalize_date_str(self, s: str) -> Optional[str]:
        """将 YYYY-MM-DD / YYYY/M/D 统一为 YYYY-MM-DD。"""
        match = self.ISO_DATE_PATTERN.search(s)
        if not match:
            return None
        return self._safe_date_from_parts(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )

    @staticmethod
    def _shift_months(dt: datetime, months: int) -> str:
        """按月偏移并返回 YYYY-MM。"""
        base = dt.year * 12 + (dt.month - 1) + months
        year = base // 12
        month = base % 12 + 1
        return f"{year:04d}-{month:02d}"


# ─── Conflict Detector ───────────────────────────────────────────────────────

class ConflictDetector:
    """
    检测同一 (patient_id, attribute, time_scope) 下的值冲突。

    冲突类型：
        - "value_change"：同一时间范围内，同一属性出现了不同的值。
        - "unit_change"：单位不一致（如 mmol/L vs mg/dL）。
    """

    # 数值比较的容差（相对误差）
    # 仅用于吸收格式差异（如 40 vs 40.0），避免把真实小幅变化吞掉。
    NUMERIC_TOLERANCE = 0.001
    MED_EQUIV_PATTERNS = [
        (re.compile(r"\bonce\s+daily\b", re.IGNORECASE), "qd"),
        (re.compile(r"\bdaily\b", re.IGNORECASE), "qd"),
        (re.compile(r"\btwice\s+daily\b", re.IGNORECASE), "bid"),
        (re.compile(r"\bbid\b", re.IGNORECASE), "bid"),
        (re.compile(r"\bonce\s+at\s+night\b", re.IGNORECASE), "qhs"),
        (re.compile(r"\bat\s+night\b", re.IGNORECASE), "qhs"),
    ]

    def detect(
        self,
        existing: CanonicalMemory,
        new_event: ExtractedEvent,
    ) -> Optional[ConflictRecord]:
        """
        检测新事件与现有记忆之间是否存在冲突。

        Args:
            existing:  现有的 CanonicalMemory 记录。
            new_event: 新的 ExtractedEvent。

        Returns:
            ConflictRecord（如果存在冲突），否则返回 None。
        """
        # 单位冲突
        if (existing.unit and new_event.unit and
                existing.unit.lower() != new_event.unit.lower()):
            return ConflictRecord(
                old_value=existing.value,
                new_value=new_event.value,
                old_provenance=existing.provenance,
                new_provenance=new_event.provenance,
                conflict_type="unit_change",
                detected_at=new_event.time_expr,
            )

        # 值冲突（先做等价判断，减少误冲突）
        if self._equivalent_value(existing.value, new_event.value, existing.attribute):
            return None

        return ConflictRecord(
            old_value=existing.value,
            new_value=new_event.value,
            old_provenance=existing.provenance,
            new_provenance=new_event.provenance,
            conflict_type="value_change",
            detected_at=new_event.time_expr,
        )

        return None

    def _equivalent_value(self, old_value: str, new_value: str, attribute: str) -> bool:
        old_raw = (old_value or "").strip()
        new_raw = (new_value or "").strip()
        if not old_raw or not new_raw:
            return False

        old_norm = self._normalize_text(old_raw)
        new_norm = self._normalize_text(new_raw)
        if old_norm == new_norm:
            return True

        # 数值等价（容差）
        try:
            old_num = float(re.sub(r"[^\d.\-]", "", old_raw))
            new_num = float(re.sub(r"[^\d.\-]", "", new_raw))
            if old_num == 0:
                return new_num == 0
            if abs(old_num - new_num) / abs(old_num) <= self.NUMERIC_TOLERANCE:
                return True
        except (ValueError, ZeroDivisionError):
            pass

        attr = (attribute or "").strip().lower()
        if attr == "medication":
            return self._normalize_medication(old_raw) == self._normalize_medication(new_raw)

        if attr == "primary_diagnosis":
            old_tokens = set(old_norm.split())
            new_tokens = set(new_norm.split())
            if old_tokens and new_tokens and (old_tokens <= new_tokens or new_tokens <= old_tokens):
                return True

        return False

    @staticmethod
    def _normalize_text(text: str) -> str:
        cleaned = re.sub(r"[^\w\s/.-]", " ", text.lower())
        return " ".join(cleaned.split())

    def _normalize_medication(self, text: str) -> str:
        normalized = self._normalize_text(text)
        for pattern, replacement in self.MED_EQUIV_PATTERNS:
            normalized = pattern.sub(replacement, normalized)
        normalized = normalized.replace("milligrams", "mg").replace("milligram", "mg")
        normalized = normalized.replace("units", "u").replace("unit", "u")
        return " ".join(normalized.split())


# ─── Uniqueness Manager ──────────────────────────────────────────────────────

class UniquenessManager:
    """
    M3 核心：基于 scope_policies 对事件进行去重和合并，输出 CanonicalMemory 列表。

    处理流程（对每个 AttributeCluster）：
        1. 对所有事件进行 Time Grounding，将 time_expr -> time_scope（标准日期）。
        2. 按 (attribute, time_scope) 分组。
        3. 对每组内的多条事件，根据 update_policy 进行合并：
           - "unique"：保留置信度最高的一条，其余标记为冲突。
           - "latest"：保留 provenance 最晚的一条（最新轮次）。
           - "append"：保留所有不重复的值。
        4. 输出最终的 CanonicalMemory 列表。
    """

    BUNDLE_STATUS_ACTIVE = "Active"
    BUNDLE_STATUS_SUPERSEDED = "Superseded"
    BUNDLE_STATUS_CONFLICTING = "Conflicting"
    BUNDLE_STATUS_RESOLVED = "Resolved"
    VALID_BUNDLE_STATUSES = {
        BUNDLE_STATUS_ACTIVE,
        BUNDLE_STATUS_SUPERSEDED,
        BUNDLE_STATUS_CONFLICTING,
        BUNDLE_STATUS_RESOLVED,
    }

    CROSS_BUNDLE_RULES = [
        {
            "med_keywords": ["insulin", "insulin glargine"],
            "symptom_keywords": ["hypoglycemia", "low blood sugar", "dizziness", "sweating", "palpitations"],
            "relation": "POTENTIAL_ADVERSE_EFFECT",
            "reason": "Medication_Risk_Heuristic:Hypoglycemia",
            "confidence": 0.72,
        },
        {
            "med_keywords": ["lisinopril", "amlodipine"],
            "symptom_keywords": ["dizziness", "lightheadedness"],
            "relation": "POTENTIAL_ADVERSE_EFFECT",
            "reason": "Medication_Risk_Heuristic:Hypotension",
            "confidence": 0.64,
        },
    ]
    MED_STRONG_ACTION_PATTERNS = [
        re.compile(r"\b(start|prescribe|prescribed|reduce|increase|continue|switch)\b", re.IGNORECASE),
        re.compile(r"\b(once daily|twice daily|bid|tid|qid|qd|qhs|at night)\b", re.IGNORECASE),
    ]
    MED_WEAK_ACTION_PATTERNS = [
        re.compile(r"\b(recommend|might recommend|may recommend|consider)\b", re.IGNORECASE),
        re.compile(r"\b(like|as needed|prn|if .* persists)\b", re.IGNORECASE),
        re.compile(r"\b(antipyretic|pain reliever)\b", re.IGNORECASE),
    ]

    def __init__(
        self,
        dialogue_date: Optional[str] = None,
        enable_time_grounding: bool = True,
        enable_conflict_detection: bool = True,
        missing_time_scope: str = "global",
        max_symptoms_per_scope: Optional[int] = None,
    ):
        """
        Args:
            dialogue_date: 对话参考日期（用于相对时间解析）。
            enable_time_grounding: 是否启用 Time Grounding；关闭时统一使用 global。
            enable_conflict_detection: 是否启用冲突检测。
        """
        self.enable_time_grounding = enable_time_grounding
        self.enable_conflict_detection = enable_conflict_detection
        self.max_symptoms_per_scope = max_symptoms_per_scope
        self.time_grounder = (
            TimeGrounder(
                dialogue_date=dialogue_date,
                missing_time_scope=missing_time_scope,
            )
            if enable_time_grounding
            else None
        )
        self.conflict_detector = ConflictDetector() if enable_conflict_detection else None
        self._event_to_event_bundle: Dict[str, str] = {}
        self._event_to_entity_bundles: Dict[str, Set[str]] = {}
        self._entity_bundle_names: Dict[str, str] = {}
        self._bundle_time_anchor: Dict[str, str] = {}
        self._event_bundle_by_id: Dict[str, object] = {}
        self._entity_bundle_by_id: Dict[str, object] = {}
        self._bundle_graph: Optional[BundleGraph] = None

    def process(
        self,
        clusters: List[AttributeCluster],
        patient_id: str,
        bundle_graph: Optional[BundleGraph] = None,
    ) -> List[CanonicalMemory]:
        """
        将 AttributeCluster 列表转换为 CanonicalMemory 列表。

        Args:
            clusters:   M2 输出的 AttributeCluster 列表。
            patient_id: 患者/对话 ID。

        Returns:
            CanonicalMemory 列表（已去重、已检测冲突）。
        """
        self._index_bundle_graph(bundle_graph)
        all_memories: List[CanonicalMemory] = []

        for cluster in clusters:
            memories = self._process_cluster(cluster, patient_id)
            all_memories.extend(memories)

        self._link_bundles(all_memories)
        return all_memories

    def _index_bundle_graph(self, bundle_graph: Optional[BundleGraph]) -> None:
        self._bundle_graph = bundle_graph
        self._event_to_event_bundle = {}
        self._event_to_entity_bundles = {}
        self._entity_bundle_names = {}
        self._bundle_time_anchor = {}
        self._event_bundle_by_id = {}
        self._entity_bundle_by_id = {}
        if bundle_graph is None:
            return

        for eb in bundle_graph.event_bundles:
            self._event_bundle_by_id[eb.bundle_id] = eb
            if not getattr(eb, "status", ""):
                eb.status = self.BUNDLE_STATUS_ACTIVE
            self._bundle_time_anchor[eb.bundle_id] = eb.time_anchor
            for eid in eb.event_ids:
                self._event_to_event_bundle[eid] = eb.bundle_id

        for ent in bundle_graph.entity_bundles:
            self._entity_bundle_by_id[ent.bundle_id] = ent
            if not getattr(ent, "status", ""):
                ent.status = self.BUNDLE_STATUS_ACTIVE
            self._entity_bundle_names[ent.bundle_id] = ent.canonical_name
            for eid in ent.event_ids:
                self._event_to_entity_bundles.setdefault(eid, set()).add(ent.bundle_id)

        for link in bundle_graph.links:
            if link.relation != "MENTIONS_MEDICATION":
                continue
            event_bundle_id = link.src_bundle_id
            entity_bundle_id = link.dst_bundle_id
            for eid, e_bid in self._event_to_event_bundle.items():
                if e_bid == event_bundle_id:
                    self._event_to_entity_bundles.setdefault(eid, set()).add(entity_bundle_id)

    def _bundle_key_for_event(self, evt: ExtractedEvent) -> str:
        return self._event_to_event_bundle.get(evt.event_id, f"event::{evt.event_id}")

    def _bundle_groups(self, events: List[ExtractedEvent]) -> Dict[str, List[ExtractedEvent]]:
        grouped: Dict[str, List[ExtractedEvent]] = {}
        for evt in events:
            grouped.setdefault(self._bundle_key_for_event(evt), []).append(evt)
        return grouped

    def _bundle_groups_grounded(
        self,
        grounded_events: List[Tuple[str, ExtractedEvent]],
    ) -> Dict[str, List[Tuple[str, ExtractedEvent]]]:
        grouped: Dict[str, List[Tuple[str, ExtractedEvent]]] = {}
        for scope, evt in grounded_events:
            grouped.setdefault(self._bundle_key_for_event(evt), []).append((scope, evt))
        return grouped

    def _bundle_event_priority(self, events: List[ExtractedEvent]) -> Tuple[int, int, float, int]:
        top = max((self._event_priority(e) for e in events), default=(0, 0, 0.0))
        return top[0], top[1], top[2], len(events)

    def _bundle_latest_priority(
        self,
        grounded_events: List[Tuple[str, ExtractedEvent]],
    ) -> Tuple[int, int, int, int, float, int]:
        top = max(
            (self._latest_priority(scope, evt) for scope, evt in grounded_events),
            default=(-1, 1, 0, 0, 0.0),
        )
        return top[0], top[1], top[2], top[3], top[4], len(grounded_events)

    def _representative_event(self, events: List[ExtractedEvent]) -> ExtractedEvent:
        return sorted(events, key=lambda e: self._event_priority(e), reverse=True)[0]

    def _representative_grounded_event(
        self,
        grounded_events: List[Tuple[str, ExtractedEvent]],
    ) -> Tuple[str, ExtractedEvent]:
        return sorted(
            grounded_events,
            key=lambda pair: self._latest_priority(pair[0], pair[1]),
            reverse=True,
        )[0]

    def _append_bundle_evidence(
        self,
        bundle_id: str,
        action: str,
        reason: str,
        extra: Optional[dict] = None,
    ) -> None:
        bundle = self._event_bundle_by_id.get(bundle_id)
        if bundle is None:
            return
        chain = getattr(bundle, "evidence_chain", None)
        if chain is None:
            chain = []
            setattr(bundle, "evidence_chain", chain)
        entry = {
            "action": action,
            "reason": reason,
        }
        if extra:
            entry["extra"] = extra
        chain.append(entry)

    def _set_bundle_status(
        self,
        bundle_id: str,
        new_status: str,
        reason: str,
        extra: Optional[dict] = None,
    ) -> None:
        if new_status not in self.VALID_BUNDLE_STATUSES:
            return
        bundle = self._event_bundle_by_id.get(bundle_id)
        if bundle is None:
            return
        old_status = getattr(bundle, "status", self.BUNDLE_STATUS_ACTIVE)
        if old_status != new_status:
            bundle.status = new_status
            self._append_bundle_evidence(
                bundle_id,
                action="status_transition",
                reason=reason,
                extra={
                    "from": old_status,
                    "to": new_status,
                    **(extra or {}),
                },
            )
        else:
            self._append_bundle_evidence(
                bundle_id,
                action="status_observed",
                reason=reason,
                extra={"status": new_status, **(extra or {})},
            )

    def _apply_bundle_decision(
        self,
        selected_bundle_id: str,
        competing_bundle_ids: List[str],
        policy: str,
        has_conflict: bool,
    ) -> None:
        if not selected_bundle_id:
            return
        if not has_conflict:
            self._set_bundle_status(
                selected_bundle_id,
                self.BUNDLE_STATUS_ACTIVE,
                reason=f"Bundle_Decision:{policy}",
            )
            for bid in competing_bundle_ids:
                self._set_bundle_status(
                    bid,
                    self.BUNDLE_STATUS_RESOLVED,
                    reason=f"Bundle_Decision:{policy}",
                    extra={"selected_bundle_id": selected_bundle_id},
                )
            return

        if policy == "latest":
            self._set_bundle_status(
                selected_bundle_id,
                self.BUNDLE_STATUS_ACTIVE,
                reason="Temporal_Recency",
                extra={"resolved_conflict": True},
            )
            for bid in competing_bundle_ids:
                self._set_bundle_status(
                    bid,
                    self.BUNDLE_STATUS_SUPERSEDED,
                    reason="Temporal_Recency",
                    extra={"selected_bundle_id": selected_bundle_id},
                )
            return

        # unique 场景：同 scope 下差异值通常无法完全确认，标为冲突待解
        self._set_bundle_status(
            selected_bundle_id,
            self.BUNDLE_STATUS_CONFLICTING,
            reason="SameScope_ValueConflict",
            extra={"requires_clarification": True},
        )
        for bid in competing_bundle_ids:
            self._set_bundle_status(
                bid,
                self.BUNDLE_STATUS_CONFLICTING,
                reason="SameScope_ValueConflict",
                extra={"selected_bundle_id": selected_bundle_id},
            )

    def _process_cluster(
        self,
        cluster: AttributeCluster,
        patient_id: str,
    ) -> List[CanonicalMemory]:
        """处理单个 AttributeCluster，输出该属性的 CanonicalMemory 列表。"""
        attribute = cluster.canonical_attribute
        policy = cluster.update_policy

        # Step 1: Time Grounding（可关闭）
        grounded_events: List[Tuple[str, ExtractedEvent]] = []
        for evt in cluster.events:
            if self.enable_time_grounding and self.time_grounder is not None:
                time_scope = self.time_grounder.ground(evt.time_expr)
            else:
                time_scope = "global"
            grounded_events.append((time_scope, evt))

        # Step 2: latest 策略是全局覆盖，不按 time_scope 切分
        if policy == "latest":
            return self._merge_latest(
                grounded_events,
                attribute=attribute,
                time_scope="global",
                patient_id=patient_id,
            )

        # Step 3: 非 latest 策略按 time_scope 分组
        scope_groups: Dict[str, List[Tuple[str, ExtractedEvent]]] = {}
        for time_scope, evt in grounded_events:
            if time_scope not in scope_groups:
                scope_groups[time_scope] = []
            scope_groups[time_scope].append((time_scope, evt))

        # Step 4: 按 update_policy 合并
        memories: List[CanonicalMemory] = []

        for time_scope, group in scope_groups.items():
            evts = [e for _, e in group]

            if policy == "unique":
                memories.extend(
                    self._merge_unique(evts, attribute, time_scope, patient_id)
                )
            elif policy == "append":
                memories.extend(
                    self._merge_append(evts, attribute, time_scope, patient_id)
                )
            else:
                memories.extend(
                    self._merge_unique(evts, attribute, time_scope, patient_id)
                )

        return memories

    def _merge_unique(
        self,
        events: List[ExtractedEvent],
        attribute: str,
        time_scope: str,
        patient_id: str,
    ) -> List[CanonicalMemory]:
        """
        "unique" 策略：同一 (attribute, time_scope) 只保留一条记录。
        如果有多条不同值，保留置信度最高的，并记录冲突。
        """
        if not events:
            return []

        # 信息团优先：先在 event bundle 层选主候选，再在 bundle 内选代表事件
        grouped = self._bundle_groups(events)
        bundle_candidates = []
        for bundle_id, group_events in grouped.items():
            rep = self._representative_event(group_events)
            bundle_candidates.append(
                {
                    "bundle_id": bundle_id,
                    "events": group_events,
                    "rep": rep,
                    "rank": self._bundle_event_priority(group_events),
                }
            )
        bundle_candidates.sort(key=lambda c: c["rank"], reverse=True)
        selected = bundle_candidates[0]
        best: ExtractedEvent = selected["rep"]

        conflict_history: List[ConflictRecord] = []
        conflict_flag = False

        # 冲突按 bundle 级比较，避免同 bundle 内重复表达导致的误冲突放大
        if self.enable_conflict_detection and self.conflict_detector is not None:
            seen_values = {best.value.strip().lower()}
            for cand in bundle_candidates[1:]:
                evt = cand["rep"]
                if evt.value.strip().lower() in seen_values:
                    continue
                temp_mem = CanonicalMemory(
                    patient_id=patient_id,
                    attribute=attribute,
                    value=best.value,
                    unit=best.unit,
                    time_scope=time_scope,
                    confidence=best.confidence,
                    provenance=best.provenance,
                    conflict_flag=False,
                    conflict_history=[],
                    update_policy="unique",
                )
                conflict = self.conflict_detector.detect(temp_mem, evt)
                if conflict:
                    conflict_history.append(conflict)
                    conflict_flag = True
                seen_values.add(evt.value.strip().lower())

        # 合并所有来源的 provenance
        all_provenance = []
        seen_prov = set()
        for evt in events:
            for p in evt.provenance:
                if p not in seen_prov:
                    all_provenance.append(p)
                    seen_prov.add(p)
        all_provenance.sort()
        self._apply_bundle_decision(
            selected_bundle_id=selected["bundle_id"],
            competing_bundle_ids=[c["bundle_id"] for c in bundle_candidates[1:]],
            policy="unique",
            has_conflict=conflict_flag,
        )

        version_events = [cand["rep"] for cand in bundle_candidates]
        versions = self._build_versions_from_events(
            events_sorted=version_events,
            selected_event_id=best.event_id,
            default_scope=time_scope,
        )
        qualifiers = self._build_bundle_qualifiers(
            events=events,
            attribute=attribute,
            value_versions=versions,
            selected_bundle_id=selected["bundle_id"],
            competing_bundle_ids=[c["bundle_id"] for c in bundle_candidates[1:]],
            decision_level="bundle",
        )

        return [CanonicalMemory(
            patient_id=patient_id,
            attribute=attribute,
            value=best.value,
            unit=best.unit,
            time_scope=time_scope,
            confidence=best.confidence,
            provenance=all_provenance,
            conflict_flag=conflict_flag,
            conflict_history=conflict_history,
            update_policy="unique",
            qualifiers=qualifiers,
        )]

    def _merge_latest(
        self,
        grounded_events: List[Tuple[str, ExtractedEvent]],
        attribute: str,
        time_scope: str,
        patient_id: str,
    ) -> List[CanonicalMemory]:
        """
        "latest" 策略：保留来自最晚轮次的记录（最新信息优先）。
        """
        if not grounded_events:
            return []

        events = [evt for _, evt in grounded_events]

        # 信息团优先：先在 bundle 层按“时间+来源优先级”排序，再选 bundle 内代表记录
        grouped = self._bundle_groups_grounded(grounded_events)
        bundle_candidates = []
        for bundle_id, group in grouped.items():
            rep_scope, rep_evt = self._representative_grounded_event(group)
            bundle_candidates.append(
                {
                    "bundle_id": bundle_id,
                    "group": group,
                    "rep_scope": rep_scope,
                    "rep_evt": rep_evt,
                    "rank": self._bundle_latest_priority(group),
                }
            )
        bundle_candidates.sort(key=lambda c: c["rank"], reverse=True)
        selected = bundle_candidates[0]
        latest = selected["rep_evt"]
        latest_scope = selected["rep_scope"]

        conflict_history: List[ConflictRecord] = []
        conflict_flag = False

        # 冲突按 bundle 级比较，减少同一事件团内噪声的误冲突
        if self.enable_conflict_detection and self.conflict_detector is not None:
            seen_values = {latest.value.strip().lower()}
            for cand in bundle_candidates[1:]:
                evt = cand["rep_evt"]
                if evt.value.strip().lower() in seen_values:
                    continue
                temp_mem = CanonicalMemory(
                    patient_id=patient_id,
                    attribute=attribute,
                    value=latest.value,
                    unit=latest.unit,
                    time_scope=time_scope,
                    confidence=latest.confidence,
                    provenance=latest.provenance,
                    conflict_flag=False,
                    conflict_history=[],
                    update_policy="latest",
                )
                conflict = self.conflict_detector.detect(temp_mem, evt)
                if conflict:
                    conflict_history.append(conflict)
                    conflict_flag = True
                seen_values.add(evt.value.strip().lower())

        all_provenance = sorted({p for evt in events for p in evt.provenance})
        self._apply_bundle_decision(
            selected_bundle_id=selected["bundle_id"],
            competing_bundle_ids=[c["bundle_id"] for c in bundle_candidates[1:]],
            policy="latest",
            has_conflict=conflict_flag,
        )

        # 为时间推理构建 medication 时间轴（不改变评测主键，仍保留 global）
        timeline = []
        if attribute == "medication":
            timeline = self._build_medication_timeline(grounded_events)

        latest_start, latest_end, _ = scope_to_interval(latest_scope)
        if latest_start is None and latest_end is None:
            latest_start = None
            latest_end = None
        if attribute == "medication":
            latest_end = None

        version_pairs = [(cand["rep_scope"], cand["rep_evt"]) for cand in bundle_candidates]
        versions = self._build_versions_from_grounded_events(
            grounded_events=version_pairs,
            selected_event_id=latest.event_id,
        )
        qualifiers = self._build_bundle_qualifiers(
            events=events,
            attribute=attribute,
            value_versions=versions,
            selected_bundle_id=selected["bundle_id"],
            competing_bundle_ids=[c["bundle_id"] for c in bundle_candidates[1:]],
            decision_level="bundle",
        )
        if timeline:
            qualifiers["med_timeline"] = timeline

        return [CanonicalMemory(
            patient_id=patient_id,
            attribute=attribute,
            value=latest.value,
            unit=latest.unit,
            time_scope=time_scope,
            confidence=latest.confidence,
            provenance=all_provenance,
            conflict_flag=conflict_flag,
            conflict_history=conflict_history,
            update_policy="latest",
            start_time=latest_start,
            end_time=latest_end,
            is_ongoing=True,
            qualifiers=qualifiers,
        )]

    @staticmethod
    def _speaker_priority(speaker: str) -> int:
        sp = (speaker or "").strip().lower()
        if sp == "doctor":
            return 2
        if sp == "patient":
            return 1
        return 0

    @staticmethod
    def _max_turn(evt: ExtractedEvent) -> int:
        return max(evt.provenance) if evt.provenance else 0

    def _medication_intent_priority(self, evt: ExtractedEvent) -> int:
        if evt.attribute != "medication":
            return 1
        text = f"{evt.value} {evt.raw_text_snippet}".strip().lower()
        if any(p.search(text) for p in self.MED_STRONG_ACTION_PATTERNS):
            return 2
        if any(p.search(text) for p in self.MED_WEAK_ACTION_PATTERNS):
            return 0
        return 1

    def _event_priority(self, evt: ExtractedEvent) -> Tuple[int, int, float]:
        return (
            self._max_turn(evt),
            self._speaker_priority(evt.speaker),
            evt.confidence,
        )

    @staticmethod
    def _scope_end_ordinal(scope: str) -> int:
        _, end, _ = scope_to_interval(scope)
        if not end:
            return -1
        try:
            return datetime.strptime(end, "%Y-%m-%d").toordinal()
        except ValueError:
            return -1

    def _latest_priority(self, scope: str, evt: ExtractedEvent) -> Tuple[int, int, int, int, float]:
        scope_rank = self._scope_end_ordinal(scope)
        intent_rank = self._medication_intent_priority(evt)
        # medication 的 latest 先看“处方意图”再看时间，减少临时建议药覆盖长期方案
        if evt.attribute == "medication":
            return (
                intent_rank,
                scope_rank,
                self._max_turn(evt),
                self._speaker_priority(evt.speaker),
                evt.confidence,
            )
        return (
            scope_rank,
            intent_rank,
            self._max_turn(evt),
            self._speaker_priority(evt.speaker),
            evt.confidence,
        )

    def _build_medication_timeline(
        self,
        grounded_events: List[Tuple[str, ExtractedEvent]],
    ) -> List[dict]:
        """
        构建药物时间轴：
        每个条目表示某个药物值在一段时间内生效，用于后续 query_meds_on 推理。
        """
        if not grounded_events:
            return []

        def max_turn(evt: ExtractedEvent) -> int:
            return max(evt.provenance) if evt.provenance else 0

        def start_key(scope: str, evt: ExtractedEvent) -> Tuple[str, int]:
            start, _, _ = scope_to_interval(scope)
            return (start or "9999-12-31", max_turn(evt))

        items = sorted(grounded_events, key=lambda x: start_key(x[0], x[1]))
        timeline: List[dict] = []

        for scope, evt in items:
            start, end, precision = scope_to_interval(scope)
            entry = {
                "value": evt.value,
                "start_time": start,
                "end_time": end,
                "time_precision": precision,
                "provenance": sorted(set(evt.provenance)),
            }

            if not timeline:
                timeline.append(entry)
                continue

            prev = timeline[-1]
            # 同药重复提及：合并时间范围
            if prev["value"].strip().lower() == evt.value.strip().lower():
                prev_start = prev.get("start_time")
                prev_end = prev.get("end_time")
                if start and (not prev_start or start < prev_start):
                    prev["start_time"] = start
                if end and (not prev_end or end > prev_end):
                    prev["end_time"] = end
                prev["provenance"] = sorted(set((prev.get("provenance") or []) + entry["provenance"]))
                continue

            # 新药开始时，将上个条目闭区间到前一天（保证区间连续）
            if start:
                try:
                    prev_end_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=1)
                    prev_end = prev_end_dt.strftime("%Y-%m-%d")
                    if not prev.get("end_time") or prev["end_time"] < prev_end:
                        prev["end_time"] = prev_end
                except ValueError:
                    pass
            timeline.append(entry)

        # 最新条目视为 ongoing
        if timeline:
            timeline[-1]["is_ongoing"] = True
            timeline[-1]["end_time"] = None

        return timeline

    def _merge_append(
        self,
        events: List[ExtractedEvent],
        attribute: str,
        time_scope: str,
        patient_id: str,
    ) -> List[CanonicalMemory]:
        """
        "append" 策略：保留所有不重复的值（如症状列表）。
        """
        if not events:
            return []

        # 先按值去重；同值保留优先级更高记录
        seen_values: Dict[str, ExtractedEvent] = {}
        bucket_counts: Dict[str, int] = {}
        for evt in events:
            key = evt.value.strip().lower()
            bucket_counts[key] = bucket_counts.get(key, 0) + 1
            if key not in seen_values or self._event_priority(evt) > self._event_priority(seen_values[key]):
                seen_values[key] = evt

        items = list(seen_values.items())
        # 症状在 benchmark 中通常是主要主诉，默认可限制为每个 scope 最多 N 条
        if attribute == "symptom" and self.max_symptoms_per_scope is not None:
            items = sorted(
                items,
                key=lambda kv: (
                    bucket_counts.get(kv[0], 1),
                    self._speaker_priority(kv[1].speaker),
                    kv[1].confidence,
                ),
                reverse=True,
            )[: max(0, self.max_symptoms_per_scope)]

        memories = []
        for _, evt in items:
            versions = self._build_versions_from_events(
                events_sorted=[evt],
                selected_event_id=evt.event_id,
                default_scope=time_scope,
            )
            qualifiers = self._build_bundle_qualifiers(
                events=[evt],
                attribute=attribute,
                value_versions=versions,
                selected_bundle_id=self._event_to_event_bundle.get(evt.event_id, ""),
                decision_level="bundle",
            )
            memories.append(CanonicalMemory(
                patient_id=patient_id,
                attribute=attribute,
                value=evt.value,
                unit=evt.unit,
                time_scope=time_scope,
                confidence=evt.confidence,
                provenance=sorted(set(evt.provenance)),
                conflict_flag=False,
                conflict_history=[],
                update_policy="append",
                qualifiers=qualifiers,
            ))
        return memories

    def _build_versions_from_events(
        self,
        events_sorted: List[ExtractedEvent],
        selected_event_id: str,
        default_scope: str,
    ) -> List[dict]:
        versions: List[dict] = []
        seen_values = set()
        for evt in events_sorted:
            key = evt.value.strip().lower()
            if key in seen_values:
                continue
            seen_values.add(key)
            versions.append(
                {
                    "value": evt.value,
                    "time_scope": default_scope,
                    "provenance": sorted(set(evt.provenance)),
                    "confidence": evt.confidence,
                    "speaker": evt.speaker,
                    "event_bundle_id": self._event_to_event_bundle.get(evt.event_id, ""),
                    "entity_bundle_ids": sorted(self._event_to_entity_bundles.get(evt.event_id, set())),
                    "is_selected": evt.event_id == selected_event_id,
                }
            )
        return versions

    def _build_versions_from_grounded_events(
        self,
        grounded_events: List[Tuple[str, ExtractedEvent]],
        selected_event_id: str,
    ) -> List[dict]:
        versions: List[dict] = []
        seen = set()
        for scope, evt in grounded_events:
            key = (evt.value.strip().lower(), scope)
            if key in seen:
                continue
            seen.add(key)
            versions.append(
                {
                    "value": evt.value,
                    "time_scope": scope,
                    "provenance": sorted(set(evt.provenance)),
                    "confidence": evt.confidence,
                    "speaker": evt.speaker,
                    "event_bundle_id": self._event_to_event_bundle.get(evt.event_id, ""),
                    "entity_bundle_ids": sorted(self._event_to_entity_bundles.get(evt.event_id, set())),
                    "is_selected": evt.event_id == selected_event_id,
                }
            )
        return versions

    def _build_bundle_qualifiers(
        self,
        events: List[ExtractedEvent],
        attribute: str,
        value_versions: Optional[List[dict]] = None,
        selected_bundle_id: str = "",
        competing_bundle_ids: Optional[List[str]] = None,
        decision_level: str = "",
    ) -> dict:
        event_bundle_ids = sorted(
            {
                self._event_to_event_bundle[e.event_id]
                for e in events
                if e.event_id in self._event_to_event_bundle
            }
        )
        entity_bundle_ids = sorted(
            {
                ent
                for e in events
                for ent in self._event_to_entity_bundles.get(e.event_id, set())
            }
        )

        qualifiers: dict = {}
        if event_bundle_ids:
            qualifiers["event_bundle_ids"] = event_bundle_ids
        if entity_bundle_ids:
            qualifiers["entity_bundle_ids"] = entity_bundle_ids
        if attribute == "medication" and len(entity_bundle_ids) == 1:
            ent_id = entity_bundle_ids[0]
            qualifiers["medication_entity_id"] = ent_id
            if ent_id in self._entity_bundle_names:
                qualifiers["medication_entity_name"] = self._entity_bundle_names[ent_id]
        if selected_bundle_id:
            qualifiers["selected_event_bundle_id"] = selected_bundle_id
            anchor = self._bundle_time_anchor.get(selected_bundle_id)
            if anchor:
                qualifiers["selected_event_bundle_time_anchor"] = anchor
            selected_bundle = self._event_bundle_by_id.get(selected_bundle_id)
            if selected_bundle is not None:
                qualifiers["selected_event_bundle_status"] = getattr(
                    selected_bundle, "status", self.BUNDLE_STATUS_ACTIVE
                )
        if competing_bundle_ids:
            qualifiers["competing_event_bundle_ids"] = sorted(set(competing_bundle_ids))
        if decision_level:
            qualifiers["decision_level"] = decision_level
        if value_versions:
            qualifiers["value_versions"] = value_versions
        return qualifiers

    @staticmethod
    def _normalize_text_key(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _selected_bundle_id_from_memory(self, mem: CanonicalMemory) -> str:
        qualifiers = mem.qualifiers or {}
        bid = qualifiers.get("selected_event_bundle_id", "")
        if isinstance(bid, str):
            return bid
        return ""

    def _append_cross_bundle_link(
        self,
        src_bundle_id: str,
        dst_bundle_id: str,
        relation: str,
        reason: str,
        confidence: float,
        metadata: Optional[dict] = None,
    ) -> None:
        if self._bundle_graph is None:
            return
        existing = {
            (l.src_bundle_id, l.dst_bundle_id, l.relation)
            for l in self._bundle_graph.links
        }
        key = (src_bundle_id, dst_bundle_id, relation)
        if key in existing:
            return
        from src.uniq_cluster_memory.m2_clustering import BundleLink

        self._bundle_graph.links.append(
            BundleLink(
                src_bundle_id=src_bundle_id,
                dst_bundle_id=dst_bundle_id,
                relation=relation,
                confidence=round(confidence, 3),
                reason=reason,
                metadata=metadata or {},
            )
        )
        self._append_bundle_evidence(
            src_bundle_id,
            action="link_out",
            reason=reason,
            extra={"dst": dst_bundle_id, "relation": relation},
        )
        self._append_bundle_evidence(
            dst_bundle_id,
            action="link_in",
            reason=reason,
            extra={"src": src_bundle_id, "relation": relation},
        )

    def _link_bundles(self, memories: List[CanonicalMemory]) -> None:
        """
        跨 Bundle 关联推理（Cross-Bundle Reasoning）：
        在 M3 合并后，基于轻量规则连接“药物团 -> 症状团”等潜在关系。
        """
        if self._bundle_graph is None or not memories:
            return

        meds: List[Tuple[str, str, CanonicalMemory]] = []
        symptoms: List[Tuple[str, str, CanonicalMemory]] = []

        for mem in memories:
            bid = self._selected_bundle_id_from_memory(mem)
            if not bid:
                continue
            value_key = self._normalize_text_key(mem.value)
            if mem.attribute == "medication":
                meds.append((bid, value_key, mem))
            elif mem.attribute == "symptom":
                symptoms.append((bid, value_key, mem))

        for med_bid, med_value, med_mem in meds:
            for sym_bid, sym_value, sym_mem in symptoms:
                if med_bid == sym_bid:
                    continue
                for rule in self.CROSS_BUNDLE_RULES:
                    if not any(k in med_value for k in rule["med_keywords"]):
                        continue
                    if not any(k in sym_value for k in rule["symptom_keywords"]):
                        continue
                    self._append_cross_bundle_link(
                        src_bundle_id=med_bid,
                        dst_bundle_id=sym_bid,
                        relation=rule["relation"],
                        reason=rule["reason"],
                        confidence=float(rule["confidence"]),
                        metadata={
                            "medication_value": med_mem.value,
                            "symptom_value": sym_mem.value,
                        },
                    )
                    break
