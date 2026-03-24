"""
m3_uniqueness/time_grounder.py
==============================
Time Grounder：将 M1 抽取的原始时间表达（time_expr）解析为标准日期（ISO 8601）。

策略优先级：
    1. 规则匹配：识别显式日期/区间/周/月/年表达（最快，无 LLM 调用）。
    2. 规则推算：处理中英文相对时间表达（"yesterday", "上周一" 等）。
    3. LLM 归一化：对复杂表达使用 Qwen-Turbo 输出结构化规范化结果。
    4. 回退：无法解析时返回 "global"。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Dict, Optional

from src.uniq_cluster_memory.utils.llm_client import get_llm_client, LLM_MODEL_FAST


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
        """将时间表达解析为标准 time_scope 字符串。"""
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
