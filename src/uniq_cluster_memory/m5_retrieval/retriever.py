"""
m5_retrieval/retrieval.py
==========================
M5 模块：查询感知的混合检索器。

核心改进：
1. 查询意图解析：自动识别 query 中的属性/时间/最新值偏好。
2. 动态权重融合：结构化、语义、时间新近度按 query 自适应加权。
3. 批量向量计算：一次编码 query + memory，避免逐条 encode 的高延迟。
4. 冲突感知排序：默认轻微下调 conflict 记录；若 query 明确问冲突则反向保留。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.uniq_cluster_memory.schema import CanonicalMemory


ATTRIBUTE_KEYWORDS: Dict[str, List[str]] = {
    "blood_glucose": ["blood glucose", "blood sugar", "glucose", "血糖"],
    "blood_pressure_sys": ["systolic", "sbp", "blood pressure", "收缩压", "血压"],
    "blood_pressure_dia": ["diastolic", "dbp", "blood pressure", "舒张压", "血压"],
    "heart_rate": ["heart rate", "pulse", "hr", "心率", "脉搏"],
    "body_temperature": ["temperature", "temp", "fever", "体温", "发烧"],
    "hemoglobin": ["hemoglobin", "hgb", "hb", "血红蛋白"],
    "primary_diagnosis": ["diagnosis", "disease", "condition", "诊断", "疾病"],
    "medication": ["medication", "medicine", "drug", "prescription", "用药", "药物", "处方"],
    "symptom": ["symptom", "complaint", "symptoms", "症状", "不适"],
}

ATTRIBUTE_ALIAS_MAP: Dict[str, str] = {
    "blood_sugar": "blood_glucose",
    "glucose": "blood_glucose",
    "bg": "blood_glucose",
    "systolic_blood_pressure": "blood_pressure_sys",
    "blood_pressure_systolic": "blood_pressure_sys",
    "bp_systolic": "blood_pressure_sys",
    "sbp": "blood_pressure_sys",
    "diastolic_blood_pressure": "blood_pressure_dia",
    "blood_pressure_diastolic": "blood_pressure_dia",
    "bp_diastolic": "blood_pressure_dia",
    "dbp": "blood_pressure_dia",
    "pulse": "heart_rate",
    "hr": "heart_rate",
    "temperature": "body_temperature",
    "temp": "body_temperature",
    "hb": "hemoglobin",
    "hgb": "hemoglobin",
    "diagnosis": "primary_diagnosis",
    "drug": "medication",
    "medicine": "medication",
}

LATEST_CUES = {
    "latest", "most recent", "current", "newest", "recent", "now", "currently",
    "最新", "最近", "当前", "现在",
}
CONFLICT_CUES = {"conflict", "inconsistent", "change", "changed", "冲突", "矛盾", "变化", "变更"}


@dataclass
class QueryIntent:
    query_attribute: Optional[str] = None
    query_time_scope: Optional[str] = None
    wants_latest: bool = False
    asks_conflict: bool = False


@dataclass
class RetrievalResult:
    """单条检索结果。"""
    memory: CanonicalMemory
    score: float
    struct_score: float
    semantic_score: float
    recency_score: float
    conflict_warning: bool      # 是否触发冲突警告
    conflict_detail: str        # 冲突详情（如有）

    def to_context_str(self) -> str:
        """将记忆记录格式化为 LLM 上下文字符串。"""
        parts = [
            f"[{self.memory.attribute}]",
            f"Value: {self.memory.value}",
        ]
        if self.memory.unit:
            parts.append(f"Unit: {self.memory.unit}")
        if self.memory.time_scope != "global":
            parts.append(f"Date: {self.memory.time_scope}")
        if self.conflict_warning:
            parts.append(f"[CONFLICT] {self.conflict_detail}")
        return " | ".join(parts)


class HybridMemoryRetriever:
    """
    M5 混合记忆检索器。

    分数定义：
        score = w_struct * struct_score + w_semantic * semantic_score + w_recency * recency_score
    """

    ISO_DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
    ISO_WEEK_PATTERN = re.compile(r"\b(\d{4})-?W(\d{1,2})\b", re.IGNORECASE)
    ISO_MONTH_PATTERN = re.compile(r"\b(\d{4})-(\d{2})\b")
    YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
    RANGE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})")

    def __init__(
        self,
        w_struct: float = 0.7,
        top_k: int = 5,
        use_embedding: bool = True,
        w_recency: float = 0.1,
    ):
        """
        Args:
            w_struct:      结构化匹配基础权重（0.0-1.0）。
            top_k:         返回的最大记录数。
            use_embedding: 是否使用 Embedding 语义检索。
            w_recency:     时间新近度基础权重（0.0-1.0）。
        """
        self.top_k = top_k
        self.use_embedding = use_embedding
        self._embedder = None
        self._embedding_cache: Dict[str, np.ndarray] = {}

        self.base_w_struct = max(0.0, min(1.0, w_struct))
        self.base_w_recency = max(0.0, min(1.0, w_recency))
        self.base_w_semantic = max(0.0, 1.0 - self.base_w_struct - self.base_w_recency)

    def _get_embedder(self):
        """懒加载 MiniLM Embedding 模型。"""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    @staticmethod
    def _normalize_attribute(attr: str) -> str:
        normalized = attr.strip().lower().replace(" ", "_").replace("-", "_")
        return ATTRIBUTE_ALIAS_MAP.get(normalized, normalized)

    def _parse_query_intent(
        self,
        query: str,
        memories: List[CanonicalMemory],
        query_attribute: Optional[str],
        query_time_scope: Optional[str],
    ) -> QueryIntent:
        q = (query or "").strip().lower()
        intent = QueryIntent()

        # 1) 属性意图
        if query_attribute:
            intent.query_attribute = self._normalize_attribute(query_attribute)
        else:
            best_attr = None
            best_score = 0
            for attr, keywords in ATTRIBUTE_KEYWORDS.items():
                score = sum(1 for kw in keywords if kw in q)
                if score > best_score:
                    best_score = score
                    best_attr = attr
            if best_attr:
                intent.query_attribute = best_attr
            else:
                # 兜底：如果 query 没有明显线索，优先对齐 memory 中唯一属性
                attrs = {self._normalize_attribute(m.attribute) for m in memories}
                if len(attrs) == 1:
                    intent.query_attribute = next(iter(attrs))

        # 2) 时间意图
        if query_time_scope:
            intent.query_time_scope = query_time_scope.strip().lower()
        else:
            explicit = self._extract_time_scope_from_query(q)
            if explicit:
                intent.query_time_scope = explicit

        # 3) 最新值 / 冲突意图
        intent.wants_latest = any(cue in q for cue in LATEST_CUES)
        intent.asks_conflict = any(cue in q for cue in CONFLICT_CUES)
        return intent

    def _extract_time_scope_from_query(self, q: str) -> Optional[str]:
        # 绝对日期
        match = self.ISO_DATE_PATTERN.search(q)
        if match:
            return match.group(1)

        # 周 / 月 / 年
        match = self.ISO_WEEK_PATTERN.search(q)
        if match:
            year = int(match.group(1))
            week = int(match.group(2))
            if 1 <= week <= 53:
                return f"{year:04d}-W{week:02d}"

        match = self.ISO_MONTH_PATTERN.search(q)
        if match and not self.ISO_DATE_PATTERN.search(q):
            year, month = int(match.group(1)), int(match.group(2))
            if 1 <= month <= 12:
                return f"{year:04d}-{month:02d}"

        match = self.YEAR_PATTERN.search(q)
        if match and ("year" in q or "年" in q):
            return match.group(1)

        # 常见相对时间（轻量规则）
        today = datetime.today()
        if "yesterday" in q or "昨天" in q:
            return (today - timedelta(days=1)).strftime("%Y-%m-%d")
        if "today" in q or "今天" in q:
            return today.strftime("%Y-%m-%d")
        if "last week" in q or "上周" in q:
            t = today - timedelta(days=7)
            iso_year, iso_week, _ = t.isocalendar()
            return f"{iso_year:04d}-W{iso_week:02d}"
        if "this week" in q or "本周" in q or "这周" in q:
            iso_year, iso_week, _ = today.isocalendar()
            return f"{iso_year:04d}-W{iso_week:02d}"
        if "last month" in q or "上个月" in q:
            year = today.year
            month = today.month - 1
            if month == 0:
                year -= 1
                month = 12
            return f"{year:04d}-{month:02d}"
        if "this month" in q or "本月" in q or "这个月" in q:
            return today.strftime("%Y-%m")

        return None

    def _memory_to_text(self, memory: CanonicalMemory) -> str:
        return " ".join(
            [
                memory.attribute.replace("_", " "),
                memory.value,
                memory.unit,
                memory.time_scope,
                memory.update_policy,
            ]
        ).strip()

    def _semantic_scores(self, query: str, memories: List[CanonicalMemory]) -> List[float]:
        if not self.use_embedding or not memories:
            return [0.0 for _ in memories]
        try:
            embedder = self._get_embedder()
            query_vec = np.asarray(
                embedder.encode([query], normalize_embeddings=True)[0],
                dtype=np.float32,
            )

            texts = [self._memory_to_text(m) for m in memories]
            missing_texts = [t for t in texts if t not in self._embedding_cache]
            if missing_texts:
                encoded = embedder.encode(missing_texts, normalize_embeddings=True)
                for text, vec in zip(missing_texts, encoded):
                    self._embedding_cache[text] = np.asarray(vec, dtype=np.float32)

            matrix = np.vstack([self._embedding_cache[t] for t in texts]).astype(np.float32)
            scores = matrix @ query_vec
            return [float(s) for s in scores.tolist()]
        except Exception:
            return [0.0 for _ in memories]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))

    def _struct_score(
        self,
        memory: CanonicalMemory,
        intent: QueryIntent,
        query: str,
    ) -> float:
        score = 0.0
        mem_attr = self._normalize_attribute(memory.attribute)

        if intent.query_attribute:
            qa = self._normalize_attribute(intent.query_attribute)
            if qa == mem_attr:
                score += 0.65
            elif qa in mem_attr or mem_attr in qa:
                score += 0.35
        else:
            # 若未显式识别到属性，但 query 文本提到了属性名，也给予弱结构化加分
            attr_text = mem_attr.replace("_", " ")
            if attr_text in query.lower():
                score += 0.25

        if intent.query_time_scope:
            score += 0.35 * self._time_match_score(memory.time_scope, intent.query_time_scope)
        elif intent.wants_latest and memory.update_policy == "latest":
            score += 0.20

        # 值层面的词汇重叠弱加分，提升细粒度命中
        q_tokens = self._tokenize(query)
        v_tokens = self._tokenize(memory.value)
        if q_tokens and v_tokens:
            overlap = len(q_tokens & v_tokens) / len(v_tokens)
            score += 0.10 * overlap

        return min(score, 1.0)

    def _time_match_score(self, memory_scope: str, target_scope: str) -> float:
        ms = (memory_scope or "").strip().lower()
        ts = (target_scope or "").strip().lower()
        if not ts:
            return 0.0
        if ms == ts:
            return 1.0
        if ms == "global":
            return 0.3

        m_range = self._scope_to_range(ms)
        t_range = self._scope_to_range(ts)
        if m_range is None or t_range is None:
            return 0.0

        m_start, m_end = m_range
        t_start, t_end = t_range
        # 时间范围相交
        if max(m_start, t_start) <= min(m_end, t_end):
            return 0.9

        # 不相交时按最近距离衰减
        if m_end < t_start:
            gap = (t_start - m_end).days
        else:
            gap = (m_start - t_end).days
        return max(0.0, 0.8 * np.exp(-gap / 30.0))

    def _scope_to_range(self, scope: str) -> Optional[Tuple[datetime, datetime]]:
        if scope == "global":
            return None

        range_match = self.RANGE_PATTERN.fullmatch(scope)
        if range_match:
            try:
                start = datetime.strptime(range_match.group(1), "%Y-%m-%d")
                end = datetime.strptime(range_match.group(2), "%Y-%m-%d")
                if start <= end:
                    return start, end
                return end, start
            except ValueError:
                return None

        if self.ISO_DATE_PATTERN.fullmatch(scope):
            try:
                d = datetime.strptime(scope, "%Y-%m-%d")
                return d, d
            except ValueError:
                return None

        week_match = self.ISO_WEEK_PATTERN.fullmatch(scope)
        if week_match:
            try:
                year = int(week_match.group(1))
                week = int(week_match.group(2))
                start = datetime.fromisocalendar(year, week, 1)
                end = datetime.fromisocalendar(year, week, 7)
                return start, end
            except ValueError:
                return None

        month_match = self.ISO_MONTH_PATTERN.fullmatch(scope)
        if month_match:
            year = int(month_match.group(1))
            month = int(month_match.group(2))
            if 1 <= month <= 12:
                start = datetime(year, month, 1)
                if month == 12:
                    end = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    end = datetime(year, month + 1, 1) - timedelta(days=1)
                return start, end

        if self.YEAR_PATTERN.fullmatch(scope):
            year = int(scope)
            return datetime(year, 1, 1), datetime(year, 12, 31)

        return None

    def _recency_score(self, memory: CanonicalMemory, intent: QueryIntent) -> float:
        scope = (memory.time_scope or "global").strip().lower()
        today = datetime.today()

        if scope == "global":
            base = 0.40 if memory.update_policy == "latest" else 0.15
        else:
            scope_range = self._scope_to_range(scope)
            if scope_range is None:
                base = 0.0
            else:
                _, end = scope_range
                delta_days = max((today - end).days, 0)
                base = float(1.0 / (1.0 + delta_days / 30.0))

        if intent.query_time_scope:
            base = max(base, self._time_match_score(memory.time_scope, intent.query_time_scope))
        if intent.wants_latest and memory.update_policy == "latest":
            base = min(1.0, base + 0.15)
        return max(0.0, min(1.0, base))

    def _dynamic_weights(self, intent: QueryIntent) -> Tuple[float, float, float]:
        w_struct = self.base_w_struct
        w_semantic = self.base_w_semantic if self.use_embedding else 0.0
        w_recency = self.base_w_recency

        if intent.query_attribute:
            w_struct += 0.10
        if intent.query_time_scope:
            w_struct += 0.05
            w_recency += 0.05
        if intent.wants_latest:
            w_recency += 0.12
        if intent.asks_conflict:
            w_struct += 0.05

        total = w_struct + w_semantic + w_recency
        if total <= 0:
            return 1.0, 0.0, 0.0
        return w_struct / total, w_semantic / total, w_recency / total

    def retrieve(
        self,
        query: str,
        memories: List[CanonicalMemory],
        query_attribute: Optional[str] = None,
        query_time_scope: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        从 CanonicalMemory 列表中检索最相关的记录。

        Args:
            query:             自然语言查询字符串。
            memories:          CanonicalMemory 列表（M4 的输出）。
            query_attribute:   可选的属性过滤（如 "blood_glucose"）。
            query_time_scope:  可选的时间范围过滤（如 "2023-02-22"）。

        Returns:
            按相关性排序的 RetrievalResult 列表（最多 top_k 条）。
        """
        if not memories:
            return []

        intent = self._parse_query_intent(
            query=query,
            memories=memories,
            query_attribute=query_attribute,
            query_time_scope=query_time_scope,
        )
        w_struct, w_semantic, w_recency = self._dynamic_weights(intent)
        semantic_scores = self._semantic_scores(query, memories)

        results: List[RetrievalResult] = []
        for mem, sem_score in zip(memories, semantic_scores):
            s_score = self._struct_score(mem, intent, query=query)
            r_score = self._recency_score(mem, intent)
            total_score = w_struct * s_score + w_semantic * sem_score + w_recency * r_score

            # 默认轻微惩罚冲突记录（除非 query 主动问冲突）
            if mem.conflict_flag and not intent.asks_conflict:
                total_score -= 0.03

            conflict_warning = mem.conflict_flag
            conflict_detail = ""
            if conflict_warning and mem.conflict_history:
                ch = mem.conflict_history[0]
                conflict_detail = f"{ch.old_value} -> {ch.new_value} ({ch.conflict_type})"

            results.append(
                RetrievalResult(
                    memory=mem,
                    score=round(float(total_score), 4),
                    struct_score=round(float(s_score), 4),
                    semantic_score=round(float(sem_score), 4),
                    recency_score=round(float(r_score), 4),
                    conflict_warning=conflict_warning,
                    conflict_detail=conflict_detail,
                )
            )

        # 按总分排序；同分时优先结构化更强、语义更强、更新更近
        def _sort_key(r: RetrievalResult) -> Tuple[float, float, float, float, int]:
            latest_turn = max(r.memory.provenance) if r.memory.provenance else 0
            return (r.score, r.struct_score, r.semantic_score, r.recency_score, latest_turn)

        results.sort(key=_sort_key, reverse=True)
        return results[: self.top_k]

    def format_context(
        self,
        results: List[RetrievalResult],
        include_conflicts: bool = True,
    ) -> str:
        """
        将检索结果格式化为 LLM 可用的上下文字符串。

        Args:
            results:           检索结果列表。
            include_conflicts: 是否在上下文中包含冲突警告。

        Returns:
            格式化的上下文字符串。
        """
        if not results:
            return "No relevant medical records found."

        lines = ["=== Patient Medical Memory ==="]
        for i, r in enumerate(results, 1):
            if include_conflicts or not r.conflict_warning:
                lines.append(f"{i}. {r.to_context_str()}")

        return "\n".join(lines)
