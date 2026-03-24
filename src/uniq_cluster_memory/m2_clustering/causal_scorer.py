"""
m2_clustering/causal_scorer.py
==============================
因果去混淆的共指评分（Causal Deconfounded Coreference Scoring）。

使用结构因果模型（Structural Causal Model）对事件对的共指概率进行去混淆调整，
避免词汇重叠、时间邻近等混淆因子导致的虚假聚合。

核心算法：
    P(coref | features, do(confounders)) = raw_similarity - λ * Σ(confounder_baselines)

    其中：
    - raw_similarity：基于词汇和属性的原始相似度
    - confounders：词汇重叠(Z1)、时间邻近(Z2)、说话者身份(Z3)
    - confounder_baselines：各混淆因子的基线共指概率
    - λ：调整强度系数

参考：Causal Graph Intervention for ECR (Scientific Reports 2025)
      基于后门调整（backdoor adjustment）的事件共指消解。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.uniq_cluster_memory.m1_event_extraction import ExtractedEvent


# ── 混淆因子基线（领域知识） ──────────────────────────────────────────────────
# 这些值表示：当两个事件共享某个混淆因子时，它们被合并的"基线概率"。
# 高 baseline = 该混淆因子对合并决策的影响大 = 去混淆调整更强。

CONFOUNDER_SAME_MEDICATION_NAME = 0.6   # 同药名事件的基线共指概率
CONFOUNDER_TEMPORAL_PROXIMITY = 0.4     # 时间邻近事件的基线共指概率
CONFOUNDER_SAME_SPEAKER = 0.2          # 同说话者事件的基线共指概率

DEFAULT_LAMBDA = 0.3                    # 调整强度系数
DEFAULT_MERGE_THRESHOLD = 0.5           # 去混淆后的合并阈值


@dataclass
class ConfoundingAnalysis:
    """混淆因子分析结果。"""
    raw_similarity: float
    deconfounded_score: float
    active_confounders: List[str]
    confounder_adjustments: dict


class CausalCoreferenceScorer:
    """
    基于结构因果模型的事件共指评分器。

    结构因果模型（SCM）：
        Treatment: T = "两个事件是否共指？"
        混淆因子:
            Z1 = 词汇重叠（同药名、同症状名）
            Z2 = 时间邻近（同一 time anchor）
            Z3 = 说话者身份（同一 speaker）
        后门调整:
            P(coref | features, do(Z)) = raw - λ * Σ(active_confounder_baselines)
    """

    def __init__(
        self,
        lambda_adjust: float = DEFAULT_LAMBDA,
        merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
    ):
        self.lambda_adjust = lambda_adjust
        self.merge_threshold = merge_threshold

    def compute_raw_similarity(
        self,
        evt_a: ExtractedEvent,
        evt_b: ExtractedEvent,
    ) -> float:
        """计算两个事件的原始相似度（基于属性、值、时间）。"""
        score = 0.0

        # 属性匹配
        if evt_a.attribute.strip().lower() == evt_b.attribute.strip().lower():
            score += 0.4

        # 值相似度（词汇重叠）
        tokens_a = set(self._tokenize(evt_a.value))
        tokens_b = set(self._tokenize(evt_b.value))
        if tokens_a and tokens_b:
            overlap = len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)
            score += 0.4 * overlap

        # 时间锚点匹配
        anchor_a = self._normalize_time_anchor(evt_a.time_expr)
        anchor_b = self._normalize_time_anchor(evt_b.time_expr)
        if anchor_a and anchor_b and anchor_a == anchor_b:
            score += 0.2

        return min(score, 1.0)

    def identify_confounders(
        self,
        evt_a: ExtractedEvent,
        evt_b: ExtractedEvent,
    ) -> dict:
        """识别两个事件之间的活跃混淆因子。"""
        confounders = {}

        # Z1: 词汇重叠（同名药物/同名症状）
        name_a = self._extract_core_name(evt_a.value, evt_a.attribute)
        name_b = self._extract_core_name(evt_b.value, evt_b.attribute)
        if name_a and name_b and name_a == name_b:
            confounders["same_name"] = CONFOUNDER_SAME_MEDICATION_NAME

        # Z2: 时间邻近
        anchor_a = self._normalize_time_anchor(evt_a.time_expr)
        anchor_b = self._normalize_time_anchor(evt_b.time_expr)
        if anchor_a and anchor_b and anchor_a == anchor_b:
            confounders["temporal_proximity"] = CONFOUNDER_TEMPORAL_PROXIMITY

        # Z3: 同说话者
        if (evt_a.speaker and evt_b.speaker and
                evt_a.speaker.strip().lower() == evt_b.speaker.strip().lower()):
            confounders["same_speaker"] = CONFOUNDER_SAME_SPEAKER

        return confounders

    def deconfounded_score(
        self,
        evt_a: ExtractedEvent,
        evt_b: ExtractedEvent,
    ) -> float:
        """
        计算去混淆后的共指概率。

        P(coref | features, do(Z)) = raw - λ * Σ(confounder_baselines)
        结果裁剪到 [0, 1]。
        """
        raw = self.compute_raw_similarity(evt_a, evt_b)
        confounders = self.identify_confounders(evt_a, evt_b)

        adjustment = sum(confounders.values())
        deconfounded = raw - self.lambda_adjust * adjustment

        return max(0.0, min(1.0, deconfounded))

    def analyze(
        self,
        evt_a: ExtractedEvent,
        evt_b: ExtractedEvent,
    ) -> ConfoundingAnalysis:
        """完整分析：返回原始分数、去混淆分数和混淆因子详情。"""
        raw = self.compute_raw_similarity(evt_a, evt_b)
        confounders = self.identify_confounders(evt_a, evt_b)
        adjustment = sum(confounders.values())
        deconfounded = max(0.0, min(1.0, raw - self.lambda_adjust * adjustment))

        return ConfoundingAnalysis(
            raw_similarity=raw,
            deconfounded_score=deconfounded,
            active_confounders=list(confounders.keys()),
            confounder_adjustments=confounders,
        )

    def should_merge(
        self,
        evt_a: ExtractedEvent,
        evt_b: ExtractedEvent,
    ) -> bool:
        """判断两个事件是否应合并到同一 bundle。"""
        return self.deconfounded_score(evt_a, evt_b) >= self.merge_threshold

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """简单分词。"""
        return re.findall(r"\w+", (text or "").strip().lower())

    @staticmethod
    def _normalize_time_anchor(time_expr: str) -> str:
        """归一化时间锚点用于比较。"""
        raw = (time_expr or "").strip().lower()
        # 统一常见变体
        for variant in ("today", "this morning", "this afternoon", "this evening", "tonight", "今天", "今早"):
            if variant in raw:
                return "today"
        for variant in ("yesterday", "昨天"):
            if variant in raw:
                return "yesterday"
        return raw

    @staticmethod
    def _extract_core_name(value: str, attribute: str) -> str:
        """提取核心实体名（去掉剂量、频率等修饰词）。"""
        text = (value or "").strip().lower()
        attr = (attribute or "").strip().lower()

        if attr == "medication":
            # 去掉剂量和频率信息，只保留药名
            text = re.sub(r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|u|iu|units?)\b", "", text)
            text = re.sub(r"\b(?:once|twice|daily|bid|tid|qid|qd|qhs|at night|per day)\b", "", text)
            text = re.sub(r"\s+", " ", text).strip()
        elif attr == "symptom":
            # 去掉严重度修饰
            text = re.sub(r"\b(?:mild|moderate|severe|slight|significant|acute|chronic)\b", "", text)
            text = re.sub(r"\s+", " ", text).strip()

        return text
