"""
m3_uniqueness/conflict_detector.py
===================================
冲突检测器 + 置信度加权多候选评分。

二元冲突检测（detect）：检测同一 (patient_id, attribute, time_scope) 下的值冲突。
多候选评分（score_candidates）：对所有候选值计算 soft confidence，
支持双时态冲突图（Bi-Temporal Conflict Graph）中的非二元冲突解决。

参考：
    - EvoKG (arXiv 2025.09): confidence-based contradiction resolution
    - Zep/Graphiti (arXiv 2025.01): bi-temporal conflict management
"""

from __future__ import annotations

import math
import re
from typing import List, Optional

from src.uniq_cluster_memory.m1_event_extraction import ExtractedEvent
from src.uniq_cluster_memory.schema import CandidateValue, CanonicalMemory, ConflictRecord


# ── 多候选评分权重 ────────────────────────────────────────────────────────────
W_RECENCY = 0.35       # 时间新近度权重
W_AUTHORITY = 0.25      # 来源权威度权重（doctor > patient > unknown）
W_EVIDENCE = 0.20       # 证据数量权重
W_PRIOR = 0.20          # 先验置信度权重

# 冲突判定阈值
CONFIDENCE_CONFLICT_THRESHOLD = 0.7     # top 候选低于此阈值 → conflict_flag
CONFIDENCE_SPREAD_THRESHOLD = 0.15      # top-2 差距小于此阈值 → conflict_flag


class ConflictDetector:
    """
    二元冲突检测 + 置信度加权多候选评分。

    detect() 保持原有行为：返回 ConflictRecord 或 None。
    score_candidates() 是新增的多候选评分方法，返回按 confidence 排序的 CandidateValue 列表。
    """

    NUMERIC_TOLERANCE = 0.001
    MED_EQUIV_PATTERNS = [
        (re.compile(r"\bonce\s+daily\b", re.IGNORECASE), "qd"),
        (re.compile(r"\bdaily\b", re.IGNORECASE), "qd"),
        (re.compile(r"\btwice\s+daily\b", re.IGNORECASE), "bid"),
        (re.compile(r"\bbid\b", re.IGNORECASE), "bid"),
        (re.compile(r"\bonce\s+at\s+night\b", re.IGNORECASE), "qhs"),
        (re.compile(r"\bat\s+night\b", re.IGNORECASE), "qhs"),
    ]

    # ── 二元冲突检测（原有逻辑不变） ──────────────────────────────────────────

    def detect(
        self,
        existing: CanonicalMemory,
        new_event: ExtractedEvent,
    ) -> Optional[ConflictRecord]:
        """检测新事件与现有记忆之间是否存在冲突。"""
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

    # ── 多候选置信度评分 ──────────────────────────────────────────────────────

    def score_candidates(
        self,
        candidates: List[CandidateValue],
        attribute: str,
    ) -> List[CandidateValue]:
        """
        对所有候选值计算综合置信度得分。

        打分公式：
            raw_score = W_RECENCY * temporal_recency
                      + W_AUTHORITY * source_authority
                      + W_EVIDENCE * log(1 + evidence_count)
                      + W_PRIOR * prior_confidence

        等价候选合并（证据计数累加），然后归一化 confidence 使之和为 1.0。
        返回按 confidence 降序排列的候选列表。
        """
        if not candidates:
            return []

        # Step 1: 合并等价候选
        merged = self._merge_equivalent_candidates(candidates, attribute)

        # Step 2: 计算综合分数
        for cand in merged:
            cand.confidence = (
                W_RECENCY * cand.temporal_recency
                + W_AUTHORITY * cand.source_authority
                + W_EVIDENCE * math.log(1 + cand.evidence_count)
                + W_PRIOR * cand.confidence
            )

        # Step 3: 归一化
        total = sum(c.confidence for c in merged)
        if total > 0:
            for cand in merged:
                cand.confidence = round(cand.confidence / total, 4)

        # Step 4: 排序
        merged.sort(key=lambda c: c.confidence, reverse=True)
        return merged

    def _merge_equivalent_candidates(
        self,
        candidates: List[CandidateValue],
        attribute: str,
    ) -> List[CandidateValue]:
        """合并语义等价的候选值（如 "daily" ≡ "qd"），证据计数累加。"""
        merged: List[CandidateValue] = []
        used = [False] * len(candidates)

        for i, cand_a in enumerate(candidates):
            if used[i]:
                continue
            # 找所有与 cand_a 等价的候选
            group_evidence = cand_a.evidence_count
            group_provenance = list(cand_a.provenance)
            for j in range(i + 1, len(candidates)):
                if used[j]:
                    continue
                if self._equivalent_value(cand_a.value, candidates[j].value, attribute):
                    group_evidence += candidates[j].evidence_count
                    group_provenance.extend(candidates[j].provenance)
                    # 保留 authority 和 recency 更高的
                    if candidates[j].source_authority > cand_a.source_authority:
                        cand_a.source_authority = candidates[j].source_authority
                    if candidates[j].temporal_recency > cand_a.temporal_recency:
                        cand_a.temporal_recency = candidates[j].temporal_recency
                    used[j] = True
            cand_a.evidence_count = group_evidence
            cand_a.provenance = sorted(set(group_provenance))
            merged.append(cand_a)
            used[i] = True

        return merged

    @staticmethod
    def should_flag_conflict(scored_candidates: List[CandidateValue]) -> bool:
        """根据多候选评分结果判断是否应标记为冲突。"""
        if len(scored_candidates) <= 1:
            return False
        top = scored_candidates[0].confidence
        if top < CONFIDENCE_CONFLICT_THRESHOLD:
            return True
        if len(scored_candidates) >= 2:
            spread = top - scored_candidates[1].confidence
            if spread < CONFIDENCE_SPREAD_THRESHOLD:
                return True
        return False

    # ── 等价判断（复用原有逻辑） ──────────────────────────────────────────────

    def _equivalent_value(self, old_value: str, new_value: str, attribute: str) -> bool:
        old_raw = (old_value or "").strip()
        new_raw = (new_value or "").strip()
        if not old_raw or not new_raw:
            return False

        old_norm = self._normalize_text(old_raw)
        new_norm = self._normalize_text(new_raw)
        if old_norm == new_norm:
            return True

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
