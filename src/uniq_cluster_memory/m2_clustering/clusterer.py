"""
m2_clustering/clusterer.py
===========================
M2 模块：事件聚类器。

核心职责：
    将 M1 输出的 ExtractedEvent 列表，按语义相似性聚类到同一"属性槽"（attribute slot）。

设计要点：
    1. 两阶段聚类：
       - Phase 1（规则）：基于 attribute 名称的规范化映射（别名表），
         将语义等价的属性名合并（如 "systolic_blood_pressure" -> "blood_pressure_sys"）。
       - Phase 2（Embedding）：对 Phase 1 无法处理的未知属性，
         使用 MiniLM Embedding 计算相似度，聚类到最近的已知属性槽。
    2. 占位设计：当前使用 all-MiniLM-L6-v2 作为 Embedding 模型，
       后续可替换为 BioLORD（临床表示）而无需修改接口。
    3. 输出：返回 AttributeCluster 列表，每个 cluster 包含一组语义等价的 ExtractedEvent。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.uniq_cluster_memory.m1_event_extraction import ExtractedEvent

# ─── 属性规范化别名表（与 uniqueness_eval.py 中的 alias_map 保持一致）────────

ATTRIBUTE_ALIAS_MAP: Dict[str, str] = {
    # 血压
    "systolic_blood_pressure": "blood_pressure_sys",
    "diastolic_blood_pressure": "blood_pressure_dia",
    "blood_pressure_systolic": "blood_pressure_sys",
    "blood_pressure_diastolic": "blood_pressure_dia",
    "bp_systolic": "blood_pressure_sys",
    "bp_diastolic": "blood_pressure_dia",
    "blood_pressure_dias": "blood_pressure_dia",
    "sbp": "blood_pressure_sys",
    "dbp": "blood_pressure_dia",
    # 血糖
    "blood_sugar": "blood_glucose",
    "glucose": "blood_glucose",
    "bg": "blood_glucose",
    "fasting_blood_glucose": "blood_glucose",
    "fasting_glucose": "blood_glucose",
    # 心率
    "pulse": "heart_rate",
    "hr": "heart_rate",
    "pulse_rate": "heart_rate",
    # 体温
    "temperature": "body_temperature",
    "temp": "body_temperature",
    "fever": "body_temperature",
    # 血红蛋白
    "hb": "hemoglobin",
    "hgb": "hemoglobin",
    "haemoglobin": "hemoglobin",
    # 诊断
    "diagnosis": "primary_diagnosis",
    "medical_diagnosis": "primary_diagnosis",
    "condition": "primary_diagnosis",
    "disease": "primary_diagnosis",
    # 症状
    "chief_complaint": "symptom",
    "complaint": "symptom",
    "presenting_symptom": "symptom",
    # 药物
    "drug": "medication",
    "medicine": "medication",
    "prescription": "medication",
    "treatment": "medication",
}

# 已知的标准属性集合
CANONICAL_ATTRIBUTES = {
    "blood_glucose", "blood_pressure_sys", "blood_pressure_dia",
    "heart_rate", "body_temperature", "hemoglobin",
    "primary_diagnosis", "medication", "symptom",
}


# ─── 数据结构 ────────────────────────────────────────────────────────────────

@dataclass
class AttributeCluster:
    """
    M2 模块的输出单元：一组语义等价的 ExtractedEvent 的聚合。

    所有 events 都被归一化到同一个 canonical_attribute 下。
    """
    cluster_id: str
    canonical_attribute: str            # 规范化后的属性名
    update_policy: str                  # 该属性的更新策略
    events: List[ExtractedEvent] = field(default_factory=list)

    @property
    def n_events(self) -> int:
        return len(self.events)


# ─── 主聚类器类 ──────────────────────────────────────────────────────────────

class EventClusterer:
    """
    M2 事件聚类器。

    Phase 1：规则别名映射（快速、确定性）。
    Phase 2：Embedding 相似度聚类（处理未知属性）。
    """

    # 属性对应的更新策略
    ATTRIBUTE_POLICIES: Dict[str, str] = {
        "blood_glucose": "unique",
        "blood_pressure_sys": "unique",
        "blood_pressure_dia": "unique",
        "heart_rate": "unique",
        "body_temperature": "unique",
        "hemoglobin": "unique",
        "primary_diagnosis": "unique",
        "medication": "latest",
        "symptom": "append",
    }

    EMBEDDING_SIMILARITY_THRESHOLD = 0.75  # 低于此阈值的未知属性保留原名

    def __init__(self, use_embedding: bool = True):
        self.use_embedding = use_embedding
        self._embedder = None
        self._canonical_embeddings: Optional[Dict[str, np.ndarray]] = None

    def _get_embedder(self):
        """懒加载 MiniLM Embedding 模型。"""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            # 预计算所有标准属性的 embedding
            canonical_list = list(CANONICAL_ATTRIBUTES)
            embeddings = self._embedder.encode(
                [attr.replace("_", " ") for attr in canonical_list],
                normalize_embeddings=True,
            )
            self._canonical_embeddings = {
                attr: emb for attr, emb in zip(canonical_list, embeddings)
            }
        return self._embedder

    def _normalize_attribute(self, attr: str) -> str:
        """
        Phase 1：通过别名表规范化属性名。
        """
        attr = attr.strip().lower().replace(" ", "_").replace("-", "_")
        # 直接匹配
        if attr in CANONICAL_ATTRIBUTES:
            return attr
        # 别名映射
        if attr in ATTRIBUTE_ALIAS_MAP:
            return ATTRIBUTE_ALIAS_MAP[attr]
        # 前缀/后缀模糊匹配
        for alias, canonical in ATTRIBUTE_ALIAS_MAP.items():
            if alias in attr or attr in alias:
                return canonical
        return attr  # 返回原始名，由 Phase 2 处理

    def _embedding_match(self, unknown_attr: str) -> str:
        """
        Phase 2：使用 Embedding 将未知属性匹配到最近的标准属性。
        """
        if not self.use_embedding:
            return unknown_attr
        try:
            embedder = self._get_embedder()
            query_emb = embedder.encode(
                [unknown_attr.replace("_", " ")],
                normalize_embeddings=True,
            )[0]
            best_attr = unknown_attr
            best_sim = -1.0
            for canonical, emb in self._canonical_embeddings.items():
                sim = float(np.dot(query_emb, emb))
                if sim > best_sim:
                    best_sim = sim
                    best_attr = canonical
            if best_sim >= self.EMBEDDING_SIMILARITY_THRESHOLD:
                return best_attr
        except Exception:
            pass
        return unknown_attr

    def cluster(
        self,
        events: List[ExtractedEvent],
        dialogue_id: str,
    ) -> List[AttributeCluster]:
        """
        将 ExtractedEvent 列表聚类为 AttributeCluster 列表。

        Args:
            events:       M1 输出的 ExtractedEvent 列表。
            dialogue_id:  对话 ID。

        Returns:
            AttributeCluster 列表，按 canonical_attribute 分组。
        """
        clusters: Dict[str, AttributeCluster] = {}
        cluster_counter = 0

        for evt in events:
            # Phase 1: 规则规范化
            canonical = self._normalize_attribute(evt.attribute)

            # Phase 2: Embedding 匹配（仅对未知属性）
            if canonical not in CANONICAL_ATTRIBUTES:
                canonical = self._embedding_match(canonical)

            # 更新事件的 attribute 为规范化名称
            evt.attribute = canonical

            # 获取或创建对应的 cluster
            if canonical not in clusters:
                cluster_counter += 1
                policy = self.ATTRIBUTE_POLICIES.get(canonical, "unique")
                clusters[canonical] = AttributeCluster(
                    cluster_id=f"cluster_{dialogue_id}_{cluster_counter:03d}",
                    canonical_attribute=canonical,
                    update_policy=policy,
                )

            clusters[canonical].events.append(evt)

        return list(clusters.values())
