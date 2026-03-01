"""
pipeline.py
===========
Uniq-Cluster Memory 完整 pipeline 主入口。

调用顺序：
    对话 -> M1（事件抽取）-> M2（聚类规范化）-> M3（Time Grounding + 唯一性管理）
         -> M4（压缩）-> M5（检索）-> 最终 CanonicalMemory 列表

对外接口：
    UniqueClusterMemoryPipeline.build_memory(dialogue, dialogue_id) -> List[CanonicalMemory]
    UniqueClusterMemoryPipeline.query(query, memories, ...) -> List[RetrievalResult]
"""

from __future__ import annotations

from typing import Any, List, Optional

from src.uniq_cluster_memory.m1_event_extraction import MedicalEventExtractor, ExtractedEvent
from src.uniq_cluster_memory.m2_clustering import EventClusterer
from src.uniq_cluster_memory.m3_uniqueness import UniquenessManager
from src.uniq_cluster_memory.m4_compression import MemoryCompressor
from src.uniq_cluster_memory.m5_retrieval import HybridMemoryRetriever, RetrievalResult
from src.uniq_cluster_memory.temporal_reasoning import TemporalReasoner
from src.uniq_cluster_memory.stores import MemoryPersistenceHub
from src.uniq_cluster_memory.schema import CanonicalMemory


class UniqueClusterMemoryPipeline:
    """
    Uniq-Cluster Memory 完整 pipeline。

    Args:
        w_struct:      M5 结构化检索权重（默认 0.7）。
        top_k:         M5 检索返回的最大记录数（默认 5）。
        use_embedding: 是否使用 Embedding 语义检索（默认 True）。
    """

    def __init__(
        self,
        w_struct: float = 0.7,
        top_k: int = 5,
        use_embedding: bool = True,
        missing_time_scope: str = "global",
        max_symptoms_per_scope: Optional[int] = None,
        enable_qdrant: bool = False,
        enable_neo4j: bool = False,
        persist_to_stores: bool = False,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_collection: str = "medical_memory",
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: str = "neo4j",
    ):
        self.m1 = MedicalEventExtractor()
        self.m2 = EventClusterer(use_embedding=use_embedding)
        self.m4 = MemoryCompressor()
        self.m5 = HybridMemoryRetriever(
            w_struct=w_struct,
            top_k=top_k,
            use_embedding=use_embedding,
        )
        self.temporal_reasoner = TemporalReasoner()
        self.missing_time_scope = (missing_time_scope or "global").strip().lower()
        self.max_symptoms_per_scope = max_symptoms_per_scope
        self.persist_to_stores = persist_to_stores
        self.persistence = MemoryPersistenceHub(
            enable_qdrant=enable_qdrant,
            enable_neo4j=enable_neo4j,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            qdrant_collection=qdrant_collection,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
        )
        self.w_struct = w_struct

    def build_memory(
        self,
        dialogue: List[Any],
        dialogue_id: str,
        dialogue_date: Optional[str] = None,
    ) -> List[CanonicalMemory]:
        """
        从对话构建 CanonicalMemory 列表（完整 M1-M4 pipeline）。

        Args:
            dialogue:      对话轮次列表。支持两种格式：
                           1) {"turn_id", "speaker", "text"}（Med-LongMem）
                           2) {"role", "content"} 或 DialogTurn 对象（benchmark）
            dialogue_id:   对话 ID（也用作 patient_id）。
            dialogue_date: 对话发生日期（ISO 格式），用于 Time Grounding。
                           如果为 None，尝试从对话内容推断。

        Returns:
            CanonicalMemory 列表。
        """
        # 统一对话字段契约（speaker/text/turn_id）
        normalized_dialogue = self._normalize_dialogue(dialogue)

        # 推断对话日期（如果未提供）
        if dialogue_date is None:
            dialogue_date = self._infer_dialogue_date(normalized_dialogue)

        # M1: 事件抽取
        events: List[ExtractedEvent] = self.m1.extract(normalized_dialogue, dialogue_id)

        if not events:
            return []

        # M2: 聚类规范化
        clusters = self.m2.cluster(events, dialogue_id)

        # M3: Time Grounding + 唯一性管理
        m3 = UniquenessManager(
            dialogue_date=dialogue_date,
            missing_time_scope=self.missing_time_scope,
            max_symptoms_per_scope=self.max_symptoms_per_scope,
        )
        memories: List[CanonicalMemory] = m3.process(clusters, patient_id=dialogue_id)

        # M4: 压缩
        memories = self.m4.compress(memories, patient_id=dialogue_id)

        # 可选持久化：写入向量库与图数据库
        if self.persist_to_stores and self.persistence.enabled:
            self.persistence.upsert_memories(memories)

        return memories

    def query(
        self,
        query: str,
        memories: List[CanonicalMemory],
        query_attribute: Optional[str] = None,
        query_time_scope: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        M5: 从 CanonicalMemory 列表中检索最相关的记录。

        Args:
            query:             自然语言查询字符串。
            memories:          CanonicalMemory 列表（build_memory 的输出）。
            query_attribute:   可选的属性过滤。
            query_time_scope:  可选的时间范围过滤。

        Returns:
            按相关性排序的 RetrievalResult 列表。
        """
        return self.m5.retrieve(
            query=query,
            memories=memories,
            query_attribute=query_attribute,
            query_time_scope=query_time_scope,
        )

    def query_meds_on(
        self,
        target_date: str,
        memories: List[CanonicalMemory],
        patient_id: Optional[str] = None,
        backend: str = "auto",
    ) -> List[CanonicalMemory]:
        """
        时间推理接口：查询某天生效的药物记录。

        Args:
            target_date: YYYY-MM-DD
            memories: build_memory 输出
            patient_id: 可选患者 ID 过滤
        """
        # 优先使用图数据库时间查询（若启用），否则回退本地规则推理
        b = (backend or "auto").strip().lower()
        if b in {"auto", "neo4j", "graph"} and self.persistence.enabled:
            graph_results = self.persistence.query_meds_on(
                target_date=target_date,
                patient_id=patient_id,
            )
            if graph_results:
                return graph_results
            if b in {"neo4j", "graph"}:
                return []

        return self.temporal_reasoner.query_meds_on(
            target_date=target_date,
            memories=memories,
            patient_id=patient_id,
        )

    def query_symptoms_between(
        self,
        start_date: str,
        end_date: str,
        memories: List[CanonicalMemory],
        patient_id: Optional[str] = None,
    ) -> List[CanonicalMemory]:
        """
        时间推理接口：查询时间区间内出现/持续的症状。
        """
        return self.temporal_reasoner.query_symptoms_between(
            start_date=start_date,
            end_date=end_date,
            memories=memories,
            patient_id=patient_id,
        )

    def query_relations_between(
        self,
        start_date: str,
        end_date: str,
        memories: List[CanonicalMemory],
        patient_id: Optional[str] = None,
        relation_type: Optional[str] = None,
    ) -> List[CanonicalMemory]:
        """
        通用时间推理接口：按关系类型查询区间重叠记录。
        """
        return self.temporal_reasoner.query_relations_between(
            start_date=start_date,
            end_date=end_date,
            memories=memories,
            patient_id=patient_id,
            relation_type=relation_type,
        )

    def vector_search_memories(
        self,
        query: str,
        patient_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[CanonicalMemory]:
        """
        向量库检索接口（Qdrant）。
        若未启用 Qdrant，则返回空列表。
        """
        return self.persistence.vector_search(
            query=query,
            patient_id=patient_id,
            relation_type=relation_type,
            limit=limit,
        )

    def close(self):
        """关闭外部资源连接（Neo4j/Qdrant）。"""
        self.persistence.close()

    def _normalize_dialogue(self, dialogue: List[Any]) -> List[dict]:
        """
        将不同来源的对话轮次统一为 M1 所需格式：
        {"turn_id": int, "speaker": "patient|doctor", "text": str}
        """
        normalized: List[dict] = []
        for i, turn in enumerate(dialogue):
            if isinstance(turn, dict):
                # Med-LongMem 风格
                if "speaker" in turn and "text" in turn:
                    text = str(turn.get("text", ""))
                    speaker = self._normalize_speaker(turn.get("speaker"))
                    turn_id = int(turn.get("turn_id", i))
                # benchmark UnifiedSample/DialogTurn 风格
                elif "role" in turn and "content" in turn:
                    text = str(turn.get("content", ""))
                    speaker = self._normalize_speaker(turn.get("role"))
                    turn_id = i
                else:
                    text = str(turn.get("text") or turn.get("content") or "")
                    speaker = self._normalize_speaker(turn.get("speaker") or turn.get("role"))
                    turn_id = int(turn.get("turn_id", i))
            else:
                # dataclass DialogTurn 等对象
                text = str(getattr(turn, "text", None) or getattr(turn, "content", ""))
                speaker = self._normalize_speaker(
                    getattr(turn, "speaker", None) or getattr(turn, "role", None)
                )
                turn_id = i

            normalized.append({
                "turn_id": turn_id,
                "speaker": speaker,
                "text": text,
            })
        return normalized

    @staticmethod
    def _normalize_speaker(raw_speaker: Any) -> str:
        """
        统一说话人标签到 patient/doctor。
        """
        s = str(raw_speaker or "").strip().lower()
        if s in {"patient", "user", "human"}:
            return "patient"
        if s in {"doctor", "assistant", "agent", "system"}:
            return "doctor"
        return "patient"

    def _infer_dialogue_date(self, dialogue: List[dict]) -> Optional[str]:
        """
        从对话内容中推断对话日期。
        简单实现：查找第一个 ISO 日期格式的字符串。
        """
        import re
        iso_pattern = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
        for turn in dialogue:
            match = iso_pattern.search(turn.get("text", ""))
            if match:
                return match.group(1)
        return None
