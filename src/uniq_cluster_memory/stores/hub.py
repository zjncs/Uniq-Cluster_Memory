"""
hub.py
======
统一持久化入口：
- Neo4j 图存储
- Qdrant 向量存储
"""

from __future__ import annotations

import os
from typing import List, Optional

from src.uniq_cluster_memory.schema import CanonicalMemory


class MemoryPersistenceHub:
    """
    可选持久化管理器。
    默认全部关闭，不影响现有离线实验。
    """

    def __init__(
        self,
        enable_qdrant: bool = False,
        enable_neo4j: bool = False,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_collection: str = "medical_memory",
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: str = "neo4j",
    ):
        self.enable_qdrant = enable_qdrant
        self.enable_neo4j = enable_neo4j

        self._qdrant = None
        self._neo4j = None

        if enable_qdrant:
            from .qdrant_store import QdrantMemoryStore

            self._qdrant = QdrantMemoryStore(
                url=qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333"),
                api_key=qdrant_api_key or os.getenv("QDRANT_API_KEY"),
                collection_name=qdrant_collection,
            )

        if enable_neo4j:
            from .neo4j_store import Neo4jMemoryStore

            uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
            pwd = neo4j_password or os.getenv("NEO4J_PASSWORD", "")
            if not pwd:
                raise RuntimeError(
                    "Neo4j password is required. Set NEO4J_PASSWORD or pass neo4j_password."
                )
            self._neo4j = Neo4jMemoryStore(
                uri=uri,
                user=user,
                password=pwd,
                database=neo4j_database,
            )

    @property
    def enabled(self) -> bool:
        return bool(self._qdrant or self._neo4j)

    def close(self):
        if self._qdrant:
            self._qdrant.close()
        if self._neo4j:
            self._neo4j.close()

    def upsert_memories(self, memories: List[CanonicalMemory]):
        if not memories:
            return
        if self._qdrant:
            self._qdrant.upsert_memories(memories)
        if self._neo4j:
            self._neo4j.upsert_memories(memories)

    def query_meds_on(self, target_date: str, patient_id: Optional[str] = None) -> List[CanonicalMemory]:
        if self._neo4j:
            return self._neo4j.query_meds_on(target_date=target_date, patient_id=patient_id)
        return []

    def vector_search(
        self,
        query: str,
        patient_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[CanonicalMemory]:
        if self._qdrant:
            return self._qdrant.search(
                query=query,
                patient_id=patient_id,
                relation_type=relation_type,
                limit=limit,
            )
        return []

