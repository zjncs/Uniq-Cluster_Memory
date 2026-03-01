"""
qdrant_store.py
===============
Qdrant 持久化向量存储适配器。
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

from src.uniq_cluster_memory.schema import CanonicalMemory
from src.uniq_cluster_memory.utils.embeddings import LocalEmbeddings


class QdrantMemoryStore:
    """
    Qdrant 记忆向量库。
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str],
        collection_name: str = "medical_memory",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client import models as qmodels  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "qdrant-client is not installed. Please install with `pip install qdrant-client`."
            ) from e

        self._qmodels = qmodels
        self._client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.embeddings = LocalEmbeddings(model_name=embedding_model)
        self._ensure_collection()

    def close(self):
        # qdrant client uses HTTP session; no explicit close required
        pass

    def _ensure_collection(self):
        try:
            self._client.get_collection(self.collection_name)
            return
        except Exception:
            pass

        dim = len(self.embeddings.embed_query("dimension probe"))
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._qmodels.VectorParams(size=dim, distance=self._qmodels.Distance.COSINE),
        )

    @staticmethod
    def _point_id(mem: CanonicalMemory) -> str:
        raw = "||".join(
            [
                mem.patient_id,
                mem.relation_type or "",
                mem.target_value or mem.value,
                mem.start_time or "",
                mem.end_time or "",
                mem.time_scope or "",
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _text(mem: CanonicalMemory) -> str:
        return " | ".join(
            [
                f"relation={mem.relation_type or mem.attribute}",
                f"value={mem.target_value or mem.value}",
                f"time={mem.start_time or ''}..{mem.end_time or ''}",
                f"scope={mem.time_scope}",
                f"dosage={mem.dosage}",
                f"frequency={mem.frequency}",
                f"time_of_day={mem.time_of_day}",
            ]
        )

    def upsert_memories(self, memories: List[CanonicalMemory]):
        if not memories:
            return

        texts = [self._text(m) for m in memories]
        vectors = self.embeddings.embed_documents(texts)

        points = []
        for m, vec in zip(memories, vectors):
            payload = m.to_dict()
            payload["embedding_text"] = self._text(m)
            points.append(
                self._qmodels.PointStruct(
                    id=self._point_id(m),
                    vector=vec,
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query: str,
        patient_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[CanonicalMemory]:
        vector = self.embeddings.embed_query(query)

        conditions = []
        if patient_id:
            conditions.append(
                self._qmodels.FieldCondition(
                    key="patient_id",
                    match=self._qmodels.MatchValue(value=patient_id),
                )
            )
        if relation_type:
            conditions.append(
                self._qmodels.FieldCondition(
                    key="relation_type",
                    match=self._qmodels.MatchValue(value=relation_type),
                )
            )

        query_filter = self._qmodels.Filter(must=conditions) if conditions else None
        hits = self._client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=limit,
            query_filter=query_filter,
        )

        out: List[CanonicalMemory] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            # 清掉无关字段
            payload.pop("embedding_text", None)
            if "patient_id" not in payload or "attribute" not in payload or "value" not in payload:
                continue
            out.append(CanonicalMemory.from_dict(payload))
        return out

