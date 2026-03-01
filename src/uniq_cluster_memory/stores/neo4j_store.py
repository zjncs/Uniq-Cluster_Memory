"""
neo4j_store.py
==============
Neo4j 持久化图存储适配器。

设计目标：
1. 将 CanonicalMemory 写入图数据库（Patient-[:HAS_MEMORY]->Memory）。
2. 提供时间查询接口 query_meds_on(date)。
3. 不依赖项目其它模块的运行时状态，可独立复用。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from src.uniq_cluster_memory.schema import CanonicalMemory


class Neo4jMemoryStore:
    """
    Neo4j 记忆存储。
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ):
        try:
            from neo4j import GraphDatabase  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "neo4j driver is not installed. Please install with `pip install neo4j`."
            ) from e

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self._ensure_indexes()

    def close(self):
        if self._driver is not None:
            self._driver.close()

    def _ensure_indexes(self):
        queries = [
            "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE",
            "CREATE CONSTRAINT patient_id_unique IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE",
        ]
        with self._driver.session(database=self.database) as session:
            for q in queries:
                session.run(q)

    @staticmethod
    def _memory_id(payload: Dict[str, Any]) -> str:
        parts = [
            str(payload.get("patient_id", "")),
            str(payload.get("relation_type", "")),
            str(payload.get("target_value", "")),
            str(payload.get("start_time", "")),
            str(payload.get("end_time", "")),
            str(payload.get("time_scope", "")),
        ]
        raw = "||".join(parts)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    def _expand_records(self, memories: List[CanonicalMemory]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for mem in memories:
            base = mem.to_dict()
            timeline = []
            if isinstance(mem.qualifiers, dict):
                timeline = mem.qualifiers.get("med_timeline") or []

            # 用药时间轴展开为多条持久化记录，方便图查询
            if timeline and (base.get("relation_type") == "TAKES_DRUG" or base.get("attribute") == "medication"):
                for entry in timeline:
                    rec = dict(base)
                    rec["value"] = entry.get("value", rec.get("value"))
                    rec["target_value"] = rec["value"]
                    rec["start_time"] = entry.get("start_time")
                    rec["end_time"] = entry.get("end_time")
                    rec["time_precision"] = entry.get("time_precision", rec.get("time_precision"))
                    rec["is_ongoing"] = bool(entry.get("is_ongoing", False))
                    rec["provenance"] = entry.get("provenance", rec.get("provenance", []))
                    rec["memory_id"] = self._memory_id(rec)
                    records.append(rec)
                continue

            base["memory_id"] = self._memory_id(base)
            records.append(base)
        return records

    def upsert_memories(self, memories: List[CanonicalMemory]):
        if not memories:
            return
        records = self._expand_records(memories)
        query = """
        UNWIND $records AS r
        MERGE (p:Patient {id: r.patient_id})
        MERGE (m:Memory {memory_id: r.memory_id})
        SET
            m.patient_id = r.patient_id,
            m.attribute = r.attribute,
            m.value = r.value,
            m.unit = r.unit,
            m.time_scope = r.time_scope,
            m.confidence = toFloat(r.confidence),
            m.provenance = coalesce(r.provenance, []),
            m.conflict_flag = coalesce(r.conflict_flag, false),
            m.update_policy = r.update_policy,
            m.subject_id = r.subject_id,
            m.relation_type = r.relation_type,
            m.target_value = r.target_value,
            m.start_time = r.start_time,
            m.end_time = r.end_time,
            m.duration_days = r.duration_days,
            m.time_precision = r.time_precision,
            m.time_source = r.time_source,
            m.is_ongoing = coalesce(r.is_ongoing, false),
            m.dosage = r.dosage,
            m.frequency = r.frequency,
            m.time_of_day = r.time_of_day,
            m.route = r.route
        MERGE (p)-[:HAS_MEMORY]->(m)
        """
        with self._driver.session(database=self.database) as session:
            session.run(query, records=records)

    def query_meds_on(
        self,
        target_date: str,
        patient_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[CanonicalMemory]:
        query = """
        MATCH (p:Patient)-[:HAS_MEMORY]->(m:Memory)
        WHERE m.relation_type = 'TAKES_DRUG'
          AND ($patient_id IS NULL OR p.id = $patient_id)
          AND (
            m.start_time IS NULL OR date(m.start_time) <= date($target_date)
          )
          AND (
            m.end_time IS NULL OR date(m.end_time) >= date($target_date)
          )
        RETURN m
        ORDER BY m.start_time DESC
        LIMIT $limit
        """
        out: List[CanonicalMemory] = []
        with self._driver.session(database=self.database) as session:
            rows = session.run(
                query,
                patient_id=patient_id,
                target_date=target_date,
                limit=limit,
            )
            for row in rows:
                m = dict(row["m"])
                # 补默认字段，保证 from_dict 稳定
                payload = {
                    "patient_id": m.get("patient_id", patient_id or ""),
                    "attribute": m.get("attribute", "medication"),
                    "value": m.get("value", ""),
                    "unit": m.get("unit", ""),
                    "time_scope": m.get("time_scope", "global"),
                    "confidence": m.get("confidence", 1.0),
                    "provenance": m.get("provenance", []),
                    "conflict_flag": m.get("conflict_flag", False),
                    "conflict_history": [],
                    "update_policy": m.get("update_policy", "latest"),
                    "subject_id": m.get("subject_id", m.get("patient_id", "")),
                    "relation_type": m.get("relation_type", "TAKES_DRUG"),
                    "target_value": m.get("target_value", m.get("value", "")),
                    "start_time": m.get("start_time"),
                    "end_time": m.get("end_time"),
                    "duration_days": m.get("duration_days"),
                    "time_precision": m.get("time_precision", ""),
                    "time_source": m.get("time_source", "neo4j"),
                    "is_ongoing": m.get("is_ongoing", False),
                    "dosage": m.get("dosage", ""),
                    "frequency": m.get("frequency", ""),
                    "time_of_day": m.get("time_of_day", ""),
                    "route": m.get("route", ""),
                    "qualifiers": {},
                }
                out.append(CanonicalMemory.from_dict(payload))
        return out

