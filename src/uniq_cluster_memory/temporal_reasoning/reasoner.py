"""
temporal_reasoning/reasoner.py
==============================
时间推理模块（最小可运行版）。

当前提供：
- query_meds_on(date): 查询某一天患者正在使用的药物记录。
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from src.uniq_cluster_memory.schema import CanonicalMemory


class TemporalReasoner:
    """
    基于结构化时间字段（start_time/end_time/is_ongoing）的规则推理器。
    """

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        return datetime.strptime(date_str, "%Y-%m-%d")

    def _memory_interval(self, mem: CanonicalMemory) -> Optional[Tuple[datetime, datetime]]:
        """
        返回记忆条目的时间区间（闭区间）。
        若缺失时间信息且不是 global/ongoing，返回 None。
        """
        if mem.start_time and mem.end_time:
            return self._parse_date(mem.start_time), self._parse_date(mem.end_time)
        if mem.start_time and mem.is_ongoing:
            # open-ended 进行保守封顶，便于区间计算
            return self._parse_date(mem.start_time), self._parse_date("2100-12-31")
        if mem.start_time:
            t = self._parse_date(mem.start_time)
            return t, t
        if mem.time_scope == "global":
            return self._parse_date("1900-01-01"), self._parse_date("2100-12-31")
        return None

    @staticmethod
    def _interval_overlap(
        a: Tuple[datetime, datetime],
        b: Tuple[datetime, datetime],
    ) -> bool:
        return max(a[0], b[0]) <= min(a[1], b[1])

    def _covers_date(self, mem: CanonicalMemory, target_date: str) -> bool:
        t = self._parse_date(target_date)

        # 明确区间
        if mem.start_time and mem.end_time:
            start = self._parse_date(mem.start_time)
            end = self._parse_date(mem.end_time)
            return start <= t <= end

        # ongoing 记录默认覆盖未来时间
        if mem.is_ongoing and mem.start_time:
            start = self._parse_date(mem.start_time)
            return t >= start
        if mem.is_ongoing and not mem.start_time:
            return True

        # 退化到 global 语义
        if mem.time_scope == "global":
            return True
        return False

    @staticmethod
    def _is_medication(mem: CanonicalMemory) -> bool:
        return mem.attribute == "medication" or mem.relation_type == "TAKES_DRUG"

    @staticmethod
    def _is_symptom(mem: CanonicalMemory) -> bool:
        return mem.attribute == "symptom" or mem.relation_type == "HAS_SYMPTOM"

    def _expand_med_timeline(
        self,
        mem: CanonicalMemory,
    ) -> List[CanonicalMemory]:
        """
        将 medication 主记录按 med_timeline 展开为可推理子记录。
        """
        timeline = []
        if isinstance(mem.qualifiers, dict):
            timeline = mem.qualifiers.get("med_timeline") or []
        if not timeline:
            return [mem]

        expanded: List[CanonicalMemory] = []
        for entry in timeline:
            payload = mem.to_dict()
            payload["value"] = entry.get("value", mem.value)
            payload["target_value"] = payload["value"]
            payload["start_time"] = entry.get("start_time")
            payload["end_time"] = entry.get("end_time")
            payload["time_precision"] = entry.get("time_precision", payload.get("time_precision", ""))
            payload["is_ongoing"] = bool(entry.get("is_ongoing", False))
            if entry.get("provenance"):
                payload["provenance"] = entry["provenance"]
            expanded.append(CanonicalMemory.from_dict(payload))
        return expanded

    def query_meds_on(
        self,
        target_date: str,
        memories: List[CanonicalMemory],
        patient_id: Optional[str] = None,
    ) -> List[CanonicalMemory]:
        """
        查询 target_date 当天生效的用药记录。
        """
        results: List[CanonicalMemory] = []
        for mem in memories:
            if patient_id and mem.patient_id != patient_id:
                continue
            if not self._is_medication(mem):
                continue

            expanded = self._expand_med_timeline(mem)
            if len(expanded) > 1:
                for em in expanded:
                    if self._covers_date(em, target_date):
                        results.append(em)
                continue

            # 回退：没有时间轴时按主记录判断
            if self._covers_date(mem, target_date):
                results.append(mem)

        # 排序：最近开始时间优先；无 start_time 的 global 放后
        def _sort_key(m: CanonicalMemory):
            if m.start_time:
                return (1, m.start_time, max(m.provenance) if m.provenance else 0)
            return (0, "", max(m.provenance) if m.provenance else 0)

        # 去重：同 patient/value/dosage/frequency 视为同药物条目
        dedup = {}
        for r in results:
            key = (
                r.patient_id,
                r.value.strip().lower(),
                (r.dosage or "").strip().lower(),
                (r.frequency or "").strip().lower(),
            )
            prev = dedup.get(key)
            if prev is None:
                dedup[key] = r
                continue
            if (r.start_time or "") > (prev.start_time or ""):
                dedup[key] = r

        results = list(dedup.values())
        results.sort(key=_sort_key, reverse=True)
        return results

    def query_symptoms_between(
        self,
        start_date: str,
        end_date: str,
        memories: List[CanonicalMemory],
        patient_id: Optional[str] = None,
    ) -> List[CanonicalMemory]:
        """
        查询某个时间区间内出现/持续的症状记录。
        """
        query_interval = (self._parse_date(start_date), self._parse_date(end_date))
        if query_interval[0] > query_interval[1]:
            query_interval = (query_interval[1], query_interval[0])

        results: List[CanonicalMemory] = []
        for mem in memories:
            if patient_id and mem.patient_id != patient_id:
                continue
            if not self._is_symptom(mem):
                continue
            interval = self._memory_interval(mem)
            if interval is None:
                continue
            if self._interval_overlap(interval, query_interval):
                results.append(mem)

        results.sort(
            key=lambda m: (
                m.start_time or "",
                max(m.provenance) if m.provenance else 0,
            ),
            reverse=True,
        )
        return results

    def query_relations_between(
        self,
        start_date: str,
        end_date: str,
        memories: List[CanonicalMemory],
        patient_id: Optional[str] = None,
        relation_type: Optional[str] = None,
    ) -> List[CanonicalMemory]:
        """
        通用关系区间查询：返回与 [start_date, end_date] 有重叠的关系记录。
        """
        query_interval = (self._parse_date(start_date), self._parse_date(end_date))
        if query_interval[0] > query_interval[1]:
            query_interval = (query_interval[1], query_interval[0])

        normalized_rel = (relation_type or "").strip().upper()
        results: List[CanonicalMemory] = []
        for mem in memories:
            if patient_id and mem.patient_id != patient_id:
                continue
            if normalized_rel and (mem.relation_type or "").strip().upper() != normalized_rel:
                continue

            candidates = self._expand_med_timeline(mem) if self._is_medication(mem) else [mem]
            for c in candidates:
                interval = self._memory_interval(c)
                if interval is None:
                    continue
                if self._interval_overlap(interval, query_interval):
                    results.append(c)

        # 去重（同关系同值同区间）
        dedup = {}
        for m in results:
            key = (
                m.patient_id,
                m.relation_type,
                m.target_value or m.value,
                m.start_time,
                m.end_time,
            )
            if key not in dedup:
                dedup[key] = m

        out = list(dedup.values())
        out.sort(
            key=lambda m: (
                m.start_time or "",
                max(m.provenance) if m.provenance else 0,
            ),
            reverse=True,
        )
        return out
