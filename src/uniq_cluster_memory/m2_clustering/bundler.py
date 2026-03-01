"""
m2_clustering/bundler.py
========================
M2.5 模块：信息团（Information Bundle）构建器。

目标：
1) 实体级聚合（Entity Bundle）：将同一药物的不同表达/剂量频次信息聚合。
2) 事件级聚合（Event Bundle）：将同一时间锚点（同天/同次复诊/同轮次）的多属性事实聚成事件团。

说明：
- 该模块不替代 M2 的 attribute 归一，而是在其后补充“信息团结构”。
- 输出 BundleGraph 供研究分析、可视化与后续版本化冲突管理使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Tuple

from src.uniq_cluster_memory.m1_event_extraction import ExtractedEvent
from src.uniq_cluster_memory.schema import extract_medication_qualifiers


@dataclass
class EntityBundle:
    bundle_id: str
    entity_type: str
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    dosage_values: List[str] = field(default_factory=list)
    frequency_values: List[str] = field(default_factory=list)
    time_of_day_values: List[str] = field(default_factory=list)
    provenance_turns: List[int] = field(default_factory=list)
    event_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bundle_id": self.bundle_id,
            "entity_type": self.entity_type,
            "canonical_name": self.canonical_name,
            "aliases": self.aliases,
            "dosage_values": self.dosage_values,
            "frequency_values": self.frequency_values,
            "time_of_day_values": self.time_of_day_values,
            "provenance_turns": self.provenance_turns,
            "event_ids": self.event_ids,
        }


@dataclass
class EventBundle:
    bundle_id: str
    time_anchor: str
    attributes: Dict[str, List[str]] = field(default_factory=dict)
    evidence_snippets: List[str] = field(default_factory=list)
    provenance_turns: List[int] = field(default_factory=list)
    event_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bundle_id": self.bundle_id,
            "time_anchor": self.time_anchor,
            "attributes": self.attributes,
            "evidence_snippets": self.evidence_snippets,
            "provenance_turns": self.provenance_turns,
            "event_ids": self.event_ids,
        }


@dataclass
class BundleLink:
    src_bundle_id: str
    dst_bundle_id: str
    relation: str

    def to_dict(self) -> dict:
        return {
            "src_bundle_id": self.src_bundle_id,
            "dst_bundle_id": self.dst_bundle_id,
            "relation": self.relation,
        }


@dataclass
class BundleGraph:
    dialogue_id: str
    entity_bundles: List[EntityBundle] = field(default_factory=list)
    event_bundles: List[EventBundle] = field(default_factory=list)
    links: List[BundleLink] = field(default_factory=list)

    @classmethod
    def empty(cls, dialogue_id: str) -> "BundleGraph":
        return cls(dialogue_id=dialogue_id, entity_bundles=[], event_bundles=[], links=[])

    def to_dict(self) -> dict:
        return {
            "dialogue_id": self.dialogue_id,
            "entity_bundles": [b.to_dict() for b in self.entity_bundles],
            "event_bundles": [b.to_dict() for b in self.event_bundles],
            "links": [l.to_dict() for l in self.links],
        }


class InformationBundleBuilder:
    """
    M2.5 信息团构建器。
    """

    TIME_ANCHOR_RULES: Dict[str, str] = {
        "today": "day:today",
        "this morning": "day:today",
        "this afternoon": "day:today",
        "this evening": "day:today",
        "tonight": "day:today",
        "yesterday": "day:yesterday",
        "last week": "week:last_week",
        "this week": "week:this_week",
        "last month": "month:last_month",
        "this month": "month:this_month",
        "global": "global",
    }

    _DOSE_RE = re.compile(r"\d+(?:\.\d+)?\s*(mg|mcg|ug|g|ml|u|iu|units?)\b", re.IGNORECASE)
    _FREQ_RE = re.compile(r"\b(qd|bid|tid|qid|qn|qhs|daily|nightly|once\s+daily|twice\s+daily)\b", re.IGNORECASE)
    _NON_WORD_RE = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fff\s-]")
    _MULTISPACE_RE = re.compile(r"\s+")
    _STOPWORDS = {
        "tablet", "capsule", "tab", "cap", "po", "oral", "take", "taking",
        "prescribed", "prescription",
    }

    def build(self, events: List[ExtractedEvent], dialogue_id: str) -> BundleGraph:
        if not events:
            return BundleGraph.empty(dialogue_id=dialogue_id)

        entity_bundles, event_to_entity = self._build_medication_entity_bundles(events, dialogue_id)
        event_bundles = self._build_event_bundles(events, dialogue_id)
        links = self._build_links(event_bundles, event_to_entity)

        return BundleGraph(
            dialogue_id=dialogue_id,
            entity_bundles=entity_bundles,
            event_bundles=event_bundles,
            links=links,
        )

    def _build_medication_entity_bundles(
        self,
        events: List[ExtractedEvent],
        dialogue_id: str,
    ) -> Tuple[List[EntityBundle], Dict[str, str]]:
        med_events = [e for e in events if e.attribute == "medication"]
        if not med_events:
            return [], {}

        grouped: Dict[str, List[ExtractedEvent]] = {}
        for evt in med_events:
            key = self._extract_medication_name(evt.value)
            grouped.setdefault(key, []).append(evt)

        bundles: List[EntityBundle] = []
        event_to_entity: Dict[str, str] = {}
        for i, (med_name, group) in enumerate(sorted(grouped.items()), start=1):
            aliases: List[str] = []
            dosage_values: List[str] = []
            frequency_values: List[str] = []
            time_of_day_values: List[str] = []
            provenance_turns: List[int] = []
            event_ids: List[str] = []

            seen_alias = set()
            for evt in group:
                if evt.value not in seen_alias:
                    aliases.append(evt.value)
                    seen_alias.add(evt.value)
                d, f, tod, _ = extract_medication_qualifiers(evt.value)
                if d and d not in dosage_values:
                    dosage_values.append(d)
                if f and f not in frequency_values:
                    frequency_values.append(f)
                if tod and tod not in time_of_day_values:
                    time_of_day_values.append(tod)
                provenance_turns.extend(evt.provenance)
                event_ids.append(evt.event_id)

            provenance_turns = sorted(set(provenance_turns))
            bundle_id = f"entity_{dialogue_id}_{i:03d}"
            bundle = EntityBundle(
                bundle_id=bundle_id,
                entity_type="medication",
                canonical_name=med_name,
                aliases=aliases,
                dosage_values=dosage_values,
                frequency_values=frequency_values,
                time_of_day_values=time_of_day_values,
                provenance_turns=provenance_turns,
                event_ids=event_ids,
            )
            bundles.append(bundle)
            for eid in event_ids:
                event_to_entity[eid] = bundle_id

        return bundles, event_to_entity

    def _build_event_bundles(self, events: List[ExtractedEvent], dialogue_id: str) -> List[EventBundle]:
        grouped: Dict[str, List[ExtractedEvent]] = {}
        for evt in events:
            anchor = self._normalize_time_anchor(evt)
            grouped.setdefault(anchor, []).append(evt)

        bundles: List[EventBundle] = []
        for i, (anchor, group) in enumerate(sorted(grouped.items()), start=1):
            attr_values: Dict[str, List[str]] = {}
            evidence_snippets: List[str] = []
            provenance_turns: List[int] = []
            event_ids: List[str] = []

            for evt in group:
                values = attr_values.setdefault(evt.attribute, [])
                if evt.value not in values:
                    values.append(evt.value)
                snippet = (evt.raw_text_snippet or "").strip()
                if snippet and snippet not in evidence_snippets:
                    evidence_snippets.append(snippet)
                provenance_turns.extend(evt.provenance)
                event_ids.append(evt.event_id)

            bundles.append(
                EventBundle(
                    bundle_id=f"event_{dialogue_id}_{i:03d}",
                    time_anchor=anchor,
                    attributes=attr_values,
                    evidence_snippets=evidence_snippets[:8],
                    provenance_turns=sorted(set(provenance_turns)),
                    event_ids=event_ids,
                )
            )

        return bundles

    @staticmethod
    def _build_links(event_bundles: List[EventBundle], event_to_entity: Dict[str, str]) -> List[BundleLink]:
        links: List[BundleLink] = []
        seen = set()
        for e_bundle in event_bundles:
            for eid in e_bundle.event_ids:
                dst = event_to_entity.get(eid)
                if not dst:
                    continue
                key = (e_bundle.bundle_id, dst, "MENTIONS_MEDICATION")
                if key in seen:
                    continue
                seen.add(key)
                links.append(
                    BundleLink(
                        src_bundle_id=e_bundle.bundle_id,
                        dst_bundle_id=dst,
                        relation="MENTIONS_MEDICATION",
                    )
                )
        return links

    def _normalize_time_anchor(self, evt: ExtractedEvent) -> str:
        raw = (evt.time_expr or "").strip()
        if not raw:
            raw = "global"
        lowered = raw.lower()
        if lowered == "global":
            turn = max(evt.provenance) if evt.provenance else 0
            return f"turn:{turn}"
        for pattern, mapped in self.TIME_ANCHOR_RULES.items():
            if pattern in lowered:
                return mapped
        return lowered

    def _extract_medication_name(self, value: str) -> str:
        text = (value or "").strip().lower()
        if not text:
            return "unknown_medication"
        text = self._DOSE_RE.sub(" ", text)
        text = self._FREQ_RE.sub(" ", text)
        text = self._NON_WORD_RE.sub(" ", text)
        tokens = [t for t in self._MULTISPACE_RE.split(text) if t and t not in self._STOPWORDS]
        if not tokens:
            return "unknown_medication"
        # 取前 3 个 token，避免带入冗长说明
        return " ".join(tokens[:3])

