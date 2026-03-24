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
from typing import Dict, List, Tuple, Optional

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
    status: str = "Active"  # Active | Superseded | Conflicting | Resolved
    evidence_chain: List[dict] = field(default_factory=list)

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
            "status": self.status,
            "evidence_chain": self.evidence_chain,
        }


@dataclass
class EventBundle:
    bundle_id: str
    time_anchor: str
    attributes: Dict[str, List[str]] = field(default_factory=dict)
    evidence_snippets: List[str] = field(default_factory=list)
    provenance_turns: List[int] = field(default_factory=list)
    event_ids: List[str] = field(default_factory=list)
    status: str = "Active"  # Active | Superseded | Conflicting | Resolved
    evidence_chain: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bundle_id": self.bundle_id,
            "time_anchor": self.time_anchor,
            "attributes": self.attributes,
            "evidence_snippets": self.evidence_snippets,
            "provenance_turns": self.provenance_turns,
            "event_ids": self.event_ids,
            "status": self.status,
            "evidence_chain": self.evidence_chain,
        }


@dataclass
class BundleLink:
    src_bundle_id: str
    dst_bundle_id: str
    relation: str
    confidence: float = 1.0
    reason: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "src_bundle_id": self.src_bundle_id,
            "dst_bundle_id": self.dst_bundle_id,
            "relation": self.relation,
            "confidence": self.confidence,
            "reason": self.reason,
            "metadata": self.metadata,
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

    核心概念：
        信息团（Information Bundle）是医疗对话中相关事实的结构化聚合单元，
        分为实体团（EntityBundle，如同一药物的不同提及）和事件团（EventBundle，
        如同一时间点的多个属性事实）。信息团是 M3 唯一性管理的基本粒度。

    因果去混淆：
        当 use_causal_scoring=True 时，使用结构因果模型对事件对的共指概率
        进行后门调整，避免词汇重叠/时间邻近等混淆因子导致的虚假聚合。
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

    def __init__(self, use_causal_scoring: bool = True):
        self.use_causal_scoring = use_causal_scoring
        self._causal_scorer = None

    def _get_causal_scorer(self):
        if self._causal_scorer is None:
            from src.uniq_cluster_memory.m2_clustering.causal_scorer import CausalCoreferenceScorer
            self._causal_scorer = CausalCoreferenceScorer()
        return self._causal_scorer

    @staticmethod
    def _evidence_entry(
        action: str,
        reason: str,
        event_id: Optional[str] = None,
        turns: Optional[List[int]] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        payload = {
            "action": action,
            "reason": reason,
        }
        if event_id:
            payload["event_id"] = event_id
        if turns:
            payload["turns"] = sorted(set(turns))
        if extra:
            payload["extra"] = extra
        return payload

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
            evidence_chain: List[dict] = []

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
                evidence_chain.append(
                    self._evidence_entry(
                        action="merge",
                        reason="Entity_Coreference",
                        event_id=evt.event_id,
                        turns=evt.provenance,
                        extra={"attribute": evt.attribute, "value": evt.value},
                    )
                )

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
                evidence_chain=evidence_chain,
            )
            bundle.evidence_chain.insert(
                0,
                self._evidence_entry(
                    action="create",
                    reason="Entity_Bundle_Initialization",
                    turns=provenance_turns,
                    extra={"canonical_name": med_name},
                ),
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

        # 因果去混淆：对同一 anchor 内的事件进一步拆分
        if self.use_causal_scoring:
            refined: Dict[str, List[ExtractedEvent]] = {}
            for anchor, group in grouped.items():
                sub_groups = self._causal_split(group, anchor)
                for j, sg in enumerate(sub_groups):
                    key = anchor if j == 0 else f"{anchor}::split_{j}"
                    refined[key] = sg
            grouped = refined

        bundles: List[EventBundle] = []
        for i, (anchor, group) in enumerate(sorted(grouped.items()), start=1):
            # 还原实际 anchor（去掉 split 后缀）
            display_anchor = anchor.split("::split_")[0]
            attr_values: Dict[str, List[str]] = {}
            evidence_snippets: List[str] = []
            provenance_turns: List[int] = []
            event_ids: List[str] = []
            evidence_chain: List[dict] = []

            for evt in group:
                values = attr_values.setdefault(evt.attribute, [])
                if evt.value not in values:
                    values.append(evt.value)
                snippet = (evt.raw_text_snippet or "").strip()
                if snippet and snippet not in evidence_snippets:
                    evidence_snippets.append(snippet)
                provenance_turns.extend(evt.provenance)
                event_ids.append(evt.event_id)
                evidence_chain.append(
                    self._evidence_entry(
                        action="merge",
                        reason="Temporal_CoAnchor",
                        event_id=evt.event_id,
                        turns=evt.provenance,
                        extra={"attribute": evt.attribute, "value": evt.value},
                    )
                )

            bundles.append(
                EventBundle(
                    bundle_id=f"event_{dialogue_id}_{i:03d}",
                    time_anchor=display_anchor,
                    attributes=attr_values,
                    evidence_snippets=evidence_snippets[:8],
                    provenance_turns=sorted(set(provenance_turns)),
                    event_ids=event_ids,
                    evidence_chain=[
                        self._evidence_entry(
                            action="create",
                            reason="Event_Bundle_Initialization",
                            turns=provenance_turns,
                            extra={"time_anchor": display_anchor},
                        ),
                        *evidence_chain,
                    ],
                )
            )

        return bundles

    def _causal_split(
        self,
        events: List[ExtractedEvent],
        anchor: str,
    ) -> List[List[ExtractedEvent]]:
        """
        对同一 time anchor 内的事件进行因果去混淆拆分。

        关键设计：只在同一 attribute 的事件之间做去混淆判断。
        不同 attribute 的事件在同一 time anchor 下应总是聚合
        （它们代表同一次临床会面的不同属性事实）。
        """
        if len(events) <= 1:
            return [events]

        scorer = self._get_causal_scorer()

        # 按 attribute 分组，仅对同 attribute 事件做去混淆拆分
        by_attr: Dict[str, List[ExtractedEvent]] = {}
        for evt in events:
            by_attr.setdefault(evt.attribute, []).append(evt)

        # 检查是否存在同 attribute 内需要拆分的情况
        needs_split = False
        split_map: Dict[str, List[List[ExtractedEvent]]] = {}
        for attr, attr_events in by_attr.items():
            if len(attr_events) <= 1:
                split_map[attr] = [attr_events]
                continue
            # 对同 attribute 事件做贪心聚类
            groups: List[List[ExtractedEvent]] = [[attr_events[0]]]
            for evt in attr_events[1:]:
                merged = False
                for group in groups:
                    if scorer.should_merge(group[0], evt):
                        group.append(evt)
                        merged = True
                        break
                if not merged:
                    groups.append([evt])
            split_map[attr] = groups
            if len(groups) > 1:
                needs_split = True

        if not needs_split:
            return [events]

        # 有拆分需求：将不同 attribute 的事件分配到对应的子组
        # 非拆分 attribute 的事件加入第一个组
        result_groups: List[List[ExtractedEvent]] = []
        non_split_events: List[ExtractedEvent] = []
        for attr, groups in split_map.items():
            if len(groups) == 1:
                non_split_events.extend(groups[0])
            else:
                result_groups.extend(groups)

        if non_split_events:
            if result_groups:
                result_groups[0].extend(non_split_events)
            else:
                result_groups.append(non_split_events)

        return result_groups

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
