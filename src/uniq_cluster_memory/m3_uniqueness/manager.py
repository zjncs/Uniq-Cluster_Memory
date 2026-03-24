"""
m3_uniqueness/manager.py
=========================
M3 核心：唯一性管理器。

基于信息团（Information Bundle）优先的策略，对同一 (patient_id, attribute, time_scope)
的多条事件进行去重和合并，输出最终的 CanonicalMemory 列表。

核心概念：
    - 唯一性（Uniqueness）：同一属性在同一时间范围内只保留一条规范记忆。
    - 信息团（Information Bundle）：M2.5 构建的事件/实体聚合单元，
      是 M3 冲突检测与候选选择的基本粒度。

协作模块（已拆分）：
    - TimeGrounder（time_grounder.py）：时间表达 → 标准 time_scope
    - ConflictDetector（conflict_detector.py）：值/单位冲突检测
    - CrossBundleLinker（cross_bundle_linker.py）：跨 bundle 关联推理
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from src.uniq_cluster_memory.m1_event_extraction import ExtractedEvent
from src.uniq_cluster_memory.m2_clustering import AttributeCluster, BundleGraph
from src.uniq_cluster_memory.schema import CanonicalMemory, ConflictRecord, scope_to_interval

from .time_grounder import TimeGrounder
from .conflict_detector import ConflictDetector
from .cross_bundle_linker import CrossBundleLinker
from .formal_constraints import ConstraintChecker

from src.uniq_cluster_memory.schema import CandidateValue


class UniquenessManager:
    """
    M3 核心：基于信息团优先策略对事件进行去重和合并，输出 CanonicalMemory 列表。

    处理流程（对每个 AttributeCluster）：
        1. 对所有事件进行 Time Grounding，将 time_expr -> time_scope（标准日期）。
        2. 按 (attribute, time_scope) 分组。
        3. 对每组内的多条事件，根据 update_policy 进行合并：
           - "unique"：保留置信度最高的一条，其余标记为冲突。
           - "latest"：保留 provenance 最晚的一条（最新信息优先）。
           - "append"：保留所有不重复的值。
        4. 输出最终的 CanonicalMemory 列表。
    """

    BUNDLE_STATUS_ACTIVE = "Active"
    BUNDLE_STATUS_SUPERSEDED = "Superseded"
    BUNDLE_STATUS_CONFLICTING = "Conflicting"
    BUNDLE_STATUS_RESOLVED = "Resolved"
    VALID_BUNDLE_STATUSES = {
        BUNDLE_STATUS_ACTIVE,
        BUNDLE_STATUS_SUPERSEDED,
        BUNDLE_STATUS_CONFLICTING,
        BUNDLE_STATUS_RESOLVED,
    }

    MED_STRONG_ACTION_PATTERNS = [
        re.compile(r"\b(start|prescribe|prescribed|reduce|increase|continue|switch)\b", re.IGNORECASE),
        re.compile(r"\b(once daily|twice daily|bid|tid|qid|qd|qhs|at night)\b", re.IGNORECASE),
    ]
    MED_WEAK_ACTION_PATTERNS = [
        re.compile(r"\b(recommend|might recommend|may recommend|consider)\b", re.IGNORECASE),
        re.compile(r"\b(like|as needed|prn|if .* persists)\b", re.IGNORECASE),
        re.compile(r"\b(antipyretic|pain reliever)\b", re.IGNORECASE),
    ]

    def __init__(
        self,
        dialogue_date: Optional[str] = None,
        enable_time_grounding: bool = True,
        enable_conflict_detection: bool = True,
        missing_time_scope: str = "global",
        max_symptoms_per_scope: Optional[int] = None,
        enable_formal_constraints: bool = True,
    ):
        self.enable_time_grounding = enable_time_grounding
        self.enable_conflict_detection = enable_conflict_detection
        self.max_symptoms_per_scope = max_symptoms_per_scope
        self.time_grounder = (
            TimeGrounder(
                dialogue_date=dialogue_date,
                missing_time_scope=missing_time_scope,
            )
            if enable_time_grounding
            else None
        )
        self.conflict_detector = ConflictDetector() if enable_conflict_detection else None
        self.constraint_checker = ConstraintChecker(enabled=enable_formal_constraints)
        self._event_to_event_bundle: Dict[str, str] = {}
        self._event_to_entity_bundles: Dict[str, Set[str]] = {}
        self._entity_bundle_names: Dict[str, str] = {}
        self._bundle_time_anchor: Dict[str, str] = {}
        self._event_bundle_by_id: Dict[str, object] = {}
        self._entity_bundle_by_id: Dict[str, object] = {}
        self._bundle_graph: Optional[BundleGraph] = None

    def process(
        self,
        clusters: List[AttributeCluster],
        patient_id: str,
        bundle_graph: Optional[BundleGraph] = None,
    ) -> List[CanonicalMemory]:
        """将 AttributeCluster 列表转换为 CanonicalMemory 列表。"""
        self._index_bundle_graph(bundle_graph)
        all_memories: List[CanonicalMemory] = []

        for cluster in clusters:
            memories = self._process_cluster(cluster, patient_id, all_memories)
            all_memories.extend(memories)

        linker = CrossBundleLinker(self._bundle_graph, self._event_bundle_by_id)
        linker.link_bundles(all_memories)
        return all_memories

    # ── Bundle Graph Indexing ─────────────────────────────────────────────────

    def _index_bundle_graph(self, bundle_graph: Optional[BundleGraph]) -> None:
        self._bundle_graph = bundle_graph
        self._event_to_event_bundle = {}
        self._event_to_entity_bundles = {}
        self._entity_bundle_names = {}
        self._bundle_time_anchor = {}
        self._event_bundle_by_id = {}
        self._entity_bundle_by_id = {}
        if bundle_graph is None:
            return

        for eb in bundle_graph.event_bundles:
            self._event_bundle_by_id[eb.bundle_id] = eb
            if not getattr(eb, "status", ""):
                eb.status = self.BUNDLE_STATUS_ACTIVE
            self._bundle_time_anchor[eb.bundle_id] = eb.time_anchor
            for eid in eb.event_ids:
                self._event_to_event_bundle[eid] = eb.bundle_id

        for ent in bundle_graph.entity_bundles:
            self._entity_bundle_by_id[ent.bundle_id] = ent
            if not getattr(ent, "status", ""):
                ent.status = self.BUNDLE_STATUS_ACTIVE
            self._entity_bundle_names[ent.bundle_id] = ent.canonical_name
            for eid in ent.event_ids:
                self._event_to_entity_bundles.setdefault(eid, set()).add(ent.bundle_id)

        for link in bundle_graph.links:
            if link.relation != "MENTIONS_MEDICATION":
                continue
            event_bundle_id = link.src_bundle_id
            entity_bundle_id = link.dst_bundle_id
            for eid, e_bid in self._event_to_event_bundle.items():
                if e_bid == event_bundle_id:
                    self._event_to_entity_bundles.setdefault(eid, set()).add(entity_bundle_id)

    # ── Bundle Grouping Helpers ───────────────────────────────────────────────

    def _bundle_key_for_event(self, evt: ExtractedEvent) -> str:
        return self._event_to_event_bundle.get(evt.event_id, f"event::{evt.event_id}")

    def _bundle_groups(self, events: List[ExtractedEvent]) -> Dict[str, List[ExtractedEvent]]:
        grouped: Dict[str, List[ExtractedEvent]] = {}
        for evt in events:
            grouped.setdefault(self._bundle_key_for_event(evt), []).append(evt)
        return grouped

    def _bundle_groups_grounded(
        self,
        grounded_events: List[Tuple[str, ExtractedEvent]],
    ) -> Dict[str, List[Tuple[str, ExtractedEvent]]]:
        grouped: Dict[str, List[Tuple[str, ExtractedEvent]]] = {}
        for scope, evt in grounded_events:
            grouped.setdefault(self._bundle_key_for_event(evt), []).append((scope, evt))
        return grouped

    def _bundle_event_priority(self, events: List[ExtractedEvent]) -> Tuple[int, int, float, int]:
        top = max((self._event_priority(e) for e in events), default=(0, 0, 0.0))
        return top[0], top[1], top[2], len(events)

    def _bundle_latest_priority(
        self,
        grounded_events: List[Tuple[str, ExtractedEvent]],
    ) -> Tuple[int, int, int, int, float, int]:
        top = max(
            (self._latest_priority(scope, evt) for scope, evt in grounded_events),
            default=(-1, 1, 0, 0, 0.0),
        )
        return top[0], top[1], top[2], top[3], top[4], len(grounded_events)

    def _representative_event(self, events: List[ExtractedEvent]) -> ExtractedEvent:
        return sorted(events, key=lambda e: self._event_priority(e), reverse=True)[0]

    def _representative_grounded_event(
        self,
        grounded_events: List[Tuple[str, ExtractedEvent]],
    ) -> Tuple[str, ExtractedEvent]:
        return sorted(
            grounded_events,
            key=lambda pair: self._latest_priority(pair[0], pair[1]),
            reverse=True,
        )[0]

    # ── Bundle Status Lifecycle ───────────────────────────────────────────────

    def _append_bundle_evidence(
        self,
        bundle_id: str,
        action: str,
        reason: str,
        extra: Optional[dict] = None,
    ) -> None:
        bundle = self._event_bundle_by_id.get(bundle_id)
        if bundle is None:
            return
        chain = getattr(bundle, "evidence_chain", None)
        if chain is None:
            chain = []
            setattr(bundle, "evidence_chain", chain)
        entry = {"action": action, "reason": reason}
        if extra:
            entry["extra"] = extra
        chain.append(entry)

    def _set_bundle_status(
        self,
        bundle_id: str,
        new_status: str,
        reason: str,
        extra: Optional[dict] = None,
    ) -> None:
        if new_status not in self.VALID_BUNDLE_STATUSES:
            return
        bundle = self._event_bundle_by_id.get(bundle_id)
        if bundle is None:
            return
        old_status = getattr(bundle, "status", self.BUNDLE_STATUS_ACTIVE)
        if old_status != new_status:
            bundle.status = new_status
            self._append_bundle_evidence(
                bundle_id,
                action="status_transition",
                reason=reason,
                extra={"from": old_status, "to": new_status, **(extra or {})},
            )
        else:
            self._append_bundle_evidence(
                bundle_id,
                action="status_observed",
                reason=reason,
                extra={"status": new_status, **(extra or {})},
            )

    def _apply_bundle_decision(
        self,
        selected_bundle_id: str,
        competing_bundle_ids: List[str],
        policy: str,
        has_conflict: bool,
    ) -> None:
        if not selected_bundle_id:
            return
        if not has_conflict:
            self._set_bundle_status(
                selected_bundle_id,
                self.BUNDLE_STATUS_ACTIVE,
                reason=f"Bundle_Decision:{policy}",
            )
            for bid in competing_bundle_ids:
                self._set_bundle_status(
                    bid,
                    self.BUNDLE_STATUS_RESOLVED,
                    reason=f"Bundle_Decision:{policy}",
                    extra={"selected_bundle_id": selected_bundle_id},
                )
            return

        if policy == "latest":
            self._set_bundle_status(
                selected_bundle_id,
                self.BUNDLE_STATUS_ACTIVE,
                reason="Temporal_Recency",
                extra={"resolved_conflict": True},
            )
            for bid in competing_bundle_ids:
                self._set_bundle_status(
                    bid,
                    self.BUNDLE_STATUS_SUPERSEDED,
                    reason="Temporal_Recency",
                    extra={"selected_bundle_id": selected_bundle_id},
                )
            return

        # unique 场景：同 scope 下差异值通常无法完全确认，标为冲突待解
        self._set_bundle_status(
            selected_bundle_id,
            self.BUNDLE_STATUS_CONFLICTING,
            reason="SameScope_ValueConflict",
            extra={"requires_clarification": True},
        )
        for bid in competing_bundle_ids:
            self._set_bundle_status(
                bid,
                self.BUNDLE_STATUS_CONFLICTING,
                reason="SameScope_ValueConflict",
                extra={"selected_bundle_id": selected_bundle_id},
            )

    # ── Cluster Processing ────────────────────────────────────────────────────

    def _process_cluster(
        self,
        cluster: AttributeCluster,
        patient_id: str,
        existing_memories: Optional[List[CanonicalMemory]] = None,
    ) -> List[CanonicalMemory]:
        """处理单个 AttributeCluster，输出该属性的 CanonicalMemory 列表。"""
        attribute = cluster.canonical_attribute
        policy = cluster.update_policy

        # Step 1: Time Grounding（可关闭）
        grounded_events: List[Tuple[str, ExtractedEvent]] = []
        for evt in cluster.events:
            if self.enable_time_grounding and self.time_grounder is not None:
                time_scope = self.time_grounder.ground(evt.time_expr)
            else:
                time_scope = "global"
            grounded_events.append((time_scope, evt))

        # Step 2: latest 策略是全局覆盖，不按 time_scope 切分
        if policy == "latest":
            return self._merge_latest(
                grounded_events,
                attribute=attribute,
                time_scope="global",
                patient_id=patient_id,
                existing_memories=existing_memories or [],
            )

        # Step 3: 非 latest 策略按 time_scope 分组
        scope_groups: Dict[str, List[Tuple[str, ExtractedEvent]]] = {}
        for time_scope, evt in grounded_events:
            if time_scope not in scope_groups:
                scope_groups[time_scope] = []
            scope_groups[time_scope].append((time_scope, evt))

        # Step 4: 按 update_policy 合并
        memories: List[CanonicalMemory] = []

        for time_scope, group in scope_groups.items():
            evts = [e for _, e in group]

            if policy == "unique":
                memories.extend(
                    self._merge_unique(evts, attribute, time_scope, patient_id, existing_memories or [])
                )
            elif policy == "append":
                memories.extend(
                    self._merge_append(evts, attribute, time_scope, patient_id)
                )
            else:
                memories.extend(
                    self._merge_unique(evts, attribute, time_scope, patient_id, existing_memories or [])
                )

        return memories

    # ── Merge Strategies ──────────────────────────────────────────────────────

    def _merge_unique(
        self,
        events: List[ExtractedEvent],
        attribute: str,
        time_scope: str,
        patient_id: str,
        existing_memories: Optional[List[CanonicalMemory]] = None,
    ) -> List[CanonicalMemory]:
        """
        "unique" 策略：同一 (attribute, time_scope) 只保留一条记录。
        如果有多条不同值，保留置信度最高的，并记录冲突。
        """
        if not events:
            return []

        # 信息团优先：先在 event bundle 层选主候选，再在 bundle 内选代表事件
        grouped = self._bundle_groups(events)
        bundle_candidates = []
        for bundle_id, group_events in grouped.items():
            rep = self._representative_event(group_events)
            bundle_candidates.append(
                {
                    "bundle_id": bundle_id,
                    "events": group_events,
                    "rep": rep,
                    "rank": self._bundle_event_priority(group_events),
                }
            )
        bundle_candidates.sort(key=lambda c: c["rank"], reverse=True)
        selected = bundle_candidates[0]
        best: ExtractedEvent = selected["rep"]

        conflict_history: List[ConflictRecord] = []
        conflict_flag = False

        # 冲突按 bundle 级比较，避免同 bundle 内重复表达导致的误冲突放大
        if self.enable_conflict_detection and self.conflict_detector is not None:
            seen_values = {best.value.strip().lower()}
            for cand in bundle_candidates[1:]:
                evt = cand["rep"]
                if evt.value.strip().lower() in seen_values:
                    continue
                temp_mem = CanonicalMemory(
                    patient_id=patient_id,
                    attribute=attribute,
                    value=best.value,
                    unit=best.unit,
                    time_scope=time_scope,
                    confidence=best.confidence,
                    provenance=best.provenance,
                    conflict_flag=False,
                    conflict_history=[],
                    update_policy="unique",
                )
                conflict = self.conflict_detector.detect(temp_mem, evt)
                if conflict:
                    conflict_history.append(conflict)
                    conflict_flag = True
                seen_values.add(evt.value.strip().lower())

        # 合并所有来源的 provenance
        all_provenance = []
        seen_prov = set()
        for evt in events:
            for p in evt.provenance:
                if p not in seen_prov:
                    all_provenance.append(p)
                    seen_prov.add(p)
        all_provenance.sort()
        self._apply_bundle_decision(
            selected_bundle_id=selected["bundle_id"],
            competing_bundle_ids=[c["bundle_id"] for c in bundle_candidates[1:]],
            policy="unique",
            has_conflict=conflict_flag,
        )

        version_events = [cand["rep"] for cand in bundle_candidates]
        versions = self._build_versions_from_events(
            events_sorted=version_events,
            selected_event_id=best.event_id,
            default_scope=time_scope,
        )
        qualifiers = self._build_bundle_qualifiers(
            events=events,
            attribute=attribute,
            value_versions=versions,
            selected_bundle_id=selected["bundle_id"],
            competing_bundle_ids=[c["bundle_id"] for c in bundle_candidates[1:]],
            decision_level="bundle",
        )

        # 多候选置信度评分（双时态冲突图核心）
        if self.conflict_detector is not None and len(bundle_candidates) > 1:
            max_turn_global = max(
                (self._max_turn(e) for e in events), default=1
            )
            cand_values = self._build_candidate_values(bundle_candidates, max_turn_global)
            scored = self.conflict_detector.score_candidates(cand_values, attribute)
            # 形式约束调整（ALICE hybrid: formal logic + LLM）
            scored = self.constraint_checker.check_and_adjust(
                scored, existing_memories or [], attribute
            )
            qualifiers["candidate_values"] = [cv.to_dict() for cv in scored]
            # 多候选评分可以补充触发 conflict_flag
            if not conflict_flag and ConflictDetector.should_flag_conflict(scored):
                conflict_flag = True

        return [CanonicalMemory(
            patient_id=patient_id,
            attribute=attribute,
            value=best.value,
            unit=best.unit,
            time_scope=time_scope,
            confidence=best.confidence,
            provenance=all_provenance,
            conflict_flag=conflict_flag,
            conflict_history=conflict_history,
            update_policy="unique",
            qualifiers=qualifiers,
        )]

    def _merge_latest(
        self,
        grounded_events: List[Tuple[str, ExtractedEvent]],
        attribute: str,
        time_scope: str,
        patient_id: str,
        existing_memories: Optional[List[CanonicalMemory]] = None,
    ) -> List[CanonicalMemory]:
        """
        "latest" 策略：保留来自最晚轮次的记录（最新信息优先）。
        """
        if not grounded_events:
            return []

        events = [evt for _, evt in grounded_events]

        # 信息团优先：先在 bundle 层按"时间+来源优先级"排序，再选 bundle 内代表记录
        grouped = self._bundle_groups_grounded(grounded_events)
        bundle_candidates = []
        for bundle_id, group in grouped.items():
            rep_scope, rep_evt = self._representative_grounded_event(group)
            bundle_candidates.append(
                {
                    "bundle_id": bundle_id,
                    "group": group,
                    "rep_scope": rep_scope,
                    "rep_evt": rep_evt,
                    "rank": self._bundle_latest_priority(group),
                }
            )
        bundle_candidates.sort(key=lambda c: c["rank"], reverse=True)
        selected = bundle_candidates[0]
        latest = selected["rep_evt"]
        latest_scope = selected["rep_scope"]

        conflict_history: List[ConflictRecord] = []
        conflict_flag = False

        # 冲突按 bundle 级比较，减少同一事件团内噪声的误冲突
        if self.enable_conflict_detection and self.conflict_detector is not None:
            seen_values = {latest.value.strip().lower()}
            for cand in bundle_candidates[1:]:
                evt = cand["rep_evt"]
                if evt.value.strip().lower() in seen_values:
                    continue
                temp_mem = CanonicalMemory(
                    patient_id=patient_id,
                    attribute=attribute,
                    value=latest.value,
                    unit=latest.unit,
                    time_scope=time_scope,
                    confidence=latest.confidence,
                    provenance=latest.provenance,
                    conflict_flag=False,
                    conflict_history=[],
                    update_policy="latest",
                )
                conflict = self.conflict_detector.detect(temp_mem, evt)
                if conflict:
                    conflict_history.append(conflict)
                    conflict_flag = True
                seen_values.add(evt.value.strip().lower())

        all_provenance = sorted({p for evt in events for p in evt.provenance})
        self._apply_bundle_decision(
            selected_bundle_id=selected["bundle_id"],
            competing_bundle_ids=[c["bundle_id"] for c in bundle_candidates[1:]],
            policy="latest",
            has_conflict=conflict_flag,
        )

        # 为时间推理构建 medication 时间轴（不改变评测主键，仍保留 global）
        timeline = []
        if attribute == "medication":
            timeline = self._build_medication_timeline(grounded_events)

        latest_start, latest_end, _ = scope_to_interval(latest_scope)
        if latest_start is None and latest_end is None:
            latest_start = None
            latest_end = None
        if attribute == "medication":
            latest_end = None

        version_pairs = [(cand["rep_scope"], cand["rep_evt"]) for cand in bundle_candidates]
        versions = self._build_versions_from_grounded_events(
            grounded_events=version_pairs,
            selected_event_id=latest.event_id,
        )
        qualifiers = self._build_bundle_qualifiers(
            events=events,
            attribute=attribute,
            value_versions=versions,
            selected_bundle_id=selected["bundle_id"],
            competing_bundle_ids=[c["bundle_id"] for c in bundle_candidates[1:]],
            decision_level="bundle",
        )
        if timeline:
            qualifiers["med_timeline"] = timeline

        # 多候选置信度评分（双时态冲突图核心）
        if self.conflict_detector is not None and len(bundle_candidates) > 1:
            max_turn_global = max(
                (self._max_turn(e) for e in events), default=1
            )
            cand_values = self._build_candidate_values(bundle_candidates, max_turn_global)
            scored = self.conflict_detector.score_candidates(cand_values, attribute)
            # 形式约束调整
            scored = self.constraint_checker.check_and_adjust(
                scored, existing_memories or [], attribute
            )
            qualifiers["candidate_values"] = [cv.to_dict() for cv in scored]
            if not conflict_flag and ConflictDetector.should_flag_conflict(scored):
                conflict_flag = True

        return [CanonicalMemory(
            patient_id=patient_id,
            attribute=attribute,
            value=latest.value,
            unit=latest.unit,
            time_scope=time_scope,
            confidence=latest.confidence,
            provenance=all_provenance,
            conflict_flag=conflict_flag,
            conflict_history=conflict_history,
            update_policy="latest",
            start_time=latest_start,
            end_time=latest_end,
            is_ongoing=True,
            qualifiers=qualifiers,
        )]

    def _merge_append(
        self,
        events: List[ExtractedEvent],
        attribute: str,
        time_scope: str,
        patient_id: str,
    ) -> List[CanonicalMemory]:
        """
        "append" 策略：保留所有不重复的值（如症状列表）。
        """
        if not events:
            return []

        # 先按值去重；同值保留优先级更高记录
        seen_values: Dict[str, ExtractedEvent] = {}
        bucket_counts: Dict[str, int] = {}
        for evt in events:
            key = evt.value.strip().lower()
            bucket_counts[key] = bucket_counts.get(key, 0) + 1
            if key not in seen_values or self._event_priority(evt) > self._event_priority(seen_values[key]):
                seen_values[key] = evt

        items = list(seen_values.items())
        # 症状在 benchmark 中通常是主要主诉，默认可限制为每个 scope 最多 N 条
        if attribute == "symptom" and self.max_symptoms_per_scope is not None:
            items = sorted(
                items,
                key=lambda kv: (
                    bucket_counts.get(kv[0], 1),
                    self._speaker_priority(kv[1].speaker),
                    kv[1].confidence,
                ),
                reverse=True,
            )[: max(0, self.max_symptoms_per_scope)]

        memories = []
        for _, evt in items:
            versions = self._build_versions_from_events(
                events_sorted=[evt],
                selected_event_id=evt.event_id,
                default_scope=time_scope,
            )
            qualifiers = self._build_bundle_qualifiers(
                events=[evt],
                attribute=attribute,
                value_versions=versions,
                selected_bundle_id=self._event_to_event_bundle.get(evt.event_id, ""),
                decision_level="bundle",
            )
            memories.append(CanonicalMemory(
                patient_id=patient_id,
                attribute=attribute,
                value=evt.value,
                unit=evt.unit,
                time_scope=time_scope,
                confidence=evt.confidence,
                provenance=sorted(set(evt.provenance)),
                conflict_flag=False,
                conflict_history=[],
                update_policy="append",
                qualifiers=qualifiers,
            ))
        return memories

    # ── Candidate Scoring (Bi-Temporal Conflict Graph) ──────────────────────

    def _build_candidate_values(
        self,
        bundle_candidates: list,
        max_turn_global: int,
    ) -> list:
        """从 bundle 候选列表构建 CandidateValue 列表，用于多候选置信度评分。"""
        candidates = []
        for cand in bundle_candidates:
            evt = cand.get("rep") or cand.get("rep_evt")
            if evt is None:
                continue
            turn = self._max_turn(evt)
            recency = turn / max(max_turn_global, 1)
            candidates.append(CandidateValue(
                value=evt.value,
                unit=evt.unit,
                confidence=evt.confidence,
                provenance=sorted(set(evt.provenance)),
                speaker=evt.speaker,
                t_event=cand.get("rep_scope"),
                t_ingest=turn,
                source_authority=self._speaker_priority(evt.speaker) / 2.0,
                temporal_recency=recency,
                evidence_count=len(cand.get("events") or cand.get("group", [])),
            ))
        return candidates

    # ── Priority Functions ────────────────────────────────────────────────────

    @staticmethod
    def _speaker_priority(speaker: str) -> int:
        sp = (speaker or "").strip().lower()
        if sp == "doctor":
            return 2
        if sp == "patient":
            return 1
        return 0

    @staticmethod
    def _max_turn(evt: ExtractedEvent) -> int:
        return max(evt.provenance) if evt.provenance else 0

    def _medication_intent_priority(self, evt: ExtractedEvent) -> int:
        if evt.attribute != "medication":
            return 1
        text = f"{evt.value} {evt.raw_text_snippet}".strip().lower()
        if any(p.search(text) for p in self.MED_STRONG_ACTION_PATTERNS):
            return 2
        if any(p.search(text) for p in self.MED_WEAK_ACTION_PATTERNS):
            return 0
        return 1

    def _event_priority(self, evt: ExtractedEvent) -> Tuple[int, int, float]:
        return (
            self._max_turn(evt),
            self._speaker_priority(evt.speaker),
            evt.confidence,
        )

    @staticmethod
    def _scope_end_ordinal(scope: str) -> int:
        _, end, _ = scope_to_interval(scope)
        if not end:
            return -1
        try:
            return datetime.strptime(end, "%Y-%m-%d").toordinal()
        except ValueError:
            return -1

    def _latest_priority(self, scope: str, evt: ExtractedEvent) -> Tuple[int, int, int, int, float]:
        scope_rank = self._scope_end_ordinal(scope)
        intent_rank = self._medication_intent_priority(evt)
        # medication 的 latest 先看"处方意图"再看时间，减少临时建议药覆盖长期方案
        if evt.attribute == "medication":
            return (
                intent_rank,
                scope_rank,
                self._max_turn(evt),
                self._speaker_priority(evt.speaker),
                evt.confidence,
            )
        return (
            scope_rank,
            intent_rank,
            self._max_turn(evt),
            self._speaker_priority(evt.speaker),
            evt.confidence,
        )

    # ── Medication Timeline ───────────────────────────────────────────────────

    def _build_medication_timeline(
        self,
        grounded_events: List[Tuple[str, ExtractedEvent]],
    ) -> List[dict]:
        """构建药物时间轴，用于后续 query_meds_on 推理。"""
        if not grounded_events:
            return []

        from datetime import timedelta

        def max_turn(evt: ExtractedEvent) -> int:
            return max(evt.provenance) if evt.provenance else 0

        def start_key(scope: str, evt: ExtractedEvent) -> Tuple[str, int]:
            start, _, _ = scope_to_interval(scope)
            return (start or "9999-12-31", max_turn(evt))

        items = sorted(grounded_events, key=lambda x: start_key(x[0], x[1]))
        timeline: List[dict] = []

        for scope, evt in items:
            start, end, precision = scope_to_interval(scope)
            entry = {
                "value": evt.value,
                "start_time": start,
                "end_time": end,
                "time_precision": precision,
                "provenance": sorted(set(evt.provenance)),
            }

            if not timeline:
                timeline.append(entry)
                continue

            prev = timeline[-1]
            # 同药重复提及：合并时间范围
            if prev["value"].strip().lower() == evt.value.strip().lower():
                prev_start = prev.get("start_time")
                prev_end = prev.get("end_time")
                if start and (not prev_start or start < prev_start):
                    prev["start_time"] = start
                if end and (not prev_end or end > prev_end):
                    prev["end_time"] = end
                prev["provenance"] = sorted(set((prev.get("provenance") or []) + entry["provenance"]))
                continue

            # 新药开始时，将上个条目闭区间到前一天（保证区间连续）
            if start:
                try:
                    prev_end_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=1)
                    prev_end = prev_end_dt.strftime("%Y-%m-%d")
                    if not prev.get("end_time") or prev["end_time"] < prev_end:
                        prev["end_time"] = prev_end
                except ValueError:
                    pass
            timeline.append(entry)

        # 最新条目视为 ongoing
        if timeline:
            timeline[-1]["is_ongoing"] = True
            timeline[-1]["end_time"] = None

        return timeline

    # ── Version & Qualifier Builders ──────────────────────────────────────────

    def _build_versions_from_events(
        self,
        events_sorted: List[ExtractedEvent],
        selected_event_id: str,
        default_scope: str,
    ) -> List[dict]:
        versions: List[dict] = []
        seen_values = set()
        for evt in events_sorted:
            key = evt.value.strip().lower()
            if key in seen_values:
                continue
            seen_values.add(key)
            versions.append(
                {
                    "value": evt.value,
                    "time_scope": default_scope,
                    "provenance": sorted(set(evt.provenance)),
                    "confidence": evt.confidence,
                    "speaker": evt.speaker,
                    "event_bundle_id": self._event_to_event_bundle.get(evt.event_id, ""),
                    "entity_bundle_ids": sorted(self._event_to_entity_bundles.get(evt.event_id, set())),
                    "is_selected": evt.event_id == selected_event_id,
                }
            )
        return versions

    def _build_versions_from_grounded_events(
        self,
        grounded_events: List[Tuple[str, ExtractedEvent]],
        selected_event_id: str,
    ) -> List[dict]:
        versions: List[dict] = []
        seen = set()
        for scope, evt in grounded_events:
            key = (evt.value.strip().lower(), scope)
            if key in seen:
                continue
            seen.add(key)
            versions.append(
                {
                    "value": evt.value,
                    "time_scope": scope,
                    "provenance": sorted(set(evt.provenance)),
                    "confidence": evt.confidence,
                    "speaker": evt.speaker,
                    "event_bundle_id": self._event_to_event_bundle.get(evt.event_id, ""),
                    "entity_bundle_ids": sorted(self._event_to_entity_bundles.get(evt.event_id, set())),
                    "is_selected": evt.event_id == selected_event_id,
                }
            )
        return versions

    def _build_bundle_qualifiers(
        self,
        events: List[ExtractedEvent],
        attribute: str,
        value_versions: Optional[List[dict]] = None,
        selected_bundle_id: str = "",
        competing_bundle_ids: Optional[List[str]] = None,
        decision_level: str = "",
    ) -> dict:
        event_bundle_ids = sorted(
            {
                self._event_to_event_bundle[e.event_id]
                for e in events
                if e.event_id in self._event_to_event_bundle
            }
        )
        entity_bundle_ids = sorted(
            {
                ent
                for e in events
                for ent in self._event_to_entity_bundles.get(e.event_id, set())
            }
        )

        qualifiers: dict = {}
        if event_bundle_ids:
            qualifiers["event_bundle_ids"] = event_bundle_ids
        if entity_bundle_ids:
            qualifiers["entity_bundle_ids"] = entity_bundle_ids
        if attribute == "medication" and len(entity_bundle_ids) == 1:
            ent_id = entity_bundle_ids[0]
            qualifiers["medication_entity_id"] = ent_id
            if ent_id in self._entity_bundle_names:
                qualifiers["medication_entity_name"] = self._entity_bundle_names[ent_id]
        if selected_bundle_id:
            qualifiers["selected_event_bundle_id"] = selected_bundle_id
            anchor = self._bundle_time_anchor.get(selected_bundle_id)
            if anchor:
                qualifiers["selected_event_bundle_time_anchor"] = anchor
            selected_bundle = self._event_bundle_by_id.get(selected_bundle_id)
            if selected_bundle is not None:
                qualifiers["selected_event_bundle_status"] = getattr(
                    selected_bundle, "status", self.BUNDLE_STATUS_ACTIVE
                )
        if competing_bundle_ids:
            qualifiers["competing_event_bundle_ids"] = sorted(set(competing_bundle_ids))
        if decision_level:
            qualifiers["decision_level"] = decision_level
        if value_versions:
            qualifiers["value_versions"] = value_versions
        return qualifiers
