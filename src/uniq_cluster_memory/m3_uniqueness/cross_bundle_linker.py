"""
m3_uniqueness/cross_bundle_linker.py
====================================
跨 Bundle 关联推理（Cross-Bundle Reasoning）：
在 M3 合并后，基于轻量规则连接"药物团 -> 症状团"等潜在关系。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from src.uniq_cluster_memory.m2_clustering import BundleGraph, BundleLink
from src.uniq_cluster_memory.schema import CanonicalMemory


CROSS_BUNDLE_RULES = [
    {
        "med_keywords": ["insulin", "insulin glargine"],
        "symptom_keywords": ["hypoglycemia", "low blood sugar", "dizziness", "sweating", "palpitations"],
        "relation": "POTENTIAL_ADVERSE_EFFECT",
        "reason": "Medication_Risk_Heuristic:Hypoglycemia",
        "confidence": 0.72,
    },
    {
        "med_keywords": ["lisinopril", "amlodipine"],
        "symptom_keywords": ["dizziness", "lightheadedness"],
        "relation": "POTENTIAL_ADVERSE_EFFECT",
        "reason": "Medication_Risk_Heuristic:Hypotension",
        "confidence": 0.64,
    },
]


class CrossBundleLinker:
    """在 M3 合并后，基于规则推理跨 bundle 的潜在关联。"""

    def __init__(
        self,
        bundle_graph: Optional[BundleGraph],
        event_bundle_by_id: Dict[str, object],
    ):
        self._bundle_graph = bundle_graph
        self._event_bundle_by_id = event_bundle_by_id

    def link_bundles(self, memories: List[CanonicalMemory]) -> None:
        """扫描所有 memories，根据规则建立跨 bundle 链接。"""
        if self._bundle_graph is None or not memories:
            return

        meds: List[Tuple[str, str, CanonicalMemory]] = []
        symptoms: List[Tuple[str, str, CanonicalMemory]] = []

        for mem in memories:
            bid = self._selected_bundle_id_from_memory(mem)
            if not bid:
                continue
            value_key = self._normalize_text_key(mem.value)
            if mem.attribute == "medication":
                meds.append((bid, value_key, mem))
            elif mem.attribute == "symptom":
                symptoms.append((bid, value_key, mem))

        for med_bid, med_value, med_mem in meds:
            for sym_bid, sym_value, sym_mem in symptoms:
                if med_bid == sym_bid:
                    continue
                for rule in CROSS_BUNDLE_RULES:
                    if not any(k in med_value for k in rule["med_keywords"]):
                        continue
                    if not any(k in sym_value for k in rule["symptom_keywords"]):
                        continue
                    self._append_cross_bundle_link(
                        src_bundle_id=med_bid,
                        dst_bundle_id=sym_bid,
                        relation=rule["relation"],
                        reason=rule["reason"],
                        confidence=float(rule["confidence"]),
                        metadata={
                            "medication_value": med_mem.value,
                            "symptom_value": sym_mem.value,
                        },
                    )
                    break

    def _append_cross_bundle_link(
        self,
        src_bundle_id: str,
        dst_bundle_id: str,
        relation: str,
        reason: str,
        confidence: float,
        metadata: Optional[dict] = None,
    ) -> None:
        if self._bundle_graph is None:
            return
        existing = {
            (link.src_bundle_id, link.dst_bundle_id, link.relation)
            for link in self._bundle_graph.links
        }
        key = (src_bundle_id, dst_bundle_id, relation)
        if key in existing:
            return

        self._bundle_graph.links.append(
            BundleLink(
                src_bundle_id=src_bundle_id,
                dst_bundle_id=dst_bundle_id,
                relation=relation,
                confidence=round(confidence, 3),
                reason=reason,
                metadata=metadata or {},
            )
        )
        self._append_bundle_evidence(
            src_bundle_id,
            action="link_out",
            reason=reason,
            extra={"dst": dst_bundle_id, "relation": relation},
        )
        self._append_bundle_evidence(
            dst_bundle_id,
            action="link_in",
            reason=reason,
            extra={"src": src_bundle_id, "relation": relation},
        )

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

    @staticmethod
    def _normalize_text_key(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    @staticmethod
    def _selected_bundle_id_from_memory(mem: CanonicalMemory) -> str:
        qualifiers = mem.qualifiers or {}
        bid = qualifiers.get("selected_event_bundle_id", "")
        if isinstance(bid, str):
            return bid
        return ""
