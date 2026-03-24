"""
temporal_reasoning/tcp_refinement.py
=====================================
TCP-in-the-Loop: Iterative temporal refinement for medical memory extraction.

Instead of running TCP as a post-hoc checker, this module integrates TCP into
the extraction pipeline as an iterative refinement loop:

    Round 1: M1 → M2 → M3 → TCP (detect inconsistencies)
    Round 2: For inconsistent memories, re-extract with temporal context
    Round 3: Merge refined memories, run TCP again to verify

This is the core algorithmic contribution: using Allen's Interval Algebra
constraint propagation to GUIDE extraction, not just CHECK it.

The key insight is that LLM-based extraction often produces temporally
inconsistent outputs (e.g., "medication started before diagnosis" or
"two different blood pressure values at the same time"). TCP can detect
these inconsistencies, and the refinement prompt provides the LLM with
the constraint violation context to produce corrected extractions.

Algorithm: TCP-Refine
    Input: memories M from initial pipeline, original dialogue D
    Output: refined memories M' with fewer temporal inconsistencies

    1. G ← BuildConstraintGraph(M)
    2. R ← Propagate(G)                    // Allen's PC-2
    3. I ← {(m_i, m_j) | R(m_i, m_j) = ∅}  // inconsistent pairs
    4. If |I| = 0: return M                 // already consistent
    5. For each inconsistent group g ∈ cluster(I):
       a. context ← ExtractTemporalContext(g, M)
       b. segments ← FindRelevantDialogue(g, D)
       c. m_refined ← LLM_ReExtract(segments, context)
       d. M ← Replace(M, g, m_refined)
    6. G' ← BuildConstraintGraph(M)
    7. R' ← Propagate(G')
    8. return M

Complexity: O(k · n² · |R|³) where k = #refinement rounds (typically 1-2)
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Set, Tuple

from src.uniq_cluster_memory.schema import CanonicalMemory
from src.uniq_cluster_memory.temporal_reasoning.constraint_propagation import (
    Rel,
    TCPResult,
    TemporalConstraintGraph,
    run_tcp,
)
from src.uniq_cluster_memory.utils.llm_client import get_llm_client, LLM_MODEL_FAST


# ── Refinement prompt ─────────────────────────────────────────────────────────

REFINEMENT_PROMPT = """You are a medical temporal reasoning expert. I found temporal inconsistencies in extracted medical memories.

## Inconsistencies detected:
{inconsistencies}

## Current memories involved:
{memories}

## Original dialogue segments:
{dialogue_segments}

## Task:
Re-examine the dialogue and fix the temporal inconsistencies. For each memory, determine:
1. Is the time_scope correct? If not, what should it be?
2. Is the value correct? If not, what should it be?
3. Should this memory be merged with another? (same fact, different time expressions)
4. Should this memory be removed? (hallucinated or duplicate)

Return a JSON array of corrected memories. Each memory should have:
- "attribute": string
- "value": string
- "time_scope": string (ISO date like "2024-01-15", or "global")
- "confidence": float (0-1)
- "action": "keep" | "update" | "remove"
- "reason": brief explanation

Return ONLY the JSON array. No markdown."""


class TCPRefinementLoop:
    """
    TCP-in-the-Loop iterative refinement.

    Detects temporal inconsistencies via constraint propagation,
    then uses LLM to re-extract and correct problematic memories.
    """

    def __init__(self, max_rounds: int = 2):
        self.max_rounds = max_rounds
        self._llm_client = None

    def _get_client(self):
        if self._llm_client is None:
            self._llm_client = get_llm_client()
        return self._llm_client

    def refine(
        self,
        memories: List[CanonicalMemory],
        dialogue: List[dict],
        patient_id: str,
    ) -> Tuple[List[CanonicalMemory], dict]:
        """
        Run TCP-in-the-loop refinement.

        Args:
            memories: Initial memories from M1→M2→M3 pipeline
            dialogue: Original dialogue turns
            patient_id: Patient/dialogue ID

        Returns:
            (refined_memories, stats)
        """
        stats = {
            "rounds": 0,
            "initial_memories": len(memories),
            "initial_inconsistencies": 0,
            "final_inconsistencies": 0,
            "memories_updated": 0,
            "memories_removed": 0,
            "memories_kept": 0,
        }

        if len(memories) < 2:
            return memories, stats

        for round_num in range(self.max_rounds):
            # Run TCP
            updated_memories, tcp_result = run_tcp(memories)

            if round_num == 0:
                stats["initial_inconsistencies"] = tcp_result.n_inconsistencies

            # No inconsistencies → done
            if tcp_result.n_inconsistencies == 0:
                stats["rounds"] = round_num + 1
                stats["final_inconsistencies"] = 0
                return updated_memories, stats

            # Cluster inconsistent memories
            inconsistent_groups = self._cluster_inconsistencies(
                tcp_result, updated_memories
            )

            if not inconsistent_groups:
                stats["rounds"] = round_num + 1
                stats["final_inconsistencies"] = tcp_result.n_inconsistencies
                return updated_memories, stats

            # Refine each group
            refined_memories = list(updated_memories)
            for group in inconsistent_groups:
                corrections = self._refine_group(
                    group, refined_memories, dialogue, patient_id
                )
                refined_memories = self._apply_corrections(
                    refined_memories, group, corrections, stats
                )

            memories = refined_memories
            stats["rounds"] = round_num + 1

        # Final TCP check
        final_memories, final_tcp = run_tcp(memories)
        stats["final_inconsistencies"] = final_tcp.n_inconsistencies
        stats["final_memories"] = len(final_memories)

        return final_memories, stats

    def _cluster_inconsistencies(
        self,
        tcp_result: TCPResult,
        memories: List[CanonicalMemory],
    ) -> List[List[CanonicalMemory]]:
        """Group inconsistent memories into clusters for batch refinement."""
        # Build memory lookup
        mem_by_id: Dict[str, CanonicalMemory] = {}
        for m in memories:
            mid = f"{m.patient_id}_{m.attribute}_{m.value}"
            mem_by_id[mid] = m

        # Union-Find to cluster connected inconsistencies
        parent: Dict[str, str] = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for mid_a, mid_b, desc in tcp_result.inconsistent_pairs:
            if mid_a in mem_by_id and mid_b in mem_by_id:
                union(mid_a, mid_b)

        # Group by root
        groups: Dict[str, List[CanonicalMemory]] = {}
        for mid_a, mid_b, desc in tcp_result.inconsistent_pairs:
            for mid in [mid_a, mid_b]:
                if mid in mem_by_id:
                    root = find(mid)
                    if root not in groups:
                        groups[root] = []
                    if mem_by_id[mid] not in groups[root]:
                        groups[root].append(mem_by_id[mid])

        return list(groups.values())

    def _refine_group(
        self,
        group: List[CanonicalMemory],
        all_memories: List[CanonicalMemory],
        dialogue: List[dict],
        patient_id: str,
    ) -> List[dict]:
        """Use LLM to refine a group of inconsistent memories."""
        # Build inconsistency description
        inconsistency_desc = []
        for i, m in enumerate(group):
            inconsistency_desc.append(
                f"Memory {i+1}: {m.attribute}={m.value} "
                f"(time={m.time_scope}, confidence={m.confidence})"
            )

        # Find relevant dialogue segments
        all_provenance = set()
        for m in group:
            all_provenance.update(m.provenance)

        # Expand to include context (±2 turns)
        expanded = set()
        for p in all_provenance:
            for offset in range(-2, 3):
                expanded.add(p + offset)

        relevant_turns = []
        for turn in dialogue:
            tid = turn.get("turn_id", 0)
            if tid in expanded:
                speaker = turn.get("speaker", "unknown")
                text = turn.get("text", "")
                relevant_turns.append(f"[Turn {tid}, {speaker}]: {text}")

        if not relevant_turns:
            # Fallback: use all turns
            for turn in dialogue[:20]:
                tid = turn.get("turn_id", 0)
                speaker = turn.get("speaker", "unknown")
                text = turn.get("text", "")
                relevant_turns.append(f"[Turn {tid}, {speaker}]: {text}")

        memories_desc = "\n".join(inconsistency_desc)
        dialogue_text = "\n".join(relevant_turns[:30])  # Limit

        prompt = REFINEMENT_PROMPT.format(
            inconsistencies=memories_desc,
            memories=memories_desc,
            dialogue_segments=dialogue_text,
        )

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=LLM_MODEL_FAST,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
            )
            result = (response.choices[0].message.content or "").strip()
            return self._parse_refinement_response(result)
        except Exception:
            return []

    def _parse_refinement_response(self, raw: str) -> List[dict]:
        """Parse LLM refinement response."""
        # Strip markdown
        if raw.startswith("```"):
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("["):
                    raw = part
                    break

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\[[\s\S]*\]", raw)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return []

    def _apply_corrections(
        self,
        memories: List[CanonicalMemory],
        group: List[CanonicalMemory],
        corrections: List[dict],
        stats: dict,
    ) -> List[CanonicalMemory]:
        """Apply LLM corrections to the memory list."""
        if not corrections:
            return memories

        # Build lookup for group memories
        group_keys = set()
        for m in group:
            group_keys.add((m.attribute, m.value, m.time_scope))

        result = []
        for mem in memories:
            key = (mem.attribute, mem.value, mem.time_scope)
            if key not in group_keys:
                result.append(mem)
                continue

            # Find matching correction
            matched = False
            for corr in corrections:
                corr_attr = corr.get("attribute", "")
                corr_val = corr.get("value", "")
                action = corr.get("action", "keep")

                # Match by attribute (fuzzy)
                if corr_attr.lower() != mem.attribute.lower():
                    continue

                if action == "remove":
                    stats["memories_removed"] = stats.get("memories_removed", 0) + 1
                    matched = True
                    break
                elif action == "update":
                    # Apply corrections
                    if corr.get("time_scope"):
                        mem.time_scope = corr["time_scope"]
                        mem.start_time = corr["time_scope"] if re.match(r"\d{4}-\d{2}-\d{2}", corr["time_scope"]) else mem.start_time
                    if corr.get("value"):
                        mem.value = corr["value"]
                    if corr.get("confidence"):
                        mem.confidence = corr["confidence"]
                    stats["memories_updated"] = stats.get("memories_updated", 0) + 1
                    result.append(mem)
                    matched = True
                    break
                else:  # keep
                    stats["memories_kept"] = stats.get("memories_kept", 0) + 1
                    result.append(mem)
                    matched = True
                    break

            if not matched:
                result.append(mem)

        return result
