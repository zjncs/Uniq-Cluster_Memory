"""
temporal_reasoning/constraint_propagation.py
=============================================
Temporal Constraint Propagation (TCP) for Medical Dialogue Memory.

Core algorithmic contribution: instead of treating each memory's time scope
independently, TCP builds a temporal constraint graph over all memories and
propagates constraints to detect and resolve temporal inconsistencies.

Theoretical basis:
    Allen's Interval Algebra (Allen 1983) defines 13 basic relations between
    temporal intervals. We use a simplified set of 7 relations suited to
    medical dialogue:
        BEFORE, AFTER, DURING, CONTAINS, OVERLAPS, MEETS, EQUALS

    Constraint propagation uses path consistency (PC-2): for any three
    intervals (i, j, k), the relation R(i,k) must be consistent with the
    composition R(i,j) ∘ R(j,k). Violations indicate temporal inconsistencies
    in the extracted memories.

Algorithm:
    1. BUILD: Construct a Temporal Constraint Graph (TCG) from memories
       - Nodes: medical events with time intervals
       - Edges: temporal relations inferred from time scopes
    2. PROPAGATE: Apply path consistency until fixpoint
       - For each triple (i,j,k): R(i,k) ← R(i,k) ∩ (R(i,j) ∘ R(j,k))
       - If any R(i,k) becomes empty → inconsistency detected
    3. RESOLVE: Use propagated constraints to:
       - Flag temporally inconsistent memories
       - Tighten ambiguous time scopes
       - Improve conflict detection via temporal ordering

Complexity: O(n³ · |R|³) where n = #memories, |R| = #relation types = 7
            In practice n < 50 per dialogue, so this is fast.

Reference:
    - Allen, J.F. (1983). "Maintaining knowledge about temporal intervals."
      Communications of the ACM, 26(11), 832-843.
    - Dechter, R., Meiri, I., & Pearl, J. (1991). "Temporal constraint networks."
      Artificial Intelligence, 49(1-3), 61-95.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntFlag, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from src.uniq_cluster_memory.schema import CanonicalMemory


# ══════════════════════════════════════════════════════════════════════════════
#  Allen's Interval Relations (simplified for medical domain)
# ══════════════════════════════════════════════════════════════════════════════

class Rel(IntFlag):
    """
    Simplified Allen's interval relations as a bitflag set.

    Using bitflags allows efficient set intersection via &, union via |.
    A constraint R(i,j) is a set of possible relations (disjunction).
    """
    BEFORE    = auto()   # i entirely before j:   [i_s, i_e] < [j_s, j_e]
    AFTER     = auto()   # i entirely after j:    [i_s, i_e] > [j_s, j_e]
    MEETS     = auto()   # i.end == j.start (or within 1 day)
    OVERLAPS  = auto()   # i and j partially overlap
    DURING    = auto()   # i is contained within j
    CONTAINS  = auto()   # i contains j
    EQUALS    = auto()   # i and j have same interval

    @classmethod
    def ALL(cls) -> 'Rel':
        """Universal relation: any relation is possible."""
        return (cls.BEFORE | cls.AFTER | cls.MEETS | cls.OVERLAPS |
                cls.DURING | cls.CONTAINS | cls.EQUALS)

    @classmethod
    def EMPTY(cls) -> 'Rel':
        return cls(0)


def inverse(r: Rel) -> Rel:
    """Compute the inverse of a relation set: R⁻¹(i,j) = R(j,i)."""
    result = Rel(0)
    if Rel.BEFORE in r:
        result |= Rel.AFTER
    if Rel.AFTER in r:
        result |= Rel.BEFORE
    if Rel.MEETS in r:
        result |= Rel.MEETS     # meets is its own inverse in simplified model
    if Rel.OVERLAPS in r:
        result |= Rel.OVERLAPS  # symmetric in simplified model
    if Rel.DURING in r:
        result |= Rel.CONTAINS
    if Rel.CONTAINS in r:
        result |= Rel.DURING
    if Rel.EQUALS in r:
        result |= Rel.EQUALS
    return result


# ── Composition Table ─────────────────────────────────────────────────────────
# compose(R1, R2) returns the set of possible relations for R1 ∘ R2
# This is the core of Allen's algebra: if i R1 j and j R2 k, what can i R? k be?
#
# Simplified composition table (conservative: returns ALL when uncertain)

_COMPOSE_TABLE: Dict[Tuple[Rel, Rel], Rel] = {}


def _init_compose_table():
    """Initialize the composition table for simplified Allen's relations."""
    ALL = Rel.ALL()

    # BEFORE ∘ X
    _COMPOSE_TABLE[(Rel.BEFORE, Rel.BEFORE)] = Rel.BEFORE
    _COMPOSE_TABLE[(Rel.BEFORE, Rel.AFTER)] = ALL
    _COMPOSE_TABLE[(Rel.BEFORE, Rel.MEETS)] = Rel.BEFORE
    _COMPOSE_TABLE[(Rel.BEFORE, Rel.OVERLAPS)] = Rel.BEFORE
    _COMPOSE_TABLE[(Rel.BEFORE, Rel.DURING)] = Rel.BEFORE | Rel.OVERLAPS | Rel.DURING
    _COMPOSE_TABLE[(Rel.BEFORE, Rel.CONTAINS)] = Rel.BEFORE
    _COMPOSE_TABLE[(Rel.BEFORE, Rel.EQUALS)] = Rel.BEFORE

    # AFTER ∘ X
    _COMPOSE_TABLE[(Rel.AFTER, Rel.BEFORE)] = ALL
    _COMPOSE_TABLE[(Rel.AFTER, Rel.AFTER)] = Rel.AFTER
    _COMPOSE_TABLE[(Rel.AFTER, Rel.MEETS)] = Rel.AFTER
    _COMPOSE_TABLE[(Rel.AFTER, Rel.OVERLAPS)] = Rel.AFTER | Rel.OVERLAPS
    _COMPOSE_TABLE[(Rel.AFTER, Rel.DURING)] = Rel.AFTER | Rel.OVERLAPS | Rel.DURING
    _COMPOSE_TABLE[(Rel.AFTER, Rel.CONTAINS)] = Rel.AFTER
    _COMPOSE_TABLE[(Rel.AFTER, Rel.EQUALS)] = Rel.AFTER

    # MEETS ∘ X
    _COMPOSE_TABLE[(Rel.MEETS, Rel.BEFORE)] = Rel.BEFORE
    _COMPOSE_TABLE[(Rel.MEETS, Rel.AFTER)] = Rel.AFTER
    _COMPOSE_TABLE[(Rel.MEETS, Rel.MEETS)] = Rel.BEFORE
    _COMPOSE_TABLE[(Rel.MEETS, Rel.OVERLAPS)] = Rel.OVERLAPS
    _COMPOSE_TABLE[(Rel.MEETS, Rel.DURING)] = Rel.OVERLAPS | Rel.DURING
    _COMPOSE_TABLE[(Rel.MEETS, Rel.CONTAINS)] = Rel.BEFORE | Rel.MEETS
    _COMPOSE_TABLE[(Rel.MEETS, Rel.EQUALS)] = Rel.MEETS

    # OVERLAPS ∘ X
    _COMPOSE_TABLE[(Rel.OVERLAPS, Rel.BEFORE)] = Rel.BEFORE
    _COMPOSE_TABLE[(Rel.OVERLAPS, Rel.AFTER)] = ALL
    _COMPOSE_TABLE[(Rel.OVERLAPS, Rel.MEETS)] = Rel.BEFORE | Rel.OVERLAPS
    _COMPOSE_TABLE[(Rel.OVERLAPS, Rel.OVERLAPS)] = Rel.BEFORE | Rel.OVERLAPS
    _COMPOSE_TABLE[(Rel.OVERLAPS, Rel.DURING)] = ALL
    _COMPOSE_TABLE[(Rel.OVERLAPS, Rel.CONTAINS)] = Rel.BEFORE | Rel.OVERLAPS | Rel.CONTAINS
    _COMPOSE_TABLE[(Rel.OVERLAPS, Rel.EQUALS)] = Rel.OVERLAPS

    # DURING ∘ X
    _COMPOSE_TABLE[(Rel.DURING, Rel.BEFORE)] = Rel.BEFORE
    _COMPOSE_TABLE[(Rel.DURING, Rel.AFTER)] = Rel.AFTER
    _COMPOSE_TABLE[(Rel.DURING, Rel.MEETS)] = Rel.BEFORE
    _COMPOSE_TABLE[(Rel.DURING, Rel.OVERLAPS)] = Rel.BEFORE | Rel.OVERLAPS | Rel.DURING
    _COMPOSE_TABLE[(Rel.DURING, Rel.DURING)] = Rel.DURING
    _COMPOSE_TABLE[(Rel.DURING, Rel.CONTAINS)] = ALL
    _COMPOSE_TABLE[(Rel.DURING, Rel.EQUALS)] = Rel.DURING

    # CONTAINS ∘ X
    _COMPOSE_TABLE[(Rel.CONTAINS, Rel.BEFORE)] = Rel.BEFORE | Rel.OVERLAPS | Rel.CONTAINS
    _COMPOSE_TABLE[(Rel.CONTAINS, Rel.AFTER)] = Rel.AFTER | Rel.OVERLAPS | Rel.CONTAINS
    _COMPOSE_TABLE[(Rel.CONTAINS, Rel.MEETS)] = Rel.OVERLAPS | Rel.CONTAINS
    _COMPOSE_TABLE[(Rel.CONTAINS, Rel.OVERLAPS)] = Rel.OVERLAPS | Rel.CONTAINS
    _COMPOSE_TABLE[(Rel.CONTAINS, Rel.DURING)] = ALL
    _COMPOSE_TABLE[(Rel.CONTAINS, Rel.CONTAINS)] = Rel.CONTAINS
    _COMPOSE_TABLE[(Rel.CONTAINS, Rel.EQUALS)] = Rel.CONTAINS

    # EQUALS ∘ X  (identity)
    for r in Rel:
        if r == Rel(0):
            continue
        _COMPOSE_TABLE[(Rel.EQUALS, r)] = r


_init_compose_table()


def compose(r1: Rel, r2: Rel) -> Rel:
    """
    Compose two relation sets: R1 ∘ R2.

    For disjunctive relations (multiple bits set), we take the union of
    all pairwise compositions.
    """
    result = Rel(0)
    for a in Rel:
        if a == Rel(0) or not (a & r1):
            continue
        for b in Rel:
            if b == Rel(0) or not (b & r2):
                continue
            comp = _COMPOSE_TABLE.get((a, b), Rel.ALL())
            result |= comp
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Temporal Constraint Graph
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TemporalNode:
    """A node in the temporal constraint graph."""
    memory_id: str          # unique identifier
    attribute: str
    value: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    is_global: bool = False
    patient_id: str = ""

    @property
    def has_interval(self) -> bool:
        return self.start is not None


@dataclass
class TCPResult:
    """Result of temporal constraint propagation."""
    n_nodes: int = 0
    n_edges: int = 0
    n_inconsistencies: int = 0
    n_tightened: int = 0
    n_iterations: int = 0
    inconsistent_pairs: List[Tuple[str, str, str]] = field(default_factory=list)
    # (mem_id_1, mem_id_2, description)
    tightened_scopes: List[Tuple[str, str, str]] = field(default_factory=list)
    # (mem_id, old_scope, new_scope)

    def to_dict(self) -> dict:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_inconsistencies": self.n_inconsistencies,
            "n_tightened": self.n_tightened,
            "n_iterations": self.n_iterations,
            "inconsistent_pairs": [
                {"mem1": a, "mem2": b, "desc": c}
                for a, b, c in self.inconsistent_pairs
            ],
            "tightened_scopes": [
                {"mem_id": a, "old": b, "new": c}
                for a, b, c in self.tightened_scopes
            ],
        }


class TemporalConstraintGraph:
    """
    Temporal Constraint Graph with Allen's Interval Algebra.

    Implements path consistency (PC-2) for detecting and resolving
    temporal inconsistencies in medical dialogue memories.
    """

    def __init__(self):
        self.nodes: Dict[str, TemporalNode] = {}
        # constraints[i][j] = Rel set (possible relations from i to j)
        self.constraints: Dict[str, Dict[str, Rel]] = {}

    def add_node(self, node: TemporalNode) -> None:
        self.nodes[node.memory_id] = node
        if node.memory_id not in self.constraints:
            self.constraints[node.memory_id] = {}

    def get_relation(self, i: str, j: str) -> Rel:
        """Get the current constraint between nodes i and j."""
        if i == j:
            return Rel.EQUALS
        return self.constraints.get(i, {}).get(j, Rel.ALL())

    def set_relation(self, i: str, j: str, r: Rel) -> None:
        """Set the constraint between nodes i and j (and its inverse for j,i)."""
        if i not in self.constraints:
            self.constraints[i] = {}
        if j not in self.constraints:
            self.constraints[j] = {}
        self.constraints[i][j] = r
        self.constraints[j][i] = inverse(r)

    def infer_relation(self, a: TemporalNode, b: TemporalNode) -> Rel:
        """
        Infer the temporal relation between two nodes from their intervals.

        Medical domain heuristics:
        - Same-day events: EQUALS or OVERLAPS
        - Medication started before diagnosis: BEFORE or OVERLAPS
        - Ongoing events: CONTAINS current events
        """
        if not a.has_interval or not b.has_interval:
            return Rel.ALL()

        a_start, a_end = a.start, a.end or a.start
        b_start, b_end = b.start, b.end or b.start

        # Tolerance: events within 1 day are considered MEETS
        TOLERANCE = timedelta(days=1)

        if a_end < b_start - TOLERANCE:
            return Rel.BEFORE
        if a_start > b_end + TOLERANCE:
            return Rel.AFTER
        if abs((a_end - b_start).days) <= 1 and a_start < b_start:
            return Rel.MEETS

        # Check containment
        if a_start <= b_start and a_end >= b_end:
            if a_start == b_start and a_end == b_end:
                return Rel.EQUALS
            return Rel.CONTAINS
        if b_start <= a_start and b_end >= a_end:
            return Rel.DURING

        # Partial overlap
        return Rel.OVERLAPS

    def build_from_memories(self, memories: List[CanonicalMemory]) -> None:
        """
        Build the TCG from a list of CanonicalMemory objects.

        Groups memories by patient_id and infers pairwise temporal relations.
        """
        for mem in memories:
            node = self._memory_to_node(mem)
            self.add_node(node)

        # Infer pairwise relations
        node_ids = list(self.nodes.keys())
        for i_idx in range(len(node_ids)):
            for j_idx in range(i_idx + 1, len(node_ids)):
                ni = self.nodes[node_ids[i_idx]]
                nj = self.nodes[node_ids[j_idx]]

                # Only constrain memories of the same patient
                if ni.patient_id != nj.patient_id:
                    continue

                rel = self.infer_relation(ni, nj)

                # Apply medical domain constraints
                rel = self._apply_medical_constraints(ni, nj, rel)

                if rel != Rel.ALL():
                    self.set_relation(ni.memory_id, nj.memory_id, rel)

    def propagate(self, max_iterations: int = 100) -> TCPResult:
        """
        Path consistency propagation (PC-2 algorithm).

        For each triple (i, j, k):
            R(i,k) ← R(i,k) ∩ compose(R(i,j), R(j,k))

        Repeat until no changes or max_iterations reached.

        Returns:
            TCPResult with inconsistency and tightening statistics.
        """
        result = TCPResult()
        result.n_nodes = len(self.nodes)

        # Count initial edges
        for i in self.constraints:
            for j in self.constraints[i]:
                if i < j:
                    result.n_edges += 1

        node_ids = list(self.nodes.keys())
        n = len(node_ids)

        for iteration in range(max_iterations):
            changed = False

            for k_idx in range(n):
                k = node_ids[k_idx]
                for i_idx in range(n):
                    if i_idx == k_idx:
                        continue
                    i = node_ids[i_idx]
                    r_ik = self.get_relation(i, k)

                    for j_idx in range(n):
                        if j_idx == i_idx or j_idx == k_idx:
                            continue
                        j = node_ids[j_idx]

                        r_ij = self.get_relation(i, j)
                        r_jk = self.get_relation(j, k)

                        # Skip if no information to propagate
                        if r_ij == Rel.ALL() or r_jk == Rel.ALL():
                            continue

                        composed = compose(r_ij, r_jk)
                        new_r_ik = r_ik & composed

                        if new_r_ik != r_ik:
                            if new_r_ik == Rel(0):
                                # Inconsistency detected!
                                ni = self.nodes[i]
                                nk = self.nodes[k]
                                desc = (
                                    f"Temporal inconsistency: "
                                    f"{ni.attribute}={ni.value} vs "
                                    f"{nk.attribute}={nk.value} "
                                    f"(via {self.nodes[j].attribute}={self.nodes[j].value})"
                                )
                                result.inconsistent_pairs.append((i, k, desc))
                                result.n_inconsistencies += 1
                                # Don't set empty constraint; keep previous
                            else:
                                self.set_relation(i, k, new_r_ik)
                                changed = True

                                # Track tightening
                                if bin(new_r_ik).count('1') < bin(r_ik).count('1'):
                                    result.n_tightened += 1

            result.n_iterations = iteration + 1
            if not changed:
                break

        return result

    def get_temporal_ordering(self) -> List[str]:
        """
        Extract a topological ordering from BEFORE/AFTER constraints.

        Returns memory_ids in temporal order (earliest first).
        """
        # Build a partial order from BEFORE constraints
        before_edges: Dict[str, Set[str]] = {nid: set() for nid in self.nodes}

        for i in self.constraints:
            for j, rel in self.constraints[i].items():
                if rel == Rel.BEFORE or rel == Rel.MEETS:
                    before_edges[i].add(j)

        # Topological sort (Kahn's algorithm)
        in_degree = {nid: 0 for nid in self.nodes}
        for src, dsts in before_edges.items():
            for dst in dsts:
                in_degree[dst] = in_degree.get(dst, 0) + 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        ordering = []

        while queue:
            # Sort by start time for deterministic ordering
            queue.sort(key=lambda nid: str(self.nodes[nid].start or ""))
            node = queue.pop(0)
            ordering.append(node)
            for neighbor in before_edges.get(node, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Add any remaining nodes (cycles)
        for nid in self.nodes:
            if nid not in ordering:
                ordering.append(nid)

        return ordering

    def tighten_memory_scopes(
        self,
        memories: List[CanonicalMemory],
    ) -> Tuple[List[CanonicalMemory], List[Tuple[str, str, str]]]:
        """
        Use propagated constraints to tighten ambiguous time scopes.

        For memories with time_scope="global", if TCP has determined
        they must be BEFORE or AFTER another memory with a known date,
        we can narrow down their time scope.

        Returns:
            (updated_memories, list of (mem_id, old_scope, new_scope))
        """
        tightened = []
        updated = []

        for mem in memories:
            mem_id = f"{mem.patient_id}_{mem.attribute}_{mem.value}"
            node = self.nodes.get(mem_id)

            if node is None or node.has_interval:
                updated.append(mem)
                continue

            # This memory has no interval; try to infer from constraints
            earliest_bound = None
            latest_bound = None

            for other_id, rel in self.constraints.get(mem_id, {}).items():
                other_node = self.nodes.get(other_id)
                if other_node is None or not other_node.has_interval:
                    continue

                if rel == Rel.AFTER and other_node.end:
                    # This memory is AFTER other → starts after other.end
                    if earliest_bound is None or other_node.end > earliest_bound:
                        earliest_bound = other_node.end

                if rel == Rel.BEFORE and other_node.start:
                    # This memory is BEFORE other → ends before other.start
                    if latest_bound is None or other_node.start < latest_bound:
                        latest_bound = other_node.start

            if earliest_bound is not None or latest_bound is not None:
                old_scope = mem.time_scope
                new_mem = copy.copy(mem)

                if earliest_bound and latest_bound:
                    new_scope = (f"{earliest_bound.strftime('%Y-%m-%d')}.."
                                f"{latest_bound.strftime('%Y-%m-%d')}")
                elif earliest_bound:
                    new_scope = f">={earliest_bound.strftime('%Y-%m-%d')}"
                else:
                    new_scope = f"<={latest_bound.strftime('%Y-%m-%d')}"

                new_mem.time_scope = new_scope
                if earliest_bound:
                    new_mem.start_time = earliest_bound.strftime('%Y-%m-%d')
                if latest_bound:
                    new_mem.end_time = latest_bound.strftime('%Y-%m-%d')

                updated.append(new_mem)
                tightened.append((mem_id, old_scope, new_scope))
            else:
                updated.append(mem)

        return updated, tightened

    # ── Medical domain constraints ────────────────────────────────────────────

    def _apply_medical_constraints(
        self,
        a: TemporalNode,
        b: TemporalNode,
        inferred: Rel,
    ) -> Rel:
        """
        Apply medical domain knowledge to constrain temporal relations.

        Medical constraints:
        C1: Diagnosis must precede or coincide with treatment
            (diagnosis BEFORE|MEETS|OVERLAPS|EQUALS medication)
        C2: Symptom onset typically precedes or coincides with diagnosis
        C3: Same vital sign at same time must be EQUALS
        C4: Medication dosage change implies temporal ordering
        C5: Lab results are point-in-time (no CONTAINS)
        """
        constraints = inferred

        # C1: Diagnosis → Treatment ordering
        if (a.attribute == "primary_diagnosis" and b.attribute == "medication"):
            constraints &= (Rel.BEFORE | Rel.MEETS | Rel.OVERLAPS | Rel.EQUALS | Rel.CONTAINS)

        if (a.attribute == "medication" and b.attribute == "primary_diagnosis"):
            constraints &= (Rel.AFTER | Rel.MEETS | Rel.OVERLAPS | Rel.EQUALS | Rel.DURING)

        # C2: Symptom → Diagnosis ordering
        if (a.attribute == "symptom" and b.attribute == "primary_diagnosis"):
            constraints &= (Rel.BEFORE | Rel.MEETS | Rel.OVERLAPS | Rel.EQUALS | Rel.CONTAINS)

        # C3: Same vital sign at same scope → EQUALS
        vitals = {"blood_pressure_sys", "blood_pressure_dia", "heart_rate",
                  "body_temperature", "blood_glucose", "hemoglobin"}
        if (a.attribute == b.attribute and a.attribute in vitals
                and a.has_interval and b.has_interval
                and a.start == b.start and a.end == b.end):
            if a.value != b.value:
                # Same vital, same time, different value → inconsistency candidate
                constraints = Rel.EQUALS  # force detection

        # C5: Lab results are instantaneous
        labs = {"blood_glucose", "hemoglobin"}
        if a.attribute in labs and not a.is_global:
            constraints &= ~Rel.CONTAINS  # labs don't contain other events

        return constraints if constraints != Rel(0) else inferred

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _memory_to_node(self, mem: CanonicalMemory) -> TemporalNode:
        """Convert a CanonicalMemory to a TemporalNode."""
        mem_id = f"{mem.patient_id}_{mem.attribute}_{mem.value}"
        start = None
        end = None
        is_global = mem.time_scope == "global"

        if mem.start_time:
            try:
                start = datetime.strptime(mem.start_time, "%Y-%m-%d")
            except ValueError:
                pass
        if mem.end_time:
            try:
                end = datetime.strptime(mem.end_time, "%Y-%m-%d")
            except ValueError:
                pass

        # If only time_scope is set (day format), use it
        if start is None and mem.time_scope and mem.time_scope != "global":
            try:
                start = datetime.strptime(mem.time_scope, "%Y-%m-%d")
                end = start
            except ValueError:
                pass

        if mem.is_ongoing and start:
            end = datetime(2100, 12, 31)

        return TemporalNode(
            memory_id=mem_id,
            attribute=mem.attribute,
            value=mem.value,
            start=start,
            end=end,
            is_global=is_global,
            patient_id=mem.patient_id,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def run_tcp(
    memories: List[CanonicalMemory],
    max_iterations: int = 50,
) -> Tuple[List[CanonicalMemory], TCPResult]:
    """
    Run Temporal Constraint Propagation on a set of memories.

    Unlike passive detection, TCP actively corrects memories:
    1. DETECT: Find temporal inconsistencies via path consistency
    2. FLAG: Mark inconsistent memories with conflict_flag=True
    3. TIGHTEN: Replace "global" time scopes with inferred ranges
    4. RESOLVE: For same-attribute conflicts at same time, keep higher confidence

    Args:
        memories: List of CanonicalMemory objects from the pipeline.
        max_iterations: Maximum propagation iterations.

    Returns:
        (updated_memories, tcp_result)
    """
    if len(memories) < 2:
        return memories, TCPResult(n_nodes=len(memories))

    # Build constraint graph
    tcg = TemporalConstraintGraph()
    tcg.build_from_memories(memories)

    # Propagate constraints
    result = tcg.propagate(max_iterations=max_iterations)

    # Step 1: Tighten ambiguous scopes
    updated_memories, tightened = tcg.tighten_memory_scopes(memories)
    result.tightened_scopes = tightened
    result.n_tightened = len(tightened)

    # Step 2: Flag inconsistent memories with conflict_flag
    inconsistent_ids = set()
    for mem_id_a, mem_id_b, desc in result.inconsistent_pairs:
        inconsistent_ids.add(mem_id_a)
        inconsistent_ids.add(mem_id_b)

    for mem in updated_memories:
        mem_id = f"{mem.patient_id}_{mem.attribute}_{mem.value}"
        if mem_id in inconsistent_ids:
            mem.conflict_flag = True

    # Step 3: Resolve same-attribute-same-time conflicts
    # Group by (patient_id, attribute, time_scope)
    groups: Dict[tuple, List[int]] = {}
    for i, mem in enumerate(updated_memories):
        key = (mem.patient_id, mem.attribute, mem.time_scope)
        groups.setdefault(key, []).append(i)

    indices_to_remove = set()
    for key, indices in groups.items():
        if len(indices) <= 1:
            continue
        # Multiple memories with same attribute and time → keep highest confidence
        group_mems = [(i, updated_memories[i]) for i in indices]
        group_mems.sort(key=lambda x: x[1].confidence, reverse=True)
        # Keep the best, remove the rest (mark as conflict)
        best_idx, best_mem = group_mems[0]
        for idx, mem in group_mems[1:]:
            if mem.value != best_mem.value:
                # Different values at same time = true conflict
                best_mem.conflict_flag = True
                indices_to_remove.add(idx)
                result.n_inconsistencies += 1

    # Remove resolved duplicates
    if indices_to_remove:
        updated_memories = [m for i, m in enumerate(updated_memories) if i not in indices_to_remove]

    return updated_memories, result
