"""
Adapter: dt RoundNode tree -> minimal NetworkX ProcessGraph for ILPScheduler.

This builds a tiny structure with the same surface the ILPScheduler expects:
- .G: nx.DiGraph with nodes carrying 'parents' and 'children' lists
- .role_schemas: optional dict (unused here)

Only AdvanceNode leaves are materialized as process nodes. RoundNode schedule
is encoded as dependency edges between the boundary leaves of child subtrees:
- sequential/interleave: linearizes children left-to-right
- parallel: leaves independent across children

Nested RoundNodes are handled recursively. This provides enough structure for
ILPScheduler to compute ASAP/ALAP levels, lifespans and an interference graph
that you can use as metadata for a higher-level simulation graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .dt_graph import AdvanceNode, RoundNode
from ...transmogrifier.ilpscheduler import ILPScheduler


@dataclass
class MinimalProcessGraph:
    """Small shim to satisfy ILPScheduler(process_graph) interface."""

    G: nx.DiGraph
    role_schemas: Dict = None  # unused by ILPScheduler for our adapter


class DtToProcessAdapter:
    """Build a minimal process graph from a RoundNode tree.

    Nodes represent AdvanceNode leaves. Edges represent schedule ordering.
    """

    def __init__(self) -> None:
        self.G: nx.DiGraph = nx.DiGraph()
        # map dt AdvanceNode object id -> process node id (we just use id itself)
        self._seen: set[int] = set()

    def build(self, root: RoundNode) -> MinimalProcessGraph:
        """Return a MinimalProcessGraph for the given dt RoundNode tree."""
        self._emit_round(root)
        return MinimalProcessGraph(G=self.G, role_schemas={})

    # ----------------------------- internals
    def _ensure_leaf(self, a: AdvanceNode) -> int:
        nid = id(a)
        if nid in self._seen:
            return nid
        self._seen.add(nid)
        label = getattr(a, "label", None) or "advance"
        self.G.add_node(
            nid,
            label=label,
            type="Advance",
            expr_obj=a.advance,  # store callable for reference only
            parents=[],
            children=[],
        )
        return nid

    def _link(self, u: int, v: int, role: str = "seq") -> None:
        if not self.G.has_edge(u, v):
            self.G.add_edge(u, v)
        # maintain parents/children lists like ProcessGraph
        if v not in [t for t, _ in self.G.nodes[u]["children"]]:
            self.G.nodes[u]["children"].append((v, role))
        if u not in [p for p, _ in self.G.nodes[v]["parents"]]:
            self.G.nodes[v]["parents"].append((u, role))

    def _emit_round(self, rnd: RoundNode) -> List[int]:
        """Emit nodes/edges for a round; return ordered leaf ids under this round."""
        leaves_by_child: List[List[int]] = []
        for child in rnd.children:
            if isinstance(child, AdvanceNode):
                leaves_by_child.append([self._ensure_leaf(child)])
            elif isinstance(child, RoundNode):
                leaves_by_child.append(self._emit_round(child))
            else:
                # Unknown child type; skip gracefully
                continue

        sched = (rnd.schedule or "sequential").lower()

        # Order across children according to schedule semantics
        if sched in ("sequential", "interleave"):
            # Connect boundary leaves between consecutive child groups
            flat: List[int] = []
            prev_tail: Optional[int] = None
            for group in leaves_by_child:
                if not group:
                    continue
                # order inside group already emitted (nested rounds)
                head = group[0]
                tail = group[-1]
                if prev_tail is not None:
                    self._link(prev_tail, head, role="seq")
                prev_tail = tail
                flat.extend(group)
            return flat
        else:  # parallel or unknown → independent groups, just flatten results
            return [nid for group in leaves_by_child for nid in group]


def schedule_dt_round(
    round_root: RoundNode,
    *,
    method: str = "asap",
    order: str = "dependency",
    interference_mode: str = "asap-maxslack",
) -> Tuple[Dict[int, int], nx.Graph, Dict[int, Tuple[int, int]], nx.DiGraph]:
    """Build a process graph from dt and run ILPScheduler.

    Returns: (levels, interference_graph, lifespans, process_graph_nx)
    """
    adapter = DtToProcessAdapter()
    proc = adapter.build(round_root)
    sched = ILPScheduler(proc)
    levels = sched.compute_levels(method, order)
    igraph, lifespans = sched.compute_asap_maxslack_interference(interference_mode)
    return levels, igraph, lifespans, proc.G


__all__ = [
    "MinimalProcessGraph",
    "DtToProcessAdapter",
    "schedule_dt_round",
]
