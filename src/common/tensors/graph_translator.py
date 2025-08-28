from __future__ import annotations
"""Translate simple networkx graphs to ILPScheduler process graphs.

This module provides a minimal adapter that converts a ``networkx`` graph
into the structure expected by :class:`ILPScheduler`.  Nodes may carry an
``op`` attribute representing the callable to execute.  The translator
computes a stable execution order via ILP scheduling and caches it for
subsequent executions.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Type

import networkx as nx

from ...transmogrifier.cycle_unroller import unroll_all_cycles_once
from ...transmogrifier.ilpscheduler import ILPScheduler


@dataclass
class MinimalProcessGraph:
    """Lightweight shim that satisfies ``ILPScheduler(process_graph)``."""

    G: nx.DiGraph
    role_schemas: Dict | None = None


class GraphTranslator:
    """Adapt a ``networkx`` graph for ILP scheduling and execution."""

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph
        self._order: List | None = None
        self._levels: Dict | None = None

    # ------------------------------------------------------------------
    # Graph -> ProcessGraph adapter
    # ------------------------------------------------------------------
    def _to_process_graph(self, graph: nx.DiGraph | None = None) -> MinimalProcessGraph:
        src = self.graph if graph is None else graph
        G = nx.DiGraph()
        for nid, data in src.nodes(data=True):
            # copy all attrs except the executable op
            attrs = {
                k: v
                for k, v in data.items()
                if k not in {"op", "parents", "children", "label"}
            }
            attrs.setdefault("parents", [])
            attrs.setdefault("children", [])
            G.add_node(nid, label=str(nid), **attrs)
        for u, v in src.edges():
            G.add_edge(u, v)
            G.nodes[u]["children"].append((v, "dep"))
            G.nodes[v]["parents"].append((u, "dep"))
        return MinimalProcessGraph(G=G, role_schemas={})

    # ------------------------------------------------------------------
    # Scheduling / execution helpers
    # ------------------------------------------------------------------
    def schedule(self, scheduler_cls: Type[ILPScheduler] = ILPScheduler) -> List:
        """Compute and cache execution order using ``scheduler_cls``."""
        if self._order is None:
            g = self.graph.copy()
            source_map: Dict[Any, Any] = unroll_all_cycles_once(g)
            proc = self._to_process_graph(g)
            sched = scheduler_cls(proc)
            raw_levels = sched.compute_levels("asap", "dependency")
            

            collapsed: Dict[Any, int] = {}
            for vid, lvl in raw_levels.items():
                orig = source_map.get(vid, vid)
                collapsed[orig] = min(collapsed.get(orig, lvl), lvl)

            unique = sorted(set(collapsed.values()))
            remap = {lvl: i for i, lvl in enumerate(unique)}
            self._levels = {nid: remap[lvl] for nid, lvl in collapsed.items()}

            for nid, lvl in self._levels.items():
                if self.graph.has_node(nid):
                    self.graph.nodes[nid]["level"] = lvl

            order_raw = [
                source_map.get(nid, nid)
                for nid, _ in sorted(raw_levels.items(), key=lambda x: x[1])
            ]
            seen: set[Any] = set()
            self._order = []
            for nid in order_raw:
                if nid not in seen:
                    self._order.append(nid)
                    seen.add(nid)
        return self._order

    def levels(self, scheduler_cls: Type[ILPScheduler] = ILPScheduler) -> Dict:
        """Return level assignments, computing them if needed."""
        if self._levels is None:
            self.schedule(scheduler_cls)
        return self._levels  # type: ignore[return-value]

    def execute(self, scheduler_cls: Type[ILPScheduler] = ILPScheduler) -> None:
        """Execute callables attached to nodes in scheduled order."""
        order = self.schedule(scheduler_cls)
        for nid in order:
            op = self.graph.nodes[nid].get("op")
            if callable(op):
                op()


__all__ = ["GraphTranslator", "MinimalProcessGraph"]
