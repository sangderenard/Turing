from __future__ import annotations
"""Translate simple networkx graphs to ILPScheduler process graphs.

This module provides a minimal adapter that converts a ``networkx`` graph
into the structure expected by :class:`ILPScheduler`.  Nodes may carry an
``op`` attribute representing the callable to execute.  The translator
computes a stable execution order via ILP scheduling and caches it for
subsequent executions.
"""

from dataclasses import dataclass
from typing import Dict, List, Type

import networkx as nx

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
    def _to_process_graph(self) -> MinimalProcessGraph:
        G = nx.DiGraph()
        for nid, data in self.graph.nodes(data=True):
            # copy all attrs except the executable op
            attrs = {k: v for k, v in data.items() if k != "op"}
            G.add_node(nid, label=str(nid), parents=[], children=[], **attrs)
        for u, v in self.graph.edges():
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
            proc = self._to_process_graph()
            sched = scheduler_cls(proc)
            self._levels = sched.compute_levels("asap", "dependency")
            for nid, lvl in self._levels.items():
                if self.graph.has_node(nid):
                    self.graph.nodes[nid]["level"] = lvl
            # Order nodes by level (stable for repeated runs)
            self._order = [nid for nid, _ in sorted(self._levels.items(), key=lambda x: x[1])]
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
