"""
Helper utilities for working with ProcessGraph instances.

This module provides a small adapter to translate the experimental
ProcessGraph into a linear sequence of ProvNode objects.  TapeCompiler still
expects a ProvenanceGraph-like interface and this helper bridges the gap.
"""

from __future__ import annotations

from typing import List
import networkx as nx

from ..turing_machine.turing_provenance import ProvNode, ProvEdge, ProvenanceGraph
from ..turing_machine.loop_structure import LoopStructureAnalyzer
from ..transmogrifier.graph.graph_express2 import ProcessGraph


def reduce_cycles_to_mu(pg: ProvenanceGraph) -> None:
    """Rewrite back-edges in ``pg`` into explicit ``mu`` nodes.

    This is a light-weight normalisation pass used by the tests in this kata.
    It walks the natural loops discovered by :class:`LoopStructureAnalyzer` and
    replaces each loop-carried dependency with a dedicated ``mu`` operator. The
    new ``mu`` node selects between the loop's initial value and the value
    produced by the latch. The original back edge is removed so the provenance
    graph presented to :func:`provenance_to_process_graph` is acyclic and can be
    scheduled deterministically.
    """

    analyzer = LoopStructureAnalyzer(pg)
    loops = analyzer.find_loops()
    for info in loops:
        for arg_pos, init_src, back_src in info.loop_vars:
            mu_idx = pg._next_idx()
            mu_out = mu_idx
            # Build the ``mu`` node and append it to the provenance lists.
            mu_node = ProvNode(mu_idx, "mu", (init_src, back_src, back_src), {}, mu_out)
            pg._nodes.append(mu_node)
            pg._edges.append(ProvEdge(init_src, mu_idx, 0))
            pg._edges.append(ProvEdge(back_src, mu_idx, 1))
            pg._edges.append(ProvEdge(back_src, mu_idx, 2))

            # Redirect the header argument to consume the ``mu`` result instead
            # of the back-edge.
            pg._edges = [
                e
                for e in pg._edges
                if not (e.src_idx == back_src and e.dst_idx == info.header and e.arg_pos == arg_pos)
            ]
            pg._edges.append(ProvEdge(mu_idx, info.header, arg_pos))

    if pg.nx is not None:  # keep networkx mirror in sync
        pg.nx.clear()
        for node in pg._nodes:
            pg.nx.add_node(node.idx, op=node.op, args=node.args, kwargs=node.kwargs, out_obj_id=node.out_obj_id)
        for e in pg._edges:
            pg.nx.add_edge(e.src_idx, e.dst_idx, arg_pos=e.arg_pos)


class ProcessGraphLinearizer:
    """Linearise a ProcessGraph into a list of ProvNode records."""

    def __init__(self, pg):
        self.pg = pg

    def linear_nodes(self) -> List[ProvNode]:
        """Return nodes in topological order as ProvNode objects."""
        order = list(nx.topological_sort(self.pg.G))
        nodes: List[ProvNode] = []
        for idx, nid in enumerate(order):
            data = self.pg.G.nodes[nid]
            label = data.get("label")
            expr_obj = data.get("expr_obj")
            if label is None and expr_obj is not None:
                label = type(expr_obj).__name__
            parents = [p for p, _ in data.get("parents", [])]
            nodes.append(ProvNode(idx=idx, op=label, arg_ids=tuple(parents), kwargs={}, out_obj_id=nid))
        return nodes


def provenance_to_process_graph(pg: "ProvenanceGraph"):
    """Return a :class:`ProcessGraph` built from a :class:`ProvenanceGraph`.

    ProcessGraph already understands how to ingest provenance graphs via
    ``ProcessGraph.build_from_expression``.  This helper simply wraps that
    behaviour to keep the compiler pipeline explicit: visualisation and
    scheduling live in the ProcessGraph world while provenance captures the
    raw data‑flow.  Keeping this bridge means any recorded provenance can
    graduate to those richer utilities without forcing callers to know
    ProcessGraph internals.
    """

    proc = ProcessGraph()
    # ProcessGraph knows how to interpret a ProvenanceGraph and will translate
    # each node/edge into its own graph form.  We funnel all conversions through
    # this single entry point to avoid duplicate logic elsewhere.
    proc.build_from_expression(pg)
    return proc
