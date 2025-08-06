"""
Helper utilities for working with ProcessGraph instances.

This module provides a small adapter to translate the experimental
ProcessGraph into a linear sequence of ProvNode objects.  TapeCompiler still
expects a ProvenanceGraph-like interface and this helper bridges the gap.
"""

from __future__ import annotations

from typing import List
import networkx as nx

from ..turing_machine.turing_provenance import ProvNode


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
