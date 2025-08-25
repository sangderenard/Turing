from __future__ import annotations

"""Render training loop diagrams for :mod:`autograd` tensors.

This helper builds a layered graph from :class:`~src.common.tensors.autograd_process.AutogradProcess`
instances and writes the result to a PNG image.  The diagram exposes the
relationship between forward computations, cached values, the loss node and the
backward pass.  Each operation is drawn in a box with its operator label so the
entire optimisation step can be visually inspected.
"""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx

from .autograd_process import AutogradProcess


def build_training_diagram(proc: AutogradProcess) -> nx.DiGraph:
    """Return a graph describing the training loop captured by ``proc``.

    Nodes are arranged into three layers:

    * Forward operations.
    * Intermediate nodes (``loss`` and cache markers).
    * Backward operations.

    Edges follow the data flow of the recorded computation while vertical
    connections indicate cached values and the mapping between forward and
    backward nodes.
    """

    if proc.forward_graph is None or proc.backward_graph is None:
        raise RuntimeError("build() must be called before requesting a diagram")

    g = nx.DiGraph()

    # forward row -----------------------------------------------------------
    for tid, data in proc.forward_graph.nodes(data=True):
        fnode = f"f{tid}"
        g.add_node(fnode, label=data.get("op"), layer=0)
        for src in proc.forward_graph.predecessors(tid):
            g.add_edge(f"f{src}", fnode)
        if data.get("loss"):
            g.add_node("loss", label="loss", layer=1)
            g.add_edge(fnode, "loss")
        if tid in proc.cache:
            cache_node = f"cache_{tid}"
            g.add_node(cache_node, label=f"cache[{tid}]", layer=1)
            g.add_edge(fnode, cache_node)

    # backward row ----------------------------------------------------------
    for tid, data in proc.backward_graph.nodes(data=True):
        bnode = f"b{tid}"
        g.add_node(bnode, label=data.get("op"), layer=2)
        for src in proc.backward_graph.predecessors(tid):
            g.add_edge(f"b{src}", bnode)
        # connect matching forward nodes to their backward counterparts
        if proc.forward_graph.has_node(tid):
            g.add_edge(f"f{tid}", bnode)

    # route loss to the roots of the backward graph
    roots: Iterable[int] = [
        nid
        for nid in proc.backward_graph.nodes
        if proc.backward_graph.in_degree(nid) == 0
    ]
    for root in roots:
        g.add_edge("loss", f"b{root}")

    return g


def render_training_diagram(proc: AutogradProcess, filename: str | Path, *, figsize: tuple[int, int] = (20, 12)) -> None:
    """Render ``proc`` as a PNG file at ``filename``."""

    diagram = build_training_diagram(proc)
    pos = nx.multipartite_layout(diagram, subset_key="layer")
    labels = {n: d.get("label", str(n)) for n, d in diagram.nodes(data=True)}

    plt.figure(figsize=figsize)
    nx.draw(
        diagram,
        pos,
        with_labels=True,
        labels=labels,
        node_shape="s",
        node_size=3000,
        font_size=8,
        arrows=True,
        node_color="#e0e0ff",
    )
    plt.axis("off")
    filename = Path(filename)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
