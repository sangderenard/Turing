from __future__ import annotations

"""Render training loop diagrams for :mod:`autograd` tensors.

This helper builds a layered graph from :class:`~src.common.tensors.autograd_process.AutogradProcess`
instances and writes the result to a PNG image.  The diagram exposes the
relationship between forward computations, cached values, the loss node and the
backward pass.  Each operation is drawn in a box with its operator label so the
entire optimisation step can be visually inspected.
"""

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import networkx as nx

from .autograd_process import AutogradProcess


def _format_label(data: Dict) -> str:
    """Return a readable label for a node ``data`` dict."""

    op = data.get("op") or "input"
    lines: List[str] = [str(op)]
    meta = data.get("metadata") or {}
    for k, v in meta.items():
        lines.append(f"{k}={v}")
    if data.get("required"):
        lines.append(f"requires={data['required']}")
    if data.get("param_id") is not None:
        lines.append(f"param={data['param_id']}")
    return "\n".join(lines)


def build_training_diagram(proc: AutogradProcess) -> nx.DiGraph:
    """Return a graph describing the training loop captured by ``proc``."""

    if proc.forward_graph is None or proc.backward_graph is None:
        raise RuntimeError("build() must be called before requesting a diagram")

    g = nx.DiGraph()

    # Determine forward layering using topological generations
    f_levels = list(nx.topological_generations(proc.forward_graph))
    inter_layer = len(f_levels)
    for lvl, nodes in enumerate(f_levels):
        for tid in nodes:
            data = proc.forward_graph.nodes[tid]
            fnode = f"f{tid}"
            g.add_node(fnode, label=_format_label(data), layer=lvl)
            for src in proc.forward_graph.predecessors(tid):
                g.add_edge(f"f{src}", fnode)
            if data.get("loss"):
                g.add_node("loss", label="loss", layer=inter_layer)
                g.add_edge(fnode, "loss")
            if tid in proc.cache:
                cache_node = f"cache_{tid}"
                g.add_node(cache_node, label=f"cache[{tid}]", layer=inter_layer)
                g.add_edge(fnode, cache_node)

    # Build backward layers following the forward/intermediate sections
    b_levels = list(nx.topological_generations(proc.backward_graph))
    b_offset = inter_layer + 1
    for lvl, nodes in enumerate(b_levels, start=b_offset):
        for tid in nodes:
            data = proc.backward_graph.nodes[tid]
            bnode = f"b{tid}"
            g.add_node(bnode, label=_format_label(data), layer=lvl)
            for src in proc.backward_graph.predecessors(tid):
                g.add_edge(f"b{src}", bnode)
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


def _row_layout(diagram: nx.DiGraph) -> Dict[str, tuple[float, float]]:
    """Lay out ``diagram`` with layers on the Y axis."""

    layers: Dict[int, List[str]] = {}
    for node, data in diagram.nodes(data=True):
        layer = int(data.get("layer", 0))
        layers.setdefault(layer, []).append(node)

    pos: Dict[str, tuple[float, float]] = {}
    for layer, nodes in layers.items():
        for i, node in enumerate(nodes):
            pos[node] = (i, -layer)
    return pos


def render_training_diagram(proc: AutogradProcess, filename: str | Path, *, figsize: tuple[int, int] = (20, 12)) -> None:
    """Render ``proc`` as a PNG file at ``filename``."""

    diagram = build_training_diagram(proc)
    pos = _row_layout(diagram)
    labels = {n: d.get("label", str(n)) for n, d in diagram.nodes(data=True)}

    layers = sorted({data.get("layer", 0) for _, data in diagram.nodes(data=True)})
    cmap = plt.get_cmap("Pastel1")
    layer_colors = {layer: cmap(i % cmap.N) for i, layer in enumerate(layers)}
    node_color = [layer_colors[data.get("layer", 0)] for _, data in diagram.nodes(data=True)]

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
        node_color=node_color,
    )
    plt.axis("off")
    filename = Path(filename)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
