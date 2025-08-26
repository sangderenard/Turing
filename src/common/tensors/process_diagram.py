from __future__ import annotations

"""Build network diagrams for :mod:`autograd` tensors.

This helper constructs a layered :class:`networkx.DiGraph` from
:class:`~src.common.tensors.autograd_process.AutogradProcess` instances. The
diagram exposes the relationship between forward computations, cached values,
the loss node and the backward pass so the entire optimisation step can be
visually inspected or further analysed. Image generation is intentionally
omitted to keep the module light-weight.
"""

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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

    # Determine forward layering from explicit annotations.  All nodes are
    # expected to carry a ``layer`` value populated by the scheduler.
    if any("layer" not in data for _, data in proc.forward_graph.nodes(data=True)):
        raise RuntimeError("forward graph missing layer annotations")
    layer_map: Dict[int, List[int]] = {}
    for tid, data in proc.forward_graph.nodes(data=True):
        layer_map.setdefault(int(data["layer"]), []).append(tid)
    f_levels = [layer_map[idx] for idx in sorted(layer_map)]
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

    # Build backward layers following the forward/intermediate sections using
    # the scheduler-provided annotations.
    if any("layer" not in data for _, data in proc.backward_graph.nodes(data=True)):
        raise RuntimeError("backward graph missing layer annotations")
    b_map: Dict[int, List[int]] = {}
    for tid, data in proc.backward_graph.nodes(data=True):
        b_map.setdefault(int(data["layer"]), []).append(tid)
    b_levels = [b_map[idx] for idx in sorted(b_map)]
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


def _layered_grid_layout(
    g: nx.DiGraph,
    *,
    max_nodes_per_row: int = 8,
    layer_gap: float = 3.0,
    col_gap: float = 1.5,
    row_gap: float = 1.5,
    jitter: float = 0.3,
) -> Dict[str, tuple[float, float]]:
    """Return coordinates for ``g`` arranging each ``layer`` in a grid.

    ``nx.multipartite_layout`` tends to stack all nodes for a layer along a
    single line which can make dense graphs unreadable. This helper spreads
    nodes within the same layer horizontally and wraps to new rows once the
    number of nodes exceeds ``max_nodes_per_row``.

    Parameters
    ----------
    g:
        Graph with ``layer`` metadata on each node.
    max_nodes_per_row:
        Number of nodes to place in a single row for a layer before
        wrapping to a new row.
    layer_gap:
        Horizontal distance between successive layers.
    col_gap:
        Additional horizontal offset applied when wrapping to a new column
        inside a layer.
    row_gap:
        Vertical distance between nodes within the same column.
    """

    layers: Dict[int, List[str]] = {}
    for node, data in g.nodes(data=True):
        layer = int(data.get("layer", 0))
        layers.setdefault(layer, []).append(node)

    pos: Dict[str, tuple[float, float]] = {}
    sorted_layers = sorted(layers)
    prev_rows = 0
    y_base = 0.0
    rng = np.random.default_rng(0)
    for i, layer in enumerate(sorted_layers):
        nodes = layers[layer]
        cols = min(max_nodes_per_row, len(nodes))
        rows = (len(nodes) + max_nodes_per_row - 1) // max_nodes_per_row

        if i > 0:
            y_base -= layer_gap + prev_rows * row_gap

        x_offset = -((cols - 1) * col_gap) / 2
        for idx, node in enumerate(nodes):
            row = idx // max_nodes_per_row
            col = idx % max_nodes_per_row
            x = x_offset + col * col_gap
            y = y_base - row * row_gap
            if jitter:
                y += rng.uniform(-jitter, jitter)
            pos[node] = (x, y)

        prev_rows = rows

    return pos


def render_training_diagram(
    proc: AutogradProcess,
    filename: str | Path | None = None,
    *,
    figsize: tuple[int, int] | None = None,
) -> nx.DiGraph:
    """Return a combined process diagram for ``proc``.

    The diagram is drawn using :mod:`matplotlib`. If ``filename`` is provided
    the image is written to that path. Otherwise a window will be shown so the
    caller can visually inspect the graph.

    Parameters
    ----------
    proc:
        The :class:`AutogradProcess` whose computation will be represented.
    filename:
        Optional output location for a PNG snapshot of the diagram. When
        omitted a GUI window is opened instead.
    figsize:
        Optional figure size passed through to
        :func:`matplotlib.pyplot.figure`. When omitted the size is determined
        from the graph layout so nodes do not crowd one another.
    """

    g = build_training_diagram(proc)

    pos = _layered_grid_layout(g)
    if figsize is None:
        xs = [x for x, _ in pos.values()]
        ys = [y for _, y in pos.values()]
        width = (max(xs) - min(xs) if xs else 1) + 2
        height = (max(ys) - min(ys) if ys else 1) + 2
        figsize = (width, height)
    plt.figure(figsize=figsize)
    # Colour nodes by their layer to make execution order visually obvious.
    layers = [data.get("layer", 0) for _, data in g.nodes(data=True)]
    node_artists = nx.draw_networkx_nodes(g, pos, node_color=layers, cmap=plt.cm.viridis)
    node_artists.set_zorder(1)
    edge_artists = nx.draw_networkx_edges(g, pos)
    for artist in edge_artists:
        artist.set_zorder(2)
    label_artists = nx.draw_networkx_labels(g, pos)
    for artist in label_artists.values():
        artist.set_zorder(3)

    if filename is not None:
        plt.savefig(Path(filename))
        plt.close()
    else:
        plt.show()

    return g
