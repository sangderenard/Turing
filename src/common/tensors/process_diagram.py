from __future__ import annotations

"""Build network diagrams for :mod:`autograd` tensors.

This helper constructs an execution‑levelled :class:`networkx.DiGraph` from
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
from matplotlib.patches import Rectangle

from .autograd_process import AutogradProcess


def _format_label(data: Dict) -> str:
    """Return a readable label for a node ``data`` dict."""

    op = data.get("op") or "input"
    lines: List[str] = [f"Operation: {op}"]
    meta = data.get("metadata") or {}
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    if data.get("required"):
        lines.append(f"Requires Grad: {data['required']}")
    if data.get("param_id") is not None:
        lines.append(f"Parameter ID: {data['param_id']}")
    return "\n".join(lines)


def build_training_diagram(proc: AutogradProcess) -> nx.DiGraph:
    """Return a graph describing the training loop captured by ``proc``."""

    if proc.forward_graph is None or proc.backward_graph is None:
        raise RuntimeError("build() must be called before requesting a diagram")

    g = nx.DiGraph()

    # Determine forward levels from explicit annotations. All nodes are
    # expected to carry a ``level`` value populated by the scheduler.
    if any("level" not in data for _, data in proc.forward_graph.nodes(data=True)):
        raise RuntimeError("forward graph missing level annotations")
    level_map: Dict[int, List[int]] = {}
    for tid, data in proc.forward_graph.nodes(data=True):
        level_map.setdefault(int(data["level"]), []).append(tid)
    max_f_level = max(level_map)
    loss_level = getattr(proc, "loss_level", None)
    if loss_level is None:
        loss_level = max_f_level + 1
    for lvl in sorted(level_map):
        nodes = level_map[lvl]
        for tid in nodes:
            data = proc.forward_graph.nodes[tid]
            fnode = f"f{tid}"
            g.add_node(fnode, label=_format_label(data), level=lvl)
            for src in proc.forward_graph.predecessors(tid):
                g.add_edge(f"f{src}", fnode)
            if data.get("loss"):
                g.add_node("loss", label="loss", level=loss_level)
                g.add_edge(fnode, "loss")
            if tid in proc.cache:
                cache_node = f"cache_{tid}"
                cache_level = proc.cache_levels.get(tid, max_f_level + 1)
                g.add_node(cache_node, label=f"cache[{tid}]", level=cache_level)
                g.add_edge(fnode, cache_node)

    # Build backward levels following the forward/intermediate sections using
    # the scheduler-provided annotations.
    if any("level" not in data for _, data in proc.backward_graph.nodes(data=True)):
        raise RuntimeError("backward graph missing level annotations")
    for tid, data in proc.backward_graph.nodes(data=True):
        lvl = int(data["level"])
        bnode = f"b{tid}"
        g.add_node(bnode, label=_format_label(data), level=lvl)
        for src in proc.backward_graph.predecessors(tid):
            g.add_edge(f"b{src}", bnode)
        if proc.forward_graph.has_node(tid):
            g.add_edge(f"f{tid}", bnode)
        if tid in proc.cache:
            g.add_edge(f"cache_{tid}", bnode)

    # route loss to the roots of the backward graph
    roots: Iterable[int] = [
        nid
        for nid in proc.backward_graph.nodes
        if proc.backward_graph.in_degree(nid) == 0
    ]
    for root in roots:
        g.add_edge("loss", f"b{root}")

    return g


def _leveled_grid_layout(
    g: nx.DiGraph,
    *,
    max_nodes_per_row: int | None = None,
    level_gap: float = 3.0,
    col_gap: float = 1.5,
    row_gap: float = 1.5,
    jitter: float = 0.3,
) -> Dict[str, tuple[float, float]]:
    """Return coordinates for ``g`` arranging each ``level`` in a grid.

    ``nx.multipartite_layout`` tends to stack all nodes for a level along a
    single line which can make dense graphs unreadable. This helper spreads
    nodes within the same level horizontally and wraps to new rows. If
    ``max_nodes_per_row`` is ``None`` the function chooses a square-ish grid
    for each level so the final image is closer to a balanced aspect ratio.

    Parameters
    ----------
    g:
        Graph with ``level`` metadata on each node.
    max_nodes_per_row:
        Optional override for the number of nodes to place in a single row
        for a level before wrapping. When ``None`` a heuristic based on the
        square‑root of the level size is used.
    level_gap:
        Horizontal distance between successive levels.
    col_gap:
        Additional horizontal offset applied when wrapping to a new column
        inside a level.
    row_gap:
        Vertical distance between nodes within the same column.
    """

    levels: Dict[int, List[str]] = {}
    for node, data in g.nodes(data=True):
        level = int(data.get("level", 0))
        levels.setdefault(level, []).append(node)

    pos: Dict[str, tuple[float, float]] = {}
    sorted_levels = sorted(levels)
    prev_rows = 0
    y_base = 0.0
    rng = np.random.default_rng(0)
    for i, level in enumerate(sorted_levels):
        nodes = levels[level]
        if max_nodes_per_row is None:
            cols = int(np.ceil(np.sqrt(len(nodes)))) or 1
        else:
            cols = min(max_nodes_per_row, len(nodes))
        rows = int(np.ceil(len(nodes) / cols))

        if i > 0:
            y_base -= level_gap + prev_rows * row_gap

        x_offset = -((cols - 1) * col_gap) / 2
        for idx, node in enumerate(nodes):
            row = idx // cols
            col = idx % cols
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
    format: str | None = None,
    figsize: tuple[int, int] | None = None,
    dpi: float | None = None,
    node_spacing: float = 1.5,
) -> nx.DiGraph:
    """Return a combined process diagram for ``proc``.

    The diagram is drawn using :mod:`matplotlib`. If ``filename`` is provided
    the image is written to that path. Otherwise a window will be shown so the
    caller can visually inspect the graph. The output format is inferred from
    the filename extension but can be explicitly set via ``format`` to force
    vector formats like SVG or PDF.

    Parameters
    ----------
    proc:
        The :class:`AutogradProcess` whose computation will be represented.
    filename:
        Optional output location for a snapshot of the diagram. When omitted a
        GUI window is opened instead.
    format:
        Optional file format passed to :func:`matplotlib.pyplot.savefig`.
        When ``None`` the format is inferred from ``filename``. Use ``"svg"``
        or ``"pdf"`` for vector output.

    dpi:
        Dots-per-inch used when rasterising formats such as PNG. If ``None``
        the value from :rc:`savefig.dpi` is used and may be automatically
        reduced to stay within backend limits.

    figsize:
        Optional figure size passed through to
        :func:`matplotlib.pyplot.figure`. When omitted the size is determined
        from the graph layout so nodes do not crowd one another.
    node_spacing:
        Distance between nodes within the same level. Larger values space the
        graph out to give edges more room.
    """

    g = build_training_diagram(proc)
    pos = _leveled_grid_layout(
        g,
        col_gap=node_spacing,
        row_gap=node_spacing,
        level_gap=node_spacing * 2,
        jitter=node_spacing / 5,
    )
    if figsize is None:
        xs = [x for x, _ in pos.values()]
        ys = [y for _, y in pos.values()]
        width = (max(xs) - min(xs) if xs else 1) + 2
        height = (max(ys) - min(ys) if ys else 1) + 2
        figsize = (width, height)
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.axis("off")
    cmap = plt.cm.viridis
    levels = [data.get("level", 0) for _, data in g.nodes(data=True)]
    min_lvl, max_lvl = min(levels), max(levels)

    # Lightly shade the background behind each level.
    for lvl in sorted({data.get("level", 0) for _, data in g.nodes(data=True)}):
        nodes = [n for n, d in g.nodes(data=True) if d.get("level", 0) == lvl]
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        if not xs or not ys:
            continue
        pad = node_spacing * 0.6
        rect = Rectangle(
            (min(xs) - pad, min(ys) - pad),
            (max(xs) - min(xs)) + 2 * pad,
            (max(ys) - min(ys)) + 2 * pad,
            color=cmap((lvl - min_lvl) / (max_lvl - min_lvl or 1)),
            alpha=0.05,
            zorder=0,
        )
        ax.add_patch(rect)

    # Colour nodes by their level to make execution order visually obvious.
    node_artists = nx.draw_networkx_nodes(g, pos, node_color=levels, cmap=cmap)
    node_artists.set_zorder(1)
    edge_levels = [g.nodes[u].get("level", 0) for u, _ in g.edges()]
    edge_artists = nx.draw_networkx_edges(
        g,
        pos,
        edge_color=edge_levels,
        edge_cmap=cmap,
        edge_vmin=min_lvl,
        edge_vmax=max_lvl,
        width=0.8,
    )
    for artist in edge_artists:
        artist.set_zorder(2)
    label_artists = nx.draw_networkx_labels(g, pos)
    for artist in label_artists.values():
        artist.set_zorder(3)

    if filename is not None:

        save_kwargs: dict[str, object] = {"format": format} if format else {}
        # Matplotlib's Agg backend limits each dimension to < 2**16 pixels.
        # When saving PNGs we may need to lower the DPI so the rasterised
        # image stays within that constraint.
        if dpi is None:
            dpi = plt.rcParams.get("savefig.dpi", 100)
            if isinstance(dpi, str):
                dpi = plt.rcParams.get("figure.dpi", 100)
        save_kwargs["dpi"] = dpi
        fmt = format or Path(filename).suffix.lstrip(".").lower()
        if fmt == "png" and figsize is not None:
            max_pixels = 2**16 - 1
            width, height = figsize
            max_inches = max(width, height)
            if max_inches * dpi > max_pixels:
                dpi = max_pixels / max_inches
                save_kwargs["dpi"] = dpi

        plt.savefig(Path(filename), **save_kwargs)
        plt.close()
    else:
        plt.show()

    return g
