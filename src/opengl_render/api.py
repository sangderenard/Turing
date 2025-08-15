"""Adapters for building renderer layers from numpy-based simulation data.

These helpers convert raw ``numpy`` arrays from an external simulation into
:class:`MeshLayer`, :class:`LineLayer` and :class:`PointLayer` instances used by
:class:`~opengl_render.renderer.GLRenderer`.

The helpers keep imports light so they can be used from headless tests. When
``rainbow`` is enabled a simple HSV→RGB mapping is applied to generate a color
spectrum for each vertex; this mirrors the "rainbow history" effect available in
older demos under ``inspiration/``.
"""
from __future__ import annotations

from typing import Mapping, Iterable, Callable
import colorsys
import numpy as np

try:  # pragma: no cover - tolerate missing OpenGL libs
    from .renderer import MeshLayer, LineLayer, PointLayer, GLRenderer
except Exception:  # noqa: BLE001
    MeshLayer = LineLayer = PointLayer = GLRenderer = object  # type: ignore


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def rainbow_colors(count: int, *, alpha: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Return ``count`` RGBA colors forming a rainbow gradient."""
    hues = (np.linspace(0.0, 1.0, count, endpoint=False) + offset) % 1.0
    rgb = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues], dtype=np.float32)
    a = np.full((count, 1), float(alpha), np.float32)
    return np.concatenate([rgb, a], axis=1)


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def rainbow_history_points(history: Iterable[np.ndarray]) -> PointLayer:
    """Pack a sequence of past positions into a rainbow-tinted point layer.

    This mirrors the ghost-trail logic from ``inspiration/particles.py`` where
    each older frame is coloured with a fading HSV tone.
    """
    hist = [np.asarray(h, dtype=np.float32) for h in history if np.asarray(h).size]
    if not hist:
        return PointLayer(positions=np.zeros((0, 3), np.float32))

    trail = [colorsys.hsv_to_rgb(h, 1.0, 1.0)
             for h in np.linspace(0, 1, len(hist), endpoint=False)]
    pos_acc: list[np.ndarray] = []
    col_acc: list[np.ndarray] = []
    n = len(hist)
    for i, (pos, (r, g, b)) in enumerate(zip(hist, trail)):
        alpha = 1.0 - (i / n)
        col = np.tile((r, g, b, alpha * 0.6), (pos.shape[0], 1))
        pos_acc.append(pos)
        col_acc.append(col)
    return PointLayer(
        positions=np.concatenate(pos_acc, axis=0),
        colors=np.concatenate(col_acc, axis=0),
    )


# ---------------------------------------------------------------------------
# Packing helpers
# ---------------------------------------------------------------------------

def pack_mesh(positions: np.ndarray,
              faces: np.ndarray,
              *,
              colors: np.ndarray | None = None,
              rainbow: bool = False) -> MeshLayer:
    """Create a :class:`MeshLayer` from vertex positions and triangle faces."""
    pos = np.asarray(positions, dtype=np.float32)
    idx = np.asarray(faces, dtype=np.uint32)
    col = colors
    if col is None and rainbow:
        col = rainbow_colors(pos.shape[0])
    return MeshLayer(positions=pos, indices=idx, colors=col)


def pack_lines(edges: np.ndarray,
               positions: np.ndarray,
               *,
               colors: np.ndarray | None = None,
               rainbow: bool = False,
               width: float = 2.0) -> LineLayer:
    """Create a :class:`LineLayer` from edge indices and vertex positions."""
    pos = np.asarray(positions, dtype=np.float32)
    e = np.asarray(edges, dtype=np.int32)
    pts = pos[e.ravel()]
    col = None
    if colors is not None:
        col = np.repeat(np.asarray(colors, dtype=np.float32), 2, axis=0)
    elif rainbow:
        col = np.repeat(rainbow_colors(e.shape[0]), 2, axis=0)
    return LineLayer(positions=pts, colors=col, width=width)


def pack_points(positions: np.ndarray,
                *,
                colors: np.ndarray | None = None,
                rainbow: bool = False,
                sizes: np.ndarray | None = None,
                default_size: float = 6.0) -> PointLayer:
    """Create a :class:`PointLayer` from raw point positions."""
    pos = np.asarray(positions, dtype=np.float32)
    col = colors
    if col is None and rainbow:
        col = rainbow_colors(pos.shape[0])
    return PointLayer(positions=pos, colors=col, sizes_px=sizes, size_px_default=default_size)


# ---------------------------------------------------------------------------
# Cellsim adapters
# ---------------------------------------------------------------------------

def cellsim_layers(h, *, rainbow: bool = False) -> Mapping[str, MeshLayer | PointLayer]:
    """Pack membrane meshes (and optionally other categories) from a cellsim ``h``.

    Parameters
    ----------
    h:
        Cellsim handle exposing ``cells`` with ``X`` (vertices) and ``F`` (faces).
    rainbow:
        If ``True``, apply a rainbow gradient to the membrane vertices.

    Returns
    -------
    Mapping[str, MeshLayer | PointLayer]
        Currently only contains ``{"membrane": MeshLayer}``. Future categories
        such as inner/outer fluids can be added without changing the caller API.
    """
    positions = []
    faces = []
    offset = 0
    for cell in getattr(h, "cells", []):
        X = np.asarray(getattr(cell, "X", np.zeros((0, 3))), dtype=np.float32)
        F = np.asarray(getattr(cell, "F", np.zeros((0, 3), dtype=np.uint32)), dtype=np.uint32)
        if X.size and F.size:
            positions.append(X)
            faces.append(F + offset)
            offset += X.shape[0]
    layers: dict[str, MeshLayer | PointLayer] = {}
    if positions and faces:
        layers["membrane"] = pack_mesh(np.concatenate(positions), np.concatenate(faces), rainbow=rainbow)
    return layers


def fluid_layers(engine, *, rainbow: bool = False) -> Mapping[str, MeshLayer | PointLayer]:
    """Pack simple fluid visuals from a fluid engine.

    Currently supports discrete particle fluids by exposing their particle
    positions as a ``PointLayer``.
    """
    layers: dict[str, MeshLayer | PointLayer] = {}
    pts = getattr(engine, "p", None)
    if pts is not None:
        layers["fluid"] = pack_points(np.asarray(pts, dtype=np.float32), rainbow=rainbow, default_size=2.0)
    return layers


# ---------------------------------------------------------------------------
# Rendering hooks
# ---------------------------------------------------------------------------

def draw_layers(
    renderer: GLRenderer,
    layers: Mapping[str, MeshLayer | LineLayer | PointLayer],
    viewport: tuple[int, int] | None = None,
) -> None:
    """Upload ``layers`` to ``renderer`` and draw a frame.

    When ``renderer`` exposes a ``print_layers`` attribute (see
    :class:`~opengl_render.renderer.DebugRenderer`), the layer mapping is passed
    directly to that method, bypassing all OpenGL work.
    """
    if hasattr(renderer, "print_layers"):
        renderer.print_layers(layers)  # type: ignore[call-arg]
        return
    if viewport is None:
        viewport = getattr(renderer, "_window_size", (640, 480))
    mesh = layers.get("membrane")
    if isinstance(mesh, MeshLayer):
        renderer.set_mesh(mesh)
    lines = layers.get("lines")
    if isinstance(lines, LineLayer):
        renderer.set_lines(lines)
    # Prefer fluid points if available, fall back to generic points.
    # Optionally merge in a ghost trail (``ghost`` layer).
    pts = layers.get("fluid") or layers.get("points")
    ghost = layers.get("ghost")
    if isinstance(pts, PointLayer) and isinstance(ghost, PointLayer):
        pos = np.concatenate([pts.positions, ghost.positions], axis=0)
        col = None
        if pts.colors is not None or ghost.colors is not None:
            col_acc: list[np.ndarray] = []
            if pts.colors is not None:
                col_acc.append(pts.colors)
            else:
                col_acc.append(np.tile(np.array([[1, 1, 1, 1]], np.float32), (pts.positions.shape[0], 1)))
            if ghost.colors is not None:
                col_acc.append(ghost.colors)
            else:
                col_acc.append(np.tile(np.array([[1, 1, 1, 1]], np.float32), (ghost.positions.shape[0], 1)))
            col = np.concatenate(col_acc, axis=0)
        merged = PointLayer(positions=pos, colors=col, size_px_default=pts.size_px_default, alpha=pts.alpha)
        renderer.set_points(merged)
    elif isinstance(pts, PointLayer):
        renderer.set_points(pts)
    elif isinstance(ghost, PointLayer):
        renderer.set_points(ghost)
    renderer.draw(viewport)



def make_draw_hook(
    renderer: GLRenderer,
    viewport: tuple[int, int] | None = None,
    *,
    history: int = 32,
    loop: bool = False,
    bounce: bool = False,
    ghost_trail: bool = True,
):
    """Return a draw hook.

    When ``renderer`` exposes ``print_layers`` (debug mode) a synchronous hook is
    returned so that output is immediate.  Otherwise a thread-backed hook is
    created via :func:`make_threaded_draw_hook`.
    """

    if hasattr(renderer, "print_layers"):
        from collections import deque
        maxlen = history if history > 0 else None
        hist: deque[Mapping[str, object]] = deque(maxlen=maxlen)

        def hook(layers: Mapping[str, MeshLayer | LineLayer | PointLayer]) -> None:
            hist.append(layers)
            frame = layers
            if ghost_trail:
                pts_hist = []
                for past in list(hist)[:-1]:
                    pts = past.get("fluid") or past.get("points")
                    if isinstance(pts, PointLayer):
                        pts_hist.append(pts.positions)
                if pts_hist:
                    frame = dict(layers)
                    frame["ghost"] = rainbow_history_points(pts_hist)
            draw_layers(renderer, frame, viewport)

        return hook

    hook, thread = make_threaded_draw_hook(
        renderer,
        viewport,
        history=history,
        loop=loop,
        bounce=bounce,
        ghost_trail=ghost_trail,
    )
    # Store it on the renderer for lifecycle control
    try:
        renderer._render_thread = thread
    except Exception:
        pass

    return hook

def make_threaded_draw_hook(
    renderer: GLRenderer,
    viewport: tuple[int, int] | None = None,
    *,
    history: int = 32,
    loop: bool = False,
    bounce: bool = False,
    ghost_trail: bool = True,
):
    """Return a thread-backed draw hook and its controller.

    The returned tuple ``(hook, thread)`` provides a callable ``hook`` that
    enqueues layer mappings to be drawn on a dedicated thread.  ``thread`` is the
    :class:`~opengl_render.renderer.GLRenderThread` instance managing the queue
    and history.
    """

    from .threaded import GLRenderThread

    thread = GLRenderThread(
        renderer,
        viewport=viewport,
        history=history,
        loop=loop,
        bounce=bounce,
        ghost_trail=ghost_trail,
    )
    return thread.get_submit_hook(), thread

