"""Adapters for building renderer layers from numpy-based simulation data.

These helpers convert raw ``numpy`` arrays from an external simulation into
:class:`MeshLayer`, :class:`LineLayer` and :class:`PointLayer` instances used by
:class:`~opengl_render.renderer.GLRenderer`.

Design notes
------------
- Headless friendly: imports of GL types are guarded so tests don't require OpenGL.
- Thread-safety: use a *renderer factory* (class or zero-arg callable) so the
  OpenGL context is created *inside* the render thread.
- Rainbow utilities: simple HSV→RGB mapping for history/“ghost trail”.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Type, Union, TYPE_CHECKING
import colorsys
import numpy as np

# ---------------------------------------------------------------------------
# Optional OpenGL types (safe under headless tests)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - tolerate missing OpenGL libs
    from .renderer import MeshLayer, LineLayer, PointLayer, GLRenderer
except Exception:  # noqa: BLE001
    MeshLayer = LineLayer = PointLayer = GLRenderer = object  # type: ignore[misc,assignment]

if TYPE_CHECKING:  # for type hints only
    from .renderer import GLRenderer as _GLRenderer_T

RendererFactory = Union[Type["_GLRenderer_T"], Callable[[], "_GLRenderer_T"]]

# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

def rainbow_colors(count: int, *, alpha: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Return ``count`` RGBA colors forming a rainbow gradient."""
    if count <= 0:
        return np.zeros((0, 4), dtype=np.float32)
    hues = (np.linspace(0.0, 1.0, count, endpoint=False) + offset) % 1.0
    rgb = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues], dtype=np.float32)
    a = np.full((count, 1), float(alpha), dtype=np.float32)
    return np.concatenate([rgb, a], axis=1)


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------

def rainbow_history_points(history: Iterable[np.ndarray]) -> PointLayer:
    """Pack a sequence of past positions into a rainbow-tinted point layer.

    Each older frame is coloured with a fading HSV tone (ghost trail).
    """
    frames = []
    for h in history:
        arr = np.asarray(h, dtype=np.float32)
        if arr.size:
            # Ensure (N, 3)
            frames.append(arr.reshape(-1, 3))
    if not frames:
        return PointLayer(positions=np.zeros((0, 3), np.float32))

    n = len(frames)
    hues = np.linspace(0.0, 1.0, n, endpoint=False)
    rgb = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues], dtype=np.float32)

    pos_acc: list[np.ndarray] = []
    col_acc: list[np.ndarray] = []
    for i, (pos, (r, g, b)) in enumerate(zip(frames, rgb)):
        alpha = 1.0 - (i / n)
        col = np.tile(np.array([r, g, b, alpha * 0.6], dtype=np.float32), (pos.shape[0], 1))
        pos_acc.append(pos)
        col_acc.append(col)

    return PointLayer(
        positions=np.concatenate(pos_acc, axis=0),
        colors=np.concatenate(col_acc, axis=0),
    )


# ---------------------------------------------------------------------------
# Packing helpers
# ---------------------------------------------------------------------------

def pack_mesh(
    positions: np.ndarray,
    faces: np.ndarray,
    *,
    colors: Optional[np.ndarray] = None,
    rainbow: bool = False,
) -> MeshLayer:
    """Create a :class:`MeshLayer` from vertex positions and triangle faces."""
    pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    idx = np.asarray(faces, dtype=np.uint32).reshape(-1, 3)
    col = colors
    if col is None and rainbow:
        col = rainbow_colors(pos.shape[0])
    return MeshLayer(positions=pos, indices=idx, colors=col)


def pack_lines(
    edges: np.ndarray,
    positions: np.ndarray,
    *,
    colors: Optional[np.ndarray] = None,
    rainbow: bool = False,
    width: float = 2.0,
) -> LineLayer:
    """Create a :class:`LineLayer` from edge indices and vertex positions."""
    pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    e = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    pts = pos[e.ravel()]
    col = None
    if colors is not None:
        col = np.repeat(np.asarray(colors, dtype=np.float32).reshape(-1, 4), 2, axis=0)
    elif rainbow:
        col = np.repeat(rainbow_colors(e.shape[0]), 2, axis=0)
    return LineLayer(positions=pts, colors=col, width=width)


def pack_points(
    positions: np.ndarray,
    *,
    colors: Optional[np.ndarray] = None,
    rainbow: bool = False,
    sizes: Optional[np.ndarray] = None,
    default_size: float = 6.0,
) -> PointLayer:
    """Create a :class:`PointLayer` from raw point positions."""
    pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    col = colors
    if col is None and rainbow:
        col = rainbow_colors(pos.shape[0])
    return PointLayer(positions=pos, colors=col, sizes_px=sizes, size_px_default=default_size)


# ---------------------------------------------------------------------------
# Cellsim adapters
# ---------------------------------------------------------------------------

def cellsim_layers(h: Any, *, rainbow: bool = False) -> Mapping[str, Union[MeshLayer, PointLayer]]:
    """Pack membrane meshes (and optionally other categories) from a cellsim ``h``.

    Parameters
    ----------
    h : object
        Cellsim handle exposing ``cells`` with ``X`` (vertices) and ``F`` (faces).
    rainbow : bool
        If ``True``, apply a rainbow gradient to the membrane vertices.

    Returns
    -------
    dict
        Currently only contains ``{"membrane": MeshLayer}``.
    """
    positions: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for cell in getattr(h, "cells", []):
        X = np.asarray(getattr(cell, "X", np.zeros((0, 3), np.float32)), dtype=np.float32).reshape(-1, 3)
        F = np.asarray(getattr(cell, "F", np.zeros((0, 3), np.uint32)), dtype=np.uint32).reshape(-1, 3)
        if X.size and F.size:
            positions.append(X)
            faces.append(F + offset)
            offset += X.shape[0]

    layers: dict[str, Union[MeshLayer, PointLayer]] = {}
    if positions and faces:
        layers["membrane"] = pack_mesh(np.concatenate(positions), np.concatenate(faces), rainbow=rainbow)
    return layers


def fluid_layers(engine: Any, *, rainbow: bool = False) -> Mapping[str, Union[MeshLayer, PointLayer]]:
    """Pack simple fluid visuals from a fluid engine.

    Currently supports discrete particle fluids by exposing their particle
    positions as a ``PointLayer``.
    """
    layers: dict[str, Union[MeshLayer, PointLayer]] = {}
    pts = getattr(engine, "p", None)
    if pts is not None:
        layers["fluid"] = pack_points(np.asarray(pts, dtype=np.float32), rainbow=rainbow, default_size=2.0)
    return layers


# ---------------------------------------------------------------------------
# Draw once (upload + frame) helper
# ---------------------------------------------------------------------------

def draw_layers(
    renderer: "_GLRenderer_T",
    layers: Mapping[str, Union[MeshLayer, LineLayer, PointLayer]],
    viewport: Optional[Tuple[int, int]] = None,
) -> None:
    """Upload ``layers`` to ``renderer`` and draw a frame.

    When ``renderer`` exposes a ``print_layers`` attribute (see
    :class:`~opengl_render.renderer.DebugRenderer`), the layer mapping is passed
    directly to that method, bypassing all OpenGL work.
    """
    # Debug/print-only path
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

        def _as_colors(pl: PointLayer, n: int) -> np.ndarray:
            if getattr(pl, "colors", None) is not None:
                return pl.colors
            # default opaque white if layer has no colors
            return np.tile(np.array([[1, 1, 1, 1]], dtype=np.float32), (n, 1))

        col = np.concatenate([
            _as_colors(pts, pts.positions.shape[0]),
            _as_colors(ghost, ghost.positions.shape[0]),
        ], axis=0)

        merged = PointLayer(
            positions=pos,
            colors=col,
            sizes_px=getattr(pts, "sizes_px", None),
            size_px_default=getattr(pts, "size_px_default", 6.0),
        )
        renderer.set_points(merged)
    elif isinstance(pts, PointLayer):
        renderer.set_points(pts)
    elif isinstance(ghost, PointLayer):
        renderer.set_points(ghost)

    renderer.draw(viewport)


# ---------------------------------------------------------------------------
# Rendering hooks (sync for debug, threaded for real GL)
# ---------------------------------------------------------------------------

def make_draw_hook(
    renderer_or_factory: Union[RendererFactory, Any],
    *,
    viewport: Optional[Tuple[int, int]] = None,
    history: int = 32,
    loop_mode: str = "idle",
    ghost_trail: bool = True,
):
    """Return a draw hook.

    - If ``renderer_or_factory`` is a **debug/print** renderer *instance* exposing
      ``print_layers``, a synchronous hook is returned (no GL/threading).
    - Otherwise, ``renderer_or_factory`` must be a **factory** (``GLRenderer`` class
      or zero-arg callable). A thread-backed hook is created via
      :func:`make_threaded_draw_hook`, which will construct the window+GL context
      **inside** the worker thread.
    - ``loop_mode`` controls behaviour when the frame queue is empty: ``"idle"``
      (default) re-draws the last frame, ``"loop"`` replays history, and
      ``"bounce"`` ping-pongs through history.
    """
    # Debug synchronous path (e.g. DebugRenderer instance)
    if hasattr(renderer_or_factory, "print_layers"):
        from collections import deque

        maxlen = history if history > 0 else None
        hist_deque: deque[Mapping[str, Union[MeshLayer, LineLayer, PointLayer]]] = deque(maxlen=maxlen)

        def hook(layers: Mapping[str, Union[MeshLayer, LineLayer, PointLayer]]) -> None:
            hist_deque.append(layers)
            frame = layers
            if ghost_trail:
                pts_hist = []
                for past in list(hist_deque)[:-1]:
                    pts = past.get("fluid") or past.get("points")
                    if isinstance(pts, PointLayer):
                        pts_hist.append(pts.positions)
                if pts_hist:
                    frame = dict(layers)
                    frame["ghost"] = rainbow_history_points(pts_hist)
            draw_layers(renderer_or_factory, frame, viewport)  # type: ignore[arg-type]

        return hook

    # Threaded GL path
    factory = _normalize_factory(renderer_or_factory)

    hook, thread = make_threaded_draw_hook(
        factory,
        viewport=viewport,
        history=history,
        loop_mode=loop_mode,
        ghost_trail=ghost_trail,
    )

    # best-effort: give caller a handle to control lifecycle if they pass a stateful object later
    try:
        setattr(hook, "_render_thread", thread)  # type: ignore[attr-defined]
    except Exception:
        pass

    return hook


def _normalize_factory(factory: RendererFactory) -> Callable[[], "_GLRenderer_T"]:
    """Normalize class or callable to a zero-arg constructor."""
    if not callable(factory):
        raise TypeError("renderer_factory must be GLRenderer class or zero-arg callable returning GLRenderer")
    return factory


def make_threaded_draw_hook(
    renderer_factory: RendererFactory,
    *,
    viewport: Optional[Tuple[int, int]] = None,
    history: int = 32,
    loop_mode: str = "idle",
    ghost_trail: bool = True,
):
    """Return a thread-backed draw hook and its controller.

    The returned tuple ``(hook, thread)`` provides a callable ``hook`` that
    enqueues layer mappings to be drawn on a dedicated thread.  ``thread`` is the
    :class:`~opengl_render.threaded.GLRenderThread` instance managing the queue
    and history. ``loop_mode`` determines behaviour when no new frames arrive:
    ``"idle"`` redraws the last frame, while ``"loop"`` and ``"bounce"`` replay
    the stored history. The thread **must** construct the GL context inside
    itself by calling the provided ``renderer_factory``.
    """
    from .threaded import GLRenderThread  # local import to keep headless tests light

    thread = GLRenderThread(
        renderer_factory=_normalize_factory(renderer_factory),
        viewport=viewport,
        history=history,
        loop_mode=loop_mode,
        ghost_trail=ghost_trail,
    )
    return thread.get_submit_hook(), thread
