"""Minimal OpenGL renderer package with mesh, line and point layers.

This module attempts to import the heavy OpenGL-backed :mod:`renderer` and
associated helpers. In headless environments without an OpenGL context these
imports may fail; in that case ``None`` placeholders are exported so that other
modules (e.g., the debug simulation coordinator) can still be imported without
raising ``ImportError``.
"""

from __future__ import annotations

from .threaded import GLRenderThread

try:  # pragma: no cover - best effort in headless CI
    from .renderer import GLRenderer, MeshLayer, LineLayer, PointLayer, DebugRenderer
except Exception:  # noqa: BLE001 - tolerate missing OpenGL libs
    GLRenderer = MeshLayer = LineLayer = PointLayer = DebugRenderer = None  # type: ignore

try:  # pragma: no cover - best effort in headless CI
    from .api import (
        rainbow_colors,
        rainbow_history_points,
        pack_mesh,
        pack_lines,
        pack_points,
        cellsim_layers,
        fluid_layers,
        make_draw_hook,
        make_threaded_draw_hook,
    )
except Exception:  # noqa: BLE001 - tolerate missing OpenGL libs
    rainbow_colors = rainbow_history_points = pack_mesh = None  # type: ignore
    pack_lines = pack_points = cellsim_layers = fluid_layers = None  # type: ignore
    make_draw_hook = make_threaded_draw_hook = None  # type: ignore

__all__ = [
    "GLRenderer",
    "GLRenderThread",
    "DebugRenderer",
    "MeshLayer",
    "LineLayer",
    "PointLayer",
    # API helpers
    "rainbow_colors",
    "rainbow_history_points",
    "pack_mesh",
    "pack_lines",
    "pack_points",
    "cellsim_layers",
    "fluid_layers",
    "make_draw_hook",
    "make_threaded_draw_hook",
]
