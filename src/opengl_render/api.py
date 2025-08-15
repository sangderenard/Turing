"""Adapters for building renderer layers from numpy-based simulation data.

This module offers small helper utilities that convert arrays produced by the
``run_numpy_demo`` cellsim into :class:`MeshLayer`, :class:`LineLayer` and
:class:`PointLayer` instances used by :class:`~opengl_render.renderer.GLRenderer`.

The helpers keep imports light so they can be used from headless tests. When
``rainbow`` is enabled a simple HSV→RGB mapping is applied to generate a color
spectrum for each vertex; this mirrors the "rainbow history" effect available in
older demos under ``inspiration/``.
"""
from __future__ import annotations

from typing import Mapping, Iterable
import colorsys
import numpy as np

from .renderer import MeshLayer, LineLayer, PointLayer


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
