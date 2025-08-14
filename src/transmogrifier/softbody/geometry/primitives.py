from __future__ import annotations
import numpy as np
from typing import Iterable, Literal, Tuple


def planar_ngon(n_segments: int,
                radius: float = 0.1,
                center: Iterable[float] = (0.0, 0.0, 0.0),
                plane: Literal["xy", "xz", "yz"] = "xy",
                dtype=np.float64,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 2D n-gon ring embedded in 3D without a central vertex.

    Returns (verts, faces_tri, edges_ring) where:
      - verts: (n, 3) float64 ring vertices
      - faces_tri: (n-2, 3) int32 triangulation fan around vertex 0
      - edges_ring: (n, 2) int32 ring edges (i, i+1 mod n)
    """
    n = int(max(3, n_segments))
    cx, cy, cz = map(float, center)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.empty((n, 3), dtype=dtype)
    if plane == "xy":
        ring[:, 0] = cx + radius * np.cos(angles)
        ring[:, 1] = cy + radius * np.sin(angles)
        ring[:, 2] = cz
    elif plane == "xz":
        ring[:, 0] = cx + radius * np.cos(angles)
        ring[:, 2] = cz + radius * np.sin(angles)
        ring[:, 1] = cy
    elif plane == "yz":
        ring[:, 1] = cy + radius * np.cos(angles)
        ring[:, 2] = cz + radius * np.sin(angles)
        ring[:, 0] = cx
    else:
        raise ValueError(f"Unknown plane: {plane}")

    # Triangulate polygon as fan from vertex 0
    i = np.arange(1, n - 1, dtype=np.int32)
    faces = np.column_stack([np.zeros(n - 2, dtype=np.int32), i, i + 1])
    # Ring edges (including wrap-around)
    idx = np.arange(0, n, dtype=np.int32)
    edges = np.column_stack([idx, np.roll(idx, -1)])
    return ring, faces, edges


def line_segment(n_segments: int,
                 radius: float = 0.1,
                 center: Iterable[float] = (0.0, 0.0, 0.0),
                 axis: Literal["x", "y", "z"] = "x",
                 dtype=np.float64,
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a straight polyline centered at `center`, spanning 2*radius along `axis`.

    Returns (verts, edges) where:
      - verts: (n_segments+1, 3) float64
      - edges: (n_segments, 2) int32 pairs
    """
    n = int(max(1, n_segments))
    cx, cy, cz = map(float, center)
    t = np.linspace(-radius, radius, n + 1)
    verts = np.empty((n + 1, 3), dtype=dtype)
    if axis == "x":
        verts[:, 0] = cx + t
        verts[:, 1] = cy
        verts[:, 2] = cz
    elif axis == "y":
        verts[:, 1] = cy + t
        verts[:, 0] = cx
        verts[:, 2] = cz
    elif axis == "z":
        verts[:, 2] = cz + t
        verts[:, 0] = cx
        verts[:, 1] = cy
    else:
        raise ValueError(f"Unknown axis: {axis}")

    edges = np.column_stack([
        np.arange(n, dtype=np.int32),
        np.arange(1, n + 1, dtype=np.int32),
    ])
    return verts, edges
