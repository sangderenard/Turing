# -*- coding: utf-8 -*-
"""Common solids interface for dt-system engines.

Purpose
-------
Provide a minimal, engine-agnostic description of solid geometry (vertex sets
or triangle meshes) that fluid, softbody, and collision engines can consume.

Goals
-----
- Lightweight: pure dataclasses + numpy arrays; no heavy deps.
- Stable contract: read-only views by default; mutability is explicit.
- Non-invasive: engines may adopt this progressively; existing APIs remain.

"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Literal
import numpy as np

Vec3 = Tuple[float, float, float]


@dataclass
class SurfaceMaterial:
        """Material model for surface contact.

        kind:
            - "elastic": elastic with Coulomb-like friction (default)
            - "soil": high-friction, non-bouncy, sticky contact (embedding-like)
            - "softbody_stub": placeholder for soft-body coupling (treated as elastic)
        """

        kind: Literal["elastic", "soil", "softbody_stub"] = "elastic"
        restitution: float = 0.2
        friction: float = 0.6
        # For soil-like surfaces, controls additional velocity damping [0..1]
        embed_damping: float = 0.5


# A few handy presets
MATERIAL_ELASTIC = SurfaceMaterial("elastic", restitution=0.25, friction=0.6)
MATERIAL_SOIL = SurfaceMaterial("soil", restitution=0.0, friction=0.9, embed_damping=0.8)
MATERIAL_SOFTBODY_STUB = SurfaceMaterial("softbody_stub", restitution=0.2, friction=0.6)
# Low-friction, weakly elastic surface useful for cages or slippery walls
MATERIAL_SLIPPERY = SurfaceMaterial("elastic", restitution=0.05, friction=0.0)


@dataclass
class SolidMesh:
    """Triangle mesh or vertex cloud with optional indexing.

    If ``faces`` is None or empty, the mesh is treated as a vertex set; engines
    may voxelize or sample SDF from points. If provided, faces are integer
    indices into ``vertices`` with shape (M, 3).
    """

    vertices: np.ndarray  # shape (N, 3), dtype=float32/64
    faces: Optional[np.ndarray] = None  # shape (M, 3), dtype=int32/64
    name: str = "solid"
    material: SurfaceMaterial = field(default_factory=lambda: MATERIAL_ELASTIC)

    def as_vertex_array(self) -> np.ndarray:
        v = np.asarray(self.vertices)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"vertices must have shape (N,3); got {v.shape}")
        return v

    def as_face_array(self) -> Optional[np.ndarray]:
        if self.faces is None:
            return None
        f = np.asarray(self.faces)
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"faces must have shape (M,3); got {f.shape}")
        return f


@dataclass
class SolidRegistry:
    """Collection of named solids, for engines to query each frame."""

    solids: Dict[str, SolidMesh] = field(default_factory=dict)

    def add(self, key: str, mesh: SolidMesh) -> None:
        self.solids[key] = mesh

    def remove(self, key: str) -> None:
        self.solids.pop(key, None)

    def get(self, key: str) -> Optional[SolidMesh]:
        return self.solids.get(key)

    def all(self) -> Iterable[Tuple[str, SolidMesh]]:
        return self.solids.items()


def make_box(center: Vec3, size: Vec3, *, name: str = "box") -> SolidMesh:
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
    v = np.array([
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
    ], dtype=np.float32)
    f = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [1, 2, 6], [1, 6, 5],  # right
        [2, 3, 7], [2, 7, 6],  # back
        [3, 0, 4], [3, 4, 7],  # left
    ], dtype=np.int32)
    return SolidMesh(vertices=v, faces=f, name=name)


# Global registry for engines that prefer a shared view
GLOBAL_SOLIDS = SolidRegistry()


# World confinement primitives ---------------------------------------------

@dataclass
class WorldPlane:
    """Half-space plane: n·x + d >= 0 is inside (allowed)."""

    normal: np.ndarray  # shape (3,)
    offset: float  # d in plane equation
    material: SurfaceMaterial = field(default_factory=lambda: MATERIAL_ELASTIC)
    # Optional plane-level fluid boundary override: "wrap" or "respawn"
    fluid_mode: Optional[Literal["wrap", "respawn"]] = None
    # Optional permeability pattern across in-plane axes as (div_u, div_v).
    # When provided, the plane alternates open and solid regions; a zero entry
    # yields stripes, while two positive entries produce a checkerboard.
    permeability: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=float).reshape(-1)
        if n.size != 3:
            raise ValueError("WorldPlane.normal must have 3 components")
        l = float(np.linalg.norm(n)) or 1.0
        self.normal = (n / l).astype(float)

    def is_fluid_permeable(self, point: np.ndarray) -> bool:
        """Return True if a fluid particle at ``point`` may pass through.

        The check assumes axis-aligned planes. When ``permeability`` is
        ``None`` the plane blocks fluid completely. If ``permeability`` is
        provided, the plane is divided into equal regions. With one positive
        division value the regions form stripes; with two they form a
        checkerboard. Regions with even parity are considered openings.
        """
        if self.permeability is None:
            return False

        div_u, div_v = self.permeability
        axis = int(np.argmax(np.abs(self.normal)))
        axes = [0, 1, 2]
        axes.pop(axis)
        u = float(point[axes[0]])
        v = float(point[axes[1]])

        iu = int(np.floor(u * div_u)) if div_u and div_u > 0 else 0
        iv = int(np.floor(v * div_v)) if div_v and div_v > 0 else 0

        if div_u and div_u > 0 and div_v and div_v > 0:
            return (iu + iv) % 2 == 0
        elif div_u and div_u > 0:
            return iu % 2 == 0
        elif div_v and div_v > 0:
            return iv % 2 == 0
        return False

    def warp_position(self, point: np.ndarray) -> np.ndarray:
        """Warp ``point`` across the plane when in wrap mode.

        A simple helper for engines: if ``fluid_mode`` is ``"wrap"`` and the
        point lies outside the plane, translate it to the opposite side while
        preserving tangential coordinates. This behaves like a portal through a
        permeable gap.
        """
        if self.fluid_mode != "wrap":
            return np.asarray(point, dtype=float)

        p = np.asarray(point, dtype=float)
        side = float(np.dot(self.normal, p) + self.offset)
        if side >= 0.0:
            return p
        # Move across the plane by twice the penetration distance
        p = p + self.normal * (-2.0 * side)
        return p


@dataclass
class WorldConfinement:
    """World rules for contact planes and fluid boundary behavior."""

    planes: List[WorldPlane] = field(default_factory=list)
    # Optional axis-aligned bounds for fluids; ((minx,miny,minz),(maxx,maxy,maxz))
    bounds: Optional[Tuple[Vec3, Vec3]] = None
    # Fluid boundary behavior: "wrap" (modulo box) or "respawn" (random re-entry)
    fluid_mode: Optional[Literal["wrap", "respawn"]] = None


# Default world: y >= 0 plane (ground), elastic-like
GLOBAL_WORLD = WorldConfinement(
    planes=[WorldPlane(normal=np.array([0.0, 1.0, 0.0], dtype=float), offset=0.0, material=MATERIAL_ELASTIC)],
    bounds=None,
    fluid_mode=None,
)

__all__ = [
    "SurfaceMaterial",
    "MATERIAL_ELASTIC",
    "MATERIAL_SOIL",
    "MATERIAL_SOFTBODY_STUB",
    "MATERIAL_SLIPPERY",
    "SolidMesh",
    "SolidRegistry",
    "make_box",
    "GLOBAL_SOLIDS",
    "WorldPlane",
    "WorldConfinement",
    "GLOBAL_WORLD",
]
