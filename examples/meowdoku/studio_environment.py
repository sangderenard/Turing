"""Authored Studio geometry at the existing fountain DEM environment seam.

This is deliberately not a second simulator. Coordinates are metres in the
native Y-up frame. Room planes and exact indexed convex parts call the
incumbent infinite-mass wall law. Optical material/triangle identities stay in
the packet; they never become invented friction coefficients.

Only certified independent convex bodies and convex planar sheets are
accepted here. A compound course is NOT a bag of independent parts. Its union
manifold must be resolved before activation, rather than silently doubling
contacts at joins. The browser bridge is responsible for one shared world
transform, not radius-dependent re-fitting of the room.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from meowdoku import fountain3d as F
from meowdoku.studio_contacts import convex_part_contact, room_planes

SCHEMA = "turing.studio-rigid-scene.v1"
ROOM_NAMES = tuple(f"studio_room_{side}_{axis}" for side in ("low", "high") for axis in "xyz")


def _points(value, name):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain finite XYZ points")
    rounded = a.astype(np.float32).astype(np.float64)
    if not np.isfinite(rounded).all():
        raise ValueError(f"{name} exceeds the native float32 coordinate range")
    return rounded


@dataclass(frozen=True)
class Part:
    id: str
    body_id: str
    closed: bool
    vertices: tuple
    triangles: tuple
    materials: tuple


def _part(record):
    ident = record.get("id")
    body = record.get("bodyId", ident)
    if not isinstance(ident, str) or not ident or not isinstance(body, str) or not body:
        raise ValueError("every part needs stable part and body IDs")
    if record.get("convex") is not True or type(record.get("closed")) is not bool:
        raise ValueError(f"{ident}: unsupported geometry needs an authored convex manifold")
    vertices = _points(record.get("vertices"), ident)
    # Admission predicates describe the actual float32 boundary, not a nearby
    # repaired mesh. Binary floats are exact rationals here: tiny real folds
    # must not be swallowed by an extent-dependent tolerance.
    exact = [tuple(Fraction(float(v)) for v in row) for row in vertices]
    def sub(a, b):
        return tuple(a[k] - b[k] for k in range(3))
    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    def dot(a, b):
        return sum(a[k]*b[k] for k in range(3))
    triangles = record.get("triangles")
    if not isinstance(triangles, (list, tuple)) or not triangles:
        raise ValueError(f"{ident}: missing indexed triangles")
    edges = {}
    normals = []
    indices = []
    volume6 = Fraction(0)
    for triangle in triangles:
        if len(triangle) != 3 or any(type(i) is not int or i < 0 or i >= len(vertices) for i in triangle):
            raise ValueError(f"{ident}: invalid triangle index")
        a, b, c = vertices[list(triangle)]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            native_normal = np.cross((b-a).astype(np.float32), (c-a).astype(np.float32))
            native_area2 = np.sum(native_normal*native_normal, dtype=np.float32)
        if not np.isfinite(native_area2) or native_area2 <= 0:
            raise ValueError(f"{ident}: triangle exceeds native float32 contact precision")
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length == 0:
            raise ValueError(f"{ident}: triangle degenerates in native float32 coordinates")
        normal /= length
        ea, eb, ec = (exact[i] for i in triangle)
        en = cross(sub(eb, ea), sub(ec, ea))
        offsets = [dot(sub(v, ea), en) for v in exact]
        if record["closed"] and max(offsets) > 0:
            raise ValueError(f"{ident}: boundary is not outward-wound convex geometry")
        if not record["closed"] and any(offsets):
            raise ValueError(f"{ident}: an open part must be a planar sheet")
        normals.append(en)
        volume6 += dot(ea, cross(eb, ec))
        indices.append(tuple(triangle))
        for u, v in zip(triangle, (*triangle[1:], triangle[0])):
            if (u, v) in edges:
                raise ValueError(f"{ident}: overlapping or inconsistently wound triangle edges")
            edges[u, v] = True
    boundary = [(u, v) for u, v in edges if (v, u) not in edges]
    if record["closed"]:
        if boundary:
            raise ValueError(f"{ident}: closed part has an open boundary")
        if volume6 <= 0:
            raise ValueError(f"{ident}: a closed convex solid must have positive volume")
    else:
        if not boundary or any(dot(n, normals[0]) <= 0 for n in normals):
            raise ValueError(f"{ident}: sheet must have one consistently wound filled boundary")
        successor = dict(boundary)
        if len(successor) != len(boundary) or len(set(successor.values())) != len(boundary):
            raise ValueError(f"{ident}: sheet boundary branches")
        seen, cursor = set(), boundary[0][0]
        while cursor not in seen:
            seen.add(cursor)
            if cursor not in successor:
                raise ValueError(f"{ident}: sheet boundary is open")
            cursor = successor[cursor]
        if cursor != boundary[0][0] or len(seen) != len(boundary):
            raise ValueError(f"{ident}: sheet has holes or disconnected boundaries")
        for u, v in boundary:
            inward = cross(normals[0], sub(exact[v], exact[u]))
            if any(dot(sub(p, exact[u]), inward) < 0 for p in exact):
                raise ValueError(f"{ident}: sheet boundary is not convex")
    used = set(i for tri in indices for i in tri)
    if len(used) != len(vertices):
        raise ValueError(f"{ident}: unreferenced vertices are not an authored boundary")
    materials = record.get("materialIds", [record.get("materialId")] * len(indices))
    if len(materials) != len(indices) or any(type(m) is not int or m < 0 for m in materials):
        raise ValueError(f"{ident}: retain a material ID for every original triangle")
    return Part(ident, body, record["closed"], tuple(map(tuple, vertices)), tuple(indices), tuple(materials))


class StudioEnvironment:
    """Compile-time topology with runtime exact vertex and room uniforms."""

    def __init__(self, packet):
        if packet.get("schema") != SCHEMA:
            raise ValueError("unsupported Studio rigid-scene schema")
        bounds = _points([packet["room"]["low"], packet["room"]["high"]], "room bounds")
        if np.any(bounds[0] >= bounds[1]):
            raise ValueError("room bounds must strictly increase")
        self.bounds = tuple(map(tuple, bounds))
        self.parts = tuple(_part(part) for part in packet.get("parts", ()))
        if len({p.id for p in self.parts}) != len(self.parts):
            raise ValueError("part IDs must be unique")
        if len({p.body_id for p in self.parts}) != len(self.parts):
            raise ValueError("compound bodies require shared union-surface ownership before activation")
        self.input_names = ROOM_NAMES + tuple(
            self.vertex_name(i, j, k) for i, part in enumerate(self.parts)
            for j in range(len(part.vertices)) for k in range(3))

    @staticmethod
    def vertex_name(part, vertex, axis):
        return f"studio_p{part}_v{vertex}_{'xyz'[axis]}"

    @property
    def arguments(self):
        return F.ARGUMENTS + self.input_names

    def uniforms(self):
        values = dict(zip(ROOM_NAMES, (v for side in self.bounds for v in side)))
        values.update({self.vertex_name(i, j, k): vertex[k]
                       for i, part in enumerate(self.parts)
                       for j, vertex in enumerate(part.vertices) for k in range(3)})
        return {name: F._f32(value) for name, value in values.items()}

    def _triangles(self, i, a):
        part = self.parts[i]
        vertices = tuple(tuple(a[self.vertex_name(i, j, k)] for k in range(3))
                         for j in range(len(part.vertices)))
        return tuple(tuple(vertices[j] for j in triangle) for triangle in part.triangles)

    def contacts(self, p, radius, live, a, wall):
        low, high = tuple(a[k] for k in ROOM_NAMES[:3]), tuple(a[k] for k in ROOM_NAMES[3:])
        for distance, normal in room_planes(p, low, high):
            wall(F._cap(radius - distance, 6.0 * radius), live, normal)
        for i, part in enumerate(self.parts):
            distance, normal, _, valid, _ = convex_part_contact(p, self._triangles(i, a), closed=part.closed)
            wall(F._cap(radius - distance, 6.0 * radius), live * valid, normal)

    def contains(self, p, radius, a):
        inside = p[0] * 0.0 + 1.0
        for k in range(3):
            inside = inside * F.as_float(p[k] >= a[ROOM_NAMES[k]] - radius)
            inside = inside * F.as_float(p[k] <= a[ROOM_NAMES[k + 3]] + radius)
        for i, part in enumerate(self.parts):
            if part.closed:
                distance, _, _, valid, _ = convex_part_contact(p, self._triangles(i, a), closed=True)
                inside = inside * valid * F.as_float(distance >= -radius)
        return inside

    def law(self, pad, active=None, *, dt=F.DT):
        return F.make_law(pad, active, sdf=False, environment=self, dt=dt)

    def grid_uniforms(self, radius, active=1):
        """Exact spatial-hash range, independent of render fitting/species clamps.

        The native grid's Y origin is fixed. The shared frame adapter must put
        the entire room above it. No coordinates are clamped or resized here.
        Float32 integer cell keys stay exact. For a bounded marble count no
        larger than the incumbent neighbor walk, coarsening only the SEARCH
        grid is exact: every potentially adjacent grain is still examined.
        Geometry, radii, positions, time and forces never change with it.
        """
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("physical radius must be positive")
        if type(active) is not int or active < 1:
            raise ValueError("active marble count must be positive")
        low, high = self.bounds
        if low[1] < F.FLOOR_Y or low[1] <= 0:
            raise ValueError("shared native frame must keep room above the parked origin and grid floor")
        cell = float(np.float32(2.1 * radius))
        if not math.isfinite(cell) or cell <= 0:
            raise ValueError("radius exceeds native float32 search-grid precision")
        half, depth = max(abs(low[0]), abs(high[0])) + radius, max(abs(low[2]), abs(high[2])) + radius
        while True:
            cols = math.ceil(2 * half / cell)
            rows = math.ceil((high[1] + radius - F.FLOOR_Y) / cell)
            lays = math.ceil(2 * depth / cell)
            cells = cols * rows * lays
            if cells + 1 < 2 ** 24:
                break
            if active > F.K_RUN_MAX:
                raise ValueError("room/radius ratio exceeds exact native spatial-hash capacity for this neighbor count")
            cell = float(np.float32(cell * max(1.01, (cells / (2 ** 24 - 2)) ** (1/3) * 1.01)))
        return {k: F._f32(v) for k, v in dict(cell=cell, half=half, dh=depth,
                                            cols=cols, rows=rows, lays=lays,
                                            k_run=F.K_RUN_MAX).items()}
