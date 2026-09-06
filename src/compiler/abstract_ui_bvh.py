"""Deterministic triangle-BVH packing for generated WebGPU contact kernels.

The layout is deliberately compatible with Spectral Analyzer's GPU BVH:
``lo.xyz,left; hi.xyz,right; start,count``.  This module isolates the spatial
contract without coupling Turing physics to Pluck's renderer or optical
materials.  Python owns mesh preprocessing; compiler backends may consume the
packed arrays to emit traversal kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


Point3 = tuple[float, float, float]
Triangle = tuple[Point3, Point3, Point3]


@dataclass(frozen=True, slots=True)
class PackedTriangleBVH:
    """A stable triangle permutation and twelve-float node records."""

    nodes: tuple[tuple[float, ...], ...]
    triangle_order: tuple[int, ...]
    leaf_size: int

    def to_data(self) -> dict[str, object]:
        return {
            "schema": "abstract-ui-packed-triangle-bvh-v0",
            "layout": ["lo.x", "lo.y", "lo.z", "left", "hi.x", "hi.y", "hi.z",
                       "right", "start", "count", "reserved.0", "reserved.1"],
            "nodes": [list(node) for node in self.nodes],
            "triangle_order": list(self.triangle_order),
            "leaf_size": self.leaf_size,
            "provenance": {
                "algorithm": "stable-median-centroid-split-on-longest-axis",
                "isolated_from": "../spectral-analyzer/demo_pluck_gl.py::_build_gpu_bvh",
            },
        }


def build_packed_triangle_bvh(triangles: Iterable[Sequence[Sequence[float]]],
                              *, leaf_size: int = 4) -> PackedTriangleBVH:
    """Build the packed deterministic BVH used by purpose-baked mesh contact.

    Equal centroids retain source order, making generated shader inputs and
    content digests reproducible across runs.
    """

    if leaf_size <= 0:
        raise ValueError("BVH leaf_size must be positive")
    values: list[Triangle] = []
    for triangle in triangles:
        if len(triangle) != 3 or any(len(point) != 3 for point in triangle):
            raise ValueError("each BVH triangle must contain three 3D points")
        values.append(tuple(tuple(float(axis) for axis in point) for point in triangle))  # type: ignore[arg-type]
    if not values:
        return PackedTriangleBVH((), (), leaf_size)

    bounds = [
        (tuple(min(point[axis] for point in triangle) for axis in range(3)),
         tuple(max(point[axis] for point in triangle) for axis in range(3)))
        for triangle in values
    ]
    centroids = [tuple(sum(point[axis] for point in triangle) / 3 for axis in range(3))
                 for triangle in values]
    nodes: list[list[float]] = []
    permutation: list[int] = []

    def emit(indices: list[int]) -> int:
        node_index = len(nodes)
        nodes.append([0.0] * 12)
        lo = [min(bounds[index][0][axis] for index in indices) for axis in range(3)]
        hi = [max(bounds[index][1][axis] for index in indices) for axis in range(3)]
        if len(indices) <= leaf_size:
            start = len(permutation)
            permutation.extend(indices)
            nodes[node_index] = [*lo, -1.0, *hi, -1.0, float(start), float(len(indices)), 0.0, 0.0]
            return node_index
        centroid_lo = [min(centroids[index][axis] for index in indices) for axis in range(3)]
        centroid_hi = [max(centroids[index][axis] for index in indices) for axis in range(3)]
        axis = max(range(3), key=lambda value: centroid_hi[value] - centroid_lo[value])
        ordered = sorted(indices, key=lambda index: centroids[index][axis])
        middle = len(ordered) // 2
        left, right = emit(ordered[:middle]), emit(ordered[middle:])
        nodes[node_index] = [*lo, float(left), *hi, float(right), 0.0, 0.0, 0.0, 0.0]
        return node_index

    emit(list(range(len(values))))
    return PackedTriangleBVH(tuple(tuple(node) for node in nodes), tuple(permutation), leaf_size)


__all__ = ["PackedTriangleBVH", "build_packed_triangle_bvh"]
