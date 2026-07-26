"""Cotangent Laplace--Beltrami geometry in arbitrary embedding dimension."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CotangentMeshGeometry:
    edges: np.ndarray
    cotangent_weights: np.ndarray
    triangle_areas: np.ndarray
    lumped_vertex_areas: np.ndarray
    boundary_vertex_mask: np.ndarray
    degenerate_triangle_mask: np.ndarray
    degenerate_vertex_mask: np.ndarray
    singular_vertex_mask: np.ndarray
    nonmanifold_edge_mask: np.ndarray
    nonmanifold_vertex_mask: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "edges",
            "cotangent_weights",
            "triangle_areas",
            "lumped_vertex_areas",
            "boundary_vertex_mask",
            "degenerate_triangle_mask",
            "degenerate_vertex_mask",
            "singular_vertex_mask",
            "nonmanifold_edge_mask",
            "nonmanifold_vertex_mask",
        ):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def invalid_vertex_mask(self) -> np.ndarray:
        return (
            self.singular_vertex_mask
            | self.degenerate_vertex_mask
            | self.nonmanifold_vertex_mask
        )

    def apply(
        self, scalar_values: np.ndarray, *, invalid_value: float = np.nan
    ) -> np.ndarray:
        values = np.asarray(scalar_values, dtype=np.float64)
        if values.shape != self.lumped_vertex_areas.shape:
            raise ValueError("scalar_values needs one value per mesh vertex")
        result = np.zeros_like(values)
        left, right = self.edges.T
        delta = self.cotangent_weights * (values[right] - values[left])
        np.add.at(result, left, delta)
        np.add.at(result, right, -delta)
        valid = ~self.invalid_vertex_mask
        result[valid] /= self.lumped_vertex_areas[valid]
        result[~valid] = invalid_value
        return result


@dataclass(frozen=True)
class MeshLaplaceResult:
    laplacian: np.ndarray
    geometry: CotangentMeshGeometry


def build_cotangent_geometry(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    degeneracy_tolerance: float = 1e-12,
) -> CotangentMeshGeometry:
    """Build lumped mass and cotangent weights using full ambient geometry."""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] < 2:
        raise ValueError("vertices must have shape (N, embedding_dimension>=2)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (T, 3)")
    if len(triangles) and (
        triangles.min() < 0 or triangles.max() >= len(vertices)
    ):
        raise ValueError("triangle index outside vertex array")

    points = vertices[triangles]
    edge_01 = points[:, 1] - points[:, 0]
    edge_02 = points[:, 2] - points[:, 0]
    gram = (
        np.einsum("ni,ni->n", edge_01, edge_01)
        * np.einsum("ni,ni->n", edge_02, edge_02)
        - np.einsum("ni,ni->n", edge_01, edge_02) ** 2
    )
    double_area = np.sqrt(np.maximum(gram, 0.0))
    triangle_areas = 0.5 * double_area
    degenerate_triangles = (
        ~np.isfinite(double_area) | (double_area <= degeneracy_tolerance)
    )
    safe_double_area = np.where(degenerate_triangles, 1.0, double_area)

    cotangent = np.empty((len(triangles), 3), dtype=np.float64)
    for corner, other_a, other_b in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        a = points[:, other_a] - points[:, corner]
        b = points[:, other_b] - points[:, corner]
        cotangent[:, corner] = (
            np.einsum("ni,ni->n", a, b) / safe_double_area
        )
    cotangent[degenerate_triangles] = 0.0

    raw_edges = np.concatenate(
        (
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
            triangles[:, (0, 1)],
        ),
        axis=0,
    )
    raw_weights = 0.5 * np.concatenate(
        (cotangent[:, 0], cotangent[:, 1], cotangent[:, 2])
    )
    raw_edges.sort(axis=1)
    edges, inverse, counts = np.unique(
        raw_edges, axis=0, return_inverse=True, return_counts=True
    )
    weights = np.zeros(len(edges), dtype=np.float64)
    np.add.at(weights, inverse, raw_weights)
    boundary_edges = counts == 1
    nonmanifold_edges = counts > 2
    nonmanifold_vertices = np.zeros(len(vertices), dtype=bool)
    if np.any(nonmanifold_edges):
        nonmanifold_vertices[edges[nonmanifold_edges].ravel()] = True
    boundary_vertices = np.zeros(len(vertices), dtype=bool)
    boundary_vertices[edges[boundary_edges].ravel()] = True

    lumped_area = np.zeros(len(vertices), dtype=np.float64)
    valid_area = np.where(degenerate_triangles, 0.0, triangle_areas) / 3.0
    for corner in range(3):
        np.add.at(lumped_area, triangles[:, corner], valid_area)
    degenerate_vertices = np.zeros(len(vertices), dtype=bool)
    if np.any(degenerate_triangles):
        degenerate_vertices[
            triangles[degenerate_triangles].ravel()
        ] = True
    singular_vertices = (
        ~np.isfinite(lumped_area) | (lumped_area <= degeneracy_tolerance)
    )
    return CotangentMeshGeometry(
        edges=edges,
        cotangent_weights=weights,
        triangle_areas=triangle_areas,
        lumped_vertex_areas=lumped_area,
        boundary_vertex_mask=boundary_vertices,
        degenerate_triangle_mask=degenerate_triangles,
        degenerate_vertex_mask=degenerate_vertices,
        singular_vertex_mask=singular_vertices,
        nonmanifold_edge_mask=nonmanifold_edges,
        nonmanifold_vertex_mask=nonmanifold_vertices,
    )


def mesh_laplace_beltrami(
    vertices: np.ndarray,
    triangles: np.ndarray,
    scalar_values: np.ndarray,
    *,
    degeneracy_tolerance: float = 1e-12,
    invalid_value: float = np.nan,
) -> MeshLaplaceResult:
    geometry = build_cotangent_geometry(
        vertices, triangles, degeneracy_tolerance=degeneracy_tolerance
    )
    return MeshLaplaceResult(
        geometry.apply(scalar_values, invalid_value=invalid_value), geometry
    )
