"""A mesh-backed parametric surface transform for Turing and Nodus consumers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh_laplace import (
    CotangentMeshGeometry,
    MeshLaplaceResult,
    build_cotangent_geometry,
)


@dataclass(frozen=True)
class TriangulatedSurfaceTransform:
    """Piecewise-affine 2D-to-M transform backed by conforming triangles."""

    parameters: np.ndarray
    embedded: np.ndarray
    triangles: np.ndarray
    geometry: CotangentMeshGeometry

    @classmethod
    def from_mesh(
        cls,
        parameters: np.ndarray,
        embedded: np.ndarray,
        triangles: np.ndarray,
    ) -> "TriangulatedSurfaceTransform":
        parameters = np.asarray(parameters, dtype=np.float64)
        embedded = np.asarray(embedded, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int64)
        if parameters.ndim != 2 or parameters.shape[1] != 2:
            raise ValueError("surface parameters must have shape (N, 2)")
        if embedded.ndim != 2 or len(embedded) != len(parameters):
            raise ValueError("embedded values need one row per parameter vertex")
        geometry = build_cotangent_geometry(embedded, triangles)
        values = []
        for value in (parameters, embedded, triangles):
            frozen = np.array(value, copy=True)
            frozen.setflags(write=False)
            values.append(frozen)
        return cls(*values, geometry)

    def _triangle_jacobians(self) -> np.ndarray:
        parameter_triangles = self.parameters[self.triangles]
        embedded_triangles = self.embedded[self.triangles]
        parameter_edges = np.stack(
            (
                parameter_triangles[:, 1] - parameter_triangles[:, 0],
                parameter_triangles[:, 2] - parameter_triangles[:, 0],
            ),
            axis=2,
        )
        embedded_edges = np.stack(
            (
                embedded_triangles[:, 1] - embedded_triangles[:, 0],
                embedded_triangles[:, 2] - embedded_triangles[:, 0],
            ),
            axis=2,
        )
        return np.einsum(
            "nmi,nij->nmj", embedded_edges, np.linalg.inv(parameter_edges)
        )

    def locate(
        self, query: np.ndarray, *, tolerance: float = 1e-10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Locate parameter queries and return triangle IDs plus barycentrics."""
        query = np.asarray(query, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError("query must have shape (N, 2)")
        owners = np.full(len(query), -1, dtype=np.int64)
        barycentric = np.full((len(query), 3), np.nan, dtype=np.float64)
        for triangle_id, indices in enumerate(self.triangles):
            pending = np.flatnonzero(owners < 0)
            if not len(pending):
                break
            vertices = self.parameters[indices]
            edges = np.stack(
                (vertices[1] - vertices[0], vertices[2] - vertices[0]), axis=1
            )
            local = (query[pending] - vertices[0]) @ np.linalg.inv(edges).T
            weights = np.column_stack(
                (1.0 - local.sum(axis=1), local[:, 0], local[:, 1])
            )
            inside = np.all(weights >= -tolerance, axis=1)
            claimed = pending[inside]
            owners[claimed] = triangle_id
            barycentric[claimed] = weights[inside]
        return owners, barycentric

    def transform(self, query: np.ndarray) -> np.ndarray:
        owners, weights = self.locate(query)
        if np.any(owners < 0):
            raise ValueError("one or more queries lie outside the mesh chart")
        return np.einsum(
            "ni,nim->nm", weights, self.embedded[self.triangles[owners]]
        )

    def metric_tensor(self, query: np.ndarray) -> np.ndarray:
        owners, _ = self.locate(query)
        if np.any(owners < 0):
            raise ValueError("one or more queries lie outside the mesh chart")
        jacobian = self._triangle_jacobians()[owners]
        return np.einsum("nmi,nmj->nij", jacobian, jacobian)

    def laplace(self, scalar_vertex_values: np.ndarray) -> MeshLaplaceResult:
        return MeshLaplaceResult(
            self.geometry.apply(scalar_vertex_values), self.geometry
        )

    def nodus_payload(self) -> dict[str, np.ndarray]:
        """Return explicit array ownership boundaries for a Nodus adapter."""
        return {
            "parameters": np.array(self.parameters, copy=True),
            "embedded": np.array(self.embedded, copy=True),
            "triangles": np.array(self.triangles, copy=True),
            "edges": np.array(self.geometry.edges, copy=True),
        }
