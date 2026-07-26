"""Conforming, curvature-sensitive triangulation of Riemannian surface maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


ArrayFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class TriangulationTolerance:
    """Independent geometric and first-derivative refinement tolerances."""

    position: float
    tangent: Optional[float] = None
    max_rounds: int = 12
    max_triangles: int = 250_000

    def __post_init__(self) -> None:
        if self.position <= 0.0:
            raise ValueError("position tolerance must be positive")
        if self.tangent is not None and self.tangent <= 0.0:
            raise ValueError("tangent tolerance must be positive")
        if self.max_rounds < 0 or self.max_triangles < 2:
            raise ValueError("invalid refinement limits")


@dataclass(frozen=True)
class TriangulationGeneration:
    """One immutable conforming mesh and its refinement certificate."""

    generation: int
    parameters: np.ndarray
    embedded: np.ndarray
    triangles: np.ndarray
    position_error: np.ndarray
    tangent_error: Optional[np.ndarray]
    surface_sample_count: int
    jacobian_sample_count: int
    converged: bool
    stopped_reason: str

    def __post_init__(self) -> None:
        for name in ("parameters", "embedded", "triangles", "position_error"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.tangent_error is not None:
            tangent = np.array(self.tangent_error, copy=True)
            tangent.setflags(write=False)
            object.__setattr__(self, "tangent_error", tangent)

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))

    @property
    def function_evaluations(self) -> int:
        """Backward-compatible count of requested surface/Jacobian sample rows."""
        return self.surface_sample_count + self.jacobian_sample_count

    @property
    def embedding_dimension(self) -> int:
        return int(self.embedded.shape[1])

    @property
    def triangle_soup(self) -> np.ndarray:
        """Return the first three embedding channels for ordinary renderers."""
        if self.embedding_dimension < 3:
            raise ValueError("rendering requires at least three embedding channels")
        return self.embedded[self.triangles, :3]

    @property
    def dec_faces(self) -> dict[int, list[int]]:
        """Return the face-map convention consumed by Turing's DEC scaffold."""
        return {
            row: [int(vertex) for vertex in triangle]
            for row, triangle in enumerate(self.triangles)
        }

    @property
    def dec_edges(self) -> np.ndarray:
        """Return canonical unoriented edges suitable for incidence assembly."""
        edges = np.concatenate(
            (
                self.triangles[:, (0, 1)],
                self.triangles[:, (1, 2)],
                self.triangles[:, (2, 0)],
            )
        )
        edges.sort(axis=1)
        return np.unique(edges, axis=0)


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


class AdaptiveSurfaceTriangulator:
    """Refine all failing regions of a parametric surface in parallel waves.

    ``surface`` accepts an ``(N, 2)`` parameter batch and returns ``(N, m)``.
    An optional ``jacobian`` returns ``(N, m, 2)`` and enables an independent
    first-derivative certificate.
    """

    def __init__(
        self,
        surface: ArrayFunction,
        *,
        bounds: tuple[tuple[float, float], tuple[float, float]] = (
            (0.0, 1.0),
            (0.0, 1.0),
        ),
        jacobian: Optional[ArrayFunction] = None,
        tolerance: TriangulationTolerance = TriangulationTolerance(1e-3),
        initial_resolution: tuple[int, int] = (1, 1),
        batch_size: Optional[int] = None,
    ) -> None:
        self.surface = surface
        self.jacobian = jacobian
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.tolerance = tolerance
        self.initial_resolution = tuple(int(value) for value in initial_resolution)
        self.batch_size = batch_size
        if self.bounds.shape != (2, 2) or np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError("bounds must contain two increasing intervals")
        if min(self.initial_resolution) < 1:
            raise ValueError("initial_resolution values must be positive")
        if tolerance.tangent is not None and jacobian is None:
            raise ValueError("tangent tolerance requires a jacobian callable")
        self._surface_sample_count = 0
        self._jacobian_sample_count = 0

    def _evaluate(
        self, function: ArrayFunction, parameters: np.ndarray, *, kind: str
    ) -> np.ndarray:
        parameters = np.asarray(parameters, dtype=np.float64)
        chunks = (
            (parameters,)
            if self.batch_size is None
            else np.array_split(
                parameters,
                max(1, int(np.ceil(len(parameters) / self.batch_size))),
            )
        )
        values = [np.asarray(function(chunk), dtype=np.float64) for chunk in chunks]
        result = np.concatenate(values, axis=0)
        if len(result) != len(parameters) or not np.isfinite(result).all():
            raise ValueError("geometry callable returned invalid batched values")
        if result.ndim < 2:
            raise ValueError("geometry callables must preserve a batch axis")
        if kind == "surface":
            if result.ndim != 2:
                raise ValueError("surface must return shape (N, embedding_dimension)")
            self._surface_sample_count += len(parameters)
        elif kind == "jacobian":
            if result.ndim != 3 or result.shape[2] != 2:
                raise ValueError(
                    "jacobian must return shape (N, embedding_dimension, 2)"
                )
            self._jacobian_sample_count += len(parameters)
        else:
            raise ValueError("unknown evaluation kind")
        return result

    def _initial_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        nu, nv = self.initial_resolution
        u = np.linspace(*self.bounds[0], nu + 1)
        v = np.linspace(*self.bounds[1], nv + 1)
        parameters = np.asarray([(x, y) for x in u for y in v])
        triangles = []
        stride = nv + 1
        for i in range(nu):
            for j in range(nv):
                a = i * stride + j
                b = (i + 1) * stride + j
                c = (i + 1) * stride + j + 1
                d = i * stride + j + 1
                triangles.extend(((a, b, c), (a, c, d)))
        return parameters, np.asarray(triangles, dtype=np.int64)

    @staticmethod
    def _unique_edges(triangles: np.ndarray) -> np.ndarray:
        edges = np.concatenate(
            (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)])
        )
        edges.sort(axis=1)
        return np.unique(edges, axis=0)

    def _measure(
        self,
        parameters: np.ndarray,
        embedded: np.ndarray,
        triangles: np.ndarray,
    ) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        edges = self._unique_edges(triangles)
        midpoint_parameters = (
            parameters[edges[:, 0]] + parameters[edges[:, 1]]
        ) * 0.5
        midpoint_values = self._evaluate(
            self.surface, midpoint_parameters, kind="surface"
        )
        midpoint_lookup = {
            _edge_key(*edge): row for row, edge in enumerate(edges)
        }
        edge_errors = np.linalg.norm(
            midpoint_values
            - (embedded[edges[:, 0]] + embedded[edges[:, 1]]) * 0.5,
            axis=1,
        )
        position_error = np.empty(len(triangles), dtype=np.float64)
        for row, triangle in enumerate(triangles):
            indices = [
                midpoint_lookup[_edge_key(triangle[0], triangle[1])],
                midpoint_lookup[_edge_key(triangle[1], triangle[2])],
                midpoint_lookup[_edge_key(triangle[2], triangle[0])],
            ]
            position_error[row] = np.max(edge_errors[indices])

        interior_barycentric = np.asarray(
            (
                (1 / 3, 1 / 3, 1 / 3),
                (0.6, 0.2, 0.2),
                (0.2, 0.6, 0.2),
                (0.2, 0.2, 0.6),
            ),
            dtype=np.float64,
        )
        triangle_parameters = parameters[triangles]
        interior_parameters = np.einsum(
            "ka,nad->nkd", interior_barycentric, triangle_parameters
        )
        interior_values = self._evaluate(
            self.surface,
            interior_parameters.reshape(-1, 2),
            kind="surface",
        ).reshape(len(triangles), len(interior_barycentric), -1)
        linear_interior = np.einsum(
            "ka,nam->nkm", interior_barycentric, embedded[triangles]
        )
        interior_error = np.linalg.norm(
            interior_values - linear_interior, axis=2
        ).max(axis=1)
        position_error = np.maximum(position_error, interior_error)

        tangent_error = None
        if self.jacobian is not None:
            endpoint_jacobian = self._evaluate(
                self.jacobian, parameters, kind="jacobian"
            )
            midpoint_jacobian = self._evaluate(
                self.jacobian, midpoint_parameters, kind="jacobian"
            )
            interior_jacobian = self._evaluate(
                self.jacobian,
                interior_parameters.reshape(-1, 2),
                kind="jacobian",
            ).reshape(
                len(triangles),
                len(interior_barycentric),
                embedded.shape[1],
                2,
            )
            parameter_edges = np.stack(
                (
                    triangle_parameters[:, 1] - triangle_parameters[:, 0],
                    triangle_parameters[:, 2] - triangle_parameters[:, 0],
                ),
                axis=2,
            )
            embedded_edges = np.stack(
                (
                    embedded[triangles[:, 1]] - embedded[triangles[:, 0]],
                    embedded[triangles[:, 2]] - embedded[triangles[:, 0]],
                ),
                axis=2,
            )
            affine_jacobian = np.einsum(
                "nmi,nij->nmj",
                embedded_edges,
                np.linalg.inv(parameter_edges),
            )
            tangent_error = np.empty(len(triangles), dtype=np.float64)
            for row, triangle in enumerate(triangles):
                indices = [
                    midpoint_lookup[_edge_key(triangle[0], triangle[1])],
                    midpoint_lookup[_edge_key(triangle[1], triangle[2])],
                    midpoint_lookup[_edge_key(triangle[2], triangle[0])],
                ]
                samples = np.concatenate(
                    (
                        endpoint_jacobian[triangle],
                        midpoint_jacobian[indices],
                        interior_jacobian[row],
                    ),
                    axis=0,
                )
                tangent_error[row] = np.linalg.norm(
                    samples - affine_jacobian[row], axis=(1, 2)
                ).max()
        return position_error, tangent_error, edges, midpoint_values

    @staticmethod
    def _split_triangle(
        triangle: tuple[int, int, int],
        marked_edges: set[tuple[int, int]],
        midpoint_ids: dict[tuple[int, int], int],
    ) -> list[tuple[int, int, int]]:
        for edge_position, (left, right, opposite) in enumerate(
            (
                (triangle[0], triangle[1], triangle[2]),
                (triangle[1], triangle[2], triangle[0]),
                (triangle[2], triangle[0], triangle[1]),
            )
        ):
            del edge_position
            key = _edge_key(left, right)
            if key in marked_edges:
                midpoint = midpoint_ids[key]
                first = (left, midpoint, opposite)
                second = (midpoint, right, opposite)
                return (
                    AdaptiveSurfaceTriangulator._split_triangle(
                        first, marked_edges, midpoint_ids
                    )
                    + AdaptiveSurfaceTriangulator._split_triangle(
                        second, marked_edges, midpoint_ids
                    )
                )
        return [triangle]

    def triangulate(self) -> TriangulationGeneration:
        self._surface_sample_count = 0
        self._jacobian_sample_count = 0
        parameters, triangles = self._initial_mesh()
        embedded = self._evaluate(self.surface, parameters, kind="surface")
        generation = 0
        stopped_reason = "tolerance"
        while True:
            position_error, tangent_error, edges, midpoint_values = self._measure(
                parameters, embedded, triangles
            )
            failing = position_error > self.tolerance.position
            if tangent_error is not None and self.tolerance.tangent is not None:
                failing |= tangent_error > self.tolerance.tangent
            if not np.any(failing):
                converged = True
                break
            if generation >= self.tolerance.max_rounds:
                converged = False
                stopped_reason = "max_rounds"
                break
            remaining = self.tolerance.max_triangles - len(triangles)
            if remaining <= 0:
                converged = False
                stopped_reason = "max_triangles"
                break

            failing_rows = np.flatnonzero(failing)
            score = position_error[failing_rows] / self.tolerance.position
            if tangent_error is not None and self.tolerance.tangent is not None:
                score = np.maximum(
                    score, tangent_error[failing_rows] / self.tolerance.tangent
                )
            failing_rows = failing_rows[np.argsort(score)[::-1]]
            selected_edges: set[tuple[int, int]] = set()
            raw_edges = np.concatenate(
                (
                    triangles[:, (0, 1)],
                    triangles[:, (1, 2)],
                    triangles[:, (2, 0)],
                )
            )
            raw_edges.sort(axis=1)
            unique_edges, edge_counts = np.unique(
                raw_edges, axis=0, return_counts=True
            )
            incidence = {
                _edge_key(*edge): int(count)
                for edge, count in zip(unique_edges, edge_counts)
            }
            additions = 0
            for row in failing_rows:
                triangle = triangles[row]
                pairs = (
                    _edge_key(triangle[0], triangle[1]),
                    _edge_key(triangle[1], triangle[2]),
                    _edge_key(triangle[2], triangle[0]),
                )
                lengths = [
                    np.linalg.norm(parameters[a] - parameters[b]) for a, b in pairs
                ]
                edge = pairs[int(np.argmax(lengths))]
                cost = 0 if edge in selected_edges else incidence[edge]
                if additions + cost <= remaining:
                    selected_edges.add(edge)
                    additions += cost
            if not selected_edges:
                converged = False
                stopped_reason = "max_triangles"
                break

            edge_rows = {_edge_key(*edge): row for row, edge in enumerate(edges)}
            new_parameters = []
            new_embedded = []
            midpoint_ids = {}
            for edge in sorted(selected_edges):
                midpoint_ids[edge] = len(parameters) + len(new_parameters)
                row = edge_rows[edge]
                new_parameters.append(
                    (parameters[edge[0]] + parameters[edge[1]]) * 0.5
                )
                new_embedded.append(midpoint_values[row])
            parameters = np.concatenate(
                (parameters, np.asarray(new_parameters)), axis=0
            )
            embedded = np.concatenate(
                (embedded, np.asarray(new_embedded)), axis=0
            )
            refined = []
            for triangle in triangles:
                refined.extend(self._split_triangle(
                    tuple(int(value) for value in triangle),
                    selected_edges,
                    midpoint_ids,
                ))
            triangles = np.asarray(refined, dtype=np.int64)
            generation += 1

        return TriangulationGeneration(
            generation=generation,
            parameters=parameters,
            embedded=embedded,
            triangles=triangles,
            position_error=position_error,
            tangent_error=tangent_error,
            surface_sample_count=self._surface_sample_count,
            jacobian_sample_count=self._jacobian_sample_count,
            converged=converged,
            stopped_reason=stopped_reason,
        )
