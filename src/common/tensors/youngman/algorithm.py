"""A compact AbstractTensor reconstruction of YoungMan's numeric core.

The original YoungManAlgorithm combines geometry compilation, field queries,
surface extraction, dynamics, and rendering.  This module deliberately keeps
only the first three responsibilities.  AbstractTensor owns the batched
numeric work; Python/NumPy owns the discrete tetrahedron case table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..abstraction import AbstractTensor


_TETRA_EDGES = np.asarray(
    ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int64
)

# Each cube is divided along the 0 -> 6 body diagonal.
_CUBE_TETRAHEDRA = np.asarray(
    ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
     (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)),
    dtype=np.int64,
)


@dataclass(frozen=True)
class ExtractionResult:
    """Surface mesh plus intermediate state useful for analysis."""

    triangles: np.ndarray
    field_values: np.ndarray
    active_edges: np.ndarray
    case_ids: np.ndarray
    elapsed_seconds: float
    solver_samples: Optional["SolverSampleBatch"] = None
    parametric_triangles: Optional[np.ndarray] = None
    triangle_tetrahedron_ids: Optional[np.ndarray] = None

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])


@dataclass(frozen=True)
class DomainTetrahedra:
    """Matching parametric and embedded tetrahedra from one domain."""

    parametric: np.ndarray
    embedded: np.ndarray


@dataclass(frozen=True)
class SolverSampleBatch:
    """Bulk export of every active-edge interpolation performed by YoungMan."""

    tetrahedron_ids: np.ndarray
    edge_ids: np.ndarray
    interpolation_weights: np.ndarray
    embedded_points: np.ndarray
    parametric_points: Optional[np.ndarray] = None
    expanded_points: Optional[np.ndarray] = None
    metric_tags: Optional["MetricSampleTagBatch"] = None

    @property
    def sample_count(self) -> int:
        return int(self.embedded_points.shape[0])


@dataclass(frozen=True)
class MetricSampleTagBatch:
    """Matrix-valued local geometry copied alongside solver movements."""

    metric: np.ndarray
    inverse_metric: np.ndarray
    determinant: np.ndarray
    parameter_positions: np.ndarray
    patch_ids: np.ndarray
    generation: Optional[int] = None
    source: str = "source"

    def __post_init__(self) -> None:
        metric = np.asarray(self.metric, dtype=np.float64)
        inverse = np.asarray(self.inverse_metric, dtype=np.float64)
        determinant = np.asarray(self.determinant, dtype=np.float64)
        parameters = np.asarray(self.parameter_positions, dtype=np.float64)
        patch_ids = np.asarray(self.patch_ids, dtype=np.int64)
        if metric.ndim != 3 or metric.shape[1] != metric.shape[2]:
            raise ValueError("metric must have shape (N, d, d)")
        if inverse.shape != metric.shape:
            raise ValueError("inverse_metric must match metric")
        if not (
            len(metric) == len(determinant) == len(parameters) == len(patch_ids)
        ):
            raise ValueError("metric tag arrays need equal row counts")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "inverse_metric", inverse)
        object.__setattr__(self, "determinant", determinant)
        object.__setattr__(self, "parameter_positions", parameters)
        object.__setattr__(self, "patch_ids", patch_ids)


def metric_sample_tags(
    parameters: np.ndarray,
    patch_ids: np.ndarray,
    metric: np.ndarray,
    *,
    generation: Optional[int] = None,
    source: str = "source",
) -> MetricSampleTagBatch:
    """Build a validated metric-matrix sidecar without collapsing its axes."""
    metric = np.asarray(metric, dtype=np.float64)
    determinant = np.linalg.det(metric)
    if np.any(determinant <= 0.0):
        raise ValueError("metric matrices must be positive definite")
    return MetricSampleTagBatch(
        metric=metric,
        inverse_metric=np.linalg.inv(metric),
        determinant=determinant,
        parameter_positions=np.asarray(parameters, dtype=np.float64),
        patch_ids=np.asarray(patch_ids, dtype=np.int64),
        generation=generation,
        source=source,
    )


def compile_grid_domain(domain) -> DomainTetrahedra:
    """Compile a domain while retaining coordinates on both sides of its map."""
    x, y, z = domain.resolve_positions(full_geometry=False)
    embedded_vertices = np.stack(
        (
            np.asarray(x.tolist()),
            np.asarray(y.tolist()),
            np.asarray(z.tolist()),
        ),
        axis=-1,
    )
    parametric_vertices = np.stack(
        (
            np.asarray(domain.U.tolist()),
            np.asarray(domain.V.tolist()),
            np.asarray(domain.W.tolist()),
        ),
        axis=-1,
    )
    resolution = np.asarray(embedded_vertices.shape[:3]) - 1
    if np.any(resolution < 1):
        raise ValueError("GridDomain needs at least two samples on every axis")
    embedded_cubes = []
    parametric_cubes = []
    for i in range(int(resolution[0])):
        for j in range(int(resolution[1])):
            for k in range(int(resolution[2])):
                indices = (
                    (i, j, k),
                    (i + 1, j, k),
                    (i + 1, j + 1, k),
                    (i, j + 1, k),
                    (i, j, k + 1),
                    (i + 1, j, k + 1),
                    (i + 1, j + 1, k + 1),
                    (i, j + 1, k + 1),
                )
                embedded_cubes.append(tuple(embedded_vertices[index] for index in indices))
                parametric_cubes.append(tuple(parametric_vertices[index] for index in indices))
    embedded = np.asarray(embedded_cubes)[:, _CUBE_TETRAHEDRA].reshape(-1, 4, 3)
    parametric = np.asarray(parametric_cubes)[:, _CUBE_TETRAHEDRA].reshape(-1, 4, 3)
    return DomainTetrahedra(parametric=parametric, embedded=embedded)


def tetrahedra_from_grid_domain(domain) -> np.ndarray:
    """Compile a Laplace ``GridDomain`` into embedded YoungMan cells."""
    return compile_grid_domain(domain).embedded


def sphere_field(
    points: AbstractTensor, radius: float = 0.8, center: float = 0.0
) -> AbstractTensor:
    """Analytical signed field used by the demo and tests."""
    relative = points - center
    return (relative * relative).sum(dim=-1) - radius * radius


def _ordered_polygon_edges(points: np.ndarray, edge_ids: np.ndarray) -> np.ndarray:
    """Order one tetrahedral crossing polygon around its centroid."""
    polygon = points[edge_ids]
    centered = polygon - polygon.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    coordinates = np.stack(
        (centered @ axes[0], centered @ axes[1]), axis=1
    )
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    return edge_ids[np.argsort(angles)]


def _triangle_selections(
    crossings: np.ndarray, topology_points: np.ndarray
) -> list[tuple[int, np.ndarray]]:
    """Return geometrically ordered selections shared by every coordinate system."""
    selections: list[tuple[int, np.ndarray]] = []
    for cell_id, mask in enumerate(crossings):
        active = np.flatnonzero(mask)
        if active.size >= 3:
            active = _ordered_polygon_edges(topology_points[cell_id], active)
        if active.size == 3:
            selections.append((cell_id, active))
        elif active.size == 4:
            selections.extend(
                (
                    (cell_id, active[[0, 1, 2]]),
                    (cell_id, active[[0, 2, 3]]),
                )
            )
    return selections


def _assemble_triangles(
    selections: list[tuple[int, np.ndarray]], crossing_points: np.ndarray
) -> np.ndarray:
    """Apply one topology selection to points in any coordinate system."""
    triangles: list[np.ndarray] = []
    for cell_id, edge_ids in selections:
        triangles.append(crossing_points[cell_id, edge_ids])
    if not triangles:
        return np.empty(
            (0, 3, crossing_points.shape[-1]), dtype=np.float64
        )
    return np.asarray(triangles, dtype=np.float64)


def extract_isosurface(
    tetrahedra: np.ndarray,
    field: Callable[[AbstractTensor], AbstractTensor] = sphere_field,
    *,
    iso_value: float = 0.0,
    parametric_tetrahedra: Optional[np.ndarray] = None,
    expanded_embedding: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> ExtractionResult:
    """Extract a triangle surface using AbstractTensor field/interpolation work."""
    import time

    started = time.perf_counter()
    tetrahedra = np.asarray(tetrahedra, dtype=np.float64)
    if tetrahedra.ndim != 3 or tetrahedra.shape[1:] != (4, 3):
        raise ValueError("tetrahedra must have shape (N, 4, 3)")
    if parametric_tetrahedra is not None:
        parametric_tetrahedra = np.asarray(parametric_tetrahedra, dtype=np.float64)
        if parametric_tetrahedra.shape[:2] != tetrahedra.shape[:2]:
            raise ValueError(
                "parametric_tetrahedra must match tetrahedra's cell and vertex axes"
            )

    vertices = AbstractTensor.get_tensor(tetrahedra)
    values = field(vertices) - iso_value
    values_np = np.asarray(values.tolist(), dtype=np.float64)
    inside = values_np < 0.0
    case_ids = (
        inside[:, 0].astype(np.uint8)
        | (inside[:, 1].astype(np.uint8) << 1)
        | (inside[:, 2].astype(np.uint8) << 2)
        | (inside[:, 3].astype(np.uint8) << 3)
    )

    edge_start = vertices[:, _TETRA_EDGES[:, 0], :]
    edge_end = vertices[:, _TETRA_EDGES[:, 1], :]
    value_start = values[:, _TETRA_EDGES[:, 0]]
    value_end = values[:, _TETRA_EDGES[:, 1]]
    active_edges = inside[:, _TETRA_EDGES[:, 0]] != inside[:, _TETRA_EDGES[:, 1]]

    denominator = value_start - value_end
    safe_denominator = AbstractTensor.where(
        denominator.abs() < 1e-12, 1.0, denominator
    )
    weight = (value_start / safe_denominator).reshape(tetrahedra.shape[0], 6, 1)
    crossing_points = edge_start + weight * (edge_end - edge_start)
    crossing_np = np.asarray(crossing_points.tolist(), dtype=np.float64)
    tetrahedron_ids, edge_ids = np.nonzero(active_edges)
    weights_np = np.asarray(weight.tolist(), dtype=np.float64)[..., 0]
    parametric_points = None
    expanded_points = None
    parametric_triangles = None
    parametric_crossings = None
    if parametric_tetrahedra is not None:
        parametric_start = parametric_tetrahedra[:, _TETRA_EDGES[:, 0], :]
        parametric_end = parametric_tetrahedra[:, _TETRA_EDGES[:, 1], :]
        parametric_crossings = (
            parametric_start
            + weights_np[..., None] * (parametric_end - parametric_start)
        )
        parametric_points = parametric_crossings[active_edges]
        if expanded_embedding is not None:
            expanded_value = expanded_embedding(parametric_points)
            if isinstance(expanded_value, AbstractTensor):
                expanded_value = expanded_value.tolist()
            expanded_points = np.asarray(expanded_value, dtype=np.float64)
            if (
                expanded_points.ndim != 2
                or len(expanded_points) != len(parametric_points)
            ):
                raise ValueError(
                    "expanded_embedding must return shape (sample_count, m)"
                )
            if not np.isfinite(expanded_points).all():
                raise ValueError("expanded_embedding returned non-finite values")
    topology_points = (
        crossing_np if parametric_crossings is None else parametric_crossings
    )
    triangle_selections = _triangle_selections(active_edges, topology_points)
    triangles = _assemble_triangles(triangle_selections, crossing_np)
    triangle_tetrahedron_ids = np.asarray(
        [cell_id for cell_id, _ in triangle_selections], dtype=np.int64
    )
    if parametric_crossings is not None:
        parametric_triangles = _assemble_triangles(
            triangle_selections, parametric_crossings
        )
    solver_samples = SolverSampleBatch(
        tetrahedron_ids=tetrahedron_ids.astype(np.int64, copy=False),
        edge_ids=edge_ids.astype(np.int8, copy=False),
        interpolation_weights=weights_np[active_edges],
        embedded_points=crossing_np[active_edges],
        parametric_points=parametric_points,
        expanded_points=expanded_points,
    )

    return ExtractionResult(
        triangles=triangles,
        field_values=values_np,
        active_edges=active_edges,
        case_ids=case_ids,
        elapsed_seconds=time.perf_counter() - started,
        solver_samples=solver_samples,
        parametric_triangles=parametric_triangles,
        triangle_tetrahedron_ids=triangle_tetrahedron_ids,
    )


def triangle_areas(triangles: np.ndarray) -> np.ndarray:
    """Return Euclidean area for every extracted triangle."""
    if triangles.size == 0:
        return np.empty(0, dtype=np.float64)
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    return 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
