"""YoungMan -> FIFO spline -> metric/Laplace round-trip demonstration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from ..abstraction import AbstractTensor
from ..abstract_convolution.laplace_nd import (
    GridDomain,
    continuous_laplace_beltrami,
)
from .algorithm import (
    DomainTetrahedra,
    compile_grid_domain,
    extract_isosurface,
    metric_sample_tags,
)
from .piecewise import StreamingPiecewiseSplineEngine


def detailed_embedding(parameters: np.ndarray) -> np.ndarray:
    """A rippled 3->5 embedding with visible and latent geometric detail."""
    u, v, w = np.asarray(parameters, dtype=np.float64).T
    tau = 2.0 * np.pi
    return np.stack(
        (
            u,
            v,
            w + 0.075 * np.sin(tau * u) * np.sin(tau * v),
            0.11 * np.sin(2.0 * tau * u) * np.cos(1.5 * tau * v) * (1.0 + 0.2 * w),
            0.09 * np.cos(1.5 * tau * u) * np.sin(2.0 * tau * w)
            + 0.035 * np.sin(2.5 * tau * v),
        ),
        axis=1,
    )


def detailed_jacobian(parameters: np.ndarray) -> np.ndarray:
    """Analytic Jacobian of :func:`detailed_embedding`."""
    u, v, w = np.asarray(parameters, dtype=np.float64).T
    tau = 2.0 * np.pi
    jacobian = np.zeros((len(parameters), 5, 3), dtype=np.float64)
    jacobian[:, 0, 0] = 1.0
    jacobian[:, 1, 1] = 1.0
    jacobian[:, 2, 0] = 0.075 * tau * np.cos(tau * u) * np.sin(tau * v)
    jacobian[:, 2, 1] = 0.075 * tau * np.sin(tau * u) * np.cos(tau * v)
    jacobian[:, 2, 2] = 1.0
    jacobian[:, 3, 0] = (
        0.22 * tau * np.cos(2.0 * tau * u)
        * np.cos(1.5 * tau * v) * (1.0 + 0.2 * w)
    )
    jacobian[:, 3, 1] = (
        -0.165 * tau * np.sin(2.0 * tau * u)
        * np.sin(1.5 * tau * v) * (1.0 + 0.2 * w)
    )
    jacobian[:, 3, 2] = (
        0.022 * np.sin(2.0 * tau * u) * np.cos(1.5 * tau * v)
    )
    jacobian[:, 4, 0] = (
        -0.135 * tau * np.sin(1.5 * tau * u) * np.sin(2.0 * tau * w)
    )
    jacobian[:, 4, 1] = 0.0875 * tau * np.cos(2.5 * tau * v)
    jacobian[:, 4, 2] = (
        0.18 * tau * np.cos(1.5 * tau * u) * np.cos(2.0 * tau * w)
    )
    return jacobian


def induced_metric(parameters: np.ndarray) -> np.ndarray:
    jacobian = detailed_jacobian(parameters)
    return np.einsum("nmi,nmj->nij", jacobian, jacobian)


def singular_metric_mask(
    metric: np.ndarray,
    *,
    determinant_floor: float = 1e-10,
    eigenvalue_floor: float = 1e-9,
    condition_limit: float = 1e10,
) -> np.ndarray:
    """Detect degenerate or numerically unsafe metric matrices."""
    metric = np.asarray(metric, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(metric)
    determinant = np.linalg.det(metric)
    smallest = eigenvalues[:, 0]
    largest = eigenvalues[:, -1]
    condition = largest / np.maximum(smallest, np.finfo(np.float64).tiny)
    return (
        ~np.isfinite(metric).all(axis=(1, 2))
        | ~np.isfinite(determinant)
        | (determinant <= determinant_floor)
        | (smallest <= eigenvalue_floor)
        | (condition >= condition_limit)
    )


def probe_gradient(parameters):
    """Gradient of the scalar probe used for both Laplace evaluations."""
    tensor_input = isinstance(parameters, AbstractTensor)
    parameters = (
        parameters
        if tensor_input
        else AbstractTensor.tensor(parameters, dtype="float64")
    )
    u, v, w = parameters[:, 0], parameters[:, 1], parameters[:, 2]
    tau = 2.0 * np.pi
    result = AbstractTensor.stack(
        (
            tau * (tau * u).cos() * (tau * v).cos() + 0.1 * v,
            -tau * (tau * u).sin() * (tau * v).sin() + 0.1 * u,
            0.75 * np.pi * (3.0 * np.pi * w).cos(),
        ),
        dim=1,
    )
    return result if tensor_input else np.asarray(result.tolist())


@dataclass(frozen=True)
class BoundaryResolution:
    """Experimental adapter for ``laplace_nd``'s six-face convention."""

    bounds: np.ndarray
    boundary_conditions: tuple[str, str, str, str, str, str]
    grid_boundaries: tuple[bool, bool, bool, bool, bool, bool]

    def __post_init__(self) -> None:
        bounds = np.asarray(self.bounds, dtype=np.float64)
        if bounds.shape != (3, 2):
            raise ValueError("bounds must have shape (3, 2)")
        if len(self.boundary_conditions) != 6 or len(self.grid_boundaries) != 6:
            raise ValueError("boundary tuples follow (u-, u+, v-, v+, w-, w+)")
        supported = {"dirichlet", "neumann", "periodic"}
        unknown = set(self.boundary_conditions) - supported
        if unknown:
            raise ValueError(f"unsupported boundary conditions: {sorted(unknown)}")
        object.__setattr__(self, "bounds", bounds)

    def contact_mask(self, parameters: np.ndarray, tolerance: float) -> np.ndarray:
        contacts = np.zeros((len(parameters), 6), dtype=bool)
        for axis in range(3):
            contacts[:, 2 * axis] = (
                self.grid_boundaries[2 * axis]
                & (parameters[:, axis] <= self.bounds[axis, 0] + tolerance)
            )
            contacts[:, 2 * axis + 1] = (
                self.grid_boundaries[2 * axis + 1]
                & (parameters[:, axis] >= self.bounds[axis, 1] - tolerance)
            )
        return contacts


def laplace_beltrami(
    parameters: np.ndarray,
    metric_function,
    *,
    step: float = 2e-5,
    boundary_resolution: BoundaryResolution | None = None,
    return_boundary_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compatibility adapter for the rank-N AbstractTensor implementation."""
    parameters = np.asarray(parameters, dtype=np.float64)

    def tensor_metric(query):
        host_query = np.asarray(query.tolist(), dtype=np.float64)
        return AbstractTensor.tensor(
            metric_function(host_query),
            dtype=query.get_dtype(),
            device=query.get_device(),
        )

    # The shared operator owns all metric arithmetic.  Boundary samples are
    # moved one stencil step inward so its centered stencil never leaves the
    # declared chart.
    evaluation_points = parameters.copy()
    contacts = (
        np.zeros((len(parameters), 6), dtype=bool)
        if boundary_resolution is None
        else boundary_resolution.contact_mask(parameters, step * 1.1)
    )
    if boundary_resolution is not None:
        for axis in range(parameters.shape[1]):
            low, high = boundary_resolution.bounds[axis]
            evaluation_points[contacts[:, 2 * axis], axis] = low + step
            evaluation_points[contacts[:, 2 * axis + 1], axis] = high - step
    result = continuous_laplace_beltrami(
        AbstractTensor.tensor(evaluation_points, dtype="float64"),
        tensor_metric,
        probe_gradient,
        step=step,
    )
    result = np.asarray(result.tolist(), dtype=np.float64)
    boundary_mask = np.any(contacts, axis=1)
    if return_boundary_mask:
        return result, boundary_mask
    return result


def _sample_simplex(
    vertices: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    return rng.dirichlet(np.ones(len(vertices)), size=count) @ vertices


def _surface_field(points: AbstractTensor) -> AbstractTensor:
    xyz = np.asarray(points.tolist(), dtype=np.float64)
    x, y, z = xyz.transpose(2, 0, 1)
    target = 0.5 + 0.12 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    return AbstractTensor.get_tensor(z - target)


def build_metric_roundtrip(
    resolution: int = 4,
    samples_per_patch: int = 10,
    fifo_batches: int = 16,
    *,
    resolve_boundaries: bool = False,
    boundary_condition: str = "dirichlet",
    display_geometry: str = "source",
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Run the complete extraction/reconstruction/geometric comparison."""
    domain = GridDomain.generate_grid_domain(
        "rectangular",
        N_u=resolution + 1,
        N_v=resolution + 1,
        N_w=resolution + 1,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        defer_resolution=True,
    )
    identity = compile_grid_domain(domain)
    expanded_vertices = detailed_embedding(identity.parametric.reshape(-1, 3))
    embedded = expanded_vertices[:, :3].reshape(identity.parametric.shape)
    compiled = DomainTetrahedra(identity.parametric, embedded)
    extraction = extract_isosurface(
        compiled.embedded,
        _surface_field,
        parametric_tetrahedra=compiled.parametric,
    )
    samples = extraction.solver_samples
    assert samples is not None and samples.parametric_points is not None
    source_tags = metric_sample_tags(
        samples.parametric_points,
        samples.tetrahedron_ids,
        induced_metric(samples.parametric_points),
        source="detailed-embedding",
    )
    extraction = replace(
        extraction,
        solver_samples=replace(samples, metric_tags=source_tags),
    )

    active_patch_ids = np.unique(samples.tetrahedron_ids)
    simplices = {
        int(patch_id): compiled.parametric[patch_id]
        for patch_id in active_patch_ids
    }
    rng = np.random.default_rng(20260725)
    support_parameters = []
    support_ids = []
    for patch_id, vertices in simplices.items():
        support_parameters.append(
            _sample_simplex(vertices, samples_per_patch, rng)
        )
        support_ids.append(
            np.full(samples_per_patch, patch_id, dtype=np.int64)
        )
    parameters = np.concatenate(
        (samples.parametric_points, np.concatenate(support_parameters))
    )
    patch_ids = np.concatenate(
        (samples.tetrahedron_ids, np.concatenate(support_ids))
    )
    values = detailed_embedding(parameters)
    jacobians = detailed_jacobian(parameters)
    engine = StreamingPiecewiseSplineEngine(
        simplices, degree=3, derivative_weight=0.25, ridge=1e-13
    )
    for rows in np.array_split(np.arange(len(parameters)), fifo_batches):
        engine.submit(
            patch_ids[rows], parameters[rows], values[rows], jacobians[rows]
        )
    generation = engine.update()
    assert generation is not None

    query_triangles = extraction.parametric_triangles
    triangle_patch_ids = extraction.triangle_tetrahedron_ids
    assert query_triangles is not None and triangle_patch_ids is not None
    query = query_triangles.reshape(-1, 3)
    query_patch_ids = np.repeat(triangle_patch_ids, 3)
    source_values = detailed_embedding(query)
    spline_values = np.empty_like(source_values)
    spline_metric = np.empty((len(query), 3, 3), dtype=np.float64)
    source_laplace = np.empty(len(query), dtype=np.float64)
    spline_laplace = np.empty(len(query), dtype=np.float64)
    boundary_vertices = np.zeros(len(query), dtype=bool)
    boundary_resolution = None
    if resolve_boundaries:
        parametric = compiled.parametric.reshape(-1, 3)
        boundary_resolution = BoundaryResolution(
            bounds=np.stack((parametric.min(axis=0), parametric.max(axis=0)), axis=1),
            boundary_conditions=(boundary_condition,) * 6,
            grid_boundaries=tuple(bool(value) for value in domain.grid_boundaries),
        )
    for patch_id in np.unique(query_patch_ids):
        mask = query_patch_ids == patch_id
        patch = generation.patches[int(patch_id)]
        spline_values[mask] = patch.evaluate(query[mask])
        spline_metric[mask] = patch.metric_tensor(query[mask])
        source_result = laplace_beltrami(
            query[mask],
            induced_metric,
            boundary_resolution=boundary_resolution,
            return_boundary_mask=True,
        )
        spline_result = laplace_beltrami(
            query[mask],
            patch.metric_tensor,
            boundary_resolution=boundary_resolution,
            return_boundary_mask=True,
        )
        source_laplace[mask], boundary_vertices[mask] = source_result
        spline_laplace[mask], _ = spline_result
    spline_tags = metric_sample_tags(
        query,
        query_patch_ids,
        spline_metric,
        generation=generation.generation,
        source="fifo-piecewise-spline",
    )
    metric_error = np.linalg.norm(
        spline_tags.metric - induced_metric(query), axis=(1, 2)
    )
    source_singular = singular_metric_mask(induced_metric(query))
    spline_singular = singular_metric_mask(spline_tags.metric)
    embedding_error = np.linalg.norm(spline_values - source_values, axis=1)
    laplace_difference = spline_laplace - source_laplace
    triangle_difference = laplace_difference.reshape(-1, 3).mean(axis=1)
    interior = ~boundary_vertices
    boundary = boundary_vertices

    def _rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(values**2))) if len(values) else float("nan")

    summary = pd.DataFrame([{
        "grid_resolution": resolution,
        "tetrahedral_patches": len(simplices),
        "surface_triangles": extraction.triangle_count,
        "youngman_crossings": samples.sample_count,
        "fifo_batches_consumed": fifo_batches,
        "fifo_batches_pending": engine.pending_batches,
        "spline_generation": generation.generation,
        "control_points": generation.control_point_count,
        "embedding_dimension": source_values.shape[1],
        "metric_matrix_shape": "3x3",
        "boundary_resolution": (
            f"experimental-{boundary_condition}" if resolve_boundaries else "off"
        ),
        "boundary_vertices": int(boundary_vertices.sum()),
        "source_singular_vertices": int(source_singular.sum()),
        "spline_singular_vertices": int(spline_singular.sum()),
        "mean_embedding_error": float(embedding_error.mean()),
        "max_embedding_error": float(embedding_error.max()),
        "mean_metric_error": float(metric_error.mean()),
        "max_metric_error": float(metric_error.max()),
        "laplace_difference_rms": float(np.sqrt(np.mean(laplace_difference**2))),
        "laplace_difference_max_abs": float(np.max(np.abs(laplace_difference))),
        "source_laplace_rms": _rms(source_laplace),
        "interior_laplace_difference_rms": _rms(laplace_difference[interior]),
        "boundary_laplace_difference_rms": _rms(laplace_difference[boundary]),
    }])
    triangle_report = pd.DataFrame({
        "triangle": np.arange(extraction.triangle_count),
        "tetrahedron": triangle_patch_ids,
        "source_laplace": source_laplace.reshape(-1, 3).mean(axis=1),
        "spline_laplace": spline_laplace.reshape(-1, 3).mean(axis=1),
        "laplace_difference": triangle_difference,
        "touches_domain_boundary": boundary_vertices.reshape(-1, 3).any(axis=1),
        "contains_singularity": (
            source_singular | spline_singular
        ).reshape(-1, 3).any(axis=1),
    })
    if display_geometry not in {"source", "spline"}:
        raise ValueError("display_geometry must be 'source' or 'spline'")
    display = (
        extraction
        if display_geometry == "source"
        else replace(
            extraction,
            triangles=spline_values[:, :3].reshape(-1, 3, 3),
        )
    )
    return summary, triangle_report, display


def _load_pluck_viewer():
    root = Path(__file__).resolve().parents[5]
    pluck = root / "spectral-analyzer"
    if str(pluck) not in sys.path:
        sys.path.insert(0, str(pluck))
    import ordinary_gl_mesh_viewer
    return ordinary_gl_mesh_viewer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--samples-per-patch", type=int, default=10)
    parser.add_argument("--fifo-batches", type=int, default=16)
    parser.add_argument("--view", action="store_true")
    parser.add_argument(
        "--render-image",
        type=Path,
        help="write a deterministic headless PNG through Pluck's mesh adapter",
    )
    parser.add_argument(
        "--display-geometry",
        choices=("source", "spline"),
        default="source",
        help="carrier surface for the error colors; spline exposes patch seams",
    )
    parser.add_argument(
        "--resolve-boundaries",
        action="store_true",
        help="use experimental laplace_nd-compatible face handling",
    )
    parser.add_argument(
        "--boundary-condition",
        choices=("dirichlet", "neumann"),
        default="dirichlet",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    summary, triangles, display = build_metric_roundtrip(
        args.resolution,
        args.samples_per_patch,
        args.fifo_batches,
        resolve_boundaries=args.resolve_boundaries,
        boundary_condition=args.boundary_condition,
        display_geometry=args.display_geometry,
    )
    print("\nMETRIC ROUND-TRIP\n", summary.to_string(index=False))
    print("\nLAPLACE SAMPLE\n", triangles.head(12).to_string(index=False))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_dir / "metric_roundtrip_summary.csv", index=False)
        triangles.to_csv(args.output_dir / "laplace_difference.csv", index=False)
    if args.view:
        _load_pluck_viewer().view_triangle_mesh(
            display.triangles,
            triangle_values=triangles["laplace_difference"].to_numpy(),
            value_label="spline minus source Laplace-Beltrami",
            title="YoungMan metric round-trip: Laplace-Beltrami difference",
        )
    if args.render_image:
        output = _load_pluck_viewer().render_triangle_mesh_image(
            display.triangles,
            args.render_image,
            triangle_values=triangles["laplace_difference"].to_numpy(),
            value_label="spline minus source Laplace-Beltrami",
            title="YoungMan metric round-trip",
        )
        print(f"\nHEADLESS IMAGE\n {output}")


if __name__ == "__main__":
    main()
