"""YoungMan -> FIFO spline -> metric/Laplace round-trip demonstration."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from ..abstraction import AbstractTensor
from ..abstract_convolution.laplace_nd import GridDomain
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


def probe_gradient(parameters: np.ndarray) -> np.ndarray:
    """Gradient of the scalar probe used for both Laplace evaluations."""
    u, v, w = np.asarray(parameters, dtype=np.float64).T
    tau = 2.0 * np.pi
    return np.stack(
        (
            tau * np.cos(tau * u) * np.cos(tau * v) + 0.1 * v,
            -tau * np.sin(tau * u) * np.sin(tau * v) + 0.1 * u,
            0.75 * np.pi * np.cos(3.0 * np.pi * w),
        ),
        axis=1,
    )


def laplace_beltrami(
    parameters: np.ndarray,
    metric_function,
    *,
    step: float = 2e-5,
) -> np.ndarray:
    """Evaluate div(sqrt(det(g)) g^-1 grad(phi))/sqrt(det(g))."""
    parameters = np.asarray(parameters, dtype=np.float64)
    center_metric = metric_function(parameters)
    center_root_det = np.sqrt(np.linalg.det(center_metric))
    divergence = np.zeros(len(parameters), dtype=np.float64)
    for axis in range(parameters.shape[1]):
        offset = np.zeros_like(parameters)
        offset[:, axis] = step
        flux = []
        for query in (parameters + offset, parameters - offset):
            metric = metric_function(query)
            root_det = np.sqrt(np.linalg.det(metric))
            inverse = np.linalg.inv(metric)
            vector = np.einsum("nij,nj->ni", inverse, probe_gradient(query))
            flux.append(root_det[:, None] * vector)
        divergence += (flux[0][:, axis] - flux[1][:, axis]) / (2.0 * step)
    return divergence / center_root_det


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
    for patch_id in np.unique(query_patch_ids):
        mask = query_patch_ids == patch_id
        patch = generation.patches[int(patch_id)]
        spline_values[mask] = patch.evaluate(query[mask])
        spline_metric[mask] = patch.metric_tensor(query[mask])
        source_laplace[mask] = laplace_beltrami(
            query[mask], induced_metric
        )
        spline_laplace[mask] = laplace_beltrami(
            query[mask], patch.metric_tensor
        )
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
    embedding_error = np.linalg.norm(spline_values - source_values, axis=1)
    laplace_difference = spline_laplace - source_laplace
    triangle_difference = laplace_difference.reshape(-1, 3).mean(axis=1)

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
        "mean_embedding_error": float(embedding_error.mean()),
        "max_embedding_error": float(embedding_error.max()),
        "mean_metric_error": float(metric_error.mean()),
        "max_metric_error": float(metric_error.max()),
        "laplace_difference_rms": float(np.sqrt(np.mean(laplace_difference**2))),
        "laplace_difference_max_abs": float(np.max(np.abs(laplace_difference))),
    }])
    triangle_report = pd.DataFrame({
        "triangle": np.arange(extraction.triangle_count),
        "tetrahedron": triangle_patch_ids,
        "source_laplace": source_laplace.reshape(-1, 3).mean(axis=1),
        "spline_laplace": spline_laplace.reshape(-1, 3).mean(axis=1),
        "laplace_difference": triangle_difference,
    })
    display = replace(
        extraction,
        triangles=spline_values[:, :3].reshape(-1, 3, 3),
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
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    summary, triangles, display = build_metric_roundtrip(
        args.resolution, args.samples_per_patch, args.fifo_batches
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


if __name__ == "__main__":
    main()
