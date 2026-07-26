"""Truthful source -> YoungMan -> spline -> mesh -> Laplace round trip."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from ..abstract_convolution.laplace_nd import GridDomain
from ..riemann import AdaptiveSurfaceTriangulator, TriangulationTolerance
from .algorithm import DomainTetrahedra, compile_grid_domain, extract_isosurface
from .metric_roundtrip_demo import (
    _surface_field,
    detailed_embedding,
    detailed_jacobian,
)
from .spline import StreamingSplineSolver


@dataclass(frozen=True)
class BlackBoxRoundTrip:
    summary: pd.DataFrame
    triangles: pd.DataFrame
    mesh: object


@dataclass(frozen=True)
class PublishedSurfaceSpline:
    """The source-free callable handed across the triangulation boundary."""

    model: object

    def __call__(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64)
        return self.model(np.column_stack((uv, np.zeros(len(uv)))))

    def jacobian(self, uv: np.ndarray) -> np.ndarray:
        return _finite_jacobian(self, uv)


def source_surface_parameters(uv: np.ndarray) -> np.ndarray:
    """Exact source chart, used only by YoungMan/reference measurements."""
    u, v = np.asarray(uv, dtype=np.float64).T
    tau = 2.0 * np.pi
    target_z = 0.5 + 0.12 * np.sin(tau * u) * np.cos(tau * v)
    visible_warp = 0.075 * np.sin(tau * u) * np.sin(tau * v)
    return np.stack((u, v, target_z - visible_warp), axis=1)


def source_surface(uv: np.ndarray) -> np.ndarray:
    return detailed_embedding(source_surface_parameters(uv))


def source_surface_jacobian(uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64)
    u, v = uv.T
    tau = 2.0 * np.pi
    parameter_jacobian = np.zeros((len(uv), 3, 2), dtype=np.float64)
    parameter_jacobian[:, 0, 0] = 1.0
    parameter_jacobian[:, 1, 1] = 1.0
    parameter_jacobian[:, 2, 0] = (
        0.12 * tau * np.cos(tau * u) * np.cos(tau * v)
        - 0.075 * tau * np.cos(tau * u) * np.sin(tau * v)
    )
    parameter_jacobian[:, 2, 1] = (
        -0.12 * tau * np.sin(tau * u) * np.sin(tau * v)
        - 0.075 * tau * np.sin(tau * u) * np.cos(tau * v)
    )
    return np.einsum(
        "nmi,nij->nmj",
        detailed_jacobian(source_surface_parameters(uv)),
        parameter_jacobian,
    )


def _finite_jacobian(function, uv: np.ndarray, step: float = 2e-5) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64)
    center = np.asarray(function(uv), dtype=np.float64)
    jacobian = np.empty((len(uv), center.shape[1], 2), dtype=np.float64)
    for axis in range(2):
        offset = np.zeros_like(uv)
        offset[:, axis] = step
        jacobian[:, :, axis] = (
            function(uv + offset) - function(uv - offset)
        ) / (2.0 * step)
    return jacobian


def _metric(jacobian: np.ndarray) -> np.ndarray:
    return np.einsum("nmi,nmj->nij", jacobian, jacobian)


def probe_values(uv: np.ndarray) -> np.ndarray:
    u, v = np.asarray(uv, dtype=np.float64).T
    tau = 2.0 * np.pi
    return np.sin(tau * u) * np.cos(tau * v) + 0.1 * u * v


def probe_gradient(uv: np.ndarray) -> np.ndarray:
    u, v = np.asarray(uv, dtype=np.float64).T
    tau = 2.0 * np.pi
    return np.stack(
        (
            tau * np.cos(tau * u) * np.cos(tau * v) + 0.1 * v,
            -tau * np.sin(tau * u) * np.sin(tau * v) + 0.1 * u,
        ),
        axis=1,
    )


def continuous_surface_laplace(
    uv: np.ndarray, metric_function, step: float = 2e-5
) -> np.ndarray:
    """Numerically evaluate the two-dimensional Laplace--Beltrami operator."""
    uv = np.asarray(uv, dtype=np.float64)

    def flux(query):
        metric = metric_function(query)
        root_det = np.sqrt(np.linalg.det(metric))
        vector = np.einsum(
            "nij,nj->ni", np.linalg.inv(metric), probe_gradient(query)
        )
        return root_det[:, None] * vector, root_det

    _, root_det = flux(uv)
    divergence = np.zeros(len(uv), dtype=np.float64)
    for axis in range(2):
        offset = np.zeros_like(uv)
        offset[:, axis] = step
        plus, _ = flux(uv + offset)
        minus, _ = flux(uv - offset)
        divergence += (plus[:, axis] - minus[:, axis]) / (2.0 * step)
    return divergence / root_det


def _domain_and_extraction(resolution: int):
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
    expanded = detailed_embedding(identity.parametric.reshape(-1, 3))
    compiled = DomainTetrahedra(
        identity.parametric,
        expanded[:, :3].reshape(identity.parametric.shape),
    )
    extraction = extract_isosurface(
        compiled.embedded,
        _surface_field,
        parametric_tetrahedra=compiled.parametric,
    )
    return domain, extraction


def publish_surface_spline(samples) -> tuple[PublishedSurfaceSpline, int]:
    """Use source access once, then publish a callable containing no source."""
    source_controls = detailed_embedding(samples.parametric_points)
    fifo = StreamingSplineSolver(
        intrinsic_axes=(0, 1),
        smoothing=2e-8,
        kernel="thin_plate_spline",
        neighbors=None,
    )
    for rows in np.array_split(np.arange(samples.sample_count), 12):
        fifo.submit(samples.parametric_points[rows], source_controls[rows])
    model = fifo.update()
    assert model is not None
    return PublishedSurfaceSpline(model), fifo.control_point_count


def build_blackbox_roundtrip(
    youngman_resolution: int = 7,
    position_tolerance: float = 2e-3,
    tangent_tolerance: float = 3.5e-1,
) -> BlackBoxRoundTrip:
    """Build every stage while enforcing the spline/triangulator black box."""
    _, extraction = _domain_and_extraction(youngman_resolution)
    samples = extraction.solver_samples
    assert samples is not None and samples.parametric_points is not None

    spline_surface, control_point_count = publish_surface_spline(samples)

    triangulator = AdaptiveSurfaceTriangulator(
        spline_surface,
        jacobian=spline_surface.jacobian,
        tolerance=TriangulationTolerance(
            position=position_tolerance,
            tangent=tangent_tolerance,
            max_rounds=9,
            max_triangles=80_000,
        ),
        initial_resolution=(4, 4),
    )
    mesh = triangulator.triangulate()

    uv = mesh.parameters
    source_values = source_surface(uv)
    spline_values = mesh.embedded
    source_jacobian = source_surface_jacobian(uv)
    spline_jacobian_values = spline_surface.jacobian(uv)
    source_metric = _metric(source_jacobian)
    spline_metric = _metric(spline_jacobian_values)
    source_continuous_laplace = continuous_surface_laplace(
        uv, lambda query: _metric(source_surface_jacobian(query))
    )
    spline_continuous_laplace = continuous_surface_laplace(
        uv, lambda query: _metric(spline_surface.jacobian(query))
    )

    from ..riemann.mesh_laplace import mesh_laplace_beltrami

    mesh_result = mesh_laplace_beltrami(
        mesh.embedded, mesh.triangles, probe_values(uv)
    )
    mesh_laplace = mesh_result.laplacian
    boundary = mesh_result.geometry.boundary_vertex_mask
    interior = (
        ~boundary
        & ~mesh_result.geometry.degenerate_vertex_mask
        & ~mesh_result.geometry.singular_vertex_mask
    )

    youngman_error = np.linalg.norm(
        samples.embedded_points
        - detailed_embedding(samples.parametric_points)[:, :3],
        axis=1,
    )
    spline_error = np.linalg.norm(spline_values - source_values, axis=1)
    metric_error = np.linalg.norm(spline_metric - source_metric, axis=(1, 2))
    continuous_laplace_error = (
        spline_continuous_laplace - source_continuous_laplace
    )
    mesh_discretization_error = mesh_laplace - spline_continuous_laplace
    mesh_laplace_error = mesh_laplace - source_continuous_laplace

    def rms(values):
        return float(np.sqrt(np.mean(np.square(values)))) if len(values) else np.nan

    summary = pd.DataFrame([{
        "youngman_resolution": youngman_resolution,
        "youngman_samples": samples.sample_count,
        "spline_controls": control_point_count,
        "spline_embedding_dimension": spline_surface.model.embedding_dimension,
        "triangulation_generation": mesh.generation,
        "mesh_vertices": len(mesh.parameters),
        "mesh_triangles": mesh.triangle_count,
        "mesh_converged": mesh.converged,
        "mesh_function_evaluations": mesh.function_evaluations,
        "youngman_error_rms": rms(youngman_error),
        "spline_position_error_rms": rms(spline_error),
        "spline_metric_error_rms": rms(metric_error),
        "triangulator_max_chord_error": float(mesh.position_error.max()),
        "triangulator_max_tangent_error": (
            float(mesh.tangent_error.max()) if mesh.tangent_error is not None else np.nan
        ),
        "continuous_spline_laplace_error_rms_interior": rms(
            continuous_laplace_error[interior]
        ),
        "mesh_discretization_error_rms_interior": rms(
            mesh_discretization_error[interior]
        ),
        "source_laplace_rms_interior": rms(
            source_continuous_laplace[interior]
        ),
        "mesh_laplace_error_rms_interior": rms(mesh_laplace_error[interior]),
        "degenerate_mesh_vertices": int(
            mesh_result.geometry.degenerate_vertex_mask.sum()
        ),
    }])
    triangle_position = spline_error[mesh.triangles].mean(axis=1)
    triangle_chord = mesh.position_error
    triangle_metric = metric_error[mesh.triangles].mean(axis=1)
    triangle_laplace = np.nanmean(
        np.where(interior, mesh_laplace_error, np.nan)[mesh.triangles], axis=1
    )
    triangle_discretization = np.nanmean(
        np.where(interior, mesh_discretization_error, np.nan)[mesh.triangles],
        axis=1,
    )
    triangle_report = pd.DataFrame({
        "triangle": np.arange(mesh.triangle_count),
        "spline_position_error": triangle_position,
        "triangulation_chord_error": triangle_chord,
        "spline_metric_error": triangle_metric,
        "mesh_laplace_error": triangle_laplace,
        "mesh_discretization_error": triangle_discretization,
        "touches_boundary": boundary[mesh.triangles].any(axis=1),
    })
    return BlackBoxRoundTrip(summary, triangle_report, mesh)


def _load_pluck_viewer():
    root = Path(__file__).resolve().parents[5]
    pluck = root / "spectral-analyzer"
    if str(pluck) not in sys.path:
        sys.path.insert(0, str(pluck))
    import ordinary_gl_mesh_viewer
    return ordinary_gl_mesh_viewer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--youngman-resolution", type=int, default=7)
    parser.add_argument("--position-tolerance", type=float, default=2e-3)
    parser.add_argument("--tangent-tolerance", type=float, default=3.5e-1)
    parser.add_argument("--render-image", type=Path)
    parser.add_argument(
        "--error-field",
        choices=("youngman", "spline", "triangulation", "metric", "laplace"),
        default="laplace",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = build_blackbox_roundtrip(
        args.youngman_resolution,
        args.position_tolerance,
        args.tangent_tolerance,
    )
    print("\nBLACK-BOX ROUND TRIP\n", result.summary.to_string(index=False))
    print("\nTRIANGLE CERTIFICATES\n", result.triangles.head(12).to_string(index=False))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result.summary.to_csv(args.output_dir / "summary.csv", index=False)
        result.triangles.to_csv(args.output_dir / "triangles.csv", index=False)
    if args.render_image:
        fields = {
            "youngman": np.full(result.mesh.triangle_count,
                                result.summary.loc[0, "youngman_error_rms"]),
            "spline": result.triangles["spline_position_error"].to_numpy(),
            "triangulation": result.triangles[
                "triangulation_chord_error"
            ].to_numpy(),
            "metric": result.triangles["spline_metric_error"].to_numpy(),
            "laplace": result.triangles["mesh_laplace_error"].to_numpy(),
        }
        output = _load_pluck_viewer().render_triangle_mesh_image(
            result.mesh.triangle_soup,
            args.render_image,
            triangle_values=fields[args.error_field],
            value_label=f"{args.error_field} stage error",
            title="Black-box geometry round trip",
        )
        print(f"\nHEADLESS IMAGE\n {output}")


if __name__ == "__main__":
    main()
