"""Truthful source -> YoungMan -> spline -> mesh -> Laplace round trip."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from ..abstract_convolution.laplace_nd import GridDomain
from ..riemann import (
    AdaptiveSurfaceTriangulator,
    TriangulatedSurfaceTransform,
    TriangulationTolerance,
)
from .algorithm import DomainTetrahedra, compile_grid_domain, extract_isosurface
from .metric_roundtrip_demo import detailed_embedding, detailed_jacobian
from .spline import StreamingSplineSolver, validate_single_valued_chart


@dataclass(frozen=True)
class BlackBoxRoundTrip:
    summary: pd.DataFrame
    triangles: pd.DataFrame
    mesh: object
    geometry_transform: TriangulatedSurfaceTransform
    profile: pd.DataFrame


@dataclass(frozen=True)
class PublishedSurfaceSpline:
    """The source-free callable handed across the triangulation boundary."""

    model: object

    def __call__(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64)
        return self.model(np.column_stack((uv, np.zeros(len(uv)))))

    def jacobian(self, uv: np.ndarray) -> np.ndarray:
        return _finite_jacobian(self, uv)


def source_surface_parameters(
    uv: np.ndarray, time_value: float = 0.0
) -> np.ndarray:
    """Exact source chart, used only by YoungMan/reference measurements."""
    u, v = np.asarray(uv, dtype=np.float64).T
    tau = 2.0 * np.pi
    phase = tau * float(time_value)
    target_z = 0.5 + 0.12 * np.sin(tau * u + phase) * np.cos(tau * v)
    visible_warp = 0.075 * np.sin(tau * u + phase) * np.sin(tau * v)
    return np.stack((u, v, target_z - visible_warp), axis=1)


def manifold_embedding(
    parameters: np.ndarray,
    time_value: float = 0.0,
    manifold: str = "ripple",
) -> np.ndarray:
    """Embed the common parameter volume into selectable visible manifolds."""
    parameters = np.asarray(parameters, dtype=np.float64)
    result = detailed_embedding(parameters)
    if manifold == "ripple":
        return result
    u, v, _ = parameters.T
    phase = 2.0 * np.pi * float(time_value)
    height = result[:, 2] - 0.5
    if manifold == "banana":
        angle = 2.35 * (u - 0.5) + 0.2 * np.sin(2.0 * np.pi * v + phase)
        radius = 1.25 + 0.42 * (v - 0.5)
        result[:, 0] = radius * np.sin(angle)
        result[:, 1] = 1.35 * (v - 0.5)
        result[:, 2] = radius * np.cos(angle) - 1.25 + 0.7 * height
    elif manifold == "saddle":
        x = 1.45 * (u - 0.5)
        y = 1.45 * (v - 0.5)
        result[:, 0] = x
        result[:, 1] = y
        result[:, 2] = 0.52 * (x * x - y * y) + height
    elif manifold == "twisted_ribbon":
        x = 1.8 * (u - 0.5)
        across = 1.1 * (v - 0.5)
        angle = 2.0 * np.pi * (u + time_value)
        result[:, 0] = x
        result[:, 1] = across * np.cos(angle) - height * np.sin(angle)
        result[:, 2] = across * np.sin(angle) + height * np.cos(angle)
    else:
        raise ValueError(f"unknown manifold preset: {manifold}")
    return result


def source_surface(
    uv: np.ndarray,
    time_value: float = 0.0,
    manifold: str = "ripple",
) -> np.ndarray:
    return manifold_embedding(
        source_surface_parameters(uv, time_value), time_value, manifold
    )


def source_surface_jacobian(
    uv: np.ndarray,
    time_value: float = 0.0,
    manifold: str = "ripple",
) -> np.ndarray:
    if manifold != "ripple":
        return _finite_jacobian(
            lambda query: source_surface(query, time_value, manifold), uv
        )
    uv = np.asarray(uv, dtype=np.float64)
    u, v = uv.T
    tau = 2.0 * np.pi
    phase = tau * float(time_value)
    parameter_jacobian = np.zeros((len(uv), 3, 2), dtype=np.float64)
    parameter_jacobian[:, 0, 0] = 1.0
    parameter_jacobian[:, 1, 1] = 1.0
    parameter_jacobian[:, 2, 0] = (
        0.12 * tau * np.cos(tau * u + phase) * np.cos(tau * v)
        - 0.075 * tau * np.cos(tau * u + phase) * np.sin(tau * v)
    )
    parameter_jacobian[:, 2, 1] = (
        -0.12 * tau * np.sin(tau * u + phase) * np.sin(tau * v)
        - 0.075 * tau * np.sin(tau * u + phase) * np.cos(tau * v)
    )
    return np.einsum(
        "nmi,nij->nmj",
        detailed_jacobian(source_surface_parameters(uv, time_value)),
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


def _domain_and_extraction(
    resolution: int,
    time_value: float = 0.0,
    manifold: str = "ripple",
):
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
    phase = 2.0 * np.pi * float(time_value)

    def time_surface_field(points):
        xyz = np.asarray(points.tolist(), dtype=np.float64)
        x, y, z = xyz.transpose(2, 0, 1)
        target = 0.5 + 0.12 * np.sin(2.0 * np.pi * x + phase) * np.cos(
            2.0 * np.pi * y
        )
        return type(points).get_tensor(z - target)

    extraction = extract_isosurface(
        compiled.embedded,
        time_surface_field,
        parametric_tetrahedra=compiled.parametric,
        expanded_embedding=lambda points: manifold_embedding(
            points, time_value, manifold
        ),
    )
    return domain, extraction


def publish_surface_spline(samples) -> tuple[PublishedSurfaceSpline, int]:
    """Publish solely from values carried across the YoungMan boundary."""
    if samples.expanded_points is None:
        raise ValueError("YoungMan samples do not contain expanded geometry")
    validate_single_valued_chart(
        samples.parametric_points, intrinsic_axes=(0, 1), tolerance=1e-8
    )
    source_controls = np.asarray(samples.expanded_points, dtype=np.float64)
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
    position_tolerance: float = 1e-6,
    tangent_tolerance: float = 6e-1,
    *,
    max_rounds: int = 14,
    max_triangles: int = 250_000,
    time_value: float = 0.0,
    manifold: str = "ripple",
) -> BlackBoxRoundTrip:
    """Build every stage while enforcing the spline/triangulator black box."""
    profile_rows = []

    def finish_stage(name, started):
        profile_rows.append({
            "stage": name,
            "elapsed_sec": perf_counter() - started,
        })

    total_started = perf_counter()
    started = perf_counter()
    _, extraction = _domain_and_extraction(
        youngman_resolution, time_value, manifold
    )
    finish_stage("youngman_extract", started)
    samples = extraction.solver_samples
    assert samples is not None and samples.parametric_points is not None

    started = perf_counter()
    spline_surface, control_point_count = publish_surface_spline(samples)
    finish_stage("fifo_spline_fit", started)

    triangulator = AdaptiveSurfaceTriangulator(
        spline_surface,
        jacobian=spline_surface.jacobian,
        tolerance=TriangulationTolerance(
            position=position_tolerance,
            tangent=tangent_tolerance,
            max_rounds=max_rounds,
            max_triangles=max_triangles,
        ),
        initial_resolution=(4, 4),
    )
    started = perf_counter()
    mesh = triangulator.triangulate()
    finish_stage("adaptive_triangulation", started)

    started = perf_counter()
    uv = mesh.parameters
    source_values = source_surface(uv, time_value, manifold)
    spline_values = mesh.embedded
    source_jacobian = source_surface_jacobian(uv, time_value, manifold)
    spline_jacobian_values = spline_surface.jacobian(uv)
    source_metric = _metric(source_jacobian)
    spline_metric = _metric(spline_jacobian_values)
    source_continuous_laplace = continuous_surface_laplace(
        uv,
        lambda query: _metric(
            source_surface_jacobian(query, time_value, manifold)
        ),
    )
    spline_continuous_laplace = continuous_surface_laplace(
        uv, lambda query: _metric(spline_surface.jacobian(query))
    )
    finish_stage("continuous_reference", started)

    started = perf_counter()
    geometry_transform = TriangulatedSurfaceTransform.from_mesh(
        mesh.parameters, mesh.embedded, mesh.triangles
    )
    mesh_result = geometry_transform.laplace(probe_values(uv))
    mesh_laplace = mesh_result.laplacian
    finish_stage("mesh_transform_laplace", started)
    boundary = mesh_result.geometry.boundary_vertex_mask
    interior = (
        ~boundary
        & ~mesh_result.geometry.invalid_vertex_mask
    )

    started = perf_counter()
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

    vertex_weights = mesh_result.geometry.lumped_vertex_areas

    def weighted_rms(values, mask=None):
        values = np.asarray(values, dtype=np.float64)
        selected = np.ones(len(values), dtype=bool) if mask is None else mask
        selected &= np.isfinite(values)
        weights = vertex_weights[selected]
        if not len(weights) or weights.sum() <= 0.0:
            return np.nan
        return float(np.sqrt(np.sum(weights * values[selected] ** 2) / weights.sum()))

    summary = pd.DataFrame([{
        "time_value": time_value,
        "manifold": manifold,
        "target_epsilon": position_tolerance,
        "youngman_resolution": youngman_resolution,
        "youngman_samples": samples.sample_count,
        "spline_controls": control_point_count,
        "spline_embedding_dimension": spline_surface.model.embedding_dimension,
        "triangulation_generation": mesh.generation,
        "mesh_vertices": len(mesh.parameters),
        "mesh_triangles": mesh.triangle_count,
        "mesh_converged": mesh.converged,
        "mesh_surface_sample_rows": mesh.surface_sample_count,
        "mesh_jacobian_sample_rows": mesh.jacobian_sample_count,
        "youngman_error_rms": rms(youngman_error),
        "spline_position_error_area_rms": weighted_rms(spline_error),
        "spline_metric_error_area_rms": weighted_rms(metric_error),
        "triangulator_max_chord_error": float(mesh.position_error.max()),
        "epsilon_ratio": float(mesh.position_error.max() / position_tolerance),
        "epsilon_achieved": bool(
            float(mesh.position_error.max()) <= position_tolerance
        ),
        "triangulator_max_tangent_error": (
            float(mesh.tangent_error.max()) if mesh.tangent_error is not None else np.nan
        ),
        "continuous_spline_laplace_error_area_rms_interior": weighted_rms(
            continuous_laplace_error, interior
        ),
        "mesh_discretization_error_area_rms_interior": weighted_rms(
            mesh_discretization_error, interior
        ),
        "source_laplace_area_rms_interior": weighted_rms(
            source_continuous_laplace, interior
        ),
        "mesh_laplace_error_area_rms_interior": weighted_rms(
            mesh_laplace_error, interior
        ),
        "degenerate_mesh_vertices": int(
            mesh_result.geometry.degenerate_vertex_mask.sum()
        ),
        "nonmanifold_mesh_edges": int(
            mesh_result.geometry.nonmanifold_edge_mask.sum()
        ),
    }])
    triangle_position = spline_error[mesh.triangles].mean(axis=1)
    triangle_chord = mesh.position_error
    triangle_metric = metric_error[mesh.triangles].mean(axis=1)
    def triangle_interior_mean(values):
        gathered = np.where(interior, values, np.nan)[mesh.triangles]
        count = np.isfinite(gathered).sum(axis=1)
        total = np.nansum(gathered, axis=1)
        return np.divide(
            total, count, out=np.full(len(count), np.nan), where=count > 0
        )

    triangle_laplace = triangle_interior_mean(mesh_laplace_error)
    triangle_discretization = triangle_interior_mean(
        mesh_discretization_error
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
    finish_stage("error_reporting", started)
    profile_rows.append({
        "stage": "total",
        "elapsed_sec": perf_counter() - total_started,
    })
    return BlackBoxRoundTrip(
        summary,
        triangle_report,
        mesh,
        geometry_transform,
        pd.DataFrame(profile_rows),
    )


def _load_pluck_viewer():
    root = Path(__file__).resolve().parents[5]
    pluck = root / "spectral-analyzer"
    if str(pluck) not in sys.path:
        sys.path.insert(0, str(pluck))
    import ordinary_gl_mesh_viewer
    return ordinary_gl_mesh_viewer


def _triangle_field(result, name: str) -> np.ndarray | None:
    if name == "geometry":
        return None
    fields = {
        "youngman": np.full(
            result.mesh.triangle_count,
            result.summary.loc[0, "youngman_error_rms"],
        ),
        "spline": result.triangles["spline_position_error"].to_numpy(),
        "triangulation": result.triangles[
            "triangulation_chord_error"
        ].to_numpy(),
        "metric": result.triangles["spline_metric_error"].to_numpy(),
        "laplace": result.triangles["mesh_laplace_error"].to_numpy(),
    }
    return fields[name]


def _profile_mapping(result: BlackBoxRoundTrip) -> dict[str, float]:
    return dict(zip(result.profile["stage"], result.profile["elapsed_sec"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--youngman-resolution", type=int, default=7)
    parser.add_argument(
        "--target-epsilon",
        type=float,
        default=1e-6,
        help="target maximum positional certificate (default: 1e-6)",
    )
    parser.add_argument(
        "--position-tolerance",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--tangent-tolerance", type=float, default=6e-1)
    parser.add_argument("--max-rounds", type=int, default=14)
    parser.add_argument("--max-triangles", type=int, default=250_000)
    parser.add_argument("--time-value", type=float, default=0.0)
    parser.add_argument(
        "--manifold",
        choices=("ripple", "banana", "saddle", "twisted_ribbon"),
        default="banana",
    )
    parser.add_argument("--animation", type=Path)
    parser.add_argument("--animation-frames", type=int, default=8)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--live-solves", type=int)
    parser.add_argument("--live-period", type=float, default=8.0)
    parser.add_argument("--live-max-frames", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--render-image", type=Path)
    parser.add_argument(
        "--error-field",
        choices=(
            "geometry", "youngman", "spline", "triangulation", "metric",
            "laplace",
        ),
        default="geometry",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-unconverged",
        action="store_true",
        help="emit diagnostics even if a triangulation tolerance or budget fails",
    )
    args = parser.parse_args()
    target_epsilon = (
        args.target_epsilon
        if args.position_tolerance is None
        else args.position_tolerance
    )
    result = build_blackbox_roundtrip(
        args.youngman_resolution,
        target_epsilon,
        args.tangent_tolerance,
        max_rounds=args.max_rounds,
        max_triangles=args.max_triangles,
        time_value=args.time_value,
        manifold=args.manifold,
    )
    print("\nBLACK-BOX ROUND TRIP\n", result.summary.to_string(index=False))
    print("\nPROFILE\n", result.profile.to_string(index=False))
    print("\nTRIANGLE CERTIFICATES\n", result.triangles.head(12).to_string(index=False))
    if not result.mesh.converged and not args.allow_unconverged:
        raise RuntimeError(
            f"triangulation did not converge: {result.mesh.stopped_reason}; "
            "use --allow-unconverged for failure diagnostics"
        )
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result.summary.to_csv(args.output_dir / "summary.csv", index=False)
        result.triangles.to_csv(args.output_dir / "triangles.csv", index=False)
        result.profile.to_csv(args.output_dir / "profile.csv", index=False)
    if args.render_image:
        viewer = _load_pluck_viewer()
        profile = _profile_mapping(result)
        panel = viewer.rolling_profile_lines(
            profile, (profile,), time_value=args.time_value
        )
        output = viewer.render_triangle_mesh_image(
            result.mesh.triangle_soup,
            args.render_image,
            triangle_values=_triangle_field(result, args.error_field),
            value_label=f"{args.error_field} stage error",
            title="Black-box geometry round trip",
            side_panel_lines=panel,
        )
        print(f"\nHEADLESS IMAGE\n {output}")
    if args.animation:
        if args.animation_frames < 2:
            raise ValueError("--animation-frames must be at least 2")
        from PIL import Image

        viewer = _load_pluck_viewer()
        frame_root = args.animation.with_suffix("")
        frame_root.mkdir(parents=True, exist_ok=True)
        history = []
        images = []
        for frame, time_value in enumerate(
            np.linspace(0.0, 1.0, args.animation_frames, endpoint=False)
        ):
            animated = result if frame == 0 and args.time_value == 0.0 else (
                build_blackbox_roundtrip(
                    args.youngman_resolution,
                    target_epsilon,
                    args.tangent_tolerance,
                    max_rounds=args.max_rounds,
                    max_triangles=args.max_triangles,
                    time_value=float(time_value),
                    manifold=args.manifold,
                )
            )
            if not animated.mesh.converged and not args.allow_unconverged:
                raise RuntimeError(
                    f"animation frame {frame} at t={time_value:.4f} did not "
                    f"converge: {animated.mesh.stopped_reason}"
                )
            profile = _profile_mapping(animated)
            history.append(profile)
            panel = viewer.rolling_profile_lines(
                profile, history, time_value=float(time_value)
            )
            frame_path = frame_root / f"frame_{frame:03d}.png"
            viewer.render_triangle_mesh_image(
                animated.mesh.triangle_soup,
                frame_path,
                triangle_values=_triangle_field(animated, args.error_field),
                value_label=f"{args.error_field} stage error",
                title="Time-varying black-box solve",
                side_panel_lines=panel,
            )
            images.append(Image.open(frame_path).convert("RGB"))
        args.animation.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            args.animation,
            save_all=True,
            append_images=images[1:],
            duration=900,
            loop=0,
        )
        for image in images:
            image.close()
        print(f"\nPROFILED ANIMATION\n {args.animation.resolve()}")
    if args.live:
        viewer = _load_pluck_viewer()
        history = []

        def solve_live_frame(index, time_value):
            solved = build_blackbox_roundtrip(
                args.youngman_resolution,
                target_epsilon,
                args.tangent_tolerance,
                max_rounds=args.max_rounds,
                max_triangles=args.max_triangles,
                time_value=time_value,
                manifold=args.manifold,
            )
            profile = _profile_mapping(solved)
            history.append(profile)
            panel = viewer.rolling_profile_lines(
                profile, history, time_value=time_value
            )
            panel.extend((
                "",
                f"solve index        {index:8d}",
                f"manifold           {args.manifold:>8}",
                f"certified          {str(solved.mesh.converged):>8}",
                f"target epsilon     {target_epsilon:8.2e}",
                f"epsilon ratio      {solved.summary.loc[0, 'epsilon_ratio']:8.3f}",
                f"vertices           {len(solved.mesh.parameters):8d}",
                f"triangles          {solved.mesh.triangle_count:8d}",
            ))
            return viewer.LiveMeshFrame(
                solved.mesh.triangle_soup,
                _triangle_field(solved, args.error_field),
                panel,
                time_value,
            )

        viewer.view_profiled_triangle_mesh_stream(
            solve_live_frame,
            period_sec=args.live_period,
            max_solves=args.live_solves,
            max_frames=args.live_max_frames,
        )


if __name__ == "__main__":
    main()
