"""Numeric YoungMan/AbstractTensor demo with tabular state and timing reports."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..abstract_convolution.laplace_nd import GridDomain, RectangularTransform
from .algorithm import (
    compile_grid_domain,
    extract_isosurface,
    tetrahedra_from_grid_domain,
    triangle_areas,
)
from .spline import StreamingSplineSolver


class _WarpedDemoTransform(RectangularTransform):
    """Smooth 3D embedding used as the spline reconstruction target."""

    def transform_spatial(self, u, v, w):
        scale = np.pi / self.Lx
        x = u + 0.08 * (scale * v).sin() * (scale * w).sin()
        y = v + 0.06 * (scale * u).sin() * (scale * w).sin()
        z = w + 0.12 * (scale * u).sin() * (scale * v).cos()
        return x, y, z


def _resolve_numpy(domain, parameters: np.ndarray) -> np.ndarray:
    """Resolve a bulk ``(N, 3)`` parameter table through the domain transform."""
    resolved = domain.resolve_positions(
        *[
            domain.U.__class__.get_tensor(parameters[:, axis])
            for axis in range(parameters.shape[1])
        ],
        full_geometry=False,
    )
    return np.stack(
        [np.asarray(component.tolist(), dtype=np.float64) for component in resolved],
        axis=-1,
    )


def build_spline_report(resolution: int = 12):
    """Reconstruct a warped extracted surface from YoungMan's live solve points."""
    extent = 2.5
    domain = GridDomain.generate_grid_domain(
        "rectangular",
        N_u=resolution + 1,
        N_v=resolution + 1,
        N_w=resolution + 1,
        Lx=extent,
        Ly=extent,
        Lz=extent,
        defer_resolution=True,
    )
    domain.transform = _WarpedDemoTransform(
        Lx=extent,
        Ly=extent,
        Lz=extent,
        N_u=resolution + 1,
        N_v=resolution + 1,
        N_w=resolution + 1,
    )
    compiled = compile_grid_domain(domain)
    center = extent / 2

    def field(points):
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        surface = center + 0.22 * (
            (2 * np.pi / extent * x).sin()
            * (2 * np.pi / extent * y).cos()
        )
        return z - surface

    result = extract_isosurface(
        compiled.embedded,
        field,
        parametric_tetrahedra=compiled.parametric,
    )
    samples = result.solver_samples
    assert samples is not None and samples.parametric_points is not None

    # A surface that is a graph over (u, v) has intrinsic dimension two even
    # though its source domain and target embedding are both three-dimensional.
    # De-duplicate chart locations before a deterministic train/validation split.
    chart = samples.parametric_points[:, :2]
    _, unique_indices = np.unique(np.round(chart, 12), axis=0, return_index=True)
    unique_indices.sort()
    parameters = samples.parametric_points[unique_indices]
    solved_values = samples.embedded_points[unique_indices]
    validation_mask = np.arange(len(parameters)) % 5 == 0
    training_mask = ~validation_mask

    solver = StreamingSplineSolver(
        intrinsic_axes=(0, 1),
        smoothing=1e-9,
        neighbors=48,
    )
    # Multiple messages exercise the same FIFO boundary used by a live solver.
    for batch in np.array_split(np.flatnonzero(training_mask), 8):
        solver.submit(parameters[batch], solved_values[batch])
    model = solver.update()
    assert model is not None

    validation_parameters = parameters[validation_mask]
    target = _resolve_numpy(domain, validation_parameters)
    prediction = model(validation_parameters)
    spline_error = np.linalg.norm(prediction - target, axis=1)
    solver_linearization_error = np.linalg.norm(
        solved_values[validation_mask] - target, axis=1
    )
    return pd.DataFrame(
        [
            {
                "parameter_dimension": model.parameter_dimension,
                "intrinsic_dimension": model.intrinsic_dimension,
                "embedding_dimension": model.embedding_dimension,
                "exported_solver_samples": samples.sample_count,
                "unique_control_points": len(parameters),
                "training_control_points": int(training_mask.sum()),
                "validation_points": int(validation_mask.sum()),
                "fifo_batches_consumed": 8,
                "fifo_batches_pending": solver.pending_batches,
                "mean_solver_linearization_error": float(
                    solver_linearization_error.mean()
                ),
                "mean_spline_target_error": float(spline_error.mean()),
                "max_spline_target_error": float(spline_error.max()),
                "total_squared_spline_error": float(np.square(spline_error).sum()),
            }
        ]
    )


def _load_pluck_viewer():
    """Import Pluck's ordinary OpenGL adapter without coupling the core to it."""
    configured = os.environ.get("PLUCK_ROOT")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path(__file__).resolve().parents[5] / "spectral-analyzer")
    for candidate in candidates:
        if (candidate / "ordinary_gl_mesh_viewer.py").is_file():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            return importlib.import_module("ordinary_gl_mesh_viewer")
    raise RuntimeError(
        "Pluck ordinary OpenGL viewer not found; set PLUCK_ROOT to the "
        "spectral-analyzer repository"
    )


def _native_field_benchmark(
    tetrahedra: np.ndarray, repeats: int, center: float
) -> float:
    started = time.perf_counter()
    for _ in range(repeats):
        relative = tetrahedra - center
        values = np.sum(relative * relative, axis=-1) - 0.8**2
        starts = values[:, (0, 0, 0, 1, 1, 2)]
        ends = values[:, (1, 2, 3, 2, 3, 3)]
        _ = starts / np.where(np.abs(starts - ends) < 1e-12, 1.0, starts - ends)
    return (time.perf_counter() - started) / repeats


def build_reports(resolution: int = 12, repeats: int = 5):
    extent = 2.5
    center = extent / 2
    domain = GridDomain.generate_grid_domain(
        "rectangular",
        N_u=resolution + 1,
        N_v=resolution + 1,
        N_w=resolution + 1,
        Lx=extent,
        Ly=extent,
        Lz=extent,
        defer_resolution=True,
    )
    tetrahedra = tetrahedra_from_grid_domain(domain)

    def implicit_sphere(x, y, z):
        return (
            (x - center) ** 2
            + (y - center) ** 2
            + (z - center) ** 2
            - 0.8**2
        )

    def field(points):
        return domain.signed_difference(
            implicit_sphere,
            points[..., 0],
            points[..., 1],
            points[..., 2],
        )
    runs = []
    result = None
    for run in range(repeats):
        result = extract_isosurface(tetrahedra, field)
        runs.append({"run": run + 1, "backend": "AbstractTensor[numpy]",
                     "seconds": result.elapsed_seconds})
    assert result is not None

    native_seconds = _native_field_benchmark(tetrahedra, repeats, center)
    runs.append({"run": 0, "backend": "native numpy numeric slice",
                 "seconds": native_seconds})
    timing = pd.DataFrame(runs)
    timing["tetrahedra_per_second"] = len(tetrahedra) / timing["seconds"]

    active_counts = result.active_edges.sum(axis=1)
    state = (
        pd.DataFrame({"case_id": result.case_ids, "active_edges": active_counts})
        .value_counts(sort=False)
        .rename("tetrahedra")
        .reset_index()
        .sort_values(["active_edges", "case_id"])
    )

    areas = triangle_areas(result.triangles)
    vertices = result.triangles.reshape(-1, 3)
    radial_error = np.abs(np.linalg.norm(vertices - center, axis=1) - 0.8)
    summary = pd.DataFrame(
        [
            {
                "resolution": resolution,
                "domain_resolution": "deferred",
                "tetrahedra": len(tetrahedra),
                "surface_tetrahedra": int(np.count_nonzero(active_counts)),
                "triangles": result.triangle_count,
                "surface_area": float(areas.sum()),
                "analytical_area": float(4 * np.pi * 0.8**2),
                "area_relative_error": float(abs(areas.sum() - 4 * np.pi * 0.8**2)
                                             / (4 * np.pi * 0.8**2)),
                "mean_radial_error": float(radial_error.mean()),
                "max_radial_error": float(radial_error.max()),
            }
        ]
    )
    return summary, state, timing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--view", action="store_true",
        help="display the extracted mesh with Pluck's BaseGLRenderer",
    )
    parser.add_argument(
        "--view-frames", type=int,
        help="close the optional Pluck view after this many frames",
    )
    args = parser.parse_args()

    summary, state, timing = build_reports(args.resolution, args.repeats)
    spline = build_spline_report(args.resolution)
    print("\nSURFACE SUMMARY\n", summary.to_string(index=False))
    print("\nTETRAHEDRON STATE\n", state.to_string(index=False))
    print("\nPERFORMANCE\n", timing.to_string(index=False))
    print("\nSPLINE RECONSTRUCTION\n", spline.to_string(index=False))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_dir / "summary.csv", index=False)
        state.to_csv(args.output_dir / "state.csv", index=False)
        timing.to_csv(args.output_dir / "performance.csv", index=False)
        spline.to_csv(args.output_dir / "spline_reconstruction.csv", index=False)
    if args.view:
        # Recompute once so the tables remain a stable reporting API and the
        # optional presentation layer receives only the resulting triangle soup.
        extent = 2.5
        domain = GridDomain.generate_grid_domain(
            "rectangular",
            N_u=args.resolution + 1,
            N_v=args.resolution + 1,
            N_w=args.resolution + 1,
            Lx=extent,
            Ly=extent,
            Lz=extent,
            defer_resolution=True,
        )
        tetrahedra = tetrahedra_from_grid_domain(domain)
        center = extent / 2

        def implicit_sphere(x, y, z):
            return (
                (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2
                - 0.8**2
            )

        result = extract_isosurface(
            tetrahedra,
            lambda points: domain.signed_difference(
                implicit_sphere, points[..., 0], points[..., 1], points[..., 2]
            ),
        )
        viewer = _load_pluck_viewer()
        viewer.view_triangle_mesh(
            result.triangles,
            title="YoungMan / AbstractTensor — Pluck ordinary OpenGL",
            max_frames=args.view_frames,
        )


if __name__ == "__main__":
    main()
