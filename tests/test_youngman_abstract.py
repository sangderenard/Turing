import numpy as np

from src.common.tensors.youngman.algorithm import (
    compile_grid_domain,
    extract_isosurface,
    sphere_field,
    tetrahedra_from_grid_domain,
    triangle_areas,
)
from src.common.tensors.abstract_convolution.laplace_nd import GridDomain
from src.common.tensors.youngman.demo import (
    _load_pluck_viewer,
    build_reports,
    build_spline_report,
)
from src.common.tensors.youngman.spline import StreamingSplineSolver


def test_sphere_extraction_is_finite_and_near_analytical_surface():
    domain = GridDomain.generate_grid_domain(
        "rectangular", N_u=9, N_v=9, N_w=9,
        Lx=2.5, Ly=2.5, Lz=2.5, defer_resolution=True,
    )
    assert domain.vertices is None
    tetrahedra = tetrahedra_from_grid_domain(domain)
    result = extract_isosurface(
        tetrahedra, lambda points: sphere_field(points, center=1.25)
    )
    assert result.triangle_count > 0
    assert np.isfinite(result.triangles).all()
    area = triangle_areas(result.triangles).sum()
    analytical = 4 * np.pi * 0.8**2
    assert abs(area - analytical) / analytical < 0.12


def test_grid_domain_signed_difference_resolves_only_query_positions():
    domain = GridDomain.generate_grid_domain(
        "rectangular", N_u=2, N_v=2, N_w=2,
        Lx=2.0, Ly=2.0, Lz=2.0, defer_resolution=True,
    )
    query = domain.U * 0 + 1.0
    difference = domain.signed_difference(
        lambda x, y, z: x + y + z, query, query, query, iso_value=2.5
    )
    assert np.allclose(np.asarray(difference), 0.5)
    assert domain.vertices is None


def test_demo_reports_topology_accuracy_and_performance():
    summary, state, timing = build_reports(resolution=5, repeats=1)
    assert summary.loc[0, "triangles"] > 0
    assert summary.loc[0, "mean_radial_error"] < 0.08
    assert set(state["active_edges"]).issubset({0, 3, 4})
    assert {"AbstractTensor[numpy]", "native numpy numeric slice"} == set(
        timing["backend"]
    )


def test_demo_finds_pluck_ordinary_gl_adapter():
    viewer = _load_pluck_viewer()
    assert callable(viewer.view_triangle_mesh)


def test_extraction_bulk_exports_parametric_solver_movements():
    domain = GridDomain.generate_grid_domain(
        "rectangular", N_u=5, N_v=5, N_w=5,
        Lx=2.5, Ly=2.5, Lz=2.5, defer_resolution=True,
    )
    compiled = compile_grid_domain(domain)
    result = extract_isosurface(
        compiled.embedded,
        lambda points: sphere_field(points, center=1.25),
        parametric_tetrahedra=compiled.parametric,
    )
    samples = result.solver_samples
    assert samples is not None
    assert samples.parametric_points is not None
    assert samples.sample_count == int(result.active_edges.sum())
    assert samples.parametric_points.shape == samples.embedded_points.shape
    assert np.all((samples.interpolation_weights >= 0.0)
                  & (samples.interpolation_weights <= 1.0))


def test_fifo_spline_solver_accumulates_control_points_in_order():
    solver = StreamingSplineSolver(intrinsic_axes=(0,), neighbors=None)
    parameters = np.linspace(0.0, 1.0, 12)[:, None]
    embedded = np.concatenate(
        (parameters, parameters**2, parameters**3), axis=1
    )
    solver.submit(parameters[:6], embedded[:6])
    solver.submit(parameters[6:], embedded[6:])
    assert solver.pending_batches == 2
    model = solver.update()
    assert model is solver.latest_model
    assert solver.pending_batches == 0
    assert solver.control_point_count == 12
    assert model.intrinsic_dimension == 1
    assert model.embedding_dimension == 3
    assert np.max(np.abs(model(parameters) - embedded)) < 1e-5


def test_demo_reconstructs_original_domain_transform_with_a_surface_spline():
    report = build_spline_report(resolution=6)
    assert report.loc[0, "parameter_dimension"] == 3
    assert report.loc[0, "intrinsic_dimension"] == 2
    assert report.loc[0, "embedding_dimension"] == 3
    assert report.loc[0, "exported_solver_samples"] > 100
    assert report.loc[0, "fifo_batches_pending"] == 0
    assert report.loc[0, "mean_spline_target_error"] < 0.04
