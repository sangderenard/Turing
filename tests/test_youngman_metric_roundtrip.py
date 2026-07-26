import numpy as np

from src.common.tensors.abstract_convolution.laplace_nd import GridDomain
from src.common.tensors.youngman.algorithm import (
    compile_grid_domain,
    extract_isosurface,
    metric_sample_tags,
    sphere_field,
)
from src.common.tensors.youngman.metric_roundtrip_demo import (
    BoundaryResolution,
    build_metric_roundtrip,
    laplace_beltrami,
    singular_metric_mask,
)


def test_extraction_preserves_parametric_triangle_provenance():
    domain = GridDomain.generate_grid_domain(
        "rectangular", N_u=3, N_v=3, N_w=3,
        Lx=1.0, Ly=1.0, Lz=1.0, defer_resolution=True,
    )
    compiled = compile_grid_domain(domain)
    result = extract_isosurface(
        compiled.embedded,
        lambda points: sphere_field(points, radius=0.35, center=0.5),
        parametric_tetrahedra=compiled.parametric,
    )
    assert result.parametric_triangles is not None
    assert result.triangle_tetrahedron_ids is not None
    assert result.parametric_triangles.shape == result.triangles.shape
    assert len(result.triangle_tetrahedron_ids) == result.triangle_count


def test_metric_tags_retain_full_matrix_and_derived_state():
    parameters = np.asarray(((0.2, 0.3, 0.4), (0.5, 0.6, 0.7)))
    metric = np.asarray((np.eye(3), np.diag((2.0, 3.0, 4.0))))
    tags = metric_sample_tags(parameters, np.asarray((4, 9)), metric)
    assert tags.metric.shape == (2, 3, 3)
    assert np.allclose(tags.inverse_metric[1], np.diag((0.5, 1 / 3, 0.25)))
    assert np.allclose(tags.determinant, (1.0, 24.0))


def test_roundtrip_compares_laplace_on_the_same_surface():
    summary, triangles, display = build_metric_roundtrip(
        resolution=1, samples_per_patch=10, fifo_batches=3
    )
    assert summary.loc[0, "embedding_dimension"] == 5
    assert summary.loc[0, "metric_matrix_shape"] == "3x3"
    assert summary.loc[0, "fifo_batches_pending"] == 0
    assert np.isfinite(triangles["laplace_difference"]).all()
    assert len(triangles) == display.triangle_count


def test_boundary_option_marks_faces_and_uses_one_sided_stencil():
    points = np.asarray(((0.0, 0.4, 0.5), (0.5, 0.4, 0.5)))
    identity_metric = lambda rows: np.repeat(np.eye(3)[None], len(rows), axis=0)
    policy = BoundaryResolution(
        np.asarray(((0.0, 1.0),) * 3),
        ("dirichlet",) * 6,
        (True,) * 6,
    )
    values, mask = laplace_beltrami(
        points,
        identity_metric,
        boundary_resolution=policy,
        return_boundary_mask=True,
    )
    assert np.array_equal(mask, (True, False))
    assert np.isfinite(values).all()


def test_singularity_detector_uses_numerical_health_not_exact_zero_only():
    metrics = np.asarray((np.eye(3), np.diag((1.0, 1.0, 1e-12))))
    assert np.array_equal(singular_metric_mask(metrics), (False, True))
