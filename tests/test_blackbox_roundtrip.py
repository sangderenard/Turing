from types import SimpleNamespace

import numpy as np

from src.common.tensors.riemann import (
    AdaptiveSurfaceTriangulator,
    TriangulationTolerance,
)
from src.common.tensors.youngman import blackbox_roundtrip_demo as demo
from src.common.tensors.youngman import validate_single_valued_chart


def test_published_spline_no_longer_needs_source(monkeypatch):
    u, v = np.meshgrid(
        np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5), indexing="ij"
    )
    parameters = np.column_stack(
        (u.ravel(), v.ravel(), np.full(u.size, 0.5))
    )
    samples = SimpleNamespace(
        parametric_points=parameters,
        expanded_points=demo.detailed_embedding(parameters),
        sample_count=len(parameters),
    )
    monkeypatch.setattr(
        demo,
        "detailed_embedding",
        lambda *_: (_ for _ in ()).throw(AssertionError("source leaked")),
    )
    published, count = demo.publish_surface_spline(samples)
    query = np.asarray(((0.2, 0.3), (0.7, 0.8)))
    assert published(query).shape == (2, 5)
    assert published.jacobian(query).shape == (2, 5, 2)
    assert count == len(parameters)
    mesh = AdaptiveSurfaceTriangulator(
        published,
        tolerance=TriangulationTolerance(position=0.1, max_rounds=2),
    ).triangulate()
    assert mesh.embedding_dimension == 5


def test_multisheet_samples_are_rejected_before_spline_fit():
    samples = np.asarray(((0.2, 0.3, 0.0), (0.2, 0.3, 1.0)))
    with np.testing.assert_raises(ValueError):
        validate_single_valued_chart(samples, intrinsic_axes=(0, 1))


def test_time_parameter_changes_geometry_periodically():
    uv = np.asarray(((0.17, 0.31), (0.63, 0.72)))
    assert not np.allclose(
        demo.source_surface(uv, 0.0), demo.source_surface(uv, 0.25)
    )
    assert np.allclose(
        demo.source_surface(uv, 0.0), demo.source_surface(uv, 1.0)
    )


def test_manifold_presets_are_distinct_periodic_five_dimensional_maps():
    uv = np.asarray(((0.17, 0.31), (0.63, 0.72)))
    surfaces = {
        name: demo.source_surface(uv, 0.0, name)
        for name in ("ripple", "banana", "saddle", "twisted_ribbon")
    }
    assert all(values.shape == (2, 5) for values in surfaces.values())
    assert not np.allclose(surfaces["banana"], surfaces["saddle"])
    for name, values in surfaces.items():
        assert np.allclose(values, demo.source_surface(uv, 1.0, name))


def test_nontrivial_manifold_jacobian_is_finite():
    uv = np.asarray(((0.17, 0.31), (0.63, 0.72)))
    jacobian = demo.source_surface_jacobian(uv, 0.2, "banana")
    assert jacobian.shape == (2, 5, 2)
    assert np.isfinite(jacobian).all()


def test_learned_mesh_deployment_requires_every_gate():
    assert demo.deployment_decision(
        model_accepted=True,
        guided_converged=True,
        objective_improved=True,
    ) == (True, "accepted")
    assert demo.deployment_decision(
        model_accepted=True,
        guided_converged=False,
        objective_improved=True,
    ) == (False, "guided_mesh_unconverged")
    assert demo.deployment_decision(
        model_accepted=True,
        guided_converged=True,
        objective_improved=False,
    ) == (False, "laplace_not_improved")
