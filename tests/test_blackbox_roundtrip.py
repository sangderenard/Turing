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
