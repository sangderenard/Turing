from types import SimpleNamespace

import numpy as np

from src.common.tensors.riemann import (
    AdaptiveSurfaceTriangulator,
    TriangulationTolerance,
)
from src.common.tensors.youngman import blackbox_roundtrip_demo as demo


def test_published_spline_no_longer_needs_source(monkeypatch):
    u, v = np.meshgrid(
        np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5), indexing="ij"
    )
    parameters = np.column_stack(
        (u.ravel(), v.ravel(), np.full(u.size, 0.5))
    )
    samples = SimpleNamespace(
        parametric_points=parameters, sample_count=len(parameters)
    )
    published, count = demo.publish_surface_spline(samples)
    monkeypatch.setattr(
        demo,
        "detailed_embedding",
        lambda *_: (_ for _ in ()).throw(AssertionError("source leaked")),
    )
    query = np.asarray(((0.2, 0.3), (0.7, 0.8)))
    assert published(query).shape == (2, 5)
    assert published.jacobian(query).shape == (2, 5, 2)
    assert count == len(parameters)
    mesh = AdaptiveSurfaceTriangulator(
        published,
        tolerance=TriangulationTolerance(position=0.1, max_rounds=2),
    ).triangulate()
    assert mesh.embedding_dimension == 5
