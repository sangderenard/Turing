import numpy as np

from src.common.tensors.youngman.piecewise import (
    SimplexBezierFactory,
    StreamingPiecewiseSplineEngine,
    simplex_multi_indices,
)
from src.common.tensors.youngman.piecewise_demo import (
    build_piecewise_report,
    expanded_embedding,
    expanded_jacobian,
)


TETRAHEDRON = np.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
     (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    dtype=np.float64,
)


def sample_tetrahedron(count=48, seed=3):
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(4), size=count) @ TETRAHEDRON


def test_n_dimensional_basis_control_count():
    assert simplex_multi_indices(1, 3).shape == (4, 2)
    assert simplex_multi_indices(2, 3).shape == (10, 3)
    assert simplex_multi_indices(3, 3).shape == (20, 4)
    assert simplex_multi_indices(4, 3).shape == (35, 5)


def test_three_to_five_patch_preserves_values_derivatives_and_metric():
    train = sample_tetrahedron()
    patch = SimplexBezierFactory.fit(
        7,
        TETRAHEDRON,
        train,
        expanded_embedding(train),
        degree=3,
        jacobians=expanded_jacobian(train),
        ridge=0.0,
    )
    query = sample_tetrahedron(20, seed=9)
    expected_values = expanded_embedding(query)
    expected_jacobian = expanded_jacobian(query)
    assert patch.embedding_dimension == 5
    assert np.max(np.abs(patch.evaluate(query) - expected_values)) < 1e-10
    assert np.max(np.abs(patch.jacobian(query) - expected_jacobian)) < 1e-10

    full, spatial, hidden = patch.collapsed_metric_components(query)
    expected_metric = np.einsum(
        "nmi,nmj->nij", expected_jacobian, expected_jacobian
    )
    assert np.max(np.abs(full - expected_metric)) < 1e-10
    assert np.max(np.abs(full - spatial - hidden)) < 1e-12
    assert np.linalg.norm(hidden) > 0.0


def test_streaming_engine_consumes_fifo_and_publishes_generations():
    samples = sample_tetrahedron()
    values = expanded_embedding(samples)
    jacobians = expanded_jacobian(samples)
    engine = StreamingPiecewiseSplineEngine({3: TETRAHEDRON}, ridge=0.0)
    for rows in np.array_split(np.arange(len(samples)), 4):
        engine.submit(
            np.full(len(rows), 3),
            samples[rows],
            values[rows],
            jacobians[rows],
        )
    assert engine.pending_batches == 4
    first = engine.update()
    assert first is engine.latest_generation
    assert first.generation == 1
    assert engine.pending_batches == 0

    extra = sample_tetrahedron(24, seed=10)
    engine.submit(
        np.full(len(extra), 3),
        extra,
        expanded_embedding(extra),
        expanded_jacobian(extra),
    )
    second = engine.update()
    assert second.generation == 2
    assert first.generation == 1
    assert np.max(np.abs(second.evaluate(extra) - expanded_embedding(extra))) < 1e-10


def test_piecewise_demo_reports_full_and_collapsed_metric_accuracy():
    summary, patches = build_piecewise_report(
        samples_per_patch=28,
        validation_per_patch=5,
        fifo_batches=7,
    )
    assert summary.loc[0, "patches"] == 6
    assert summary.loc[0, "embedding_dimension"] == 5
    assert summary.loc[0, "fifo_batches_pending"] == 0
    assert summary.loc[0, "owner_mismatches"] == 0
    assert summary.loc[0, "max_expanded_error"] < 1e-9
    assert summary.loc[0, "max_jacobian_error"] < 1e-9
    assert summary.loc[0, "max_metric_error"] < 1e-9
    assert summary.loc[0, "mean_hidden_metric_contribution"] > 0.0
    assert set(patches["intrinsic_dimension"]) == {3}
    assert set(patches["embedding_dimension"]) == {5}
