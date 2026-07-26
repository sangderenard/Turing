import numpy as np

from src.common.tensors.youngman.refinement_network import (
    train_refinement_predictor,
    triangle_refinement_features,
    triangle_spring_edges,
)


def test_abstract_refinement_predictor_trains_and_exports_schedules():
    u, v = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5), indexing="ij")
    parameters = np.column_stack((u.ravel(), v.ravel()))
    triangles = []
    for i in range(4):
        for j in range(4):
            a = i * 5 + j
            triangles.extend(((a, a + 5, a + 6), (a, a + 6, a + 1)))
    triangles = np.asarray(triangles)
    features = triangle_refinement_features(parameters, triangles)
    centroids = parameters[triangles].mean(axis=1)
    errors = 1e-4 * (
        1.0 + 2.0 * centroids[:, 0] + 0.5 * centroids[:, 1]
    )
    result = train_refinement_predictor(
        features,
        errors,
        epsilon=1e-5,
        epochs=200,
        learning_rate=0.05,
        spring_edges=triangle_spring_edges(triangles),
        spring_strength=0.02,
    )
    assert result.final_loss < result.initial_loss * 0.05
    assert result.forward_nodes > 0
    assert result.backward_nodes > 0
    assert result.concurrent_forward_width > 1
    assert np.corrcoef(result.predictions, np.log1p(errors / 1e-5))[0, 1] > 0.98
    assert result.validation_loss < result.baseline_validation_loss
    assert result.validation_correlation > 0.95
    assert result.accepted
    alpha = result.predict_alpha(features)
    assert alpha.shape == errors.shape
    assert np.corrcoef(alpha, errors / 1e-5)[0, 1] > 0.98


def test_triangle_springs_connect_only_edge_neighbors():
    triangles = np.asarray(((0, 1, 2), (0, 2, 3), (2, 4, 3)))
    assert triangle_spring_edges(triangles).tolist() == [[0, 1], [1, 2]]


def test_predictor_backpropagates_with_full_geometry_feature_width():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(160, 26))
    errors = np.exp(0.2 * features[:, 0] - 0.1 * features[:, 15])
    result = train_refinement_predictor(
        features, errors, epsilon=1.0, epochs=3
    )
    assert np.isfinite(result.final_loss)
