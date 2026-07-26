import numpy as np

from src.common.tensors.riemann import TriangulatedSurfaceTransform


def _transform():
    parameters = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    embedded = np.column_stack((
        parameters,
        parameters[:, 0] + 2.0 * parameters[:, 1],
        np.zeros((4, 2)),
    ))
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)))
    return TriangulatedSurfaceTransform.from_mesh(parameters, embedded, triangles)


def test_piecewise_transform_and_metric_are_exact_for_affine_embedding():
    transform = _transform()
    query = np.asarray(((0.2, 0.1), (0.2, 0.8), (0.5, 0.5)))
    expected = np.column_stack((
        query, query[:, 0] + 2.0 * query[:, 1], np.zeros((3, 2))
    ))
    assert np.allclose(transform.transform(query), expected)
    expected_metric = np.asarray(((2.0, 2.0), (2.0, 5.0)))
    assert np.allclose(transform.metric_tensor(query), expected_metric)


def test_queries_outside_chart_are_rejected():
    transform = _transform()
    with np.testing.assert_raises(ValueError):
        transform.transform(np.asarray(((1.1, 0.5),)))


def test_nodus_payload_is_detached_from_transform_storage():
    transform = _transform()
    payload = transform.nodus_payload()
    payload["parameters"][0] = 99.0
    assert np.allclose(transform.parameters[0], 0.0)
