import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_convolution.laplace_nd import (
    continuous_laplace_beltrami,
)


def _identity_metric(points):
    count, rank = points.get_shape()
    identity = np.repeat(np.eye(rank)[None], count, axis=0)
    return AbstractTensor.tensor(
        identity, dtype=points.get_dtype(), device=points.get_device()
    )


@pytest.mark.parametrize("backend", ["numpy", "c", "torch"])
def test_surface_is_the_rank_two_case_of_continuous_laplace_nd(backend):
    with AbstractTensor.use_backend(backend):
        points = AbstractTensor.tensor(
            ((0.17, 0.23), (0.31, 0.42), (0.63, 0.71)),
            dtype="float64",
        )
        tau = 2.0 * np.pi

        def gradient(rows):
            u, v = rows[:, 0], rows[:, 1]
            return AbstractTensor.stack(
                (
                    tau * (tau * u).cos() * (tau * v).cos(),
                    -tau * (tau * u).sin() * (tau * v).sin(),
                ),
                dim=1,
            )

        actual = np.asarray(
            continuous_laplace_beltrami(
                points, _identity_metric, gradient, step=1e-5
            ).tolist()
        )
    host = np.asarray(points.tolist())
    expected = -2.0 * tau**2 * np.sin(tau * host[:, 0]) * np.cos(
        tau * host[:, 1]
    )
    assert np.allclose(actual, expected, rtol=2e-7, atol=2e-7)


def test_continuous_laplace_nd_reads_rank_from_geometry():
    points = AbstractTensor.tensor(
        ((0.17, 0.23, 0.29), (0.31, 0.42, 0.53)), dtype="float64"
    )

    def quadratic_gradient(rows):
        return 2.0 * rows

    actual = np.asarray(
        continuous_laplace_beltrami(
            points, _identity_metric, quadratic_gradient, step=1e-5
        ).tolist()
    )
    assert np.allclose(actual, 6.0, rtol=1e-6, atol=1e-6)
