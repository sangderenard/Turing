from src.common.tensors import AbstractTensor
from src.common.tensors.filtered_poisson import filtered_poisson


def test_residual_poisson_smoothing_reduces_peak():
    rhs = AbstractTensor.get_tensor([0.0, 2.0, 0.0])
    adjacency = AbstractTensor.get_tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], like=rhs
    )
    smoothed = filtered_poisson(rhs, iterations=20, adjacency=adjacency)
    assert float(smoothed[0]) > 0.0 and float(smoothed[2]) > 0.0
