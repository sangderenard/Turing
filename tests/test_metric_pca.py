from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.abstract_convolution.ndpca3transform import fit_metric_pca


def test_fit_metric_pca_returns_expected_shapes():
    AT = AbstractTensor
    B, n = 10, 3

    t = AT.linspace(0.0, 1.0, B)
    samples = AT.stack([t, t**2, t**3], dim=-1)
    weights = AT.linspace(1.0, 2.0, B)

    M = AT.eye(n)
    diag = AT.get_tensor([1.0, 0.5, 2.0])
    M = M * diag.reshape(1, -1)
    M = M.swapaxes(-1, -2) * diag.reshape(1, -1)

    basis = fit_metric_pca(samples, weights=weights, metric_M=M)

    assert basis.mu.shape == (n,)
    assert basis.P.shape == (n, n)
    assert basis.n == n
