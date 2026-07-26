import numpy as np

from src.common.tensors import AbstractTensor


def test_large_tiled_matmul_preserves_forward_and_backward():
    rng = np.random.default_rng(73)
    a_host = rng.normal(size=(4097, 3))
    b_host = rng.normal(size=(3, 2))
    a = AbstractTensor.tensor(a_host)
    b = AbstractTensor.tensor(b_host)
    a.requires_grad_(True)
    b.requires_grad_(True)

    product = a @ b
    loss = product.sum()
    grad_a, grad_b = AbstractTensor.autograd.grad(loss, [a, b])

    assert np.allclose(np.asarray(product.tolist()), a_host @ b_host)
    assert np.allclose(
        np.asarray(grad_a.tolist()),
        np.ones((4097, 2)) @ b_host.T,
    )
    assert np.allclose(
        np.asarray(grad_b.tolist()),
        a_host.T @ np.ones((4097, 2)),
    )
