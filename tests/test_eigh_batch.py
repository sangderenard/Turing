import numpy as np

from src.common.tensors import AbstractTensor
from src.common.tensors.numpy_backend import NumPyTensorOperations


def _to_numpy(t):
    np_backend = AbstractTensor.get_tensor(cls=NumPyTensorOperations)
    return t.to_backend(np_backend).numpy()


def test_eigh_handles_batched_matrices():
    AT = AbstractTensor
    rng = np.random.default_rng(0)
    B, n = 4, 3
    M = rng.standard_normal((B, n, n))
    A_np = np.matmul(np.transpose(M, (0, 2, 1)), M)
    A = AT.get_tensor(A_np)

    w, V = AT.linalg.eigh(A)
    w_np = _to_numpy(w)
    V_np = _to_numpy(V)

    w_ref, V_ref = np.linalg.eigh(A_np)

    assert w_np.shape == (B, n)
    assert V_np.shape == (B, n, n)
    assert np.allclose(w_np, w_ref, atol=1e-5)

    for i in range(B):
        A_recon = V_np[i] @ np.diag(w_np[i]) @ V_np[i].T
        assert np.allclose(A_recon, A_np[i], atol=1e-5)
