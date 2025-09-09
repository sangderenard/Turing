import numpy as np
import pytest

from src.common.tensors.numpy_backend import NumPyTensorOperations as T


@pytest.mark.skipif(T is None, reason="NumPy backend required")
def test_fft_backward_matches_ifft():
    x = T.arange(8, dtype=np.float32)
    x.requires_grad_(True)

    y = x.fft().ifft()
    loss = y.sum()
    loss.backward()

    expected = T.ones_like(x)
    assert np.allclose(x.grad.data, expected.data)

