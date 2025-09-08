import importlib.util
import numpy as np
import pytest

try:
    from src.common.tensors.numpy_backend import NumPyTensorOperations
except Exception:  # pragma: no cover - optional dependency
    NumPyTensorOperations = None

BACKENDS = []
if NumPyTensorOperations is not None:
    BACKENDS.append(("NumPy", NumPyTensorOperations))


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_gather_and_mul_add_broadcast(backend_name, Backend):
    batch, n, features = 2, 3, 4
    base = Backend.tensor(np.ones((n, features)))
    indices = Backend.tensor(np.tile(np.arange(n), (batch, 1)))
    params = Backend.tensor(np.vstack([np.arange(n), np.arange(n) + 10]))
    fn_specs = [(Backend.__mul__, 0), (Backend.__add__, 1)]
    out = base.gather_and(0, indices, fn_specs, params)
    expected = (
        np.ones((batch, n, features)) * np.arange(n).reshape(1, n, 1)
        + (np.arange(n) + 10).reshape(1, n, 1)
    )
    assert np.allclose(out.data, expected)
