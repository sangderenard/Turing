import numpy as np
import pytest

from src.common.tensors.pure_backend import PurePythonTensorOperations

try:  # optional dependency
    from src.common.tensors.numpy_backend import NumPyTensorOperations
except Exception:  # pragma: no cover
    NumPyTensorOperations = None

BACKENDS = [("PurePython", PurePythonTensorOperations)]
if NumPyTensorOperations is not None:
    BACKENDS.append(("NumPy", NumPyTensorOperations))


@pytest.mark.parametrize("name,Backend", BACKENDS)
def test_expand_with_negative_one(name, Backend):
    data = np.zeros((1, 1, 2, 3, 4)).tolist()
    t = Backend.tensor(data)
    expanded = t.expand(2, 3, -1, -1, -1)
    assert expanded.shape == (2, 3, 2, 3, 4)
