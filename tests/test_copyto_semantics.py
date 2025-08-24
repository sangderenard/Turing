import importlib.util
import pytest

from src.common.tensors import AbstractTensor
from src.common.tensors.pure_backend import PurePythonTensorOperations

try:
    from src.common.tensors.numpy_backend import NumPyTensorOperations
except Exception:  # pragma: no cover - optional dependency
    NumPyTensorOperations = None

jax_spec = importlib.util.find_spec("jax")
if jax_spec is not None:
    try:
        from src.common.tensors.jax_backend import JAXTensorOperations
    except Exception:  # pragma: no cover - optional dependency
        JAXTensorOperations = None
else:  # jax not available
    JAXTensorOperations = None

BACKENDS = [("PurePython", PurePythonTensorOperations)]
if NumPyTensorOperations is not None:
    BACKENDS.append(("NumPy", NumPyTensorOperations))
if JAXTensorOperations is not None:
    BACKENDS.append(("JAX", JAXTensorOperations))


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_copyto_scalar_broadcast_and_where(backend_name, Backend):
    dst = AbstractTensor.get_tensor([[0, 0], [0, 0]], cls=Backend)
    src = AbstractTensor.get_tensor(5, cls=Backend)
    AbstractTensor.copyto(dst, src)
    assert dst.tolist() == [[5, 5], [5, 5]]

    dst2 = AbstractTensor.get_tensor([[0, 0], [0, 0]], cls=Backend)
    mask = AbstractTensor.get_tensor([[True, False], [False, True]], cls=Backend)
    AbstractTensor.copyto(dst2, src, where=mask)
    assert dst2.tolist() == [[5, 0], [0, 5]]


@pytest.mark.skipif(NumPyTensorOperations is None, reason="NumPy backend not available")
def test_copyto_broadcast_and_casting_numpy():
    import numpy as np

    dst = AbstractTensor.get_tensor(np.zeros((2, 2), dtype=np.int32), cls=NumPyTensorOperations)
    src = AbstractTensor.get_tensor([1, 2], cls=NumPyTensorOperations)
    AbstractTensor.copyto(dst, src)
    assert dst.tolist() == [[1, 2], [1, 2]]

    dst = AbstractTensor.get_tensor(np.zeros((2, 2), dtype=np.int32), cls=NumPyTensorOperations)
    src = AbstractTensor.get_tensor(np.ones((2, 2), dtype=np.float32), cls=NumPyTensorOperations)
    with pytest.raises(TypeError):
        AbstractTensor.copyto(dst, src)
    AbstractTensor.copyto(dst, src, casting="unsafe")
    assert dst.tolist() == [[1, 1], [1, 1]]
