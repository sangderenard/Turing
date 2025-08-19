import importlib.util
import pytest

from src.common.tensors.pure_backend import PurePythonTensorOperations

try:
    from src.common.tensors.torch_backend import PyTorchTensorOperations
except Exception:  # pragma: no cover - optional dependency
    PyTorchTensorOperations = None

try:
    from src.common.tensors.numpy_backend import NumPyTensorOperations
except Exception:  # pragma: no cover - optional dependency
    NumPyTensorOperations = None

try:
    from src.common.tensors.jax_backend import JAXTensorOperations
except Exception:  # pragma: no cover - optional dependency
    JAXTensorOperations = None

BACKENDS = [("PurePython", PurePythonTensorOperations)]
torch_spec = importlib.util.find_spec("torch")
if PyTorchTensorOperations is not None and torch_spec is not None:
    BACKENDS.append(("PyTorch", PyTorchTensorOperations))
if NumPyTensorOperations is not None:
    BACKENDS.append(("NumPy", NumPyTensorOperations))
jax_spec = importlib.util.find_spec("jax")
if JAXTensorOperations is not None and jax_spec is not None:
    BACKENDS.append(("JAX", JAXTensorOperations))


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_basic_add_and_zeros(backend_name, Backend):
    a = Backend.tensor_from_list([[1, 2], [3, 4]])
    b = Backend.tensor_from_list([[5, 6], [7, 8]])
    result = a + b
    assert result.tolist() == [[6, 8], [10, 12]]
    zeros = Backend.zeros((2, 2))
    assert zeros.tolist() == [[0, 0], [0, 0]]
