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
def test_python_list_operands(backend_name, Backend):
    t = Backend.tensor_from_list([1, 2, 3])
    assert (t - [1, 1, 1]).tolist() == [0, 1, 2]
    assert ([1, 1, 1] - t).tolist() == [0, -1, -2]
