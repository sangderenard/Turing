import importlib.util
import pytest

from src.common.tensors.pure_backend import PurePythonTensorOperations

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    try:
        from src.common.tensors.torch_backend import PyTorchTensorOperations
    except Exception:  # pragma: no cover - optional dependency
        PyTorchTensorOperations = None
else:  # torch not available
    PyTorchTensorOperations = None

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
if PyTorchTensorOperations is not None:
    BACKENDS.append(("PyTorch", PyTorchTensorOperations))
if NumPyTensorOperations is not None:
    BACKENDS.append(("NumPy", NumPyTensorOperations))
if JAXTensorOperations is not None:
    BACKENDS.append(("JAX", JAXTensorOperations))


def to_list(x):
    return x.tolist() if hasattr(x, "tolist") else x


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_basic_add_and_zeros(backend_name, Backend):
    a = Backend.tensor_from_list([[1, 2], [3, 4]])
    b = Backend.tensor_from_list([[5, 6], [7, 8]])
    result = a + b
    assert result.tolist() == [[6, 8], [10, 12]]
    zeros = Backend.zeros((2, 2))
    assert zeros.tolist() == [[0, 0], [0, 0]]


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_flatten(backend_name, Backend):
    a = Backend.tensor_from_list([[1, 2], [3, 4]])
    flat = a.flatten()
    assert flat.tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_prod(backend_name, Backend):
    a = Backend.tensor_from_list([[1, 2], [3, 4]])
    assert to_list(a.prod()) == 24
    assert to_list(a.prod(dim=0)) == [3, 8]
    assert to_list(a.prod(dim=1, keepdim=True)) == [[2], [12]]
    assert to_list(a.prod(dim=-1)) == [2, 12]
