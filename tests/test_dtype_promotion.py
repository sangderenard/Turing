import importlib.util
import pytest

from src.common.tensors.pure_backend import PurePythonTensorOperations

try:
    from src.common.tensors.numpy_backend import NumPyTensorOperations
except Exception:  # optional dependency
    NumPyTensorOperations = None

jax_spec = importlib.util.find_spec("jax")
if jax_spec is not None:
    try:
        from src.common.tensors.jax_backend import JAXTensorOperations
    except Exception:
        JAXTensorOperations = None
else:
    JAXTensorOperations = None

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    try:
        from src.common.tensors.torch_backend import PyTorchTensorOperations
    except Exception:
        PyTorchTensorOperations = None
else:
    PyTorchTensorOperations = None

BACKENDS = [("PurePython", PurePythonTensorOperations)]
if NumPyTensorOperations is not None:
    BACKENDS.append(("NumPy", NumPyTensorOperations))
if JAXTensorOperations is not None:
    BACKENDS.append(("JAX", JAXTensorOperations))
if PyTorchTensorOperations is not None:
    BACKENDS.append(("PyTorch", PyTorchTensorOperations))


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_zeros_default_float(backend_name, Backend):
    t = Backend.zeros((2, 2))
    assert "float" in str(t.get_dtype()).lower()


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_matmul_promotes_int_to_float(backend_name, Backend):
    a = Backend.ones((2, 2))
    b = Backend.ones((2, 2), dtype="int")
    c = a @ b
    assert "float" in str(c.get_dtype()).lower()
