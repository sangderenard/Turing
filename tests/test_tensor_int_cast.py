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

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    try:
        from src.common.tensors.torch_backend import PyTorchTensorOperations
    except Exception:  # pragma: no cover - optional dependency
        PyTorchTensorOperations = None
else:  # torch not available
    PyTorchTensorOperations = None

BACKENDS = [("PurePython", PurePythonTensorOperations)]
if NumPyTensorOperations is not None:
    BACKENDS.append(("NumPy", NumPyTensorOperations))
if JAXTensorOperations is not None:
    BACKENDS.append(("JAX", JAXTensorOperations))
if PyTorchTensorOperations is not None:
    BACKENDS.append(("PyTorch", PyTorchTensorOperations))


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_int_conversion(backend_name, Backend):
    t = AbstractTensor.get_tensor(3.9, cls=Backend)
    assert int(t) == 3


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_int_conversion_requires_scalar(backend_name, Backend):
    t = AbstractTensor.get_tensor([1.0, 2.0], cls=Backend)
    with pytest.raises(TypeError, match="Only scalar tensors can be converted to int"):
        int(t)
