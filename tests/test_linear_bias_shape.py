import importlib.util
import pytest

from src.common.tensors.pure_backend import PurePythonTensorOperations
from src.common.tensors.abstract_nn.core import Linear

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

BACKENDS = [("PurePython", PurePythonTensorOperations)]
if PyTorchTensorOperations is not None:
    BACKENDS.append(("PyTorch", PyTorchTensorOperations))
if NumPyTensorOperations is not None:
    BACKENDS.append(("NumPy", NumPyTensorOperations))


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_linear_bias_shape_is_row_vector(backend_name, Backend):
    like = Backend.tensor_from_list([[0.0]])
    layer = Linear(in_dim=1, out_dim=1, like=like)
    assert tuple(layer.b.shape) == (1, 1)
