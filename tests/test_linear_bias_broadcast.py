import importlib.util
import pytest

from src.common.tensors.pure_backend import PurePythonTensorOperations
from src.common.tensors.abstract_nn.core import Linear

try:
    from src.common.tensors.torch_backend import PyTorchTensorOperations
except Exception:  # pragma: no cover - optional dependency
    PyTorchTensorOperations = None

try:
    from src.common.tensors.numpy_backend import NumPyTensorOperations
except Exception:  # pragma: no cover - optional dependency
    NumPyTensorOperations = None

BACKENDS = [("PurePython", PurePythonTensorOperations)]
if PyTorchTensorOperations is not None and importlib.util.find_spec("torch") is not None:
    BACKENDS.append(("PyTorch", PyTorchTensorOperations))
if NumPyTensorOperations is not None:
    BACKENDS.append(("NumPy", NumPyTensorOperations))


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_linear_bias_broadcast_and_grad(backend_name, Backend):
    like = Backend.tensor_from_list([[0.0]])
    layer = Linear(in_dim=2, out_dim=2, like=like)
    layer.W = Backend.zeros((2, 2))
    layer.b = Backend.tensor_from_list([[1.0, 2.0]])
    x = Backend.tensor_from_list([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    out = layer.forward(x)
    assert out.tolist() == [[1.0, 2.0]] * 3
    grad_out = Backend.tensor_from_list([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    layer.backward(grad_out)
    assert tuple(layer.gb.shape) == (1, 2)
    assert layer.gb.tolist() == [[3.0, 3.0]]


def test_expand_is_view_on_torch():
    if PyTorchTensorOperations is None or importlib.util.find_spec("torch") is None:
        pytest.skip("torch not available")
    t = PyTorchTensorOperations.tensor_from_list([[1.0, 2.0]])
    expanded = t.expand((3, 2))
    assert expanded.data.storage().data_ptr() == t.data.storage().data_ptr()
