import importlib.util
import numpy as np
import pytest

pytest.skip("Linear.backward not implemented", allow_module_level=True)

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
def test_linear_bias_broadcast_and_grad(backend_name, Backend):
    like = Backend.tensor([[0.0]])
    layer = Linear(in_dim=2, out_dim=2, like=like)
    layer.W = Backend.zeros((2, 2))
    layer.b = Backend.tensor([[1.0, 2.0]])
    x = Backend.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    out = layer.forward(x)
    assert out.tolist() == [[1.0, 2.0]] * 3
    grad_out = Backend.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    layer.backward(grad_out)
    assert tuple(layer.gb.shape) == (1, 2)
    assert layer.gb.tolist() == [[3.0, 3.0]]


def test_expand_is_view_on_numpy():
    if NumPyTensorOperations is None:
        pytest.skip("numpy backend not available")
    t = NumPyTensorOperations.tensor([[1.0, 2.0]])
    expanded = t.expand((3, 2))
    assert np.shares_memory(expanded.data, t.data)


def test_expand_accepts_varargs():
    if NumPyTensorOperations is None:
        pytest.skip("numpy backend not available")
    t = NumPyTensorOperations.tensor([[1.0, 2.0]])
    expanded = t.expand(3, 2)
    assert expanded.shape == (3, 2)
