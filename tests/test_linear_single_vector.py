import importlib.util
import pytest

from src.common.tensors.pure_backend import PurePythonTensorOperations
from src.common.tensors.abstract_nn.core import Linear
from src.common.tensors.autograd import autograd

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
def test_linear_single_vector_trains(backend_name, Backend):
    if backend_name == "PurePython":
        pytest.skip("PurePython backend lacks full autograd support for this test")
    like = Backend.tensor([[0.0]])
    layer = Linear(in_dim=2, out_dim=3, like=like)
    x = Backend.tensor([1.0, -1.0], requires_grad=True)
    autograd.tape.create_tensor_node(x)
    out = layer.forward(x)
    assert tuple(out.shape) == (1, 3)
    loss = out.reshape((3,)).sum()
    loss.backward()
    assert tuple(layer.W.grad.shape) == (2, 3)
    assert layer.b is not None and tuple(layer.b.grad.shape) == (1, 3)
