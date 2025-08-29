import importlib.util
import pytest

from src.common.tensors.abstract_nn.losses import MSELoss

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

BACKENDS = []
if PyTorchTensorOperations is not None:
    BACKENDS.append(("PyTorch", PyTorchTensorOperations))
if NumPyTensorOperations is not None:
    BACKENDS.append(("NumPy", NumPyTensorOperations))
if JAXTensorOperations is not None:
    BACKENDS.append(("JAX", JAXTensorOperations))


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_mse_loss_backward_multi_dim(backend_name, Backend):
    pred = Backend.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = Backend.zeros((2, 2))
    loss = MSELoss()
    grad = loss.backward(pred, target)
    assert grad.tolist() == [[0.5, 1.0], [1.5, 2.0]]


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_mse_loss_forward_multi_dim(backend_name, Backend):
    pred = Backend.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = Backend.zeros((2, 2))
    loss = MSELoss()
    val = loss.forward(pred, target)
    if hasattr(val, "tolist"):
        val = val.tolist()
    assert pytest.approx(float(val), rel=1e-6) == 7.5
