import importlib.util
import pytest

from src.common.tensors.abstract_nn.losses import BCEWithLogitsLoss

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
def test_bce_with_logits_forward_backward(backend_name, Backend):
    logits = Backend.tensor_from_list([[0.0], [2.0], [-2.0]])
    target = Backend.tensor_from_list([[0.0], [1.0], [0.0]])
    loss = BCEWithLogitsLoss()
    val = loss.forward(logits, target)
    if hasattr(val, "tolist"):
        val = val.tolist()
    assert pytest.approx(float(val), rel=1e-6) == pytest.approx(0.3156677342, rel=1e-6)

    grad = loss.backward(logits, target)
    if hasattr(grad, "tolist"):
        grad_list = grad.tolist()
    else:
        grad_list = grad
    assert pytest.approx(grad_list[0][0], rel=1e-6) == 0.1666667
    assert pytest.approx(grad_list[1][0], rel=1e-6) == -0.0397343
    assert pytest.approx(grad_list[2][0], rel=1e-6) == 0.0397343

