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
    a = Backend.tensor([[1, 2], [3, 4]])
    b = Backend.tensor([[5, 6], [7, 8]])
    result = a + b
    assert result.tolist() == [[6, 8], [10, 12]]
    zeros = Backend.zeros((2, 2))
    assert zeros.tolist() == [[0, 0], [0, 0]]


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_flatten(backend_name, Backend):
    a = Backend.tensor([[1, 2], [3, 4]])
    flat = a.flatten()
    assert flat.tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_prod(backend_name, Backend):
    a = Backend.tensor([[1, 2], [3, 4]])
    assert to_list(a.prod()) == 24
    assert to_list(a.prod(dim=0)) == [3, 8]
    assert to_list(a.prod(dim=1, keepdim=True)) == [[2], [12]]
    assert to_list(a.prod(dim=-1)) == [2, 12]


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_view_flat_handles_non_contiguous(backend_name, Backend):
    # swapaxes produces a non-contiguous view for some backends (e.g., PyTorch)
    t = Backend.tensor([[1, 2], [3, 4]]).swapaxes(0, 1)
    flat = t.view_flat()
    assert flat.tolist() == [1, 3, 2, 4]


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_transpose_negative_indices(backend_name, Backend):
    t = Backend.tensor([[1, 2, 3], [4, 5, 6]])
    neg = t.transpose(-2, -1)
    pos = t.transpose(0, 1)
    assert neg.tolist() == pos.tolist()


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_transpose_invalid_indices(backend_name, Backend):
    t = Backend.tensor([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        t.transpose(2, 0)
    with pytest.raises(ValueError):
        t.transpose(0, 2)
    with pytest.raises(ValueError):
        t.transpose(-3, 0)
    with pytest.raises(ValueError):
        t.transpose(0, -3)


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_swapaxes_negative_indices(backend_name, Backend):
    t = Backend.tensor([[1, 2, 3], [4, 5, 6]])
    neg = t.swapaxes(-2, -1)
    pos = t.swapaxes(0, 1)
    assert neg.tolist() == pos.tolist()


@pytest.mark.parametrize("backend_name,Backend", BACKENDS)
def test_swapaxes_invalid_indices(backend_name, Backend):
    t = Backend.tensor([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        t.swapaxes(2, 0)
    with pytest.raises(ValueError):
        t.swapaxes(0, 2)
    with pytest.raises(ValueError):
        t.swapaxes(-3, 0)
    with pytest.raises(ValueError):
        t.swapaxes(0, -3)
