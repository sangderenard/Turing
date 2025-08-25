import pytest
from src.common.tensors.autograd import autograd, GradTape

try:  # NumPy backend is optional
    from src.common.tensors.numpy_backend import NumPyTensorOperations as Tensor
except Exception:  # pragma: no cover - optional dependency
    Tensor = None  # type: ignore


@pytest.fixture(autouse=True)
def _reset_tape():
    autograd.tape = GradTape()
    yield
    autograd.tape = GradTape()


def _tensor(data):
    t = Tensor.tensor_from_list(data)
    t.requires_grad_(True)
    return t


@pytest.mark.skipif(Tensor is None, reason="NumPy backend not available")
def test_required_cache_simple_chain():
    a = _tensor([1.0, 2.0])
    b = _tensor([3.0, 4.0])
    c = _tensor([5.0, 6.0])
    inter = a * b
    result = inter + c
    required = autograd.tape.required_cache(result)
    assert required == {id(a), id(b)}
