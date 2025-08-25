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
def test_add_records_metadata():
    a = _tensor([1, 2])
    b = _tensor([3, 4])
    c = a + b
    node = autograd.tape.node(c)
    assert node is not None
    ctx = node.ctx
    assert ctx["input_dtypes"] == [a.dtype, b.dtype]
    assert ctx["result_dtype"] == c.dtype
    assert ctx["input_devices"] == [a.device, b.device]
    assert ctx["result_device"] == c.device
    assert ctx["input_backends"] == [type(a).__name__, type(b).__name__]
    assert ctx["result_backend"] == type(c).__name__
    assert ctx["input_strides"] == [a.data.strides, b.data.strides]
    assert ctx["result_strides"] == c.data.strides
    assert ctx["params"] == {}


@pytest.mark.skipif(Tensor is None, reason="NumPy backend not available")
def test_manual_params_recording():
    a = _tensor([1, 2])
    b = _tensor([3, 4])
    with autograd.no_grad():
        res = a + b
    autograd.record("add", [a, b], res, params={"alpha": 2})
    node = autograd.tape.node(res)
    assert node is not None
    assert node.ctx["params"] == {"alpha": 2}
    assert node.ctx["input_dtypes"] == [a.dtype, b.dtype]
    assert node.ctx["result_dtype"] == res.dtype

