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
    t = Tensor.tensor(data)
    t.requires_grad_(True)
    return t


@pytest.mark.skipif(Tensor is None, reason="NumPy backend not available")
def test_cache_tags_and_zero_grad():
    a = _tensor([1.0, 2.0])
    b = _tensor([3.0, 4.0])
    c = _tensor([5.0, 6.0])
    inter = a * b
    result = inter + c
    tape = autograd.tape
    for t in (a, b):
        anns = tape.graph.nodes[id(t)].get("annotations", {})
        assert anns.get("cache") is True
    anns_c = tape.graph.nodes[id(c)].get("annotations", {})
    assert anns_c.get("cache") is not True
    a.zero_grad()
    b.zero_grad()
    for t in (a, b):
        anns = tape.graph.nodes[id(t)].get("annotations", {})
        assert anns.get("cache") is True
    a.zero_grad(clear_cache=True)
    anns = tape.graph.nodes[id(a)].get("annotations", {})
    assert "cache" not in anns
