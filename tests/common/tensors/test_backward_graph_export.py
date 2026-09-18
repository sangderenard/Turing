import pytest
import networkx as nx
from src.common.tensors.abstraction import tensor_identity

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
def test_export_backward_graph_structure():
    a = _tensor([1.0, 2.0])
    b = _tensor([3.0, 4.0])
    c = _tensor([5.0, 6.0])
    inter = a * b
    result = inter + c

    g = autograd.tape.export_backward_graph(result)
    assert isinstance(g, nx.DiGraph)

    assert g.nodes[tensor_identity(result)]["op"] == "add"
    assert g.nodes[tensor_identity(inter)]["op"] == "mul"
    assert set(g.nodes[tensor_identity(inter)]["required"]) == {tensor_identity(a), tensor_identity(b)}
    assert g.nodes[tensor_identity(result)]["required"] == []

    assert g.has_edge(tensor_identity(result), tensor_identity(inter))
    assert g.has_edge(tensor_identity(result), tensor_identity(c))
    assert g.has_edge(tensor_identity(inter), tensor_identity(a))
    assert g.has_edge(tensor_identity(inter), tensor_identity(b))
