import pytest
import networkx as nx

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


def _param(value):
    t = Tensor.tensor([value])
    t.requires_grad_(True)
    return t


@pytest.mark.skipif(Tensor is None, reason="NumPy backend not available")
def test_export_training_state():
    w = _param(1.0)
    b = _param(0.5)
    x = Tensor.tensor([2.0])
    y = Tensor.tensor([5.0])
    pred = w * x + b
    err = pred - y
    loss = err * err
    autograd.tape.mark_loss(loss)

    fwd, bwd, params_tensor, id_map = autograd.tape.export_training_state()
    assert isinstance(fwd, nx.DiGraph)
    assert isinstance(bwd, nx.DiGraph)
    assert params_tensor is not None
    assert params_tensor.get_shape()[0] == 2
    assert id_map[id(w)] != id_map[id(b)]
    assert fwd.nodes[id(w)]["param_id"] == id_map[id(w)]
    assert fwd.nodes[id(w)]["stateful"] is True
    assert bwd.nodes[id(w)]["param_id"] == id_map[id(w)]
    assert bwd.nodes[id(loss)]["loss"] is True


@pytest.mark.skipif(Tensor is None, reason="NumPy backend not available")
def test_train_updates_parameter():
    w = _param(0.0)
    x = Tensor.tensor([1.0])
    y = Tensor.tensor([2.0])

    def loss_fn():
        pred = w * x
        loss = (pred - y) * (pred - y)
        return loss

    proc = autograd.train(loss_fn, epochs=1, lr=0.1, params=[w])
    assert isinstance(proc.forward_graph, nx.DiGraph)
    assert abs(w.data[0] - 0.4) < 1e-6
