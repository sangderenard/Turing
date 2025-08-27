from src.common.tensors.numpy_backend import NumPyTensorOperations as T
import pytest


def test_grads_returns_grad_attribute():
    x = T.tensor_from_list([[1.0, 2.0], [3.0, 4.0]])
    x.requires_grad_(True)
    (x * x).sum().backward()
    assert x.grads() is x.grad


def test_grads_fallback_to_gW():
    w = T.tensor_from_list([[1.0, 2.0], [3.0, 4.0]])
    w._grad = None
    w.gW = T.ones_like(w)
    assert w.grads() is w.gW


def test_grads_raises_when_missing():
    z = T.tensor_from_list([[0.0]])
    with pytest.raises(AttributeError):
        z.grads()


def test_grads_invokes_autograd_when_grad_absent():
    x = T.tensor_from_list([1.0, 2.0])
    x.requires_grad_(True)
    loss = (x * x).sum()
    # Register loss but do not run backward to keep x.grad unset
    x.autograd.tape.mark_loss(loss)
    assert x.grad is None
    g = x.grads()
    # gradient of sum(x^2) is 2*x
    expected = (x * 2).data
    assert g.data.tolist() == expected.tolist()


def test_grads_prefers_autograd_over_legacy_attributes():
    x = T.tensor_from_list([1.0, 2.0])
    x.requires_grad_(True)
    x.gW = T.ones_like(x)  # legacy attribute that should be ignored
    loss = (x * x).sum()
    x.autograd.tape.mark_loss(loss)
    g = x.grads()
    expected = (x * 2).data
    assert g.data.tolist() == expected.tolist()
    assert g is not x.gW
