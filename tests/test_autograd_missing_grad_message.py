import pytest
from src.common.tensors import AbstractTensor
from src.common.tensors.abstraction import tensor_identity


def test_grad_error_identifies_missing_input():
    autograd = AbstractTensor.autograd
    autograd.tape._nodes.clear()

    a = AbstractTensor.tensor([1.0])
    b = AbstractTensor.tensor([2.0])
    a.requires_grad = True
    b.requires_grad = True

    y = a * 3.0

    with pytest.raises(ValueError) as excinfo:
        autograd.grad(y, [a, b], allow_unused=False)
    msg = str(excinfo.value)
    assert "index 1" in msg
    assert str(tensor_identity(b)) in msg
