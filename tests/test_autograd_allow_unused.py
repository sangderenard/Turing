import pytest
from src.common.tensors import AbstractTensor


def test_grad_allow_unused_skips_strict_connectivity():
    autograd = AbstractTensor.autograd
    prev_strict = autograd.strict
    autograd.strict = True
    autograd.tape._nodes.clear()

    try:
        a = AbstractTensor.tensor([1.0])
        b = AbstractTensor.tensor([2.0])
        a.requires_grad = True
        b.requires_grad = True

        y = a * 3.0
        grads = autograd.grad(y, [a, b], allow_unused=True)
        assert grads[0] is not None
        assert grads[1] is None
    finally:
        autograd.strict = prev_strict
