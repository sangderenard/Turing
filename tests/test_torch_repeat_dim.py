import pytest

torch = pytest.importorskip("torch")
from src.common.tensors.torch_backend import PyTorchTensorOperations


def test_repeat_clone_when_no_repeats():
    t = PyTorchTensorOperations.tensor_from_list([[1, 2], [3, 4]])
    r = t.repeat()
    assert torch.equal(r.data, t.data)
    r.data[0, 0] = 99
    assert t.data[0, 0] == 1
    r2 = t.repeat(repeats=())
    assert torch.equal(r2.data, t.data)
    r2.data[0, 0] = 88
    assert t.data[0, 0] == 1


def test_repeat_scalar():
    s = PyTorchTensorOperations.tensor_from_list(5)
    r = s.repeat(repeats=3, dim=0)
    assert r.data.tolist() == [5, 5, 5]


def test_repeat_along_dims():
    t = PyTorchTensorOperations.tensor_from_list([[1, 2], [3, 4]])
    r0 = t.repeat(repeats=2, dim=0)
    assert r0.data.tolist() == [[1, 2], [3, 4], [1, 2], [3, 4]]
    r1 = t.repeat(repeats=2, dim=1)
    assert r1.data.tolist() == [[1, 2, 1, 2], [3, 4, 3, 4]]
