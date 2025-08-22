from src.common.tensors import AbstractTensor


def test_from_nested_basic():
    t = AbstractTensor.from_nested([[1, 2], [3, 4]])
    assert t.shape == (2, 2)
    assert t.tolist() == [[1, 2], [3, 4]]


def test_from_nested_with_existing_tensors():
    a = AbstractTensor.tensor([1, 2])
    b = AbstractTensor.tensor([3, 4])
    t = AbstractTensor.from_nested([a, b])
    assert t.shape == (2, 2)
    assert t.tolist() == [[1, 2], [3, 4]]


def test_from_nested_ragged_padding():
    t = AbstractTensor.from_nested([[1, 2], [3]])
    assert t.shape == (2, 2)
    assert t.tolist() == [[1, 2], [3, 0]]


def test_get_tensor_dispatches_nested():
    t = AbstractTensor.get_tensor([1, [2, 3]])
    assert t.shape == (2, 2)
    assert t.tolist() == [[1, 0], [2, 3]]
