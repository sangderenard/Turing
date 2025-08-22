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
