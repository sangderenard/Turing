from src.common.tensors.abstraction import AbstractTensor


def test_pad_cat_basic():
    a = AbstractTensor.tensor([[1, 2], [3, 4]])
    b = AbstractTensor.tensor([[5, 6, 7]])
    result = AbstractTensor.pad_cat([a, b], dim=0, pad_value=0)
    assert result.tolist() == [[1, 2, 0], [3, 4, 0], [5, 6, 7]]


def test_pad_cat_dim1_padding():
    a = AbstractTensor.tensor([[1, 2], [3, 4], [5, 6]])
    b = AbstractTensor.tensor([[7], [8]])
    result = AbstractTensor.pad_cat([a, b], dim=1, pad_value=0)
    assert result.tolist() == [[1, 2, 7], [3, 4, 8], [5, 6, 0]]
