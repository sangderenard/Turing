from src.common.tensors.abstraction import AbstractTensor


def test_unravel_index_accepts_python_scalar():
    coordinates = AbstractTensor.unravel_index(17, (3, 3, 3))
    assert tuple(int(axis) for axis in coordinates) == (1, 2, 2)


def test_reduced_scalars_remain_closed_under_arithmetic():
    values = AbstractTensor.get_tensor([1.0, 4.0, 2.0])
    extent = values.max() - values.min()
    assert extent.item() == 3.0
