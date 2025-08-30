import numpy as np
from src.common.tensors.abstraction import AbstractTensor


def test_eye_three_plus_batch_dims():
    E = AbstractTensor.eye(4, batch_shape=(2, 3, 4))
    assert E.get_shape() == (2, 3, 4, 4, 4)
    expected = np.eye(4, dtype=E.data.dtype)
    expected = expected.reshape((1, 1, 1, 4, 4))
    expected = np.broadcast_to(expected, (2, 3, 4, 4, 4))
    assert np.allclose(E.data, expected)
