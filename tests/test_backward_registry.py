import numpy as np

from src.common.tensors.abstraction import get_backward_tool
from src.common.tensors.backward import BackwardRegistry
from src.common.tensors.backward_registry import T as transpose_helper
from src.common.tensors.numpy_backend import NumPyTensorOperations as T


def test_get_backward_tool_add():
    tool = get_backward_tool(['add'])
    g1, g2 = tool(np.array([1.0, 2.0]), np.array([3.0, 3.0]), np.array([4.0, 5.0]))
    assert np.allclose(g1, np.array([1.0, 2.0]))
    assert np.allclose(g2, np.array([1.0, 2.0]))


def test_backward_pipeline_sequence():
    def f1(x):
        return x + 1

    def f2(x):
        return x * 2

    reg = BackwardRegistry()
    reg.register('f1', f1)
    reg.register('f2', f2)
    pipeline = reg.build(['f1', 'f2'])
    assert pipeline(3) == 8


def test_T_handles_1d_input():
    x = T.tensor([1.0, 2.0, 3.0])
    y = transpose_helper(x)
    assert y.shape == (3, 1)
    assert np.allclose(y.data, np.array([[1.0], [2.0], [3.0]]))
