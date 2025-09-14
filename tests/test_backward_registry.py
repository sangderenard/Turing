import numpy as np
import pytest

from src.common.tensors.abstraction import AbstractTensor, get_backward_tool
from src.common.tensors.backward import BackwardRegistry, BACKWARD_REGISTRY
from src.common.tensors.backward_registry import T as transpose_helper
from src.common.tensors.numpy_backend import NumPyTensorOperations as T


def test_get_backward_tool_add():
    tool = get_backward_tool(['add'])
    g = T.tensor([1.0, 2.0])
    x = T.tensor([3.0, 3.0])
    y = T.tensor([4.0, 5.0])
    g1, g2 = tool(g, x, y)
    assert np.allclose(g1.data, np.array([1.0, 2.0]))
    assert np.allclose(g2.data, np.array([1.0, 2.0]))


def test_backward_pipeline_sequence():
    def f1(x):
        return x + 1

    def f2(x):
        return x * 2

    reg = BackwardRegistry()
    reg.register('f1', f1)
    reg.register('f2', f2)
    pipeline = reg.build(['f1', 'f2'])
    result = pipeline(3)
    assert isinstance(result, AbstractTensor)
    assert np.allclose(result.data, 8)


def test_T_handles_1d_input():
    x = T.tensor([1.0, 2.0, 3.0])
    y = transpose_helper(x)
    assert y.shape == (3, 1)
    assert np.allclose(y.data, np.array([[1.0], [2.0], [3.0]]))


def test_backward_wraps_scalar_inputs():
    bw_add = BACKWARD_REGISTRY._methods["add"]
    x = AbstractTensor.tensor([1.0, 2.0])
    gx, gy = bw_add(1.0, x, 3.0)
    assert isinstance(gx, AbstractTensor)
    assert isinstance(gy, AbstractTensor)
    assert np.allclose(gx.data, np.array([1.0, 1.0]))
    assert np.allclose(gy.data, np.array(1.0))


def test_backward_rejects_non_tensor_inputs():
    bw_add = BACKWARD_REGISTRY._methods["add"]
    x = AbstractTensor.tensor([1.0, 2.0])
    with pytest.raises(TypeError):
        bw_add(1.0, x, [1, 2])
