import numpy as np
import pytest

from src.common.tensors import AbstractTensor
from src.common.tensors import AbstractTensor as AT
from src.compiler.fused_program_python_backend import lower_functions_to_numpy


def numeric(value):
    return AT.tensor(value, dtype='complex128', device='cpu')


def shapes(a, b):
    return AT.broadcast_shapes(a, b)


def linear_solve(value):
    matrix = AT.eye(2, dtype='complex128', device='cpu')
    zero = AT.zeros((2,), dtype='complex128', device='cpu')
    ones = AT.ones((2,), dtype='complex128', device='cpu')
    rhs = (value.to('complex128') + zero) * ones
    return AT.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1).detach().clone()


def tensor_metadata(value):
    return value.dim(), value.numel(), value.abs()


class Base:
    def __init__(self):
        self.initialized = True


class Child(Base):
    def __init__(self, value):
        super().__init__()
        self.value = value


def lower(*roots, subclasses=None):
    product = lower_functions_to_numpy(roots, namespace=globals(), owned=lambda _: False,
                                      subclass_methods=subclasses)
    scope = {}
    exec(product.source, scope)
    return scope


@pytest.mark.parametrize('value', [0.4, 1 + 2j, [0.2, 0.5], np.array([0.3])])
def test_dtype_device_and_alias(value):
    actual = lower(numeric)['numeric'](value)
    expected = numeric(value).data
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_broadcast():
    assert lower(shapes)['shapes']((1, 2, 1), (3, 1, 4)) == (3, 2, 4)


def test_existing_complex_tensor_boundary_preserves_imaginary_component():
    value = AT.tensor([0.4 + 0.3j], dtype='complex128')
    np.testing.assert_array_equal(lower(numeric)['numeric'](value), value.data)


def test_lowered_constructor_preserves_authored_super():
    cls = lower(subclasses={Child: ('__init__',)})['Child']
    value = cls(7)
    assert isinstance(value, Child)
    assert value.initialized and value.value == 7


def test_linear_complex_solve_matches_abstract():
    raw = np.array([0.4 + 0.3j, 0.8 - 0.1j])
    actual = lower(linear_solve)['linear_solve'](raw)
    expected = linear_solve(AT.tensor(raw)).data
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert not np.shares_memory(actual, raw)


def test_tensor_metadata_matches_abstract():
    raw = np.array([[-0.5, 0.2]])
    actual = lower(tensor_metadata)['tensor_metadata'](raw)
    expected = tensor_metadata(AT.tensor(raw))
    assert actual[:2] == expected[:2]
    np.testing.assert_array_equal(actual[2], expected[2].data)
