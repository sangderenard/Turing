"""Runtime reshape validation uses the native fault catalog."""

import pytest

from src.compiler.native_faults import (
    NativeValueError,
    RESHAPE_INVALID_SHAPE,
    validation_fault,
)


@pytest.mark.parametrize("detail", (
    "width is zero", "width is negative", "count is not divisible",
))
def test_runtime_reshape_fault_is_a_python_value_error(detail):
    error = validation_fault(RESHAPE_INVALID_SHAPE, detail)
    assert isinstance(error, NativeValueError)
    assert isinstance(error, ValueError)
    assert detail in str(error)
    assert "reshape" in str(error)


def test_unknown_native_fault_remains_generic():
    assert type(validation_fault(RESHAPE_INVALID_SHAPE + 1, "unknown")) is RuntimeError
