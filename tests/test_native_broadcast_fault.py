"""A failed authored broadcast law stops native elementwise execution."""

from src.compiler.native_faults import (
    BROADCAST_INCOMPATIBLE_SHAPE,
    NativeValueError,
    RESHAPE_INVALID_SHAPE,
    validation_fault,
)


def test_broadcast_incompatibility_has_distinct_value_error_code():
    assert BROADCAST_INCOMPATIBLE_SHAPE == 100
    assert BROADCAST_INCOMPATIBLE_SHAPE != RESHAPE_INVALID_SHAPE
    fault = validation_fault(BROADCAST_INCOMPATIBLE_SHAPE, "rank2 and rank1")
    assert isinstance(fault, NativeValueError)
    assert "incompatible broadcast shapes" in str(fault)
    assert "rank2 and rank1" in str(fault)
