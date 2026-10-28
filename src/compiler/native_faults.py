"""Python exceptions a native module reports through ``turing_validation_error``.

The native entry points record one integer code.  A code that stands for a
Python exception is raised as that exception (still a ``RuntimeError``, so
callers that caught the generic failure keep catching it).
"""

from __future__ import annotations

#: ``pop`` from an empty list, or a list read/write outside ``-len <= i < len``.
LIST_INDEX_OUT_OF_RANGE = 71

#: A subscript read of a mapping key it does not hold (``required_key_absent``).
MAPPING_KEY_ABSENT = 84

#: A Python ``int`` result no interval proves inside int64 left int64.  Python
#: would have continued exactly; until that result continues as Precision
#: limbs, the native module reports it rather than return a wrapped value.
INT_OVERFLOW = 86

#: ``a << b`` or ``a >> b`` with ``b < 0``: Python raises ``ValueError``.
NEGATIVE_SHIFT_COUNT = 87

#: ``a // 0`` or ``a % 0`` on ints: Python raises ``ZeroDivisionError``; the
#: native divide would trap the host process instead.
INT_ZERO_DIVISION = 88

#: ``a / b`` on ints where ``a`` or ``b`` is not exactly a float64.  Python
#: returns the correctly rounded quotient of the exact ints; ``float(a) /
#: float(b)`` rounds twice and can differ in the last place.  Until that
#: quotient is computed on Precision limbs, the native module reports it.
INT_TRUE_DIVISION_INEXACT = 89


class NativeIndexError(RuntimeError, IndexError):
    """``IndexError`` raised from a native validation code."""


class NativeIntOverflow(RuntimeError, OverflowError):
    """An int64 result past int64 that Python would have kept exact."""


class NativeValueError(RuntimeError, ValueError):
    """``ValueError`` raised from a native validation code."""


class NativeZeroDivisionError(RuntimeError, ZeroDivisionError):
    """``ZeroDivisionError`` raised from a native validation code."""


class NativeKeyError(RuntimeError, KeyError):
    """``KeyError`` raised from a native validation code."""

    def __str__(self) -> str:
        return RuntimeError.__str__(self)


def validation_fault(code: int, runtime_message: str) -> RuntimeError:
    """The exception for one recorded validation code."""

    if int(code) == LIST_INDEX_OUT_OF_RANGE:
        return NativeIndexError(f"list index out of range ({runtime_message})")
    if int(code) == MAPPING_KEY_ABSENT:
        return NativeKeyError(f"key not in mapping ({runtime_message})")
    if int(code) == INT_OVERFLOW:
        return NativeIntOverflow(
            f"int result left int64; no interval proves it ({runtime_message})"
        )
    if int(code) == NEGATIVE_SHIFT_COUNT:
        return NativeValueError(f"negative shift count ({runtime_message})")
    if int(code) == INT_TRUE_DIVISION_INEXACT:
        return NativeIntOverflow(
            "int / int operand is not exact in float64; Python's correctly "
            f"rounded quotient needs Precision limbs ({runtime_message})"
        )
    if int(code) == INT_ZERO_DIVISION:
        return NativeZeroDivisionError(
            f"integer division or modulo by zero ({runtime_message})"
        )
    return RuntimeError(runtime_message)
