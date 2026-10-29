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

#: ``a ** b`` / ``pow(a, b, m)`` typed int reached ``b < 0``.  Python's
#: result is a float (``2 ** -1 == 0.5``) or, with a modulus, a modular
#: inverse; the int-typed native power reports it instead of returning an
#: int Python never computes.
INT_POWER_NEGATIVE_EXPONENT = 89

#: ``pow(a, b, 0)``: Python raises ``ValueError``.
POW_MODULUS_ZERO = 90

#: ``a / b`` on ints where ``a`` or ``b`` is not exactly a float64.  Python
#: returns the correctly rounded quotient of the exact ints; ``float(a) /
#: float(b)`` rounds twice and can differ in the last place.  Until that
#: quotient is computed on Precision limbs, the native module reports it.
INT_TRUE_DIVISION_INEXACT = 91

#: ``0.0 ** y`` (``-0.0`` too) with a finite ``y < 0``: Python raises
#: ``ZeroDivisionError`` ("0.0 cannot be raised to a negative power").
FLOAT_POW_ZERO_NEGATIVE = 92

#: ``x ** y`` on floats with finite ``x < 0`` and finite non-integer ``y``:
#: Python returns a ``complex``.  A float-typed native power cannot carry
#: it, so the module reports it by name rather than return a float Python
#: never computes (whether to lower it to complex is the user's choice).
FLOAT_POW_COMPLEX_RESULT = 93

#: ``math.pow`` (and ``math`` functions generally) outside the domain:
#: ``ValueError("math domain error")``.
FLOAT_MATH_DOMAIN = 94

#: ``x ** y`` on floats whose finite operands give a result past the
#: largest double: ``OverflowError((34, 'Numerical result out of range'))``.
FLOAT_POW_OVERFLOW = 95

#: ``math.pow`` overflow: ``OverflowError("math range error")``.
FLOAT_MATH_RANGE = 96


class NativeIndexError(RuntimeError, IndexError):
    """``IndexError`` raised from a native validation code."""


class NativeIntOverflow(RuntimeError, OverflowError):
    """An int64 result past int64 that Python would have kept exact."""


class NativeValueError(RuntimeError, ValueError):
    """``ValueError`` raised from a native validation code."""


class NativeZeroDivisionError(RuntimeError, ZeroDivisionError):
    """``ZeroDivisionError`` raised from a native validation code."""


class NativeIntPowerNegativeExponent(RuntimeError):
    """An int-typed ``a ** b`` whose exponent was negative at run time.

    Python returns a float there (or a modular inverse for three-argument
    ``pow``); the program's int type cannot carry it.
    """


class NativeFloatOverflow(RuntimeError, OverflowError):
    """A float result Python reports as ``OverflowError``."""


class NativeFloatPowComplexResult(RuntimeError):
    """A negative float ``**`` a non-integer float: complex in Python."""


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
    if int(code) == INT_POWER_NEGATIVE_EXPONENT:
        return NativeIntPowerNegativeExponent(
            "int ** negative int is not an int in Python "
            f"({runtime_message})"
        )
    if int(code) == POW_MODULUS_ZERO:
        return NativeValueError(
            f"pow() 3rd argument cannot be 0 ({runtime_message})"
        )
    if int(code) == FLOAT_POW_ZERO_NEGATIVE:
        return NativeZeroDivisionError(
            f"0.0 cannot be raised to a negative power ({runtime_message})"
        )
    if int(code) == FLOAT_POW_COMPLEX_RESULT:
        return NativeFloatPowComplexResult(
            "negative float ** non-integer float is complex in Python "
            f"({runtime_message})"
        )
    if int(code) == FLOAT_MATH_DOMAIN:
        return NativeValueError(f"math domain error ({runtime_message})")
    if int(code) == FLOAT_POW_OVERFLOW:
        return NativeFloatOverflow(
            f"(34, 'Numerical result out of range') ({runtime_message})"
        )
    if int(code) == FLOAT_MATH_RANGE:
        return NativeFloatOverflow(f"math range error ({runtime_message})")
    return RuntimeError(runtime_message)
