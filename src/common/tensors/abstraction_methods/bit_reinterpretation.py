"""Same-width bit reinterpretation: one value's bits read as another type.

``float_bits_to_uint(1.0)`` is ``0x3F800000``: no value conversion, the
32 bits are kept and only their type changes.  These are GLSL's
``floatBitsToUint``/``floatBitsToInt``/``uintBitsToFloat``/
``intBitsToFloat``, C's ``memcpy`` into the other type and LLVM's
``bitcast``.  Each is a direct ``BitCast`` in the compiler's operation table
(``python_identity_programs``), which every backend emits in its own
native form.

Reversal.  Each function's exact inverse is the opposite one
(``BIT_REINTERPRETATION_INVERSES``): the bits are unchanged, so the round
trip is the identity on every bit pattern.  (GLSL leaves the float a
NaN/Inf pattern produces unspecified, so a GPU need not preserve a NaN
payload.)  A reinterpretation has no derivative, since ``1.0`` and
``1065353216`` are not related by any slope, so graph reversal treats it as
non-differentiable (``backward_registry["bitcast"]``) rather than as the
identity adjoint of a value cast.
"""

from __future__ import annotations

import numpy as np

#: The function name -> (source dtype, target dtype) it reinterprets.
BIT_REINTERPRETATIONS = {
    "float_bits_to_uint": ("float32", "uint32"),
    "float_bits_to_int": ("float32", "int32"),
    "uint_bits_to_float": ("uint32", "float32"),
    "int_bits_to_float": ("int32", "float32"),
    "int_bits_to_uint": ("int32", "uint32"),
    "uint_bits_to_int": ("uint32", "int32"),
}

#: Each reinterpretation's exact inverse.
BIT_REINTERPRETATION_INVERSES = {
    "float_bits_to_uint": "uint_bits_to_float",
    "uint_bits_to_float": "float_bits_to_uint",
    "float_bits_to_int": "int_bits_to_float",
    "int_bits_to_float": "float_bits_to_int",
    "int_bits_to_uint": "uint_bits_to_int",
    "uint_bits_to_int": "int_bits_to_uint",
}


def _reinterpret(value, source: str, target: str):
    held = np.asarray(value, dtype=np.dtype(source))
    reinterpreted = held.view(np.dtype(target))
    return reinterpreted[()] if reinterpreted.ndim == 0 else reinterpreted


def float_bits_to_uint(value):
    """The float32 ``value``'s bits as a uint32 (GLSL ``floatBitsToUint``)."""

    return _reinterpret(value, "float32", "uint32")


def float_bits_to_int(value):
    """The float32 ``value``'s bits as an int32 (GLSL ``floatBitsToInt``)."""

    return _reinterpret(value, "float32", "int32")


def uint_bits_to_float(value):
    """The uint32 ``value``'s bits as a float32 (GLSL ``uintBitsToFloat``)."""

    return _reinterpret(value, "uint32", "float32")


def int_bits_to_float(value):
    """The int32 ``value``'s bits as a float32 (GLSL ``intBitsToFloat``)."""

    return _reinterpret(value, "int32", "float32")


def int_bits_to_uint(value):
    """The int32 ``value``'s bits as a uint32 (GLSL ``uint(int)``)."""

    return _reinterpret(value, "int32", "uint32")


def uint_bits_to_int(value):
    """The uint32 ``value``'s bits as an int32 (GLSL ``int(uint)``)."""

    return _reinterpret(value, "uint32", "int32")


__all__ = [
    "BIT_REINTERPRETATIONS", "BIT_REINTERPRETATION_INVERSES",
    "float_bits_to_int", "float_bits_to_uint",
    "int_bits_to_float", "int_bits_to_uint",
    "uint_bits_to_float", "uint_bits_to_int",
]
