"""In-memory tensor backend routed through the nodus arena.

Storage, shape, and reduction logic are inherited wholesale from
:class:`NumPyTensorOperations` -- ``.data`` stays a plain ``numpy.ndarray``, so
every other piece of code that assumes that representation keeps working.
Only the elementwise operator dispatch (``_apply_operator__``, the single
designated hook a backend implements) is overridden, to route the operations
nodus's ``InMemoryBackend`` actually exposes -- the ``CanonicalOp`` set in
:mod:`nodus_arena` -- through the arena instead of through NumPy.

Anything the ABI does not cover, or does not cover *safely* for the operands
in hand, falls back to the inherited NumPy implementation: matmul (not
elementwise), mismatched shapes (the ABI has no broadcasting -- it reads
whatever bytes sit at the mismatched extent instead of refusing, so a shape
mismatch must never reach ``nodus_tensor_binary``), and dtypes nodus has no
code for.
"""

from __future__ import annotations

import numpy as np

from .. import abstraction as _abstraction
from ..numpy_backend import NumPyTensorOperations
from . import nodus_arena as _na

_NUMPY_TO_NODUS = {
    np.dtype(np.float32): _na.F32,
    np.dtype(np.float64): _na.F64,
    np.dtype(np.int8): _na.I8,
    np.dtype(np.int16): _na.I16,
    np.dtype(np.int32): _na.I32,
    np.dtype(np.int64): _na.I64,
    np.dtype(np.uint8): _na.U8,
    np.dtype(np.uint16): _na.U16,
    np.dtype(np.uint32): _na.U32,
    np.dtype(np.uint64): _na.U64,
    np.dtype(np.bool_): _na.BOOL,
}

# Measured against the live ABI (see test in test_nodus_backend.py): a
# comparison op comes back with 0/1 written in the *input* dtype, not a BOOL
# tensor -- nodus_tensor_binary refuses a BOOL output (status -5,
# "unsupported"). Cast to bool on the way out so this backend still returns
# what NumPyTensorOperations returns for the same op.
_BOOL_RESULT_OPS = frozenset(
    {
        "less", "less_equal", "greater", "greater_equal", "equal",
        "not_equal", "isfinite", "isnan", "isinf", "logical_not",
    }
)

_INPLACE_TO_BASE = {
    "iadd": "add", "isub": "sub", "imul": "mul", "itruediv": "truediv",
    "ifloordiv": "floordiv", "imod": "mod", "ipow": "pow",
    "imatmul": "matmul",
}


class NodusTensorOperations(NumPyTensorOperations):
    """NumPy-shaped storage, nodus-arena elementwise compute."""

    def _apply_operator__(self, op: str, left, right):
        canonical = _INPLACE_TO_BASE.get(op, op)
        if canonical not in _na.CANONICAL_OPS:
            return super()._apply_operator__(op, left, right)

        arena = _na.connect(warn=False)
        if arena is None:
            return super()._apply_operator__(op, left, right)

        try:
            if canonical in _na.UNARY_OPS:
                result = self._unary_via_nodus(arena, canonical, left)
            else:
                result = self._binary_via_nodus(arena, canonical, left, right)
        except (_na.NodusArenaError, _na.NodusArenaUnavailable, TypeError, ValueError):
            result = None

        if result is None:
            return super()._apply_operator__(op, left, right)
        if canonical in _BOOL_RESULT_OPS:
            result = result.astype(np.bool_)
        return result

    @staticmethod
    def _dtype_code(array: np.ndarray):
        if array.ndim == 0:
            # Rank-0 handling is untested against the ABI; fall back rather
            # than guess.
            return None
        return _NUMPY_TO_NODUS.get(array.dtype)

    @staticmethod
    def _push(arena, array: np.ndarray) -> int:
        contiguous = np.ascontiguousarray(array)
        code = NodusTensorOperations._dtype_code(contiguous)
        if code is None:
            raise TypeError(f"no nodus dtype for {array.dtype}")
        handle = arena.create(code, contiguous.shape)
        arena.write_bytes(handle, contiguous.tobytes())
        return handle

    @staticmethod
    def _pull(arena, handle: int, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
        # Read exactly the payload nodus_tensor_describe reports, not the
        # (possibly larger, 64-byte-aligned) lease nodus_tensor_map would
        # hand back -- that mismatch is the exact trap the coverage audit in
        # nodus_arena.py flags for any future map() caller.
        desc = arena.describe(handle)
        raw = arena.read_bytes(handle, int(desc.total_bytes))
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

    def _unary_via_nodus(self, arena, op: str, a):
        if not isinstance(a, np.ndarray):
            return None
        handle = self._push(arena, a)
        try:
            out = arena.unary(op, handle)
            try:
                return self._pull(arena, out, a.dtype, a.shape)
            finally:
                arena.destroy(out)
        finally:
            arena.destroy(handle)

    def _binary_via_nodus(self, arena, op: str, a, b):
        a_is_array = isinstance(a, np.ndarray)
        b_is_array = isinstance(b, np.ndarray)

        if a_is_array and b_is_array:
            if a.shape != b.shape or a.dtype != b.dtype:
                return None
            left = self._push(arena, a)
            right = self._push(arena, b)
            try:
                out = arena.binary(op, left, right)
                try:
                    return self._pull(arena, out, a.dtype, a.shape)
                finally:
                    arena.destroy(out)
            finally:
                arena.destroy(left)
                arena.destroy(right)

        if a_is_array or b_is_array:
            array, scalar, scalar_on_left = (
                (a, b, False) if a_is_array else (b, a, True)
            )
            scalar = float(scalar)
            handle = self._push(arena, array)
            try:
                out = arena.scalar(op, handle, scalar, scalar_on_left=scalar_on_left)
                try:
                    return self._pull(arena, out, array.dtype, array.shape)
                finally:
                    arena.destroy(out)
            finally:
                arena.destroy(handle)

        return None


_abstraction.register_backend("nodus", NodusTensorOperations)
_abstraction.register_backend("in_memory", NodusTensorOperations)

__all__ = ["NodusTensorOperations"]
