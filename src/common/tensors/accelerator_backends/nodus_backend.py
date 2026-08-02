"""In-memory tensor backend routed through the nodus arena.

Storage, shape, and reduction logic are inherited wholesale from
:class:`NumPyTensorOperations` -- ``.data`` stays a plain ``numpy.ndarray``, so
every other piece of code that assumes that representation keeps working.
Only the elementwise operator dispatch (``_apply_operator__``, the single
designated hook a backend implements) is overridden, to route the operations
nodus's ``InMemoryBackend`` actually exposes -- the ``CanonicalOp`` set in
:mod:`nodus_arena` -- through the arena instead of through NumPy.

There are no silent fallbacks here. Every operation nodus implements goes to
nodus, and a failure is raised rather than answered by NumPy behind the
caller's back -- otherwise a broken or missing arena looks like nothing worse
than a slow run, and the matrix of what nodus actually serves cannot be
trusted. Only operations nodus genuinely does not implement reach the
inherited NumPy code, and each is named explicitly at the dispatch rather
than being caught by an ``except``.

Matmul is included: it is not elementwise and so is not a ``CanonicalOp``,
but ``tensor_matmul_f32``/``f64`` exist as the "in-memory backend only" dense
helpers in tensor_math.h, and ``nodus_tensor_matmul`` reaches them.
Broadcasting is nodus's own (verified against the ABI for row, column, and
outer broadcasts); only dtype promotion happens on this side, using NumPy's
rule, because the ABI takes one dtype for both operands.
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

class NodusUnsupported(NotImplementedError):
    """nodus genuinely cannot serve this operation on these operands.

    Raised rather than quietly answered by another backend: which operations
    nodus actually implements is exactly the thing this backend exists to
    report honestly.
    """


def _arena():
    """The connected arena, or a hard failure explaining why there isn't one.

    ``connect()`` returns None on an unavailable core, which is right for a
    caller deciding whether to use nodus at all. Here the decision is already
    made -- this backend was selected -- so a missing core is an error, not a
    cue to compute the answer somewhere else.
    """

    arena = _na.connect(warn=False)
    if arena is None:
        raise NodusUnsupported(
            "the nodus tensor core is not connected, so NodusTensorOperations "
            "cannot compute. Build the nodus_tensor_core target or set "
            "NODUS_TENSOR_CORE, or select a different backend explicitly with "
            "AbstractTensor.set_default_backend()."
        )
    return arena


def _promote_for(canonical: str, operand):
    """Widen an integer operand to f64 for an operation that is real-valued.

    Not a workaround for a nodus limitation -- it is the same promotion NumPy
    performs for these operations, moved to the point where the dtype is
    chosen. nodus refuses ``sqrt`` on an int32 tensor because an elementwise
    result carries its operand's dtype and ``sqrt(10)`` is not an int32; the
    answer is to ask it for the f64 the caller actually wants.
    """

    if canonical in _na.INTEGER_EXACT_OPS or not isinstance(operand, np.ndarray):
        return operand
    if operand.dtype.kind in "iub":
        return operand.astype(np.float64)
    return operand


_INPLACE_TO_BASE = {
    "iadd": "add", "isub": "sub", "imul": "mul", "itruediv": "truediv",
    "ifloordiv": "floordiv", "imod": "mod", "ipow": "pow",
    "imatmul": "matmul",
}


class NodusTensorOperations(NumPyTensorOperations):
    """NumPy-shaped storage, nodus-arena elementwise compute."""

    def _apply_operator__(self, op: str, left, right):
        canonical = _INPLACE_TO_BASE.get(op, op)

        # nodus is this backend's compute, not an optimization layered over
        # NumPy's. Every operation it covers goes to nodus and its failures
        # are raised, never swallowed: a quiet fall back to NumPy would make
        # a broken or missing arena look like nothing worse than a slow run,
        # and would misreport which operations nodus actually serves. Only
        # operations nodus genuinely does not implement reach super(), and
        # each of those is named below rather than reached by an except.
        if canonical == "matmul":
            # The connected arena ABI exposes matrix-matrix multiplication,
            # including the rank-1 views normalized below, but not NumPy's
            # batched matmul contract.  This is an explicit unsupported shape
            # class, so it belongs to the inherited implementation just like
            # other operations absent from the arena vocabulary.
            if np.asarray(left).ndim > 2 or np.asarray(right).ndim > 2:
                return super()._apply_operator__(op, left, right)
            return self._matmul_via_nodus(_arena(), left, right)

        if canonical not in _na.CANONICAL_OPS:
            # Not a canonical nodus operation at all (e.g. clamp, where, the
            # cast ops) -- NumPy's implementation is the definition here, not
            # a substitute for one nodus has.
            return super()._apply_operator__(op, left, right)

        arena = _arena()
        if canonical in _na.UNARY_OPS:
            result = self._unary_via_nodus(
                arena, canonical, _promote_for(canonical, left)
            )
        else:
            left, right = (
                _promote_for(canonical, left),
                _promote_for(canonical, right),
            )
            result = self._binary_via_nodus(arena, canonical, left, right)
        if canonical in _BOOL_RESULT_OPS:
            result = result.astype(np.bool_)
        return result

    @staticmethod
    def _dtype_code(array: np.ndarray) -> int:
        code = _NUMPY_TO_NODUS.get(array.dtype)
        if code is None:
            raise NodusUnsupported(
                f"nodus has no dtype for {array.dtype}; the arena cannot "
                "hold this tensor"
            )
        return code

    @staticmethod
    def _push(arena, array: np.ndarray) -> int:
        contiguous = np.ascontiguousarray(array)
        if contiguous.shape != array.shape:
            # ascontiguousarray promotes a 0-d array to shape (1,). Left
            # alone that pushes a rank-1 operand against a rank-0 output and
            # nodus refuses the pair, so the rank is restored here rather
            # than letting the ranks disagree.
            contiguous = contiguous.reshape(array.shape)
        handle = arena.create(
            NodusTensorOperations._dtype_code(contiguous), contiguous.shape
        )
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
        array = np.asarray(a)
        handle = self._push(arena, array)
        try:
            out = arena.unary(op, handle)
            try:
                return self._pull(arena, out, array.dtype, array.shape)
            finally:
                arena.destroy(out)
        finally:
            arena.destroy(handle)

    def _binary_via_nodus(self, arena, op: str, a, b):
        # A Python number stays a number: nodus_tensor_scalar exists so the
        # scalar side never has to be materialized as a tensor.
        a_is_array = isinstance(a, np.ndarray)
        b_is_array = isinstance(b, np.ndarray)
        if a_is_array != b_is_array:
            array, scalar, scalar_on_left = (
                (a, b, False) if a_is_array else (b, a, True)
            )
            handle = self._push(arena, array)
            try:
                out = arena.scalar(
                    op, handle, float(scalar), scalar_on_left=scalar_on_left
                )
                try:
                    return self._pull(arena, out, array.dtype, array.shape)
                finally:
                    arena.destroy(out)
            finally:
                arena.destroy(handle)

        left_array = np.asarray(a)
        right_array = np.asarray(b)
        # nodus broadcasts (verified against the ABI: row, column, and outer
        # broadcasts all agree with NumPy), so the shapes are handed over as
        # they are. Only the output extent has to be stated, and NumPy's own
        # rule decides it so the two cannot disagree about what broadcasting
        # means.
        result_shape = np.broadcast_shapes(left_array.shape, right_array.shape)
        # Dtypes are promoted before the boundary rather than inside it: the
        # ABI takes one dtype for both operands, and doing the promotion here
        # with NumPy's rule keeps the result type identical to what
        # NumPyTensorOperations would have produced.
        result_dtype = np.result_type(left_array, right_array)
        left = self._push(arena, left_array.astype(result_dtype, copy=False))
        right = self._push(arena, right_array.astype(result_dtype, copy=False))
        try:
            out = arena.create(self._dtype_code(np.empty(0, result_dtype)), result_shape)
            try:
                arena.binary(op, left, right, out=out)
                return self._pull(arena, out, result_dtype, result_shape)
            finally:
                arena.destroy(out)
        finally:
            arena.destroy(left)
            arena.destroy(right)

    def _matmul_via_nodus(self, arena, a, b):
        """Matrix multiply through ``tensor_matmul_f32``/``f64``.

        Reached over ``nodus_tensor_matmul``, which is not a CanonicalOp --
        matmul is not elementwise, so it is not in the canonical operator
        table, but nodus implements it and this backend uses it.
        """

        left_array = np.asarray(a)
        right_array = np.asarray(b)
        result_dtype = np.result_type(left_array, right_array, np.float32)
        if result_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise NodusUnsupported(
                f"nodus matmul is f32/f64 only; operands promote to "
                f"{result_dtype}"
            )
        if left_array.ndim not in (1, 2) or right_array.ndim not in (1, 2):
            raise NodusUnsupported(
                "nodus matmul supports rank-1/rank-2 NumPy matmul; got shapes "
                f"{left_array.shape} and {right_array.shape}"
            )
        # The core ABI is matrix-matrix only. Rank-1 operands are ABI views of
        # the same data, not new operators: promote them to a row/column and
        # remove the corresponding singleton result axis afterward.
        left_was_vector = left_array.ndim == 1
        right_was_vector = right_array.ndim == 1
        left_matrix = left_array.reshape(1, -1) if left_was_vector else left_array
        right_matrix = right_array.reshape(-1, 1) if right_was_vector else right_array
        left = self._push(arena, left_matrix.astype(result_dtype, copy=False))
        right = self._push(arena, right_matrix.astype(result_dtype, copy=False))
        try:
            out = arena.matmul(left, right)
            try:
                matrix_shape = (left_matrix.shape[0], right_matrix.shape[1])
                result = self._pull(arena, out, result_dtype, matrix_shape)
                if left_was_vector and right_was_vector:
                    return result.reshape(())
                if left_was_vector:
                    return result.reshape(matrix_shape[1])
                if right_was_vector:
                    return result.reshape(matrix_shape[0])
                return result
            finally:
                arena.destroy(out)
        finally:
            arena.destroy(left)
            arena.destroy(right)


_abstraction.register_backend("nodus", NodusTensorOperations)
_abstraction.register_backend("in_memory", NodusTensorOperations)

__all__ = ["NodusTensorOperations"]
