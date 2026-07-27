"""GLSL compute-shader execution target for FusedProgram elementwise regions.

This is the GPU sibling of the C backend. It deliberately mirrors that design
rather than inventing a second one:

    c_backend/ctensor_ops.c      one flat CTensorOp vocabulary + a switch dispatcher
    fused_ir.py                  one backend-neutral semantic program
    c_primitive_program.py       private lowering to the native C slot ABI
    glsl_backend.py  (this)      direct lowering to a fused compute shader

The one structural difference is where the win comes from. The C interpreter walks
instructions and writes every intermediate slot to memory. A GPU does not want
that: an elementwise program of N instructions compiles to **one shader with N
lines and a single dispatch**, where every intermediate is a register-resident
local. Only feeds and the final output ever touch a buffer.

Memory model
------------
``GLChunk`` is a typed chunk of equally-shaped data that lives on the CPU (numpy),
in an OpenGL buffer (SSBO), or both, with explicit transfer and an explicit
residency flag. It can also *wrap a buffer this module did not allocate*
(``GLChunk.wrap``), which is how a host that already owns a GL context -- a nodus
or pluck renderer -- hands tensor data across without a round trip through system
memory.

Context policy
--------------
This backend **demands** an OpenGL 4.3+ context and fails loudly when it cannot
have one. It never silently falls back to CPU. Acquisition is delegated to
``gl_context``, which borrows the host's context (a pluck or nodus frontend)
before ever creating its own -- see that module for the ordering and for why
nodus's registration-based model means "be registerable" rather than "ask nodus
for a context".

Storage types
-------------
GPU storage uses the native 32-bit scalar types accepted by std430 SSBO arrays:
``float``, ``int`` and ``uint``. Logical tensors use uint storage with a bool
logical dtype. Floating values are narrowed to float32 at the backend boundary.
The typed storage is required by the shared bitwise and cast primitives; treating
those values as floats would silently change their meaning.
"""

from __future__ import annotations

import ctypes
import hashlib
import itertools
import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..fused_ir import (
    FusedProgram,
    Meta,
    OpStep,
    ordered_feed_ids,
    primary_output_id,
)
from .gl_context import (  # re-exported: the backend's public context surface
    GLContextUnavailable,
    gl_context_info,
    register_context_provider,
    release_gl_context,
    require_gl_context,
)

__all__ = [
    "GLContextUnavailable",
    "GLSLCompileError",
    "GLSLUnsupportedOp",
    "GLComputeLimits",
    "GLLaunchPlan",
    "require_gl_context",
    "gl_context_info",
    "register_context_provider",
    "release_gl_context",
    "GLChunk",
    "FusedProgram",
    "OpStep",
    "emit_program_source",
    "emit_multi_output_program_source",
    "emit_op_source",
    "emit_cat_source",
    "emit_arange_source",
    "emit_cumsum_source",
    "emit_expand_source",
    "emit_gather_source",
    "emit_index_assign_source",
    "emit_index_select_source",
    "emit_slice_axis_source",
    "emit_matmul_source",
    "emit_permute_source",
    "emit_repeat_source",
    "emit_reduce_source",
    "emit_stack_source",
    "emit_topk_offsets_source",
    "run_op",
    "cat_chunks",
    "arange_chunk",
    "cumsum_chunk",
    "expand_chunk",
    "gather_offsets_chunk",
    "index_assign_offsets_chunk",
    "index_assign_index_chunk",
    "index_select_chunk",
    "matmul_chunks",
    "permute_chunk",
    "repeat_chunk",
    "reduce_chunk",
    "plan_launch",
    "reshape_chunk",
    "slice_axis_chunk",
    "stack_chunks",
    "topk_chunks",
    "execute_program",
    "execute_multi_output_program",
    "fuse_elementwise",
    "GLSL_OPS",
    "shader_cache_stats",
]


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class GLSLCompileError(RuntimeError):
    """A generated shader failed to compile or link; carries the driver log."""


class GLSLUnsupportedOp(KeyError):
    """An op has no GLSL lowering. Raised rather than silently skipped."""


# ---------------------------------------------------------------------------
# the op table -- the GLSL surface's half of the canonical vocabulary
#
# Keys are the canonical op names used by AbstractTensor's ``_apply_operator__``
# and by c_primitive_program's ``_OP_NAMES``. Values are GLSL expression
# templates over ``$a`` (left) and ``$b`` (right).
#
# This is a function table, deliberately: one dict, one lookup, a hard KeyError
# on a miss. It is not an if-chain, and it is not a second numbering -- the
# integer opcode space belongs to turing's CTensorOp (see
# nodus/ops/canonical_ops.json's numbering contract).
# ---------------------------------------------------------------------------

_BINARY: dict[str, str] = {
    "add": "$a + $b",
    "sub": "$a - $b",
    "mul": "$a * $b",
    "truediv": "$a / $b",
    "pow": "pow($a, $b)",
    # Floored modulo, matching ctensor_ops.c's `a - floor(a/b)*b` and numpy/torch.
    # NOT GLSL's mod(), which differs for mixed signs on some drivers.
    "mod": "($a - floor($a / $b) * $b)",
    "floordiv": "floor($a / $b)",
    "maximum": "max($a, $b)",
    "minimum": "min($a, $b)",
    "less": "$out($a < $b)",
    "less_equal": "$out($a <= $b)",
    "greater": "$out($a > $b)",
    "greater_equal": "$out($a >= $b)",
    "equal": "$out($a == $b)",
    "not_equal": "$out($a != $b)",
    "bitand": "$a & $b",
    "bitor": "$a | $b",
    "bitxor": "$a ^ $b",
    "shl": "$a << $b",
    "shr": "$a >> $b",
    "logical_and": "$out(($a != 0) && ($b != 0))",
    "logical_or": "$out(($a != 0) || ($b != 0))",
}

_UNARY: dict[str, str] = {
    "sqrt": "sqrt($a)",
    "exp": "exp($a)",
    "log": "log($a)",
    "tanh": "tanh($a)",
    "sin": "sin($a)",
    "cos": "cos($a)",
    "tan": "tan($a)",
    "asin": "asin($a)",
    "acos": "acos($a)",
    "atan": "atan($a)",
    "sinh": "sinh($a)",
    "cosh": "cosh($a)",
    "asinh": "asinh($a)",
    "acosh": "acosh($a)",
    "atanh": "atanh($a)",
    "neg": "-$a",
    "abs": "abs($a)",
    # C round() is half-away-from-zero; GLSL round() is permitted to break ties
    # either way (roundEven() is the explicit one). Spell it out so the GPU and
    # the C backend agree on x.5 instead of disagreeing per driver.
    "round": "(sign($a) * floor(abs($a) + 0.5))",
    "trunc": "trunc($a)",
    "floor": "floor($a)",
    "ceil": "ceil($a)",
    "isfinite": "$out(!isinf($a) && !isnan($a))",
    "isnan": "$out(isnan($a))",
    "isinf": "$out(isinf($a))",
    "logical_not": "$out($a == 0)",
    "sign": "sign($a)",
    "invert": "~$a",
    "int_trunc": "int($a)",
    "zext": "uint($a)",
    "sext": "int($a)",
    "fptosi": "int($a)",
    "fptoui": "uint($a)",
    "sitofp": "float($a)",
    "uitofp": "float($a)",
}

GLSL_OPS: frozenset[str] = frozenset(_BINARY) | frozenset(_UNARY)

_LOCAL_SIZE = 256

_ALIASES = {
    "div": "truediv",
    "lt": "less",
    "le": "less_equal",
    "gt": "greater",
    "ge": "greater_equal",
    "eq": "equal",
    "ne": "not_equal",
}


def canonical_op(op: str) -> tuple[str, bool]:
    """Resolve an op string to ``(canonical_name, reverse_operands)``.

    Mirrors the C backend's i/r-prefix normalization, but only strips a prefix
    when doing so actually yields a known op -- so ``isnan``, ``isinf``,
    ``isfinite``, ``invert`` and ``round`` survive intact instead of becoming
    ``snan``, ``sinf``, ``sfinite``, ``nvert`` and ``ound``.
    """
    name = _ALIASES.get(op, op)
    if name in GLSL_OPS:
        return name, False
    if name[:1] in ("i", "r"):
        base = _ALIASES.get(name[1:], name[1:])
        if base in GLSL_OPS:
            return base, name[0] == "r"
    raise GLSLUnsupportedOp(
        f"no GLSL lowering for op {op!r}; "
        f"known ops: {', '.join(sorted(GLSL_OPS))}"
    )


def _expr(
    op: str,
    a: str,
    b: str | None,
    reverse: bool,
    *,
    out_type: str = "float",
) -> str:
    if op in _UNARY:
        if b is not None:
            raise ValueError(f"unary op {op!r} given a right operand")
        template = _UNARY[op]
        return template.replace("$a", a).replace("$out", out_type)
    if b is None:
        raise ValueError(f"binary op {op!r} missing its right operand")
    left, right = (b, a) if reverse else (a, b)
    return (
        _BINARY[op]
        .replace("$a", left)
        .replace("$b", right)
        .replace("$out", out_type)
    )


# ---------------------------------------------------------------------------
# memory chunks
# ---------------------------------------------------------------------------

def _normalize_dtype(dtype: Any) -> np.dtype:
    """Return one of the logical scalar dtypes supported by GLSL SSBO storage."""
    if dtype is None:
        return np.dtype(np.float32)
    if isinstance(dtype, str):
        key = dtype.lower().replace("torch.", "").replace("numpy.", "")
        if key in {"float", "float16", "float32", "float64", "double", "single"}:
            return np.dtype(np.float32)
        if key in {"int", "int8", "int16", "int32", "int64", "long"}:
            return np.dtype(np.int32)
        if key in {"uint", "uint8", "uint16", "uint32", "uint64"}:
            return np.dtype(np.uint32)
        if key in {"bool", "boolean"}:
            return np.dtype(np.bool_)
    value = np.dtype(dtype)
    if value.kind == "b":
        return np.dtype(np.bool_)
    if value.kind == "f":
        return np.dtype(np.float32)
    if value.kind == "i":
        return np.dtype(np.int32)
    if value.kind == "u":
        return np.dtype(np.uint32)
    raise TypeError(f"GLSL backend does not support dtype {dtype!r}")


def _storage_dtype(dtype: Any) -> np.dtype:
    dtype = _normalize_dtype(dtype)
    return np.dtype(np.uint32) if dtype.kind == "b" else dtype


def _glsl_type(dtype: Any) -> str:
    kind = _normalize_dtype(dtype).kind
    return {"f": "float", "i": "int", "u": "uint", "b": "uint"}[kind]


class _GLStorage:
    """Shared physical storage for one or more differently shaped GL views."""

    __slots__ = ("dtype", "host", "buffer", "owns_buffer", "gpu_valid", "refs")

    def __init__(
        self,
        dtype: Any,
        host: np.ndarray | None = None,
        *,
        buffer: int | None = None,
        owns_buffer: bool = False,
        gpu_valid: bool = False,
    ) -> None:
        self.dtype = _normalize_dtype(dtype)
        self.host = (
            None
            if host is None
            else np.ascontiguousarray(host, dtype=self.dtype).reshape(-1)
        )
        self.buffer = buffer
        self.owns_buffer = bool(owns_buffer)
        self.gpu_valid = bool(gpu_valid)
        self.refs = 1


class GLChunk:
    """An equally-shaped typed block resident on the CPU, the GPU, or both.

    Residency is explicit and observable (``.on_cpu`` / ``.on_gpu``) rather than
    implied. Transfers are explicit calls. Nothing here silently moves data
    behind the caller's back, because a hidden readback is a performance cliff
    that is very hard to notice after the fact.
    """

    __slots__ = (
        "_shape",
        "_count",
        "_storage",
        "_deferred",
        "_released",
    )

    def __init__(
        self,
        shape: Sequence[int],
        host: np.ndarray | None = None,
        *,
        dtype: Any = np.float32,
    ) -> None:
        self._shape = tuple(int(d) for d in shape)
        self._count = int(np.prod(self._shape)) if self._shape else 1
        logical_dtype = _normalize_dtype(
            host.dtype if host is not None and dtype is None else dtype
        )
        host_array = None
        if host is not None:
            host_array = np.asarray(host, dtype=logical_dtype)
            if host_array.size != self._count:
                raise ValueError(
                    f"host contains {host_array.size} elements for shape "
                    f"{self._shape} ({self._count} required)"
                )
        self._storage = _GLStorage(logical_dtype, host_array)
        self._deferred = None
        self._released = False

    # -- construction ------------------------------------------------------

    @classmethod
    def from_numpy(cls, array: Any, dtype: Any = None) -> "GLChunk":
        raw = np.asarray(array)
        logical_dtype = _normalize_dtype(raw.dtype if dtype is None else dtype)
        arr = np.asarray(raw, dtype=logical_dtype)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return cls(arr.shape, arr, dtype=logical_dtype)

    @classmethod
    def zeros(cls, shape: Sequence[int], dtype: Any = np.float32) -> "GLChunk":
        shape = tuple(int(d) for d in shape)
        logical_dtype = _normalize_dtype(dtype)
        return cls(
            shape,
            np.zeros(shape, dtype=logical_dtype),
            dtype=logical_dtype,
        )

    @classmethod
    def wrap(
        cls,
        buffer_id: int,
        shape: Sequence[int],
        dtype: Any = np.float32,
    ) -> "GLChunk":
        """Adopt an SSBO this module did not allocate.

        The interop path: a host that already owns the GL context (a nodus or
        pluck renderer) passes its buffer name and shape, and computation happens
        in place with no system-memory round trip. Ownership stays with the host --
        ``release()`` will not delete a wrapped buffer.
        """
        chunk = cls(shape, None, dtype=dtype)
        chunk._storage.buffer = int(buffer_id)
        chunk._storage.owns_buffer = False
        chunk._storage.gpu_valid = True
        return chunk

    def view(self, shape: Sequence[int]) -> "GLChunk":
        """Return a zero-copy shape view sharing one coherent physical buffer."""
        if self._released:
            raise RuntimeError("cannot view a released GLChunk")
        shape = tuple(int(d) for d in shape)
        count = int(np.prod(shape)) if shape else 1
        if count != self._count:
            raise ValueError(
                f"view shape {shape} has {count} elements, expected {self._count}"
            )
        if self._deferred is not None:
            deferred = self._deferred
            output_id = primary_output_id(deferred.program)
            metadata = dict(deferred.program.meta or {})
            output_meta = metadata.get(output_id)
            metadata[output_id] = Meta(
                shape=shape,
                dtype=(
                    getattr(output_meta, "dtype", None)
                    if output_meta is not None
                    else self.dtype.name
                ),
                device=(
                    getattr(output_meta, "device", None)
                    if output_meta is not None
                    else "glsl"
                ),
            )
            program = FusedProgram(
                version=deferred.program.version,
                feeds=set(deferred.program.feeds),
                steps=list(deferred.program.steps),
                outputs=dict(deferred.program.outputs),
                state_in=deferred.program.state_in,
                meta=metadata,
                extras=deferred.program.extras,
            )
            if hasattr(deferred.program, "feed_order"):
                program.feed_order = deferred.program.feed_order
            program.glsl_linear_output_shape = shape
            chunk = GLChunk(shape, dtype=self.dtype)
            chunk._deferred = _DeferredElementwise(program, deferred.feeds)
            return chunk
        chunk = object.__new__(type(self))
        chunk._shape = shape
        chunk._count = count
        chunk._storage = self._storage
        chunk._storage.refs += 1
        chunk._deferred = None
        chunk._released = False
        return chunk

    # -- properties --------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def count(self) -> int:
        return self._count

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return self._count

    @property
    def dtype(self):
        return self._storage.dtype

    @property
    def nbytes(self) -> int:
        return self._count * 4

    @property
    def on_cpu(self) -> bool:
        return not self._released and self._storage.host is not None

    @property
    def on_gpu(self) -> bool:
        return (
            not self._released
            and self._storage.buffer is not None
            and self._storage.gpu_valid
        )

    @property
    def buffer_id(self) -> int | None:
        if not self._released and self._deferred is not None:
            self.to_gpu()
        return None if self._released else self._storage.buffer

    # -- transfer ----------------------------------------------------------

    def to_gpu(self) -> "GLChunk":
        """Ensure GPU residency, allocating and uploading if needed."""
        require_gl_context()
        return self._to_gpu_current()

    def _to_gpu_current(self) -> "GLChunk":
        """Ensure residency when the caller has already established a GL context."""
        if self._released:
            raise RuntimeError("cannot upload a released GLChunk")
        if self._deferred is not None:
            deferred = self._deferred
            self._deferred = None
            try:
                execute_program(deferred.program, deferred.feeds, out=self)
            except Exception:
                self._deferred = deferred
                raise
            return self
        from OpenGL import GL

        storage = self._storage
        if storage.buffer is None:
            storage.buffer = int(GL.glGenBuffers(1))
            storage.owns_buffer = True
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, storage.buffer)
            GL.glBufferData(
                GL.GL_SHADER_STORAGE_BUFFER, self.nbytes, None, GL.GL_DYNAMIC_DRAW
            )
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            storage.gpu_valid = False

        if not storage.gpu_valid:
            if storage.host is None:
                # Allocated but never written and nothing to upload: it is an
                # output slot. Leave contents undefined but mark it live.
                storage.gpu_valid = True
                return self
            data = np.ascontiguousarray(
                storage.host, dtype=_storage_dtype(storage.dtype)
            )
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, storage.buffer)
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, self.nbytes, data)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            storage.gpu_valid = True
        return self

    def update_numpy(self, array: Any) -> "GLChunk":
        """Replace host contents while preserving an allocated GPU buffer.

        The shape may not change.  The next :meth:`to_gpu` uploads the new
        contents into the existing SSBO instead of allocating another buffer.
        """
        if self._released:
            raise RuntimeError("cannot update a released GLChunk")
        self._deferred = None
        data = np.ascontiguousarray(np.asarray(array, dtype=self.dtype))
        if data.shape != self._shape:
            raise ValueError(
                f"updated data must keep shape {self._shape}, got {data.shape}"
            )
        self._storage.host = data.reshape(-1)
        self._storage.gpu_valid = False
        return self

    def discard_host(self) -> "GLChunk":
        """Drop a staging/readback copy while preserving the live SSBO."""
        if not self.on_gpu:
            raise RuntimeError("cannot discard the only valid GLChunk storage")
        self._storage.host = None
        return self

    def _mark_gpu_written(self) -> None:
        """Mark host contents stale after a shader writes the SSBO."""
        if self._released:
            raise RuntimeError("cannot mark a released GLChunk")
        self._deferred = None
        self._storage.host = None
        self._storage.gpu_valid = True

    def __len__(self) -> int:
        if not self._shape:
            raise TypeError("len() of a scalar GLChunk")
        return self._shape[0]

    def to_cpu(self) -> np.ndarray:
        """Read back into host memory and return the host array."""
        if self._released:
            raise RuntimeError("cannot read a released GLChunk")
        if self._deferred is not None:
            self.to_gpu()
        if self._storage.host is not None:
            return self._storage.host.reshape(self._shape)
        if not self.on_gpu:
            raise RuntimeError("chunk has no CPU data and no live GPU buffer")
        require_gl_context()
        from OpenGL import GL

        out = np.empty(self._count, dtype=_storage_dtype(self.dtype))
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._storage.buffer)
        GL.glGetBufferSubData(
            GL.GL_SHADER_STORAGE_BUFFER, 0, self.nbytes,
            out.ctypes.data_as(ctypes.c_void_p),
        )
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        self._storage.host = out.astype(self.dtype, copy=False).reshape(-1)
        return self._storage.host.reshape(self._shape)

    def numpy(self) -> np.ndarray:
        """Host view, reading back from the GPU when that is the live copy."""
        return self.to_cpu()

    def release(self) -> None:
        """Delete the GL buffer if we allocated it. Wrapped buffers are left alone."""
        if self._released:
            return
        self._released = True
        self._deferred = None
        storage = self._storage
        storage.refs -= 1
        if storage.refs:
            return
        if storage.buffer is not None and storage.owns_buffer:
            try:
                from OpenGL import GL
                GL.glDeleteBuffers(1, [storage.buffer])
            except Exception:
                pass  # context may already be gone during interpreter teardown
        storage.buffer = None
        storage.owns_buffer = False
        storage.gpu_valid = False
        storage.host = None

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    def __repr__(self) -> str:
        where = []
        if self.on_cpu:
            where.append("cpu")
        if self.on_gpu:
            where.append("gpu")
        if self._deferred is not None:
            where.append("deferred")
        return (
            f"GLChunk(shape={self._shape}, dtype={self.dtype}, "
            f"on={'+'.join(where) or 'none'})"
        )


# ---------------------------------------------------------------------------
# shader emission
# ---------------------------------------------------------------------------

_SHADER_HEADER = """#version 430
// GENERATED by turing glsl_backend.emit_program_source -- do not edit by hand.
//
// Fused elementwise program: every intermediate value is a local (a register),
// so only feeds and the single output ever touch memory. This is the whole
// reason a FusedProgram is worth running on a GPU rather than operation
// by instruction.
layout(local_size_x = {local_size}) in;

uint turing_linear_gid() {{
    uvec3 launch_size = gl_NumWorkGroups * gl_WorkGroupSize;
    return gl_GlobalInvocationID.x
         + gl_GlobalInvocationID.y * launch_size.x
         + gl_GlobalInvocationID.z * launch_size.x * launch_size.y;
}}
"""

_STRUCTURAL_SHADER_HEADER = """#version 430
// GENERATED by turing glsl_backend structural lowering -- do not edit by hand.
//
// One output invocation maps directly to one source-buffer address.  The full
// cat/stack layout operation therefore needs one compute dispatch and no
// intermediate buffer or host readback.
layout(local_size_x = {local_size}) in;

uint turing_linear_gid() {{
    uvec3 launch_size = gl_NumWorkGroups * gl_WorkGroupSize;
    return gl_GlobalInvocationID.x
         + gl_GlobalInvocationID.y * launch_size.x
         + gl_GlobalInvocationID.z * launch_size.x * launch_size.y;
}}
"""


def _validate_program_outputs(
    program: FusedProgram,
) -> tuple[tuple[int, ...], tuple[tuple[str, int], ...]]:
    feed_ids = ordered_feed_ids(program)
    if not feed_ids:
        raise ValueError("a fused program needs at least one feed")
    defined = set(feed_ids)
    for step in program.steps:
        op, _ = canonical_op(step.op_name)
        if op not in GLSL_OPS:
            raise GLSLUnsupportedOp(
                f"fused GLSL program does not support typed primitive {op!r}"
            )
        scalar = step.attrs.get("right_scalar")
        unknown = set(step.attrs) - {"right_scalar", "reverse"}
        if unknown:
            raise ValueError(
                f"step {step.step_id} has unsupported attrs: {sorted(unknown)}"
            )
        if any(value_id not in defined for value_id in step.input_ids):
            raise ValueError(
                f"step {step.step_id} reads a value before it is written"
            )
        if op in _UNARY:
            valid = len(step.input_ids) == 1 and scalar is None
        else:
            valid = (
                len(step.input_ids) == 2 and scalar is None
            ) or (
                len(step.input_ids) == 1 and scalar is not None
            )
        if not valid:
            raise ValueError(f"step {step.step_id} has an invalid operand layout")
        defined.add(step.result_id)
    outputs = tuple((str(name), value_id) for name, value_id in program.outputs.items())
    if not outputs:
        raise ValueError("a fused program needs at least one output")
    missing = [name for name, value_id in outputs if value_id not in defined]
    if missing:
        raise ValueError(
            "FusedProgram outputs are not produced: " + ", ".join(missing)
        )
    return feed_ids, outputs


def _validate_program(program: FusedProgram) -> tuple[tuple[int, ...], int]:
    feed_ids, outputs = _validate_program_outputs(program)
    if len(outputs) != 1:
        raise ValueError("elementwise fused backends require exactly one output")
    return feed_ids, outputs[0][1]


def _emit_program_source(
    program: FusedProgram,
    local_size: int = _LOCAL_SIZE,
    *,
    scalar_feeds: Iterable[int] = (),
    feed_shapes: Mapping[int, Sequence[int]] | None = None,
    output_shape: Sequence[int] | None = None,
    allow_multiple_outputs: bool = False,
) -> str:
    feed_ids, outputs = _validate_program_outputs(program)
    if not allow_multiple_outputs and len(outputs) != 1:
        raise ValueError("elementwise fused backends require exactly one output")
    scalar_feeds = frozenset(int(value_id) for value_id in scalar_feeds)
    unknown_scalars = scalar_feeds - set(feed_ids)
    if unknown_scalars:
        raise ValueError(
            f"scalar feed IDs are not program feeds: {sorted(unknown_scalars)}"
        )
    normalized_feed_shapes = None
    if feed_shapes is not None:
        missing_shapes = set(feed_ids) - set(feed_shapes)
        if missing_shapes:
            raise ValueError(
                f"missing FusedProgram feed shapes: {sorted(missing_shapes)}"
            )
        normalized_feed_shapes = {
            feed_id: tuple(int(size) for size in feed_shapes[feed_id])
            for feed_id in feed_ids
        }
        if output_shape is None:
            resolved: tuple[int, ...] = ()
            for feed_id in feed_ids:
                resolved = _broadcast_shape(
                    resolved, normalized_feed_shapes[feed_id]
                )
            output_shape = resolved
        else:
            output_shape = tuple(int(size) for size in output_shape)
    metadata = program.meta or {}

    def metadata_dtype(value_id: int, default: Any = np.float32) -> np.dtype:
        meta = metadata.get(value_id)
        value = getattr(meta, "dtype", None) if meta is not None else None
        if value in (None, "None"):
            value = default
        return _normalize_dtype(value)

    value_dtypes: dict[int, np.dtype] = {
        feed_id: metadata_dtype(feed_id) for feed_id in feed_ids
    }
    output_dtypes = {
        name: metadata_dtype(output_id) for name, output_id in outputs
    }
    declarations: list[str] = [_SHADER_HEADER.format(local_size=local_size)]

    for i, feed_id in enumerate(feed_ids):
        declarations.append(
            f"layout(std430, binding = {i}) readonly buffer Feed{i} "
            f"{{ {_glsl_type(value_dtypes[feed_id])} feed{i}[]; }};"
        )
    for output_index, (name, _output_id) in enumerate(outputs):
        suffix = "" if len(outputs) == 1 else str(output_index)
        out_binding = len(feed_ids) + output_index
        declarations.append(
            f"layout(std430, binding = {out_binding}) writeonly buffer "
            f"OutBuf{suffix} {{ {_glsl_type(output_dtypes[name])} "
            f"outbuf{suffix}[]; }};"
        )
    body = [
        "",
        "uniform uint u_count;",
        "",
        "void main() {",
        "    uint gid = turing_linear_gid();",
        "    if (gid >= u_count) { return; }",
    ]

    value_names: dict[int, str] = {}
    for i, feed_id in enumerate(feed_ids):
        if feed_id in scalar_feeds:
            index = "0"
        elif normalized_feed_shapes is not None:
            index_lines, index = _broadcast_index_source(
                f"feed{i}",
                normalized_feed_shapes[feed_id],
                output_shape,
            )
            body.extend(index_lines)
        else:
            index = "gid"
        body.append(
            f"    {_glsl_type(value_dtypes[feed_id])} s{i} = "
            f"feed{i}[{index}];"
        )
        value_names[feed_id] = f"s{i}"

    helpers: list[str] = []
    for index, step in enumerate(program.steps, len(feed_ids)):
        op, reverse = canonical_op(step.op_name)
        reverse = reverse ^ bool(step.attrs.get("reverse", False))
        a = value_names[step.input_ids[0]]
        left_dtype = value_dtypes[step.input_ids[0]]
        if op in _UNARY:
            b = None
            right_dtype = None
        elif len(step.input_ids) == 2:
            b = value_names[step.input_ids[1]]
            right_dtype = value_dtypes[step.input_ids[1]]
        elif "right_scalar" in step.attrs:
            scalar = step.attrs["right_scalar"]
            right_dtype = _normalize_dtype(np.asarray(scalar).dtype)
            b = _glsl_literal(scalar, right_dtype)
        else:
            raise ValueError(f"binary op {step.op_name!r} has no right operand")
        inferred_dtype = _result_dtype(op, left_dtype, right_dtype)
        result_dtype = metadata_dtype(step.result_id, inferred_dtype)
        helper, expression = _typed_expr(
            op,
            a,
            b,
            reverse,
            left_dtype,
            right_dtype,
            result_dtype,
        )
        if helper and helper not in helpers:
            helpers.append(helper)
        value_names[step.result_id] = f"s{index}"
        value_dtypes[step.result_id] = result_dtype
        body.append(
            f"    {_glsl_type(result_dtype)} s{index} = {expression};"
        )

    for output_index, (name, output_id) in enumerate(outputs):
        suffix = "" if len(outputs) == 1 else str(output_index)
        output_dtype = output_dtypes[name]
        output_value = value_names[output_id]
        if value_dtypes[output_id] != output_dtype:
            output_value = f"{_glsl_type(output_dtype)}({output_value})"
        body.append(f"    outbuf{suffix}[gid] = {output_value};")
    body.append("}")
    return "\n".join(
        declarations + ([""] + helpers if helpers else []) + body
    ) + "\n"


def emit_program_source(
    program: FusedProgram,
    local_size: int = _LOCAL_SIZE,
    *,
    scalar_feeds: Iterable[int] = (),
    feed_shapes: Mapping[int, Sequence[int]] | None = None,
    output_shape: Sequence[int] | None = None,
) -> str:
    """Lower a typed, single-output FusedProgram to one compute shader."""
    return _emit_program_source(
        program,
        local_size,
        scalar_feeds=scalar_feeds,
        feed_shapes=feed_shapes,
        output_shape=output_shape,
        allow_multiple_outputs=False,
    )


def emit_multi_output_program_source(
    program: FusedProgram,
    local_size: int = _LOCAL_SIZE,
    *,
    scalar_feeds: Iterable[int] = (),
    feed_shapes: Mapping[int, Sequence[int]] | None = None,
    output_shape: Sequence[int] | None = None,
) -> str:
    """Lower a same-shape, multi-output FusedProgram to one compute shader."""
    return _emit_program_source(
        program,
        local_size,
        scalar_feeds=scalar_feeds,
        feed_shapes=feed_shapes,
        output_shape=output_shape,
        allow_multiple_outputs=True,
    )


def _glsl_float(value: float) -> str:
    """A GLSL float literal that survives round-tripping and is never an int."""
    v = float(value)
    if v != v:
        return "(0.0 / 0.0)"
    if v == float("inf"):
        return "(1.0 / 0.0)"
    if v == float("-inf"):
        return "(-1.0 / 0.0)"
    return f"float({v!r})"


_BOOL_RESULTS = frozenset(
    {
        "isfinite",
        "isnan",
        "isinf",
        "logical_not",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "equal",
        "not_equal",
        "logical_and",
        "logical_or",
    }
)
_CAST_RESULTS = {
    "int_trunc": np.dtype(np.int32),
    "zext": np.dtype(np.uint32),
    "sext": np.dtype(np.int32),
    "fptosi": np.dtype(np.int32),
    "fptoui": np.dtype(np.uint32),
    "sitofp": np.dtype(np.float32),
    "uitofp": np.dtype(np.float32),
}
_FLOAT_MATH = frozenset(
    {
        "sqrt",
        "exp",
        "log",
        "tanh",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "asinh",
        "acosh",
        "atanh",
    }
)
_BITWISE = frozenset({"invert", "bitand", "bitor", "bitxor", "shl", "shr"})


def _promote_dtype(left: Any, right: Any | None = None) -> np.dtype:
    left = _normalize_dtype(left)
    if right is None:
        return left
    right = _normalize_dtype(right)
    kinds = {left.kind, right.kind}
    if "f" in kinds:
        return np.dtype(np.float32)
    if kinds == {"u"}:
        return np.dtype(np.uint32)
    return np.dtype(np.int32)


def _result_dtype(op: str, left: Any, right: Any | None = None) -> np.dtype:
    if op in _BOOL_RESULTS:
        return np.dtype(np.bool_)
    if op in _CAST_RESULTS:
        return _CAST_RESULTS[op]
    if op in _FLOAT_MATH or op == "truediv":
        return np.dtype(np.float32)
    return _promote_dtype(left, right)


def _glsl_literal(value: Any, dtype: Any) -> str:
    dtype = _normalize_dtype(dtype)
    if dtype.kind == "f":
        return _glsl_float(float(value))
    if dtype.kind == "u" or dtype.kind == "b":
        return f"uint({int(value)})"
    return f"int({int(value)})"


def _cast_expr(value: str, glsl_type: str) -> str:
    return f"{glsl_type}({value})"


def _typed_expr(
    op: str,
    a: str,
    b: str | None,
    reverse: bool,
    left_dtype: Any,
    right_dtype: Any | None,
    out_dtype: Any,
) -> tuple[str, str]:
    """Return ``(helper_source, expression)`` for one canonical primitive."""
    if reverse:
        if b is None:
            raise ValueError(f"unary op {op!r} cannot reverse operands")
        a, b = b, a
        left_dtype, right_dtype = right_dtype, left_dtype

    left_type = _glsl_type(left_dtype)
    right_type = _glsl_type(right_dtype) if right_dtype is not None else None
    out_type = _glsl_type(out_dtype)
    helper = ""

    if op in _CAST_RESULTS:
        return "", _expr(op, a, None, False, out_type=out_type)

    if op in _FLOAT_MATH:
        return "", _expr(
            op,
            a if left_type == "float" else _cast_expr(a, "float"),
            None,
            False,
            out_type=out_type,
        )

    if op in {"round", "trunc", "floor", "ceil"} and left_type != "float":
        return "", _cast_expr(a, out_type)

    if op == "abs" and left_type == "uint":
        return "", a
    if op == "sign" and left_type == "uint":
        return "", f"uint({a} != 0u)"

    if op in {"isfinite", "isnan", "isinf"} and left_type != "float":
        value = "true" if op == "isfinite" else "false"
        return "", f"{out_type}({value})"

    if op == "logical_not":
        return "", f"{out_type}({a} == {_glsl_literal(0, left_dtype)})"

    if op in {"logical_and", "logical_or"}:
        if b is None or right_dtype is None:
            raise ValueError(f"binary op {op!r} missing its right operand")
        join = "&&" if op == "logical_and" else "||"
        return "", (
            f"{out_type}(({a} != {_glsl_literal(0, left_dtype)}) {join} "
            f"({b} != {_glsl_literal(0, right_dtype)}))"
        )

    if op in {
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "equal",
        "not_equal",
    }:
        if b is None or right_dtype is None:
            raise ValueError(f"binary op {op!r} missing its right operand")
        common = _glsl_type(_promote_dtype(left_dtype, right_dtype))
        return "", _expr(
            op,
            a if left_type == common else _cast_expr(a, common),
            b if right_type == common else _cast_expr(b, common),
            False,
            out_type=out_type,
        )

    if op in _BITWISE:
        if _normalize_dtype(left_dtype).kind == "f":
            raise TypeError(f"{op} requires an integer or unsigned tensor")
        if b is None:
            return "", _expr(op, a, None, False, out_type=out_type)
        if right_dtype is None or _normalize_dtype(right_dtype).kind == "f":
            raise TypeError(f"{op} requires integer or unsigned operands")
        if op in {"shl", "shr"}:
            return "", _expr(
                op, _cast_expr(a, out_type), _cast_expr(b, "uint"), False,
                out_type=out_type,
            )
        return "", _expr(
            op, _cast_expr(a, out_type), _cast_expr(b, out_type), False,
            out_type=out_type,
        )

    if b is None:
        return "", _expr(op, a, None, False, out_type=out_type)
    if right_dtype is None:
        raise ValueError(f"binary op {op!r} missing its right dtype")

    common = _glsl_type(out_dtype)
    aa = a if left_type == common else _cast_expr(a, common)
    bb = b if right_type == common else _cast_expr(b, common)

    if op == "truediv":
        return "", f"float({a}) / float({b})"
    if op == "pow" and common != "float":
        helper_name = "ipow_u" if common == "uint" else "ipow_i"
        scalar_type = "uint" if common == "uint" else "int"
        one = "1u" if common == "uint" else "1"
        helper = (
            f"{scalar_type} {helper_name}({scalar_type} base, uint exponent) {{\n"
            f"    {scalar_type} result = {one};\n"
            "    while (exponent != 0u) {\n"
            "        if ((exponent & 1u) != 0u) { result *= base; }\n"
            "        exponent >>= 1u;\n"
            "        if (exponent != 0u) { base *= base; }\n"
            "    }\n"
            "    return result;\n"
            "}\n"
        )
        return helper, f"{helper_name}({aa}, uint({b}))"
    if op in {"mod", "floordiv"} and common == "int":
        helper = (
            "int floor_div_i(int x, int y) {\n"
            "    int q = x / y;\n"
            "    int r = x % y;\n"
            "    return q - int((r != 0) && ((r < 0) != (y < 0)));\n"
            "}\n"
        )
        if op == "floordiv":
            return helper, f"floor_div_i({aa}, {bb})"
        return helper, f"({aa} - floor_div_i({aa}, {bb}) * {bb})"
    if op == "mod" and common == "uint":
        return "", f"{aa} % {bb}"
    if op == "floordiv" and common == "uint":
        return "", f"{aa} / {bb}"
    return "", _expr(op, aa, bb, False, out_type=out_type)


def _broadcast_shape(
    left: Sequence[int], right: Sequence[int]
) -> tuple[int, ...]:
    """Return the NumPy/Torch broadcast result without materializing either input."""
    left = tuple(int(size) for size in left)
    right = tuple(int(size) for size in right)
    rank = max(len(left), len(right))
    left = (1,) * (rank - len(left)) + left
    right = (1,) * (rank - len(right)) + right
    result = []
    for left_size, right_size in zip(left, right):
        if left_size == right_size:
            result.append(left_size)
        elif left_size == 1:
            result.append(right_size)
        elif right_size == 1:
            result.append(left_size)
        else:
            raise ValueError(
                f"operands are not broadcastable: {left} and {right}"
            )
    return tuple(result)


def _broadcast_index_source(
    name: str,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
) -> tuple[list[str], str]:
    """Emit output-linear-index to broadcast-input-index mapping."""
    input_shape = tuple(int(size) for size in input_shape)
    output_shape = tuple(int(size) for size in output_shape)
    if input_shape == output_shape:
        return [], "gid"
    if _broadcast_shape(input_shape, output_shape) != output_shape:
        raise ValueError(
            f"input shape {input_shape} does not broadcast to {output_shape}"
        )
    if not input_shape:
        return [], "uint(0)"

    input_strides = _row_major_strides(input_shape)
    output_strides = _row_major_strides(output_shape)
    offset = len(output_shape) - len(input_shape)
    lines = [
        f"    uint {name}_remaining = gid;",
        f"    uint {name}_index = uint(0);",
    ]
    for output_axis, output_stride in enumerate(output_strides):
        coordinate = f"{name}_coord{output_axis}"
        lines.extend(
            [
                f"    uint {coordinate} = "
                f"{name}_remaining / uint({output_stride});",
                f"    {name}_remaining %= uint({output_stride});",
            ]
        )
        input_axis = output_axis - offset
        if input_axis >= 0 and input_shape[input_axis] != 1:
            lines.append(
                f"    {name}_index += {coordinate} * "
                f"uint({input_strides[input_axis]});"
            )
    return lines, f"{name}_index"


def _emit_primitive_source(
    op: str,
    *,
    left_dtype: Any,
    right_dtype: Any | None,
    out_dtype: Any,
    left_shape: Sequence[int],
    right_shape: Sequence[int] | None,
    out_shape: Sequence[int],
    left_scalar: Any | None = None,
    right_scalar: Any | None = None,
    reverse: bool = False,
    local_size: int = _LOCAL_SIZE,
) -> str:
    feeds: list[tuple[str, Any, Sequence[int]]] = []
    index_lines: list[str] = []
    if left_scalar is None:
        feeds.append(("lhs", left_dtype, left_shape))
        left_lines, left_index = _broadcast_index_source(
            "lhs", left_shape, out_shape
        )
        index_lines.extend(left_lines)
        a = f"lhs[{left_index}]"
    else:
        a = _glsl_literal(left_scalar, left_dtype)
    if right_dtype is None:
        b = None
    elif right_scalar is None:
        assert right_shape is not None
        feeds.append(("rhs", right_dtype, right_shape))
        right_lines, right_index = _broadcast_index_source(
            "rhs", right_shape, out_shape
        )
        index_lines.extend(right_lines)
        b = f"rhs[{right_index}]"
    else:
        b = _glsl_literal(right_scalar, right_dtype)

    helper, expression = _typed_expr(
        op, a, b, reverse, left_dtype, right_dtype, out_dtype
    )
    lines = [_SHADER_HEADER.format(local_size=local_size)]
    for binding, (name, dtype, _) in enumerate(feeds):
        lines.append(
            f"layout(std430, binding = {binding}) readonly buffer "
            f"{name.title()}Buf {{ {_glsl_type(dtype)} {name}[]; }};"
        )
    lines.append(
        f"layout(std430, binding = {len(feeds)}) writeonly buffer OutBuf "
        f"{{ {_glsl_type(out_dtype)} outbuf[]; }};"
    )
    lines.extend(["", "uniform uint u_count;", ""])
    if helper:
        lines.extend([helper.rstrip(), ""])
    lines.extend(
        [
            "void main() {",
            "    uint gid = turing_linear_gid();",
            "    if (gid >= u_count) { return; }",
            *index_lines,
            f"    outbuf[gid] = {expression};",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def emit_op_source(
    op: str,
    *,
    scalar: Any | None = None,
    left_dtype: Any = np.float32,
    right_dtype: Any | None = None,
    output_dtype: Any | None = None,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Lower one canonical primitive to a standalone typed compute shader."""
    name, reverse = canonical_op(op)
    left_dtype = _normalize_dtype(left_dtype)
    if name in _UNARY:
        if scalar is not None:
            raise ValueError(f"unary op {op!r} given a scalar right operand")
        right_dtype = None
    else:
        right_dtype = _normalize_dtype(
            left_dtype if right_dtype is None else right_dtype
        )
    output_dtype = (
        _result_dtype(name, left_dtype, right_dtype)
        if output_dtype is None
        else _normalize_dtype(output_dtype)
    )
    return _emit_primitive_source(
        name,
        left_dtype=left_dtype,
        right_dtype=right_dtype,
        out_dtype=output_dtype,
        left_shape=(1,),
        right_shape=(1,) if right_dtype is not None else None,
        out_shape=(1,),
        right_scalar=scalar,
        reverse=reverse,
        local_size=local_size,
    )


# ---------------------------------------------------------------------------
# device-native creation
# ---------------------------------------------------------------------------

def _arange_count(start: Any, end: Any, step: Any) -> int:
    if step == 0:
        raise ValueError("arange step must be nonzero")
    distance = end - start
    if (step > 0 and distance <= 0) or (step < 0 and distance >= 0):
        return 0
    if all(isinstance(value, (int, np.integer)) for value in (start, end, step)):
        distance = abs(int(distance))
        stride = abs(int(step))
        return (distance + stride - 1) // stride
    count = int(math.ceil(float(distance) / float(step)))
    return max(0, count)


def _arange_dtype(start: Any, end: Any, step: Any, dtype: Any) -> np.dtype:
    if dtype is not None:
        result = _normalize_dtype(dtype)
    elif all(
        isinstance(value, (int, np.integer)) for value in (start, end, step)
    ):
        result = np.dtype(np.int32)
    else:
        result = np.dtype(np.float32)
    if result.kind == "b":
        raise TypeError("arange does not support boolean dtype")
    if result.kind == "u" and (start < 0 or end < 0 or step < 0):
        raise ValueError("unsigned arange requires non-negative bounds and step")
    return result


def emit_arange_source(
    start: Any,
    step: Any,
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Emit a device-native 1-D arithmetic-sequence creation shader."""
    dtype = _normalize_dtype(dtype)
    if dtype.kind == "b":
        raise TypeError("arange does not support boolean dtype")
    scalar_type = _glsl_type(dtype)
    start_literal = _glsl_literal(start, dtype)
    step_literal = _glsl_literal(step, dtype)
    gid_value = f"{scalar_type}(gid)"
    return "\n".join(
        [
            _STRUCTURAL_SHADER_HEADER.format(local_size=local_size),
            "layout(std430, binding = 0) writeonly buffer OutBuf "
            f"{{ {scalar_type} outbuf[]; }};",
            "",
            "uniform uint u_count;",
            "",
            "void main() {",
            "    uint gid = turing_linear_gid();",
            "    if (gid >= u_count) { return; }",
            f"    outbuf[gid] = {start_literal} + "
            f"{gid_value} * {step_literal};",
            "}",
            "",
        ]
    )


def _resolve_expand_shape(
    source_shape: Sequence[int], target_shape: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    source_shape = tuple(int(size) for size in source_shape)
    target = [int(size) for size in target_shape]
    if len(target) < len(source_shape):
        raise ValueError("cannot expand to fewer dimensions")
    aligned_source = (1,) * (len(target) - len(source_shape)) + source_shape
    for axis, (current, desired) in enumerate(zip(aligned_source, target)):
        if desired == -1:
            target[axis] = current
        elif desired < 0 or (current != desired and current != 1):
            raise ValueError(
                f"cannot expand dimension {axis} from {current} to {desired}"
            )
    return aligned_source, tuple(target)


def emit_expand_source(
    source_shape: Sequence[int],
    target_shape: Sequence[int],
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Emit one direct broadcast-copy shader without an expanded intermediate."""
    source_shape, target_shape = _resolve_expand_shape(
        source_shape, target_shape
    )
    dtype = _normalize_dtype(dtype)
    scalar_type = _glsl_type(dtype)
    index_lines, source_index = _broadcast_index_source(
        "source", source_shape, target_shape
    )
    return "\n".join(
        [
            _STRUCTURAL_SHADER_HEADER.format(local_size=local_size),
            "layout(std430, binding = 0) readonly buffer Input0 "
            f"{{ {scalar_type} input0[]; }};",
            "layout(std430, binding = 1) writeonly buffer OutBuf "
            f"{{ {scalar_type} outbuf[]; }};",
            "",
            "uniform uint u_count;",
            "",
            "void main() {",
            "    uint gid = turing_linear_gid();",
            "    if (gid >= u_count) { return; }",
            *index_lines,
            f"    outbuf[gid] = input0[{source_index}];",
            "}",
            "",
        ]
    )


# ---------------------------------------------------------------------------
# structural shader emission
#
# ``cat`` and ``stack`` are intentionally not new elementwise primitives.
# AbstractTensor has always exposed them as backend structural hooks, alongside
# the shape/layout family, because unlike metadata-only reshape they move an
# entire buffer and NumPy/Torch do not share one native calling convention.
# A composition such as unsqueeze-plus-cat remains mathematically valid, but a
# resident GPU backend can do better: map each output address directly to its
# source and complete the whole layout transform in one compute dispatch.
#
# These emitters keep that specialization isolated below the common tensor
# semantics.  They preserve arbitrary rank and dtype, never read an SSBO back
# through NumPy, and leave room for later GLSL-specific improvements (subgroup
# copies, shared-memory tiling, or multi-stage plans for very large input lists)
# without changing AbstractTensor's public ``cat``/``stack`` contract.
# ---------------------------------------------------------------------------

def _shape_product(shape: Sequence[int]) -> int:
    result = 1
    for size in shape:
        result *= int(size)
    return result


def _normalize_structural_dim(dim: int, rank: int, *, inserts_axis: bool) -> int:
    limit = rank + 1 if inserts_axis else rank
    if dim < 0:
        dim += limit
    if dim < 0 or dim >= limit:
        raise ValueError("dim out of range")
    return dim


def _validate_cat_layout(
    shapes: Sequence[Sequence[int]], dim: int
) -> tuple[tuple[tuple[int, ...], ...], int, tuple[int, ...]]:
    if not shapes:
        raise ValueError("tensors list cannot be empty")
    normalized = tuple(tuple(int(size) for size in shape) for shape in shapes)
    rank = len(normalized[0])
    if rank == 0:
        raise ValueError("zero-dimensional tensors cannot be concatenated")
    dim = _normalize_structural_dim(dim, rank, inserts_axis=False)
    base = normalized[0]
    for shape in normalized:
        if len(shape) != rank:
            raise ValueError("All tensors must have the same rank")
        if any(
            shape[axis] != base[axis]
            for axis in range(rank)
            if axis != dim
        ):
            raise ValueError("Non-concat dimensions must match")
    output = list(base)
    output[dim] = sum(shape[dim] for shape in normalized)
    return normalized, dim, tuple(output)


def emit_cat_source(
    shapes: Sequence[Sequence[int]],
    dim: int = 0,
    *,
    dtype: Any = np.float32,
    input_dtypes: Sequence[Any] | None = None,
    output_dtype: Any | None = None,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Emit one arbitrary-rank concatenate shader for homogeneous SSBOs."""
    shapes, dim, output_shape = _validate_cat_layout(shapes, dim)
    if input_dtypes is None:
        input_dtypes = [dtype] * len(shapes)
    if len(input_dtypes) != len(shapes):
        raise ValueError("input_dtypes must match the number of cat inputs")
    input_dtypes = tuple(_normalize_dtype(value) for value in input_dtypes)
    output_dtype = _normalize_dtype(
        dtype if output_dtype is None else output_dtype
    )
    output_type = _glsl_type(output_dtype)
    after = _shape_product(output_shape[dim + 1:])
    output_axis = output_shape[dim]

    lines = [_STRUCTURAL_SHADER_HEADER.format(local_size=local_size)]
    for index, input_dtype in enumerate(input_dtypes):
        lines.append(
            f"layout(std430, binding = {index}) readonly buffer Input{index} "
            f"{{ {_glsl_type(input_dtype)} input{index}[]; }};"
        )
    lines.extend(
        [
            f"layout(std430, binding = {len(shapes)}) writeonly buffer OutBuf "
            f"{{ {output_type} outbuf[]; }};",
            "",
            "uniform uint u_count;",
            "",
            "void main() {",
            "    uint gid = turing_linear_gid();",
            "    if (gid >= u_count) { return; }",
            f"    uint inner = gid % uint({after});",
            f"    uint block = gid / uint({after});",
            f"    uint axis_index = block % uint({output_axis});",
            f"    uint outer = block / uint({output_axis});",
        ]
    )

    prefix = 0
    for index, shape in enumerate(shapes):
        axis_size = shape[dim]
        condition = "if" if index == 0 else "else if"
        lines.extend(
            [
                f"    {condition} (axis_index < uint({prefix + axis_size})) {{",
                f"        uint local_axis = axis_index - uint({prefix});",
                "        uint source_index = "
                f"(outer * uint({axis_size}) + local_axis) * uint({after}) + inner;",
                f"        outbuf[gid] = "
                f"{output_type}(input{index}[source_index]);",
                "    }",
            ]
        )
        prefix += axis_size
    lines.extend(["}", ""])
    return "\n".join(lines)


def _validate_stack_layout(
    shape: Sequence[int], input_count: int, dim: int
) -> tuple[tuple[int, ...], int, tuple[int, ...]]:
    if input_count <= 0:
        raise ValueError("tensors list cannot be empty")
    shape = tuple(int(size) for size in shape)
    dim = _normalize_structural_dim(dim, len(shape), inserts_axis=True)
    output = shape[:dim] + (int(input_count),) + shape[dim:]
    return shape, dim, output


def _validate_permute_layout(
    shape: Sequence[int], dims: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    shape = tuple(int(size) for size in shape)
    rank = len(shape)
    if len(dims) != rank:
        raise ValueError("permute requires dims to match tensor dimensions")
    normalized = []
    for axis in dims:
        axis = int(axis)
        if not -rank <= axis < rank:
            raise ValueError("permute dimension out of range")
        normalized.append(axis % rank)
    if sorted(normalized) != list(range(rank)):
        raise ValueError("dims must be a permutation of tensor axes")
    output_shape = tuple(shape[axis] for axis in normalized)
    return shape, tuple(normalized), output_shape


def _resolve_reshape_shape(shape: Sequence[int], count: int) -> tuple[int, ...]:
    shape = [int(size) for size in shape]
    inferred = [index for index, size in enumerate(shape) if size == -1]
    if len(inferred) > 1:
        raise ValueError("only one inferred dimension is permitted")
    if any(size < -1 for size in shape):
        raise ValueError("reshape dimensions must be non-negative or -1")

    known = _shape_product(size for size in shape if size != -1)
    if inferred:
        if known == 0 or count % known:
            raise ValueError("shape is incompatible with tensor size")
        shape[inferred[0]] = count // known
    if _shape_product(shape) != count:
        raise ValueError("shape is incompatible with tensor size")
    return tuple(shape)


def _row_major_strides(shape: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    result = [1] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        result[axis] = stride
        stride *= int(shape[axis])
    return tuple(result)


def emit_permute_source(
    shape: Sequence[int],
    dims: Sequence[int],
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Emit one arbitrary-rank row-major axis-permutation shader."""
    shape, dims, output_shape = _validate_permute_layout(shape, dims)
    scalar_type = _glsl_type(dtype)
    input_strides = _row_major_strides(shape)
    output_strides = _row_major_strides(output_shape)
    lines = [
        _STRUCTURAL_SHADER_HEADER.format(local_size=local_size),
        "layout(std430, binding = 0) readonly buffer Input0 "
        f"{{ {scalar_type} input0[]; }};",
        "layout(std430, binding = 1) writeonly buffer OutBuf "
        f"{{ {scalar_type} outbuf[]; }};",
        "",
        "uniform uint u_count;",
        "",
        "void main() {",
        "    uint gid = turing_linear_gid();",
        "    if (gid >= u_count) { return; }",
        "    uint remaining = gid;",
        "    uint source_index = uint(0);",
    ]
    for output_axis, source_axis in enumerate(dims):
        output_stride = output_strides[output_axis]
        source_stride = input_strides[source_axis]
        lines.extend(
            [
                f"    uint coord{output_axis} = "
                f"remaining / uint({output_stride});",
                f"    remaining %= uint({output_stride});",
                f"    source_index += coord{output_axis} * "
                f"uint({source_stride});",
            ]
        )
    lines.extend(["    outbuf[gid] = input0[source_index];", "}", ""])
    return "\n".join(lines)


def _matmul_layout(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    left_shape = tuple(int(size) for size in left_shape)
    right_shape = tuple(int(size) for size in right_shape)
    if len(left_shape) < 2 or len(right_shape) < 2:
        raise ValueError("matmul expects tensors with at least two dimensions")
    rows, inner = left_shape[-2:]
    inner_right, columns = right_shape[-2:]
    if inner != inner_right:
        raise ValueError(
            f"matmul inner dimensions do not match: {inner} != {inner_right}"
        )
    batch_shape = _broadcast_shape(left_shape[:-2], right_shape[:-2])
    batch_rank = len(batch_shape)
    padded_left = (
        (1,) * (batch_rank - (len(left_shape) - 2)) + left_shape
    )
    padded_right = (
        (1,) * (batch_rank - (len(right_shape) - 2)) + right_shape
    )
    output_shape = batch_shape + (rows, columns)
    return padded_left, padded_right, batch_shape, output_shape


def emit_matmul_source(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
    *,
    left_dtype: Any = np.float32,
    right_dtype: Any = np.float32,
    output_dtype: Any | None = None,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Emit one cooperative tiled, broadcasted batched matmul shader."""
    left_shape, right_shape, batch_shape, output_shape = _matmul_layout(
        left_shape, right_shape
    )
    left_dtype = _normalize_dtype(left_dtype)
    right_dtype = _normalize_dtype(right_dtype)
    output_dtype = (
        _promote_dtype(left_dtype, right_dtype)
        if output_dtype is None
        else _normalize_dtype(output_dtype)
    )
    if output_dtype.kind == "b":
        raise TypeError("matmul does not support boolean output")

    rows, inner = left_shape[-2:]
    columns = right_shape[-1]
    left_strides = _row_major_strides(left_shape)
    right_strides = _row_major_strides(right_shape)
    batch_strides = _row_major_strides(batch_shape)
    batch_count = _shape_product(batch_shape)
    output_type = _glsl_type(output_dtype)
    tile = 1 << max(0, int(math.isqrt(int(local_size))).bit_length() - 1)
    tile = min(tile, 16)
    thread_count = tile * tile
    row_tiles = (rows + tile - 1) // tile
    column_tiles = (columns + tile - 1) // tile
    group_count = batch_count * row_tiles * column_tiles
    lines = [
        _STRUCTURAL_SHADER_HEADER.format(local_size=thread_count),
        "layout(std430, binding = 0) readonly buffer LeftBuf "
        f"{{ {_glsl_type(left_dtype)} lhs[]; }};",
        "layout(std430, binding = 1) readonly buffer RightBuf "
        f"{{ {_glsl_type(right_dtype)} rhs[]; }};",
        "layout(std430, binding = 2) writeonly buffer OutBuf "
        f"{{ {output_type} outbuf[]; }};",
        "",
        "uniform uint u_count;",
        f"shared {output_type} left_tile[{tile}][{tile}];",
        f"shared {output_type} right_tile[{tile}][{tile}];",
        "",
        "void main() {",
        "    uint group_index = gl_WorkGroupID.x",
        "        + gl_WorkGroupID.y * gl_NumWorkGroups.x",
        "        + gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y;",
        f"    if (group_index >= uint({group_count})) {{ return; }}",
        f"    uint tile_column = group_index % uint({column_tiles});",
        f"    uint matrix_tile = group_index / uint({column_tiles});",
        f"    uint tile_row = matrix_tile % uint({row_tiles});",
        f"    uint batch_index = matrix_tile / uint({row_tiles});",
        "    uint local_index = gl_LocalInvocationIndex;",
        f"    uint local_row = local_index / uint({tile});",
        f"    uint local_column = local_index % uint({tile});",
        f"    uint row = tile_row * uint({tile}) + local_row;",
        f"    uint column = tile_column * uint({tile}) + local_column;",
        "    uint batch_remaining = batch_index;",
        "    uint left_offset = uint(0);",
        "    uint right_offset = uint(0);",
    ]
    for axis, batch_stride in enumerate(batch_strides):
        lines.extend(
            [
                f"    uint batch_coord{axis} = "
                f"batch_remaining / uint({batch_stride});",
                f"    batch_remaining %= uint({batch_stride});",
            ]
        )
        if left_shape[axis] != 1:
            lines.append(
                f"    left_offset += batch_coord{axis} * "
                f"uint({left_strides[axis]});"
            )
        if right_shape[axis] != 1:
            lines.append(
                f"    right_offset += batch_coord{axis} * "
                f"uint({right_strides[axis]});"
            )
    lines.extend(
        [
            f"    {output_type} total = {output_type}(0);",
            f"    for (uint tile_k = uint(0); tile_k < "
            f"uint({(inner + tile - 1) // tile}); ++tile_k) {{",
            f"        uint left_k = tile_k * uint({tile}) + local_column;",
            f"        uint right_k = tile_k * uint({tile}) + local_row;",
            f"        left_tile[local_row][local_column] = "
            f"(row < uint({rows}) && left_k < uint({inner}))",
            f"            ? {output_type}(lhs[left_offset + "
            f"row * uint({inner}) + left_k]) : {output_type}(0);",
            f"        right_tile[local_row][local_column] = "
            f"(right_k < uint({inner}) && column < uint({columns}))",
            f"            ? {output_type}(rhs[right_offset + "
            f"right_k * uint({columns}) + column]) : {output_type}(0);",
            "        barrier();",
            f"        for (uint k = uint(0); k < uint({tile}); ++k) {{",
            "            total += left_tile[local_row][k] "
            "* right_tile[k][local_column];",
            "        }",
            "        barrier();",
            "    }",
            f"    if (row < uint({rows}) && column < uint({columns})) {{",
            f"        uint output_index = (batch_index * uint({rows}) + row) "
            f"* uint({columns}) + column;",
            "        outbuf[output_index] = total;",
            "    }",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_repeat_layout(
    source_shape: Sequence[int],
    repeats: Any,
    dim: int = 0,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    source_shape = tuple(int(size) for size in source_shape)
    if repeats is None:
        raise ValueError("repeats must be specified")
    if isinstance(repeats, (int, np.integer)):
        if not source_shape:
            source_shape = (1,)
        factors = [1] * len(source_shape)
        factors[int(dim) % len(source_shape)] = int(repeats)
    else:
        factors = [int(factor) for factor in repeats]
        if len(factors) < len(source_shape):
            factors = [1] * (len(source_shape) - len(factors)) + factors
        elif len(factors) > len(source_shape):
            source_shape = (
                (1,) * (len(factors) - len(source_shape)) + source_shape
            )
    if any(factor < 0 for factor in factors):
        raise ValueError("repeat factors must be non-negative")
    output_shape = tuple(
        size * factor for size, factor in zip(source_shape, factors)
    )
    return source_shape, tuple(factors), output_shape


def emit_repeat_source(
    source_shape: Sequence[int],
    repeats: Any,
    dim: int = 0,
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Emit one arbitrary-rank tile/repeat shader."""
    source_shape, _, output_shape = _resolve_repeat_layout(
        source_shape, repeats, dim
    )
    scalar_type = _glsl_type(dtype)
    source_strides = _row_major_strides(source_shape)
    output_strides = _row_major_strides(output_shape)
    lines = [
        _STRUCTURAL_SHADER_HEADER.format(local_size=local_size),
        "layout(std430, binding = 0) readonly buffer Input0 "
        f"{{ {scalar_type} input0[]; }};",
        "layout(std430, binding = 1) writeonly buffer OutBuf "
        f"{{ {scalar_type} outbuf[]; }};",
        "",
        "uniform uint u_count;",
        "",
        "void main() {",
        "    uint gid = turing_linear_gid();",
        "    if (gid >= u_count) { return; }",
        "    uint remaining = gid;",
        "    uint source_index = uint(0);",
    ]
    for axis, (source_size, source_stride, output_stride) in enumerate(
        zip(source_shape, source_strides, output_strides)
    ):
        lines.extend(
            [
                f"    uint coord{axis} = "
                f"remaining / uint({output_stride});",
                f"    remaining %= uint({output_stride});",
                f"    source_index += (coord{axis} % uint({source_size})) "
                f"* uint({source_stride});",
            ]
        )
    lines.extend(["    outbuf[gid] = input0[source_index];", "}", ""])
    return "\n".join(lines)


def emit_gather_source(
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Emit an arbitrary-offset gather over a resident flat tensor."""
    scalar_type = _glsl_type(dtype)
    return "\n".join(
        [
            _STRUCTURAL_SHADER_HEADER.format(local_size=local_size),
            "layout(std430, binding = 0) readonly buffer Input0 "
            f"{{ {scalar_type} input0[]; }};",
            "layout(std430, binding = 1) readonly buffer OffsetBuf "
            "{ int offsets[]; };",
            "layout(std430, binding = 2) writeonly buffer OutBuf "
            f"{{ {scalar_type} outbuf[]; }};",
            "",
            "uniform uint u_count;",
            "",
            "void main() {",
            "    uint gid = turing_linear_gid();",
            "    if (gid >= u_count) { return; }",
            "    outbuf[gid] = input0[uint(offsets[gid])];",
            "}",
            "",
        ]
    )


def emit_topk_offsets_source(
    shape: Sequence[int],
    k: int,
    dim: int,
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Emit a deterministic arbitrary-axis top-k offset selector.

    Each invocation owns one output rank in one axis slice. It performs a small
    repeated selection locally and writes the selected flat source offset.
    Values are then obtained through the backend's ordinary resident gather,
    keeping selection and transport as reusable structural primitives.
    """

    shape = tuple(int(size) for size in shape)
    if not shape:
        raise ValueError("topk requires a tensor with at least one dimension")
    dim = int(dim)
    if dim < 0:
        dim += len(shape)
    if dim < 0 or dim >= len(shape):
        raise ValueError("topk dimension out of range")
    axis_size = shape[dim]
    k = int(k)
    if k < 1 or k > axis_size:
        raise ValueError("topk k must be between one and the selected axis size")
    dtype = _normalize_dtype(dtype)
    scalar_type = _glsl_type(dtype)
    inner = _shape_product(shape[dim + 1:])
    nan_order = ""
    if dtype.kind == "f":
        nan_order = (
            "            bool candidate_nan = isnan(candidate);\n"
            "            bool best_nan = isnan(best);\n"
            "            bool better = !found\n"
            "                || (candidate_nan && !best_nan)\n"
            "                || (candidate_nan == best_nan\n"
            "                    && (candidate > best\n"
            "                        || (candidate == best\n"
            "                            && axis_index < best_axis)));"
        )
    else:
        nan_order = (
            "            bool better = !found || candidate > best\n"
            "                || (candidate == best && axis_index < best_axis);"
        )
    return "\n".join(
        [
            _STRUCTURAL_SHADER_HEADER.format(local_size=local_size),
            "layout(std430, binding = 0) readonly buffer Input0 "
            f"{{ {scalar_type} input0[]; }};",
            "layout(std430, binding = 1) writeonly buffer OffsetBuf "
            "{ int offsets[]; };",
            "",
            "uniform uint u_count;",
            "",
            "void main() {",
            "    uint gid = turing_linear_gid();",
            "    if (gid >= u_count) { return; }",
            f"    uint inner_index = gid % uint({inner});",
            f"    uint slot = gid / uint({inner});",
            f"    uint rank = slot % uint({k});",
            f"    uint outer = slot / uint({k});",
            f"    uint base = outer * uint({axis_size * inner}) + inner_index;",
            f"    uint chosen[{k}];",
            "    uint best_axis = uint(0);",
            "    for (uint selection = uint(0); selection <= rank; "
            "++selection) {",
            "        bool found = false;",
            f"        {scalar_type} best = {scalar_type}(0);",
            f"        for (uint axis_index = uint(0); axis_index < "
            f"uint({axis_size}); ++axis_index) {{",
            "            bool used = false;",
            "            for (uint prior = uint(0); prior < selection; "
            "++prior) {",
            "                used = used || chosen[prior] == axis_index;",
            "            }",
            "            if (used) { continue; }",
            f"            {scalar_type} candidate = "
            f"input0[base + axis_index * uint({inner})];",
            nan_order,
            "            if (better) {",
            "                found = true;",
            "                best = candidate;",
            "                best_axis = axis_index;",
            "            }",
            "        }",
            "        chosen[selection] = best_axis;",
            "    }",
            f"    offsets[gid] = int(base + best_axis * uint({inner}));",
            "}",
            "",
        ]
    )


def emit_index_assign_source(
    *,
    dtype: Any = np.float32,
    index_dtype: Any = np.int32,
    scalar_value: bool = False,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Emit arbitrary-offset assignment into an existing resident tensor."""
    scalar_type = _glsl_type(dtype)
    index_type = _glsl_type(index_dtype)
    value_index = "uint(0)" if scalar_value else "gid"
    return "\n".join(
        [
            _STRUCTURAL_SHADER_HEADER.format(local_size=local_size),
            "layout(std430, binding = 0) readonly buffer OffsetBuf "
            f"{{ {index_type} offsets[]; }};",
            "layout(std430, binding = 1) readonly buffer ValueBuf "
            f"{{ {scalar_type} values[]; }};",
            "layout(std430, binding = 2) buffer OutBuf "
            f"{{ {scalar_type} outbuf[]; }};",
            "",
            "uniform uint u_count;",
            "",
            "void main() {",
            "    uint gid = turing_linear_gid();",
            "    if (gid >= u_count) { return; }",
            f"    outbuf[uint(offsets[gid])] = values[{value_index}];",
            "}",
            "",
        ]
    )


def emit_index_select_source(
    shape: Sequence[int],
    dim: int,
    index_count: int,
    *,
    dtype: Any = np.float32,
    index_dtype: Any = np.int32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Emit a shaped integer-array selection along one source axis."""
    shape = tuple(int(size) for size in shape)
    dim = int(dim) % len(shape)
    axis_size = shape[dim]
    after = _shape_product(shape[dim + 1:])
    scalar_type = _glsl_type(dtype)
    index_type = _glsl_type(index_dtype)
    selected_lines = (
        [
            "    int selected = indices[index_position];",
            f"    if (selected < 0) {{ selected += int({axis_size}); }}",
            "    uint selected_index = uint(selected);",
        ]
        if _normalize_dtype(index_dtype).kind == "i"
        else ["    uint selected_index = uint(indices[index_position]);"]
    )
    return "\n".join(
        [
            _STRUCTURAL_SHADER_HEADER.format(local_size=local_size),
            "layout(std430, binding = 0) readonly buffer Input0 "
            f"{{ {scalar_type} input0[]; }};",
            "layout(std430, binding = 1) readonly buffer IndexBuf "
            f"{{ {index_type} indices[]; }};",
            "layout(std430, binding = 2) writeonly buffer OutBuf "
            f"{{ {scalar_type} outbuf[]; }};",
            "",
            "uniform uint u_count;",
            "",
            "void main() {",
            "    uint gid = turing_linear_gid();",
            "    if (gid >= u_count) { return; }",
            f"    uint inner = gid % uint({after});",
            f"    uint block = gid / uint({after});",
            f"    uint index_position = block % uint({index_count});",
            f"    uint outer = block / uint({index_count});",
            *selected_lines,
            f"    uint source_index = (outer * uint({axis_size}) + "
            f"selected_index) * uint({after}) + inner;",
            "    outbuf[gid] = input0[source_index];",
            "}",
            "",
        ]
    )


def emit_slice_axis_source(
    shape: Sequence[int],
    dim: int,
    start: int,
    step: int,
    count: int,
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Emit an affine slice along one axis without an index buffer."""
    shape = tuple(int(size) for size in shape)
    dim = int(dim) % len(shape)
    axis_size = shape[dim]
    after = _shape_product(shape[dim + 1:])
    scalar_type = _glsl_type(dtype)
    return "\n".join(
        [
            _STRUCTURAL_SHADER_HEADER.format(local_size=local_size),
            "layout(std430, binding = 0) readonly buffer Input0 "
            f"{{ {scalar_type} input0[]; }};",
            "layout(std430, binding = 1) writeonly buffer OutBuf "
            f"{{ {scalar_type} outbuf[]; }};",
            "",
            "uniform uint u_count;",
            "",
            "void main() {",
            "    uint gid = turing_linear_gid();",
            "    if (gid >= u_count) { return; }",
            f"    uint inner = gid % uint({after});",
            f"    uint block = gid / uint({after});",
            f"    uint selected_position = block % uint({count});",
            f"    uint outer = block / uint({count});",
            f"    int selected = int({start}) + "
            f"int(selected_position) * int({step});",
            f"    uint source_index = (outer * uint({axis_size}) + "
            f"uint(selected)) * uint({after}) + inner;",
            "    outbuf[gid] = input0[source_index];",
            "}",
            "",
        ]
    )


def _reduce_layout(
    shape: Sequence[int],
    dim: int | None,
    keepdim: bool,
) -> tuple[tuple[int, ...], int, tuple[int, ...], int]:
    shape = tuple(int(size) for size in shape)
    if dim is None:
        source_shape = (_shape_product(shape),)
        axis = 0
        output_shape = (1,) * len(shape) if keepdim else ()
    else:
        if not shape:
            raise ValueError("cannot reduce a scalar along an explicit dimension")
        axis = int(dim)
        if axis < 0:
            axis += len(shape)
        if axis < 0 or axis >= len(shape):
            raise ValueError("dim out of range")
        source_shape = shape
        output_shape = (
            shape[:axis] + (1,) + shape[axis + 1:]
            if keepdim
            else shape[:axis] + shape[axis + 1:]
        )
    return source_shape, axis, output_shape, source_shape[axis]


def _reduction_dtype(op: str, dtype: Any) -> np.dtype:
    dtype = _normalize_dtype(dtype)
    if op in {"any", "all"}:
        return np.dtype(np.bool_)
    if op == "mean":
        return np.dtype(np.float32)
    if op == "sum" and dtype.kind == "b":
        return np.dtype(np.int32)
    return dtype


def emit_reduce_source(
    op: str,
    shape: Sequence[int],
    dim: int | None = None,
    keepdim: bool = False,
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Emit one axis reduction shader with one invocation per output value."""
    if op not in {"sum", "mean", "min", "max", "any", "all"}:
        raise ValueError(f"unsupported GLSL reduction {op!r}")
    source_shape, axis, _, extent = _reduce_layout(shape, dim, keepdim)
    input_dtype = _normalize_dtype(dtype)
    output_dtype = _reduction_dtype(op, input_dtype)
    input_type = _glsl_type(input_dtype)
    output_type = _glsl_type(output_dtype)
    after = _shape_product(source_shape[axis + 1:])

    if op in {"min", "max"} and extent == 0:
        raise ValueError(f"{op} reduction has no identity for an empty axis")
    if op == "all":
        initial = "uint(1)"
    else:
        initial = f"{output_type}(0)"

    value = f"{output_type}(input0[source_index])"
    if op == "sum":
        update = f"total += {value};"
    elif op == "mean":
        update = f"total += float(input0[source_index]);"
    elif op == "min":
        update = (
            f"total = (k == uint(0)) ? {value} : min(total, {value});"
        )
    elif op == "max":
        update = (
            f"total = (k == uint(0)) ? {value} : max(total, {value});"
        )
    elif op == "any":
        update = "total |= uint(input0[source_index] != 0);"
    else:
        update = "total &= uint(input0[source_index] != 0);"

    final = f"total / float({extent})" if op == "mean" and extent else "total"
    lines = [
        _STRUCTURAL_SHADER_HEADER.format(local_size=local_size),
        "layout(std430, binding = 0) readonly buffer Input0 "
        f"{{ {input_type} input0[]; }};",
        "layout(std430, binding = 1) writeonly buffer OutBuf "
        f"{{ {output_type} outbuf[]; }};",
        "",
        "uniform uint u_count;",
        "",
        "void main() {",
        "    uint gid = turing_linear_gid();",
        "    if (gid >= u_count) { return; }",
        f"    uint inner = gid % uint({after});",
        f"    uint outer = gid / uint({after});",
        f"    uint base = outer * uint({extent * after}) + inner;",
        f"    {output_type} total = {initial};",
        f"    for (uint k = uint(0); k < uint({extent}); ++k) {{",
        f"        uint source_index = base + k * uint({after});",
        f"        {update}",
        "    }",
        f"    outbuf[gid] = {final};",
        "}",
        "",
    ]
    return "\n".join(lines)


def emit_cumsum_source(
    shape: Sequence[int],
    dim: int = 0,
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Emit an axis-line prefix-sum kernel with one invocation per line."""
    shape = tuple(int(size) for size in shape)
    if not shape:
        raise ValueError("cumsum requires at least one dimension")
    dim = int(dim)
    if dim < 0:
        dim += len(shape)
    if dim < 0 or dim >= len(shape):
        raise ValueError("dim out of range")
    input_dtype = _normalize_dtype(dtype)
    output_dtype = _reduction_dtype("sum", input_dtype)
    input_type = _glsl_type(input_dtype)
    output_type = _glsl_type(output_dtype)
    extent = shape[dim]
    after = _shape_product(shape[dim + 1:])
    return "\n".join(
        [
            _STRUCTURAL_SHADER_HEADER.format(local_size=local_size),
            "layout(std430, binding = 0) readonly buffer Input0 "
            f"{{ {input_type} input0[]; }};",
            "layout(std430, binding = 1) writeonly buffer OutBuf "
            f"{{ {output_type} outbuf[]; }};",
            "",
            "uniform uint u_count;",
            "",
            "void main() {",
            "    uint line = turing_linear_gid();",
            "    if (line >= u_count) { return; }",
            f"    uint inner = line % uint({after});",
            f"    uint outer = line / uint({after});",
            f"    uint base = outer * uint({extent * after}) + inner;",
            f"    {output_type} total = {output_type}(0);",
            f"    for (uint k = uint(0); k < uint({extent}); ++k) {{",
            f"        uint index = base + k * uint({after});",
            f"        total += {output_type}(input0[index]);",
            "        outbuf[index] = total;",
            "    }",
            "}",
            "",
        ]
    )


def emit_stack_source(
    shape: Sequence[int],
    input_count: int,
    dim: int = 0,
    *,
    dtype: Any = np.float32,
    input_dtypes: Sequence[Any] | None = None,
    output_dtype: Any | None = None,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Emit one arbitrary-rank stack shader for equally-shaped SSBOs."""
    shape, dim, _ = _validate_stack_layout(shape, input_count, dim)
    if input_dtypes is None:
        input_dtypes = [dtype] * input_count
    if len(input_dtypes) != input_count:
        raise ValueError("input_dtypes must match the number of stack inputs")
    input_dtypes = tuple(_normalize_dtype(value) for value in input_dtypes)
    output_dtype = _normalize_dtype(
        dtype if output_dtype is None else output_dtype
    )
    output_type = _glsl_type(output_dtype)
    after = _shape_product(shape[dim:])

    lines = [_STRUCTURAL_SHADER_HEADER.format(local_size=local_size)]
    for index, input_dtype in enumerate(input_dtypes):
        lines.append(
            f"layout(std430, binding = {index}) readonly buffer Input{index} "
            f"{{ {_glsl_type(input_dtype)} input{index}[]; }};"
        )
    lines.extend(
        [
            f"layout(std430, binding = {input_count}) writeonly buffer OutBuf "
            f"{{ {output_type} outbuf[]; }};",
            "",
            "uniform uint u_count;",
            "",
            "void main() {",
            "    uint gid = turing_linear_gid();",
            "    if (gid >= u_count) { return; }",
            f"    uint inner = gid % uint({after});",
            f"    uint block = gid / uint({after});",
            f"    uint source_number = block % uint({input_count});",
            f"    uint outer = block / uint({input_count});",
            f"    uint source_index = outer * uint({after}) + inner;",
        ]
    )
    for index in range(input_count):
        condition = "if" if index == 0 else "else if"
        lines.extend(
            [
                f"    {condition} (source_number == uint({index})) {{",
                f"        outbuf[gid] = "
                f"{output_type}(input{index}[source_index]);",
                "    }",
            ]
        )
    lines.extend(["}", ""])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# compilation + cache
# ---------------------------------------------------------------------------

_program_cache: dict[str, int] = {}
_uniform_location_cache: dict[tuple[int, str], int] = {}
_cache_stats = {"hits": 0, "misses": 0}


@dataclass(frozen=True)
class _DeferredElementwise:
    """One not-yet-dispatched GLSL expression region and its concrete feeds."""

    program: FusedProgram
    feeds: Mapping[int, GLChunk]


_fusion_depth: ContextVar[int] = ContextVar("glsl_fusion_depth", default=0)
_deferred_value_ids = itertools.count(start=-1, step=-1)


@contextmanager
def fuse_elementwise():
    """Defer compatible GLSL primitives and emit each region as one shader.

    The scope is deliberately opt-in. Layout changes, reductions, indexing,
    matmul, explicit transfers, and access to a buffer ID materialize the
    current region. This preserves ordinary eager AbstractTensor behavior while
    allowing a caller with a dense calculation to request register-resident
    intermediates without constructing a second graph format.
    """

    token = _fusion_depth.set(_fusion_depth.get() + 1)
    try:
        yield
    finally:
        _fusion_depth.reset(token)


def shader_cache_stats() -> dict[str, int]:
    return dict(_cache_stats, size=len(_program_cache))


def _compile(source: str) -> int:
    """Compile+link a compute shader, caching by source hash."""
    key = hashlib.sha256(source.encode("utf-8")).hexdigest()
    cached = _program_cache.get(key)
    if cached is not None:
        _cache_stats["hits"] += 1
        return cached
    _cache_stats["misses"] += 1

    require_gl_context()
    from OpenGL import GL

    shader = GL.glCreateShader(GL.GL_COMPUTE_SHADER)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader)
        GL.glDeleteShader(shader)
        raise GLSLCompileError(_annotate(source, log))

    program = GL.glCreateProgram()
    GL.glAttachShader(program, shader)
    GL.glLinkProgram(program)
    GL.glDeleteShader(shader)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(program)
        GL.glDeleteProgram(program)
        raise GLSLCompileError(_annotate(source, log))

    _program_cache[key] = program
    return program


def _annotate(source: str, log: Any) -> str:
    """Driver logs cite line numbers; print the numbered source beside them."""
    if isinstance(log, bytes):
        log = log.decode("utf-8", "replace")
    numbered = "\n".join(
        f"{n:4d} | {line}" for n, line in enumerate(source.splitlines(), 1)
    )
    return f"{str(log).strip()}\n--- generated shader ---\n{numbered}"


# ---------------------------------------------------------------------------
# execution
#
# Backend-surface inventory note: this is currently a narrow 1-D launcher, not
# yet the generalized launch strategy the expanding GLSL backend will need.
# All present shaders linearize their work and compile with a fixed local size,
# so ceil(element_count / local_size) is sufficient today.  A general wrapper
# must plan *before source emission*, because GLSL 4.3 bakes local_size_x/y/z
# into the shader.  It should cache per-context compute limits, choose a
# kernel-appropriate local shape, fold oversized logical grids across the
# available x/y/z group counts (or issue base-offset tiles), validate SSBO
# bindings, skip zero-work dispatches, bind typed uniforms/outputs, and own the
# visibility barrier policy.  Keeping that machinery here rather than in each
# operator will let elementwise, structural, tiled matmul, convolution, and
# future indirect launches share one device-aware planner without changing
# their mathematical lowering.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GLComputeLimits:
    """Compute-launch limits for the active OpenGL device."""

    max_group_count: tuple[int, int, int]
    max_group_size: tuple[int, int, int]
    max_invocations: int
    max_ssbo_bindings: int
    max_compute_ssbo_blocks: int

    @property
    def max_dispatch_ssbo_blocks(self) -> int:
        return min(self.max_ssbo_bindings, self.max_compute_ssbo_blocks)


@dataclass(frozen=True)
class GLLaunchPlan:
    """A device-valid launch shape shared by emission and dispatch."""

    count: int
    local_size: int
    groups: tuple[int, int, int]
    limits: GLComputeLimits

    @property
    def skipped(self) -> bool:
        return self.count == 0


_compute_limits_cache: dict[tuple[str, str, str], GLComputeLimits] = {}


def _first_gl_integer(value: Any) -> int:
    """Normalize PyOpenGL scalar/one-element/vector query return styles."""
    values = np.asarray(value).reshape(-1)
    if values.size == 0:
        raise RuntimeError("OpenGL returned an empty integer capability")
    return int(values[0])


def _compute_limits() -> GLComputeLimits:
    info = require_gl_context()
    key = (
        str(info.get("vendor", "")),
        str(info.get("renderer", "")),
        str(info.get("version", "")),
    )
    cached = _compute_limits_cache.get(key)
    if cached is not None:
        return cached

    from OpenGL import GL

    limits = GLComputeLimits(
        max_group_count=tuple(
            _first_gl_integer(
                GL.glGetIntegeri_v(GL.GL_MAX_COMPUTE_WORK_GROUP_COUNT, axis)
            )
            for axis in range(3)
        ),
        max_group_size=tuple(
            _first_gl_integer(
                GL.glGetIntegeri_v(GL.GL_MAX_COMPUTE_WORK_GROUP_SIZE, axis)
            )
            for axis in range(3)
        ),
        max_invocations=_first_gl_integer(
            GL.glGetIntegerv(GL.GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS)
        ),
        max_ssbo_bindings=_first_gl_integer(
            GL.glGetIntegerv(GL.GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS)
        ),
        max_compute_ssbo_blocks=_first_gl_integer(
            GL.glGetIntegerv(GL.GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS)
        ),
    )
    _compute_limits_cache[key] = limits
    return limits


def _power_of_two_at_most(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (int(value).bit_length() - 1)


def plan_launch(
    count: int,
    *,
    preferred_local_size: int = _LOCAL_SIZE,
    binding_count: int = 0,
) -> GLLaunchPlan:
    """Choose a valid local size and x/y/z dispatch grid for ``count`` items.

    Local size is selected from the active device limits and small workloads
    use a smaller power-of-two group (down to 32 when the device permits).
    Workgroup counts that exceed the x dimension are folded across y and z.
    Generated shaders use :func:`turing_linear_gid` so that this remains one
    contiguous logical index space.
    """
    count = int(count)
    if count < 0:
        raise ValueError("launch count cannot be negative")
    if count > 0xFFFFFFFF:
        raise ValueError(
            "launch count exceeds the uint u_count contract; use a tiled "
            "base-offset launch"
        )
    if preferred_local_size <= 0:
        raise ValueError("preferred local size must be positive")
    if binding_count < 0:
        raise ValueError("binding count cannot be negative")

    limits = _compute_limits()
    if binding_count > limits.max_dispatch_ssbo_blocks:
        raise ValueError(
            f"launch requires {binding_count} SSBO bindings, but the active "
            "compute stage supports "
            f"{limits.max_dispatch_ssbo_blocks}"
        )

    local_cap = min(
        int(preferred_local_size),
        limits.max_group_size[0],
        limits.max_invocations,
    )
    local_size = _power_of_two_at_most(local_cap)
    if count:
        small_target = max(1, 1 << (count - 1).bit_length())
        minimum_group = min(32, local_size)
        local_size = min(local_size, max(minimum_group, small_target))

    if count == 0:
        return GLLaunchPlan(count, local_size, (0, 0, 0), limits)

    groups_needed = (count + local_size - 1) // local_size
    group_x = min(groups_needed, limits.max_group_count[0])
    remaining = (groups_needed + group_x - 1) // group_x
    group_y = min(remaining, limits.max_group_count[1])
    remaining = (remaining + group_y - 1) // group_y
    group_z = remaining
    if group_z > limits.max_group_count[2]:
        capacity = (
            limits.max_group_count[0]
            * limits.max_group_count[1]
            * limits.max_group_count[2]
            * local_size
        )
        raise ValueError(
            f"launch count {count} exceeds one-dispatch capacity {capacity}; "
            "the caller must use a base-offset tiled launch"
        )
    return GLLaunchPlan(
        count,
        local_size,
        (int(group_x), int(group_y), int(group_z)),
        limits,
    )


def _dispatch(
    program_id: int,
    chunks: Sequence[GLChunk],
    out: GLChunk,
    plan: GLLaunchPlan,
) -> None:
    _dispatch_many(program_id, chunks, (out,), plan)


def _dispatch_many(
    program_id: int,
    chunks: Sequence[GLChunk],
    outputs: Sequence[GLChunk],
    plan: GLLaunchPlan,
) -> None:
    from OpenGL import GL

    if plan.skipped:
        return
    if not outputs:
        raise ValueError("a compute dispatch needs at least one output")
    binding_count = len(chunks) + len(outputs)
    if binding_count > plan.limits.max_dispatch_ssbo_blocks:
        raise ValueError(
            f"launch requires {binding_count} SSBO bindings, but the active "
            "compute stage supports "
            f"{plan.limits.max_dispatch_ssbo_blocks}"
        )

    # Deferred inputs can themselves execute a fused program. Materialize every
    # nested region before binding this dispatch's program; otherwise the inner
    # dispatch correctly restores program 0 and accidentally unbinds the outer
    # program before its launch.
    for chunk in chunks:
        chunk._to_gpu_current()
    for output in outputs:
        output._to_gpu_current()

    # PyOpenGL normally calls glGetError after every individual state change.
    # A dispatch performs a dozen tightly related calls, so that policy can do
    # more driver round-trips than the shader itself. Preserve error detection
    # while checking once at the dispatch boundary.
    checker = getattr(GL.glUseProgram, "error_checker", None)
    previous_checker = None
    if checker is not None and hasattr(checker, "_currentChecker"):
        previous_checker = checker._currentChecker
        checker._currentChecker = checker.nullGetError
    succeeded = False
    try:
        GL.glUseProgram(program_id)
        uniform_key = (program_id, "u_count")
        loc = _uniform_location_cache.get(uniform_key)
        if loc is None:
            loc = int(GL.glGetUniformLocation(program_id, "u_count"))
            _uniform_location_cache[uniform_key] = loc
        if loc != -1:
            GL.glUniform1ui(loc, plan.count)

        for binding, chunk in enumerate(chunks):
            GL.glBindBufferBase(
                GL.GL_SHADER_STORAGE_BUFFER, binding, chunk.buffer_id
            )
        for output_index, output in enumerate(outputs, len(chunks)):
            GL.glBindBufferBase(
                GL.GL_SHADER_STORAGE_BUFFER,
                output_index,
                output.buffer_id,
            )

        GL.glDispatchCompute(*plan.groups)
        # Without this the readback may observe stale memory. It is the GPU
        # analogue of the substrate-visibility problems in research/06.
        GL.glMemoryBarrier(
            GL.GL_SHADER_STORAGE_BARRIER_BIT
            | GL.GL_BUFFER_UPDATE_BARRIER_BIT
        )
        for output in outputs:
            output._mark_gpu_written()

        for binding in range(binding_count):
            GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, binding, 0)
        GL.glUseProgram(0)
        succeeded = True
    finally:
        if previous_checker is not None:
            checker._currentChecker = previous_checker
    if succeeded:
        error = int(GL.glGetError())
        if error != int(GL.GL_NO_ERROR):
            raise RuntimeError(
                f"OpenGL error 0x{error:04X} during compute dispatch"
            )


def _structural_chunks(values: Sequence[Any]) -> list[GLChunk]:
    if not values:
        raise ValueError("tensors list cannot be empty")
    chunks = [
        value if isinstance(value, GLChunk) else GLChunk.from_numpy(value)
        for value in values
    ]
    return chunks


def _structural_dtype(chunks: Sequence[GLChunk]) -> np.dtype:
    dtype = chunks[0].dtype
    for chunk in chunks[1:]:
        if chunk.dtype != dtype:
            dtype = _promote_dtype(dtype, chunk.dtype)
    return dtype


def arange_chunk(
    start: Any,
    end: Any,
    step: Any = 1,
    *,
    dtype: Any = None,
) -> GLChunk:
    """Create an arithmetic sequence directly in one resident output SSBO."""
    count = _arange_count(start, end, step)
    dtype = _arange_dtype(start, end, step, dtype)
    if count == 0:
        return GLChunk.from_numpy(np.empty((0,), dtype=dtype), dtype=dtype)

    out = GLChunk((count,), dtype=dtype)
    plan = plan_launch(count, binding_count=1)
    source = emit_arange_source(
        start,
        step,
        dtype=dtype,
        local_size=plan.local_size,
    )
    _dispatch(_compile(source), [], out, plan)
    return out


def expand_chunk(chunk: GLChunk, shape: Sequence[int]) -> GLChunk:
    """Materialize a broadcasted shape in one planned resident dispatch."""
    if not isinstance(chunk, GLChunk):
        chunk = GLChunk.from_numpy(chunk)
    source_shape, target_shape = _resolve_expand_shape(chunk.shape, shape)
    if _shape_product(target_shape) == 0:
        return GLChunk.from_numpy(
            np.empty(target_shape, dtype=chunk.dtype), dtype=chunk.dtype
        )

    out = GLChunk(target_shape, dtype=chunk.dtype)
    plan = plan_launch(out.count, binding_count=2)
    source = emit_expand_source(
        source_shape,
        target_shape,
        dtype=chunk.dtype,
        local_size=plan.local_size,
    )
    _dispatch(_compile(source), [chunk], out, plan)
    return out


def gather_offsets_chunk(
    chunk: GLChunk,
    offsets: Any,
    output_shape: Sequence[int],
) -> GLChunk:
    """Gather prevalidated row-major offsets without reading source data back."""
    if not isinstance(chunk, GLChunk):
        chunk = GLChunk.from_numpy(chunk)
    output_shape = tuple(int(size) for size in output_shape)
    output_count = _shape_product(output_shape)
    owns_offsets = not isinstance(offsets, GLChunk)
    if owns_offsets:
        flat_offsets = np.ascontiguousarray(
            np.asarray(offsets, dtype=np.int32).reshape(-1)
        )
        offset_chunk = GLChunk.from_numpy(flat_offsets, dtype=np.int32)
    else:
        offset_chunk = offsets
        if offset_chunk.dtype.kind != "i":
            raise TypeError("gather offsets must have signed integer dtype")
    if offset_chunk.count != output_count:
        raise ValueError(
            f"gather has {offset_chunk.count} offsets for {output_count} outputs"
        )
    if output_count == 0:
        if owns_offsets:
            offset_chunk.release()
        return GLChunk.from_numpy(
            np.empty(output_shape, dtype=chunk.dtype), dtype=chunk.dtype
        )
    out = GLChunk(output_shape, dtype=chunk.dtype)
    plan = plan_launch(output_count, binding_count=3)
    try:
        _dispatch(
            _compile(
                emit_gather_source(
                    dtype=chunk.dtype,
                    local_size=plan.local_size,
                )
            ),
            [chunk, offset_chunk],
            out,
            plan,
        )
    finally:
        if owns_offsets:
            offset_chunk.release()
    return out


def topk_chunks(
    chunk: GLChunk,
    k: int,
    dim: int = -1,
) -> tuple[GLChunk, GLChunk]:
    """Return resident top-k values and axis indices for arbitrary rank."""

    if not isinstance(chunk, GLChunk):
        chunk = GLChunk.from_numpy(chunk)
    if not chunk.shape:
        raise ValueError("topk requires a tensor with at least one dimension")
    dim = int(dim)
    if dim < 0:
        dim += chunk.ndim
    if dim < 0 or dim >= chunk.ndim:
        raise ValueError("topk dimension out of range")
    axis_size = chunk.shape[dim]
    k = int(k)
    if k < 0:
        raise ValueError("topk k cannot be negative")
    k = min(k, axis_size)
    output_shape = chunk.shape[:dim] + (k,) + chunk.shape[dim + 1:]
    if k == 0 or _shape_product(output_shape) == 0:
        return (
            GLChunk.from_numpy(
                np.empty(output_shape, dtype=chunk.dtype), dtype=chunk.dtype
            ),
            GLChunk.from_numpy(
                np.empty(output_shape, dtype=np.int32), dtype=np.int32
            ),
        )

    offsets = GLChunk(output_shape, dtype=np.int32)
    plan = plan_launch(offsets.count, binding_count=2)
    _dispatch(
        _compile(
            emit_topk_offsets_source(
                chunk.shape,
                k,
                dim,
                dtype=chunk.dtype,
                local_size=plan.local_size,
            )
        ),
        [chunk],
        offsets,
        plan,
    )
    values = gather_offsets_chunk(chunk, offsets, output_shape)
    inner = _shape_product(chunk.shape[dim + 1:])
    quotient = run_op("floordiv", offsets, inner)
    indices = run_op("mod", quotient, axis_size)
    return values, indices


def index_assign_offsets_chunk(
    chunk: GLChunk,
    offsets: Any,
    values: Any,
) -> GLChunk:
    """Assign prevalidated row-major offsets while keeping tensor data resident."""
    if not isinstance(chunk, GLChunk):
        raise TypeError("indexed assignment target must be a GLChunk")
    flat_offsets = np.ascontiguousarray(
        np.asarray(offsets, dtype=np.int32).reshape(-1)
    )
    if flat_offsets.size == 0:
        return chunk
    value_chunk = (
        values if isinstance(values, GLChunk) else GLChunk.from_numpy(values)
    )
    if value_chunk.dtype != chunk.dtype:
        raise TypeError(
            f"assignment dtype {value_chunk.dtype} does not match {chunk.dtype}"
        )
    if value_chunk.count not in (1, flat_offsets.size):
        raise ValueError(
            f"assignment value has {value_chunk.count} elements; selection "
            f"requires 1 or {flat_offsets.size}"
        )
    offset_chunk = GLChunk.from_numpy(flat_offsets, dtype=np.int32)
    plan = plan_launch(flat_offsets.size, binding_count=3)
    try:
        _dispatch(
            _compile(
                emit_index_assign_source(
                    dtype=chunk.dtype,
                    index_dtype=np.int32,
                    scalar_value=value_chunk.count == 1,
                    local_size=plan.local_size,
                )
            ),
            [offset_chunk, value_chunk],
            chunk,
            plan,
        )
    finally:
        offset_chunk.release()
    return chunk


def index_select_chunk(
    chunk: GLChunk,
    dim: int,
    indices: GLChunk,
) -> GLChunk:
    """Select a shaped resident integer index along one axis."""
    if not isinstance(chunk, GLChunk) or not isinstance(indices, GLChunk):
        raise TypeError("index_select requires GLChunk data and indices")
    if indices.dtype.kind not in {"i", "u"}:
        raise TypeError("advanced tensor indices must be integers")
    dim = int(dim)
    if dim < 0:
        dim += chunk.ndim
    if dim < 0 or dim >= chunk.ndim:
        raise IndexError("index axis out of range")
    output_shape = chunk.shape[:dim] + indices.shape + chunk.shape[dim + 1:]
    if indices.count == 0:
        return GLChunk.from_numpy(
            np.empty(output_shape, dtype=chunk.dtype), dtype=chunk.dtype
        )
    out = GLChunk(output_shape, dtype=chunk.dtype)
    plan = plan_launch(out.count, binding_count=3)
    _dispatch(
        _compile(
            emit_index_select_source(
                chunk.shape,
                dim,
                indices.count,
                dtype=chunk.dtype,
                index_dtype=indices.dtype,
                local_size=plan.local_size,
            )
        ),
        [chunk, indices],
        out,
        plan,
    )
    return out


def slice_axis_chunk(
    chunk: GLChunk,
    dim: int,
    start: int,
    step: int,
    count: int,
) -> GLChunk:
    """Select one affine slice while keeping source and result resident."""
    if not isinstance(chunk, GLChunk):
        raise TypeError("slice requires a GLChunk")
    dim = int(dim)
    if dim < 0:
        dim += chunk.ndim
    if dim < 0 or dim >= chunk.ndim:
        raise IndexError("slice axis out of range")
    output_shape = chunk.shape[:dim] + (int(count),) + chunk.shape[dim + 1:]
    if count == 0:
        return GLChunk.from_numpy(
            np.empty(output_shape, dtype=chunk.dtype), dtype=chunk.dtype
        )
    if start == 0 and step == 1 and count == chunk.shape[dim]:
        return chunk.view(output_shape)
    out = GLChunk(output_shape, dtype=chunk.dtype)
    plan = plan_launch(out.count, binding_count=2)
    _dispatch(
        _compile(
            emit_slice_axis_source(
                chunk.shape,
                dim,
                start,
                step,
                count,
                dtype=chunk.dtype,
                local_size=plan.local_size,
            )
        ),
        [chunk],
        out,
        plan,
    )
    return out


def index_assign_index_chunk(
    chunk: GLChunk,
    indices: GLChunk,
    values: Any,
) -> GLChunk:
    """Assign a resident integer index into a flat resident target."""
    if chunk.ndim != 1:
        raise ValueError("direct index assignment currently requires a flat target")
    if indices.dtype.kind not in {"i", "u"}:
        raise TypeError("advanced tensor indices must be integers")
    value_chunk = (
        values if isinstance(values, GLChunk) else GLChunk.from_numpy(values)
    )
    if value_chunk.dtype != chunk.dtype:
        raise TypeError(
            f"assignment dtype {value_chunk.dtype} does not match {chunk.dtype}"
        )
    if value_chunk.count not in (1, indices.count):
        raise ValueError(
            f"assignment value has {value_chunk.count} elements; selection "
            f"requires 1 or {indices.count}"
        )
    if indices.count == 0:
        return chunk
    plan = plan_launch(indices.count, binding_count=3)
    _dispatch(
        _compile(
            emit_index_assign_source(
                dtype=chunk.dtype,
                index_dtype=indices.dtype,
                scalar_value=value_chunk.count == 1,
                local_size=plan.local_size,
            )
        ),
        [indices, value_chunk],
        chunk,
        plan,
    )
    return chunk


def cat_chunks(chunks: Sequence[Any], dim: int = 0) -> GLChunk:
    """Concatenate resident chunks in one structural GPU dispatch."""
    require_gl_context()
    chunks = _structural_chunks(chunks)
    shapes, dim, output_shape = _validate_cat_layout(
        [chunk.shape for chunk in chunks], dim
    )
    output_dtype = _structural_dtype(chunks)
    max_inputs = _compute_limits().max_dispatch_ssbo_blocks - 1
    if len(chunks) > max_inputs:
        partials = [
            cat_chunks(chunks[start:start + max_inputs], dim)
            for start in range(0, len(chunks), max_inputs)
        ]
        try:
            return cat_chunks(partials, dim)
        finally:
            for partial in partials:
                partial.release()
    out = GLChunk(output_shape, dtype=output_dtype)
    if out.count == 0:
        return GLChunk.from_numpy(
            np.empty(output_shape, dtype=output_dtype),
            dtype=output_dtype,
        )
    plan = plan_launch(out.count, binding_count=len(chunks) + 1)
    source = emit_cat_source(
        shapes,
        dim,
        input_dtypes=[chunk.dtype for chunk in chunks],
        output_dtype=output_dtype,
        local_size=plan.local_size,
    )
    _dispatch(_compile(source), chunks, out, plan)
    return out


def reshape_chunk(chunk: GLChunk, shape: Sequence[int]) -> GLChunk:
    """Return a zero-copy resident view with a different logical shape."""
    if not isinstance(chunk, GLChunk):
        chunk = GLChunk.from_numpy(chunk)
    output_shape = _resolve_reshape_shape(shape, chunk.count)
    return chunk.view(output_shape)


def permute_chunk(chunk: GLChunk, dims: Sequence[int]) -> GLChunk:
    """Permute arbitrary-rank resident storage in one planned GPU dispatch."""
    if not isinstance(chunk, GLChunk):
        chunk = GLChunk.from_numpy(chunk)
    shape, dims, output_shape = _validate_permute_layout(chunk.shape, dims)
    if chunk.count == 0:
        return GLChunk.from_numpy(
            np.empty(output_shape, dtype=chunk.dtype), dtype=chunk.dtype
        )

    out = GLChunk(output_shape, dtype=chunk.dtype)
    plan = plan_launch(out.count, binding_count=2)
    source = emit_permute_source(
        shape,
        dims,
        dtype=chunk.dtype,
        local_size=plan.local_size,
    )
    _dispatch(_compile(source), [chunk], out, plan)
    return out


def matmul_chunks(
    left: Any,
    right: Any,
    *,
    reverse: bool = False,
) -> GLChunk:
    """Execute one native 2-D or broadcasted batched matrix multiplication."""
    left = left if isinstance(left, GLChunk) else GLChunk.from_numpy(left)
    right = right if isinstance(right, GLChunk) else GLChunk.from_numpy(right)
    if reverse:
        left, right = right, left
    left_shape, right_shape, _, output_shape = _matmul_layout(
        left.shape, right.shape
    )
    output_dtype = _promote_dtype(left.dtype, right.dtype)
    if _shape_product(output_shape) == 0:
        return GLChunk.from_numpy(
            np.empty(output_shape, dtype=output_dtype), dtype=output_dtype
        )

    out = GLChunk(output_shape, dtype=output_dtype)
    limits = _compute_limits()
    tile_cap = min(
        16,
        int(math.isqrt(limits.max_invocations)),
        int(math.isqrt(limits.max_group_size[0])),
    )
    tile = 1 << max(0, tile_cap.bit_length() - 1)
    thread_count = tile * tile
    rows, columns = output_shape[-2:]
    group_count = (
        _shape_product(output_shape[:-2])
        * ((rows + tile - 1) // tile)
        * ((columns + tile - 1) // tile)
    )
    plan = plan_launch(
        group_count * thread_count,
        preferred_local_size=thread_count,
        binding_count=3,
    )
    source = emit_matmul_source(
        left_shape,
        right_shape,
        left_dtype=left.dtype,
        right_dtype=right.dtype,
        output_dtype=output_dtype,
        local_size=thread_count,
    )
    _dispatch(_compile(source), [left, right], out, plan)
    return out


def repeat_chunk(
    chunk: GLChunk,
    repeats: Any,
    dim: int = 0,
) -> GLChunk:
    """Tile resident storage in one planned arbitrary-rank dispatch."""
    if not isinstance(chunk, GLChunk):
        chunk = GLChunk.from_numpy(chunk)
    source_shape, factors, output_shape = _resolve_repeat_layout(
        chunk.shape, repeats, dim
    )
    if _shape_product(output_shape) == 0:
        return GLChunk.from_numpy(
            np.empty(output_shape, dtype=chunk.dtype), dtype=chunk.dtype
        )

    out = GLChunk(output_shape, dtype=chunk.dtype)
    plan = plan_launch(out.count, binding_count=2)
    source = emit_repeat_source(
        source_shape,
        factors,
        dtype=chunk.dtype,
        local_size=plan.local_size,
    )
    _dispatch(_compile(source), [chunk], out, plan)
    return out


def reduce_chunk(
    chunk: GLChunk,
    op: str,
    dim: int | None = None,
    keepdim: bool = False,
) -> GLChunk:
    """Reduce one dimension directly into a resident output buffer."""
    if not isinstance(chunk, GLChunk):
        chunk = GLChunk.from_numpy(chunk)
    _, _, output_shape, _ = _reduce_layout(chunk.shape, dim, keepdim)
    output_dtype = _reduction_dtype(op, chunk.dtype)
    out = GLChunk(output_shape, dtype=output_dtype)
    plan = plan_launch(out.count, binding_count=2)
    source = emit_reduce_source(
        op,
        chunk.shape,
        dim,
        keepdim,
        dtype=chunk.dtype,
        local_size=plan.local_size,
    )
    _dispatch(_compile(source), [chunk], out, plan)
    return out


def cumsum_chunk(chunk: GLChunk, dim: int = 0) -> GLChunk:
    """Prefix-sum every line of one axis while preserving GPU residency."""
    if not isinstance(chunk, GLChunk):
        chunk = GLChunk.from_numpy(chunk)
    if not chunk.shape:
        raise ValueError("cumsum requires at least one dimension")
    dim = int(dim)
    if dim < 0:
        dim += len(chunk.shape)
    if dim < 0 or dim >= len(chunk.shape):
        raise ValueError("dim out of range")
    output_dtype = _reduction_dtype("sum", chunk.dtype)
    if chunk.count == 0:
        return GLChunk.from_numpy(
            np.empty(chunk.shape, dtype=output_dtype), dtype=output_dtype
        )

    out = GLChunk(chunk.shape, dtype=output_dtype)
    line_count = chunk.count // chunk.shape[dim]
    plan = plan_launch(line_count, binding_count=2)
    source = emit_cumsum_source(
        chunk.shape,
        dim,
        dtype=chunk.dtype,
        local_size=plan.local_size,
    )
    _dispatch(_compile(source), [chunk], out, plan)
    return out


def stack_chunks(chunks: Sequence[Any], dim: int = 0) -> GLChunk:
    """Stack equally-shaped resident chunks in one structural GPU dispatch."""
    require_gl_context()
    chunks = _structural_chunks(chunks)
    base_shape = chunks[0].shape
    if any(chunk.shape != base_shape for chunk in chunks[1:]):
        raise ValueError("All tensors must have the same shape")
    _, dim, output_shape = _validate_stack_layout(
        base_shape, len(chunks), dim
    )
    output_dtype = _structural_dtype(chunks)
    max_inputs = _compute_limits().max_dispatch_ssbo_blocks - 1
    if len(chunks) > max_inputs:
        partials = [
            stack_chunks(chunks[start:start + max_inputs], dim)
            for start in range(0, len(chunks), max_inputs)
        ]
        try:
            return cat_chunks(partials, dim)
        finally:
            for partial in partials:
                partial.release()
    out = GLChunk(output_shape, dtype=output_dtype)
    if out.count == 0:
        return GLChunk.from_numpy(
            np.empty(output_shape, dtype=output_dtype),
            dtype=output_dtype,
        )
    plan = plan_launch(out.count, binding_count=len(chunks) + 1)
    source = emit_stack_source(
        base_shape,
        len(chunks),
        dim,
        input_dtypes=[chunk.dtype for chunk in chunks],
        output_dtype=output_dtype,
        local_size=plan.local_size,
    )
    _dispatch(_compile(source), chunks, out, plan)
    return out


def execute_multi_output_program(
    program: FusedProgram,
    feeds: Mapping[int, Any] | Sequence[Any],
    *,
    outs: Mapping[str, GLChunk] | None = None,
) -> dict[str, GLChunk]:
    """Run every same-shape program output in one fused dispatch.

    Caller-owned ``outs`` keep all result buffers resident across repeated
    launches. Multi-output regions are important for backend graph planners:
    a shared producer can feed a renderer and several encoder planes without
    recomputation or an intermediate host boundary.
    """
    require_gl_context()
    feed_ids, output_items = _validate_program_outputs(program)
    if isinstance(feeds, Mapping):
        missing = set(feed_ids) - set(feeds)
        if missing:
            raise ValueError(f"missing FusedProgram feeds: {sorted(missing)}")
        feeds = [feeds[value_id] for value_id in feed_ids]
    if len(feeds) != len(feed_ids):
        raise ValueError(
            f"expected {len(feed_ids)} feeds, received {len(feeds)}"
        )
    if not feeds:
        raise ValueError("a fused program needs at least one feed")

    chunks = [f if isinstance(f, GLChunk) else GLChunk.from_numpy(f) for f in feeds]
    metadata = program.meta or {}
    for feed_id, chunk in zip(feed_ids, chunks):
        meta = metadata.get(feed_id)
        declared = getattr(meta, "dtype", None) if meta is not None else None
        expected_dtype = _normalize_dtype(declared or np.float32)
        if chunk.dtype != expected_dtype:
            raise TypeError(
                f"feed {feed_id} dtype must be {expected_dtype}, "
                f"got {chunk.dtype}"
            )
    linear_shape = getattr(program, "glsl_linear_output_shape", None)
    if linear_shape is not None:
        shape = tuple(int(size) for size in linear_shape)
        output_count = _shape_product(shape)
        runtime_feed_shapes = {}
        for feed_id, chunk in zip(feed_ids, chunks):
            if chunk.count == 1 and output_count != 1:
                runtime_feed_shapes[feed_id] = chunk.shape
            elif chunk.count == output_count:
                # A deferred reshape changes logical coordinates but not the
                # elementwise program's row-major lane correspondence.
                runtime_feed_shapes[feed_id] = shape
            else:
                raise ValueError(
                    f"linear fused feed {feed_id} has {chunk.count} elements; "
                    f"expected one or {output_count}"
                )
    else:
        shape: tuple[int, ...] = ()
        runtime_feed_shapes = {
            feed_id: chunk.shape
            for feed_id, chunk in zip(feed_ids, chunks)
        }
        for feed_shape in runtime_feed_shapes.values():
            shape = _broadcast_shape(shape, feed_shape)

    expected_names = tuple(name for name, _ in output_items)
    if outs is None:
        result = {}
    else:
        unknown = set(outs) - set(expected_names)
        missing = set(expected_names) - set(outs)
        if unknown or missing:
            raise ValueError(
                "caller-owned outputs must exactly match program outputs; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        result = dict(outs)
    for name, output_id in output_items:
        output_meta = metadata.get(output_id)
        output_dtype = _normalize_dtype(
            getattr(output_meta, "dtype", None) or np.float32
        )
        out = result.get(name)
        if out is None:
            out = GLChunk(shape, dtype=output_dtype)
            result[name] = out
        elif out.shape != shape:
            if len(output_items) == 1:
                raise ValueError(
                    "output must share the feed shape; "
                    f"got {out.shape} and {shape}"
                )
            raise ValueError(
                f"output {name!r} must have shape {shape}; got {out.shape}"
            )
        elif out.dtype != output_dtype:
            raise TypeError(
                f"output {name!r} dtype must be {output_dtype}, got {out.dtype}"
            )

    plan = plan_launch(
        _shape_product(shape),
        binding_count=len(chunks) + len(output_items),
    )
    source = emit_multi_output_program_source(
        program,
        local_size=plan.local_size,
        feed_shapes=runtime_feed_shapes,
        output_shape=shape,
    )
    ordered_outputs = [result[name] for name in expected_names]
    program_id = _compile(source)
    if len(ordered_outputs) == 1:
        # Preserve the long-standing single-output dispatch seam used by
        # instrumentation and hosts that wrap `_dispatch`.
        _dispatch(program_id, chunks, ordered_outputs[0], plan)
    else:
        _dispatch_many(program_id, chunks, ordered_outputs, plan)
    return result


def execute_program(
    program: FusedProgram,
    feeds: Mapping[int, Any] | Sequence[Any],
    *,
    out: GLChunk | None = None,
) -> GLChunk:
    """Run a single-output elementwise program as one fused dispatch."""
    _validate_program(program)
    output_name = next(iter(program.outputs))
    outs = None if out is None else {output_name: out}
    return execute_multi_output_program(program, feeds, outs=outs)[output_name]


def _defer_elementwise(
    name: str,
    *,
    reverse: bool,
    lhs: GLChunk | None,
    rhs: GLChunk | None,
    left_scalar: Any | None,
    right_scalar: Any | None,
    output_shape: tuple[int, ...],
    output_dtype: np.dtype,
) -> GLChunk | None:
    """Append one primitive to compatible deferred regions.

    Returns ``None`` when the merged region would exceed a hardware binding
    limit or the conservative shader-size ceiling; the caller then executes
    eagerly, which first materializes any deferred inputs.
    """

    steps: list[OpStep] = []
    seen_results: set[int] = set()
    feeds: dict[int, GLChunk] = {}
    metadata: dict[int, Meta] = {}

    def operand_id(chunk: GLChunk) -> int:
        deferred = chunk._deferred
        if deferred is None:
            value_id = id(chunk)
            feeds[value_id] = chunk
            metadata[value_id] = Meta(
                shape=chunk.shape,
                dtype=chunk.dtype.name,
                device="glsl",
            )
            return value_id

        feeds.update(deferred.feeds)
        if deferred.program.meta:
            metadata.update(deferred.program.meta)
        for step in deferred.program.steps:
            if step.result_id not in seen_results:
                steps.append(step)
                seen_results.add(step.result_id)
        return primary_output_id(deferred.program)

    attrs: dict[str, Any] = {}
    if name in _UNARY:
        assert lhs is not None
        input_ids = [operand_id(lhs)]
    elif lhs is None:
        assert rhs is not None and left_scalar is not None
        input_ids = [operand_id(rhs)]
        attrs["right_scalar"] = left_scalar
        if not reverse:
            attrs["reverse"] = True
    elif rhs is None:
        assert right_scalar is not None
        input_ids = [operand_id(lhs)]
        attrs["right_scalar"] = right_scalar
        if reverse:
            attrs["reverse"] = True
    else:
        input_ids = [operand_id(lhs), operand_id(rhs)]
        if reverse:
            attrs["reverse"] = True

    if len(feeds) + 1 > _compute_limits().max_dispatch_ssbo_blocks:
        return None
    if len(steps) >= 512:
        return None

    result_id = next(_deferred_value_ids)
    steps.append(
        OpStep(
            step_id=result_id,
            op_name=name,
            input_ids=input_ids,
            attrs=attrs,
            result_id=result_id,
        )
    )
    # Deferred result IDs come from one process-wide negative sequence, so they
    # also make collision-free step IDs when expression branches are merged.
    # Keeping the existing immutable step objects avoids an O(region²)
    # renumber/copy cycle while a long expression is being assembled.
    metadata[result_id] = Meta(
        shape=output_shape,
        dtype=output_dtype.name,
        device="glsl",
    )
    program = FusedProgram(
        version=1,
        feeds=set(feeds),
        steps=steps,
        outputs={"result": result_id},
        meta=metadata,
    )
    # Object IDs identify live chunks but their numeric sort order changes
    # between frames. Preserve expression traversal order so equivalent regions
    # emit byte-identical shader sources and hit the compilation cache.
    program.feed_order = tuple(feeds)
    program.glsl_linear_output_shape = output_shape
    out = GLChunk(output_shape, dtype=output_dtype)
    out._deferred = _DeferredElementwise(program, feeds)
    return out


def run_op(op: str, left: Any, right: Any = None) -> GLChunk:
    """Execute a single op -- the ``_apply_operator__`` analogue.

    Inputs may be resident chunks, host arrays, or one scalar. Non-scalar
    operands must already have the same shape; AbstractTensor owns any
    higher-level broadcasting policy.
    """
    require_gl_context()
    name, reverse = canonical_op(op)
    unary = name in _UNARY
    if unary and right is not None:
        raise ValueError(f"unary op {op!r} given a right operand")
    if not unary and right is None:
        raise ValueError(f"binary op {op!r} requires a right operand")

    left_is_scalar = np.isscalar(left)
    right_is_scalar = right is not None and np.isscalar(right)
    if left_is_scalar and (unary or right_is_scalar):
        raise TypeError("run_op requires at least one tensor operand")

    owned: list[GLChunk] = []
    lhs: GLChunk | None = None
    rhs: GLChunk | None = None
    try:
        if left_is_scalar:
            left_dtype = _normalize_dtype(np.asarray(left).dtype)
            left_shape: tuple[int, ...] = ()
        else:
            lhs = left if isinstance(left, GLChunk) else GLChunk.from_numpy(left)
            if not isinstance(left, GLChunk):
                owned.append(lhs)
            left_dtype = lhs.dtype
            left_shape = lhs.shape

        if unary:
            right_dtype = None
            right_shape = None
        elif right_is_scalar:
            right_dtype = _normalize_dtype(np.asarray(right).dtype)
            right_shape = ()
        else:
            rhs = right if isinstance(right, GLChunk) else GLChunk.from_numpy(right)
            if not isinstance(right, GLChunk):
                owned.append(rhs)
            right_dtype = rhs.dtype
            right_shape = rhs.shape

        if lhs is not None and rhs is not None:
            out_shape = _broadcast_shape(lhs.shape, rhs.shape)
        else:
            out_shape = lhs.shape if lhs is not None else rhs.shape
        out_dtype = _result_dtype(name, left_dtype, right_dtype)
        feeds = [chunk for chunk in (lhs, rhs) if chunk is not None]
        if _shape_product(out_shape) == 0:
            return GLChunk.from_numpy(
                np.empty(out_shape, dtype=out_dtype), dtype=out_dtype
            )
        can_fuse = (
            _fusion_depth.get() > 0
            and not owned
            and name in GLSL_OPS
            and all(
                chunk.shape == out_shape or chunk.count == 1
                for chunk in feeds
            )
        )
        if can_fuse:
            deferred = _defer_elementwise(
                name,
                reverse=reverse,
                lhs=lhs,
                rhs=rhs,
                left_scalar=left if left_is_scalar else None,
                right_scalar=right if right_is_scalar else None,
                output_shape=out_shape,
                output_dtype=out_dtype,
            )
            if deferred is not None:
                return deferred
        out = GLChunk(out_shape, dtype=out_dtype)
        plan = plan_launch(out.count, binding_count=len(feeds) + 1)
        source = _emit_primitive_source(
            name,
            left_dtype=left_dtype,
            right_dtype=right_dtype,
            out_dtype=out_dtype,
            left_shape=left_shape,
            right_shape=right_shape,
            out_shape=out_shape,
            left_scalar=left if left_is_scalar else None,
            right_scalar=right if right_is_scalar else None,
            reverse=reverse,
            local_size=plan.local_size,
        )
        _dispatch(_compile(source), feeds, out, plan)
        return out
    finally:
        for chunk in owned:
            chunk.release()
