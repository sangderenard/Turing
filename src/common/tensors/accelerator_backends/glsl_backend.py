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

import ast
import ctypes
from collections import Counter
import hashlib
import itertools
import json
import math
import re
import os
from pathlib import Path
import struct
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
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
    "emit_native_for_loop",
    "compose_control_shader",
    "ComposedGLSLControlArtifact",
    "InstalledGLSLControlShell",
    "build_control_shader_artifact",
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
    "emit_where_source",
    "compile_glsl_source",
    "run_op",
    "cat_chunks",
    "arange_chunk",
    "full_chunk",
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
    "where_chunks",
    "execute_program",
    "execute_multi_output_program",
    "execute_captured_fused_program",
    "compile_captured_fused_program",
    "dispatch_batch",
    "fuse_elementwise",
    "GLSL_OPS",
    "GLSL_FLOAT_SCALAR_OPS",
    "emit_scalar_expression",
    "shader_cache_stats",
    "dispatch_stats",
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

# WebGL 2 has the GLSL ES scalar math used by these entries, but it does not
# share the desktop compute backend's typed SSBO ABI.  Keep the reusable
# float-expression surface beside the authoritative expression tables; the
# WebGL fragment adapter consumes this view instead of copying the operators.
_NON_FLOAT_SCALAR_OPS = frozenset(
    {
        "bitand", "bitor", "bitxor", "shl", "shr", "invert",
        "int_trunc", "zext", "sext", "fptosi", "fptoui", "sitofp",
        "uitofp",
    }
)
GLSL_FLOAT_SCALAR_OPS: frozenset[str] = GLSL_OPS - _NON_FLOAT_SCALAR_OPS

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


def emit_scalar_expression(
    op: str,
    left: str,
    right: str | None = None,
    *,
    reverse: bool = False,
    out_type: str = "float",
) -> str:
    """Render one scalar expression from the canonical GLSL op tables.

    This is the deliberately small reuse seam for GLSL-family targets.  It
    exposes expression spelling, not the desktop compute shader's SSBO,
    dispatch, or context policy.
    """

    canonical, prefix_reverse = canonical_op(op)
    return _expr(
        canonical,
        left,
        right,
        bool(reverse) ^ prefix_reverse,
        out_type=out_type,
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


class _GLArena:
    """The one shader storage buffer that backs every resident value.

    A dataflow graph has no races, so a buffer per value buys nothing and
    spends the scarcest resource a compute shader has: SSBO binding points,
    sixteen on this device.  Every value is a slot in one arena, so a shader
    binds the arena once and reaches any value by index.

    Slots are four-byte words whatever the logical type -- ``_glsl_type``
    admits only ``float``, ``int`` and ``uint`` -- so one ``uint`` arena
    carries every value losslessly and callers reinterpret on access.  Freed
    slots return to a size-keyed free list; without one the arena could only
    grow, and pinned GL storage is what kills this process first.
    """

    __slots__ = (
        "buffer", "capacity", "high_water", "_free", "_alignment",
        "_execution_lock", "_executing",
    )

    def __init__(self) -> None:
        self.buffer: int | None = None
        self.capacity = 0
        self.high_water = 0
        self._free: dict[int, list[int]] = {}
        self._alignment = 0
        self._execution_lock = threading.Lock()
        self._executing = False

    def _slot_words(self) -> int:
        """Slot granularity, in words.

        While any emitter still declares its own storage block, slots are
        reached with ``glBindBufferRange`` and must satisfy the device's SSBO
        offset alignment.  Once every shader indexes the arena instead, this
        can drop to one.
        """

        if not self._alignment:
            try:
                self._alignment = max(
                    1, _compute_limits().ssbo_offset_alignment // 4
                )
            except Exception:
                self._alignment = 1
        return self._alignment

    def allocate(self, count: int) -> tuple[int, int]:
        if self._executing:
            raise RuntimeError(
                "GL arena allocation is forbidden during planned execution"
            )
        granularity = self._slot_words()
        reserved = max(1, int(count))
        reserved = ((reserved + granularity - 1) // granularity) * granularity
        pool = self._free.get(reserved)
        if pool:
            return pool.pop(), reserved
        offset = self.high_water
        self.high_water += reserved
        return offset, reserved

    def free(self, offset: int, reserved: int) -> None:
        self._free.setdefault(int(reserved), []).append(int(offset))

    def reserve(self) -> int:
        """Grow the device buffer so every allocated slot is addressable."""

        from OpenGL import GL

        if self.buffer is not None and self.capacity >= self.high_water:
            return self.buffer
        target = max(1 << 20, 1 << max(0, (self.high_water - 1).bit_length()))
        grown = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, grown)
        GL.glBufferData(
            GL.GL_SHADER_STORAGE_BUFFER, target * 4, None, GL.GL_DYNAMIC_DRAW
        )
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        if self.buffer is not None and self.capacity:
            GL.glBindBuffer(GL.GL_COPY_READ_BUFFER, self.buffer)
            GL.glBindBuffer(GL.GL_COPY_WRITE_BUFFER, grown)
            GL.glCopyBufferSubData(
                GL.GL_COPY_READ_BUFFER, GL.GL_COPY_WRITE_BUFFER,
                0, 0, self.capacity * 4,
            )
            GL.glBindBuffer(GL.GL_COPY_READ_BUFFER, 0)
            GL.glBindBuffer(GL.GL_COPY_WRITE_BUFFER, 0)
            GL.glDeleteBuffers(1, [self.buffer])
        self.buffer = grown
        self.capacity = target
        return self.buffer

    @contextmanager
    def execution(self):
        """Exclusively lease stable arena storage for one planned launch."""

        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError(
                "GL arena is already owned by another shell execution"
            )
        try:
            if self._executing:
                raise RuntimeError("recursive GL arena execution is forbidden")
            self.reserve()
            self._executing = True
            yield
        finally:
            self._executing = False
            self._execution_lock.release()


_ARENA = _GLArena()

ARENA_BINDING = 0

_ARENA_BLOCK = (
    f"layout(std430, binding = {ARENA_BINDING}) buffer Arena "
    "{ uint arena[]; };"
)


class _GLStorage:
    """Shared physical storage for one or more differently shaped GL views."""

    __slots__ = (
        "dtype",
        "capacity",
        "host",
        "buffer",
        "owns_buffer",
        "gpu_valid",
        "refs",
    )

    def __init__(
        self,
        dtype: Any,
        capacity: int,
        host: np.ndarray | None = None,
        *,
        buffer: int | None = None,
        owns_buffer: bool = False,
        gpu_valid: bool = False,
    ) -> None:
        self.dtype = _normalize_dtype(dtype)
        self.capacity = int(capacity)
        self.host = (
            None
            if host is None
            else np.ascontiguousarray(host, dtype=self.dtype).reshape(-1)
        )
        self.buffer = buffer
        self.owns_buffer = bool(owns_buffer)
        self.gpu_valid = bool(gpu_valid)
        self.refs = 1


class _ArenaAllocation:
    """Reference-count one allocation shared by zero-copy arena views.

    A view is not merely an offset: it is an owner of the allocation containing
    that offset.  In particular, expressions such as ``arange(8).unsqueeze(1)``
    immediately discard the temporary source chunk.  Returning the source slot
    to the arena at that point leaves the still-live view pointing into storage
    that the next tensor may overwrite.  Keep the allocation alive until the
    final base chunk, reshape, prefix, or range view releases it.
    """

    __slots__ = ("offset", "reserved", "refs", "released")

    def __init__(self, offset: int, reserved: int) -> None:
        self.offset = int(offset)
        self.reserved = int(reserved)
        self.refs = 1
        self.released = False

    def retain(self) -> "_ArenaAllocation":
        if self.released:
            raise RuntimeError("cannot retain a released GL arena allocation")
        self.refs += 1
        return self

    def release(self) -> None:
        if self.released:
            return
        self.refs -= 1
        if self.refs < 0:
            raise RuntimeError("GL arena allocation reference count underflow")
        if self.refs == 0:
            self.released = True
            _ARENA.free(self.offset, self.reserved)


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
        "_offset",
        "_reserved",
        "_dtype",
        "_host",
        "_gpu_valid",
        "_storage",
        "_allocation",
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
            host_array = np.ascontiguousarray(host_array).reshape(-1)
        self._dtype = logical_dtype
        self._host = host_array
        self._gpu_valid = False
        self._offset, self._reserved = _ARENA.allocate(self._count)
        self._allocation = _ArenaAllocation(self._offset, self._reserved)
        self._storage = None
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
        # A foreign buffer cannot live in the arena, so give the slot back and
        # keep a private storage record pointing at the host's own SSBO.
        chunk._allocation.release()
        chunk._allocation = None
        chunk._offset = 0
        chunk._reserved = 0
        chunk._storage = _GLStorage(
            dtype,
            chunk._count,
            buffer=int(buffer_id),
            owns_buffer=False,
            gpu_valid=True,
        )
        chunk._gpu_valid = True
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
        if self._deferred is not None and any(
            feed.count not in (1, self._count)
            for feed in self._deferred.feeds.values()
        ):
            # A row-major reshape is transparent only when every non-scalar
            # feed already spans the complete logical output. If the deferred
            # expression broadcasts (for example, (..., 8, 8) / (8, 8)),
            # changing the final coordinates before emission would reinterpret
            # that broadcast against the reshaped axes. Materialize the valid
            # broadcast region first, then keep the reshape itself zero-copy.
            self._to_gpu_current()
        if self._deferred is not None:
            deferred = self._deferred
            output_id = primary_output_id(deferred.program)
            metadata = dict(deferred.program.meta or {})
            for feed_id, feed in deferred.feeds.items():
                if feed.count != self._count:
                    continue
                feed_meta = metadata.get(feed_id)
                metadata[feed_id] = Meta(
                    shape=shape,
                    dtype=(
                        getattr(feed_meta, "dtype", None)
                        if feed_meta is not None
                        else feed.dtype.name
                    ),
                    device=(
                        getattr(feed_meta, "device", None)
                        if feed_meta is not None
                        else "glsl"
                    ),
                )
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
        self._to_gpu_current()
        chunk = object.__new__(type(self))
        chunk._shape = shape
        chunk._count = count
        chunk._offset = self._offset
        chunk._reserved = 0          # a borrowed slot: the source still owns it
        chunk._dtype = self._dtype
        chunk._host = None
        chunk._gpu_valid = True
        chunk._storage = self._storage
        chunk._allocation = (
            self._allocation.retain()
            if self._allocation is not None
            else None
        )
        chunk._deferred = None
        chunk._released = False
        return chunk

    def prefix_view(self, shape: Sequence[int]) -> "GLChunk":
        """Return a zero-copy contiguous prefix with a smaller logical count.

        This is valid only for a row-major prefix beginning at element zero.
        It exists primarily for first-axis slicing, where dispatching a copy
        shader would add synchronization without changing a single address.
        """

        if self._released:
            raise RuntimeError("cannot view a released GLChunk")
        shape = tuple(int(d) for d in shape)
        count = int(np.prod(shape)) if shape else 1
        if count < 0 or count > self._count:
            raise ValueError(
                f"prefix shape {shape} has {count} elements, but source has "
                f"{self._count}"
            )
        # A deferred expression must first produce the complete source buffer;
        # the prefix then aliases that resident storage.
        self._to_gpu_current()
        chunk = object.__new__(type(self))
        chunk._shape = shape
        chunk._count = count
        chunk._offset = self._offset
        chunk._reserved = 0          # a borrowed slot: the source still owns it
        chunk._dtype = self._dtype
        chunk._host = None
        chunk._gpu_valid = True
        chunk._storage = self._storage
        chunk._allocation = (
            self._allocation.retain()
            if self._allocation is not None
            else None
        )
        chunk._deferred = None
        chunk._released = False
        return chunk

    def range_view(
        self,
        shape: Sequence[int],
        *,
        offset: int,
    ) -> "GLChunk":
        """Return an aligned zero-copy contiguous subrange of this chunk."""

        if self._released:
            raise RuntimeError("cannot view a released GLChunk")
        shape = tuple(int(d) for d in shape)
        count = int(np.prod(shape)) if shape else 1
        offset = int(offset)
        if offset < 0 or count < 0 or offset + count > self._count:
            raise ValueError(
                f"range [{offset}, {offset + count}) is outside "
                f"{self._count} source elements"
            )
        byte_offset = (self._offset + offset) * 4
        alignment = _compute_limits().ssbo_offset_alignment
        if byte_offset % alignment:
            raise ValueError(
                f"range byte offset {byte_offset} is not aligned to "
                f"the device SSBO requirement {alignment}"
            )
        self._to_gpu_current()
        chunk = object.__new__(type(self))
        chunk._shape = shape
        chunk._count = count
        chunk._offset = self._offset + offset
        chunk._reserved = 0          # a borrowed slot: the source still owns it
        chunk._dtype = self._dtype
        chunk._host = None
        chunk._gpu_valid = True
        chunk._storage = self._storage
        chunk._allocation = (
            self._allocation.retain()
            if self._allocation is not None
            else None
        )
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
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self._count * 4

    @property
    def on_cpu(self) -> bool:
        return not self._released and self._host is not None

    @property
    def on_gpu(self) -> bool:
        return (
            not self._released
            and _ARENA.buffer is not None
            and self._gpu_valid
        )

    @property
    def buffer_id(self) -> int | None:
        if not self._released and self._deferred is not None:
            self.to_gpu()
        if self._released:
            return None
        if self._storage is not None:
            return self._storage.buffer
        return _ARENA.buffer

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

        buffer = _ARENA.reserve()
        if not self._gpu_valid:
            if self._host is None:
                # Allocated but never written and nothing to upload: it is an
                # output slot. Leave contents undefined but mark it live.
                self._gpu_valid = True
                return self
            data = np.ascontiguousarray(
                self._host, dtype=_storage_dtype(self._dtype)
            )
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, buffer)
            GL.glBufferSubData(
                GL.GL_SHADER_STORAGE_BUFFER,
                self._offset * 4,
                self.nbytes,
                data,
            )
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            self._gpu_valid = True
        return self

    def update_numpy(self, array: Any) -> "GLChunk":
        """Replace host contents while preserving an allocated GPU buffer.

        The shape may not change.  The next :meth:`to_gpu` uploads the new
        contents into the existing SSBO instead of allocating another buffer.
        """
        if self._released:
            raise RuntimeError("cannot update a released GLChunk")
        if self._reserved == 0 and self._allocation is not None:
            raise RuntimeError(
                "cannot replace a partial GLChunk view; update its owning "
                "allocation or use an explicit resident-range upload"
            )
        self._deferred = None
        data = np.ascontiguousarray(np.asarray(array, dtype=self.dtype))
        if data.shape != self._shape:
            raise ValueError(
                f"updated data must keep shape {self._shape}, got {data.shape}"
            )
        self._host = data.reshape(-1)
        self._gpu_valid = False
        return self

    def upload_numpy(self, array: Any) -> "GLChunk":
        """Upload exactly this logical view into its resident buffer range."""

        if self._released:
            raise RuntimeError("cannot upload into a released GLChunk")
        require_gl_context()
        from OpenGL import GL

        data = np.ascontiguousarray(
            np.asarray(array, dtype=_storage_dtype(self.dtype))
        )
        if data.shape != self._shape:
            raise ValueError(
                f"uploaded data must keep shape {self._shape}, got {data.shape}"
            )
        self._to_gpu_current()
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, _ARENA.buffer)
        GL.glBufferSubData(
            GL.GL_SHADER_STORAGE_BUFFER,
            self._offset * 4,
            self.nbytes,
            data,
        )
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        self._host = None
        self._gpu_valid = True
        return self

    def discard_host(self) -> "GLChunk":
        """Drop a staging/readback copy while preserving the live SSBO."""
        if not self.on_gpu:
            raise RuntimeError("cannot discard the only valid GLChunk storage")
        self._host = None
        return self

    def _mark_gpu_written(self) -> None:
        """Mark host contents stale after a shader writes the SSBO."""
        if self._released:
            raise RuntimeError("cannot mark a released GLChunk")
        self._deferred = None
        self._host = None
        self._gpu_valid = True

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
        if self._host is not None:
            return self._host[:self._count].reshape(self._shape)
        if not self.on_gpu:
            raise RuntimeError("chunk has no CPU data and no live GPU buffer")
        require_gl_context()
        from OpenGL import GL

        out = np.empty(self._count, dtype=_storage_dtype(self.dtype))
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.buffer_id)
        GL.glGetBufferSubData(
            GL.GL_SHADER_STORAGE_BUFFER, self._offset * 4, self.nbytes,
            out.ctypes.data_as(ctypes.c_void_p),
        )
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        return out.astype(self.dtype, copy=False).reshape(self._shape)

    def numpy(self) -> np.ndarray:
        """Host view, reading back from the GPU when that is the live copy."""
        return self.to_cpu()

    def release(self) -> None:
        """Return this value's arena slot; the arena buffer itself persists."""
        if self._released:
            return
        self._released = True
        self._deferred = None
        self._host = None
        self._gpu_valid = False
        if self._allocation is not None:
            self._allocation.release()
            self._allocation = None

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

_WORKGROUP_CONTROL_SHADER_HEADER = """#version 430
// GENERATED by turing glsl_backend control lowering -- do not edit by hand.
//
// One planner-proved independent loop iteration is assigned to each
// workgroup.  Numerical lanes are local to that iteration/frame.
layout(local_size_x = {local_size}) in;

uint turing_linear_gid() {{
    return gl_LocalInvocationID.x;
}}
"""


@dataclass(frozen=True)
class ShaderSnippet:
    """One operation's body lines, not a shader.

    An operation contributes statements, the arena slots it reads and writes,
    and any helper functions its expressions call.  It does not carry a
    version directive, a storage declaration, or a ``main``: those exist once
    per compiled program, and a program is only finished when a shell control
    interruption actually forces a dispatch.  Until then operations keep
    accumulating into the same body.
    """

    lines: tuple[str, ...]
    slots: int
    helpers: tuple[str, ...] = ()
    shared: tuple[str, ...] = ()
    guard: bool = True

    def guarded(
        self,
        index: int,
        *,
        device_resident: bool = False,
    ) -> tuple[str, ...]:
        """Bracket the body so this operation runs over its own extent.

        A snippet that does its own launch-geometry arithmetic -- a tiled
        kernel indexing by workgroup rather than by flat output element --
        sets ``guard`` false and is emitted in a bare scope instead.
        """

        if not self.guard:
            return ("    {", *("        " + line for line in self.lines), "    }")
        if device_resident:
            return (
                "    {",
                "        for (uint gid = gl_LocalInvocationID.x; "
                f"gid < u_extent[{index}]; "
                "gid += gl_WorkGroupSize.x) {",
                *("            " + line for line in self.lines),
                "        }",
                "    }",
            )
        return (
            "    {",
            "        uint gid = turing_linear_gid();",
            f"        if (gid < u_extent[{index}]) {{",
            *("            " + line for line in self.lines),
            "        }",
            "    }",
        )

    def rebased(self, base: int) -> "ShaderSnippet":
        """Relocate this fragment's local arena slots into a larger program."""

        offset = int(base)
        if offset == 0:
            return self

        def relocate(line: str) -> str:
            return re.sub(
                r"u_slot\[(\d+)\]",
                lambda match: f"u_slot[{int(match.group(1)) + offset}]",
                line,
            )

        return ShaderSnippet(
            lines=tuple(relocate(line) for line in self.lines),
            slots=self.slots,
            helpers=self.helpers,
            shared=self.shared,
            guard=self.guard,
        )


@dataclass(frozen=True)
class ComposedGLSLControlArtifact:
    """One fully selected GLSL shell before device installation."""

    source: str
    slot_value_ids: tuple[int, ...]
    extents: tuple[int, ...]
    slot_extents: tuple[int, ...]
    value_meta: Mapping[int, Meta]
    external_value_ids: tuple[int, ...]
    terminal_outputs: Mapping[str, int]
    uniform_value_ids: Mapping[str, int]
    value_aliases: Mapping[int, int]
    contiguous_plan: Any = None
    phase_sources: tuple[str, ...] = ()
    specialized_values: Mapping[int, Any] = field(default_factory=dict)
    instrumentation: bool = False
    debug_capacity: int = 65536
    device_resident: bool = False
    local_size: int = _LOCAL_SIZE
    stream_publications: tuple[Any, ...] = ()
    stream_outputs: Mapping[str, int] = field(default_factory=dict)
    stream_continuation_count: int = 0
    slot_contract_diagnostics: Mapping[int, Any] = field(
        default_factory=dict
    )
    snippet_diagnostics: tuple[Mapping[str, Any], ...] = ()
    stream_word_capacity: int = 1 << 24
    stream_descriptor_capacity: int = 4096
    phase_cache_identities: tuple[str, ...] = ()
    workgroup_loop_bounds: tuple[str, str, str] | None = None
    c_dispatch_loop_bounds: tuple[str, str, str] | None = None
    private_value_capacities: Mapping[int, int] = field(
        default_factory=dict
    )


def _control_stream_publications(root: Any) -> tuple[Any, ...]:
    from src.compiler.control_source import (
        CallBlock,
        LoopBlock,
        ParallelDeployment,
        SequenceBlock,
        StateMachineTick,
        StreamPublishBlock,
    )

    if isinstance(root, StreamPublishBlock):
        return (root,)
    children = ()
    if isinstance(root, SequenceBlock):
        children = root.blocks
    elif isinstance(root, LoopBlock):
        children = (root.body,)
    elif isinstance(root, StateMachineTick):
        children = tuple(body for _case, body in root.cases)
    elif isinstance(root, ParallelDeployment):
        children = root.lanes
    elif isinstance(root, CallBlock):
        children = (root.callee,)
    return tuple(
        publication
        for child in children
        for publication in _control_stream_publications(child)
    )


def _selected_workgroup_loop(root: Any):
    """Return the outermost planner-approved loop for the dispatch x axis."""

    from src.compiler.control_source import (
        CallBlock,
        LoopBlock,
        ParallelDeployment,
        SequenceBlock,
        StateMachineTick,
    )

    if isinstance(root, LoopBlock):
        if root.parallel_iterations:
            return root
        return _selected_workgroup_loop(root.body)
    if isinstance(root, SequenceBlock):
        children = root.blocks
    elif isinstance(root, StateMachineTick):
        children = tuple(body for _case, body in root.cases)
    elif isinstance(root, ParallelDeployment):
        children = root.lanes
    elif isinstance(root, CallBlock):
        children = (root.callee,)
    else:
        children = ()
    for child in children:
        selected = _selected_workgroup_loop(child)
        if selected is not None:
            return selected
    return None


def _selected_c_dispatch_loop(root: Any):
    """Return the outermost loop dissolved into C-planned dispatches."""

    from src.compiler.control_source import (
        CallBlock,
        LoopBlock,
        ParallelDeployment,
        SequenceBlock,
        StateMachineTick,
    )

    def children(block):
        if isinstance(block, LoopBlock):
            return (block.body,)
        if isinstance(block, SequenceBlock):
            return block.blocks
        if isinstance(block, StateMachineTick):
            return tuple(body for _case, body in block.cases)
        if isinstance(block, ParallelDeployment):
            return block.lanes
        if isinstance(block, CallBlock):
            return (block.callee,)
        return ()

    candidates = []

    def gather(block):
        if isinstance(block, LoopBlock) and block.dispatch_shell == "c":
            candidates.append(block)
        for child in children(block):
            gather(child)

    def contains(block, target) -> bool:
        return block is target or any(
            contains(child, target) for child in children(block)
        )

    gather(root)
    if not candidates:
        return None
    outer = candidates[0]
    # One C command stream can dissolve nested loops into the selected
    # closure, but cannot represent two disjoint control segments without
    # separate shader variants.  Keep disjoint loops native until that richer
    # schedule is explicitly constructed.
    if all(contains(outer.body, candidate) for candidate in candidates[1:]):
        return outer
    return None


def _control_scheduled_regions(root: Any) -> tuple[int, ...]:
    from src.compiler.control_source import (
        CallBlock,
        LoopBlock,
        ParallelDeployment,
        SequenceBlock,
        StateMachineTick,
        StatementBlock,
    )

    if isinstance(root, StatementBlock):
        if (
            len(root.lines) == 1
            and root.lines[0].startswith("__scheduled_region_")
            and root.lines[0].endswith("__")
        ):
            return (
                int(root.lines[0][len("__scheduled_region_"):-2]),
            )
        return ()
    if isinstance(root, SequenceBlock):
        children = root.blocks
    elif isinstance(root, LoopBlock):
        children = (root.body,)
    elif isinstance(root, StateMachineTick):
        children = tuple(body for _case, body in root.cases)
    elif isinstance(root, ParallelDeployment):
        children = root.lanes
    elif isinstance(root, CallBlock):
        children = (root.callee,)
    else:
        children = ()
    return tuple(
        region
        for child in children
        for region in _control_scheduled_regions(child)
    )


def _direct_stream_publications(root: Any) -> tuple[Any, ...]:
    """Return publications in one lexical execution scope.

    Sequence and absorbed-call blocks remain in the same scope.  A nested
    loop/state-machine/parallel lane owns its own continuation and is therefore
    deliberately not traversed here.
    """

    from src.compiler.control_source import (
        CallBlock,
        SequenceBlock,
        StreamPublishBlock,
    )

    if isinstance(root, StreamPublishBlock):
        return (root,)
    if isinstance(root, SequenceBlock):
        children = root.blocks
    elif isinstance(root, CallBlock):
        children = (root.callee,)
    else:
        return ()
    return tuple(
        publication
        for child in children
        for publication in _direct_stream_publications(child)
    )


def _stream_continuation_count(root: Any, *, _root: bool = True) -> int:
    """Count planner scopes which need persistent resident continuation."""

    from src.compiler.control_source import (
        CallBlock,
        LoopBlock,
        ParallelDeployment,
        SequenceBlock,
        StateMachineTick,
    )

    count = int(
        _root
        and not isinstance(root, LoopBlock)
        and bool(_direct_stream_publications(root))
    )
    if isinstance(root, LoopBlock):
        count += int(bool(_direct_stream_publications(root.body)))
        children = (root.body,)
    elif isinstance(root, SequenceBlock):
        children = root.blocks
    elif isinstance(root, StateMachineTick):
        children = tuple(body for _case, body in root.cases)
    elif isinstance(root, ParallelDeployment):
        children = root.lanes
    elif isinstance(root, CallBlock):
        children = (root.callee,)
    else:
        children = ()
    return count + sum(
        _stream_continuation_count(child, _root=False)
        for child in children
    )


def compose_shader(
    snippets: Sequence[ShaderSnippet],
    *,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Assemble accumulated operation bodies into one compilable program.

    This is the only place a shader is finished.  Everything upstream composes
    snippets; nothing upstream emits a header, a storage block, or a ``main``.
    """

    snippets = tuple(snippets)
    total_slots = sum(snippet.slots for snippet in snippets)
    helpers: list[str] = []
    shared: list[str] = []
    for snippet in snippets:
        for helper in snippet.helpers:
            if helper and helper not in helpers:
                helpers.append(helper)
        for declaration in snippet.shared:
            if declaration and declaration not in shared:
                shared.append(declaration)
    body: list[str] = []
    for index, snippet in enumerate(snippets):
        if index:
            # Later operations may read what earlier ones wrote.
            body.append("    barrier();")
            body.append("    memoryBarrierBuffer();")
        body.extend(snippet.guarded(index))
    return "\n".join(
        [
            _SHADER_HEADER.format(local_size=local_size),
            _ARENA_BLOCK,
            "",
            "uniform uint u_count;",
            f"uniform uint u_extent[{max(1, len(snippets))}];",
            f"uniform uint u_slot[{max(1, total_slots)}];",
            *shared,
            "",
            *([*helpers, ""] if helpers else []),
            "void main() {",
            *body,
            "}",
            "",
        ]
    ) + "\n"


def compose_control_shader(
    control_program,
    captured_regions: Mapping[int, Any],
    *,
    local_size: int = _LOCAL_SIZE,
    instrumentation: bool = False,
    active_program_indices: Iterable[int] | None = None,
    emit_validations: bool = True,
    device_resident: bool = False,
) -> str:
    """Absorb scheduled elementwise regions into one compiled control shader.

    ``control_program`` is planner-owned structure from ``control_source``.
    Region markers are substituted exactly once and in the order selected by
    that plan.  This function never discovers control flow or computes a new
    schedule.
    """

    from src.compiler.control_source import (
        CallBlock,
        ControlTarget,
        LoopBlock,
        ParallelDeployment,
        SequenceBlock,
        StateMachineTick,
        StatementBlock,
        StreamPublishBlock,
        ValidationBlock,
        render_control_block,
    )

    def contains_validation(block) -> bool:
        if isinstance(block, ValidationBlock):
            return True
        if isinstance(block, SequenceBlock):
            return any(contains_validation(child) for child in block.blocks)
        if isinstance(block, LoopBlock):
            return contains_validation(block.body)
        if isinstance(block, StateMachineTick):
            return any(
                contains_validation(body) for _value, body in block.cases
            )
        if isinstance(block, ParallelDeployment):
            return any(contains_validation(lane) for lane in block.lanes)
        if isinstance(block, CallBlock):
            return contains_validation(block.callee)
        if isinstance(block, StreamPublishBlock):
            return False
        return False

    stream_publications = _control_stream_publications(
        control_program.root
    )
    selected_workgroup_loop = _selected_workgroup_loop(
        control_program.root
    )
    selected_workgroup_induction = (
        None
        if selected_workgroup_loop is None
        else str(selected_workgroup_loop.induction)
    )
    selected_c_dispatch_loop = _selected_c_dispatch_loop(
        control_program.root
    )
    selected_c_dispatch_induction = (
        None
        if selected_c_dispatch_loop is None
        else str(selected_c_dispatch_loop.induction)
    )

    trace_instrumentation = bool(instrumentation)
    debug_enabled = bool(
        trace_instrumentation
        or contains_validation(control_program.root)
    )

    expected = tuple(control_program.region_indices)
    if set(captured_regions) != set(expected):
        raise ValueError(
            "control shader regions do not match the planner submission: "
            f"expected={expected!r}, supplied={tuple(captured_regions)!r}"
        )

    active_program_indices = (
        None
        if active_program_indices is None
        else frozenset(int(index) for index in active_program_indices)
    )
    region_snippets: dict[
        int, tuple[tuple[int, int, ShaderSnippet], ...]
    ] = {}
    region_slot_ids: dict[int, tuple[int, ...]] = {}
    all_slot_ids: list[int] = []
    all_slot_meta: list[Meta | None] = []
    static_scalar_values: dict[int, Any] = {}
    scalar_trace_slots: dict[int, tuple[int, ...]] = {}
    closure_source_trace_indices: set[int] = set()
    closure_scalar_source_producers: dict[int, tuple[int, int]] = {}
    closure_source_ids = {
        int(source_id)
        for _iterable_id, _target_id, _induction, source_ids
        in control_program.closure_iterable_bindings
        for source_id in source_ids
    }
    base = 0
    snippet_index = 0
    program_index = 0
    for region_index in expected:
        captured = captured_regions[region_index]
        stages = tuple(getattr(captured, "stages", ()) or ())
        parts = (
            tuple(type(captured)(stage, {}) for stage in stages)
            if stages
            else (captured,)
        )
        lowered = []
        region_values = []
        for part in parts:
            part_base = base
            current_snippet_index = snippet_index
            try:
                snippet = captured_program_snippet(
                    part,
                    base=base,
                    local_size=local_size,
                )
            except Exception as error:
                fragment_error = RuntimeError(
                    "failed to emit captured control-shader fragment: "
                    f"region={region_index}, stage={snippet_index}, "
                    f"feeds={tuple(ordered_feed_ids(part.program))!r}, "
                    f"outputs={tuple(part.program.outputs.items())!r}, "
                    f"steps={tuple((step.op_name, tuple(step.input_ids), int(step.result_id)) for step in part.program.steps)!r}, "
                    f"shapes={{{', '.join(f'{int(value_id)}: {tuple(meta.shape or ())!r}' for value_id, meta in (part.program.meta or {}).items())}}}"
                )
                fragment_error.region_index = int(region_index)
                fragment_error.program = part.program
                raise fragment_error from error
            lowered.append((snippet_index, program_index, snippet))
            snippet_index += 1
            program_index += 1
            base += snippet.slots
            local_values = (
                *ordered_feed_ids(part.program),
                *tuple(part.program.outputs.values()),
            )
            # Native snippets consume operands in the operation's positional
            # ABI.  A FusedProgram's ``feeds`` is a set and cannot preserve
            # either that order or repetition such as stack(a, b, b).
            # Elementwise snippets are different: program_snippet reads the
            # canonical ordered feed set, so forcing positional order there
            # swaps slots whenever value IDs sort differently.
            kernel_kind = (part.program.extras or {}).get("kernel_kind")
            if (
                len(part.program.steps) == 1
                and kernel_kind not in {None, "linear_reshape_copy"}
            ):
                positional_values = (
                    *tuple(part.program.steps[0].input_ids),
                    *tuple(part.program.outputs.values()),
                )
                if len(positional_values) == snippet.slots:
                    local_values = positional_values
            if len(local_values) != snippet.slots:
                raise ValueError(
                    "captured fragment slot count does not match its value "
                    f"ABI: region={region_index}, stage={snippet_index - 1}, "
                    f"snippet_slots={snippet.slots}, "
                    f"feed_ids={ordered_feed_ids(part.program)!r}, "
                    f"outputs={tuple(part.program.outputs.items())!r}, "
                    "steps="
                    f"{tuple((step.op_name, tuple(step.input_ids), step.result_id) for step in part.program.steps)!r}"
                )
            kernel_kind = (part.program.extras or {}).get("kernel_kind")
            if (
                len(part.program.steps) == 1
                and kernel_kind not in {None, "linear_reshape_copy"}
            ):
                step = part.program.steps[0]
                values = tuple(step.attrs.get("values", ()))
                if (
                    step.op_name == "tensor_from_list"
                    and len(values) == 1
                ):
                    static_scalar_values[int(step.result_id)] = values[0]
            local_meta = tuple(
                (part.program.meta or {}).get(int(value_id))
                for value_id in local_values
            )
            output_metas = tuple(
                (part.program.meta or {}).get(int(value_id))
                for value_id in part.program.outputs.values()
            )
            scalar_output_slots = tuple(
                part_base + len(local_values) - len(output_metas) + offset
                for offset, meta in enumerate(output_metas)
                if (
                    meta is not None
                    and meta.shape is not None
                    and _shape_product(tuple(meta.shape or ())) == 1
                )
            )
            if scalar_output_slots:
                scalar_trace_slots[current_snippet_index] = (
                    scalar_output_slots
                )
                output_ids = tuple(
                    map(int, part.program.outputs.values())
                )
                if closure_source_ids.intersection(output_ids):
                    closure_source_trace_indices.add(
                        current_snippet_index
                    )
                    for output_id, output_slot, meta in zip(
                        output_ids,
                        range(
                            part_base + len(local_values) - len(output_metas),
                            part_base + len(local_values),
                        ),
                        output_metas,
                    ):
                        if (
                            output_id in closure_source_ids
                            and meta is not None
                            and meta.shape is not None
                            and _shape_product(
                                tuple(meta.shape or ())
                            ) == 1
                        ):
                            closure_scalar_source_producers[output_id] = (
                                int(current_snippet_index),
                                int(output_slot),
                            )
            region_values.extend(int(value_id) for value_id in local_values)
            all_slot_ids.extend(int(value_id) for value_id in local_values)
            all_slot_meta.extend(local_meta)
        region_snippets[region_index] = tuple(lowered)
        region_slot_ids[region_index] = tuple(region_values)

    # Iterable control operands are part of the shell ABI even when no
    # numerical region consumes the iterable as an ordinary tensor input.
    # Give each one a resident arena slot so the shader can bind the current
    # element directly, without a host-side assignment per iteration.
    for iterable_id, _target_id, _induction in (
        control_program.iterable_bindings
    ):
        if int(iterable_id) not in all_slot_ids:
            all_slot_ids.append(int(iterable_id))
            all_slot_meta.append(None)
            base += 1
    for _iterable_id, target_id, _induction, values in (
        control_program.static_iterable_bindings
    ):
        if int(target_id) in all_slot_ids:
            continue
        values = tuple(values)
        dtype = (
            "float32"
            if any(isinstance(value, float) for value in values)
            else "bool"
            if values and all(isinstance(value, bool) for value in values)
            else "int32"
        )
        all_slot_ids.append(int(target_id))
        all_slot_meta.append(Meta(shape=(1,), dtype=dtype, device="glsl"))
        base += 1
    for _source_id, collection_id, _induction, _start in (
        control_program.collection_bindings
    ):
        if int(collection_id) in all_slot_ids:
            continue
        all_slot_ids.append(int(collection_id))
        collection_meta = next(
            (
                (program.meta or {}).get(int(collection_id))
                for captured in captured_regions.values()
                for program in (
                    tuple(getattr(captured, "stages", ()) or ())
                    or (captured.program,)
                )
                if (program.meta or {}).get(int(collection_id)) is not None
            ),
            None,
        )
        all_slot_meta.append(collection_meta)
        base += 1
    for _iterable_id, target_id, _induction, source_ids in (
        control_program.closure_iterable_bindings
    ):
        for value_id in (int(target_id), *map(int, source_ids)):
            if value_id in all_slot_ids:
                continue
            all_slot_ids.append(value_id)
            all_slot_meta.append(None)
            base += 1
    # Loop-carried endpoints are executable data dependencies even when one
    # endpoint is hidden behind a structural LoopExit and therefore is not a
    # public numerical terminal.  Control composition addresses both ranges
    # for the state commit, so both identities belong to the slot ABI.
    for updated_id, initial_id in control_program.value_aliases:
        for value_id in (int(updated_id), int(initial_id)):
            if value_id in all_slot_ids:
                continue
            all_slot_ids.append(value_id)
            all_slot_meta.append(None)
            base += 1
    for publication in stream_publications:
        for value_id in (
            publication.value_id,
            publication.count_value_id,
            publication.predicate_value_id,
        ):
            if value_id is None or int(value_id) in all_slot_ids:
                continue
            all_slot_ids.append(int(value_id))
            all_slot_meta.append(None)
            base += 1

    def scheduled_regions(block) -> tuple[int, ...]:
        if isinstance(block, StatementBlock):
            if (
                len(block.lines) == 1
                and block.lines[0].startswith("__scheduled_region_")
                and block.lines[0].endswith("__")
            ):
                return (int(block.lines[0][len("__scheduled_region_"):-2]),)
            return ()
        if isinstance(block, SequenceBlock):
            children = block.blocks
        elif isinstance(block, LoopBlock):
            children = (block.body,)
        elif isinstance(block, StateMachineTick):
            children = tuple(body for _case, body in block.cases)
        elif isinstance(block, ParallelDeployment):
            children = block.lanes
        elif isinstance(block, CallBlock):
            children = (block.callee,)
        else:
            children = ()
        return tuple(
            region
            for child in children
            for region in scheduled_regions(child)
        )

    parallel_private_value_ids: set[int] = set()
    if selected_workgroup_loop is not None:
        for region_index in scheduled_regions(
            selected_workgroup_loop.body
        ):
            captured = captured_regions[int(region_index)]
            stages = tuple(getattr(captured, "stages", ()) or ())
            programs = (
                stages if stages else (captured.program,)
            )
            parallel_private_value_ids.update(
                int(value_id)
                for program in programs
                for value_id in program.outputs.values()
            )
    parallel_private_slots = frozenset(
        index
        for index, value_id in enumerate(all_slot_ids)
        if int(value_id) in parallel_private_value_ids
    )

    consumed: list[int] = []
    next_stream_continuation = 0

    def publication_source(block: StreamPublishBlock) -> tuple[str, str]:
        """Return the predicate and one boolean resident publication call."""

        try:
            payload_slot = all_slot_ids.index(int(block.value_id))
        except ValueError as error:
            raise ValueError(
                "stream payload has no composed shader slot: "
                f"{block.value_id}"
            ) from error
        count_expression = (
            f"u_extent_control[{payload_slot}]"
            if block.count_value_id is None
            else (
                "arena[u_slot["
                f"{all_slot_ids.index(int(block.count_value_id))}"
                "]]"
            )
        )
        predicate = "true"
        if block.predicate_value_id is not None:
            predicate_slot = all_slot_ids.index(
                int(block.predicate_value_id)
            )
            predicate = f"arena[u_slot[{predicate_slot}]] != 0u"
        call = (
            f"turing_stream_publish({int(block.stream_id)}u, "
            f"u_slot[{payload_slot}], {count_expression}, "
            f"{'true' if block.final else 'false'})"
        )
        return predicate, call

    def flatten_execution_scope(block):
        """Flatten only sequence/call composition, preserving nested control."""

        if isinstance(block, SequenceBlock):
            return tuple(
                item
                for child in block.blocks
                for item in flatten_execution_scope(child)
            )
        if isinstance(block, CallBlock):
            return flatten_execution_scope(block.callee)
        return (block,)

    def resumable_loop_source(
        block: LoopBlock,
        continuation_index: int,
    ) -> StatementBlock:
        """Lower one planner loop to a lossless resident continuation.

        Completed iterations are never replayed.  If a publication cannot fit,
        the exact induction value and publication point are saved after all
        invocations converge.  The next dispatch of this same installed shell
        resumes at the publication, preserving its already-computed payload.
        """

        items = flatten_execution_scope(block.body)
        publications = tuple(
            item for item in items if isinstance(item, StreamPublishBlock)
        )
        if not publications:
            raise ValueError("resumable loop has no direct publication")
        continuation_base = (
            "(8u + stream_state[5] * 4u + "
            f"{2 * int(continuation_index)}u)"
        )
        marker_name = (
            f"turing_resume_marker_{int(continuation_index)}"
        )
        active_name = (
            f"turing_resume_active_{int(continuation_index)}"
        )
        start_name = f"turing_resume_start_{int(continuation_index)}"
        lines = [
            "{",
            f"    uint turing_resume_base = {continuation_base};",
            f"    uint {marker_name} = "
            "stream_state[turing_resume_base + 1u];",
            f"    int {start_name} = "
            f"({marker_name} == 0u) ? int({block.start}) : "
            "int(stream_state[turing_resume_base]);",
            f"    for (int {block.induction} = {start_name}; "
            f"{block.induction} < {block.stop}; "
            f"{block.induction} += {block.step}) {{",
            f"        bool {active_name} = ({marker_name} == 0u);",
        ]
        publication_index = 0
        for item in items:
            if isinstance(item, StreamPublishBlock):
                publication_index += 1
                predicate, call = publication_source(item)
                marker = int(publication_index)
                lines.extend((
                    f"        if ({active_name} || "
                    f"{marker_name} == {marker}u) {{",
                    f"            if ({predicate} && !{call}) {{",
                    "                if (gl_LocalInvocationID.x == 0u) {",
                    "                    stream_state[turing_resume_base] = "
                    f"uint({block.induction});",
                    "                    stream_state["
                    "turing_resume_base + 1u] = "
                    f"{marker}u;",
                    "                }",
                    "                barrier();",
                    "                return;",
                    "            }",
                    f"            {marker_name} = 0u;",
                    f"            {active_name} = true;",
                    "        }",
                ))
                continue
            rendered = render_control_block(item, ControlTarget.GLSL)
            lines.append(f"        if ({active_name}) {{")
            lines.extend(f"            {line}" for line in rendered)
            lines.append("        }")
        lines.extend((
            "        if (gl_LocalInvocationID.x == 0u) {",
            "            stream_state[turing_resume_base] = "
            f"uint({block.induction} + ({block.step}));",
            "            stream_state[turing_resume_base + 1u] = 0u;",
            "        }",
            "        barrier();",
            f"        {marker_name} = 0u;",
            "    }",
            "    if (gl_LocalInvocationID.x == 0u) {",
            "        stream_state[turing_resume_base] = "
            f"uint(int({block.start}));",
            "        stream_state[turing_resume_base + 1u] = 0u;",
            "    }",
            "    barrier();",
            "}",
        ))
        return StatementBlock(tuple(lines))

    def parallelize_private_access(line: str) -> str:
        for slot in parallel_private_slots:
            line = line.replace(
                f"arena[u_slot[{slot}] + ",
                f"arena[u_slot[{slot}] + "
                f"uint(gl_WorkGroupID.x) * "
                f"u_extent_control[{slot}] + ",
            )
        return line

    def substitute(block, parallel_scope: bool = False):
        nonlocal next_stream_continuation
        if isinstance(block, StatementBlock):
            if len(block.lines) != 1:
                return block
            marker = block.lines[0]
            prefix = "__scheduled_region_"
            if not marker.startswith(prefix) or not marker.endswith("__"):
                return block
            region_index = int(marker[len(prefix):-2])
            if region_index not in region_snippets:
                raise ValueError(
                    f"control program references absent region {region_index}"
                )
            consumed.append(region_index)
            blocks = []
            if trace_instrumentation:
                blocks.append(StatementBlock((
                    f"turing_debug_event(3u, {region_index}u, "
                    f"{sum(1 for _snippet_index, program_index, _snippet in region_snippets[region_index] if active_program_indices is None or program_index in active_program_indices)}u, u_count);",
                )))
            for index, current_program_index, snippet in (
                region_snippets[region_index]
            ):
                if (
                    active_program_indices is not None
                    and current_program_index not in active_program_indices
                ):
                    continue
                checkpoint = ()
                if (
                    trace_instrumentation
                    and index in scalar_trace_slots
                    and (
                        index >= max(0, snippet_index - 128)
                        or index in closure_source_trace_indices
                    )
                ):
                    checkpoint = tuple(
                            f"turing_debug_event(8u, {int(index)}u, "
                            f"arena[u_slot[{output_slot}]], "
                            f"u_extent[{output_slot}]);"
                            for output_slot in scalar_trace_slots[index]
                        )
                guarded = snippet.guarded(
                        index,
                        device_resident=device_resident,
                    )
                if parallel_scope:
                    guarded = tuple(
                        parallelize_private_access(line)
                        for line in guarded
                    )
                blocks.append(StatementBlock((
                    *guarded,
                    # This orders one invocation's arena spill before that
                    # invocation consumes it in the next same-index snippet.
                    # It is deliberately not ``barrier()`` and makes no claim
                    # of synchronizing workgroups; cross-invocation edges are
                    # separated by the contiguation plan into real dispatches.
                    "memoryBarrierBuffer();",
                    *(("barrier();",) if device_resident else ()),
                    *checkpoint,
                )))
            active_region_snippets = tuple(
                item
                for item in region_snippets[region_index]
                if (
                    active_program_indices is None
                    or item[1] in active_program_indices
                )
            )
            if trace_instrumentation and active_region_snippets:
                last_region_snippet = max(
                    index
                    for index, _program_index, _snippet
                    in active_region_snippets
                )
                lifetime_checks = tuple(
                    f"turing_debug_event(9u, {source_id}u, "
                    f"arena[u_slot[{source_slot}]], {region_index}u);"
                    for source_id, (producer_index, source_slot)
                    in closure_scalar_source_producers.items()
                    if producer_index <= last_region_snippet
                )
                if lifetime_checks:
                    blocks.append(StatementBlock(lifetime_checks))
            return SequenceBlock(tuple(blocks))
        if isinstance(block, SequenceBlock):
            return SequenceBlock(tuple(
                substitute(child, parallel_scope)
                for child in block.blocks
            ))
        if isinstance(block, LoopBlock):
            is_selected_workgroup = (
                str(block.induction) == selected_workgroup_induction
            )
            body = substitute(
                block.body,
                parallel_scope or is_selected_workgroup,
            )
            stop = block.stop
            static_bindings = []
            closure_bindings = []
            for iterable_id, target_id, induction in (
                control_program.iterable_bindings
            ):
                if induction != block.induction:
                    continue
                iterable_slot = all_slot_ids.index(int(iterable_id))
                stop = stop.replace(
                    f"__iterable_extent_{int(iterable_id)}__",
                    f"int(u_extent_control[{iterable_slot}])",
                )
                target_slots = tuple(
                    index
                    for index, value_id in enumerate(all_slot_ids)
                    if int(value_id) == int(target_id)
                )

                def bind_iterable(child):
                    if isinstance(child, StatementBlock):
                        replacement = (
                            f"arena[u_slot[{iterable_slot}] + "
                            f"uint({block.induction})]"
                        )
                        lines = []
                        for line in child.lines:
                            for target_slot in target_slots:
                                line = line.replace(
                                    f"arena[u_slot[{target_slot}] + (gid)]",
                                    replacement,
                                ).replace(
                                    f"arena[u_slot[{target_slot}] + (0)]",
                                    replacement,
                                )
                            lines.append(line)
                        return StatementBlock(tuple(lines))
                    if isinstance(child, SequenceBlock):
                        return SequenceBlock(tuple(
                            bind_iterable(item) for item in child.blocks
                        ))
                    if isinstance(child, CallBlock):
                        return CallBlock(
                            child.callsite_id,
                            bind_iterable(child.callee),
                            child.argument_bindings,
                            child.result_bindings,
                        )
                    if isinstance(child, ValidationBlock):
                        return child
                    return child

                body = bind_iterable(body)
            for _iterable_id, target_id, induction, values in (
                control_program.static_iterable_bindings
            ):
                if induction != block.induction:
                    continue
                target_slots = tuple(
                    index
                    for index, value_id in enumerate(all_slot_ids)
                    if int(value_id) == int(target_id)
                )
                if not target_slots:
                    raise ValueError(
                        "static iterable target has no composed shader slot: "
                        f"{target_id}"
                    )
                static_bindings.append((
                    target_slots,
                    tuple(values),
                ))
            for _iterable_id, target_id, induction, source_ids in (
                control_program.closure_iterable_bindings
            ):
                if induction != block.induction:
                    continue
                target_slots = tuple(
                    index
                    for index, value_id in enumerate(all_slot_ids)
                    if int(value_id) == int(target_id)
                )
                source_slots = tuple(
                    all_slot_ids.index(int(source_id))
                    for source_id in source_ids
                )
                if not target_slots or not source_slots:
                    raise ValueError(
                        "closure iterable has no composed resident slots: "
                        f"target={target_id}, sources={tuple(source_ids)!r}"
                    )
                closure_bindings.append((target_slots, source_slots))
            if static_bindings and len({
                len(values) for _slots, values in static_bindings
            }) != 1:
                raise ValueError(
                    "destructured static iterable bindings have unequal "
                    "iteration counts"
                )
            commits = []
            for updated, initial in block.carried_aliases:
                try:
                    initial_slot = all_slot_ids.index(int(initial))
                    updated_slot = all_slot_ids.index(int(updated))
                except ValueError as error:
                    raise ValueError(
                        "loop-carried value has no composed shader slot: "
                        f"{initial}->{updated}"
                    ) from error
                commits.append(
                    "uint control_gid = turing_linear_gid();"
                    if not commits else ""
                )
                commits.extend((
                    "if (control_gid < u_count) {",
                    f"    arena[u_slot[{initial_slot}] + control_gid] = "
                    f"arena[u_slot[{updated_slot}] + control_gid];",
                    "}",
                ))
            for binding_index, (
                source_id,
                collection_id,
                induction,
                start,
            ) in enumerate(control_program.collection_bindings):
                if induction != block.induction:
                    continue
                try:
                    source_slot = all_slot_ids.index(int(source_id))
                    collection_slot = all_slot_ids.index(
                        int(collection_id)
                    )
                except ValueError as error:
                    raise ValueError(
                        "loop collection value has no composed shader slot: "
                        f"{source_id}->{collection_id}"
                    ) from error
                gid_name = f"collection_gid_{binding_index}"
                extent_name = f"collection_extent_{binding_index}"
                iteration_name = f"collection_iteration_{binding_index}"
                source_frame_offset = (
                    f"uint(gl_WorkGroupID.x) * "
                    f"u_extent_control[{source_slot}] + "
                    if (
                        str(block.induction)
                        == selected_workgroup_induction
                        and source_slot in parallel_private_slots
                    )
                    else ""
                )
                commits.extend((
                    f"uint {extent_name} = "
                    f"u_extent_control[{source_slot}];",
                    f"for (uint {gid_name} = gl_LocalInvocationID.x; "
                    f"{gid_name} < {extent_name}; "
                    f"{gid_name} += gl_WorkGroupSize.x) {{",
                    f"    uint {iteration_name} = uint("
                    f"({block.induction} - ({int(start)})) / "
                    f"int({block.step}));",
                    f"    arena[u_slot[{collection_slot}] + "
                    f"{iteration_name} * {extent_name} + "
                    f"{gid_name}] = "
                    f"arena[u_slot[{source_slot}] + "
                    f"{source_frame_offset}{gid_name}];",
                    "}",
                ))
            if commits:
                body = SequenceBlock((
                    body,
                    StatementBlock(tuple(commits)),
                    *(
                        (StatementBlock((
                            f"turing_debug_event(4u, 0u, "
                            f"uint({block.induction}), u_count);",
                        )),)
                        if trace_instrumentation else ()
                    ),
                ))
            if trace_instrumentation and not static_bindings:
                body = SequenceBlock((
                    StatementBlock((
                        f"turing_debug_event(2u, 0u, "
                        f"uint({block.induction}), u_count);",
                    )),
                    body,
                ))
            if static_bindings:
                iteration_count = len(static_bindings[0][1])
                selection = [
                    f"switch (int({block.induction})) {{"
                ]
                for iteration_index in range(iteration_count):
                    selection.append(f"case {int(iteration_index)}:")
                    for target_slots, values in static_bindings:
                        value = values[iteration_index]
                        for target_slot in target_slots:
                            meta = all_slot_meta[target_slot]
                            dtype = (
                                "" if meta is None or meta.dtype is None
                                else str(meta.dtype)
                            )
                            if "float" in dtype:
                                encoded = (
                                    f"floatBitsToUint(float({value!r}))"
                                )
                            elif "bool" in dtype:
                                encoded = (
                                    "uint(1)" if bool(value) else "uint(0)"
                                )
                            else:
                                encoded = f"uint(int({value!r}))"
                            selection.append(
                                f"    arena[u_slot[{target_slot}]] = "
                                f"{encoded};"
                            )
                    selection.append("    break;")
                selection.extend(("}", "memoryBarrierBuffer();"))
                body = SequenceBlock((
                    StatementBlock(tuple(selection)),
                    body,
                ))
                stop = str(iteration_count)
            if closure_bindings:
                source_counts = {
                    len(source_slots)
                    for _target_slots, source_slots in closure_bindings
                }
                if len(source_counts) != 1:
                    raise ValueError(
                        "destructured closure aggregate bindings have unequal "
                        "iteration counts"
                    )

                def bind_source(
                    child,
                    target_slots,
                    source_offset,
                    source_extent,
                ):
                    if isinstance(child, StatementBlock):
                        lines = []
                        for line in child.lines:
                            for target_slot in target_slots:
                                line = line.replace(
                                    f"u_slot[{target_slot}]",
                                    source_offset,
                                ).replace(
                                    f"u_extent_control[{target_slot}]",
                                    source_extent,
                                )
                            lines.append(line)
                        return StatementBlock(tuple(lines))
                    if isinstance(child, SequenceBlock):
                        return SequenceBlock(tuple(
                            bind_source(
                                item,
                                target_slots,
                                source_offset,
                                source_extent,
                            )
                            for item in child.blocks
                        ))
                    if isinstance(child, CallBlock):
                        return CallBlock(
                            child.callsite_id,
                            bind_source(
                                child.callee,
                                target_slots,
                                source_offset,
                                source_extent,
                            ),
                            child.argument_bindings,
                            child.result_bindings,
                        )
                    return child

                selection = []
                for binding_index, (
                    target_slots,
                    source_slots,
                ) in enumerate(closure_bindings):
                    offset_name = (
                        f"closure_source_{block.induction}_{binding_index}"
                    )
                    extent_name = (
                        f"closure_extent_{block.induction}_{binding_index}"
                    )
                    selection.extend((
                        f"uint {offset_name} = 0u;",
                        f"uint {extent_name} = 0u;",
                        f"switch (int({block.induction})) {{",
                    ))
                    for iteration_index, source_slot in enumerate(
                        source_slots
                    ):
                        selection.extend((
                            f"case {int(iteration_index)}:",
                            f"    {offset_name} = u_slot[{source_slot}];",
                            f"    {extent_name} = "
                            f"u_extent_control[{source_slot}];",
                            "    break;",
                        ))
                    selection.append("}")
                    body = bind_source(
                        body,
                        target_slots,
                        offset_name,
                        extent_name,
                    )
                body = SequenceBlock((
                    StatementBlock(tuple(selection)),
                    body,
                ))
                stop = str(next(iter(source_counts)))
            lowered_loop = LoopBlock(
                block.induction,
                block.start,
                stop,
                block.step,
                body,
                block.carried_aliases,
                bool(
                    block.parallel_iterations
                    and str(block.induction)
                    == selected_workgroup_induction
                ),
                (
                    "c"
                    if str(block.induction)
                    == selected_c_dispatch_induction
                    else "glsl"
                ),
            )
            if (
                device_resident
                and _direct_stream_publications(lowered_loop.body)
            ):
                continuation_index = next_stream_continuation
                next_stream_continuation += 1
                return resumable_loop_source(
                    lowered_loop,
                    continuation_index,
                )
            return lowered_loop
        if isinstance(block, StateMachineTick):
            return StateMachineTick(
                block.state,
                tuple(
                    (value, substitute(body, parallel_scope))
                    for value, body in block.cases
                ),
            )
        if isinstance(block, ParallelDeployment):
            return ParallelDeployment(tuple(
                substitute(lane, parallel_scope) for lane in block.lanes
            ))
        if isinstance(block, CallBlock):
            return CallBlock(
                block.callsite_id,
                substitute(block.callee, parallel_scope),
                block.argument_bindings,
                block.result_bindings,
            )
        if isinstance(block, ValidationBlock):
            if not emit_validations:
                return StatementBlock(())
            try:
                predicate_slot = all_slot_ids.index(
                    int(block.predicate_value_id)
                )
            except ValueError as error:
                raise ValueError(
                    "validation predicate has no composed shader slot: "
                    f"{block.predicate_value_id}"
                ) from error
            comparison = "== 0u" if block.expect_true else "!= 0u"
            return StatementBlock((
                "if (turing_linear_gid() == 0u && "
                f"arena[u_slot[{predicate_slot}]] {comparison}) {{",
                f"    turing_debug_event(6u, "
                f"{int(block.error_code)}u, 0u, 0u);",
                "}",
            ))
        if isinstance(block, StreamPublishBlock):
            if device_resident:
                # Keep the logical effect intact until its owning loop has
                # converted it into a resident suspension point.
                return block
            predicate, call = publication_source(block)
            return StatementBlock((
                f"if ({predicate}) {{",
                f"    {call};",
                "}",
            ))
        raise TypeError(f"unknown control block {type(block).__name__}")

    substituted_root = substitute(control_program.root)
    remaining_publications = _control_stream_publications(substituted_root)
    if device_resident and remaining_publications:
        direct_publications = _direct_stream_publications(substituted_root)
        if direct_publications != remaining_publications:
            raise ValueError(
                "device-resident stream publication lacks a planner-owned "
                "resumable execution scope: "
                f"{remaining_publications!r}"
            )
        # A root-level publication is a one-iteration execution scope.  Give
        # it the same persistent continuation semantics as an explicit loop
        # so a full downstream queue never causes producer replay.
        continuation_index = next_stream_continuation
        next_stream_continuation += 1
        substituted_root = resumable_loop_source(
            LoopBlock(
                f"turing_root_once_{continuation_index}",
                "0",
                "1",
                "1",
                substituted_root,
            ),
            continuation_index,
        )
    body = render_control_block(substituted_root, ControlTarget.GLSL)
    # Control values belong to the installed shell, not to the driver's tiny
    # legacy uniform register file.  Large, honestly composed programs can
    # contain thousands of scheduled extents and loop controls; spelling each
    # one as a GLSL uniform makes program size determine whether the driver can
    # compile it.  Index them through one resident table instead.  This is an
    # ABI rewrite of planner identities, not specialization from discovery
    # values and emphatically not another tape/capture pass.
    control_table_indices = {
        uniform.name: index
        for index, uniform in enumerate(control_program.uniforms)
    }
    for name, index in control_table_indices.items():
        body = tuple(
            re.sub(
                rf"\b{re.escape(name)}\b",
                f"int(u_control[{index}])",
                line,
            )
            for line in body
        )
    # A scalar constant stage has extent one, so only invocation zero executes
    # its arena write.  Textually placing a later broadcast consumer after it
    # does not create a device-wide execution barrier: other workgroups may
    # read the old word.  Constants are compile-time values, not runtime state,
    # so replace every *read* of their slots with the typed literal.  This is
    # both the correct vertical-chain reduction and avoids pretending that
    # source order is a global synchronization primitive.
    static_slots = {
        index: (
            static_scalar_values[value_id],
            all_slot_meta[index],
        )
        for index, value_id in enumerate(all_slot_ids)
        if value_id in static_scalar_values
    }
    for slot, (value, meta) in static_slots.items():
        dtype = "" if meta is None else str(meta.dtype or "")
        if "float" in dtype:
            encoded = (
                f"floatBitsToUint(float({float(value)!r}))"
            )
        elif "bool" in dtype:
            encoded = (
                "uint(1)" if bool(value) else "uint(0)"
            )
        else:
            encoded = f"uint(int({int(value)}))"
        static_slots[slot] = (encoded, meta)
    if static_slots:
        static_read = re.compile(
            r"arena\[u_slot\[(\d+)\] \+ \([^)]*\)\]"
        )

        def replace_static_read(match):
            fixed = static_slots.get(int(match.group(1)))
            return match.group(0) if fixed is None else fixed[0]

        rewritten = []
        for line in body:
            if "=" not in line:
                rewritten.append(line)
                continue
            lhs, rhs = line.split("=", 1)
            rewritten.append(
                lhs + "=" + static_read.sub(replace_static_read, rhs)
            )
        body = tuple(rewritten)
    if Counter(consumed) != Counter(expected):
        raise ValueError(
            "control program must consume each submitted region exactly once: "
            f"expected={expected!r}, consumed={tuple(consumed)!r}"
        )
    helpers = tuple(dict.fromkeys(
        helper
        for region_index in expected
        for _index, current_program_index, snippet
        in region_snippets[region_index]
        if (
            active_program_indices is None
            or current_program_index in active_program_indices
        )
        for helper in snippet.helpers
        if helper
    ))
    shared = tuple(dict.fromkeys(
        declaration
        for region_index in expected
        for _index, current_program_index, snippet
        in region_snippets[region_index]
        if (
            active_program_indices is None
            or current_program_index in active_program_indices
        )
        for declaration in snippet.shared
        if declaration
    ))
    return "\n".join([
        (
            _WORKGROUP_CONTROL_SHADER_HEADER
            if selected_workgroup_induction is not None
            else _SHADER_HEADER
        ).format(local_size=local_size),
        _ARENA_BLOCK,
        *(
            (
                "layout(std430, binding = 1) buffer TuringDebugLog "
                "{ uint debug_words[]; };",
                "uniform uint u_debug_capacity;",
            )
            if debug_enabled else ()
        ),
        *(
            (
                "layout(std430, binding = 2) buffer "
                "TuringStreamState { uint stream_state[]; };",
                "layout(std430, binding = 3) buffer "
                "TuringStreamWords { uint stream_words[]; };",
                "shared uint turing_stream_start;",
                "shared uint turing_stream_count;",
                "shared uint turing_stream_descriptor;",
                "shared uint turing_stream_accept;",
            )
            if stream_publications else ()
        ),
        (
            "layout(std430, binding = 4) readonly buffer "
            "TuringSlotTable { uint u_slot[]; };"
        ),
        (
            "layout(std430, binding = 5) readonly buffer "
            "TuringControlExtents { uint u_extent_control[]; };"
        ),
        (
            "layout(std430, binding = 6) readonly buffer "
            "TuringDispatchExtents { uint u_extent[]; };"
        ),
        (
            "layout(std430, binding = 7) readonly buffer "
            "TuringControlValues { uint u_control[]; };"
        ),
        "",
        "uniform uint u_count;",
        *(
            ("uniform int u_dispatch_iteration;",)
            if selected_c_dispatch_loop is not None
            else ()
        ),
        *(
            (
                "bool turing_stream_publish(",
                "    uint stream_id, uint source, uint count, bool final",
                ") {",
                "    if (gl_LocalInvocationID.x == 0u) {",
                "        uint used_words = stream_state[0] - stream_state[1];",
                "        uint used_desc = stream_state[2] - stream_state[3];",
                "        uint word_capacity = stream_state[4];",
                "        uint desc_capacity = stream_state[5];",
                "        turing_stream_accept = uint(",
                "            count <= word_capacity - used_words &&",
                "            used_desc < desc_capacity",
                "        );",
                "        if (turing_stream_accept != 0u) {",
                "            turing_stream_start = stream_state[0];",
                "            turing_stream_count = count;",
                "            turing_stream_descriptor = stream_state[2];",
                "        } else {",
                "            stream_state[6] = "
                "(count > word_capacity) ? 2u : 1u;",
                "            stream_state[7] = count;",
                "        }",
                "    }",
                "    barrier();",
                "    if (turing_stream_accept == 0u) return false;",
                "    uint word_capacity = stream_state[4];",
                "    for (uint index = gl_LocalInvocationID.x; index < count;",
                "         index += gl_WorkGroupSize.x) {",
                "        stream_words[(turing_stream_start + index) % "
                "word_capacity] = arena[source + index];",
                "    }",
                "    barrier();",
                "    if (gl_LocalInvocationID.x == 0u) {",
                "        uint desc_capacity = stream_state[5];",
                "        uint desc = 8u + "
                "(turing_stream_descriptor % desc_capacity) * 4u;",
                "        stream_state[desc] = turing_stream_start;",
                "        stream_state[desc + 1u] = turing_stream_count;",
                "        stream_state[desc + 2u] = stream_id;",
                "        stream_state[desc + 3u] = uint(final);",
                "        memoryBarrierBuffer();",
                "        stream_state[0] += turing_stream_count;",
                "        stream_state[2] += 1u;",
                "        stream_state[6] = 0u;",
                "    }",
                "    barrier();",
                "    return true;",
                "}",
                "",
            )
            if stream_publications else ()
        ),
        *shared,
        "",
        *([*helpers, ""] if helpers else []),
        *(
            (
                "void turing_debug_event(",
                "    uint code, uint subject, uint payload0, uint payload1",
                ") {",
                "    if (turing_linear_gid() != 0u) return;",
                "    uint index = atomicAdd(debug_words[0], 1u);",
                "    if (index >= u_debug_capacity) {",
                "        atomicAdd(debug_words[1], 1u);",
                "        return;",
                "    }",
                "    uint base = 4u + index * 4u;",
                "    debug_words[base] = code;",
                "    debug_words[base + 1u] = subject;",
                "    debug_words[base + 2u] = payload0;",
                "    debug_words[base + 3u] = payload1;",
                "}",
                "",
            )
            if debug_enabled else ()
        ),
        "void main() {",
        *(
            ("    turing_debug_event(1u, 0u, u_count, 0u);",)
            if trace_instrumentation else ()
        ),
        *_indent_control_shader_body(body),
        *(
            ("    turing_debug_event(5u, 0u, u_count, 0u);",)
            if trace_instrumentation else ()
        ),
        "}",
        "",
    ]) + "\n"


def _canonical_cache_value(value: Any) -> Any:
    """Convert compiler IR into deterministic, address-free JSON data."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return {
            "bytes_type": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            ),
            "bytes_length": len(payload),
            "bytes_sha256": hashlib.sha256(payload).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "array_dtype": str(contiguous.dtype),
            "array_shape": list(contiguous.shape),
            "array_sha256": hashlib.sha256(
                contiguous.tobytes()
            ).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _canonical_cache_value(value.item())
    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _canonical_cache_value(value.value),
        }
    if isinstance(value, FusedProgram):
        return {
            "type": "FusedProgram",
            "version": int(value.version),
            "feeds": sorted(map(int, value.feeds)),
            # Fused steps are SSA operations.  Their list order is a lowering
            # schedule, not semantics, so use producer identity as the stable
            # order while retaining every dependency and attribute.
            "steps": [
                _canonical_cache_value(step)
                for step in sorted(
                    value.steps,
                    key=lambda step: (
                        int(step.result_id),
                        int(step.step_id),
                    ),
                )
            ],
            "outputs": _canonical_cache_value(value.outputs),
            "state_in": _canonical_cache_value(value.state_in),
            "meta": _canonical_cache_value(value.meta),
            "extras": _canonical_cache_value(value.extras),
        }
    if isinstance(value, OpStep):
        return {
            "type": "OpStep",
            "op_name": str(value.op_name),
            "input_ids": tuple(map(int, value.input_ids)),
            "attrs": _canonical_cache_value(value.attrs),
            "result_id": int(value.result_id),
            "mode_sensitive": bool(value.mode_sensitive),
        }
    if is_dataclass(value):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _canonical_cache_value(
                    getattr(value, item.name)
                )
                for item in fields(value)
            },
        }
    if isinstance(value, Mapping):
        items = [
            (
                _canonical_cache_value(key),
                _canonical_cache_value(item),
            )
            for key, item in value.items()
        ]
        items.sort(
            key=lambda pair: json.dumps(
                pair[0], sort_keys=True, separators=(",", ":")
            )
        )
        return {"mapping": items}
    if isinstance(value, (set, frozenset)):
        items = [_canonical_cache_value(item) for item in value]
        items.sort(
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":")
            )
        )
        return {"set": items}
    if isinstance(value, (tuple, list)):
        return [_canonical_cache_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.dtype, type)):
        return str(value)
    raise TypeError(
        "GLSL semantic cache identity cannot encode "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _semantic_cache_digest(value: Any) -> str:
    payload = json.dumps(
        _canonical_cache_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _lowering_implementation_digest() -> str:
    """Fingerprint code that turns the semantic artifact into GLSL."""

    digest = hashlib.sha256()
    paths = (
        Path(__file__),
        Path(__file__).parents[3] / "compiler" / "control_source.py",
        Path(__file__).parents[3] / "compiler" / "loop_ir.py",
        Path(__file__).parents[3] / "compiler" / "loop_composer.py",
        Path(__file__).parents[3] / "compiler"
        / "hierarchical_control.py",
        Path(__file__).parents[3] / "compiler"
        / "glsl_deployment_strategy.py",
    )
    for path in paths:
        digest.update(str(path.name).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_control_shader_artifact(
    control_program,
    captured_regions: Mapping[int, Any],
    *,
    local_size: int = _LOCAL_SIZE,
    value_meta: Mapping[int, Meta] | None = None,
    value_contract_diagnostics: Mapping[
        int, Sequence[Mapping[str, Any]]
    ] | None = None,
    instrumentation: bool = False,
    terminal_outputs: Mapping[str, int] | None = None,
    stream_outputs: Mapping[str, int] | None = None,
    specialized_values: Mapping[int, Any] | None = None,
    device_resident: bool = False,
) -> ComposedGLSLControlArtifact:
    """Build source and the value-routing plan required to execute it."""

    source = compose_control_shader(
        control_program,
        captured_regions,
        local_size=local_size,
        instrumentation=instrumentation,
        device_resident=device_resident,
    )
    instrumentation = bool(
        instrumentation or "turing_debug_event(6u" in source
    )
    program_records = []
    snippet_diagnostics = []
    slot_base = 0
    for region_index in control_program.region_indices:
        captured = captured_regions[region_index]
        stages = tuple(getattr(captured, "stages", ()) or ())
        parts = (
            tuple(type(captured)(stage, {}) for stage in stages)
            if stages
            else (captured,)
        )
        for stage_index, part in enumerate(parts):
            snippet = captured_program_snippet(
                part,
                base=slot_base,
                local_size=local_size,
            )
            local_values = (
                *ordered_feed_ids(part.program),
                *tuple(part.program.outputs.values()),
            )
            kernel_kind = (part.program.extras or {}).get("kernel_kind")
            if (
                len(part.program.steps) == 1
                and kernel_kind not in {None, "linear_reshape_copy"}
            ):
                positional_values = (
                    *tuple(part.program.steps[0].input_ids),
                    *tuple(part.program.outputs.values()),
                )
                if len(positional_values) == snippet.slots:
                    local_values = positional_values
            if len(local_values) != snippet.slots:
                raise ValueError(
                    "artifact slot ABI disagrees with emitted shader: "
                    f"region={region_index}, base={slot_base}, "
                    f"snippet_slots={snippet.slots}, values={local_values!r}"
                )
            program_records.append((part.program, tuple(
                int(value_id) for value_id in local_values
            )))
            snippet_diagnostics.append({
                "index": len(snippet_diagnostics),
                "region": int(region_index),
                "stage": int(stage_index),
                "slot_base": int(slot_base),
                "slot_count": int(snippet.slots),
                "output_slot": int(slot_base + snippet.slots - 1),
                "feeds": tuple(map(int, ordered_feed_ids(part.program))),
                "outputs": tuple(
                    (str(name), int(value_id))
                    for name, value_id in part.program.outputs.items()
                ),
                "operations": tuple(
                    str(step.op_name) for step in part.program.steps
                ),
            })
            slot_base += snippet.slots
    programs = tuple(program for program, _slots in program_records)

    raw_aliases = {
        int(alias): int(owner)
        for alias, owner in control_program.value_aliases
    }

    def canonical_alias(value_id: int) -> int:
        current = int(value_id)
        seen = set()
        while current in raw_aliases:
            if current in seen:
                raise ValueError(
                    "compiled GLSL value-alias cycle: "
                    + " -> ".join(map(str, (*seen, current)))
                )
            seen.add(current)
            current = int(raw_aliases[current])
        return current

    aliases = {
        alias: canonical_alias(alias)
        for alias in raw_aliases
    }

    produced = {
        int(value_id)
        for program in programs
        for value_id in program.outputs.values()
    }
    consumed = {
        int(value_id)
        for program in programs
        for value_id in program.feeds
    }
    slot_value_ids = []
    extents = []
    metadata: dict[int, Meta] = {}
    output_names: dict[int, str] = {}
    for program, local_slots in program_records:
        slot_value_ids.extend(local_slots)
        for value_id, meta in (program.meta or {}).items():
            metadata.setdefault(int(value_id), meta)
        output_id = next(iter(program.outputs.values()))
        output_meta = (program.meta or {})[output_id]
        extents.append(_shape_product(tuple(output_meta.shape or ())))
        for name, value_id in program.outputs.items():
            output_names[int(value_id)] = str(name)

    for value_id, meta in (value_meta or {}).items():
        metadata.setdefault(int(value_id), meta)

    def propagate_storage_contracts() -> None:
        """Close storage facts over every identity relation to a fixed point.

        Metadata discovery is intentionally distributed: numerical lowering,
        lexical/static loop analysis, hierarchy projection, and collection
        planning each learn different structural facts.  Their order must not
        decide whether an otherwise identical SSA value receives storage.
        """

        changed = True
        while changed:
            changed = False
            for alias_id, owner_id in aliases.items():
                alias_id = int(alias_id)
                owner_id = int(owner_id)
                alias_meta = metadata.get(alias_id)
                owner_meta = metadata.get(owner_id)
                if owner_meta is None and alias_meta is not None:
                    metadata[owner_id] = alias_meta
                    changed = True
                elif alias_meta is None and owner_meta is not None:
                    metadata[alias_id] = owner_meta
                    changed = True
            for _iterable_id, target_id, _induction, source_ids in (
                control_program.closure_iterable_bindings
            ):
                target_id = int(target_id)
                target_meta = metadata.get(target_id)
                if target_meta is None:
                    target_meta = next(
                        (
                            metadata.get(int(aliases.get(
                                int(source_id), int(source_id)
                            )))
                            or metadata.get(int(source_id))
                            for source_id in source_ids
                            if (
                                metadata.get(int(aliases.get(
                                    int(source_id), int(source_id)
                                )))
                                or metadata.get(int(source_id))
                            ) is not None
                        ),
                        None,
                    )
                    if target_meta is not None:
                        metadata[target_id] = target_meta
                        changed = True
                if target_meta is None:
                    continue
                # Every projected source occupies the same lexical field
                # position consumed by the one compiled loop body.  Therefore
                # the target ABI is also the source ABI.
                for source_id in source_ids:
                    source_id = int(source_id)
                    owner_id = int(aliases.get(source_id, source_id))
                    if metadata.get(source_id) is None:
                        metadata[source_id] = target_meta
                        changed = True
                    if metadata.get(owner_id) is None:
                        metadata[owner_id] = target_meta
                        changed = True

    # Structural lowering can split one capture into stages.  Scalar values
    # crossing those stage boundaries are genuine program feeds, but older
    # captures did not attach Meta to them.  Recover only their storage
    # contract from the typed operation relation; never inspect a captured
    # payload or bake its value into the shader.
    for program in programs:
        for feed_id in program.feeds:
            feed_id = int(feed_id)
            if feed_id in metadata:
                continue
            inferred_dtype = None
            for step in program.steps:
                if feed_id not in step.input_ids:
                    continue
                candidates = (
                    metadata.get(int(step.result_id)),
                    *(
                        metadata.get(int(other_id))
                        for other_id in step.input_ids
                        if int(other_id) != feed_id
                    ),
                )
                inferred_dtype = next(
                    (
                        str(meta.dtype)
                        for meta in candidates
                        if meta is not None and meta.dtype is not None
                    ),
                    None,
                )
                if inferred_dtype is not None:
                    break
            if inferred_dtype is not None:
                metadata[feed_id] = Meta((), inferred_dtype, "glsl")

    for _iterable_id, target_id, _induction, values in (
        control_program.static_iterable_bindings
    ):
        target_id = int(target_id)
        if target_id in metadata:
            continue
        values = tuple(values)
        if values and all(
            isinstance(value, (bool, np.bool_)) for value in values
        ):
            dtype = "bool"
        elif values and all(
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            for value in values
        ):
            dtype = "int32"
        elif values and all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            for value in values
        ):
            dtype = "float32"
        else:
            raise TypeError(
                "static loop binding needs a homogeneous scalar storage "
                f"contract: target={target_id}, values={values!r}"
            )
        # This records storage for a planner-owned source literal; it does not
        # convert a tensor or override AbstractTensor dtype selection.
        metadata[target_id] = Meta((), dtype, "glsl")

    for publication in _control_stream_publications(control_program.root):
        if publication.count_value_id is not None:
            metadata.setdefault(
                int(publication.count_value_id),
                Meta((), "int32", "glsl"),
            )
        if publication.predicate_value_id is not None:
            metadata.setdefault(
                int(publication.predicate_value_id),
                Meta((), "bool", "glsl"),
            )

    propagate_storage_contracts()

    from src.compiler.control_source import (
        CallBlock,
        LoopBlock,
        ParallelDeployment,
        SequenceBlock,
        StateMachineTick,
    )

    loop_trip_counts: dict[str, int] = {}

    specialized_controls = {
        f"u_control_{int(value_id)}": int(
            value.item() if hasattr(value, "item") else value
        )
        for value_id, value in (specialized_values or {}).items()
        if isinstance(
            value.item() if hasattr(value, "item") else value,
            (bool, int, np.bool_, np.integer),
        )
    }

    def resolve_control_int(source: str) -> int:
        expression = ast.parse(str(source), mode="eval").body

        def evaluate(node) -> int:
            if isinstance(node, ast.Constant) and isinstance(
                node.value, (bool, int)
            ):
                return int(node.value)
            if isinstance(node, ast.Name) and node.id in specialized_controls:
                return int(specialized_controls[node.id])
            if isinstance(node, ast.UnaryOp):
                operand = evaluate(node.operand)
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.UAdd):
                    return operand
            if isinstance(node, ast.BinOp):
                left = evaluate(node.left)
                right = evaluate(node.right)
                operations = {
                    ast.Add: lambda: left + right,
                    ast.Sub: lambda: left - right,
                    ast.Mult: lambda: left * right,
                    ast.FloorDiv: lambda: left // right,
                    ast.Mod: lambda: left % right,
                }
                operation = operations.get(type(node.op))
                if operation is not None:
                    return int(operation())
            raise ValueError(source)

        return evaluate(expression)

    def gather_loop_trip_counts(block) -> None:
        if isinstance(block, LoopBlock):
            try:
                loop_range = range(
                    resolve_control_int(block.start),
                    resolve_control_int(block.stop),
                    resolve_control_int(block.step),
                )
            except (SyntaxError, ValueError, ZeroDivisionError):
                pass
            else:
                loop_trip_counts[str(block.induction)] = len(loop_range)
            gather_loop_trip_counts(block.body)
        elif isinstance(block, SequenceBlock):
            for child in block.blocks:
                gather_loop_trip_counts(child)
        elif isinstance(block, StateMachineTick):
            for _case, body in block.cases:
                gather_loop_trip_counts(body)
        elif isinstance(block, ParallelDeployment):
            for lane in block.lanes:
                gather_loop_trip_counts(lane)
        elif isinstance(block, CallBlock):
            gather_loop_trip_counts(block.callee)

    gather_loop_trip_counts(control_program.root)
    for source_id, collection_id, induction, _start in (
        control_program.collection_bindings
    ):
        source_id = int(source_id)
        collection_id = int(collection_id)
        if collection_id in metadata:
            continue
        source_meta = metadata.get(
            int(aliases.get(source_id, source_id))
        )
        iterations = loop_trip_counts.get(str(induction))
        if (
            source_meta is not None
            and source_meta.shape is not None
            and source_meta.dtype is not None
            and iterations is not None
        ):
            metadata[collection_id] = Meta(
                (int(iterations), *tuple(source_meta.shape or ())),
                source_meta.dtype,
                "glsl",
            )

    # Collection inference happens after loop-bound resolution, so close the
    # same relation set once more.  This is one monotone fixed-point analysis,
    # not a phase-order-dependent series of special cases.
    propagate_storage_contracts()

    for iterable_id, _target_id, _induction in (
        control_program.iterable_bindings
    ):
        iterable_id = int(iterable_id)
        if iterable_id not in slot_value_ids:
            slot_value_ids.append(iterable_id)

    for _iterable_id, target_id, _induction, _values in (
        control_program.static_iterable_bindings
    ):
        target_id = int(target_id)
        if target_id not in slot_value_ids:
            slot_value_ids.append(target_id)

    for _source_id, collection_id, _induction, _start in (
        control_program.collection_bindings
    ):
        collection_id = int(collection_id)
        if collection_id not in slot_value_ids:
            slot_value_ids.append(collection_id)

    for _iterable_id, target_id, _induction, source_ids in (
        control_program.closure_iterable_bindings
    ):
        for value_id in (int(target_id), *map(int, source_ids)):
            if value_id not in slot_value_ids:
                slot_value_ids.append(value_id)
    for updated_id, initial_id in control_program.value_aliases:
        for value_id in (int(updated_id), int(initial_id)):
            if value_id not in slot_value_ids:
                slot_value_ids.append(value_id)
    # Keep the executable slot ABI identical to compose_control_shader().
    # Stream-only values may not be feeds or outputs of a numerical region,
    # but the control shader still addresses their resident ranges.  Omitting
    # them here leaves valid source indexing beyond the uploaded slot/extent
    # tables and silently publishes zero words.
    for publication in _control_stream_publications(
        control_program.root
    ):
        for value_id in (
            publication.value_id,
            publication.count_value_id,
            publication.predicate_value_id,
        ):
            if value_id is None:
                continue
            value_id = int(value_id)
            if value_id not in slot_value_ids:
                slot_value_ids.append(value_id)

    static_targets = {
        int(target_id)
        for _iterable_id, target_id, _induction, _values
        in control_program.static_iterable_bindings
    }
    static_iterables = {
        int(iterable_id)
        for iterable_id, _target_id, _induction, _values
        in control_program.static_iterable_bindings
    }
    collection_targets = {
        int(collection_id)
        for _source_id, collection_id, _induction, _start
        in control_program.collection_bindings
    }
    closure_targets = {
        int(target_id)
        for _iterable_id, target_id, _induction, _sources
        in control_program.closure_iterable_bindings
    }
    closure_aggregates = {
        int(iterable_id)
        for iterable_id, _target_id, _induction, _sources
        in control_program.closure_iterable_bindings
    }
    external = tuple(dict.fromkeys(
        int(value_id)
        for program in programs
        for value_id in ordered_feed_ids(program)
        if int(value_id) not in produced
        and int(value_id) not in static_targets
        and int(value_id) not in static_iterables
        and int(value_id) not in collection_targets
        and int(value_id) not in closure_targets
        and int(value_id) not in closure_aggregates
        and int(value_id) not in aliases
    ))
    if terminal_outputs is not None:
        terminal = {
            str(name): int(value_id)
            for name, value_id in terminal_outputs.items()
        }
        absent = set(terminal.values()) - set(slot_value_ids)
        if absent:
            raise ValueError(
                "planner-declared terminal values have no composed shader "
                f"slot: {tuple(sorted(absent))!r}"
            )
    else:
        terminal = {
            output_names[value_id]: int(value_id)
            for value_id in produced - consumed
            if value_id in output_names
        }
    if not terminal and programs:
        terminal = {
            str(name): int(value_id)
            for name, value_id in programs[-1].outputs.items()
        }
    from src.compiler.contiguous_execution import contiguate

    contiguous_plan = contiguate(programs)
    phase_sources = (
        (source,)
        if device_resident
        else tuple(
            compose_control_shader(
                control_program,
                captured_regions,
                local_size=local_size,
                instrumentation=instrumentation,
                active_program_indices=phase.program_indices,
                emit_validations=(
                    phase_index == len(contiguous_plan.phases) - 1
                ),
            )
            for phase_index, phase in enumerate(contiguous_plan.phases)
        )
    )
    slot_contract_diagnostics: dict[int, list[dict[str, Any]]] = {
        int(value_id): [dict(row) for row in rows]
        for value_id, rows in (value_contract_diagnostics or {}).items()
    }
    for source_id, collection_id, induction, start in (
        control_program.collection_bindings
    ):
        slot_contract_diagnostics.setdefault(
            int(collection_id), []
        ).append({
            "kind": "collection",
            "source": int(source_id),
            "source_owner": int(aliases.get(int(source_id), int(source_id))),
            "source_meta": metadata.get(
                int(aliases.get(int(source_id), int(source_id)))
            ),
            "induction": str(induction),
            "trip_count": loop_trip_counts.get(str(induction)),
            "start": int(start),
        })
    for iterable_id, target_id, induction, source_ids in (
        control_program.closure_iterable_bindings
    ):
        row = {
            "kind": "closure-iterable",
            "iterable": int(iterable_id),
            "target": int(target_id),
            "target_meta": metadata.get(int(target_id)),
            "induction": str(induction),
            "sources": tuple(
                (
                    int(source_id),
                    int(aliases.get(int(source_id), int(source_id))),
                    metadata.get(
                        int(aliases.get(int(source_id), int(source_id)))
                    ),
                )
                for source_id in source_ids
            ),
        }
        for value_id in (int(target_id), *map(int, source_ids)):
            slot_contract_diagnostics.setdefault(value_id, []).append(row)
    for publication in _control_stream_publications(control_program.root):
        row = {
            "kind": "stream",
            "stream": int(publication.stream_id),
            "value": int(publication.value_id),
            "count": (
                None
                if publication.count_value_id is None
                else int(publication.count_value_id)
            ),
        }
        for value_id in (
            publication.value_id,
            publication.count_value_id,
            publication.predicate_value_id,
        ):
            if value_id is not None:
                slot_contract_diagnostics.setdefault(
                    int(value_id), []
                ).append(row)
    selected_workgroup_loop = _selected_workgroup_loop(
        control_program.root
    )
    workgroup_loop_bounds = (
        None
        if selected_workgroup_loop is None
        else (
            str(selected_workgroup_loop.start),
            str(selected_workgroup_loop.stop),
            str(selected_workgroup_loop.step),
        )
    )
    selected_c_dispatch_loop = _selected_c_dispatch_loop(
        control_program.root
    )
    c_dispatch_loop_bounds = (
        None
        if selected_c_dispatch_loop is None
        else (
            str(selected_c_dispatch_loop.start),
            str(selected_c_dispatch_loop.stop),
            str(selected_c_dispatch_loop.step),
        )
    )
    private_value_capacities: dict[int, int] = {}
    if selected_workgroup_loop is not None:
        induction = str(selected_workgroup_loop.induction)
        trip_capacity = loop_trip_counts.get(induction)
        if trip_capacity is None:
            for source_id, collection_id, binding_induction, _start in (
                control_program.collection_bindings
            ):
                if str(binding_induction) != induction:
                    continue
                source_meta = metadata.get(
                    int(aliases.get(int(source_id), int(source_id)))
                )
                collection_meta = metadata.get(
                    int(aliases.get(
                        int(collection_id), int(collection_id)
                    ))
                )
                if source_meta is None or collection_meta is None:
                    continue
                source_extent = _shape_product(
                    tuple(source_meta.shape or ())
                )
                collection_extent = _shape_product(
                    tuple(collection_meta.shape or ())
                )
                if (
                    source_extent > 0
                    and collection_extent % source_extent == 0
                ):
                    trip_capacity = collection_extent // source_extent
                    break
        if trip_capacity is None:
            raise ValueError(
                "workgroup-parallel loop lacks a resident batch capacity: "
                f"{induction}"
            )
        private_ids = set()
        for region_index in _control_scheduled_regions(
            selected_workgroup_loop.body
        ):
            captured = captured_regions[int(region_index)]
            stages = tuple(getattr(captured, "stages", ()) or ())
            for program in (
                stages if stages else (captured.program,)
            ):
                private_ids.update(
                    map(int, program.outputs.values())
                )
        for value_id in private_ids:
            owner = int(aliases.get(value_id, value_id))
            meta = metadata.get(owner)
            if meta is None or meta.shape is None:
                raise ValueError(
                    "workgroup-private value lacks a storage contract: "
                    f"{value_id}->{owner}"
                )
            private_value_capacities[owner] = (
                int(trip_capacity)
                * _shape_product(tuple(meta.shape or ()))
            )
    semantic_cache_record = {
        "cache_schema": "turing-glsl-semantic-v2",
        "lowering_implementation": _lowering_implementation_digest(),
        "control_program": control_program,
        "program_records": tuple(program_records),
        "slot_value_ids": tuple(slot_value_ids),
        "extents": tuple(extents),
        "metadata": metadata,
        "aliases": aliases,
        "external": external,
        "terminal": terminal,
        "specialized_values": {
            int(value_id): value
            for value_id, value in (specialized_values or {}).items()
        },
        "instrumentation": bool(instrumentation),
        "device_resident": bool(device_resident),
        "local_size": int(local_size),
        "workgroup_loop_bounds": workgroup_loop_bounds,
        "c_dispatch_loop_bounds": c_dispatch_loop_bounds,
        "private_value_capacities": private_value_capacities,
    }
    semantic_base = _semantic_cache_digest(semantic_cache_record)
    phase_cache_identities = tuple(
        _semantic_cache_digest({
            "semantic_base": semantic_base,
            "phase": int(phase_index),
            "program_indices": (
                tuple(range(len(programs)))
                if device_resident
                else tuple(
                    contiguous_plan.phases[
                        phase_index
                    ].program_indices
                )
            ),
            "emits_validations": bool(
                device_resident
                or phase_index == len(phase_sources) - 1
            ),
        })
        for phase_index in range(len(phase_sources))
    )
    return ComposedGLSLControlArtifact(
        source=source,
        slot_value_ids=tuple(slot_value_ids),
        extents=tuple(extents),
        slot_extents=tuple(
            (
                _shape_product(tuple(
                    metadata[
                        int(aliases.get(value_id, value_id))
                    ].shape or ()
                ))
                if int(aliases.get(value_id, value_id)) in metadata
                else 0
            )
            for value_id in slot_value_ids
        ),
        value_meta=metadata,
        external_value_ids=external,
        terminal_outputs=terminal,
        uniform_value_ids={
            uniform.name: int(uniform.value_id)
            for uniform in control_program.uniforms
        },
        value_aliases=aliases,
        contiguous_plan=contiguous_plan,
        phase_sources=phase_sources,
        specialized_values={
            int(value_id): value
            for value_id, value in (specialized_values or {}).items()
        },
        instrumentation=bool(instrumentation),
        device_resident=bool(device_resident),
        local_size=int(local_size),
        stream_publications=_control_stream_publications(
            control_program.root
        ),
        stream_outputs={
            str(name): int(stream_id)
            for name, stream_id in (stream_outputs or {}).items()
        },
        stream_continuation_count=_stream_continuation_count(
            control_program.root
        ),
        slot_contract_diagnostics={
            value_id: tuple(rows)
            for value_id, rows in slot_contract_diagnostics.items()
        },
        snippet_diagnostics=tuple(snippet_diagnostics),
        phase_cache_identities=phase_cache_identities,
        workgroup_loop_bounds=workgroup_loop_bounds,
        c_dispatch_loop_bounds=c_dispatch_loop_bounds,
        private_value_capacities=private_value_capacities,
    )


def _indent_control_shader_body(lines: Iterable[str]) -> tuple[str, ...]:
    return tuple("    " + line if line else line for line in lines)


def emit_native_for_loop(
    body: Iterable[str],
    *,
    induction: str,
    start: str | int,
    stop: str | int,
    step: str | int = 1,
    indent: int = 4,
) -> tuple[str, ...]:
    """Wrap an already-lowered GLSL region in one device-native loop."""

    name = str(induction)
    if not name.isidentifier():
        raise ValueError(f"invalid GLSL loop induction name {name!r}")
    step_text = str(step)
    try:
        numeric_step = int(step_text)
    except ValueError:
        comparison = "<"
    else:
        if numeric_step == 0:
            raise ValueError("GLSL loop step cannot be zero")
        comparison = "<" if numeric_step > 0 else ">"
    prefix = " " * int(indent)
    nested = prefix + "    "
    return (
        prefix
        + f"for (int {name} = int({start}); "
        + f"{name} {comparison} int({stop}); "
        + f"{name} += int({step_text})) {{",
        *(
            nested + line if line else line
            for line in body
        ),
        prefix + "}",
    )


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



def program_snippet(
    program: FusedProgram,
    *,
    scalar_feeds: Iterable[int] = (),
    feed_shapes: Mapping[int, Sequence[int]] | None = None,
    output_shape: Sequence[int] | None = None,
    allow_multiple_outputs: bool = False,
    base: int = 0,
) -> ShaderSnippet:
    """Contribute a straight-line elementwise region to the program.

    Every intermediate stays a register; only the region's feeds and published
    results touch the arena.
    """

    feed_ids, outputs = _validate_program_outputs(program)
    if not allow_multiple_outputs and len(outputs) != 1:
        raise ValueError(
            "elementwise fused backends require exactly one output"
        )
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

    lines: list[str] = []
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
            lines.extend(line.strip() for line in index_lines)
        else:
            index = "gid"
        lines.append(
            f"{_glsl_type(value_dtypes[feed_id])} s{i} = "
            f"{_arena_read(value_dtypes[feed_id], i, index, base)};"
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
            raise ValueError(
                f"binary op {step.op_name!r} has no right operand"
            )
        inferred_dtype = _result_dtype(op, left_dtype, right_dtype)
        result_dtype = metadata_dtype(step.result_id, inferred_dtype)
        helper, expression = _typed_expr(
            op, a, b, reverse, left_dtype, right_dtype, result_dtype,
        )
        if helper and helper not in helpers:
            helpers.append(helper)
        value_names[step.result_id] = f"s{index}"
        value_dtypes[step.result_id] = result_dtype
        lines.append(f"{_glsl_type(result_dtype)} s{index} = {expression};")

    for output_index, (name, output_id) in enumerate(outputs):
        output_dtype = output_dtypes[name]
        output_value = value_names[output_id]
        if value_dtypes[output_id] != output_dtype:
            output_value = f"{_glsl_type(output_dtype)}({output_value})"
        slot = len(feed_ids) + output_index
        lines.append(
            _arena_write(output_dtype, slot, "gid", output_value, base) + ";"
        )

    return ShaderSnippet(
        lines=tuple(lines),
        slots=len(feed_ids) + len(outputs),
        helpers=tuple(helpers),
    )


def _emit_program_source(
    program: FusedProgram,
    local_size: int = _LOCAL_SIZE,
    *,
    scalar_feeds: Iterable[int] = (),
    feed_shapes: Mapping[int, Sequence[int]] | None = None,
    output_shape: Sequence[int] | None = None,
    allow_multiple_outputs: bool = False,
) -> str:
    """Finish a program containing one elementwise region."""

    return compose_shader(
        [program_snippet(
            program,
            scalar_feeds=scalar_feeds,
            feed_shapes=feed_shapes,
            output_shape=output_shape,
            allow_multiple_outputs=allow_multiple_outputs,
        )],
        local_size=local_size,
    )


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
            "    if ((r != 0) && ((x < 0) != (y < 0))) {\n"
            "        q -= 1;\n"
            "    }\n"
            "    return q;\n"
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


def _arena_read(dtype: Any, slot: int, index: str, base: int = 0) -> str:
    """Read one value's element out of the arena, reinterpreting the word."""

    kind = _normalize_dtype(dtype).kind
    word = f"arena[u_slot[{slot + base}] + ({index})]"
    if kind == "f":
        return f"uintBitsToFloat({word})"
    if kind == "u" or kind == "b":
        return word
    return f"int({word})"


def _arena_write(
    dtype: Any, slot: int, index: str, value: str, base: int = 0
) -> str:
    """Store one value's element into the arena as a raw word."""

    kind = _normalize_dtype(dtype).kind
    word = f"arena[u_slot[{slot + base}] + ({index})]"
    if kind == "f":
        return f"{word} = floatBitsToUint({value})"
    if kind == "u" or kind == "b":
        return f"{word} = uint({value})"
    return f"{word} = uint({value})"


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
    if _shape_product(input_shape) == _shape_product(output_shape):
        # AbstractTensor has already established the result shape.  Equal
        # extents require no broadcast reconstruction: corresponding values
        # have the same linear resident index even when their logical ranks
        # differ (notably ``(1,)`` and ``()``).  Do not second-guess that
        # recorded shape with a stricter compiler-only broadcasting rule.
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



def primitive_snippet(
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
    base: int = 0,
) -> ShaderSnippet:
    """Contribute one typed primitive operation to the program being built."""

    feeds: list[tuple[str, Any, Sequence[int]]] = []
    index_lines: list[str] = []
    if left_scalar is None:
        feeds.append(("lhs", left_dtype, left_shape))
        left_lines, left_index = _broadcast_index_source(
            "lhs", left_shape, out_shape
        )
        index_lines.extend(left_lines)
        a = _arena_read(left_dtype, len(feeds) - 1, left_index, base)
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
        b = _arena_read(right_dtype, len(feeds) - 1, right_index, base)
    else:
        b = _glsl_literal(right_scalar, right_dtype)

    helper, expression = _typed_expr(
        op, a, b, reverse, left_dtype, right_dtype, out_dtype
    )
    return ShaderSnippet(
        lines=(
            *(line.strip() for line in index_lines),
            _arena_write(out_dtype, len(feeds), "gid", expression, base) + ";",
        ),
        slots=len(feeds) + 1,
        helpers=(helper.rstrip(),) if helper else (),
    )


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
    """Finish a program containing one typed primitive operation."""

    return compose_shader(
        [primitive_snippet(
            op,
            left_dtype=left_dtype,
            right_dtype=right_dtype,
            out_dtype=out_dtype,
            left_shape=left_shape,
            right_shape=right_shape,
            out_shape=out_shape,
            left_scalar=left_scalar,
            right_scalar=right_scalar,
            reverse=reverse,
        )],
        local_size=local_size,
    )


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


def where_snippet(
    condition_shape: Sequence[int],
    true_shape: Sequence[int],
    false_shape: Sequence[int],
    *,
    condition_dtype: Any,
    true_dtype: Any,
    false_dtype: Any,
    output_dtype: Any,
    output_shape: Sequence[int],
    base: int = 0,
) -> ShaderSnippet:
    """Contribute one typed, broadcast-aware conditional selection."""

    shapes = (
        tuple(condition_shape),
        tuple(true_shape),
        tuple(false_shape),
    )
    dtypes = (
        _normalize_dtype(condition_dtype),
        _normalize_dtype(true_dtype),
        _normalize_dtype(false_dtype),
    )
    output_dtype = _normalize_dtype(output_dtype)
    output_shape = tuple(output_shape)
    values = []
    lines = []
    for slot, (name, shape, dtype) in enumerate(
        zip(("condition", "if_true", "if_false"), shapes, dtypes)
    ):
        index_lines, index = _broadcast_index_source(
            name, shape, output_shape
        )
        lines.extend(line.strip() for line in index_lines)
        values.append(_arena_read(dtype, slot, index, base))
    output_type = _glsl_type(output_dtype)
    expression = (
        f"(({values[0]}) != {_glsl_literal(0, dtypes[0])} ? "
        f"{_cast_expr(values[1], output_type)} : "
        f"{_cast_expr(values[2], output_type)})"
    )
    lines.append(
        _arena_write(output_dtype, 3, "gid", expression, base) + ";"
    )
    return ShaderSnippet(lines=tuple(lines), slots=4)


def emit_where_source(
    condition_shape: Sequence[int],
    true_shape: Sequence[int],
    false_shape: Sequence[int],
    *,
    condition_dtype: Any,
    true_dtype: Any,
    false_dtype: Any,
    output_dtype: Any,
    output_shape: Sequence[int],
    local_size: int = _LOCAL_SIZE,
) -> str:
    return compose_shader(
        [where_snippet(
            condition_shape,
            true_shape,
            false_shape,
            condition_dtype=condition_dtype,
            true_dtype=true_dtype,
            false_dtype=false_dtype,
            output_dtype=output_dtype,
            output_shape=output_shape,
        )],
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



def arange_snippet(
    start: Any, step: Any, *, dtype: Any = np.float32, base: int = 0
) -> ShaderSnippet:
    """Contribute an arithmetic sequence to the program being built."""

    dtype = _normalize_dtype(dtype)
    if _normalize_dtype(dtype).kind == "b":
        raise TypeError("arange does not support boolean dtype")
    scalar_type = _glsl_type(dtype)
    value = (
        f"{_glsl_literal(start, dtype)} + {scalar_type}(gid) * "
        f"{_glsl_literal(step, dtype)}"
    )
    return ShaderSnippet(
        lines=(_arena_write(dtype, 0, "gid", value, base) + ";",),
        slots=1,
    )


def emit_arange_source(
    start: Any,
    step: Any = 1,
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Finish a program whose only operation is an arithmetic sequence."""

    return compose_shader(
        [arange_snippet(start, step, dtype=dtype)], local_size=local_size
    )


def fill_snippet(
    value: Any, *, dtype: Any = np.float32, base: int = 0
) -> ShaderSnippet:
    """Contribute a typed constant fill to the program being built."""

    dtype = _normalize_dtype(dtype)
    literal = _glsl_literal(value, dtype)
    return ShaderSnippet(
        lines=(_arena_write(dtype, 0, "gid", literal, base) + ";",),
        slots=1,
    )


def emit_fill_source(
    value: Any,
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Finish a program whose only operation is a constant fill."""

    return compose_shader(
        [fill_snippet(value, dtype=dtype)], local_size=local_size
    )



def constant_snippet(
    values: Sequence[Any], *, dtype: Any = np.float32, base: int = 0
) -> ShaderSnippet:
    """Contribute an immutable literal payload to the program being built."""

    dtype = _normalize_dtype(dtype)
    scalar_type = _glsl_type(dtype)
    literals = ", ".join(_glsl_literal(value, dtype) for value in values)
    count = len(values)
    table = f"table{base}"
    return ShaderSnippet(
        lines=(
            f"const {scalar_type} {table}[{count}] = "
            f"{scalar_type}[{count}]({literals});",
            _arena_write(dtype, 0, "gid", f"{table}[gid]", base) + ";",
        ),
        slots=1,
    )


def emit_constant_source(
    values: Sequence[Any],
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Finish a program whose only operation is a literal payload."""

    return compose_shader(
        [constant_snippet(values, dtype=dtype)], local_size=local_size
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



def expand_snippet(
    source_shape: Sequence[int],
    target_shape: Sequence[int],
    *,
    dtype: Any = np.float32,
    base: int = 0,
) -> ShaderSnippet:
    """Contribute a direct broadcast copy to the program being built."""

    source_shape, target_shape = _resolve_expand_shape(
        source_shape, target_shape
    )
    dtype = _normalize_dtype(dtype)
    index_lines, source_index = _broadcast_index_source(
        "source", source_shape, target_shape
    )
    return ShaderSnippet(
        lines=(
            *(line.strip() for line in index_lines),
            _arena_write(
                dtype, 1, "gid",
                _arena_read(dtype, 0, source_index, base), base,
            ) + ";",
        ),
        slots=2,
    )


def emit_expand_source(
    source_shape: Sequence[int],
    target_shape: Sequence[int],
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Finish a program whose only operation is a broadcast copy."""

    return compose_shader(
        [expand_snippet(source_shape, target_shape, dtype=dtype)],
        local_size=local_size,
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
# These snippets keep that specialization isolated below the common tensor
# semantics.  They preserve arbitrary rank and dtype, never read an SSBO back
# through NumPy, and leave room for later GLSL-specific improvements (subgroup
# copies, shared-memory tiling, or multi-stage plans for very large input
# lists) without changing AbstractTensor's public ``cat``/``stack`` contract.
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



def cat_snippet(
    shapes: Sequence[Sequence[int]],
    dim: int = 0,
    *,
    dtype: Any = np.float32,
    input_dtypes: Sequence[Any] | None = None,
    output_dtype: Any | None = None,
    base: int = 0,
) -> ShaderSnippet:
    """Contribute an arbitrary-rank concatenate to the program being built."""

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

    lines = [
        f"uint inner = gid % uint({after});",
        f"uint block = gid / uint({after});",
        f"uint axis_index = block % uint({output_axis});",
        f"uint outer = block / uint({output_axis});",
    ]
    prefix = 0
    for index, shape in enumerate(shapes):
        axis_size = shape[dim]
        condition = "if" if index == 0 else "else if"
        lines.extend([
            f"{condition} (axis_index < uint({prefix + axis_size})) {{",
            f"    uint local_axis = axis_index - uint({prefix});",
            f"    uint source_index = (outer * uint({axis_size}) + "
            f"local_axis) * uint({after}) + inner;",
            "    " + _arena_write(
                output_dtype, len(shapes), "gid",
                f"{output_type}("
                + _arena_read(input_dtypes[index], index, "source_index", base)
                + ")",
                base,
            ) + ";",
            "}",
        ])
        prefix += axis_size
    return ShaderSnippet(lines=tuple(lines), slots=len(shapes) + 1)


def emit_cat_source(
    shapes: Sequence[Sequence[int]],
    dim: int = 0,
    *,
    dtype: Any = np.float32,
    input_dtypes: Sequence[Any] | None = None,
    output_dtype: Any | None = None,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Finish a program whose only operation is a concatenate."""

    return compose_shader(
        [cat_snippet(
            shapes, dim, dtype=dtype,
            input_dtypes=input_dtypes, output_dtype=output_dtype,
        )],
        local_size=local_size,
    )


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



def permute_snippet(
    shape: Sequence[int],
    dims: Sequence[int],
    *,
    dtype: Any = np.float32,
    base: int = 0,
) -> ShaderSnippet:
    """Contribute a row-major axis permutation to the program being built."""

    shape, dims, output_shape = _validate_permute_layout(shape, dims)
    input_strides = _row_major_strides(shape)
    output_strides = _row_major_strides(output_shape)
    lines = ["uint remaining = gid;", "uint source_index = uint(0);"]
    for output_axis, source_axis in enumerate(dims):
        output_stride = output_strides[output_axis]
        source_stride = input_strides[source_axis]
        lines.extend([
            f"uint coord{output_axis} = remaining / uint({output_stride});",
            f"remaining %= uint({output_stride});",
            f"source_index += coord{output_axis} * uint({source_stride});",
        ])
    lines.append(
        _arena_write(
            dtype, 1, "gid",
            _arena_read(dtype, 0, "source_index", base), base,
        ) + ";"
    )
    return ShaderSnippet(lines=tuple(lines), slots=2)


def emit_permute_source(
    shape: Sequence[int],
    dims: Sequence[int],
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Finish a program whose only operation is an axis permutation."""

    return compose_shader(
        [permute_snippet(shape, dims, dtype=dtype)], local_size=local_size
    )


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



def matmul_snippet(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
    *,
    left_dtype: Any = np.float32,
    right_dtype: Any = np.float32,
    output_dtype: Any | None = None,
    local_size: int = _LOCAL_SIZE,
    base: int = 0,
) -> ShaderSnippet:
    """Contribute a cooperative tiled batched matmul to the program.

    This snippet derives its own launch geometry from the workgroup builtins
    and cooperates through shared tiles, so it opts out of the flat linear
    guard and carries its tile declarations up to file scope.
    """

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
    row_tiles = (rows + tile - 1) // tile
    column_tiles = (columns + tile - 1) // tile
    group_count = batch_count * row_tiles * column_tiles

    lines = [
        "uint group_index = gl_WorkGroupID.x",
        "    + gl_WorkGroupID.y * gl_NumWorkGroups.x",
        "    + gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y;",
        f"if (group_index < uint({group_count})) {{",
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
        lines.extend([
            f"    uint batch_coord{axis} = "
            f"batch_remaining / uint({batch_stride});",
            f"    batch_remaining %= uint({batch_stride});",
        ])
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
    lines.extend([
        f"    {output_type} total = {output_type}(0);",
        f"    for (uint tile_k = uint(0); tile_k < "
        f"uint({(inner + tile - 1) // tile}); ++tile_k) {{",
        f"        uint left_k = tile_k * uint({tile}) + local_column;",
        f"        uint right_k = tile_k * uint({tile}) + local_row;",
        f"        left_tile[local_row][local_column] = "
        f"(row < uint({rows}) && left_k < uint({inner}))",
        f"            ? {output_type}("
        + _arena_read(
            left_dtype, 0,
            f"left_offset + row * uint({inner}) + left_k", base,
        )
        + f") : {output_type}(0);",
        f"        right_tile[local_row][local_column] = "
        f"(right_k < uint({inner}) && column < uint({columns}))",
        f"            ? {output_type}("
        + _arena_read(
            right_dtype, 1,
            f"right_offset + right_k * uint({columns}) + column", base,
        )
        + f") : {output_type}(0);",
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
        "        " + _arena_write(
            output_dtype, 2, "output_index", "total", base
        ) + ";",
        "    }",
        "}",
    ])
    return ShaderSnippet(
        lines=tuple(lines),
        slots=3,
        shared=(
            f"shared {output_type} left_tile[{tile}][{tile}];",
            f"shared {output_type} right_tile[{tile}][{tile}];",
        ),
        guard=False,
    )


def emit_matmul_source(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
    *,
    left_dtype: Any = np.float32,
    right_dtype: Any = np.float32,
    output_dtype: Any | None = None,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Finish a program whose only operation is a batched matmul."""

    return compose_shader(
        [matmul_snippet(
            left_shape, right_shape,
            left_dtype=left_dtype, right_dtype=right_dtype,
            output_dtype=output_dtype, local_size=local_size,
        )],
        local_size=local_size,
    )


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



def repeat_snippet(
    source_shape: Sequence[int],
    repeats: Any,
    dim: int = 0,
    *,
    dtype: Any = np.float32,
    base: int = 0,
) -> ShaderSnippet:
    """Contribute a tile/repeat to the program being built."""

    source_shape, _, output_shape = _resolve_repeat_layout(
        source_shape, repeats, dim
    )
    source_strides = _row_major_strides(source_shape)
    output_strides = _row_major_strides(output_shape)
    lines = ["uint remaining = gid;", "uint source_index = uint(0);"]
    for axis, (source_size, source_stride, output_stride) in enumerate(
        zip(source_shape, source_strides, output_strides)
    ):
        lines.extend([
            f"uint coord{axis} = remaining / uint({output_stride});",
            f"remaining %= uint({output_stride});",
            f"source_index += (coord{axis} % uint({source_size})) "
            f"* uint({source_stride});",
        ])
    lines.append(
        _arena_write(
            dtype, 1, "gid",
            _arena_read(dtype, 0, "source_index", base), base,
        ) + ";"
    )
    return ShaderSnippet(lines=tuple(lines), slots=2)


def emit_repeat_source(
    source_shape: Sequence[int],
    repeats: Any,
    dim: int = 0,
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Finish a program whose only operation is a tile/repeat."""

    return compose_shader(
        [repeat_snippet(source_shape, repeats, dim, dtype=dtype)],
        local_size=local_size,
    )



def gather_snippet(
    *, dtype: Any = np.float32, base: int = 0
) -> ShaderSnippet:
    """Contribute an arbitrary-offset gather to the program being built."""

    return ShaderSnippet(
        lines=(
            _arena_write(
                dtype, 2, "gid",
                _arena_read(
                    dtype, 0,
                    "uint(" + _arena_read(np.int32, 1, "gid", base) + ")",
                    base,
                ),
                base,
            ) + ";",
        ),
        slots=3,
    )


def emit_gather_source(
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Finish a program whose only operation is a gather."""

    return compose_shader(
        [gather_snippet(dtype=dtype)], local_size=local_size
    )



def scatter_snippet(
    base_shape: Sequence[int],
    index_shape: Sequence[int],
    dim: int,
    *,
    dtype: Any = np.float32,
    base: int = 0,
) -> ShaderSnippet:
    """Contribute a resident scatter-add to the program being built."""

    base_shape = tuple(int(size) for size in base_shape)
    index_shape = tuple(int(size) for size in index_shape)
    dim = int(dim)
    if dim < 0:
        dim += len(base_shape)
    if len(index_shape) != len(base_shape):
        raise ValueError(
            "captured scatter requires index rank to match base rank"
        )
    scalar_type = _glsl_type(dtype)
    base_strides = tuple(
        _shape_product(base_shape[axis + 1:])
        for axis in range(len(base_shape))
    )
    index_strides = tuple(
        _shape_product(index_shape[axis + 1:])
        for axis in range(len(index_shape))
    )
    target_terms = []
    for axis, stride in enumerate(base_strides):
        if axis == dim:
            target_terms.append(
                "(uint(" + _arena_read(np.int32, 1, "k", base)
                + f") * {stride}u)"
            )
        else:
            target_terms.append(
                f"(((k / {index_strides[axis]}u) % "
                f"{index_shape[axis]}u) * {stride}u)"
            )
    target = " + ".join(target_terms) or "0u"
    return ShaderSnippet(
        lines=(
            f"{scalar_type} accumulated = "
            + _arena_read(dtype, 0, "gid", base) + ";",
            f"for (uint k = 0u; k < {_shape_product(index_shape)}u; ++k) {{",
            f"    uint target = {target};",
            "    if (target == gid) { accumulated += "
            + _arena_read(dtype, 2, "k", base) + "; }",
            "}",
            _arena_write(dtype, 3, "gid", "accumulated", base) + ";",
        ),
        slots=4,
    )


def emit_scatter_source(
    base_shape: Sequence[int],
    index_shape: Sequence[int],
    dim: int,
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Finish a program whose only operation is a scatter-add."""

    return compose_shader(
        [scatter_snippet(base_shape, index_shape, dim, dtype=dtype)],
        local_size=local_size,
    )



def topk_offsets_snippet(
    shape: Sequence[int],
    k: int,
    dim: int,
    *,
    dtype: Any = np.float32,
    base: int = 0,
) -> ShaderSnippet:
    """Contribute a deterministic top-k offset selection to the program.

    Each invocation owns one output rank in one axis slice, performs a small
    repeated selection locally, and writes the selected flat source offset.
    Values are then obtained through the ordinary resident gather, keeping
    selection and transport separate reusable primitives.
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
        raise ValueError(
            "topk k must be between one and the selected axis size"
        )
    dtype = _normalize_dtype(dtype)
    scalar_type = _glsl_type(dtype)
    inner = _shape_product(shape[dim + 1:])
    if dtype.kind == "f":
        nan_order = (
            "        bool candidate_nan = isnan(candidate);\n"
            "        bool best_nan = isnan(best);\n"
            "        bool better = !found\n"
            "            || (candidate_nan && !best_nan)\n"
            "            || (candidate_nan == best_nan\n"
            "                && (candidate > best\n"
            "                    || (candidate == best\n"
            "                        && axis_index < best_axis)));"
        )
    else:
        nan_order = (
            "        bool better = !found || candidate > best\n"
            "            || (candidate == best && axis_index < best_axis);"
        )
    return ShaderSnippet(
        lines=(
            f"uint inner_index = gid % uint({inner});",
            f"uint slot = gid / uint({inner});",
            f"uint rank = slot % uint({k});",
            f"uint outer = slot / uint({k});",
            f"uint base_offset = outer * uint({axis_size * inner}) "
            "+ inner_index;",
            f"uint chosen[{k}];",
            "uint best_axis = uint(0);",
            "for (uint selection = uint(0); selection <= rank; ++selection) {",
            "    bool found = false;",
            f"    {scalar_type} best = {scalar_type}(0);",
            f"    for (uint axis_index = uint(0); axis_index < "
            f"uint({axis_size}); ++axis_index) {{",
            "        bool used = false;",
            "        for (uint prior = uint(0); prior < selection; ++prior) {",
            "            used = used || chosen[prior] == axis_index;",
            "        }",
            "        if (used) { continue; }",
            f"        {scalar_type} candidate = "
            + _arena_read(
                dtype, 0, f"base_offset + axis_index * uint({inner})", base
            ) + ";",
            nan_order,
            "        if (better) {",
            "            found = true;",
            "            best = candidate;",
            "            best_axis = axis_index;",
            "        }",
            "    }",
            "    chosen[selection] = best_axis;",
            "}",
            _arena_write(
                np.int32, 1, "gid",
                f"int(base_offset + best_axis * uint({inner}))", base,
            ) + ";",
        ),
        slots=2,
    )


def emit_topk_offsets_source(
    shape: Sequence[int],
    k: int,
    dim: int,
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Finish a program whose only operation is a top-k offset selection."""

    return compose_shader(
        [topk_offsets_snippet(shape, k, dim, dtype=dtype)],
        local_size=local_size,
    )



def index_assign_snippet(
    *,
    dtype: Any = np.float32,
    index_dtype: Any = np.int32,
    scalar_value: bool = False,
    base: int = 0,
) -> ShaderSnippet:
    """Contribute an arbitrary-offset assignment to the program being built."""

    value_index = "uint(0)" if scalar_value else "gid"
    return ShaderSnippet(
        lines=(
            _arena_write(
                dtype, 2,
                "uint(" + _arena_read(index_dtype, 0, "gid", base) + ")",
                _arena_read(dtype, 1, value_index, base),
                base,
            ) + ";",
        ),
        slots=3,
    )


def emit_index_assign_source(
    *,
    dtype: Any = np.float32,
    index_dtype: Any = np.int32,
    scalar_value: bool = False,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Finish a program whose only operation is an indexed assignment."""

    return compose_shader(
        [index_assign_snippet(
            dtype=dtype, index_dtype=index_dtype, scalar_value=scalar_value
        )],
        local_size=local_size,
    )


def index_select_snippet(
    shape: Sequence[int],
    dim: int,
    index_count: int,
    *,
    dtype: Any = np.float32,
    index_dtype: Any = np.int32,
    base: int = 0,
) -> ShaderSnippet:
    """Contribute a shaped integer selection to the program being built."""

    shape = tuple(int(size) for size in shape)
    dim = int(dim) % len(shape)
    axis_size = shape[dim]
    after = _shape_product(shape[dim + 1:])
    read_index = _arena_read(index_dtype, 1, "index_position", base)
    selected_lines = (
        [
            f"int selected = {read_index};",
            f"if (selected < 0) {{ selected += int({axis_size}); }}",
            "uint selected_index = uint(selected);",
        ]
        if _normalize_dtype(index_dtype).kind == "i"
        else [f"uint selected_index = uint({read_index});"]
    )
    return ShaderSnippet(
        lines=(
            f"uint inner = gid % uint({after});",
            f"uint block = gid / uint({after});",
            f"uint index_position = block % uint({index_count});",
            f"uint outer = block / uint({index_count});",
            *selected_lines,
            f"uint source_index = (outer * uint({axis_size}) + "
            f"selected_index) * uint({after}) + inner;",
            _arena_write(
                dtype, 2, "gid",
                _arena_read(dtype, 0, "source_index", base), base,
            ) + ";",
        ),
        slots=3,
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
    """Finish a program whose only operation is an index selection."""

    return compose_shader(
        [index_select_snippet(
            shape, dim, index_count, dtype=dtype, index_dtype=index_dtype
        )],
        local_size=local_size,
    )


def slice_axis_snippet(
    shape: Sequence[int],
    dim: int,
    start: int,
    step: int,
    count: int,
    *,
    dtype: Any = np.float32,
    base: int = 0,
) -> ShaderSnippet:
    """Contribute an affine single-axis slice to the program being built."""

    shape = tuple(int(size) for size in shape)
    dim = int(dim) % len(shape)
    axis_size = shape[dim]
    after = _shape_product(shape[dim + 1:])
    return ShaderSnippet(
        lines=(
            f"uint inner = gid % uint({after});",
            f"uint block = gid / uint({after});",
            f"uint selected_position = block % uint({count});",
            f"uint outer = block / uint({count});",
            f"int selected = int({start}) + "
            f"int(selected_position) * int({step});",
            f"uint source_index = (outer * uint({axis_size}) + "
            f"uint(selected)) * uint({after}) + inner;",
            _arena_write(
                dtype, 1, "gid",
                _arena_read(dtype, 0, "source_index", base), base,
            ) + ";",
        ),
        slots=2,
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
    """Finish a program whose only operation is an affine slice."""

    return compose_shader(
        [slice_axis_snippet(shape, dim, start, step, count, dtype=dtype)],
        local_size=local_size,
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



def reduce_snippet(
    op: str,
    shape: Sequence[int],
    dim: int | None = None,
    keepdim: bool = False,
    *,
    dtype: Any = np.float32,
    base: int = 0,
) -> ShaderSnippet:
    """Contribute an axis reduction to the program being built."""

    if op not in {"sum", "mean", "min", "max", "any", "all"}:
        raise ValueError(f"unsupported GLSL reduction {op!r}")
    source_shape, axis, _, extent = _reduce_layout(shape, dim, keepdim)
    input_dtype = _normalize_dtype(dtype)
    output_dtype = _reduction_dtype(op, input_dtype)
    output_type = _glsl_type(output_dtype)
    after = _shape_product(source_shape[axis + 1:])

    if op in {"min", "max"} and extent == 0:
        raise ValueError(f"{op} reduction has no identity for an empty axis")
    initial = "uint(1)" if op == "all" else f"{output_type}(0)"

    src = _arena_read(input_dtype, 0, "source_index", base)
    value = f"{output_type}({src})"
    if op == "sum":
        update = f"total += {value};"
    elif op == "mean":
        update = f"total += float({src});"
    elif op == "min":
        update = f"total = (k == uint(0)) ? {value} : min(total, {value});"
    elif op == "max":
        update = f"total = (k == uint(0)) ? {value} : max(total, {value});"
    elif op == "any":
        update = f"total |= uint({src} != 0);"
    else:
        update = f"total &= uint({src} != 0);"

    final = f"total / float({extent})" if op == "mean" and extent else "total"
    return ShaderSnippet(
        lines=(
            f"uint inner = gid % uint({after});",
            f"uint outer = gid / uint({after});",
            f"uint line_base = outer * uint({extent * after}) + inner;",
            f"{output_type} total = {initial};",
            f"for (uint k = uint(0); k < uint({extent}); ++k) {{",
            f"    uint source_index = line_base + k * uint({after});",
            f"    {update}",
            "}",
            _arena_write(output_dtype, 1, "gid", final, base) + ";",
        ),
        slots=2,
    )


def emit_reduce_source(
    op: str,
    shape: Sequence[int],
    dim: int | None = None,
    keepdim: bool = False,
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Finish a program whose only operation is an axis reduction."""

    return compose_shader(
        [reduce_snippet(op, shape, dim, keepdim, dtype=dtype)],
        local_size=local_size,
    )



def cumsum_snippet(
    shape: Sequence[int],
    dim: int = 0,
    *,
    dtype: Any = np.float32,
    base: int = 0,
) -> ShaderSnippet:
    """Contribute one bounded prefix result per output invocation.

    Every invocation owns exactly ``output[gid]``.  Treating ``gid`` as a
    whole axis line while dispatching the full output extent makes each
    invocation write ``extent`` words and overruns the resident allocation by
    that factor.  Recover the output element's axis coordinate instead, scan
    only through that coordinate, and perform one in-range write.
    """

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
    output_type = _glsl_type(output_dtype)
    extent = shape[dim]
    after = _shape_product(shape[dim + 1:])
    if extent == 0:
        return ShaderSnippet(lines=(), slots=2)
    return ShaderSnippet(
        lines=(
            f"uint inner = gid % uint({after});",
            f"uint block = gid / uint({after});",
            f"uint axis_index = block % uint({extent});",
            f"uint outer = block / uint({extent});",
            f"uint line_base = outer * uint({extent * after}) + inner;",
            f"{output_type} total = {output_type}(0);",
            "for (uint k = uint(0); k <= axis_index; ++k) {",
            f"    uint source_index = line_base + k * uint({after});",
            f"    total += {output_type}("
            + _arena_read(input_dtype, 0, "source_index", base) + ");",
            "}",
            _arena_write(output_dtype, 1, "gid", "total", base) + ";",
        ),
        slots=2,
    )


def emit_cumsum_source(
    shape: Sequence[int],
    dim: int = 0,
    *,
    dtype: Any = np.float32,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Finish a program whose only operation is a prefix sum."""

    return compose_shader(
        [cumsum_snippet(shape, dim, dtype=dtype)], local_size=local_size
    )


def stack_snippet(
    shape: Sequence[int],
    input_count: int,
    dim: int = 0,
    *,
    dtype: Any = np.float32,
    input_dtypes: Sequence[Any] | None = None,
    output_dtype: Any | None = None,
    base: int = 0,
) -> ShaderSnippet:
    """Contribute an arbitrary-rank stack to the program being built."""

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

    lines = [
        f"uint inner = gid % uint({after});",
        f"uint block = gid / uint({after});",
        f"uint source_number = block % uint({input_count});",
        f"uint outer = block / uint({input_count});",
        f"uint source_index = outer * uint({after}) + inner;",
    ]
    for index in range(input_count):
        condition = "if" if index == 0 else "else if"
        lines.extend([
            f"{condition} (source_number == uint({index})) {{",
            "    " + _arena_write(
                output_dtype, input_count, "gid",
                f"{output_type}("
                + _arena_read(input_dtypes[index], index, "source_index", base)
                + ")",
                base,
            ) + ";",
            "}",
        ])
    return ShaderSnippet(lines=tuple(lines), slots=input_count + 1)


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
    """Finish a program whose only operation is a stack."""

    return compose_shader(
        [stack_snippet(
            shape, input_count, dim, dtype=dtype,
            input_dtypes=input_dtypes, output_dtype=output_dtype,
        )],
        local_size=local_size,
    )


# ---------------------------------------------------------------------------
# compilation + cache
# ---------------------------------------------------------------------------

_program_cache: dict[tuple[int, str], int] = {}
_uniform_location_cache: dict[tuple[int, str], int] = {}
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "persistent_hits": 0,
    "persistent_misses": 0,
    "persistent_writes": 0,
}
_dispatch_stats = {"calls": 0, "work_items": 0}
_PROGRAM_BINARY_MAGIC = b"TURGLSL1"
_c_dispatch_planner_cache: dict[
    tuple[tuple[str, str, str], tuple[str, ...]], Any
] = {}


def _c_dispatch_expression_names(
    bounds: tuple[str, str, str],
) -> tuple[str, ...]:
    """Validate compiler-generated integer expressions and list their inputs."""

    names: set[str] = set()
    allowed = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Load,
    )
    for source in bounds:
        expression = ast.parse(str(source), mode="eval")
        for node in ast.walk(expression):
            if not isinstance(node, allowed):
                raise ValueError(
                    "C dispatch loop bound contains unsupported syntax: "
                    f"{source!r} ({type(node).__name__})"
                )
            if isinstance(node, ast.Name):
                names.add(str(node.id))
    return tuple(sorted(names))


def _compile_c_dispatch_planner(
    bounds: tuple[str, str, str],
    uniform_names: Iterable[str],
):
    """Compile the retained loop itself as a C command-producing shell."""

    names = _c_dispatch_expression_names(bounds)
    unknown = set(names) - set(map(str, uniform_names))
    if unknown:
        raise ValueError(
            "C dispatch loop references unavailable controls: "
            + ", ".join(sorted(unknown))
        )
    key = (tuple(map(str, bounds)), names)
    cached = _c_dispatch_planner_cache.get(key)
    if cached is not None:
        return cached

    from cffi import FFI

    start, stop, step = map(str, bounds)
    parameters = ", ".join(
        [*(f"int {name}" for name in names), "int *commands", "size_t capacity"]
    )
    declaration = ", ".join(
        [*("int" for _name in names), "int *", "size_t"]
    )
    source = f"""
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
size_t turing_plan_dispatch({parameters}) {{
    const int turing_start = (int)({start});
    const int turing_stop = (int)({stop});
    const int turing_step = (int)({step});
    size_t count = 0;
    if (turing_step == 0) return SIZE_MAX;
    for (
        int iteration = turing_start;
        turing_step > 0 ? iteration < turing_stop : iteration > turing_stop;
        iteration += turing_step
    ) {{
        if (commands != NULL && count < capacity) commands[count] = iteration;
        if (count == (size_t)INT_MAX) return SIZE_MAX;
        ++count;
    }}
    return count;
}}
"""
    ffi = FFI()
    ffi.cdef(
        f"size_t turing_plan_dispatch({declaration});"
    )
    library = ffi.verify(source)
    planner = (ffi, library.turing_plan_dispatch, names)
    _c_dispatch_planner_cache[key] = planner
    return planner


def _c_dispatch_iterations(
    bounds: tuple[str, str, str],
    uniforms: Mapping[str, int],
) -> tuple[int, ...]:
    ffi, planner, names = _compile_c_dispatch_planner(
        bounds, uniforms
    )
    arguments = tuple(int(uniforms[name]) for name in names)
    count = int(planner(*arguments, ffi.NULL, 0))
    if count == int(ffi.cast("size_t", -1)):
        raise ValueError("C dispatch loop has a zero step or excessive extent")
    commands = ffi.new("int[]", max(1, count))
    written = int(planner(*arguments, commands, count))
    if written != count:
        raise RuntimeError(
            f"C dispatch planner changed extent: {count} -> {written}"
        )
    return tuple(int(commands[index]) for index in range(count))


def _persistent_shader_cache_directory() -> Path:
    configured = os.environ.get("TURING_GLSL_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    from .artifact_cache import repository_cache_root

    return repository_cache_root() / "glsl-programs"


def _program_binary_cache_path(GL, cache_identity: str) -> Path | None:
    try:
        if int(GL.glGetIntegerv(GL.GL_NUM_PROGRAM_BINARY_FORMATS)) <= 0:
            return None
        driver = b"\0".join(
            bytes(GL.glGetString(token) or b"")
            for token in (GL.GL_VENDOR, GL.GL_RENDERER, GL.GL_VERSION)
        )
    except Exception:
        return None
    driver_digest = hashlib.sha256(driver).hexdigest()[:24]
    return (
        _persistent_shader_cache_directory()
        / driver_digest
        / f"{cache_identity}.bin"
    )


def _load_program_binary(GL, path: Path) -> int | None:
    program = None
    try:
        payload = path.read_bytes()
        header_size = struct.calcsize("<8sI")
        magic, binary_format = struct.unpack(
            "<8sI", payload[:header_size]
        )
        binary = payload[header_size:]
        if magic != _PROGRAM_BINARY_MAGIC or not binary:
            raise ValueError("invalid persistent GLSL program cache entry")
        program = GL.glCreateProgram()
        GL.glProgramBinary(
            program,
            int(binary_format),
            binary,
            len(binary),
        )
        if bool(GL.glGetProgramiv(program, GL.GL_LINK_STATUS)):
            return int(program)
        GL.glDeleteProgram(program)
    except Exception:
        if program is not None:
            try:
                GL.glDeleteProgram(program)
            except Exception:
                pass
        # A program binary is an optional driver-specific cache artifact.
        # Corruption or driver rejection must become a cache miss, not prevent
        # recompilation from authoritative GLSL source.
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    return None


def _store_program_binary(GL, program: int, path: Path) -> bool:
    try:
        length = int(
            GL.glGetProgramiv(program, GL.GL_PROGRAM_BINARY_LENGTH)
        )
        if length <= 0:
            return False
        written = ctypes.c_int()
        binary_format = ctypes.c_uint()
        binary = (ctypes.c_ubyte * length)()
        GL.glGetProgramBinary(
            program,
            length,
            ctypes.byref(written),
            ctypes.byref(binary_format),
            binary,
        )
        payload = (
            struct.pack(
                "<8sI",
                _PROGRAM_BINARY_MAGIC,
                int(binary_format.value),
            )
            + bytes(binary[:written.value])
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_bytes(payload)
        os.replace(temporary, path)
        return True
    except Exception:
        return False


def _write_shader_cache_manifest(
    binary_path: Path | None,
    fields: Mapping[str, Any],
) -> None:
    if binary_path is None:
        return
    path = binary_path.with_suffix(".json")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(dict(fields), sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError):
        pass


@dataclass(frozen=True)
class _DeferredElementwise:
    """One not-yet-dispatched GLSL expression region and its concrete feeds."""

    program: FusedProgram
    feeds: Mapping[int, GLChunk]


_fusion_depth: ContextVar[int] = ContextVar("glsl_fusion_depth", default=0)
_deferred_value_ids = itertools.count(start=-1, step=-1)


@dataclass
class _DispatchBatchState:
    """Mutable state for one lock-free sequence of dependent GL launches."""

    max_bindings: int = 0
    depth: int = 1


_dispatch_batch_state: ContextVar[_DispatchBatchState | None] = ContextVar(
    "glsl_dispatch_batch_state",
    default=None,
)


@contextmanager
def dispatch_batch():
    """Submit a staged compute graph with one GL state/error boundary.

    Individual launches retain their memory barriers, so dependent kernels see
    prior writes exactly as they do outside this scope. The scope only removes
    redundant program resets, SSBO unbinding, and driver error polling between
    known stages. Nested callers share the outer batch.
    """

    existing = _dispatch_batch_state.get()
    if existing is not None:
        existing.depth += 1
        try:
            yield
        finally:
            existing.depth -= 1
        return

    require_gl_context()
    from OpenGL import GL

    state = _DispatchBatchState()
    token = _dispatch_batch_state.set(state)
    checker = getattr(GL.glUseProgram, "error_checker", None)
    previous_checker = None
    if checker is not None and hasattr(checker, "_currentChecker"):
        previous_checker = checker._currentChecker
        checker._currentChecker = checker.nullGetError
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        try:
            for binding in range(state.max_bindings):
                GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, binding, 0)
            GL.glUseProgram(0)
        finally:
            if previous_checker is not None:
                checker._currentChecker = previous_checker
            _dispatch_batch_state.reset(token)
    if succeeded:
        error = int(GL.glGetError())
        if error != int(GL.GL_NO_ERROR):
            raise RuntimeError(
                f"OpenGL error 0x{error:04X} during compute dispatch batch"
            )


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


def dispatch_stats(*, reset: bool = False) -> dict[str, int]:
    """Return physical compute-launch totals for transparent live profiling."""

    snapshot = dict(_dispatch_stats)
    if reset:
        _dispatch_stats.update(calls=0, work_items=0)
    return snapshot


def _compile(
    source: str,
    *,
    cache_identity: str | None = None,
) -> int:
    """Compile+link a compute shader using planner semantics as cache identity."""
    require_gl_context()
    from OpenGL import GL
    from OpenGL import platform

    context = int(platform.PLATFORM.GetCurrentContext() or 0)
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    identity = str(cache_identity or source_digest)
    key = (context, identity)
    cached = _program_cache.get(key)
    if cached is not None and bool(GL.glIsProgram(int(cached))):
        _cache_stats["hits"] += 1
        return int(cached)
    if cached is not None:
        # A context teardown, or older code which deleted a borrowed cached
        # program, may leave a numeric name behind.  Never hand that stale name
        # to glUseProgram.
        _program_cache.pop(key, None)
    _cache_stats["misses"] += 1
    binary_path = _program_binary_cache_path(GL, identity)
    cache_trace = os.environ.get("TURING_GLSL_CACHE_TRACE") == "1"
    trace_entry = bool(
        cache_trace
        and (cache_identity is not None or len(source) >= 100_000)
    )
    manifest = {
        "cache_identity": identity,
        "source_sha256": source_digest,
        "semantic": cache_identity is not None,
        "source_characters": len(source),
        "source_lines": source.count("\n") + 1,
        "pid": os.getpid(),
        "updated_unix": time.time(),
    }
    if (
        binary_path is not None
        and (
            os.environ.get("TURING_GLSL_CACHE_SOURCES") == "1"
            or len(source) >= 100_000
        )
    ):
        try:
            binary_path.parent.mkdir(parents=True, exist_ok=True)
            binary_path.with_suffix(".glsl").write_text(
                source,
                encoding="utf-8",
            )
        except OSError:
            # Source dumps are diagnostics.  Cache permissions must not turn a
            # valid shader into a compilation failure.
            pass
        _write_shader_cache_manifest(
            binary_path,
            dict(manifest, status="source-ready"),
        )
    if binary_path is not None and binary_path.is_file():
        program = _load_program_binary(GL, binary_path)
        if program is not None:
            _cache_stats["persistent_hits"] += 1
            _write_shader_cache_manifest(
                binary_path,
                dict(
                    manifest,
                    status="ready",
                    result="persistent-hit",
                    updated_unix=time.time(),
                ),
            )
            if trace_entry:
                print(
                    "[glsl-cache] persistent hit "
                    f"identity={identity} source={source_digest}",
                    flush=True,
                )
            _program_cache[key] = program
            return program
    if binary_path is not None:
        _cache_stats["persistent_misses"] += 1
        if trace_entry:
            print(
                "[glsl-cache] persistent miss "
                f"identity={identity} source={source_digest}",
                flush=True,
            )

    compile_started = time.perf_counter()
    _write_shader_cache_manifest(
        binary_path,
        dict(
            manifest,
            status="compiling",
            started_unix=time.time(),
        ),
    )
    if trace_entry:
        print(
            "[glsl-cache] driver compile begin "
            f"identity={identity} chars={len(source)} "
            f"lines={source.count(chr(10)) + 1}",
            flush=True,
        )
    shader = GL.glCreateShader(GL.GL_COMPUTE_SHADER)
    GL.glShaderSource(shader, source)
    _write_shader_cache_manifest(
        binary_path,
        dict(
            manifest,
            status="compiling",
            stage="shader-compile",
            started_unix=time.time(),
        ),
    )
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader)
        GL.glDeleteShader(shader)
        _write_shader_cache_manifest(
            binary_path,
            dict(
                manifest,
                status="failed",
                stage="compile",
                elapsed_seconds=(
                    time.perf_counter() - compile_started
                ),
                driver_log=str(log)[:4096],
                updated_unix=time.time(),
            ),
        )
        raise GLSLCompileError(_annotate(source, log))

    program = GL.glCreateProgram()
    if binary_path is not None:
        GL.glProgramParameteri(
            program,
            GL.GL_PROGRAM_BINARY_RETRIEVABLE_HINT,
            GL.GL_TRUE,
        )
    GL.glAttachShader(program, shader)
    _write_shader_cache_manifest(
        binary_path,
        dict(
            manifest,
            status="compiling",
            stage="program-link",
            started_unix=time.time(),
            elapsed_seconds=time.perf_counter() - compile_started,
        ),
    )
    GL.glLinkProgram(program)
    GL.glDeleteShader(shader)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(program)
        GL.glDeleteProgram(program)
        _write_shader_cache_manifest(
            binary_path,
            dict(
                manifest,
                status="failed",
                stage="link",
                elapsed_seconds=(
                    time.perf_counter() - compile_started
                ),
                driver_log=str(log)[:4096],
                updated_unix=time.time(),
            ),
        )
        raise GLSLCompileError(_annotate(source, log))

    _program_cache[key] = program
    binary_written = bool(
        binary_path is not None
        and _store_program_binary(GL, program, binary_path)
    )
    if binary_written:
        _cache_stats["persistent_writes"] += 1
        if trace_entry:
            print(
                "[glsl-cache] persistent write "
                f"identity={identity} source={source_digest}",
                flush=True,
            )
    elapsed_seconds = time.perf_counter() - compile_started
    _write_shader_cache_manifest(
        binary_path,
        dict(
            manifest,
            status="ready",
            result="compiled",
            elapsed_seconds=elapsed_seconds,
            binary_written=binary_written,
            updated_unix=time.time(),
        ),
    )
    if trace_entry:
        print(
            "[glsl-cache] driver compile complete "
            f"identity={identity} seconds={elapsed_seconds:.3f}",
            flush=True,
        )
    return program


def compile_glsl_source(
    source: str,
    *,
    cache_identity: str | None = None,
) -> int:
    """Compile one fully composed GLSL compute-shader source."""

    return _compile(source, cache_identity=cache_identity)


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
    ssbo_offset_alignment: int = 1

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
        ssbo_offset_alignment=_first_gl_integer(
            GL.glGetIntegerv(GL.GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT)
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


def _bind_chunk(binding: int, chunk: GLChunk) -> None:
    """Bind a whole buffer or an aligned logical subrange."""

    from OpenGL import GL

    if chunk._offset:
        GL.glBindBufferRange(
            GL.GL_SHADER_STORAGE_BUFFER,
            binding,
            chunk.buffer_id,
            chunk._offset * 4,
            chunk.nbytes,
        )
    else:
        GL.glBindBufferBase(
            GL.GL_SHADER_STORAGE_BUFFER,
            binding,
            chunk.buffer_id,
        )


def _uniform_uint_array(program_id: int, name: str, values) -> None:
    """Upload one ``uniform uint[]`` by name, if the program declares it."""

    from OpenGL import GL

    key = (program_id, name)
    loc = _uniform_location_cache.get(key)
    if loc is None:
        loc = int(GL.glGetUniformLocation(program_id, name))
        _uniform_location_cache[key] = loc
    values = [int(value) for value in values]
    if loc != -1 and values:
        GL.glUniform1uiv(
            loc, len(values), (GL.GLuint * len(values))(*values)
        )


def _bind_arena(
    program_id: int, chunks, outputs, extents=None
) -> None:
    """Bind the one arena and hand the shader its slot offsets and extents.

    Nothing else is bound.  A value is reached by index, so the number of
    values a shader touches no longer competes for binding points.  Each
    composed operation also gets its own element count, because operations
    sharing a program do not have to share an extent.
    """

    from OpenGL import GL

    _ARENA.reserve()
    GL.glBindBufferBase(
        GL.GL_SHADER_STORAGE_BUFFER, ARENA_BINDING, _ARENA.buffer
    )
    _uniform_uint_array(
        program_id, "u_slot",
        [chunk._offset for chunk in (*chunks, *outputs)],
    )
    _uniform_uint_array(program_id, "u_extent", extents or ())


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
    binding_count = 1
    _dispatch_stats["calls"] += 1
    _dispatch_stats["work_items"] += int(plan.count)

    # Deferred inputs can themselves execute a fused program. Materialize every
    # nested region before binding this dispatch's program; otherwise the inner
    # dispatch correctly restores program 0 and accidentally unbinds the outer
    # program before its launch.
    for chunk in chunks:
        chunk._to_gpu_current()
    for output in outputs:
        output._to_gpu_current()

    batch = _dispatch_batch_state.get()
    if batch is not None:
        batch.max_bindings = max(batch.max_bindings, binding_count)
        GL.glUseProgram(program_id)
        uniform_key = (program_id, "u_count")
        loc = _uniform_location_cache.get(uniform_key)
        if loc is None:
            loc = int(GL.glGetUniformLocation(program_id, "u_count"))
            _uniform_location_cache[uniform_key] = loc
        if loc != -1:
            GL.glUniform1ui(loc, plan.count)
        _bind_arena(program_id, chunks, outputs, (plan.count,))
        GL.glDispatchCompute(*plan.groups)
        GL.glMemoryBarrier(
            GL.GL_SHADER_STORAGE_BARRIER_BIT
            | GL.GL_BUFFER_UPDATE_BARRIER_BIT
        )
        for output in outputs:
            output._mark_gpu_written()
        return

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

        _bind_arena(program_id, chunks, outputs, (plan.count,))

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


def _dispatch_composed_control(
    program_id: int,
    slots: Sequence[GLChunk],
    written: Sequence[GLChunk],
    plan: GLLaunchPlan,
    debug_buffer: int = 0,
    debug_capacity: int = 0,
    stream_state_buffer: int = 0,
    stream_words_buffer: int = 0,
    slot_table_buffer: int = 0,
    extent_table_buffer: int = 0,
    dispatch_extent_buffer: int = 0,
    control_value_buffer: int = 0,
    dispatch_iteration: int | None = None,
) -> None:
    """Launch one already-composed shell with its exact slot table."""

    from OpenGL import GL

    if plan.skipped:
        return
    for chunk in slots:
        chunk._to_gpu_current()
    GL.glUseProgram(program_id)
    if debug_buffer:
        GL.glBindBufferBase(
            GL.GL_SHADER_STORAGE_BUFFER, 1, int(debug_buffer)
        )
        location = int(
            GL.glGetUniformLocation(program_id, "u_debug_capacity")
        )
        if location != -1:
            GL.glUniform1ui(location, int(debug_capacity))
    if stream_state_buffer:
        GL.glBindBufferBase(
            GL.GL_SHADER_STORAGE_BUFFER, 2, int(stream_state_buffer)
        )
        GL.glBindBufferBase(
            GL.GL_SHADER_STORAGE_BUFFER, 3, int(stream_words_buffer)
        )
    count_location = int(GL.glGetUniformLocation(program_id, "u_count"))
    if count_location != -1:
        GL.glUniform1ui(count_location, plan.count)
    if dispatch_iteration is not None:
        iteration_location = int(
            GL.glGetUniformLocation(
                program_id, "u_dispatch_iteration"
            )
        )
        if iteration_location == -1:
            raise RuntimeError(
                "C-planned GLSL closure lacks u_dispatch_iteration"
            )
        GL.glUniform1i(iteration_location, int(dispatch_iteration))
    _bind_arena(program_id, slots, (), ())
    for binding, buffer_id in (
        (4, slot_table_buffer),
        (5, extent_table_buffer),
        (6, dispatch_extent_buffer),
        (7, control_value_buffer),
    ):
        GL.glBindBufferBase(
            GL.GL_SHADER_STORAGE_BUFFER,
            int(binding),
            int(buffer_id),
        )
    _dispatch_stats["calls"] += 1
    _dispatch_stats["work_items"] += int(plan.count)
    GL.glDispatchCompute(*plan.groups)
    GL.glMemoryBarrier(
        GL.GL_SHADER_STORAGE_BARRIER_BIT
        | GL.GL_BUFFER_UPDATE_BARRIER_BIT
    )
    for chunk in written:
        chunk._mark_gpu_written()
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, ARENA_BINDING, 0)
    if debug_buffer:
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 1, 0)
    if stream_state_buffer:
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 2, 0)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 3, 0)
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 4, 0)
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 5, 0)
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 6, 0)
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 7, 0)
    GL.glUseProgram(0)
    error = int(GL.glGetError())
    if error != int(GL.GL_NO_ERROR):
        raise RuntimeError(
            f"OpenGL error 0x{error:04X} during composed control dispatch"
        )


class InstalledGLSLControlShell:
    """Device-installed single-dispatch realization of a control artifact."""

    def __init__(self, artifact: ComposedGLSLControlArtifact):
        self.artifact = artifact
        contract_errors = []
        for alias, owner in artifact.value_aliases.items():
            alias_meta = artifact.value_meta.get(alias)
            owner_meta = artifact.value_meta.get(owner)
            if (
                alias_meta is not None
                and owner_meta is not None
                and (
                    tuple(alias_meta.shape or ())
                    != tuple(owner_meta.shape or ())
                    or str(alias_meta.dtype) != str(owner_meta.dtype)
                )
            ):
                contract_errors.append(
                    f"alias {alias}->{owner} changes storage contract"
                )
        owner_ids = {
            int(artifact.value_aliases.get(value_id, value_id))
            for value_id in artifact.slot_value_ids
        }
        for value_id in sorted(owner_ids):
            meta = artifact.value_meta.get(value_id)
            if meta is not None and meta.shape is not None and meta.dtype is not None:
                continue
            slot_indices = tuple(
                index
                for index, candidate in enumerate(artifact.slot_value_ids)
                if int(artifact.value_aliases.get(candidate, candidate))
                == value_id
            )
            source_uses = tuple(
                line.strip()
                for line in artifact.source.splitlines()
                if any(
                    f"u_slot[{index}]" in line
                    for index in slot_indices
                )
            )
            contract_errors.append(
                f"value {value_id} lacks shape/dtype metadata; "
                f"slots={slot_indices!r}; source_uses={source_uses[:8]!r}; "
                "bindings="
                f"{artifact.slot_contract_diagnostics.get(value_id, ())!r}"
            )
        if contract_errors:
            raise ValueError(
                "composed GLSL storage contract audit failed before driver "
                "compilation: "
                + " | ".join(contract_errors)
            )
        sources = artifact.phase_sources or (artifact.source,)
        cache_identities = artifact.phase_cache_identities
        if cache_identities and len(cache_identities) != len(sources):
            raise ValueError(
                "GLSL artifact phase cache identities do not match phase "
                f"sources: {len(cache_identities)} != {len(sources)}"
            )
        self.program_ids = tuple(
            compile_glsl_source(
                source,
                cache_identity=(
                    cache_identities[index]
                    if cache_identities
                    else None
                ),
            )
            for index, source in enumerate(sources)
        )
        self.program_id = self.program_ids[0]
        self.debug_buffer = 0
        self.stream_state_buffer = 0
        self.stream_words_buffer = 0
        self.slot_table_buffer = 0
        self.extent_table_buffer = 0
        self.dispatch_extent_buffer = 0
        self.control_value_buffer = 0
        self.last_debug_records: tuple[tuple[int, int, int, int], ...] = ()
        self.last_debug_header: tuple[int, int, int, int] = (0, 0, 1, 0)
        self.last_gpu_ms = 0.0
        self.last_dispatches = 0
        self.last_stream_status = 0
        if artifact.instrumentation:
            from OpenGL import GL

            self.debug_buffer = int(GL.glGenBuffers(1))
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self.debug_buffer)
            GL.glBufferData(
                GL.GL_SHADER_STORAGE_BUFFER,
                (4 + 4 * int(artifact.debug_capacity)) * 4,
                None,
                GL.GL_DYNAMIC_READ,
            )
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        if artifact.stream_publications:
            from OpenGL import GL

            word_capacity = int(artifact.stream_word_capacity)
            descriptor_capacity = int(
                artifact.stream_descriptor_capacity
            )
            if word_capacity < 1 or descriptor_capacity < 1:
                raise ValueError("resident stream capacities must be positive")
            self.stream_state_buffer = int(GL.glGenBuffers(1))
            self.stream_words_buffer = int(GL.glGenBuffers(1))
            state = np.zeros(
                8
                + descriptor_capacity * 4
                + 2 * int(artifact.stream_continuation_count),
                dtype=np.uint32,
            )
            state[4] = word_capacity
            state[5] = descriptor_capacity
            GL.glBindBuffer(
                GL.GL_SHADER_STORAGE_BUFFER,
                self.stream_state_buffer,
            )
            GL.glBufferData(
                GL.GL_SHADER_STORAGE_BUFFER,
                state.nbytes,
                state,
                GL.GL_DYNAMIC_READ,
            )
            GL.glBindBuffer(
                GL.GL_SHADER_STORAGE_BUFFER,
                self.stream_words_buffer,
            )
            GL.glBufferData(
                GL.GL_SHADER_STORAGE_BUFFER,
                word_capacity * 4,
                None,
                GL.GL_DYNAMIC_READ,
            )
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        from OpenGL import GL

        self.slot_table_buffer = int(GL.glGenBuffers(1))
        self.extent_table_buffer = int(GL.glGenBuffers(1))
        self.dispatch_extent_buffer = int(GL.glGenBuffers(1))
        self.control_value_buffer = int(GL.glGenBuffers(1))
        self.values: dict[int, GLChunk] = {}
        for alias, owner in artifact.value_aliases.items():
            alias_meta = artifact.value_meta.get(alias)
            owner_meta = artifact.value_meta.get(owner)
            if (
                alias_meta is not None
                and owner_meta is not None
                and (
                    tuple(alias_meta.shape or ())
                    != tuple(owner_meta.shape or ())
                    or str(alias_meta.dtype) != str(owner_meta.dtype)
                )
            ):
                raise ValueError(
                    "loop-carried GLSL alias changes storage contract: "
                    f"{alias}->{owner}"
                )
        owner_ids = {
            int(artifact.value_aliases.get(value_id, value_id))
            for value_id in artifact.slot_value_ids
        }
        private_backings: list[GLChunk] = []
        for value_id in owner_ids:
            meta = artifact.value_meta.get(value_id)
            if meta is None or meta.shape is None or meta.dtype is None:
                slot_indices = tuple(
                    index
                    for index, candidate in enumerate(
                        artifact.slot_value_ids
                    )
                    if candidate == value_id
                )
                source_uses = tuple(
                    line.strip()
                    for line in artifact.source.splitlines()
                    if any(
                        f"u_slot[{index}]" in line
                        for index in slot_indices
                    )
                )
                raise ValueError(
                    f"composed GLSL value {value_id} lacks shape/dtype "
                    f"metadata; slots={slot_indices!r}; "
                    f"source_uses={source_uses[:8]!r}"
                )
            shape = tuple(int(size) for size in meta.shape)
            logical_extent = _shape_product(shape)
            storage_extent = int(
                artifact.private_value_capacities.get(
                    value_id, logical_extent
                )
            )
            if storage_extent < logical_extent:
                raise ValueError(
                    "workgroup-private storage capacity is smaller than its "
                    f"logical value: {value_id} "
                    f"{storage_extent} < {logical_extent}"
                )
            if storage_extent > logical_extent:
                backing = GLChunk(
                    (storage_extent,), dtype=meta.dtype
                ).to_gpu()
                private_backings.append(backing)
                self.values[value_id] = backing.range_view(
                    shape, offset=0
                )
            else:
                self.values[value_id] = GLChunk(
                    shape, dtype=meta.dtype
                ).to_gpu()
        for value_id in set(artifact.slot_value_ids):
            owner = int(artifact.value_aliases.get(value_id, value_id))
            self.values[value_id] = self.values[owner]
        _ARENA.reserve()
        self._owned_chunks = tuple({
            id(chunk): chunk
            for chunk in (
                *self.values.values(),
                *private_backings,
            )
        }.values())
        slots = tuple(
            self.values[value_id] for value_id in artifact.slot_value_ids
        )
        static_tables = (
            (
                self.slot_table_buffer,
                np.asarray(
                    [chunk._offset for chunk in slots],
                    dtype=np.uint32,
                ),
            ),
            (
                self.extent_table_buffer,
                np.asarray(artifact.slot_extents, dtype=np.uint32),
            ),
            (
                self.dispatch_extent_buffer,
                np.asarray(artifact.extents, dtype=np.uint32),
            ),
        )
        for buffer_id, values in static_tables:
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, int(buffer_id))
            GL.glBufferData(
                GL.GL_SHADER_STORAGE_BUFFER,
                max(4, int(values.nbytes)),
                values if values.size else None,
                GL.GL_STATIC_DRAW,
            )
        GL.glBindBuffer(
            GL.GL_SHADER_STORAGE_BUFFER, self.control_value_buffer
        )
        GL.glBufferData(
            GL.GL_SHADER_STORAGE_BUFFER,
            max(4, 4 * len(artifact.uniform_value_ids)),
            None,
            GL.GL_DYNAMIC_DRAW,
        )
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        self._last_control_values: tuple[int, ...] | None = None

    def _upload_control_values(self, values: Sequence[int]) -> None:
        """Update only dynamic control words, and only when they changed.

        Slot offsets, tensor extents, and dispatch extents are properties of
        the installed compiled shell.  Reallocating and re-uploading those
        tables for each dispatch turns immutable program metadata into host
        churn and can force driver synchronization.
        """

        normalized = tuple(int(value) for value in values)
        if normalized == self._last_control_values:
            return
        from OpenGL import GL

        words = np.asarray(normalized, dtype=np.uint32)
        if words.size:
            GL.glBindBuffer(
                GL.GL_SHADER_STORAGE_BUFFER, self.control_value_buffer
            )
            GL.glBufferSubData(
                GL.GL_SHADER_STORAGE_BUFFER,
                0,
                words.nbytes,
                words,
            )
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        self._last_control_values = normalized

    def execute(
        self,
        feeds: Mapping[int, Any],
        *,
        uniform_feeds: Mapping[int, Any] | None = None,
        _resume: bool = False,
    ) -> dict[str, GLChunk]:
        runtime_values = dict(self.values)
        uniforms: dict[str, int] = {}
        if _resume and feeds:
            raise ValueError(
                "resident continuation resumes installed ranges and accepts "
                "no replacement feeds"
            )
        for value_id, expected in (
            ()
            if _resume
            else self.artifact.specialized_values.items()
        ):
            # Specialization records the value used while compiling shape or
            # control structure.  It is not, by itself, a runtime ABI input:
            # nested closures commonly specialize ordinary lexical constants.
            # If the value is also an external/uniform input it is supplied
            # and checked here; otherwise the compiled internal specialization
            # is authoritative and no fake host feed is invented for it.
            if value_id not in feeds:
                continue
            actual = feeds[value_id]
            if hasattr(actual, "item"):
                actual = actual.item()
            if hasattr(expected, "item"):
                expected = expected.item()
            if actual != expected:
                raise ValueError(
                    "GLSL shell specialization mismatch for value "
                    f"{value_id}: compiled={expected!r}, runtime={actual!r}"
                )
        for value_id in (
            () if _resume else self.artifact.external_value_ids
        ):
            try:
                value = feeds[value_id]
            except KeyError as error:
                raise KeyError(
                    f"missing composed GLSL feed {value_id}"
                ) from error
            if isinstance(value, GLChunk):
                raise TypeError(
                    "a planned GLSL shell owns its feed ranges; resident "
                    "GLChunk handoff needs an explicit arena-range transfer "
                    "plan and cannot replace a slot during execution"
                )
            runtime_values[value_id].upload_numpy(value)
        if not _resume:
            scalar_values = dict(feeds)
            scalar_values.update(uniform_feeds or {})
            for name, value_id in self.artifact.uniform_value_ids.items():
                if value_id in scalar_values:
                    value = scalar_values[value_id]
                elif value_id in self.artifact.specialized_values:
                    value = self.artifact.specialized_values[value_id]
                else:
                    raise KeyError(
                        f"missing composed GLSL control value {value_id}"
                    )
                if hasattr(value, "item"):
                    value = value.item()
                uniforms[name] = int(value)
            self._upload_control_values(tuple(uniforms.values()))
        elif (
            self.artifact.uniform_value_ids
            and self._last_control_values is None
        ):
            raise RuntimeError(
                "resident shell cannot resume before its initial controls "
                "have been installed"
            )
        slots = tuple(
            runtime_values[value_id]
            for value_id in self.artifact.slot_value_ids
        )
        count = max(self.artifact.extents, default=0)
        plan = plan_launch(count, binding_count=1)
        if self.artifact.device_resident and count:
            plan = GLLaunchPlan(
                count,
                int(self.artifact.local_size),
                (1, 1, 1),
                plan.limits,
            )
        if self.artifact.workgroup_loop_bounds is not None:
            if _resume:
                raise RuntimeError(
                    "workgroup-parallel loops do not use resident resume"
                )
            workgroup_iterations = _c_dispatch_iterations(
                self.artifact.workgroup_loop_bounds,
                uniforms,
            )
            workgroup_count = len(workgroup_iterations)
            if workgroup_count > plan.limits.max_group_count[0]:
                raise ValueError(
                    "frame batch exceeds the GLSL x workgroup limit: "
                    f"{workgroup_count} > "
                    f"{plan.limits.max_group_count[0]}"
                )
            capacities = [
                int(capacity)
                // max(
                    1,
                    _shape_product(tuple(
                        self.artifact.value_meta[value_id].shape or ()
                    )),
                )
                for value_id, capacity
                in self.artifact.private_value_capacities.items()
            ]
            if capacities and workgroup_count > min(capacities):
                raise ValueError(
                    "frame batch exceeds compiled workgroup-private "
                    f"capacity: {workgroup_count} > {min(capacities)}"
                )
            plan = GLLaunchPlan(
                (count if workgroup_count else 0),
                int(self.artifact.local_size),
                (max(1, workgroup_count), 1, 1),
                plan.limits,
            )
        dispatch_iterations: tuple[int | None, ...] = (None,)
        if self.artifact.c_dispatch_loop_bounds is not None:
            if _resume:
                raise RuntimeError(
                    "C-planned dispatch loops do not use resident resume"
                )
            dispatch_iterations = tuple(
                _c_dispatch_iterations(
                    self.artifact.c_dispatch_loop_bounds,
                    uniforms,
                )
            )
        written = tuple(dict.fromkeys(
            runtime_values[value_id]
            for value_id in self.artifact.terminal_outputs.values()
        ))
        with _ARENA.execution():
            if self.debug_buffer:
                from OpenGL import GL

                zero = np.asarray((0, 0, 1, 0), dtype=np.uint32)
                GL.glBindBuffer(
                    GL.GL_SHADER_STORAGE_BUFFER, self.debug_buffer
                )
                GL.glBufferSubData(
                    GL.GL_SHADER_STORAGE_BUFFER, 0, zero.nbytes, zero
                )
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            query = 0
            if self.debug_buffer:
                from OpenGL import GL

                query = int(
                    np.asarray(GL.glGenQueries(1)).reshape(-1)[0]
                )
                GL.glBeginQuery(GL.GL_TIME_ELAPSED, query)
            self.last_dispatches = 0
            for dispatch_iteration in dispatch_iterations:
                for program_id in self.program_ids:
                    _dispatch_composed_control(
                        program_id,
                        slots,
                        written,
                        plan,
                        self.debug_buffer,
                        self.artifact.debug_capacity,
                        self.stream_state_buffer,
                        self.stream_words_buffer,
                        self.slot_table_buffer,
                        self.extent_table_buffer,
                        self.dispatch_extent_buffer,
                        self.control_value_buffer,
                        dispatch_iteration,
                    )
                    self.last_dispatches += 1
            if query:
                from OpenGL import GL

                GL.glEndQuery(GL.GL_TIME_ELAPSED)
                elapsed_ns = ctypes.c_uint64()
                GL.glGetQueryObjectui64v(
                    query,
                    GL.GL_QUERY_RESULT,
                    ctypes.byref(elapsed_ns),
                )
                GL.glDeleteQueries(1, (query,))
                self.last_gpu_ms = elapsed_ns.value / 1e6
            if self.debug_buffer:
                from OpenGL import GL

                GL.glBindBuffer(
                    GL.GL_SHADER_STORAGE_BUFFER, self.debug_buffer
                )
                header = np.empty(4, dtype=np.uint32)
                GL.glGetBufferSubData(
                    GL.GL_SHADER_STORAGE_BUFFER,
                    0,
                    header.nbytes,
                    header.ctypes.data_as(ctypes.c_void_p),
                )
                count = min(
                    int(header[0]), int(self.artifact.debug_capacity)
                )
                self.last_debug_header = tuple(
                    int(value) for value in header
                )
                words = np.empty(count * 4, dtype=np.uint32)
                if count:
                    GL.glGetBufferSubData(
                        GL.GL_SHADER_STORAGE_BUFFER,
                        16,
                        words.nbytes,
                        words.ctypes.data_as(ctypes.c_void_p),
                    )
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
                self.last_debug_records = tuple(
                    tuple(int(value) for value in row)
                    for row in words.reshape(-1, 4)
                )
                validation_errors = tuple(
                    record
                    for record in self.last_debug_records
                    if record[0] == 6
                )
                if validation_errors:
                    codes = tuple(
                        record[1] for record in validation_errors
                    )
                    raise RuntimeError(
                        "compiled GLSL shell device validation failed; "
                        f"error_codes={codes!r}"
                    )
        return {
            name: runtime_values[value_id]
            for name, value_id in self.artifact.terminal_outputs.items()
        }

    def resume(self) -> dict[str, GLChunk]:
        """Continue the same installed shell from resident suspension state."""

        if not self.stream_state_buffer:
            raise RuntimeError(
                "compiled GLSL shell has no resident stream continuation"
            )
        if self.last_stream_status != 1:
            raise RuntimeError(
                "compiled GLSL shell is not suspended on downstream capacity"
            )
        return self.execute({}, _resume=True)

    def drain_stream(self, max_items: int | None = None):
        """Consume published resident ranges in descriptor order.

        This is an ABI operation for a C/host shell, not shader orchestration:
        the compiled GLSL shell owns production and backpressure state.  A
        caller may drain any prefix, update the consumer sequences once, and
        dispatch the same installed shell again if it had suspended on a full
        queue.
        """

        if not self.stream_state_buffer:
            return ()
        from OpenGL import GL

        header = np.empty(8, dtype=np.uint32)
        GL.glBindBuffer(
            GL.GL_SHADER_STORAGE_BUFFER, self.stream_state_buffer
        )
        GL.glGetBufferSubData(
            GL.GL_SHADER_STORAGE_BUFFER,
            0,
            header.nbytes,
            header.ctypes.data_as(ctypes.c_void_p),
        )
        write_words, read_words, write_desc, read_desc = (
            int(value) for value in header[:4]
        )
        self.last_stream_status = int(header[6])
        word_capacity = int(header[4])
        descriptor_capacity = int(header[5])
        if self.last_stream_status == 2:
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            raise RuntimeError(
                "resident stream publication exceeds its fixed payload "
                f"capacity of {word_capacity} words; "
                f"attempted={int(header[7])}"
            )
        available = write_desc - read_desc
        if max_items is not None:
            available = min(available, max(0, int(max_items)))
        descriptors = np.empty((available, 4), dtype=np.uint32)
        first_descriptors = min(
            available,
            descriptor_capacity - (read_desc % descriptor_capacity),
        )
        if first_descriptors:
            GL.glGetBufferSubData(
                GL.GL_SHADER_STORAGE_BUFFER,
                (8 + (read_desc % descriptor_capacity) * 4) * 4,
                first_descriptors * 16,
                descriptors[:first_descriptors].ctypes.data_as(
                    ctypes.c_void_p
                ),
            )
        if first_descriptors < available:
            tail = descriptors[first_descriptors:]
            GL.glGetBufferSubData(
                GL.GL_SHADER_STORAGE_BUFFER,
                8 * 4,
                tail.nbytes,
                tail.ctypes.data_as(ctypes.c_void_p),
            )
        consumed_words = int(
            np.asarray(descriptors[:, 1], dtype=np.uint64).sum()
        )
        if consumed_words > write_words - read_words:
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            raise RuntimeError(
                "resident stream descriptors exceed published word range"
            )
        payload_words = np.empty(consumed_words, dtype=np.uint32)
        first_words = min(
            consumed_words,
            word_capacity - (read_words % word_capacity),
        )
        GL.glBindBuffer(
            GL.GL_SHADER_STORAGE_BUFFER, self.stream_words_buffer
        )
        if first_words:
            GL.glGetBufferSubData(
                GL.GL_SHADER_STORAGE_BUFFER,
                (read_words % word_capacity) * 4,
                first_words * 4,
                payload_words.ctypes.data_as(ctypes.c_void_p),
            )
        if first_words < consumed_words:
            tail = payload_words[first_words:]
            GL.glGetBufferSubData(
                GL.GL_SHADER_STORAGE_BUFFER,
                0,
                tail.nbytes,
                tail.ctypes.data_as(ctypes.c_void_p),
            )
        GL.glBindBuffer(
            GL.GL_SHADER_STORAGE_BUFFER, self.stream_state_buffer
        )
        items = []
        payload_offset = 0
        expected_start = read_words
        for descriptor in descriptors:
            start, count, stream_id, final = (
                int(value) for value in descriptor
            )
            if start != (expected_start & 0xFFFFFFFF):
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
                raise RuntimeError(
                    "resident stream descriptor order is not contiguous: "
                    f"expected word sequence {expected_start}, got {start}"
                )
            payload = payload_words[payload_offset:payload_offset + count]
            payload_offset += count
            expected_start += count
            items.append({
                "stream_id": stream_id,
                "words": payload.copy(),
                "values": self._decode_stream_words(stream_id, payload),
                "final": bool(final),
            })
        if available:
            next_read_words = read_words + consumed_words
            next_read_desc = read_desc + available
            if (
                next_read_words == write_words
                and next_read_desc == write_desc
            ):
                # Empty rings have no ordering history to preserve.  Reset all
                # four monotone sequences together so a long-lived installed
                # shell cannot eventually wrap uint32 counters.
                zero_sequences = np.zeros(4, dtype=np.uint32)
                GL.glBufferSubData(
                    GL.GL_SHADER_STORAGE_BUFFER,
                    0,
                    zero_sequences.nbytes,
                    zero_sequences,
                )
            else:
                updated_words = np.asarray(
                    (next_read_words,), dtype=np.uint32
                )
                updated_descriptors = np.asarray(
                    (next_read_desc,), dtype=np.uint32
                )
                GL.glBufferSubData(
                    GL.GL_SHADER_STORAGE_BUFFER,
                    4,
                    updated_words.nbytes,
                    updated_words,
                )
                GL.glBufferSubData(
                    GL.GL_SHADER_STORAGE_BUFFER,
                    12,
                    updated_descriptors.nbytes,
                    updated_descriptors,
                )
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        return tuple(items)

    def _decode_stream_words(self, stream_id: int, words: np.ndarray):
        publication = next(
            (
                item
                for item in self.artifact.stream_publications
                if int(item.stream_id) == int(stream_id)
            ),
            None,
        )
        meta = (
            None
            if publication is None
            else self.artifact.value_meta.get(
                int(self.artifact.value_aliases.get(
                    int(publication.value_id),
                    int(publication.value_id),
                ))
            )
        )
        if meta is None or meta.dtype is None:
            return words
        dtype = np.dtype(meta.dtype)
        if dtype.itemsize == 0 or dtype.kind in {"S", "a"}:
            # Python ``bytes`` is the terminal host view of one octet per
            # arena word.  NumPy represents the abstract boundary as S0; an
            # astype(S0) conversion silently produces a zero-byte array.
            # Preserve the compiler's numerical storage contract and perform
            # the one intentional byte materialization here.
            return words.astype(np.uint8, copy=False)
        if dtype.itemsize < np.dtype(np.uint32).itemsize:
            return words.astype(dtype, copy=False)
        if dtype.itemsize == np.dtype(np.uint32).itemsize:
            return words.view(dtype)
        raise ValueError(
            "resident stream element exceeds one arena word: "
            f"stream={stream_id}, dtype={dtype}"
        )

    def release(self) -> None:
        if self.debug_buffer:
            from OpenGL import GL

            GL.glDeleteBuffers(1, [self.debug_buffer])
            self.debug_buffer = 0
        if self.stream_state_buffer:
            from OpenGL import GL

            GL.glDeleteBuffers(
                2,
                [self.stream_state_buffer, self.stream_words_buffer],
            )
            self.stream_state_buffer = 0
            self.stream_words_buffer = 0
        if self.slot_table_buffer:
            from OpenGL import GL

            GL.glDeleteBuffers(
                4,
                [
                    self.slot_table_buffer,
                    self.extent_table_buffer,
                    self.dispatch_extent_buffer,
                    self.control_value_buffer,
                ],
            )
            self.slot_table_buffer = 0
            self.extent_table_buffer = 0
            self.dispatch_extent_buffer = 0
            self.control_value_buffer = 0
        for chunk in self._owned_chunks:
            chunk.release()
        if self.program_ids:
            # Compiled programs are context-scoped cache assets shared by
            # installed shells.  Buffers/ranges belong to this installation,
            # but deleting a borrowed program here poisons the cache and makes
            # the next identical shell receive an invalid GL name.
            self.program_ids = ()
            self.program_id = 0
        self.values.clear()


def _structural_chunks(values: Sequence[Any]) -> list[GLChunk]:
    if not values:
        raise ValueError("tensors list cannot be empty")
    chunks = [
        value if isinstance(value, GLChunk) else GLChunk.from_numpy(value)
        for value in values
    ]
    _materialize_structural_fanout(chunks)
    return chunks


def _materialize_structural_fanout(chunks: Sequence[GLChunk]) -> None:
    """Coalesce compatible deferred branches before stack/cat boundaries."""

    candidates: list[GLChunk] = []
    seen_chunks: set[int] = set()
    for chunk in chunks:
        if chunk._deferred is None or id(chunk) in seen_chunks:
            continue
        candidates.append(chunk)
        seen_chunks.add(id(chunk))
    if len(candidates) < 2:
        return

    groups: list[list[GLChunk]] = []
    for chunk in candidates:
        placed = False
        for group in groups:
            if group[0].shape != chunk.shape:
                continue
            tentative = group + [chunk]
            feed_ids = {
                feed_id
                for value in tentative
                for feed_id in value._deferred.feeds
            }
            step_ids = {
                step.result_id
                for value in tentative
                for step in value._deferred.program.steps
            }
            binding_count = len(feed_ids) + len(tentative)
            if (
                binding_count
                <= _compute_limits().max_dispatch_ssbo_blocks
                and len(step_ids) <= 512
            ):
                group.append(chunk)
                placed = True
                break
        if not placed:
            groups.append([chunk])

    for group in groups:
        if len(group) < 2:
            continue
        snapshots = [chunk._deferred for chunk in group]
        feeds: dict[int, GLChunk] = {}
        metadata: dict[int, Meta] = {}
        steps: list[OpStep] = []
        seen_results: set[int] = set()
        outputs: dict[str, int] = {}
        feed_order: list[int] = []
        for output_index, deferred in enumerate(snapshots):
            assert deferred is not None
            for feed_id, feed in deferred.feeds.items():
                if feed_id not in feeds:
                    feed_order.append(feed_id)
                feeds[feed_id] = feed
            if deferred.program.meta:
                metadata.update(deferred.program.meta)
            for step in deferred.program.steps:
                if step.result_id not in seen_results:
                    steps.append(step)
                    seen_results.add(step.result_id)
            outputs[f"result_{output_index}"] = primary_output_id(
                deferred.program
            )
        program = FusedProgram(
            version=1,
            feeds=set(feeds),
            steps=steps,
            outputs=outputs,
            meta=metadata,
        )
        program.feed_order = tuple(feed_order)
        program.glsl_linear_output_shape = group[0].shape
        for chunk in group:
            chunk._deferred = None
        try:
            execute_multi_output_program(
                program,
                feeds,
                outs={
                    f"result_{index}": chunk
                    for index, chunk in enumerate(group)
                },
            )
        except Exception:
            for chunk, deferred in zip(group, snapshots):
                chunk._deferred = deferred
            raise


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


def where_chunks(condition: Any, if_true: Any, if_false: Any) -> GLChunk:
    chunks = _structural_chunks((condition, if_true, if_false))
    condition_chunk, true_chunk, false_chunk = chunks
    output_shape = _broadcast_shape(
        _broadcast_shape(condition_chunk.shape, true_chunk.shape),
        false_chunk.shape,
    )
    output_dtype = _promote_dtype(true_chunk.dtype, false_chunk.dtype)
    out = GLChunk(output_shape, dtype=output_dtype)
    plan = plan_launch(out.count, binding_count=4)
    source = emit_where_source(
        condition_chunk.shape,
        true_chunk.shape,
        false_chunk.shape,
        condition_dtype=condition_chunk.dtype,
        true_dtype=true_chunk.dtype,
        false_dtype=false_chunk.dtype,
        output_dtype=output_dtype,
        output_shape=output_shape,
        local_size=plan.local_size,
    )
    _dispatch(_compile(source), chunks, out, plan)
    return out


def full_chunk(
    shape: Sequence[int],
    fill_value: Any,
    *,
    dtype: Any = None,
) -> GLChunk:
    """Create a constant tensor in one resident GPU dispatch."""

    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(size) for size in shape)
    out = GLChunk(shape, dtype=_normalize_dtype(dtype))
    if out.count == 0:
        return out
    plan = plan_launch(out.count, binding_count=1)
    source = emit_fill_source(
        fill_value,
        dtype=out.dtype,
        local_size=plan.local_size,
    )
    _dispatch(_compile(source), [], out, plan)
    return out


def constant_chunk(
    shape: Sequence[int],
    values: Sequence[Any],
    *,
    dtype: Any = None,
) -> GLChunk:
    """Materialize compile-time literal values in one resident dispatch."""

    shape = tuple(int(size) for size in shape)
    out = GLChunk(shape, dtype=_normalize_dtype(dtype))
    if out.count == 0:
        return out
    if len(values) != out.count:
        raise ValueError("constant payload size does not match output shape")
    plan = plan_launch(out.count, binding_count=1)
    source = emit_constant_source(
        values,
        dtype=out.dtype,
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
    if dim == 0 and step == 1:
        if start == 0:
            return chunk.prefix_view(output_shape)
        source_row_size = _shape_product(chunk.shape[1:])
        try:
            return chunk.range_view(
                output_shape,
                offset=start * source_row_size,
            )
        except ValueError:
            # SSBO range offsets are device-aligned. Small/misaligned slices
            # retain the ordinary one-dispatch copy path below.
            pass
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
    plan = plan_launch(out.count, binding_count=2)
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
            meta = metadata.get(feed_id)
            declared_shape = tuple(
                int(size)
                for size in (
                    getattr(meta, "shape", None) or chunk.shape
                )
            )
            if _shape_product(declared_shape) != chunk.count:
                declared_shape = chunk.shape
            if chunk.count == 1 and output_count != 1:
                runtime_feed_shapes[feed_id] = declared_shape
            elif chunk.count == output_count:
                # A deferred reshape changes logical coordinates but not the
                # elementwise program's row-major lane correspondence.
                runtime_feed_shapes[feed_id] = shape
            elif _broadcast_shape(declared_shape, shape) == shape:
                # A smaller non-scalar feed may be broadcast directly by
                # the fused shader. This keeps branches such as (N, 1) and
                # (1, M) register-resident instead of materializing each
                # side before their common (N, M) consumer.
                runtime_feed_shapes[feed_id] = declared_shape
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


def execute_captured_fused_program(
    captured,
    runtime_feeds: Mapping[int, Any],
) -> dict[str, GLChunk]:
    """Execute one captured ProcessGraph region through compiled GLSL stages."""

    program = captured.program
    merged = dict(captured.feeds)
    merged.update(runtime_feeds)
    chunks = {
        value_id: (
            value.data
            if hasattr(value, "data") and isinstance(value.data, GLChunk)
            else value
        )
        for value_id, value in merged.items()
    }
    stages = tuple(getattr(captured, "stages", ()) or ())
    if stages:
        for stage in stages:
            missing = set(stage.feeds) - set(chunks)
            if missing:
                raise KeyError(
                    "captured GLSL stage is missing routed values: "
                    + ", ".join(map(str, sorted(missing)))
                )
            stage_capture = type(captured)(
                stage,
                {
                    value_id: chunks[value_id]
                    for value_id in stage.feeds
                },
            )
            results = execute_captured_fused_program(stage_capture, {})
            chunks.update({
                value_id: results[name]
                for name, value_id in stage.outputs.items()
            })
        return {
            name: chunks[value_id]
            for name, value_id in program.outputs.items()
        }
    kind = (program.extras or {}).get("kernel_kind")
    if kind in {None, "linear_reshape_copy"}:
        return execute_multi_output_program(program, chunks)

    if len(program.steps) != 1 or len(program.outputs) != 1:
        raise ValueError(
            f"GLSL {kind!r} captured regions require one operation/output"
        )
    step = program.steps[0]
    output_name = next(iter(program.outputs))
    input_chunks = [chunks[value_id] for value_id in step.input_ids]
    if kind == "fill":
        result = full_chunk(
            step.attrs["shape"],
            step.attrs["fill_value"],
            dtype=(program.meta or {})[
                next(iter(program.outputs.values()))
            ].dtype,
        )
    elif kind == "constant":
        result = constant_chunk(
            step.attrs["shape"],
            step.attrs["values"],
            dtype=(program.meta or {})[
                next(iter(program.outputs.values()))
            ].dtype,
        )
    elif kind == "arange":
        result = arange_chunk(
            step.attrs["start"],
            step.attrs["end"],
            step.attrs.get("step", 1),
            dtype=(program.meta or {})[
                next(iter(program.outputs.values()))
            ].dtype,
        )
    elif kind == "stack":
        result = stack_chunks(
            input_chunks,
            int(step.attrs.get("dim", 0)),
        )
    elif kind == "cat":
        result = cat_chunks(
            input_chunks,
            int(step.attrs.get("dim", 0)),
        )
    elif kind == "expand":
        result = expand_chunk(input_chunks[0], step.attrs["shape"])
    elif kind == "permute":
        result = permute_chunk(input_chunks[0], step.attrs["dims"])
    elif kind == "repeat":
        result = repeat_chunk(
            input_chunks[0],
            step.attrs.get("repeats"),
            int(step.attrs.get("dim", 0)),
        )
    elif kind == "matmul":
        result = matmul_chunks(input_chunks[0], input_chunks[1])
    elif kind == "index_select":
        result = index_select_chunk(
            input_chunks[0],
            int(step.attrs.get("dim", 0)),
            input_chunks[1],
        )
    elif kind == "reduce":
        result = reduce_chunk(
            input_chunks[0],
            step.attrs["reduce_op"],
            step.attrs.get("axis"),
            bool(step.attrs.get("keepdim", False)),
        )
    elif kind == "cumsum":
        result = cumsum_chunk(
            input_chunks[0],
            int(step.attrs.get("dim", 0)),
        )
    elif kind == "where":
        result = where_chunks(*input_chunks)
    elif kind == "scatter":
        output_id = next(iter(program.outputs.values()))
        output_meta = (program.meta or {})[output_id]
        result = GLChunk(
            tuple(int(size) for size in output_meta.shape),
            dtype=output_meta.dtype,
        )
        plan = plan_launch(result.count, binding_count=4)
        _dispatch(
            _compile(_emit_captured_fused_program_source(
                captured,
                local_size=plan.local_size,
            )),
            input_chunks,
            result,
            plan,
        )
    elif kind == "slice":
        output_id = next(iter(program.outputs.values()))
        output_meta = (program.meta or {})[output_id]
        result = GLChunk(
            tuple(int(size) for size in output_meta.shape),
            dtype=output_meta.dtype,
        )
        plan = plan_launch(result.count, binding_count=len(step.input_ids) + 1)
        source = _emit_captured_fused_program_source(
            captured,
            local_size=plan.local_size,
        )
        _dispatch(
            _compile(source),
            input_chunks,
            result,
            plan,
        )
    else:
        raise ValueError(f"unsupported captured GLSL kernel kind {kind!r}")
    return {output_name: result}


def _emit_captured_fused_program_source(
    captured,
    *,
    local_size: int = _LOCAL_SIZE,
) -> str:
    """Emit one complete shader for one captured numerical region."""

    program = captured.program
    kind = (program.extras or {}).get("kernel_kind")
    if kind in {None, "linear_reshape_copy"}:
        output_id = next(iter(program.outputs.values()))
        output_shape = tuple(
            int(size)
            for size in ((program.meta or {})[output_id].shape or ())
        )
        feed_shapes = {
            value_id: (
                output_shape
                if kind == "linear_reshape_copy"
                else tuple(
                    int(size)
                    for size in (
                        (program.meta or {})[value_id].shape or ()
                    )
                )
            )
            for value_id in program.feeds
        }
        return emit_multi_output_program_source(
            program,
            local_size=local_size,
            feed_shapes=feed_shapes,
            output_shape=output_shape,
        )

    step = program.steps[0]
    metadata = program.meta or {}
    output_id = next(iter(program.outputs.values()))
    output_meta = metadata[output_id]
    if kind == "fill":
        return emit_fill_source(
            step.attrs["fill_value"],
            dtype=output_meta.dtype,
            local_size=local_size,
        )
    if kind == "constant":
        return emit_constant_source(
            step.attrs["values"],
            dtype=output_meta.dtype,
            local_size=local_size,
        )
    if kind == "arange":
        return emit_arange_source(
            step.attrs["start"],
            step.attrs.get("step", 1),
            dtype=output_meta.dtype,
            local_size=local_size,
        )
    if kind == "scatter":
        return emit_scatter_source(
            metadata[step.input_ids[0]].shape,
            metadata[step.input_ids[1]].shape,
            int(step.attrs.get("dim", 0)),
            dtype=output_meta.dtype,
            local_size=local_size,
        )
    source_meta = metadata[step.input_ids[0]]
    if kind == "stack":
        return emit_stack_source(
            tuple(source_meta.shape or ()),
            len(step.input_ids),
            int(step.attrs.get("dim", 0)),
            input_dtypes=[
                metadata[value_id].dtype for value_id in step.input_ids
            ],
            output_dtype=output_meta.dtype,
            local_size=local_size,
        )
    if kind == "cat":
        return emit_cat_source(
            [tuple(metadata[value_id].shape or ()) for value_id in step.input_ids],
            int(step.attrs.get("dim", 0)),
            input_dtypes=[
                metadata[value_id].dtype for value_id in step.input_ids
            ],
            output_dtype=output_meta.dtype,
            local_size=local_size,
        )
    if kind == "expand":
        return emit_expand_source(
            tuple(source_meta.shape or ()),
            tuple(step.attrs["shape"]),
            dtype=source_meta.dtype,
            local_size=local_size,
        )
    if kind == "permute":
        return emit_permute_source(
            tuple(source_meta.shape or ()),
            tuple(step.attrs["dims"]),
            dtype=source_meta.dtype,
            local_size=local_size,
        )
    if kind == "repeat":
        return emit_repeat_source(
            tuple(source_meta.shape or ()),
            step.attrs.get("repeats"),
            int(step.attrs.get("dim", 0)),
            dtype=source_meta.dtype,
            local_size=local_size,
        )
    if kind == "matmul":
        right_meta = metadata[step.input_ids[1]]
        return emit_matmul_source(
            tuple(source_meta.shape or ()),
            tuple(right_meta.shape or ()),
            left_dtype=source_meta.dtype,
            right_dtype=right_meta.dtype,
            output_dtype=output_meta.dtype,
            local_size=local_size,
        )
    if kind == "index_select":
        index_meta = metadata[step.input_ids[1]]
        return emit_index_select_source(
            tuple(source_meta.shape or ()),
            int(step.attrs.get("dim", 0)),
            _shape_product(tuple(index_meta.shape or ())),
            dtype=source_meta.dtype,
            index_dtype=index_meta.dtype,
            local_size=local_size,
        )
    if kind == "reduce":
        return emit_reduce_source(
            step.attrs["reduce_op"],
            tuple(source_meta.shape or ()),
            step.attrs.get("axis"),
            bool(step.attrs.get("keepdim", False)),
            dtype=source_meta.dtype,
            local_size=local_size,
        )
    if kind == "cumsum":
        return emit_cumsum_source(
            tuple(source_meta.shape or ()),
            int(step.attrs.get("dim", 0)),
            dtype=source_meta.dtype,
            local_size=local_size,
        )
    if kind == "where":
        condition_meta, true_meta, false_meta = (
            metadata[value_id] for value_id in step.input_ids
        )
        return emit_where_source(
            tuple(condition_meta.shape or ()),
            tuple(true_meta.shape or ()),
            tuple(false_meta.shape or ()),
            condition_dtype=condition_meta.dtype,
            true_dtype=true_meta.dtype,
            false_dtype=false_meta.dtype,
            output_dtype=output_meta.dtype,
            output_shape=tuple(output_meta.shape or ()),
            local_size=local_size,
        )
    if kind == "slice":
        slice_kind = step.attrs.get("slice_kind")
        if slice_kind == "axis":
            return emit_slice_axis_source(
                tuple(source_meta.shape or ()),
                int(step.attrs["dim"]),
                int(step.attrs["start"]),
                int(step.attrs["step"]),
                int(step.attrs["count"]),
                dtype=source_meta.dtype,
                local_size=local_size,
            )
        if slice_kind == "flat":
            return emit_slice_axis_source(
                (int(step.attrs["source_count"]),),
                0,
                int(step.attrs["start"]),
                int(step.attrs.get("step", 1)),
                int(step.attrs["count"]),
                dtype=source_meta.dtype,
                local_size=local_size,
            )
        if slice_kind == "index_select":
            index_meta = metadata[step.input_ids[1]]
            index_count = _shape_product(tuple(index_meta.shape or ()))
            return emit_index_select_source(
                tuple(source_meta.shape or ()),
                int(step.attrs["dim"]),
                index_count,
                dtype=source_meta.dtype,
                index_dtype=index_meta.dtype,
                local_size=local_size,
            )
    raise ValueError(f"unsupported captured GLSL kernel kind {kind!r}")


def captured_program_snippet(
    captured,
    *,
    base: int = 0,
    local_size: int = _LOCAL_SIZE,
) -> ShaderSnippet:
    """Lower one captured numerical stage without finishing a shader."""

    if getattr(captured, "stages", ()):
        raise ValueError(
            "lower captured stages individually before control composition"
        )
    program = captured.program
    kind = (program.extras or {}).get("kernel_kind")
    metadata = program.meta or {}
    output_id = next(iter(program.outputs.values()))
    output_meta = metadata[output_id]
    if kind in {None, "linear_reshape_copy"}:
        output_shape = tuple(int(size) for size in (output_meta.shape or ()))
        return program_snippet(
            program,
            feed_shapes={
                value_id: (
                    output_shape
                    if kind == "linear_reshape_copy"
                    else tuple(int(size) for size in (
                        metadata[value_id].shape or ()
                    ))
                )
                for value_id in program.feeds
            },
            output_shape=output_shape,
            allow_multiple_outputs=True,
            base=base,
        )

    step = program.steps[0]
    if kind == "fill":
        return fill_snippet(
            step.attrs["fill_value"], dtype=output_meta.dtype, base=base
        )
    if kind == "constant":
        return constant_snippet(
            step.attrs["values"], dtype=output_meta.dtype, base=base
        )
    if kind == "arange":
        return arange_snippet(
            step.attrs["start"],
            step.attrs.get("step", 1),
            dtype=output_meta.dtype,
            base=base,
        )
    if kind == "scatter":
        return scatter_snippet(
            metadata[step.input_ids[0]].shape,
            metadata[step.input_ids[1]].shape,
            int(step.attrs.get("dim", 0)),
            dtype=output_meta.dtype,
            base=base,
        )
    source_meta = metadata[step.input_ids[0]]
    if kind == "stack":
        return stack_snippet(
            tuple(source_meta.shape or ()),
            len(step.input_ids),
            int(step.attrs.get("dim", 0)),
            input_dtypes=[
                metadata[value_id].dtype for value_id in step.input_ids
            ],
            output_dtype=output_meta.dtype,
            base=base,
        )
    if kind == "cat":
        return cat_snippet(
            [
                tuple(metadata[value_id].shape or ())
                for value_id in step.input_ids
            ],
            int(step.attrs.get("dim", 0)),
            input_dtypes=[
                metadata[value_id].dtype for value_id in step.input_ids
            ],
            output_dtype=output_meta.dtype,
            base=base,
        )
    if kind == "expand":
        return expand_snippet(
            tuple(source_meta.shape or ()),
            tuple(step.attrs["shape"]),
            dtype=source_meta.dtype,
            base=base,
        )
    if kind == "permute":
        return permute_snippet(
            tuple(source_meta.shape or ()),
            tuple(step.attrs["dims"]),
            dtype=source_meta.dtype,
            base=base,
        )
    if kind == "repeat":
        return repeat_snippet(
            tuple(source_meta.shape or ()),
            step.attrs.get("repeats"),
            int(step.attrs.get("dim", 0)),
            dtype=source_meta.dtype,
            base=base,
        )
    if kind == "matmul":
        right_meta = metadata[step.input_ids[1]]
        return matmul_snippet(
            tuple(source_meta.shape or ()),
            tuple(right_meta.shape or ()),
            left_dtype=source_meta.dtype,
            right_dtype=right_meta.dtype,
            output_dtype=output_meta.dtype,
            local_size=local_size,
            base=base,
        )
    if kind == "index_select":
        index_meta = metadata[step.input_ids[1]]
        return index_select_snippet(
            tuple(source_meta.shape or ()),
            int(step.attrs.get("dim", 0)),
            _shape_product(tuple(index_meta.shape or ())),
            dtype=source_meta.dtype,
            index_dtype=index_meta.dtype,
            base=base,
        )
    if kind == "reduce":
        return reduce_snippet(
            step.attrs["reduce_op"],
            tuple(source_meta.shape or ()),
            step.attrs.get("axis"),
            bool(step.attrs.get("keepdim", False)),
            dtype=source_meta.dtype,
            base=base,
        )
    if kind == "cumsum":
        return cumsum_snippet(
            tuple(source_meta.shape or ()),
            int(step.attrs.get("dim", 0)),
            dtype=source_meta.dtype,
            base=base,
        )
    if kind == "where":
        condition_meta, true_meta, false_meta = (
            metadata[value_id] for value_id in step.input_ids
        )
        return where_snippet(
            tuple(condition_meta.shape or ()),
            tuple(true_meta.shape or ()),
            tuple(false_meta.shape or ()),
            condition_dtype=condition_meta.dtype,
            true_dtype=true_meta.dtype,
            false_dtype=false_meta.dtype,
            output_dtype=output_meta.dtype,
            output_shape=tuple(output_meta.shape or ()),
            base=base,
        )
    if kind == "slice":
        slice_kind = step.attrs.get("slice_kind")
        if slice_kind in {"axis", "flat"}:
            shape = (
                tuple(source_meta.shape or ())
                if slice_kind == "axis"
                else (int(step.attrs["source_count"]),)
            )
            return slice_axis_snippet(
                shape,
                int(step.attrs["dim"]) if slice_kind == "axis" else 0,
                int(step.attrs["start"]),
                int(step.attrs.get("step", 1)),
                int(step.attrs["count"]),
                dtype=source_meta.dtype,
                base=base,
            )
        if slice_kind == "index_select":
            index_meta = metadata[step.input_ids[1]]
            return index_select_snippet(
                tuple(source_meta.shape or ()),
                int(step.attrs["dim"]),
                _shape_product(tuple(index_meta.shape or ())),
                dtype=source_meta.dtype,
                index_dtype=index_meta.dtype,
                base=base,
            )
    raise ValueError(f"unsupported captured GLSL kernel kind {kind!r}")


def compile_captured_fused_program(captured) -> str:
    """Compile and cache the shader for one captured numerical region."""

    stages = tuple(getattr(captured, "stages", ()) or ())
    if stages:
        return "\n".join(
            f"// captured stage {index}\n"
            + compile_captured_fused_program(type(captured)(stage, {}))
            for index, stage in enumerate(stages)
        )
    program = captured.program
    kind = (program.extras or {}).get("kernel_kind")
    if kind == "stack":
        step = program.steps[0]
        max_inputs = _compute_limits().max_dispatch_ssbo_blocks - 1
        if len(step.input_ids) > max_inputs:
            metadata = program.meta or {}
            dim = int(step.attrs.get("dim", 0))
            groups = [
                step.input_ids[start:start + max_inputs]
                for start in range(0, len(step.input_ids), max_inputs)
            ]
            sources = []
            partial_shapes = []
            output_dtype = metadata[
                next(iter(program.outputs.values()))
            ].dtype
            for group in groups:
                base_shape = tuple(metadata[group[0]].shape or ())
                _, normalized_dim, partial_shape = _validate_stack_layout(
                    base_shape, len(group), dim
                )
                partial_shapes.append(partial_shape)
                source = emit_stack_source(
                    base_shape,
                    len(group),
                    normalized_dim,
                    input_dtypes=[metadata[value_id].dtype for value_id in group],
                    output_dtype=output_dtype,
                )
                _compile(source)
                sources.append(source)
            cat_source = emit_cat_source(
                partial_shapes,
                normalized_dim,
                input_dtypes=[output_dtype] * len(partial_shapes),
                output_dtype=output_dtype,
            )
            _compile(cat_source)
            sources.append(cat_source)
            return "\n".join(sources)
    source = _emit_captured_fused_program_source(captured)
    _compile(source)
    return source


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
                _broadcast_shape(chunk.shape, out_shape) == out_shape
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
