"""GLSL compute-shader execution target for elementwise primitive programs.

This is the GPU sibling of the C backend. It deliberately mirrors that design
rather than inventing a second one:

    c_backend/ctensor_ops.c      one flat CTensorOp vocabulary + a switch dispatcher
    c_primitive_program.py       PrimitiveProgram: slots, feeds, instructions
    glsl_backend.py  (this)      the same vocabulary + the same program shape,
                                 executed as a fused GLSL compute shader

The one structural difference is where the win comes from. The C interpreter walks
instructions and writes every intermediate slot to memory. A GPU does not want
that: an elementwise program of N instructions compiles to **one shader with N
lines and a single dispatch**, where every intermediate slot is a register-resident
local. Only feeds and the final output ever touch a buffer. So a PrimitiveProgram
is not merely runnable here, it is the natural input format.

Memory model
------------
``GLChunk`` is a chunk of equally-shaped float data that lives on the CPU (numpy),
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

Precision
---------
GPU storage is **float32**. The C backend is double-only; numpy defaults to
float64. Values crossing into this backend are narrowed, and results come back
float32. This is a real, documented contract -- not an accident -- in the same
spirit as ``docs/c_backend_status.md`` being explicit that CTensor is double-only
and has no dtype field. ``assert_close`` in the tests uses tolerances chosen for
float32 accordingly.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

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
    "require_gl_context",
    "gl_context_info",
    "register_context_provider",
    "release_gl_context",
    "GLChunk",
    "GlslInstruction",
    "GlslProgram",
    "emit_program_source",
    "emit_op_source",
    "run_op",
    "execute_program",
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
    # Predicates return 1.0/0.0 floats, matching the C backend's double storage
    # convention (neither backend has a bool dtype).
    "lt": "float($a < $b)",
    "le": "float($a <= $b)",
    "gt": "float($a > $b)",
    "ge": "float($a >= $b)",
    "eq": "float($a == $b)",
    "ne": "float($a != $b)",
}

_UNARY: dict[str, str] = {
    "sqrt": "sqrt($a)",
    "exp": "exp($a)",
    "log": "log($a)",
    "neg": "-$a",
    "abs": "abs($a)",
    # C round() is half-away-from-zero; GLSL round() is permitted to break ties
    # either way (roundEven() is the explicit one). Spell it out so the GPU and
    # the C backend agree on x.5 instead of disagreeing per driver.
    "round": "(sign($a) * floor(abs($a) + 0.5))",
    "trunc": "trunc($a)",
    "floor": "floor($a)",
    "ceil": "ceil($a)",
    "isfinite": "float(!isinf($a) && !isnan($a))",
    "isnan": "float(isnan($a))",
    "isinf": "float(isinf($a))",
    "logical_not": "float($a == 0.0)",
}

# Aliases accepted on the way in, matching c_primitive_program.compile_elementwise_tape.
_ALIASES: dict[str, str] = {
    "div": "truediv",
    "less": "lt",
    "less_equal": "le",
    "greater": "gt",
    "greater_equal": "ge",
    "equal": "eq",
    "not_equal": "ne",
}

GLSL_OPS: frozenset[str] = frozenset(_BINARY) | frozenset(_UNARY)

_LOCAL_SIZE = 256


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


def _expr(op: str, a: str, b: str | None, reverse: bool) -> str:
    if op in _UNARY:
        if b is not None:
            raise ValueError(f"unary op {op!r} given a right operand")
        return _UNARY[op].replace("$a", a)
    if b is None:
        raise ValueError(f"binary op {op!r} missing its right operand")
    left, right = (b, a) if reverse else (a, b)
    return _BINARY[op].replace("$a", left).replace("$b", right)


# ---------------------------------------------------------------------------
# memory chunks
# ---------------------------------------------------------------------------

class GLChunk:
    """An equally-shaped block of float32 resident on the CPU, the GPU, or both.

    Residency is explicit and observable (``.on_cpu`` / ``.on_gpu``) rather than
    implied. Transfers are explicit calls. Nothing here silently moves data
    behind the caller's back, because a hidden readback is a performance cliff
    that is very hard to notice after the fact.
    """

    __slots__ = ("_shape", "_count", "_host", "_buffer", "_owns_buffer", "_gpu_valid")

    def __init__(self, shape: Sequence[int], host: np.ndarray | None = None) -> None:
        self._shape = tuple(int(d) for d in shape)
        self._count = int(np.prod(self._shape)) if self._shape else 1
        self._host = host
        self._buffer: int | None = None
        self._owns_buffer = False
        self._gpu_valid = False

    # -- construction ------------------------------------------------------

    @classmethod
    def from_numpy(cls, array: Any) -> "GLChunk":
        arr = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
        return cls(arr.shape, arr)

    @classmethod
    def zeros(cls, shape: Sequence[int]) -> "GLChunk":
        shape = tuple(int(d) for d in shape)
        return cls(shape, np.zeros(shape, dtype=np.float32))

    @classmethod
    def wrap(cls, buffer_id: int, shape: Sequence[int]) -> "GLChunk":
        """Adopt an SSBO this module did not allocate.

        The interop path: a host that already owns the GL context (a nodus or
        pluck renderer) passes its buffer name and shape, and computation happens
        in place with no system-memory round trip. Ownership stays with the host --
        ``release()`` will not delete a wrapped buffer.
        """
        chunk = cls(shape, None)
        chunk._buffer = int(buffer_id)
        chunk._owns_buffer = False
        chunk._gpu_valid = True
        return chunk

    # -- properties --------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def count(self) -> int:
        return self._count

    @property
    def nbytes(self) -> int:
        return self._count * 4

    @property
    def on_cpu(self) -> bool:
        return self._host is not None

    @property
    def on_gpu(self) -> bool:
        return self._buffer is not None and self._gpu_valid

    @property
    def buffer_id(self) -> int | None:
        return self._buffer

    # -- transfer ----------------------------------------------------------

    def to_gpu(self) -> "GLChunk":
        """Ensure GPU residency, allocating and uploading if needed."""
        require_gl_context()
        from OpenGL import GL

        if self._buffer is None:
            self._buffer = int(GL.glGenBuffers(1))
            self._owns_buffer = True
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buffer)
            GL.glBufferData(
                GL.GL_SHADER_STORAGE_BUFFER, self.nbytes, None, GL.GL_DYNAMIC_DRAW
            )
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            self._gpu_valid = False

        if not self._gpu_valid:
            if self._host is None:
                # Allocated but never written and nothing to upload: it is an
                # output slot. Leave contents undefined but mark it live.
                self._gpu_valid = True
                return self
            data = np.ascontiguousarray(self._host, dtype=np.float32)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buffer)
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, self.nbytes, data)
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
            self._gpu_valid = True
        return self

    def to_cpu(self) -> np.ndarray:
        """Read back into host memory and return the host array."""
        if not self.on_gpu:
            if self._host is None:
                raise RuntimeError("chunk has no CPU data and no live GPU buffer")
            return self._host
        require_gl_context()
        from OpenGL import GL

        out = np.empty(self._count, dtype=np.float32)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, self._buffer)
        GL.glGetBufferSubData(
            GL.GL_SHADER_STORAGE_BUFFER, 0, self.nbytes,
            out.ctypes.data_as(ctypes.c_void_p),
        )
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, 0)
        self._host = out.reshape(self._shape) if self._shape else out
        return self._host

    def numpy(self) -> np.ndarray:
        """Host view, reading back from the GPU when that is the live copy."""
        if self.on_gpu:
            return self.to_cpu()
        if self._host is None:
            raise RuntimeError("chunk has no data")
        return self._host

    def release(self) -> None:
        """Delete the GL buffer if we allocated it. Wrapped buffers are left alone."""
        if self._buffer is not None and self._owns_buffer:
            try:
                from OpenGL import GL
                GL.glDeleteBuffers(1, [self._buffer])
            except Exception:
                pass  # context may already be gone during interpreter teardown
        self._buffer = None
        self._owns_buffer = False
        self._gpu_valid = False

    def __repr__(self) -> str:
        where = []
        if self.on_cpu:
            where.append("cpu")
        if self.on_gpu:
            where.append("gpu")
        return f"GLChunk(shape={self._shape}, on={'+'.join(where) or 'none'})"


# ---------------------------------------------------------------------------
# program IR
#
# Structurally identical to c_primitive_program.PrimitiveInstruction/Program. It
# is redefined rather than imported because importing that module pulls in the
# CFFI C library build, and the GPU path must not require a working C toolchain.
# ``from_c_program`` adapts one without importing it.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GlslInstruction:
    """One equally-shaped elementwise primitive operation."""

    op: str
    out_slot: int
    left_slot: int
    right_slot: int | None = None
    right_scalar: float | None = None
    reverse: bool = False


@dataclass(frozen=True)
class GlslProgram:
    """A validated elementwise program lowered to a single fused shader."""

    instructions: Sequence[GlslInstruction]
    feed_count: int
    slot_count: int
    output_slot: int

    @classmethod
    def from_c_program(cls, program: Any) -> "GlslProgram":
        """Adapt a ``c_primitive_program.PrimitiveProgram`` (duck-typed)."""
        return cls(
            instructions=tuple(
                GlslInstruction(
                    op=i.op,
                    out_slot=i.out_slot,
                    left_slot=i.left_slot,
                    right_slot=getattr(i, "right_slot", None),
                    right_scalar=getattr(i, "right_scalar", None),
                    reverse=bool(getattr(i, "reverse", False)),
                )
                for i in program.instructions
            ),
            feed_count=int(program.feed_count),
            slot_count=int(program.slot_count),
            output_slot=int(program.output_slot),
        )

    def validate(self) -> None:
        if self.slot_count < self.feed_count:
            raise ValueError("slot_count must be at least feed_count")
        if not 0 <= self.output_slot < self.slot_count:
            raise ValueError("output_slot out of range")
        defined = set(range(self.feed_count))
        for n, ins in enumerate(self.instructions):
            canonical_op(ins.op)  # raises GLSLUnsupportedOp on an unknown op
            if ins.right_slot is not None and ins.right_scalar is not None:
                raise ValueError(
                    f"instruction {n}: cannot have both slot and scalar right operands"
                )
            for slot in (ins.left_slot, ins.right_slot):
                if slot is None:
                    continue
                if slot not in defined:
                    raise ValueError(
                        f"instruction {n} reads slot {slot} before it is written"
                    )
            if not 0 <= ins.out_slot < self.slot_count:
                raise ValueError(f"instruction {n}: out_slot out of range")
            defined.add(ins.out_slot)
        if self.output_slot not in defined:
            raise ValueError("output_slot is never written")


# ---------------------------------------------------------------------------
# shader emission
# ---------------------------------------------------------------------------

_SHADER_HEADER = """#version 430
// GENERATED by turing glsl_backend.emit_program_source -- do not edit by hand.
//
// Fused elementwise program: every intermediate slot is a local (a register),
// so only feeds and the single output ever touch memory. This is the whole
// reason a PrimitiveProgram is worth running on a GPU rather than instruction
// by instruction.
layout(local_size_x = {local_size}) in;
"""


def emit_program_source(program: GlslProgram, local_size: int = _LOCAL_SIZE) -> str:
    """Lower a whole elementwise program to one compute shader."""
    program.validate()
    lines: list[str] = [_SHADER_HEADER.format(local_size=local_size)]

    for i in range(program.feed_count):
        lines.append(
            f"layout(std430, binding = {i}) readonly buffer Feed{i} "
            f"{{ float feed{i}[]; }};"
        )
    out_binding = program.feed_count
    lines.append(
        f"layout(std430, binding = {out_binding}) writeonly buffer OutBuf "
        f"{{ float outbuf[]; }};"
    )
    lines.append("")
    lines.append("uniform uint u_count;")
    lines.append("")
    lines.append("void main() {")
    lines.append("    uint gid = gl_GlobalInvocationID.x;")
    lines.append("    if (gid >= u_count) { return; }")

    for i in range(program.feed_count):
        lines.append(f"    float s{i} = feed{i}[gid];")

    for ins in program.instructions:
        op, reverse = canonical_op(ins.op)
        reverse = reverse or ins.reverse
        a = f"s{ins.left_slot}"
        if op in _UNARY:
            b = None
        elif ins.right_slot is not None:
            b = f"s{ins.right_slot}"
        elif ins.right_scalar is not None:
            b = _glsl_float(ins.right_scalar)
        else:
            raise ValueError(f"binary op {ins.op!r} has no right operand")
        lines.append(f"    float s{ins.out_slot} = {_expr(op, a, b, reverse)};")

    lines.append(f"    outbuf[gid] = s{program.output_slot};")
    lines.append("}")
    return "\n".join(lines) + "\n"


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


def emit_op_source(op: str, *, scalar: float | None = None,
                   local_size: int = _LOCAL_SIZE) -> str:
    """Lower a single op to a shader -- the ``_apply_operator__`` fast path."""
    name, reverse = canonical_op(op)
    if name in _UNARY:
        program = GlslProgram((GlslInstruction(name, 1, 0),), 1, 2, 1)
    elif scalar is not None:
        program = GlslProgram(
            (GlslInstruction(name, 1, 0, right_scalar=scalar, reverse=reverse),), 1, 2, 1
        )
    else:
        program = GlslProgram(
            (GlslInstruction(name, 2, 0, right_slot=1, reverse=reverse),), 2, 3, 2
        )
    return emit_program_source(program, local_size=local_size)


# ---------------------------------------------------------------------------
# compilation + cache
# ---------------------------------------------------------------------------

_program_cache: dict[str, int] = {}
_cache_stats = {"hits": 0, "misses": 0}


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
# ---------------------------------------------------------------------------

def _dispatch(program_id: int, chunks: Sequence[GLChunk], out: GLChunk,
              count: int, local_size: int = _LOCAL_SIZE) -> None:
    from OpenGL import GL

    GL.glUseProgram(program_id)
    loc = GL.glGetUniformLocation(program_id, "u_count")
    if loc != -1:
        GL.glUniform1ui(loc, count)

    for binding, chunk in enumerate(chunks):
        chunk.to_gpu()
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, binding, chunk.buffer_id)
    out.to_gpu()
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, len(chunks), out.buffer_id)

    groups = (count + local_size - 1) // local_size
    GL.glDispatchCompute(groups, 1, 1)
    # Without this the readback may observe stale memory. It is the GPU analogue
    # of the substrate-visibility problems in research/06: correct-looking code
    # over data that was never actually made visible.
    GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_BUFFER_UPDATE_BARRIER_BIT)

    for binding in range(len(chunks) + 1):
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, binding, 0)
    GL.glUseProgram(0)


def execute_program(
    program: GlslProgram,
    feeds: Sequence[Any],
    *,
    out: GLChunk | None = None,
) -> GLChunk:
    """Run an elementwise program as one fused dispatch.

    ``out`` may be a caller-owned chunk of the shared shape. Supplying one
    keeps both input and output buffers resident across repeated dispatches,
    which is the stateful path used by renderers and native hosts. A fresh
    output is otherwise allocated without uploading meaningless zeroes: every
    invocation writes all ``count`` elements before the chunk is observable.
    """
    require_gl_context()
    program.validate()
    if len(feeds) != program.feed_count:
        raise ValueError(
            f"expected {program.feed_count} feeds, received {len(feeds)}"
        )
    if not feeds:
        raise ValueError("a primitive program needs at least one feed")

    chunks = [f if isinstance(f, GLChunk) else GLChunk.from_numpy(f) for f in feeds]
    shape = chunks[0].shape
    for chunk in chunks[1:]:
        if chunk.shape != shape:
            raise ValueError(
                f"program feeds must share one shape; got {shape} and {chunk.shape}"
            )

    if out is None:
        out = GLChunk(shape)
    elif out.shape != shape:
        raise ValueError(
            f"output must share the feed shape; got {out.shape} and {shape}"
        )
    _dispatch(_compile(emit_program_source(program)), chunks, out, chunks[0].count)
    return out


def run_op(op: str, left: Any, right: Any = None) -> GLChunk:
    """Execute a single op -- the ``_apply_operator__`` analogue.

    ``right`` may be a chunk/array of matching shape, a Python scalar, or None
    for unary ops.
    """
    require_gl_context()
    name, _ = canonical_op(op)
    lhs = left if isinstance(left, GLChunk) else GLChunk.from_numpy(left)

    if name in _UNARY:
        if right is not None:
            raise ValueError(f"unary op {op!r} given a right operand")
        source = emit_op_source(op)
        feeds: list[GLChunk] = [lhs]
    elif isinstance(right, (int, float, np.floating, np.integer)) and not isinstance(right, bool):
        source = emit_op_source(op, scalar=float(right))
        feeds = [lhs]
    elif right is None:
        raise ValueError(f"binary op {op!r} requires a right operand")
    else:
        rhs = right if isinstance(right, GLChunk) else GLChunk.from_numpy(right)
        if rhs.shape != lhs.shape:
            raise ValueError(
                f"operands must share one shape; got {lhs.shape} and {rhs.shape}"
            )
        source = emit_op_source(op)
        feeds = [lhs, rhs]

    out = GLChunk(lhs.shape)
    _dispatch(_compile(source), feeds, out, lhs.count)
    return out
