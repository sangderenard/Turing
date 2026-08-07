"""Emit WebAssembly -- both the readable text and the runnable binary -- from
a ``FusedProgram``.

.. warning::

    **FULLY OFF-LIMITS INTERNAL IMPLEMENTATION.**

    **FUSED PROGRAM IS NOT THE COMPILER.**  This module is not a compiler
    entrypoint, public API, architecture description, or capability boundary.
    Do not inspect, invoke, reference, modify, or use this file to infer any
    compiler capability or limitation unless the task explicitly names this
    internal backend implementation.

No toolchain is required, and that includes producing something runnable.
WAT is the human-readable form; a browser only ever executes the binary, so
emitting text alone would leave every compiled program readable but
unrunnable unless ``wat2wasm`` happened to be installed. ``wasm_binary.py``
assembles the module here instead, from this same lowering, so ``.wat`` and
``.wasm`` describe the same program by construction. ``compile_wat`` remains
for callers who would rather round-trip through WABT.

Layout: every array is a byte offset into the module's exported linear
memory, passed as an ``i32`` parameter, in the order the API descriptor
records. The caller owns memory -- it writes feeds in and reads the output
back -- because a fused elementwise program has no state of its own.

**WebAssembly has no transcendental instructions.** ``exp``, ``log``, the
trigonometric family and ``pow`` are simply not in the instruction set; there
is no lowering for them that is not a hand-written polynomial. They are
reported as named shortfalls rather than approximated silently, so a program
containing one fails to emit instead of returning a plausible wrong number.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..common.tensors.fused_ir import (
    ELEMENTWISE_BINARY,
    ELEMENTWISE_UNARY,
    FusedProgram,
    OpStep,
    flatten_tensor_constant,
    ordered_feed_ids,
    resolve_view_source,
    uniform_tensor_constant,
    unroll_feed_axis_reductions,
    view_offset_stride,
)
from .wasm_binary import (
    OP_F32_CONVERT_I32_S,
    OP_F32_CONVERT_I64_S,
    OP_F64_CONVERT_I32_S,
    OP_F64_CONVERT_I64_S,
    OP_I32_TRUNC_F32_S,
    OP_I32_TRUNC_F64_S,
    OP_I64_TRUNC_F32_S,
    OP_I64_TRUNC_F64_S,
)


class WasmEmissionError(ValueError):
    """The program cannot be expressed in WebAssembly."""


@dataclass(frozen=True)
class WasmShortfall:
    """One operation with no WebAssembly instruction behind it."""

    step_id: int
    op_name: str
    reason: str

    def format(self) -> str:
        return f"step {self.step_id} ({self.op_name}): {self.reason}"


# Numeric types, as (WAT type, element bytes, load, store).
_TYPES: dict[str, tuple[str, int, str, str]] = {
    "float64": ("f64", 8, "f64.load", "f64.store"),
    "f64": ("f64", 8, "f64.load", "f64.store"),
    "double": ("f64", 8, "f64.load", "f64.store"),
    "float32": ("f32", 4, "f32.load", "f32.store"),
    "f32": ("f32", 4, "f32.load", "f32.store"),
    "float": ("f32", 4, "f32.load", "f32.store"),
    # Integer working types. Until these existed this module could only
    # compile a program whose *own* arithmetic was floating point --
    # integers were supported solely as separate memory buffers to convert
    # in and out of (``_MEMORY_DTYPE_OPS`` below). That is fine for a
    # numeric kernel and wrong for a program that is genuinely integral:
    # a decoder, a register file, a state machine. Such a program has no
    # meaningful f64 working type -- ``//`` and ``%`` are not float
    # operations, bit masks are not float operations, and a 64-bit value
    # does not survive f64's 2**53 exact-integer range. WebAssembly has
    # native i32/i64 arithmetic; this simply stops pretending it does not.
    "int64": ("i64", 8, "i64.load", "i64.store"),
    "i64": ("i64", 8, "i64.load", "i64.store"),
    "long": ("i64", 8, "i64.load", "i64.store"),
    "int32": ("i32", 4, "i32.load", "i32.store"),
    "i32": ("i32", 4, "i32.load", "i32.store"),
    "int": ("i32", 4, "i32.load", "i32.store"),
}

# Which working types are integral. Everything that differs between the two
# families keys off this rather than re-parsing the type name at each site.
_INTEGER_VALUE_TYPES = frozenset({"i32", "i64"})


def _is_integer_type(value_type: str) -> bool:
    return str(value_type) in _INTEGER_VALUE_TYPES


# The saturating bounds of each integer working type. A min/max reduction's
# fold identity is spelled as an infinity (``_REDUCE_FOLD``), which has no
# integer literal -- but the *meaning* of that identity, "a value no operand
# can beat", is exactly the type's extreme, so an integral fold uses that
# rather than refusing or overflowing on int(float("inf")).
_INTEGER_BOUNDS: dict[str, tuple[int, int]] = {
    "i32": (-(2 ** 31), 2 ** 31 - 1),
    "i64": (-(2 ** 63), 2 ** 63 - 1),
}


def _typed_constant(value_type: str, value: Any) -> str:
    """One scalar literal, spelled for its working type.

    ``f64.const 0`` and ``i64.const 0.0`` are both invalid WAT, so the
    literal's spelling has to follow the type rather than the Python value's
    own repr.
    """

    if _is_integer_type(value_type):
        numeric = float(value)
        low, high = _INTEGER_BOUNDS[str(value_type)]
        if numeric == float("inf"):
            return f"{value_type}.const {high}"
        if numeric == float("-inf"):
            return f"{value_type}.const {low}"
        return f"{value_type}.const {int(numeric)}"
    return f"{value_type}.const {float(value)!r}"

# A parameter whose real dtype is not the module's f32/f64 working type
# still gets loaded and stored at its own byte width and WASM memory
# instruction -- not silently reinterpreted as the working type's bytes.
# "bool"/"logical" are deliberately absent: that tag arises naturally on an
# ordinary comparison op's result flowing through this module's own uniform
# elementwise array (Meta.dtype records what an operation *produced*, not
# only what a caller declared), and narrowing it to 1 byte here corrupts
# that array's stride for every other value sharing it. This table exists
# for a caller-declared, genuinely separate byte/integer buffer (a register
# file, raw memory) -- not for a value this module computed for itself.
# (byte_width, load_instruction, store_instruction, wasm_value_kind)
_MEMORY_DTYPE_OPS: dict[str, tuple[int, str, str, str]] = {
    "uint8": (1, "i32.load8_u", "i32.store8", "i32"),
    "u8": (1, "i32.load8_u", "i32.store8", "i32"),
    "int32": (4, "i32.load", "i32.store", "i32"),
    "i32": (4, "i32.load", "i32.store", "i32"),
    "int": (4, "i32.load", "i32.store", "i32"),
    "int64": (8, "i64.load", "i64.store", "i64"),
    "i64": (8, "i64.load", "i64.store", "i64"),
}

# Converting an integer parameter to/from the module's f32/f64 working type
# for arithmetic. Signed conversion is correct even for the unsigned kinds
# above (uint8/bool load already zero-extends to a non-negative i32, and a
# non-negative i32/i64 converts identically whether treated as signed or
# unsigned), so one conversion table covers both.
_TO_WORKING_TYPE = {
    ("i32", "f64"): "f64.convert_i32_s",
    ("i32", "f32"): "f32.convert_i32_s",
    ("i64", "f64"): "f64.convert_i64_s",
    ("i64", "f32"): "f32.convert_i64_s",
}
_FROM_WORKING_TYPE = {
    ("f64", "i32"): "i32.trunc_f64_s",
    ("f32", "i32"): "i32.trunc_f32_s",
    ("f64", "i64"): "i64.trunc_f64_s",
    ("f32", "i64"): "i64.trunc_f32_s",
}

# Same conversions as raw opcodes, for the binary assembler below -- the WAT
# text and the binary are two independent emitters describing one program;
# both need the same conversions, one as mnemonic text, one as opcode bytes.
_TO_WORKING_OPCODE = {
    ("i32", "f64"): OP_F64_CONVERT_I32_S,
    ("i32", "f32"): OP_F32_CONVERT_I32_S,
    ("i64", "f64"): OP_F64_CONVERT_I64_S,
    ("i64", "f32"): OP_F32_CONVERT_I64_S,
}
_FROM_WORKING_OPCODE = {
    ("f64", "i32"): OP_I32_TRUNC_F64_S,
    ("f32", "i32"): OP_I32_TRUNC_F32_S,
    ("f64", "i64"): OP_I64_TRUNC_F64_S,
    ("f32", "i64"): OP_I64_TRUNC_F32_S,
}

# Operations that are one native instruction, given the value type prefix.
_BINARY_INSTRUCTION = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "truediv": "div",
    "minimum": "min",
    "maximum": "max",
}

_UNARY_INSTRUCTION = {
    "neg": "neg",
    "abs": "abs",
    "sqrt": "sqrt",
    "floor": "floor",
    "ceil": "ceil",
    "trunc": "trunc",
    "round": "nearest",
}

# Integer counterparts. WebAssembly's integer instruction set is not the
# float one with a different prefix, so these cannot be derived from the
# tables above:
#
#   * division and remainder need an explicit signedness (``div_s``), where
#     float division does not. Signed is correct here because every dtype
#     this module maps to an integer working type is signed; ``uint8`` is a
#     memory dtype (``_MEMORY_DTYPE_OPS``) that loads zero-extended into a
#     non-negative i32, never a working type of its own.
#   * ``min``/``max`` genuinely do not exist for integers in WebAssembly --
#     they are lowered from a comparison plus ``select`` in
#     ``_integer_binary_instructions`` rather than listed here.
#   * ``mod``/``floordiv`` become expressible for the first time. They sit in
#     ``_NO_WASM_INSTRUCTION`` below, whose comment already anticipated this
#     exact unlock ("mod/floordiv await an integer-remainder lowering"); the
#     integer working type is that lowering, so they are filtered back out
#     of the unsupported set when the working type is integral.
_INTEGER_BINARY_INSTRUCTION = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "truediv": "div_s",
    "floordiv": "div_s",
    "mod": "rem_s",
}

_INTEGER_COMPARISON_INSTRUCTION = {
    "less": "lt_s",
    "less_equal": "le_s",
    "greater": "gt_s",
    "greater_equal": "ge_s",
    "equal": "eq",
    "not_equal": "ne",
}

# Integer ops this module reaches by composition rather than one opcode.
_INTEGER_COMPOSED_BINARY = frozenset({"minimum", "maximum"})

# Boolean connectives over integers. WebAssembly has bitwise and/or, which
# are *not* these: `2 & 1` is 0 while `bool(2) and bool(1)` is true. So each
# operand is first reduced to a 0/1 truth value with ``eqz`` (which tests
# "is zero", so the sense is inverted once and corrected by De Morgan) and
# the connective is then applied to those. This keeps a predicate mask
# exactly 0/1, which is what every other backend reports for these ops and
# what ``select``-style lowerings downstream assume.
_INTEGER_LOGICAL_BINARY = frozenset({"logical_and", "logical_or"})
_INTEGER_LOGICAL_UNARY = frozenset({"logical_not"})

# Float rounding that is the identity on an integer, so an integral program
# asking for it is satisfied by doing nothing rather than being refused.
_INTEGER_IDENTITY_UNARY = frozenset({"floor", "ceil", "trunc", "round"})

# Comparisons return i32 0/1 in WebAssembly, so the result is converted back
# to the value type -- every other backend in this repository reports a
# comparison as 0.0/1.0 in the operand's own type, and the torture matrix
# compares outputs numerically rather than as booleans.
_COMPARISON_INSTRUCTION = {
    "less": "lt",
    "less_equal": "le",
    "greater": "gt",
    "greater_equal": "ge",
    "equal": "eq",
    "not_equal": "ne",
}

# Named so the failure explains itself rather than reading as an oversight.
# Genuinely out of reach: no instruction, and no table either. tan has poles
# inside any interval worth covering, so no bounded table describes it -- a
# program wanting it should divide sin by cos and decide for itself what
# happens near the pole. mod/floordiv await an integer-remainder lowering;
# the predicates return a boolean mask this elementwise float pass does not
# model yet. (sign and pow ARE lowered -- see _step_instructions / _assemble.)
_NO_WASM_INSTRUCTION = {
    "mod", "floordiv", "tan",
    "isfinite", "isnan", "isinf", "logical_not",
}
# Reachable through a baked table rather than an instruction. Taken from
# the catalogue itself so the two cannot drift apart: adding a function to
# wasm_math_tables makes it emittable here without a second edit. f64 only --
# an f32 table would need its own sampling and has no caller yet.
def _tabulated_ops() -> frozenset[str]:
    from .wasm_math_tables import TABULATED

    return TABULATED


_LUT_OPS = _tabulated_ops()


@dataclass(frozen=True)
class WasmModule:
    """Emitted WAT plus whatever could not be expressed."""

    name: str
    source: str
    shortfalls: tuple[WasmShortfall, ...] = ()
    parameters: tuple[str, ...] = ()
    value_type: str = "f64"
    api: Any = None
    # The assembled module. Emitted here rather than left to wat2wasm: a
    # browser only executes the binary, so without this the program could be
    # read but never run. Built from this same lowering, so the two forms
    # cannot disagree.
    binary: bytes | None = None

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def shortfall_report(self) -> str:
        return "WebAssembly emission shortfalls:\n" + "\n".join(
            "- " + s.format() for s in self.shortfalls
        )

    def write(self, directory: str | Path) -> Path:
        path = Path(directory) / f"{self.name}.wat"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.source, encoding="utf-8")
        if self.api is not None:
            self.api.write(path.with_suffix(".api.yaml"))
        if self.binary is not None:
            path.with_suffix(".wasm").write_bytes(self.binary)
        return path


def program_feed_order(program: FusedProgram) -> tuple[int, ...]:
    """Feed order as the program itself uses them, not as their ids sort.

    ``ordered_feed_ids`` sorts by value id, and a value id is an allocation
    address -- so for a program with more than one feed the parameter order
    was effectively arbitrary. A caller then wrote its first array into
    whichever parameter happened to sort first, which does not fail: it
    computes a wrong answer from correctly-shaped inputs. Measured on an
    11-input network, sorted order was a scrambled permutation of the
    source's own parameter order.

    First use is the program's own statement about order, and for anything
    the AOT front end emits it is source order, because the first thing a
    program does with an argument is in the order the arguments appear.
    Feeds nothing reads are appended, so the count still matches.
    """

    feeds = set(program.feeds)
    order: list[int] = []
    for step in program.steps:
        for value_id in step.input_ids:
            if value_id in feeds and value_id not in order:
                order.append(value_id)
    order.extend(sorted(feeds - set(order)))
    return tuple(order)


def required_steps(program: FusedProgram) -> list[OpStep]:
    """The steps the requested outputs actually depend on, in program order.

    A captured program records every value the observation produced, and an
    AST-compiled one keeps intermediates no output reads -- a comparison's
    recorded result sitting beside the live comparison that produced it, for
    instance. Emitting those is not wrong, but it costs locals and, for an
    array-valued constant, would demand space in linear memory for something
    nothing reads. Same traversal c_jit_backend._required_nodes performs for
    the C and Fortran backends.
    """

    producers = {step.result_id: step for step in program.steps}
    required: set[int] = set()
    stack = list(program.outputs.values())
    while stack:
        value_id = stack.pop()
        if value_id in required:
            continue
        step = producers.get(value_id)
        if step is None:
            continue
        required.add(value_id)
        stack.extend(step.input_ids)
    return [step for step in program.steps if step.result_id in required]


def _flat_sum_steps(live: Sequence[OpStep]) -> tuple[OpStep, ...]:
    """Validate and return whole-tensor ``sum`` reductions in dependency order.

    SSA already names and specifies this operator.  The Wasm backend merely
    supplies its storage/control implementation: one counted pass accumulates
    the input tensor into a scalar local, which subsequent elementwise steps
    naturally broadcast.  Axis reductions need shape/stride information and
    remain explicit shortfalls until that memory contract reaches this IR.
    """

    return tuple(step for step in live if step.op_name == "sum")


def _sum_dependencies(step: OpStep, live: Sequence[OpStep]) -> tuple[OpStep, ...]:
    producers = {candidate.result_id: candidate for candidate in live}
    required: set[int] = set()
    stack = list(step.input_ids)
    while stack:
        value_id = stack.pop()
        producer = producers.get(value_id)
        if producer is None or producer.result_id in required:
            continue
        required.add(producer.result_id)
        stack.extend(producer.input_ids)
    return tuple(candidate for candidate in live if candidate.result_id in required)


# One axis reduction folds a trailing axis away. The accumulator starts at the
# identity and each grid element is combined with the value-type instruction
# named here; ``mean`` uses the ``sum`` fold and divides by K afterwards.
_REDUCE_FOLD: dict[str, tuple[str, float]] = {
    "sum": ("add", 0.0),
    "mean": ("add", 0.0),
    "prod": ("mul", 1.0),
    "min": ("min", float("inf")),
    "amin": ("min", float("inf")),
    "max": ("max", float("-inf")),
    "amax": ("max", float("-inf")),
}


def _shape_product(shape: Any) -> int | None:
    if shape is None:
        return None
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


@dataclass(frozen=True)
class _AxisReductionPlan:
    """How to iterate a program whose outputs survive a trailing-axis reduce.

    An axis reduction is a rank change (N*K -> N) the flat ``run(count, ...)``
    model cannot express directly.  It is lowered as a nested walk: the outer
    loop iterates the surviving extent N -- the independent, per-output lane,
    exactly as parallel as an elementwise pass -- and an inner counted loop of
    K accumulates the reduced axis.  Keeping N as the outer lane is deliberate:
    a materialised pre-reduction segment stays parallel over N, so it can still
    be scored as a parallel deployment region downstream; only the K fold is
    serial.  ``ok`` is False when the program contains an axis reduction this
    backend cannot yet express (the reasons are recorded as shortfalls).
    """

    ok: bool
    axis_extent: int
    reductions: tuple[OpStep, ...]
    reduce_op: dict[int, str]
    inner_dependencies: dict[int, tuple[OpStep, ...]]
    pre_steps: tuple[OpStep, ...]
    post_steps: tuple[OpStep, ...]
    value_class: dict[int, str]
    feed_class: dict[int, str]
    grid_output_steps: tuple[OpStep, ...]


def _plan_axis_reductions(
    program: FusedProgram,
    live: Sequence[OpStep],
    feed_ids: Sequence[int],
    shortfalls: list[WasmShortfall],
) -> _AxisReductionPlan | None:
    """Classify one reduction program, or return None when it has no axis reduce.

    Shared by both emitters so the WAT and binary forms cannot disagree about
    which values are grid/row/kaxis/scalar or which steps run in the inner
    fold.  Records a shortfall (and returns an ``ok=False`` plan) for every
    reduction shape this backend cannot express, so emission fails loudly
    rather than returning a plausible wrong number.
    """

    reductions = tuple(
        step for step in live if step.attrs.get("axis") is not None
    )
    if not reductions:
        return None

    def fail(step_id: int, op: str, reason: str) -> _AxisReductionPlan:
        shortfalls.append(WasmShortfall(step_id, op, reason))
        return _AxisReductionPlan(
            ok=False, axis_extent=0, reductions=reductions, reduce_op={},
            inner_dependencies={}, pre_steps=(), post_steps=(),
            value_class={}, feed_class={}, grid_output_steps=(),
        )

    meta = program.meta or {}
    whole_sums = tuple(
        step for step in live
        if step.op_name == "sum" and step.attrs.get("axis") is None
    )
    if whole_sums:
        return fail(
            reductions[0].step_id, "reduce",
            "whole-tensor and axis reductions in one program are not lowered "
            "together",
        )

    extents_n: set[int] = set()
    extents_k: set[int] = set()
    reduce_op: dict[int, str] = {}
    for step in reductions:
        if not step.input_ids:
            return fail(step.step_id, step.op_name, "reduction has no operand")
        entry = meta.get(step.input_ids[0])
        shape = tuple(entry.shape) if entry is not None and entry.shape is not None else None
        if shape is None:
            return fail(
                step.step_id, step.op_name,
                "axis reduction input has no shape metadata",
            )
        axis = int(step.attrs["axis"])
        norm = axis if axis >= 0 else len(shape) + axis
        if norm != len(shape) - 1:
            return fail(
                step.step_id, step.op_name,
                f"only trailing-axis (axis=-1) reduction is lowered; got "
                f"axis={axis} on shape {shape}",
            )
        k = int(shape[-1])
        total = _shape_product(shape) or 0
        op = str(step.attrs.get("reduce_op") or step.op_name)
        if op not in _REDUCE_FOLD:
            return fail(
                step.step_id, step.op_name,
                f"reduce_op {op!r} has no WebAssembly accumulator",
            )
        reduce_op[step.result_id] = op
        extents_k.add(k)
        extents_n.add(total // k if k else 0)

    if len(extents_k) != 1 or len(extents_n) != 1:
        return fail(
            reductions[0].step_id, "reduce",
            f"mixed reduction extents N={sorted(extents_n)} K={sorted(extents_k)} "
            "are not lowered in one program",
        )
    outer_n = next(iter(extents_n))
    axis_k = next(iter(extents_k))
    if outer_n == axis_k:
        return fail(
            reductions[0].step_id, "reduce",
            f"ambiguous reduction where N == K == {outer_n}; broadcasting "
            "cannot be classified by extent alone",
        )

    def classify(value_id: int) -> str:
        entry = meta.get(value_id)
        product = _shape_product(entry.shape) if entry is not None else None
        if product is None:
            return "unknown"
        if product == outer_n * axis_k:
            return "grid"
        if product == outer_n:
            return "row"
        if product == axis_k:
            return "kaxis"
        if product == 1:
            return "scalar"
        return "other"

    value_class: dict[int, str] = {}
    for value_id in feed_ids:
        value_class[value_id] = classify(value_id)
    for step in live:
        value_class[step.result_id] = classify(step.result_id)

    reduction_results = {step.result_id for step in reductions}
    producers = {step.result_id: step for step in live}

    def depends_on_reduction(step: OpStep) -> bool:
        seen: set[int] = set()
        stack = list(step.input_ids)
        while stack:
            value_id = stack.pop()
            if value_id in seen:
                continue
            seen.add(value_id)
            if value_id in reduction_results:
                return True
            producer = producers.get(value_id)
            if producer is not None:
                stack.extend(producer.input_ids)
        return False

    feed_class: dict[int, str] = {}
    for value_id in feed_ids:
        klass = value_class.get(value_id, "unknown")
        if klass == "grid":
            return fail(
                -1, "feed",
                f"feed {value_id} is grid-shaped (N*K); the count-based ABI "
                "cannot size it yet",
            )
        if klass in ("other", "unknown"):
            return fail(
                -1, "feed",
                f"feed {value_id} has a shape that is not row/kaxis/scalar "
                "relative to the reduction extents",
            )
        feed_class[value_id] = klass

    pre_steps: list[OpStep] = []
    post_steps: list[OpStep] = []
    inner_dependencies: dict[int, tuple[OpStep, ...]] = {}
    for step in live:
        if step.result_id in reduction_results:
            continue
        klass = value_class.get(step.result_id, "unknown")
        if klass in ("grid", "kaxis"):
            # Inner-scope steps are emitted per reduction, in the fold loop.
            # A grid/kaxis value may depend on an earlier reduction's row
            # result (e.g. mask = dist <= dist.min(axis)): that reduction is
            # emitted first and its result stays live in the outer scope, so a
            # later reduction's inner loop can read it as a per-row scalar.
            continue
        if klass == "other" or klass == "unknown":
            return fail(
                step.step_id, step.op_name,
                f"value {step.result_id} has an unclassifiable shape relative "
                "to the reduction extents",
            )
        if depends_on_reduction(step):
            post_steps.append(step)
        else:
            pre_steps.append(step)

    for step in reductions:
        deps = _sum_dependencies(step, live)
        inner = tuple(
            dep for dep in deps
            if dep.result_id not in reduction_results
            and value_class.get(dep.result_id) in ("grid", "kaxis")
        )
        inner_dependencies[step.result_id] = inner

    # Grid/kaxis outputs need their own K-walk to be written at every (i, k).
    # Gather the grid/kaxis steps that feed them (row/scalar ancestors are
    # already computed once in the outer scope, so recursion stops there).
    grid_output_ids = [
        output_id for output_id in program.outputs.values()
        if value_class.get(output_id) in ("grid", "kaxis")
    ]
    grid_needed: set[int] = set()
    grid_stack = list(grid_output_ids)
    while grid_stack:
        value_id = grid_stack.pop()
        if value_id in grid_needed:
            continue
        if value_class.get(value_id) not in ("grid", "kaxis"):
            continue
        producer = producers.get(value_id)
        if producer is None:
            continue
        grid_needed.add(value_id)
        grid_stack.extend(producer.input_ids)
    grid_output_steps = tuple(
        step for step in live if step.result_id in grid_needed
    )

    return _AxisReductionPlan(
        ok=True,
        axis_extent=axis_k,
        reductions=reductions,
        reduce_op=reduce_op,
        inner_dependencies=inner_dependencies,
        pre_steps=tuple(pre_steps),
        post_steps=tuple(post_steps),
        value_class=value_class,
        feed_class=feed_class,
        grid_output_steps=grid_output_steps,
    )


def _constant_scalar(step: OpStep) -> float | None:
    """The scalar a ``tensor_from_list`` step contributes, if it is one.

    An AST-compiled program carries its literals as recorded constructor
    steps rather than as inline scalars (see c_primitive_program: turning
    one back into a Python number would contradict AbstractTensor's own type
    decision). A one-element constant is still just a number here, and
    materialising it as an array in linear memory would be wasteful, so it
    becomes a constant local instead.
    """

    if step.op_name != "tensor_from_list":
        return None
    return uniform_tensor_constant(step.attrs.get("values"))


def _value_type(program: FusedProgram, dtype: str | None) -> tuple[str, int, str, str]:
    if dtype is None:
        meta = program.meta or {}
        for value_id in ordered_feed_ids(program):
            entry = meta.get(value_id)
            if entry is not None and entry.dtype:
                dtype = str(entry.dtype)
                break
    resolved = _TYPES.get(str(dtype or "float64"))
    if resolved is None:
        raise WasmEmissionError(
            f"no WebAssembly value type for dtype {dtype!r}; "
            f"one of {sorted(set(_TYPES))}"
        )
    return resolved


def _emit_reduction_body_wat(
    body: list[str],
    plan: _AxisReductionPlan,
    names: Mapping[int, str],
    feed_label: Mapping[int, str],
    feed_ids: Sequence[int],
    value_type: str,
    load: str,
    store: str,
    element_bytes: int,
    static_data: Mapping[str, Any],
    shortfalls: list[WasmShortfall],
    output_ids: Sequence[int],
) -> None:
    """Append the nested reduction body for one axis-reduction program.

    Runs inside the outer ``loop $body`` (which owns ``$i`` over N).  Row and
    scalar feeds are read once per ``$i``; each reduction then runs an inner
    ``$k`` loop over K that reads kaxis feeds/constants and folds the reduced
    axis.  Post-reduction elementwise steps close out the iteration.
    """

    axis_k = plan.axis_extent

    def index_lines(klass: str) -> list[str]:
        if klass == "grid":
            return [
                "local.get $i", f"i32.const {axis_k}", "i32.mul",
                "local.get $k", "i32.add",
            ]
        if klass == "kaxis":
            return ["local.get $k"]
        if klass == "row":
            return ["local.get $i"]
        return ["i32.const 0"]

    def emit_load(pointer_expr: str, klass: str, destination: str) -> None:
        body.append(f"      {pointer_expr}")
        for line in index_lines(klass):
            body.append(f"      {line}")
        body.append(f"      i32.const {element_bytes}")
        body.append("      i32.mul")
        body.append("      i32.add")
        body.append(f"      {load}")
        body.append(f"      local.set {destination}")

    def emit_feed(feed_id: int) -> None:
        emit_load(
            f"local.get {feed_label[feed_id]}",
            plan.feed_class[feed_id],
            names[feed_id],
        )

    def emit_step(step: OpStep) -> None:
        constant = _constant_scalar(step)
        if step.op_name == "tensor_from_list" and constant is None:
            entry = static_data["constants"].get(step.result_id)
            if entry is None:
                shortfalls.append(WasmShortfall(
                    step.step_id, step.op_name, "array constant was not baked",
                ))
                return
            emit_load(
                f"i32.const {entry['base']}",
                plan.value_class.get(step.result_id, "row"),
                names[step.result_id],
            )
            return
        instructions = _step_instructions(
            step, names, value_type, element_bytes,
            static_data["constants"], shortfalls,
        )
        if instructions is None:
            return
        body.extend(instructions)
        body.append(f"      local.set {names[step.result_id]}")

    # Row/scalar feeds: read once for this $i, stable across every inner loop.
    for feed_id in feed_ids:
        if plan.feed_class[feed_id] in ("row", "scalar"):
            emit_feed(feed_id)
    for step in plan.pre_steps:
        emit_step(step)

    grid_slots = [
        (output_id, slot, plan.value_class.get(output_id))
        for slot, output_id in enumerate(output_ids)
        if plan.value_class.get(output_id) in ("grid", "kaxis")
    ]
    for region_index, reduction in enumerate(plan.reductions):
        op = plan.reduce_op[reduction.result_id]
        fold, identity = _REDUCE_FOLD[op]
        accumulator = names[reduction.result_id]
        body.append(f"      {_typed_constant(value_type, identity)}")
        body.append(f"      local.set {accumulator}")
        body.append("      i32.const 0")
        body.append("      local.set $k")
        body.append(f"      (block $rdone_{region_index}")
        body.append(f"        (loop $rbody_{region_index}")
        body.append("          local.get $k")
        body.append(f"          i32.const {axis_k}")
        body.append("          i32.ge_s")
        body.append(f"          br_if $rdone_{region_index}")
        for feed_id in feed_ids:
            if plan.feed_class[feed_id] == "kaxis":
                emit_feed(feed_id)
        for step in plan.inner_dependencies[reduction.result_id]:
            emit_step(step)
        body.append(f"          local.get {accumulator}")
        body.append(f"          local.get {names[reduction.input_ids[0]]}")
        if _is_integer_type(value_type) and fold in {"min", "max"}:
            # No integer min/max instruction; fold by compare-and-select
            # over the same two operands (see _INTEGER_COMPOSED_BINARY).
            body.append(f"          local.get {accumulator}")
            body.append(f"          local.get {names[reduction.input_ids[0]]}")
            body.append(
                f"          {value_type}.{'lt_s' if fold == 'min' else 'gt_s'}"
            )
            body.append("          select")
        else:
            body.append(f"          {value_type}.{fold}")
        body.append(f"          local.set {accumulator}")
        body.append("          local.get $k")
        body.append("          i32.const 1")
        body.append("          i32.add")
        body.append("          local.set $k")
        body.append(f"          br $rbody_{region_index}")
        body.append("        )")
        body.append("      )")
        if op == "mean":
            body.append(f"      local.get {accumulator}")
            body.append(f"      {_typed_constant(value_type, axis_k)}")
            body.append(
                f"      {value_type}."
                f"{'div_s' if _is_integer_type(value_type) else 'div'}"
            )
            body.append(f"      local.set {accumulator}")

    for step in plan.post_steps:
        emit_step(step)

    # Grid/kaxis outputs get their own K-walk: they are full N*K tensors, so
    # each is written at every (i, k) cell.  Independent of the folds above --
    # an output that feeds no reduction is still materialised here.
    if grid_slots:
        body.append("      i32.const 0")
        body.append("      local.set $k")
        body.append("      (block $gdone")
        body.append("        (loop $gbody")
        body.append("          local.get $k")
        body.append(f"          i32.const {axis_k}")
        body.append("          i32.ge_s")
        body.append("          br_if $gdone")
        for feed_id in feed_ids:
            if plan.feed_class[feed_id] == "kaxis":
                emit_feed(feed_id)
        for step in plan.grid_output_steps:
            emit_step(step)
        for output_id, slot, klass in grid_slots:
            body.append(f"          local.get $out{slot}")
            for line in index_lines(klass):
                body.append(f"          {line}")
            body.append(f"          i32.const {element_bytes}")
            body.append("          i32.mul")
            body.append("          i32.add")
            body.append(f"          local.get {names[output_id]}")
            body.append(f"          {store}")
        body.append("          local.get $k")
        body.append("          i32.const 1")
        body.append("          i32.add")
        body.append("          local.set $k")
        body.append("          br $gbody")
        body.append("        )")
        body.append("      )")


def emit_wasm_module(
    program: FusedProgram,
    *,
    name: str = "fused_program",
    function_name: str = "run",
    dtype: str | None = None,
    imports: Sequence[object] = (),
    static_data_offset: int = 0,
) -> WasmModule:
    """Lower one internal elementwise ``FusedProgram`` to a WAT module.

    This is a backend emission primitive, not a source compiler.  A direct
    call is appropriate for backend tests and for compiler stages that already
    own a ``FusedProgram``.  It is not an application workflow and says
    nothing about whether the Python frontend can represent classes, control
    flow, state machines, or other source constructs; those were handled
    before this function is reached.

    The emitted function is ``(count, feed0, feed1, ..., out0, ...)`` where
    every argument after ``count`` is a byte offset into the exported memory.

    ``imports`` (``wasm_binary.WasmImport`` entries) wires this module to
    another module's exported function/memory -- see
    ``wasm_class_modules.py``. It defaults to empty, reproducing the exact
    single-module output this function has always produced; every existing
    caller (``build_homepage.py`` included) is unaffected.
    """

    # A trailing-axis reduction over a genuine feed buffer (not one
    # composed in-program from lower-rank operands) has no memory contract
    # under the flat run(count, ...) ABI otherwise; unroll it into an
    # elementwise fold over K strided views of that same buffer first, so
    # the rest of this emitter never has to special-case it.
    program = unroll_feed_axis_reductions(program)
    value_type, element_bytes, load, store = _value_type(program, dtype)
    shortfalls: list[WasmShortfall] = []
    live = required_steps(program)
    static_data = plan_static_data(
        live, value_type, data_offset=static_data_offset,
    )
    shortfalls.extend(static_data["shortfalls"])

    feed_ids = program_feed_order(program)
    output_ids = list(program.outputs.values())
    names: dict[int, str] = {}
    # A view feed (Meta.source_id set) reads another feed's buffer under its
    # own offset/stride and must not get a second byte-offset parameter for
    # the same memory; parameters are allocated per unique buffer owner.
    parameter_feed_ids: list[int] = []
    seen_sources: set[int] = set()
    for feed_id in feed_ids:
        source_id = resolve_view_source(program.meta, feed_id)
        if source_id not in seen_sources:
            seen_sources.add(source_id)
            parameter_feed_ids.append(source_id)
    labels = feed_names(program, parameter_feed_ids)
    parameters: list[str] = ["$count"]
    for index, feed_id in enumerate(parameter_feed_ids):
        parameters.append("$" + labels[index])
    for index, _ in enumerate(output_ids):
        parameters.append(f"$out{index}")

    # A value whose real dtype isn't the module's f32/f64 working type is
    # still loaded/stored at its own byte width and WASM instruction; only
    # the arithmetic in between stays the working type (converted at the
    # boundary), not the parameter's memory representation.
    program_meta = program.meta or {}

    def _memory_ops(value_id: int) -> tuple[int, str, str, str | None]:
        meta = program_meta.get(int(value_id))
        entry = _MEMORY_DTYPE_OPS.get(str(meta.dtype)) if meta and meta.dtype else None
        if entry is None:
            return element_bytes, load, store, None
        width, typed_load, typed_store, wasm_kind = entry
        return width, typed_load, typed_store, wasm_kind

    body: list[str] = []
    locals_declared: list[str] = ["(local $i i32)", "(local $addr i32)"]

    def direct_element_address(pointer: str, byte_width: int = element_bytes) -> list[str]:
        # addr = pointer + i * byte_width
        return [
            f"      local.get {pointer}",
            "      local.get $i",
            f"      i32.const {byte_width}",
            "      i32.mul",
            "      i32.add",
        ]

    def element_address(feed_id: int) -> list[str]:
        # addr = source_pointer + offset*bytes + i * stride*bytes
        # The offset/stride descriptor is the IR's memory manager
        # (fused_ir.Meta); default (0, 1) reproduces a plain contiguous read.
        source_id = resolve_view_source(program.meta, feed_id)
        offset, stride = view_offset_stride(program.meta, feed_id)
        byte_width, *_ = _memory_ops(source_id)
        instructions = [f"      local.get {feed_label[source_id]}"]
        byte_offset = offset * byte_width
        if byte_offset:
            instructions.extend([
                f"      i32.const {byte_offset}",
                "      i32.add",
            ])
        instructions.extend([
            "      local.get $i",
            f"      i32.const {stride * byte_width}",
            "      i32.mul",
            "      i32.add",
        ])
        return instructions

    def gather_address(table_id: int, index_local: str) -> list[str] | None:
        """``table[index]`` -- the table's base plus a *computed* offset.

        ``element_address`` walks with the loop cursor ``$i``; this walks
        with a value the program computed, which is the whole difference
        between an elementwise read and a table lookup.
        """

        source_id = resolve_view_source(program.meta, int(table_id))
        label = feed_label.get(source_id)
        if label is None:
            return None
        byte_width, typed_load, _store, wasm_kind = _memory_ops(source_id)
        offset, _stride = view_offset_stride(program.meta, int(table_id))
        # Address arithmetic is i32, so a wider index is narrowed for the
        # address only -- the index value itself keeps its own type.
        if value_type == "i64":
            narrow = ["      i32.wrap_i64"]
        elif value_type == "i32":
            narrow = []
        else:
            narrow = [f"      {_FROM_WORKING_TYPE[(value_type, 'i32')]}"]
        instructions = [f"      local.get {label}"]
        if offset:
            instructions.extend([
                f"      i32.const {offset * byte_width}",
                "      i32.add",
            ])
        instructions.extend([
            f"      local.get {index_local}",
            *narrow,
            f"      i32.const {byte_width}",
            "      i32.mul",
            "      i32.add",
            f"      {typed_load}",
        ])
        if wasm_kind is not None:
            conversion = _TO_WORKING_TYPE.get((wasm_kind, value_type))
            if conversion is not None:
                instructions.append(f"      {conversion}")
        return instructions

    # Allocate stable locals before emitting either reduction or output passes.
    for index, feed_id in enumerate(feed_ids):
        local = f"$v{len(names)}"
        names[feed_id] = local
        locals_declared.append(f"(local {local} {value_type})")
    for step in live:
        local = f"$v{len(names)}"
        names[step.result_id] = local
        locals_declared.append(f"(local {local} {value_type})")

    sums = tuple(
        step for step in _flat_sum_steps(live) if step.attrs.get("axis") is None
    )
    plan = _plan_axis_reductions(program, live, feed_ids, shortfalls)
    if plan is not None:
        locals_declared.append("(local $k i32)")

    feed_label = {
        feed_id: "$" + labels[index] for index, feed_id in enumerate(parameter_feed_ids)
    }

    def load_feeds(target: list[str]) -> None:
        # Feeds are read once per iteration into locals, so a value used by
        # more than one step is loaded once rather than re-read from memory.
        for feed_id in feed_ids:
            _, feed_load, _, feed_kind = _memory_ops(feed_id)
            target.extend(element_address(feed_id))
            target.append(f"      {feed_load}")
            if feed_kind is not None:
                conversion = _TO_WORKING_TYPE.get((feed_kind, value_type))
                if conversion is not None:
                    target.append(f"      {conversion}")
            target.append(f"      local.set {names[feed_id]}")

    def evaluate_steps(target: list[str], steps: Sequence[OpStep]) -> None:
        for candidate in steps:
            if candidate.op_name == "sum":
                continue
            instructions = _step_instructions(
                candidate,
                names,
                value_type,
                element_bytes,
                static_data["constants"],
                shortfalls,
                gather_address,
            )
            if instructions is None:
                continue
            target.extend(instructions)
            target.append(f"      local.set {names[candidate.result_id]}")

    reduction_passes: list[str] = []
    if plan is not None and plan.ok:
        # Nested walk: outer loop iterates N (added by the assembly below),
        # each reduction runs an inner counted loop over K.
        _emit_reduction_body_wat(
            body, plan, names, feed_label, feed_ids, value_type,
            load, store, element_bytes, static_data, shortfalls, output_ids,
        )
    elif plan is not None:
        # Unsupported axis reduction: the plan already recorded a shortfall, so
        # no binary is built. The illustrative text body is left empty.
        pass
    else:
        for reduction_index, step in enumerate(sums):
            dependencies = _sum_dependencies(step, live)
            source = names.get(step.input_ids[0]) if step.input_ids else None
            if source is None:
                shortfalls.append(WasmShortfall(
                    step.step_id, step.op_name, "reduction operand was never produced",
                ))
                continue
            reduction_passes.extend((
                "    i32.const 0",
                "    local.set $i",
                f"    (block $sum_done_{reduction_index}",
                f"      (loop $sum_body_{reduction_index}",
                "        local.get $i",
                "        local.get $count",
                "        i32.ge_s",
                f"        br_if $sum_done_{reduction_index}",
            ))
            load_feeds(reduction_passes)
            evaluate_steps(reduction_passes, dependencies)
            reduction_passes.extend((
                f"      local.get {names[step.result_id]}",
                f"      local.get {source}",
                f"      {value_type}.add",
                f"      local.set {names[step.result_id]}",
                "        local.get $i",
                "        i32.const 1",
                "        i32.add",
                "        local.set $i",
                f"        br $sum_body_{reduction_index}",
                "      )",
                "    )",
            ))

        load_feeds(body)
        evaluate_steps(body, live)

    if plan is None or plan.ok:
        for index, output_id in enumerate(output_ids):
            if plan is not None and plan.value_class.get(output_id) in ("grid", "kaxis"):
                continue  # written by the grid K-walk in the reduction body
            target = names.get(output_id)
            if target is None:
                shortfalls.append(
                    WasmShortfall(-1, "output", f"value {output_id} is never produced")
                )
                continue
            out_width, _, out_store, out_kind = _memory_ops(output_id)
            body.extend(direct_element_address(f"$out{index}", out_width))
            body.append(f"      local.get {target}")
            if out_kind is not None:
                conversion = _FROM_WORKING_TYPE.get((value_type, out_kind))
                if conversion is not None:
                    body.append(f"      {conversion}")
            body.append(f"      {out_store}")

    parameter_text = " ".join(f"(param {p} i32)" for p in parameters)
    memory_import = next(
        (entry for entry in imports if getattr(entry, "kind", None) == "memory"),
        None,
    )
    memory_declaration = (
        f'  (import "{memory_import.module}" "{memory_import.field}" (memory '
        f'{memory_import.memory_pages}))'
        if memory_import is not None
        else '  (memory (export "memory") 1)'
    )
    lines = [
        f"(module ;; {name}",
        "  ;; The coordinator owns memory and passes byte offsets. A fused",
        "  ;; elementwise program keeps no private tensor state.",
        memory_declaration,
        f"  (func (export \"{function_name}\") {parameter_text}",
        *(f"    {declaration}" for declaration in locals_declared),
        *reduction_passes,
        "    i32.const 0",
        "    local.set $i",
        "    (block $done",
        "      (loop $body",
        "        ;; while i < count",
        "        local.get $i",
        "        local.get $count",
        "        i32.ge_s",
        "        br_if $done",
        *body,
        "        local.get $i",
        "        i32.const 1",
        "        i32.add",
        "        local.set $i",
        "        br $body",
        "      )",
        "    )",
        "  )",
        ")",
        "",
    ]
    source = "\n".join(lines)

    reserved = static_data["reserved_bytes"]
    parameter_dtypes = {
        value_id: meta.dtype
        for value_id in (*parameter_feed_ids, *output_ids)
        if (meta := program_meta.get(value_id)) is not None and meta.dtype
    }
    api = _describe(name, function_name, parameter_feed_ids, output_ids, value_type,
                    element_bytes, reserved,
                    static_data_offset=static_data_offset,
                    shared_memory_import=(
                        {"module": memory_import.module, "field": memory_import.field}
                        if memory_import is not None else None
                    ),
                    input_names=labels,
                    output_names=list(program.outputs.keys()),
                    parameter_dtypes=parameter_dtypes)
    binary = None
    if not shortfalls:
        binary = _assemble(
            program, feed_ids, output_ids, value_type, element_bytes,
            function_name, static_data=static_data, imports=imports,
        )
    return WasmModule(
        name=name,
        source=source,
        shortfalls=tuple(shortfalls),
        parameters=tuple(parameters),
        value_type=value_type,
        api=api,
        binary=binary,
    )


def _step_instructions(
    step: OpStep,
    names: Mapping[int, str],
    value_type: str,
    element_bytes: int,
    constant_entries: Mapping[int, Mapping[str, Any]],
    shortfalls: list[WasmShortfall],
    gather_address: Any = None,
) -> list[str] | None:
    op = step.op_name
    constant = _constant_scalar(step)
    if constant is not None:
        return [f"      {_typed_constant(value_type, constant)}"]
    if op == "tensor_from_list":
        entry = constant_entries.get(step.result_id)
        if entry is None:
            return None
        return [
            f"      i32.const {entry['base']}",
            "      local.get $i",
            f"      i32.const {element_bytes}",
            "      i32.mul",
            "      i32.add",
            f"      {value_type}.load",
        ]
    integral = _is_integer_type(value_type)
    if op in _NO_WASM_INSTRUCTION and not (
        integral
        and (
            op in _INTEGER_BINARY_INSTRUCTION
            or op in _INTEGER_LOGICAL_BINARY
            or op in _INTEGER_LOGICAL_UNARY
        )
    ):
        # mod/floordiv and the boolean connectives are only unsupported for
        # a *float* working type; an integral program reaches them through
        # rem_s/div_s and eqz below.
        shortfalls.append(
            WasmShortfall(
                step.step_id,
                op,
                "WebAssembly has no instruction for this; it would need a "
                "hand-written polynomial approximation, which is not a "
                "translation",
            )
        )
        return None

    left = names.get(step.input_ids[0]) if step.input_ids else None
    if left is None:
        shortfalls.append(
            WasmShortfall(step.step_id, op, "operand was never produced")
        )
        return None

    if op == "sign":
        # sign(x) = (x>0) - (x<0): exact, using the comparison opcodes WASM has.
        if integral:
            # An integer comparison already yields i32 0/1. For an i64
            # working type that still has to widen before the subtraction;
            # for i32 the comparison result is already the working type.
            widen = ["      i64.extend_i32_u"] if value_type == "i64" else []
            return [
                f"      local.get {left}",
                f"      {_typed_constant(value_type, 0)}",
                f"      {value_type}.gt_s",
                *widen,
                f"      local.get {left}",
                f"      {_typed_constant(value_type, 0)}",
                f"      {value_type}.lt_s",
                *widen,
                f"      {value_type}.sub",
            ]
        return [
            f"      local.get {left}",
            f"      {value_type}.const 0.0",
            f"      {value_type}.gt",
            f"      {value_type}.convert_i32_u",
            f"      local.get {left}",
            f"      {value_type}.const 0.0",
            f"      {value_type}.lt",
            f"      {value_type}.convert_i32_u",
            f"      {value_type}.sub",
        ]

    if op == "pow":
        if step.attrs.get("reverse", False):
            shortfalls.append(WasmShortfall(
                step.step_id, "pow",
                "reverse pow (base on the right) is not lowered yet"))
            return None
        # pow(x, y) = exp(y*log(x)); the binary composes the baked exp/log
        # tables. Text names the step rather than inlining it, like any LUT op.
        return [f"      ;; pow via exp(y*log(x)) baked tables (see the .wasm)",
                f"      local.get {left}"]

    if op in _LUT_OPS:
        # The binary carries the table and the interpolation; the text form
        # names the step instead of inlining it, so the two are not pretending
        # to be line-for-line the same. The binary is the artifact that runs.
        return [f"      ;; {op} via baked lookup table (see the .wasm)",
                f"      local.get {left}"]

    if op in ELEMENTWISE_UNARY:
        if integral:
            if op in _INTEGER_IDENTITY_UNARY:
                # Rounding an integer is the integer.
                return [f"      local.get {left}"]
            if op in _INTEGER_LOGICAL_UNARY:
                # not(x) is exactly "x is zero", which eqz answers directly.
                widen = (
                    ["      i64.extend_i32_u"] if value_type == "i64" else []
                )
                return [
                    f"      local.get {left}",
                    f"      {value_type}.eqz",
                    *widen,
                ]
            if op == "neg":
                # No i32.neg/i64.neg exists; negation is 0 - x.
                return [
                    f"      {_typed_constant(value_type, 0)}",
                    f"      local.get {left}",
                    f"      {value_type}.sub",
                ]
            if op == "abs":
                # No integer abs either: select between x and -x on x < 0.
                return [
                    f"      local.get {left}",
                    f"      {_typed_constant(value_type, 0)}",
                    f"      local.get {left}",
                    f"      {value_type}.sub",
                    f"      local.get {left}",
                    f"      {_typed_constant(value_type, 0)}",
                    f"      {value_type}.lt_s",
                    "      select",
                ]
            shortfalls.append(WasmShortfall(
                step.step_id, op,
                f"no integer instruction for this unary operation on "
                f"{value_type}; it is defined only for floating point",
            ))
            return None
        instruction = _UNARY_INSTRUCTION.get(op)
        if instruction is None:
            shortfalls.append(
                WasmShortfall(step.step_id, op, "no unary instruction registered")
            )
            return None
        return [f"      local.get {left}", f"      {value_type}.{instruction}"]

    if op == "gather" and len(step.input_ids) == 2:
        # table[index]: a read at a *computed* offset rather than at the
        # elementwise walk's own cursor. This is what a table-driven decoder
        # is made of -- an opcode byte selecting a row of an encoding table --
        # so without it a state machine's every table lookup is a shortfall.
        index_local = names.get(step.input_ids[1])
        if index_local is None:
            shortfalls.append(WasmShortfall(
                step.step_id, op, "gather index was never produced",
            ))
            return None
        if int(step.attrs.get("dim", 0)) != 0:
            shortfalls.append(WasmShortfall(
                step.step_id, op,
                f"only dim=0 gather is lowered; got dim="
                f"{step.attrs.get('dim')}",
            ))
            return None
        instructions = (
            None if gather_address is None
            else gather_address(step.input_ids[0], index_local)
        )
        if instructions is None:
            shortfalls.append(WasmShortfall(
                step.step_id, op,
                "gather source is not an addressable feed buffer",
            ))
            return None
        return instructions

    if op == "where" and len(step.input_ids) == 3:
        # Ternary select: (condition, if_true, if_false). WebAssembly's own
        # `select` is exactly this, so a predicated program -- which is what
        # a branchless state machine compiles to, every update guarded by a
        # mask rather than a jump -- needs no control flow to express.
        #
        # `select` pops (val1, val2, condition) and keeps val1 when the
        # condition is a non-zero *i32*, so a wider or floating working type
        # is first reduced to that. i64 goes through eqz twice rather than
        # i32.wrap_i64: wrapping a value whose low 32 bits happen to be zero
        # would silently invert the test, and nothing here guarantees a mask
        # is only ever 0/1.
        condition, if_true, if_false = (
            names.get(value_id) for value_id in step.input_ids
        )
        if None in (condition, if_true, if_false):
            shortfalls.append(WasmShortfall(
                step.step_id, op, "an operand was never produced",
            ))
            return None
        if value_type == "i32":
            narrow: list[str] = []          # already an i32 truth value
        elif value_type == "i64":
            narrow = ["      i64.eqz", "      i32.eqz"]
        else:
            narrow = [
                f"      {_typed_constant(value_type, 0)}",
                f"      {value_type}.ne",
            ]
        return [
            f"      local.get {if_true}",
            f"      local.get {if_false}",
            f"      local.get {condition}",
            *narrow,
            "      select",
        ]

    if op not in ELEMENTWISE_BINARY:
        shortfalls.append(
            WasmShortfall(step.step_id, op, "not an elementwise operation")
        )
        return None

    if len(step.input_ids) == 2:
        right_source = [f"      local.get {names[step.input_ids[1]]}"]
    elif "right_scalar" in step.attrs:
        right_source = [
            f"      {_typed_constant(value_type, step.attrs['right_scalar'])}"
        ]
    else:
        shortfalls.append(
            WasmShortfall(step.step_id, op, "binary step has no right operand")
        )
        return None

    operands = [f"      local.get {left}", *right_source]
    if step.attrs.get("reverse", False):
        operands = [*right_source, f"      local.get {left}"]

    if integral:
        instruction = _INTEGER_BINARY_INSTRUCTION.get(op)
        if instruction is not None:
            return [*operands, f"      {value_type}.{instruction}"]
        if op in _INTEGER_LOGICAL_BINARY:
            # Truth-value connectives, not bitwise ones. eqz gives "is zero",
            # so De Morgan applies: a&&b == !(!a || !b), a||b == !(!a && !b).
            combine = "or" if op == "logical_and" else "and"
            widen = ["      i64.extend_i32_u"] if value_type == "i64" else []
            return [
                operands[0],
                f"      {value_type}.eqz",
                operands[1],
                f"      {value_type}.eqz",
                f"      i32.{combine}",
                "      i32.eqz",
                *widen,
            ]
        if op in _INTEGER_COMPOSED_BINARY:
            # WebAssembly has no integer min/max, so compare and select.
            # ``select`` pops (val1, val2, condition) and yields val1 when
            # the condition is non-zero, so the operands are pushed twice:
            # once as the candidate pair, once to compute the predicate.
            keep_left = "lt_s" if op == "minimum" else "gt_s"
            return [
                *operands,
                *operands,
                f"      {value_type}.{keep_left}",
                "      select",
            ]
        comparison = _INTEGER_COMPARISON_INSTRUCTION.get(op)
        if comparison is not None:
            # An integer comparison already produces i32 0/1. Only an i64
            # working type needs a widening; i32 is already the value type.
            widen = ["      i64.extend_i32_u"] if value_type == "i64" else []
            return [*operands, f"      {value_type}.{comparison}", *widen]
        shortfalls.append(WasmShortfall(
            step.step_id, op,
            f"no integer instruction registered for this operation on "
            f"{value_type}",
        ))
        return None

    instruction = _BINARY_INSTRUCTION.get(op)
    if instruction is not None:
        return [*operands, f"      {value_type}.{instruction}"]

    comparison = _COMPARISON_INSTRUCTION.get(op)
    if comparison is not None:
        # i32 0/1 back to the value type, so the result matches what every
        # other backend reports for a comparison.
        return [
            *operands,
            f"      {value_type}.{comparison}",
            f"      {value_type}.convert_i32_u",
        ]

    shortfalls.append(
        WasmShortfall(step.step_id, op, "no binary instruction registered")
    )
    return None



def _assemble(
    program: FusedProgram,
    feed_ids: Sequence[int],
    output_ids: Sequence[int],
    value_type: str,
    element_bytes: int,
    function_name: str,
    *,
    static_data: Mapping[str, Any],
    imports: Sequence[object] = (),
) -> bytes:
    """Assemble the same program as a binary module.

    Mirrors the WAT lowering above step for step -- same order, same
    operands, same instructions -- because the two forms describing different
    programs would be worse than having only one.
    """

    from .wasm_binary import CodeBuilder, build_module

    live = required_steps(program)
    lut_ops = {step.op_name for step in live} & _LUT_OPS
    if any(step.op_name == "pow" for step in live):
        lut_ops |= {"exp", "log"}
    lut_ops = sorted(lut_ops)
    if lut_ops and value_type != "f64":
        raise WasmEmissionError(
            f"{lut_ops} is only baked for f64; an f32 table would need its "
            "own sampling and has no caller yet"
        )
    tables = static_data
    reserved_bytes = tables["reserved_bytes"]

    parameter_count = 1 + len(feed_ids) + len(output_ids)
    builder = CodeBuilder(value_type=value_type, parameter_count=parameter_count)
    count_param = 0
    # A view feed (Meta.source_id set) reads another feed's buffer under its
    # own offset/stride and must not get a second byte-offset parameter for
    # the same memory; parameters are allocated per unique buffer owner, same
    # as the WAT lowering above.
    parameter_feed_ids: list[int] = []
    seen_sources: set[int] = set()
    for feed_id in feed_ids:
        source_id = resolve_view_source(program.meta, feed_id)
        if source_id not in seen_sources:
            seen_sources.add(source_id)
            parameter_feed_ids.append(source_id)
    feed_params = {feed_id: 1 + i for i, feed_id in enumerate(parameter_feed_ids)}
    output_params = [1 + len(parameter_feed_ids) + i for i in range(len(output_ids))]

    index_local = builder.declare_local("i32")
    locals_for: dict[int, int] = {}
    for feed_id in feed_ids:
        locals_for[feed_id] = builder.declare_local(value_type)
    for step in live:
        locals_for[step.result_id] = builder.declare_local(value_type)

    # Same boundary-typing rule as the WAT lowering above: a value whose
    # real dtype isn't this module's f32/f64 working type is still
    # loaded/stored at its own byte width and WASM instruction, converted
    # to/from the working type only for the arithmetic in between.
    program_meta = program.meta or {}

    def _memory_ops(value_id: int) -> tuple[int, str | None]:
        meta = program_meta.get(int(value_id))
        dtype = str(meta.dtype) if meta and meta.dtype else None
        entry = _MEMORY_DTYPE_OPS.get(dtype) if dtype else None
        if entry is None:
            return element_bytes, None
        width, _load_text, _store_text, wasm_kind = entry
        return width, wasm_kind

    def _typed_load(width: int, wasm_kind: str | None) -> None:
        if wasm_kind == "i32":
            builder.i32_load_width(8 if width == 1 else 32)
        elif wasm_kind == "i64":
            builder.i64_load_width(64)
        else:
            builder.load()

    def _typed_store(width: int, wasm_kind: str | None) -> None:
        if wasm_kind == "i32":
            builder.i32_store_width(8 if width == 1 else 32)
        elif wasm_kind == "i64":
            builder.i64_store_width(64)
        else:
            builder.store()

    def _convert_to_working(wasm_kind: str | None) -> None:
        if wasm_kind is None:
            return
        opcode = _TO_WORKING_OPCODE.get((wasm_kind, value_type))
        if opcode is not None:
            builder.raw(opcode)

    def _convert_from_working(wasm_kind: str | None) -> None:
        if wasm_kind is None:
            return
        opcode = _FROM_WORKING_OPCODE.get((value_type, wasm_kind))
        if opcode is not None:
            builder.raw(opcode)

    def direct_element_address(pointer_param: int, byte_width: int = element_bytes) -> None:
        builder.local_get(pointer_param)
        builder.local_get(index_local)
        builder.i32_const(byte_width)
        builder.raw(0x6C)  # i32.mul
        builder.raw(0x6A)  # i32.add

    def element_address(feed_id: int) -> None:
        source_id = resolve_view_source(program.meta, feed_id)
        offset, stride = view_offset_stride(program.meta, feed_id)
        byte_width, _ = _memory_ops(source_id)
        builder.local_get(feed_params[source_id])
        byte_offset = offset * byte_width
        if byte_offset:
            builder.i32_const(byte_offset)
            builder.raw(0x6A)  # i32.add
        builder.local_get(index_local)
        builder.i32_const(stride * byte_width)
        builder.raw(0x6C)  # i32.mul
        builder.raw(0x6A)  # i32.add

    def load_feeds() -> None:
        for feed_id in feed_ids:
            width, kind = _memory_ops(feed_id)
            element_address(feed_id)
            _typed_load(width, kind)
            _convert_to_working(kind)
            builder.local_set(locals_for[feed_id])

    def evaluate_steps(steps: Sequence[OpStep]) -> None:
        for step in steps:
            if step.op_name == "sum":
                continue
            emit_step(step, push_index_flat)

    def push_index_flat() -> None:
        builder.local_get(index_local)

    def emit_step(step: OpStep, push_index) -> None:
        local = locals_for[step.result_id]
        constant = _constant_scalar(step)
        if constant is not None:
            builder.value_const(constant)
            builder.local_set(local)
            return
        if step.op_name == "tensor_from_list":
            entry = tables["constants"][step.result_id]
            builder.i32_const(entry["base"])
            push_index()
            builder.i32_const(element_bytes)
            builder.raw(0x6C)  # i32.mul
            builder.raw(0x6A)  # i32.add
            builder.load()
            builder.local_set(local)
            return
        left = locals_for[step.input_ids[0]]

        integral = _is_integer_type(value_type)

        def push_right() -> None:
            if len(step.input_ids) == 2:
                builder.local_get(locals_for[step.input_ids[1]])
            else:
                builder.value_const(float(step.attrs["right_scalar"]))

        if step.op_name == "sign":
            # sign(x) = (x>0) - (x<0): exact, no table.
            if integral:
                for comparison in ("gt_s", "lt_s"):
                    builder.local_get(left)
                    builder.value_const(0.0)
                    builder.op(comparison)
                    if value_type == "i64":
                        builder.op("extend_i32_u")
                builder.op("sub")
                builder.local_set(local)
                return
            builder.local_get(left)
            builder.value_const(0.0)
            builder.op("gt")
            builder.op("convert_i32_u")
            builder.local_get(left)
            builder.value_const(0.0)
            builder.op("lt")
            builder.op("convert_i32_u")
            builder.op("sub")
            builder.local_set(local)
            return

        if step.op_name == "pow":
            # pow(x, y) = exp(y * log(x)), composing the baked tables.
            log_entry = tables["entries"]["log"]
            exp_entry = tables["entries"]["exp"]
            product = builder.declare_local(value_type)
            _emit_lut(
                builder, left, "log", log_entry["base"],
                log_entry["intervals"], log_entry["lower"],
                log_entry["upper"], log_entry["periodic"],
            )
            push_right()
            builder.op("mul")
            builder.local_set(product)
            _emit_lut(
                builder, product, "exp", exp_entry["base"],
                exp_entry["intervals"], exp_entry["lower"],
                exp_entry["upper"], exp_entry["periodic"],
            )
            builder.local_set(local)
            return

        if step.op_name == "where" and len(step.input_ids) == 3:
            # See the WAT emitter for why the condition is narrowed this way.
            condition, if_true, if_false = (
                locals_for[value_id] for value_id in step.input_ids
            )
            builder.local_get(if_true)
            builder.local_get(if_false)
            builder.local_get(condition)
            if value_type == "i64":
                builder.op("eqz")
                builder.raw(0x45)  # i32.eqz
            elif value_type != "i32":
                builder.value_const(0.0)
                builder.op("ne")
            builder.select()
            builder.local_set(local)
            return

        if step.op_name in _LUT_OPS:
            entry = tables["entries"][step.op_name]
            _emit_lut(
                builder, left, step.op_name, entry["base"], entry["intervals"],
                entry["lower"], entry["upper"], entry["periodic"],
            )
        elif step.op_name in ELEMENTWISE_UNARY:
            if integral and step.op_name in _INTEGER_IDENTITY_UNARY:
                builder.local_get(left)  # rounding an integer is the integer
            elif integral and step.op_name in _INTEGER_LOGICAL_UNARY:
                builder.local_get(left)
                builder.op("eqz")
                if value_type == "i64":
                    builder.op("extend_i32_u")
            elif integral and step.op_name == "neg":
                builder.value_const(0.0)  # no integer neg: 0 - x
                builder.local_get(left)
                builder.op("sub")
            elif integral and step.op_name == "abs":
                builder.local_get(left)
                builder.value_const(0.0)
                builder.local_get(left)
                builder.op("sub")
                builder.local_get(left)
                builder.value_const(0.0)
                builder.op("lt_s")
                builder.select()
            elif integral:
                raise WasmEmissionError(
                    f"no integer instruction for unary {step.op_name!r} on "
                    f"{value_type}; it is defined only for floating point"
                )
            else:
                builder.local_get(left)
                builder.op(_UNARY_INSTRUCTION[step.op_name])
        elif step.attrs.get("reverse", False):
            push_right()
            builder.local_get(left)
        else:
            builder.local_get(left)
            push_right()

        if step.op_name in ELEMENTWISE_BINARY:
            if integral:
                instruction = _INTEGER_BINARY_INSTRUCTION.get(step.op_name)
                if instruction is not None:
                    builder.op(instruction)
                elif step.op_name in _INTEGER_LOGICAL_BINARY:
                    # Operands are already on the stack in order; reduce each
                    # to a truth value and combine by De Morgan (see the WAT
                    # emitter for the full explanation).
                    combine_and = step.op_name == "logical_or"
                    builder.op("eqz")          # top operand -> "is zero"
                    swap = builder.declare_local("i32")
                    builder.local_set(swap)
                    builder.op("eqz")          # lower operand -> "is zero"
                    builder.local_get(swap)
                    builder.raw(0x71 if combine_and else 0x72)  # i32.and/or
                    builder.raw(0x45)  # i32.eqz
                    if value_type == "i64":
                        builder.op("extend_i32_u")
                elif step.op_name in _INTEGER_COMPOSED_BINARY:
                    # No integer min/max: re-push the pair and select.
                    if step.attrs.get("reverse", False):
                        push_right()
                        builder.local_get(left)
                    else:
                        builder.local_get(left)
                        push_right()
                    builder.op(
                        "lt_s" if step.op_name == "minimum" else "gt_s"
                    )
                    builder.select()
                else:
                    builder.op(
                        _INTEGER_COMPARISON_INSTRUCTION[step.op_name]
                    )
                    if value_type == "i64":
                        builder.op("extend_i32_u")
            else:
                instruction = _BINARY_INSTRUCTION.get(step.op_name)
                if instruction is not None:
                    builder.op(instruction)
                else:
                    builder.op(_COMPARISON_INSTRUCTION[step.op_name])
                    builder.op("convert_i32_u")
        builder.local_set(local)

    plan = _plan_axis_reductions(program, live, feed_ids, [])
    if plan is not None and plan.ok:
        k_local = builder.declare_local("i32")
        axis_k = plan.axis_extent

        def push_class_index(klass: str) -> None:
            if klass == "grid":
                builder.local_get(index_local)
                builder.i32_const(axis_k)
                builder.raw(0x6C)  # i32.mul
                builder.local_get(k_local)
                builder.raw(0x6A)  # i32.add
            elif klass == "kaxis":
                builder.local_get(k_local)
            elif klass == "row":
                builder.local_get(index_local)
            else:
                builder.i32_const(0)

        def load_feed_class(feed_id: int) -> None:
            builder.local_get(feed_params[feed_id])
            push_class_index(plan.feed_class[feed_id])
            builder.i32_const(element_bytes)
            builder.raw(0x6C)  # i32.mul
            builder.raw(0x6A)  # i32.add
            builder.load()
            builder.local_set(locals_for[feed_id])

        def emit_reduction_step(step: OpStep) -> None:
            klass = plan.value_class.get(step.result_id, "row")
            emit_step(step, lambda: push_class_index(klass))

        builder.i32_const(0)
        builder.local_set(index_local)
        builder.block()
        builder.loop()
        builder.local_get(index_local)
        builder.local_get(count_param)
        builder.raw(0x4E)  # i32.ge_s
        builder.br_if(1)
        for feed_id in feed_ids:
            if plan.feed_class[feed_id] in ("row", "scalar"):
                load_feed_class(feed_id)
        for step in plan.pre_steps:
            emit_reduction_step(step)
        grid_slots = [
            (output_id, slot, plan.value_class.get(output_id))
            for slot, output_id in enumerate(output_ids)
            if plan.value_class.get(output_id) in ("grid", "kaxis")
        ]
        for reduction in plan.reductions:
            op = plan.reduce_op[reduction.result_id]
            fold, identity = _REDUCE_FOLD[op]
            accumulator = locals_for[reduction.result_id]
            builder.value_const(identity)
            builder.local_set(accumulator)
            builder.i32_const(0)
            builder.local_set(k_local)
            builder.block()
            builder.loop()
            builder.local_get(k_local)
            builder.i32_const(axis_k)
            builder.raw(0x4E)  # i32.ge_s
            builder.br_if(1)
            for feed_id in feed_ids:
                if plan.feed_class[feed_id] == "kaxis":
                    load_feed_class(feed_id)
            for step in plan.inner_dependencies[reduction.result_id]:
                emit_reduction_step(step)
            builder.local_get(accumulator)
            builder.local_get(locals_for[reduction.input_ids[0]])
            if _is_integer_type(value_type) and fold in {"min", "max"}:
                # No integer min/max instruction; compare and select over the
                # same operand pair (matches the WAT emitter's fold above).
                builder.local_get(accumulator)
                builder.local_get(locals_for[reduction.input_ids[0]])
                builder.op("lt_s" if fold == "min" else "gt_s")
                builder.select()
            else:
                builder.op(fold)
            builder.local_set(accumulator)
            builder.local_get(k_local)
            builder.i32_const(1)
            builder.raw(0x6A)  # i32.add
            builder.local_set(k_local)
            builder.br(0)
            builder.end()  # loop
            builder.end()  # block
            if op == "mean":
                builder.local_get(accumulator)
                builder.value_const(float(axis_k))
                builder.op("div_s" if _is_integer_type(value_type) else "div")
                builder.local_set(accumulator)
        for step in plan.post_steps:
            emit_reduction_step(step)
        if grid_slots:
            builder.i32_const(0)
            builder.local_set(k_local)
            builder.block()
            builder.loop()
            builder.local_get(k_local)
            builder.i32_const(axis_k)
            builder.raw(0x4E)  # i32.ge_s
            builder.br_if(1)
            for feed_id in feed_ids:
                if plan.feed_class[feed_id] == "kaxis":
                    load_feed_class(feed_id)
            for step in plan.grid_output_steps:
                emit_reduction_step(step)
            for output_id, slot, klass in grid_slots:
                grid_width, grid_kind = _memory_ops(output_id)
                builder.local_get(output_params[slot])
                push_class_index(klass)
                builder.i32_const(grid_width)
                builder.raw(0x6C)  # i32.mul
                builder.raw(0x6A)  # i32.add
                builder.local_get(locals_for[output_id])
                _convert_from_working(grid_kind)
                _typed_store(grid_width, grid_kind)
            builder.local_get(k_local)
            builder.i32_const(1)
            builder.raw(0x6A)  # i32.add
            builder.local_set(k_local)
            builder.br(0)
            builder.end()  # loop
            builder.end()  # block
        for slot, output_id in enumerate(output_ids):
            if plan.value_class.get(output_id) in ("grid", "kaxis"):
                continue  # written by the grid K-walk above
            out_width, out_kind = _memory_ops(output_id)
            direct_element_address(output_params[slot], out_width)
            builder.local_get(locals_for[output_id])
            _convert_from_working(out_kind)
            _typed_store(out_width, out_kind)
        builder.local_get(index_local)
        builder.i32_const(1)
        builder.raw(0x6A)  # i32.add
        builder.local_set(index_local)
        builder.br(0)
        builder.end()  # loop
        builder.end()  # block

        data = tables["data"]
        pages = max(1, (reserved_bytes + 65535) // 65536 + 1)
        return build_module(
            function_name=function_name,
            parameter_types=["i32"] * parameter_count,
            body=builder,
            memory_pages=pages,
            data=data,
            data_offset=int(tables.get("data_offset", 0)),
            imports=imports,
        )

    # Each SSA sum gets an explicit whole-tensor pass. Its scalar local is
    # then consumed by later tensor expressions as a broadcast value.
    for reduction in _flat_sum_steps(live):
        builder.i32_const(0)
        builder.local_set(index_local)
        builder.block()
        builder.loop()
        builder.local_get(index_local)
        builder.local_get(count_param)
        builder.raw(0x4E)  # i32.ge_s
        builder.br_if(1)
        load_feeds()
        evaluate_steps(_sum_dependencies(reduction, live))
        builder.local_get(locals_for[reduction.result_id])
        builder.local_get(locals_for[reduction.input_ids[0]])
        builder.op("add")
        builder.local_set(locals_for[reduction.result_id])
        builder.local_get(index_local)
        builder.i32_const(1)
        builder.raw(0x6A)
        builder.local_set(index_local)
        builder.br(0)
        builder.end()
        builder.end()

    builder.i32_const(0)
    builder.local_set(index_local)
    # block { loop { if i >= count break; ...; i += 1; continue } }
    builder.block()
    builder.loop()
    builder.local_get(index_local)
    builder.local_get(count_param)
    builder.raw(0x4E)  # i32.ge_s
    builder.br_if(1)  # out of the enclosing block
    load_feeds()
    evaluate_steps(live)

    for slot, output_id in enumerate(output_ids):
        out_width, out_kind = _memory_ops(output_id)
        direct_element_address(output_params[slot], out_width)
        builder.local_get(locals_for[output_id])
        _convert_from_working(out_kind)
        _typed_store(out_width, out_kind)

    builder.local_get(index_local)
    builder.i32_const(1)
    builder.raw(0x6A)  # i32.add
    builder.local_set(index_local)
    builder.br(0)  # continue the loop
    builder.end()  # loop
    builder.end()  # block

    data = tables["data"]
    # One page for the table, plus room for whatever the caller lays out
    # after it. A caller that needs more grows the memory itself.
    pages = max(1, (reserved_bytes + 65535) // 65536 + 1)
    return build_module(
        function_name=function_name,
        parameter_types=["i32"] * parameter_count,
        body=builder,
        memory_pages=pages,
        data=data,
        data_offset=int(tables.get("data_offset", 0)),
        imports=imports,
    )



# --- baked lookup tables ---------------------------------------------------
#
# WebAssembly has no transcendental instructions, so a function like tanh has
# to arrive as data rather than as an opcode. That is what llvm_signal_math
# already does for sine on the LLVM path: size a table from an absolute error
# target, interpolate linearly between entries, and state the bound. The same
# reasoning is used here so the two paths agree about what "accurate enough"
# means -- linear interpolation error is bounded by M*h^2/8, where M bounds
# the second derivative over the sampled interval.
#
# This is an approximation, and that is exactly why it is declared. The
# refusal elsewhere in this file is aimed at silently substituting a guess for
# an operation a caller asked for; a table whose error is chosen, bounded and
# tested is a different thing.

import math as _math

# max|tanh''| = 4/(3*sqrt(3)), at x = +/- asinh(1/sqrt(2)).
_TANH_CURVATURE = 4.0 / (3.0 * _math.sqrt(3.0))

# Periodic functions are sampled over one full turn and the argument is
# reduced into it. Reduction needs only floor, which WebAssembly has, so a
# periodic table stays exact for arguments of any size -- which matters here
# because the camera's angle grows without bound as the frame counter does.
_TAU = 2.0 * _math.pi

# tanh(8) = 0.99999977..., so clamping outside this costs less than the
# interpolation does inside it.
_TANH_LIMIT = 8.0

# Default absolute error target for a baked table, matching the scale
# llvm_signal_math treats as conservative for its own solvers.
DEFAULT_LUT_EPSILON = 1.0e-6


def _power_of_two_ceiling(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def lut_for(op: str, epsilon: float | None = None):
    """The table for one op, plus how its argument maps onto it.

    Returns (values, achieved_error, lower, upper, periodic). Sourced from
    the repository cache when it is present, because sampling fifteen
    functions to 1e-6 is deterministic and not worth repeating on every
    build; falls back to computing it, so a fresh checkout that has not run
    build_math_cache still works rather than failing at emission time.

    The error reported is the one the table was *measured* to deliver, not
    the one its sizing predicted. For a function whose curvature is singular
    at an endpoint the prediction is optimistic, and a caller reasoning about
    accuracy should see the worse number.
    """

    from .wasm_math_tables import DEFAULT_EPSILON, FUNCTIONS, build_table

    if epsilon is None:
        epsilon = DEFAULT_EPSILON
    function = FUNCTIONS.get(op)
    if function is None:
        raise WasmEmissionError(
            f"no baked table defined for {op!r}; the catalogue is "
            f"{sorted(FUNCTIONS)}"
        )
    if epsilon == DEFAULT_EPSILON:
        try:
            from .build_math_cache import load_manifest, load_table

            manifest = load_manifest()
            entry = manifest["tables"].get(op)
            if entry is not None:
                values = load_table(op)
                achieved = entry.get("achieved", entry.get("bound"))
                return (
                    list(values), achieved,
                    entry["lower"], entry["upper"], entry["periodic"],
                )
        except (FileNotFoundError, KeyError, OSError):
            pass
    table = build_table(op, epsilon)
    return (
        list(table.values), table.bound,
        table.lower, table.upper, table.periodic,
    )


def tanh_table(epsilon: float | None = None):
    """Kept because callers and tests already use this name; the definition
    now lives in the wasm_math_tables catalogue."""

    values, achieved, _lower, _upper, _periodic = lut_for("tanh", epsilon)
    return values, achieved


def plan_tables(ops, epsilon: float | None = None) -> dict:
    """Lay every table a program needs into one block of memory.

    Returns the packed bytes, the byte each table starts at, and the total
    reservation the caller must start past. Sorted, so the same program
    always produces the same module rather than one that depends on set
    iteration order.
    """

    import struct as _struct

    entries: dict[str, dict] = {}
    payload = bytearray()
    for op in sorted(ops):
        values, achieved, lower, upper, periodic = lut_for(op, epsilon)
        entries[op] = {
            "base": len(payload),
            "intervals": len(values) - 1,
            "lower": lower,
            "upper": upper,
            "periodic": periodic,
            "bound": achieved,
        }
        for value in values:
            payload += _struct.pack("<d", value)
    return {
        "entries": entries,
        "data": bytes(payload),
        "reserved_bytes": len(payload),
    }


def plan_static_data(
    steps: Sequence[OpStep],
    value_type: str,
    *,
    data_offset: int = 0,
) -> dict[str, Any]:
    """Pack lookup tables and varying tensor constants into module memory."""

    if data_offset < 0:
        raise ValueError("static data offset must be non-negative")

    import struct as _struct

    lut_ops = {step.op_name for step in steps} & _LUT_OPS
    if any(step.op_name == "pow" for step in steps):
        # pow composes exp and log tables even though it is not itself tabulated.
        lut_ops |= {"exp", "log"}
    tables = plan_tables(sorted(lut_ops))
    for entry in tables["entries"].values():
        entry["base"] += data_offset
    payload = bytearray(tables["data"])
    element_bytes = 4 if value_type == "f32" else 8
    pack_format = "<f" if value_type == "f32" else "<d"
    constants: dict[int, dict[str, int]] = {}
    shortfalls: list[WasmShortfall] = []

    for step in steps:
        if step.op_name != "tensor_from_list" or _constant_scalar(step) is not None:
            continue
        try:
            values = flatten_tensor_constant(step.attrs.get("values"))
        except (TypeError, ValueError) as exc:
            shortfalls.append(
                WasmShortfall(step.step_id, step.op_name, str(exc))
            )
            continue
        padding = (-len(payload)) % element_bytes
        if padding:
            payload.extend(bytes(padding))
        base = data_offset + len(payload)
        for value in values:
            payload.extend(_struct.pack(pack_format, value))
        constants[step.result_id] = {
            "base": base,
            "count": len(values),
        }

    return {
        "entries": tables["entries"],
        "constants": constants,
        "data": bytes(payload),
        "data_offset": data_offset,
        "reserved_bytes": data_offset + len(payload),
        "shortfalls": tuple(shortfalls),
    }


def _emit_lut(builder, source_local: int, op: str, table_base: int,
              intervals: int, lower: float, upper: float, periodic: bool) -> None:
    """Interpolate a baked table for the value in ``source_local``.

    Leaves the result on the stack. ``table_base`` is the byte offset the
    table was placed at; several tables share linear memory, so each is
    addressed from its own base rather than assuming offset zero.
    """

    from .wasm_binary import OP_F64_CONVERT_I32_S, OP_I32_TRUNC_F64_S

    span = upper - lower
    scale = intervals / span
    position = builder.declare_local("f64")
    index = builder.declare_local("i32")
    lower_value = builder.declare_local("f64")

    builder.local_get(source_local)
    if periodic:
        # x - floor(x / span) * span, so any argument lands in one period.
        builder.local_get(source_local)
        builder.value_const(1.0 / span)
        builder.op("mul")
        builder.op("floor")
        builder.value_const(span)
        builder.op("mul")
        builder.op("sub")
    else:
        builder.value_const(upper)
        builder.op("min")
        builder.value_const(lower)
        builder.op("max")
        builder.value_const(-lower)
        builder.op("add")
    builder.value_const(scale)
    builder.op("mul")
    builder.value_const(float(intervals) - 1.0e-9)
    builder.op("min")
    builder.value_const(0.0)
    builder.op("max")
    builder.local_set(position)

    builder.local_get(position)
    builder.raw(OP_I32_TRUNC_F64_S)
    builder.local_set(index)

    builder.local_get(index)
    builder.i32_const(8)
    builder.raw(0x6C)
    builder.load(offset=table_base)
    builder.local_set(lower_value)

    builder.local_get(lower_value)
    builder.local_get(index)
    builder.i32_const(8)
    builder.raw(0x6C)
    builder.load(offset=table_base + 8)
    builder.local_get(lower_value)
    builder.op("sub")
    builder.local_get(position)
    builder.local_get(index)
    builder.raw(OP_F64_CONVERT_I32_S)
    builder.op("sub")
    builder.op("mul")
    builder.op("add")


def feed_names(program: FusedProgram, feed_ids: Sequence[int]) -> list[str]:
    """What to call each feed in the descriptor.

    A program carries the source parameter each feed was bound to
    (``capture_feed_origins[...]["binding_name"]``), so the contract can say
    ``cx`` rather than ``feed0`` and a caller stops having to work out which
    array goes where. Falls back to a positional name when the program does
    not know -- a hand-built program, or one from a front end that does not
    record it -- because a positional name is still better than none.

    Names are made unique and identifier-safe: two parameters that collide,
    or one that is not a usable identifier, would produce a descriptor a
    caller cannot bind against.
    """

    origins = (program.extras or {}).get("capture_feed_origins", {}) or {}
    used: set[str] = set()
    names: list[str] = []
    for index, feed_id in enumerate(feed_ids):
        raw = (origins.get(feed_id) or {}).get("binding_name")
        candidate = str(raw) if raw else f"feed{index}"
        if not candidate.isidentifier():
            candidate = f"feed{index}"
        if candidate in used:
            candidate = f"{candidate}_{index}"
        used.add(candidate)
        names.append(candidate)
    return names


def _describe(
    name: str,
    function_name: str,
    feed_ids: Sequence[int],
    output_ids: Sequence[int],
    value_type: str,
    element_bytes: int,
    reserved_bytes: int = 0,
    static_data_offset: int = 0,
    shared_memory_import: Mapping[str, str] | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    parameter_dtypes: Mapping[int, str] | None = None,
):
    """The same calling-contract descriptor the Fortran path emits.

    Every value here is still stored and computed as ``value_type`` -- this
    module's arithmetic is one uniform elementwise numeric type, and that is
    unchanged. ``parameter_dtypes`` (keyed by feed/output value id, from the
    same ``FusedProgram.meta`` the code generator already reads) only
    corrects the *label* a caller sees for a parameter that logically holds
    something else -- a byte buffer, a small integer counter -- so a reader
    interprets the bytes it gets back correctly instead of assuming every
    parameter is the module's numeric working type. A value that does not
    survive its real dtype through this uniform-arithmetic path (anything
    needing exact precision ``value_type`` cannot represent, such as a full
    64-bit integer through float64 arithmetic) is not fixed by this label;
    that is a real backend limitation, not a caller-facing one.
    """

    from .compiled_program_api import CompiledProgramAPI, EntryPoint, Parameter, _C_TYPES

    dtypes = {
        int(key): str(value)
        for key, value in (parameter_dtypes or {}).items()
        if str(value) in _C_TYPES
    }

    def dtype_for(value_id: int) -> str:
        return dtypes.get(int(value_id), value_type)

    parameters = [
        Parameter(
            name="count",
            role="extent",
            dtype="int32",
            c_type="int32_t",
            ctypes_name="c_int32",
            passing="value",
        )
    ]
    labels = list(input_names or [f"feed{i}" for i in range(len(feed_ids))])
    for index, feed_id in enumerate(feed_ids):
        parameters.append(
            Parameter(
                name=labels[index],
                role="input",
                dtype=dtype_for(feed_id),
                c_type="int32_t",
                ctypes_name="c_int32",
                # A WebAssembly parameter is always by value; what it holds
                # is a byte offset into the exported memory, which is what a
                # caller needs to know.
                passing="value",
                extent="count",
            )
        )
    for index, output_id in enumerate(output_ids):
        # A program that named its outputs gets to keep those names: the
        # caller reads them, and "red" carries information "out0" does not.
        parameters.append(
            Parameter(
                name=(output_names[index] if output_names and index < len(output_names)
                      else f"out{index}"),
                role="output",
                dtype=dtype_for(output_id),
                c_type="int32_t",
                ctypes_name="c_int32",
                passing="value",
                extent="count",
            )
        )
    return CompiledProgramAPI(
        module=name,
        language="wasm",
        entry=function_name,
        entry_points=(
            EntryPoint(
                name=function_name,
                symbol=function_name,
                kind="numerical",
                parameters=tuple(parameters),
                note=(
                    "every argument after count is a byte offset into the "
                    "module's exported memory, which the caller owns and "
                    "fills"
                ),
            ),
        ),
        metadata={
            "value_type": value_type,
            "element_bytes": element_bytes,
            "memory_export": "memory",
            # A baked table sits at offset 0, so a caller's arrays start
            # here. Zero when the program needed no table.
            "reserved_bytes": int(reserved_bytes),
            "static_data_offset": int(static_data_offset),
            "shared_memory_import": dict(shared_memory_import or {}),
        },
    )


def wat_assembler() -> str | None:
    """``wat2wasm`` if it is installed. Emission never needs it."""

    return shutil.which("wat2wasm")


def compile_wat(module: WasmModule, *, directory: str | Path | None = None) -> Path:
    """Assemble WAT to a ``.wasm`` binary, if an assembler is present."""

    if not module.complete:
        raise WasmEmissionError(module.shortfall_report())
    assembler = wat_assembler()
    if assembler is None:
        raise WasmEmissionError(
            "no wat2wasm found; emission does not require one, but "
            "compile_wat does. Install WABT, or assemble the .wat yourself."
        )
    workdir = Path(directory or tempfile.mkdtemp(prefix="turing_wasm_"))
    source_path = module.write(workdir)
    binary_path = source_path.with_suffix(".wasm")
    completed = subprocess.run(
        [assembler, str(source_path), "-o", str(binary_path)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise WasmEmissionError(
            "wat2wasm failed:\n" + (completed.stderr or completed.stdout)
        )
    return binary_path


__all__ = [
    "WasmEmissionError",
    "WasmModule",
    "WasmShortfall",
    "compile_wat",
    "emit_wasm_module",
    "wat_assembler",
]
