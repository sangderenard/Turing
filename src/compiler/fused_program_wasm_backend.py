"""Emit WebAssembly -- both the readable text and the runnable binary -- from
a ``FusedProgram``.

Why this IR and not SSA or Fortran: WebAssembly has no ``goto``. Its control
flow is structured (``block``/``loop``/``br``), so lowering an arbitrary SSA
control-flow graph needs a relooper -- a real algorithm, not a translation.
``FusedProgram`` is the one intermediary that sidesteps it entirely: a flat,
topologically ordered list of ``OpStep`` with no branches at all (see
``fused_ir.py``). The only loop in the emitted module is the elementwise walk
over the extent, which this file writes itself.

So this is the same shape as ``fused_program_python_backend.py`` -- one
lowering per ``OpStep``, in the program's own order -- with a different
instruction set underneath.

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
    ordered_feed_ids,
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
# happens near the pole. The rest are shape or predicate operations rather
# than functions of one float.
_NO_WASM_INSTRUCTION = {
    "pow", "mod", "floordiv", "tan",
    "sign", "isfinite", "isnan", "isinf", "logical_not",
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
    values = step.attrs.get("values")
    while isinstance(values, (list, tuple)) and len(values) == 1:
        values = values[0]
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        return float(values)
    return None


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


def emit_wasm_module(
    program: FusedProgram,
    *,
    name: str = "fused_program",
    function_name: str = "run",
    dtype: str | None = None,
    imports: Sequence[object] = (),
) -> WasmModule:
    """Lower one elementwise ``FusedProgram`` to a WAT module.

    The emitted function is ``(count, feed0, feed1, ..., out0, ...)`` where
    every argument after ``count`` is a byte offset into the exported memory.

    ``imports`` (``wasm_binary.WasmImport`` entries) wires this module to
    another module's exported function/memory -- see
    ``wasm_class_modules.py``. It defaults to empty, reproducing the exact
    single-module output this function has always produced; every existing
    caller (``build_homepage.py`` included) is unaffected.
    """

    value_type, element_bytes, load, store = _value_type(program, dtype)
    shortfalls: list[WasmShortfall] = []

    feed_ids = program_feed_order(program)
    output_ids = list(program.outputs.values())
    names: dict[int, str] = {}
    labels = feed_names(program, feed_ids)
    parameters: list[str] = ["$count"]
    for index, feed_id in enumerate(feed_ids):
        parameters.append("$" + labels[index])
    for index, _ in enumerate(output_ids):
        parameters.append(f"$out{index}")

    body: list[str] = []
    locals_declared: list[str] = ["(local $i i32)", "(local $addr i32)"]

    def element_address(pointer: str) -> list[str]:
        # addr = pointer + i * element_bytes
        return [
            f"      local.get {pointer}",
            "      local.get $i",
            f"      i32.const {element_bytes}",
            "      i32.mul",
            "      i32.add",
        ]

    # Feeds are read once per iteration into locals, so a value used by more
    # than one step is loaded once rather than re-read from memory.
    for index, feed_id in enumerate(feed_ids):
        local = f"$v{len(names)}"
        names[feed_id] = local
        locals_declared.append(f"(local {local} {value_type})")
        # The same label the signature declares -- emitting $feed{index}
        # here while the header says $unit_x produces WAT that does not
        # refer to its own parameters.
        body.extend(element_address("$" + labels[index]))
        body.append(f"      {load}")
        body.append(f"      local.set {local}")

    for step in required_steps(program):
        local = f"$v{len(names)}"
        locals_declared.append(f"(local {local} {value_type})")
        instructions = _step_instructions(step, names, value_type, shortfalls)
        if instructions is None:
            # Still bind a name so later steps referring to this result do
            # not also fail; the module is incomplete either way.
            names[step.result_id] = local
            continue
        body.extend(instructions)
        body.append(f"      local.set {local}")
        names[step.result_id] = local

    for index, output_id in enumerate(output_ids):
        target = names.get(output_id)
        if target is None:
            shortfalls.append(
                WasmShortfall(-1, "output", f"value {output_id} is never produced")
            )
            continue
        body.extend(element_address(f"$out{index}"))
        body.append(f"      local.get {target}")
        body.append(f"      {store}")

    parameter_text = " ".join(f"(param {p} i32)" for p in parameters)
    lines = [
        f"(module ;; {name}",
        "  ;; The caller owns memory: it writes the feeds in and reads the",
        "  ;; outputs back. A fused elementwise program keeps no state.",
        "  (memory (export \"memory\") 1)",
        f"  (func (export \"{function_name}\") {parameter_text}",
        *(f"    {declaration}" for declaration in locals_declared),
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

    reserved = plan_tables(
        sorted({step.op_name for step in required_steps(program)} & _LUT_OPS)
    )["reserved_bytes"]
    api = _describe(name, function_name, feed_ids, output_ids, value_type,
                    element_bytes, reserved,
                    input_names=feed_names(program, feed_ids),
                    output_names=list(program.outputs.keys()))
    binary = None
    if not shortfalls:
        binary = _assemble(
            program, feed_ids, output_ids, value_type, element_bytes,
            function_name, imports=imports,
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
    shortfalls: list[WasmShortfall],
) -> list[str] | None:
    op = step.op_name
    constant = _constant_scalar(step)
    if constant is not None:
        return [f"      {value_type}.const {constant!r}"]
    if op == "tensor_from_list":
        shortfalls.append(
            WasmShortfall(
                step.step_id, op,
                "only a one-element constant can become an immediate; a real "
                "array constant would have to be placed in linear memory",
            )
        )
        return None
    if op in _NO_WASM_INSTRUCTION:
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

    if op in _LUT_OPS:
        # The binary carries the table and the interpolation; the text form
        # names the step instead of inlining it, so the two are not pretending
        # to be line-for-line the same. The binary is the artifact that runs.
        return [f"      ;; {op} via baked lookup table (see the .wasm)",
                f"      local.get {left}"]

    if op in ELEMENTWISE_UNARY:
        instruction = _UNARY_INSTRUCTION.get(op)
        if instruction is None:
            shortfalls.append(
                WasmShortfall(step.step_id, op, "no unary instruction registered")
            )
            return None
        return [f"      local.get {left}", f"      {value_type}.{instruction}"]

    if op not in ELEMENTWISE_BINARY:
        shortfalls.append(
            WasmShortfall(step.step_id, op, "not an elementwise operation")
        )
        return None

    if len(step.input_ids) == 2:
        right_source = [f"      local.get {names[step.input_ids[1]]}"]
    elif "right_scalar" in step.attrs:
        right_source = [
            f"      {value_type}.const {float(step.attrs['right_scalar'])!r}"
        ]
    else:
        shortfalls.append(
            WasmShortfall(step.step_id, op, "binary step has no right operand")
        )
        return None

    operands = [f"      local.get {left}", *right_source]
    if step.attrs.get("reverse", False):
        operands = [*right_source, f"      local.get {left}"]

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
    imports: Sequence[object] = (),
) -> bytes:
    """Assemble the same program as a binary module.

    Mirrors the WAT lowering above step for step -- same order, same
    operands, same instructions -- because the two forms describing different
    programs would be worse than having only one.
    """

    from .wasm_binary import CodeBuilder, build_module

    live = required_steps(program)
    lut_ops = sorted({step.op_name for step in live} & _LUT_OPS)
    if lut_ops and value_type != "f64":
        raise WasmEmissionError(
            f"{lut_ops} is only baked for f64; an f32 table would need its "
            "own sampling and has no caller yet"
        )
    tables = plan_tables(lut_ops)
    reserved_bytes = tables["reserved_bytes"]

    parameter_count = 1 + len(feed_ids) + len(output_ids)
    builder = CodeBuilder(value_type=value_type, parameter_count=parameter_count)
    count_param = 0
    feed_params = {feed_id: 1 + i for i, feed_id in enumerate(feed_ids)}
    output_params = [1 + len(feed_ids) + i for i in range(len(output_ids))]

    index_local = builder.declare_local("i32")
    locals_for: dict[int, int] = {}

    def element_address(pointer_param: int) -> None:
        builder.local_get(pointer_param)
        builder.local_get(index_local)
        builder.i32_const(element_bytes)
        builder.raw(0x6C)  # i32.mul
        builder.raw(0x6A)  # i32.add

    # block { loop { if i >= count break; ...; i += 1; continue } }
    builder.block()
    builder.loop()
    builder.local_get(index_local)
    builder.local_get(count_param)
    builder.raw(0x4E)  # i32.ge_s
    builder.br_if(1)  # out of the enclosing block

    for feed_id in feed_ids:
        local = builder.declare_local(value_type)
        locals_for[feed_id] = local
        element_address(feed_params[feed_id])
        builder.load()
        builder.local_set(local)

    for step in live:
        local = builder.declare_local(value_type)
        constant = _constant_scalar(step)
        if constant is not None:
            builder.value_const(constant)
            locals_for[step.result_id] = local
            builder.local_set(local)
            continue
        left = locals_for[step.input_ids[0]]

        def push_right() -> None:
            if len(step.input_ids) == 2:
                builder.local_get(locals_for[step.input_ids[1]])
            else:
                builder.value_const(float(step.attrs["right_scalar"]))

        if step.op_name in _LUT_OPS:
            entry = tables["entries"][step.op_name]
            _emit_lut(
                builder, left, step.op_name, entry["base"], entry["intervals"],
                entry["lower"], entry["upper"], entry["periodic"],
            )
        elif step.op_name in ELEMENTWISE_UNARY:
            builder.local_get(left)
            builder.op(_UNARY_INSTRUCTION[step.op_name])
        elif step.attrs.get("reverse", False):
            push_right()
            builder.local_get(left)
        else:
            builder.local_get(left)
            push_right()

        if step.op_name in ELEMENTWISE_BINARY:
            instruction = _BINARY_INSTRUCTION.get(step.op_name)
            if instruction is not None:
                builder.op(instruction)
            else:
                builder.op(_COMPARISON_INSTRUCTION[step.op_name])
                # A comparison yields i32; convert so the result carries the
                # operand's own type, as every other backend reports it.
                builder.op("convert_i32_u")
        locals_for[step.result_id] = local
        builder.local_set(local)

    for slot, output_id in enumerate(output_ids):
        element_address(output_params[slot])
        builder.local_get(locals_for[output_id])
        builder.store()

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
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
):
    """The same calling-contract descriptor the Fortran path emits."""

    from .compiled_program_api import CompiledProgramAPI, EntryPoint, Parameter

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
    for index, _ in enumerate(feed_ids):
        parameters.append(
            Parameter(
                name=labels[index],
                role="input",
                dtype=value_type,
                c_type="int32_t",
                ctypes_name="c_int32",
                # A WebAssembly parameter is always by value; what it holds
                # is a byte offset into the exported memory, which is what a
                # caller needs to know.
                passing="value",
                extent="count",
            )
        )
    for index, _ in enumerate(output_ids):
        # A program that named its outputs gets to keep those names: the
        # caller reads them, and "red" carries information "out0" does not.
        parameters.append(
            Parameter(
                name=(output_names[index] if output_names and index < len(output_names)
                      else f"out{index}"),
                role="output",
                dtype=value_type,
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
