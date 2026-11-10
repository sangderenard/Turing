"""Direct scalar repository-SSA to WebAssembly emission."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..transmogrifier.ssa import Function, IRModule
from .hierarchical_plan import TENSOR_OPERATION_SCALAR_SPELLING
from .output_publication import (
    function_output_publications,
    publication_surface_plan,
)
from .wasm_binary import (
    CodeBuilder,
    OP_F64_CONVERT_I32_S,
    OP_I32_ADD,
    OP_I32_AND,
    OP_I32_GE_S,
    OP_I32_TRUNC_F64_S,
    build_module,
)


def _settled_dtype(dtype, value_id, record):
    """Return a dtype the pipeline settled, or refuse by name."""

    if dtype is None or str(dtype) == "":
        raise ValueError(
            f"backend: value {value_id} has no settled dtype; missing "
            f"record: {record}"
        )
    return dtype


@dataclass(frozen=True, slots=True)
class WasmEmissionShortfall:
    operation: str
    reason: str


@dataclass(frozen=True, slots=True)
class SSAWasmArtifact:
    name: str
    wat: str
    binary: bytes
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    input_offsets: tuple[int, ...]
    output_offsets: tuple[int, ...]
    shortfalls: tuple[WasmEmissionShortfall, ...]
    output_publications: tuple[Mapping[str, Any], ...] = ()
    output_surfaces: Mapping[str, Any] | None = None

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def write(self, directory: str | Path) -> tuple[Path, Path]:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        wat_path = destination / f"{self.name}.wat"
        wasm_path = destination / f"{self.name}.wasm"
        wat_path.write_text(self.wat, encoding="utf-8")
        wasm_path.write_bytes(self.binary)
        return wat_path, wasm_path


#: How far a whole-number exponent is unrolled into multiplies.
#: Eight covers every power that appears in section properties,
#: modal frequencies and contact laws; past that the instruction
#: count stops being worth it and a shortfall is more honest.
_MAX_UNROLLED_POWER = 8

#: SSA comparison names to the f64 opcode that spells each.
_F64_COMPARISONS = {"Lt": "lt", "Le": "le", "Gt": "gt",
                    "Ge": "ge", "Eq": "eq", "Ne": "ne"}


def emit_ssa_function_to_wasm(
    module: IRModule, function_name: str, *, entry_name: str | None = None,
    trig_solver: str = "lut", trig_epsilon: float | None = None,
    work_contract: str | None = None,
) -> SSAWasmArtifact:
    """Emit one scalar SSA function as a WebAssembly module.

    ``trig_solver`` selects how a function WebAssembly has no instruction for
    is realised, mirroring ``LLVMTrigSolver`` in the LLVM lane: ``"lut"``
    bakes the sampled table into linear memory, ``"continuous"`` evaluates a
    reduced argument series in arithmetic alone and needs no data segment.
    Both are available deliberately -- a table costs memory and an interpolation,
    a series costs multiplies, and which is cheaper depends on the target.
    """
    from .ssa_llvm_backend import (
        DOUBLE_EXACT, INT32_EXACT, python_int_unchecked_refusal,
    )
    from .work_contract import PRESETS, active_contract

    function: Function = module.functions[function_name]
    name = str(entry_name or function_name)
    if work_contract is None:
        contract = active_contract()
    else:
        contract = PRESETS.get(str(work_contract).strip().lower())
        if contract is None:
            raise ValueError(
                f"unknown work contract {work_contract!r}; presets: "
                f"{sorted(PRESETS)}"
            )
    input_names = tuple(function.metadata.get("argument_names", ()))
    output_names = tuple(function.metadata.get("output_names", ()))
    if set(function.blocks) != {"entry"}:
        return SSAWasmArtifact(
            name, "", b"", input_names, output_names, (), (),
            (WasmEmissionShortfall("control", "direct scalar WASM requires one entry block"),),
        )
    if len(input_names) != len(function.args):
        input_names = tuple(f"arg{index}" for index in range(len(function.args)))

    # WebAssembly has no transcendental instruction, so these reach the module
    # as baked lookup tables -- the same tables, laid out by the same planner,
    # that the fused lane already uses. The tables occupy the front of linear
    # memory, so the value arena starts past them.
    from .fused_program_wasm_backend import plan_tables

    _TABLE_OPS = {"Sin": "sin", "Cos": "cos", "Tan": "tan", "Exp": "exp",
                  "Log": "log", "Tanh": "tanh"}
    required_tables = sorted({
        _TABLE_OPS[str(instruction.op)]
        for instruction in function.blocks["entry"].instrs
        if str(instruction.op) in _TABLE_OPS
    })
    if str(trig_solver) not in {"lut", "continuous"}:
        raise ValueError(
            f"unknown trig solver {trig_solver!r}; expected 'lut' or "
            "'continuous'"
        )
    tables = (
        plan_tables(required_tables, trig_epsilon)
        if required_tables and str(trig_solver) == "lut"
        else {"entries": {}, "data": b"", "reserved_bytes": 0}
    )
    # Keep the arena eight-byte aligned behind the tables.
    arena_base = (int(tables["reserved_bytes"]) + 7) // 8 * 8

    builder = CodeBuilder("f64", parameter_count=1)
    locals_by_id: dict[int, int] = {}
    wat_lines = [
        "(module",
        "  (memory (export \"memory\") 1)",
        f"  (func (export \"{name}\") (param $io i32)",
    ]
    all_result_ids = {
        int(instruction.res.id)
        for instruction in function.blocks["entry"].instrs
        if instruction.res is not None
    }
    for value in function.args:
        all_result_ids.add(int(value.id))
    for value_id in sorted(all_result_ids):
        locals_by_id[value_id] = builder.declare_local("f64")
        wat_lines.append(f"    (local $t{value_id} f64)")
    input_offsets = tuple(
        arena_base + index * 8 for index in range(len(function.args))
    )
    for value, offset in zip(function.args, input_offsets):
        value_id = int(value.id)
        builder.local_get(0).i32_const(offset).raw(OP_I32_ADD).load().local_set(
            locals_by_id[value_id]
        )
        wat_lines.append(
            f"    local.get $io i32.const {offset} i32.add f64.load local.set $t{value_id}"
        )

    constants: dict[int, float] = {}
    outputs: tuple[int, ...] = ()
    shortfalls: list[WasmEmissionShortfall] = []
    from .ir_identities import precision_backend_shortfalls
    shortfalls.extend(
        WasmEmissionShortfall(
            "precision_section",
            "backend cannot honour precision obligations "
            + repr(item["missing"]),
        )
        for item in precision_backend_shortfalls(
            module, "wasm", (function_name,)
        )
    )

    def get(value_id: int) -> None:
        builder.local_get(locals_by_id[int(value_id)])

    for instruction in function.blocks["entry"].instrs:
        # Ingested scalar SSA arrives with graph-native spellings
        # (``maximum``/``minimum``); read them through the one table every
        # SSA reader shares, as ``ssa_c_backend.emit_ssa_function_to_c`` does.
        op = str(instruction.op)
        op = TENSOR_OPERATION_SCALAR_SPELLING.get(op.casefold(), op)
        if op == "Ret":
            outputs = tuple(int(value.id) for value in instruction.args)
            continue
        if instruction.res is None:
            shortfalls.append(WasmEmissionShortfall(op, "instruction has no result"))
            continue
        unchecked = python_int_unchecked_refusal(
            instruction, "direct scalar WASM f64", DOUBLE_EXACT,
        )
        if unchecked is not None:
            shortfalls.append(WasmEmissionShortfall(op, unchecked))
            continue
        result_id = int(instruction.res.id)
        if op == "Const":
            value = float(instruction.attributes.get("constant", instruction.attributes.get("value")))
            constants[result_id] = value
            builder.value_const(value).local_set(locals_by_id[result_id])
            wat_lines.append(f"    f64.const {value.hex()} local.set $t{result_id}")
            continue
        if op == "Pi":
            # The shared materialisation, as in the C and LLVM lanes, so the
            # four backends carry one constant with one declared bound rather
            # than four literals that can drift apart.
            from .bounded_constants import materialize_pi

            materialization = materialize_pi(
                instruction.attributes.get("constant_solver") or "literal",
                instruction.attributes.get("requested_epsilon"),
            )
            if materialization.value is None:
                shortfalls.append(WasmEmissionShortfall(
                    op, "pi materialisation was rejected",
                ))
                continue
            value = float(materialization.value)
            constants[result_id] = value
            builder.value_const(value).local_set(locals_by_id[result_id])
            wat_lines.append(f"    f64.const {value.hex()} local.set $t{result_id}")
            continue
        args = tuple(int(value.id) for value in instruction.args)
        wat_operation = None
        if op in {"Cast", "CastLike", "cast_like"} and len(args) >= 1:
            target = str(
                instruction.attributes.get("target_dtype")
                or _settled_dtype(
                    instruction.res.dtype, instruction.res.id,
                    "Cast target_dtype attribute or result dtype")
            ).casefold()
            get(args[0])
            if target in {"bool", "i1"}:
                builder.value_const(0.0).op("ne").op("convert_i32_u")
                wat_operation = (
                    f"local.get $t{args[0]} f64.const 0x0p+0 "
                    "f64.ne f64.convert_i32_u"
                )
            elif target in {"int", "int32", "i32"}:
                builder.raw(OP_I32_TRUNC_F64_S, OP_F64_CONVERT_I32_S)
                wat_operation = (
                    f"local.get $t{args[0]} "
                    "i32.trunc_f64_s f64.convert_i32_s"
                )
            else:
                wat_operation = f"local.get $t{args[0]}"
        elif op.casefold() == "fma" and len(args) == 3:
            # WebAssembly HAS NO FMA INSTRUCTION, and this is the expansion
            # into a multiply and an add -- which rounds twice and is
            # therefore NOT an fma. Emitted anyway so the four lanes present
            # the same operation and a program compiles here at all, but the
            # difference is not cosmetic: on a precision dual, where the
            # whole value of the fma is that `a * b - fl(a * b)` keeps the
            # residual, two roundings return exactly zero.
            #
            # So this lane does not declare FMA_MANDATORY in
            # BACKEND_PRECISION_CAPABILITIES, and a precision section asking
            # for it is told before emission rather than after. Ordinary
            # code that merely wanted the accuracy gets the arithmetic it
            # asked for; precision code gets refused. Both are correct, and
            # the ledger is what tells them apart.
            get(args[0]); get(args[1]); builder.op("mul")
            get(args[2]); builder.op("add")
            wat_operation = (
                f"local.get $t{args[0]} local.get $t{args[1]} f64.mul "
                f"local.get $t{args[2]} f64.add"
            )
        elif op in {"Add", "Sub", "Mul", "Div", "Max", "Min"} and len(args) == 2:
            get(args[0]); get(args[1]); builder.op(op.lower())
            wat_operation = f"local.get $t{args[0]} local.get $t{args[1]} f64.{op.lower()}"
        elif str(op) in _TABLE_OPS and str(trig_solver) == "continuous" and (
            str(op) in {"Sin", "Cos"} and len(args) == 1
        ):
            # Argument reduced onto [-pi/2, pi/2], then the odd series through
            # r^13, whose truncation error there is (pi/2)^15 / 15! ~ 7e-10.
            # No data segment is needed, which is the point of the option.
            from .bounded_constants import materialize_pi

            pi = float(materialize_pi("literal").value)
            shifted = builder.declare_local("f64")
            rounded = builder.declare_local("f64")
            reduced = builder.declare_local("f64")
            squared = builder.declare_local("f64")
            series = builder.declare_local("f64")
            parity = builder.declare_local("i32")
            get(args[0])
            if op == "Cos":
                builder.value_const(pi * 0.5).op("add")
            builder.local_set(shifted)
            builder.local_get(shifted).value_const(1.0 / pi).op("mul")
            builder.op("nearest").local_set(rounded)
            builder.local_get(rounded).raw(OP_I32_TRUNC_F64_S)
            builder.i32_const(1).raw(OP_I32_AND).local_set(parity)
            builder.local_get(shifted)
            builder.local_get(rounded).value_const(pi).op("mul")
            builder.op("sub").local_set(reduced)
            builder.local_get(reduced).local_get(reduced).op("mul")
            builder.local_set(squared)
            for index, coefficient in enumerate((
                1.0 / 6227020800.0, -1.0 / 39916800.0, 1.0 / 362880.0,
                -1.0 / 5040.0, 1.0 / 120.0, -1.0 / 6.0, 1.0,
            )):
                if index == 0:
                    builder.value_const(coefficient)
                else:
                    builder.local_get(squared).op("mul")
                    builder.value_const(coefficient).op("add")
            builder.local_get(reduced).op("mul").local_set(series)
            builder.local_get(series).op("neg")
            builder.local_get(series)
            builder.local_get(parity)
            builder.select()
            wat_operation = f"(; {op.lower()} by reduced-argument series ;)"
        elif str(op) in _TABLE_OPS and len(args) == 1:
            from .fused_program_wasm_backend import _emit_lut

            entry = tables["entries"][_TABLE_OPS[str(op)]]
            _emit_lut(
                builder, locals_by_id[args[0]], _TABLE_OPS[str(op)],
                entry["base"], entry["intervals"], entry["lower"],
                entry["upper"], entry["periodic"],
            )
            wat_operation = (
                f"(; {_TABLE_OPS[str(op)]} from the baked table at "
                f"byte {entry['base']}, measured error {entry['bound']:.2e} ;)"
            )
        elif op in {"Abs", "Sqrt", "Neg"} and len(args) == 1:
            get(args[0]); builder.op(op.lower())
            wat_operation = f"local.get $t{args[0]} f64.{op.lower()}"
        elif str(op) in _F64_COMPARISONS and len(args) == 2:
            # A COMPARISON IS AN INSTRUCTION HERE, not a missing feature.
            # WebAssembly has f64.le and the whole family; the table in
            # wasm_binary has carried their opcodes all along and this
            # emitter simply never reached for them, so any law
            # containing a Piecewise -- steel softening with temperature,
            # a contact that switches on, a limit that saturates -- came
            # back as "no direct scalar WASM spelling" and could not be
            # published as a component at all.
            #
            # The result is i32; everything else on this stack is f64, so
            # it is converted back the same way the mask reaches i1 in
            # the LLVM lane: by its truthiness, not by a special type.
            get(args[0]); get(args[1])
            builder.op(_F64_COMPARISONS[str(op)])
            builder.op("convert_i32_u")
            wat_operation = (f"local.get $t{args[0]} local.get $t{args[1]} "
                             f"f64.{_F64_COMPARISONS[str(op)]} "
                             f"f64.convert_i32_u")
        elif str(op) in {"Select", "where"} and len(args) == 3:
            # Select(mask, when_true, when_false) -- the same order the
            # LLVM lane uses, so one SSA program means one thing on both
            # targets. WASM's own `select` pops (val1, val2, condition)
            # and keeps val1 when the condition is non-zero, so the two
            # values go on in that order and the mask is reduced to i32
            # by comparing it against zero.
            get(args[1]); get(args[2]); get(args[0])
            builder.value_const(0.0)
            builder.op("ne")
            builder.select()
            wat_operation = (f"local.get $t{args[1]} local.get $t{args[2]} "
                             f"local.get $t{args[0]} f64.const 0 f64.ne select")
        elif op == "Pow" and len(args) == 2:
            exponent = constants.get(args[1])
            if exponent == 2.0:
                get(args[0]); get(args[0]); builder.op("mul")
                wat_operation = f"local.get $t{args[0]} local.get $t{args[0]} f64.mul"
            elif exponent == -1.0:
                builder.value_const(1.0); get(args[0]); builder.op("div")
                wat_operation = f"f64.const 0x1.0000000000000p+0 local.get $t{args[0]} f64.div"
            elif (exponent is not None and float(exponent).is_integer()
                  and 2 <= abs(int(exponent)) <= _MAX_UNROLLED_POWER):
                # A WHOLE-NUMBER POWER IS REPEATED MULTIPLICATION, and
                # that is not a reduction, an identity or a policy
                # choice -- it is what the exponent MEANS. Scalar WASM
                # has no pow instruction, so multiplication is not a
                # cheaper way of getting there, it is the only way; and
                # x**2 was already being spelled exactly like this three
                # lines up. Stopping at 2 meant a second moment of area
                # -- pi/4 (ro^4 - ri^4), the most ordinary expression in
                # beam theory there is -- could not lower to the game
                # engine's own target at all, under any contract.
                #
                # Left-to-right, so the association is the one the
                # source would have had if it were written out, and the
                # rounding is therefore the rounding a reader expects.
                n = abs(int(exponent))
                get(args[0])
                for _ in range(n - 1):
                    get(args[0])
                    builder.op("mul")
                chain = " ".join([f"local.get $t{args[0]}"]
                                 + [f"local.get $t{args[0]} f64.mul"] * (n - 1))
                if int(exponent) < 0:
                    # 1/x^n rather than x^-n, for the same reason -1.0
                    # above is spelled as a divide
                    builder.local_set(temp := builder.temp())
                    builder.value_const(1.0)
                    builder.local_get(temp)
                    builder.op("div")
                    chain = (f"f64.const 0x1.0000000000000p+0 {chain} f64.div")
                wat_operation = chain
            elif exponent == -2.0 and contract.inexact_identities:
                builder.value_const(1.0); get(args[0]); get(args[0]); builder.op("mul"); builder.op("div")
                wat_operation = (
                    "f64.const 0x1.0000000000000p+0 "
                    f"local.get $t{args[0]} local.get $t{args[0]} f64.mul f64.div"
                )
            elif exponent == 0.5 and contract.inexact_identities:
                get(args[0])
                builder.op("sqrt")
                wat_operation = f"local.get $t{args[0]} f64.sqrt"
            elif exponent == -0.5 and contract.inexact_identities:
                builder.value_const(1.0)
                get(args[0])
                builder.op("sqrt")
                builder.op("div")
                wat_operation = (
                    "f64.const 0x1.0000000000000p+0 "
                    f"local.get $t{args[0]} f64.sqrt f64.div"
                )
            elif exponent in (0.5, -0.5, -2.0):
                # These spellings change bits (the sqrt family); scalar WASM
                # has no pow instruction to fall back on, so exact-only
                # contracts get an honest shortfall instead of a silent
                # policy violation.
                shortfalls.append(WasmEmissionShortfall(
                    op,
                    f"no exact scalar WASM spelling for exponent {exponent!r} "
                    f"under contract {contract.name!r}; deploy/fast permit "
                    "the sqrt-family reduction",
                ))
                continue
            else:
                shortfalls.append(WasmEmissionShortfall(
                    op, f"WebAssembly has no direct power for exponent {exponent!r}",
                ))
                continue
        else:
            shortfalls.append(WasmEmissionShortfall(op, "no direct scalar WASM spelling"))
            continue
        builder.local_set(locals_by_id[result_id])
        wat_lines.append(f"    {wat_operation} local.set $t{result_id}")

    if not output_names:
        output_names = tuple(f"output{index}" for index in range(len(outputs)))
    if len(output_names) != len(outputs):
        shortfalls.append(WasmEmissionShortfall("Ret", "output names do not match return arity"))
    output_offsets = tuple(
        arena_base + (len(function.args) + index) * 8
        for index in range(len(outputs))
    )
    for value_id, offset in zip(outputs, output_offsets):
        builder.local_get(0).i32_const(offset).raw(OP_I32_ADD)
        get(value_id)
        builder.store()
        wat_lines.append(
            f"    local.get $io i32.const {offset} i32.add local.get $t{value_id} f64.store"
        )
    wat_lines.extend(("  )", ")", ""))
    arena_bytes = arena_base + (len(function.args) + len(outputs)) * 8
    binary = b"" if shortfalls else build_module(
        function_name=name,
        parameter_types=["i32"],
        body=builder,
        memory_pages=max(1, (arena_bytes + 65535) // 65536),
        data=tables["data"],
        data_offset=0,
    )
    publications = function_output_publications(function)
    return SSAWasmArtifact(
        name,
        "\n".join(wat_lines),
        binary,
        input_names,
        output_names,
        input_offsets,
        output_offsets,
        tuple(shortfalls),
        publications,
        publication_surface_plan(publications, target="wasm"),
    )


# ---------------------------------------------------------------------------
# Repository-call module emission: the batched lane.
#
# The scalar artifact above is one straight-line block behind slot-in-memory
# scalars. The precision benchmark's kernels are a counted loop calling a
# planned region once per element, addressing arrays through
# GetElementPtr/Load/Store. This emitter takes that whole module and produces
# ONE exported function: the wrapper's CFG is translated with the classic
# label-dispatcher (a loop over nested blocks selected by a label local --
# WebAssembly's structured control flow spelling of an arbitrary branch), and
# the region is INLINED at its call site, because a single-block callee
# invoked at a single site is cheaper to splice than to give its own function
# and marshalling.
#
# The ABI mirrors the classification the C module lane uses: each root formal
# becomes one parameter -- an i32 byte offset into exported linear memory for
# a formal something addresses through, the value itself (i32/f64) for a
# scalar. The host lays the arrays out in memory, passes their offsets, and
# reads results back from memory.
# ---------------------------------------------------------------------------

import base64 as _base64
import json as _json
import struct as _struct
import subprocess as _subprocess

# Opcodes the module lane needs beyond the scalar lane's imports; numeric
# values are the WebAssembly spec's, mirrored from wasm_binary's tables.
_OP_UNREACHABLE = 0x00
_OP_RETURN = 0x0F
_OP_I32_EQ = 0x46
_OP_I32_LT_S = 0x48
_OP_I32_SUB = 0x6B
_OP_I32_MUL = 0x6C
_OP_I32_DIV_S = 0x6D
_OP_I32_SHL = 0x74
_F64_OPS = {"Add": 0xA0, "Sub": 0xA1, "Mul": 0xA2, "Div": 0xA3}
_F32_OPS = {"Add": 0x92, "Sub": 0x93, "Mul": 0x94, "Div": 0x95}
_F32_COMPARISONS = {"Lt": 0x5D, "Le": 0x5F, "Gt": 0x5E, "Ge": 0x60, "Eq": 0x5B, "Ne": 0x5C}
_I32_OPS = {"Add": 0x6A, "Sub": 0x6B, "Mul": 0x6C, "Div": 0x6D}
_I32_COMPARISONS = {
    "Lt": 0x48, "Le": 0x4C, "Gt": 0x4A, "Ge": 0x4E, "Eq": 0x46, "Ne": 0x47,
}
# Python int is i64 here (Lane X2), so int arithmetic proven inside int64
# lowers; unproven arithmetic still refuses by name (no checked spelling).
_INT64_EXACT = (-(1 << 63), (1 << 63) - 1)
_I64_OPS = {"Add": 0x7C, "Sub": 0x7D, "Mul": 0x7E}
_I64_COMPARISONS = {
    "Lt": 0x53, "Le": 0x57, "Gt": 0x55, "Ge": 0x59, "Eq": 0x51, "Ne": 0x52,
}
_F64_COMPARISONS = {
    "Lt": 0x63, "Le": 0x65, "Gt": 0x64, "Ge": 0x66, "Eq": 0x61, "Ne": 0x62,
}
_OP_I32_NE = 0x47
_OP_I64_NE = 0x52
_OP_F64_NE = 0x62
_OP_I64_SUB = 0x7D
_OP_I64_STORE = 0x37
# Python int is int64 (rule 7; ``int`` is its provisional spelling); bool
# and addresses are i32.
_I64_DTYPES = frozenset({"int", "int64", "i64", "long"})
_I32_DTYPES = frozenset({"int32", "i32", "bool", "i1"})
_F64_DTYPES = frozenset({"float", "float64", "f64", "double"})
_F32_DTYPES = frozenset({"float32", "f32", "single"})
# The C lane's implicit conversions: (double) of an integer and bool widened
# unsigned.  Narrowings are never implicit: a Cast truncates a double the
# way C does (i64.trunc_f64_s traps where C's is undefined), and only an
# address index is wrapped to i32.
_CONVERSIONS = {
    ("i64", "f64"): (0xB9,),  # f64.convert_i64_s
    ("i32", "f64"): (0xB7,),  # f64.convert_i32_s
    ("i32", "i64"): (0xAD,),  # i64.extend_i32_u
    ("f32", "f64"): (0xBB,),  # f64.promote_f32 (fpext)
}
_CAST_CONVERSIONS = {
    **_CONVERSIONS, ("f64", "i64"): (0xB0,),
    ("f32", "i32"): (0xA8,),  # explicit GLSL int(f32), truncate toward zero
    ("i32", "f32"): (0xB2,),  # explicit signed int32 -> float32
    ("f64", "f32"): (0xB6,),  # f32.demote_f64 (fptrunc)
}
_INDEX_CONVERSIONS = {**_CONVERSIONS, ("i64", "i32"): (0xA7,)}


@dataclass(frozen=True, slots=True)
class WasmCoreArtifact:
    """A whole repository-call module as one wasm module.

    ``parameters`` is the exported entry's call ABI in formal order:
    ``("buffer", id)`` for an i32 byte-offset parameter, ``("i32", id)`` /
    ``("i64", id)`` / ``("f64", id)`` for scalars passed by value.
    ``buffer_order`` lists the buffer-parameter value ids, which is what a
    feed dictionary keys.  Every other reachable SSA function is its own
    wasm function inside the module (see ``emit_ssa_module_to_wasm_core``).
    """

    name: str
    binary: bytes
    parameters: tuple[tuple[str, int], ...]
    shortfalls: tuple[WasmEmissionShortfall, ...]
    output_order: tuple[int, ...] = ()
    precision_sections: bool = False
    #: ``(value_id, numpy dtype)`` of each output buffer cell: an i64 result
    #: (Python ``int`` or ``bool``) is stored as int64, everything else f64.
    output_dtypes: tuple[tuple[int, str], ...] = ()
    output_elements: tuple[tuple[int, int], ...] = ()
    # Static aggregate cells emitted by the direct core occupy this prefix.
    # The existing host buffer layout starts after it.
    reserved_bytes: int = 8
    #: Host functions the module imports: ``(module, field, arity)``, all
    #: ``f64 -> f64`` by arity.  The same libm names the LLVM and C lanes
    #: declare (``pow``, ``exp``, ``tanh`` ...); a host binds them from
    #: ``Math``.  Empty when the program reaches none.
    imports: tuple[tuple[str, str, int], ...] = ()
    #: ``(name, parameter kinds, result kinds)`` of every further function,
    #: in function-index order after the entry.  Diagnostics and shells.
    functions: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = ()
    #: Per buffer parameter (``buffer_order`` order): the host dtype and the
    #: declared shape, the same two tables the C and LLVM artifacts carry so
    #: ``NativeSystem.layout`` reads any lane alike.
    buffer_dtypes: tuple[str, ...] = ()
    buffer_shapes: tuple[tuple[int, ...], ...] = ()
    #: ``emission_concordance.ArtifactEmission``: the book and the
    #: MODULE_TEXT / BUFFER_ORDER cells this emission posted (None without an
    #: attached book).  ``NativeSystem.slots`` reads the ProgramABI slot rows
    #: posted beside them under ``emission.backend``.
    emission: Any = field(default=None, repr=False, compare=False)
    #: The calling contract (``compiled_program_api.CompiledProgramAPI``)
    #: described from the same root ``Function`` this emission lowered; None
    #: when the emission has shortfalls.
    api: Any = field(default=None, repr=False, compare=False)

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def prepare_execution(self, feeds: Mapping[int, Any], *, shapes=None):
        """Allocate the public buffers from real feed values, ready to run:
        the same name and contract as ``CModuleArtifact.prepare_execution``,
        so a host (``NativeSystem.prepare``) drives this lane unchanged."""

        return prepare_wasm_core_execution(self, feeds)

    @property
    def buffer_order(self) -> tuple[int, ...]:
        return tuple(
            value_id for kind, value_id in self.parameters
            if kind == "buffer"
        )


#: Scalar externs the repository's imported LLVM kernels call, and the one
#: Wasm instruction that spells each exactly.
# The module lane's unary math, spelled as the LLVM lane spells it
# (ssa_llvm_backend ``_UNARY`` / the ``unary_double`` scalar table): an f64
# operand through the f64 instruction.  ``math.floor`` of a float lowers to
# ``floor`` with an integer result (Python's floor returns int): the f64
# instruction, then the exact truncation the cast lane uses (the value is
# already integral, so truncation is exact).  The floor of an integer is the
# integer itself, as the LLVM lane spells it.
_WASM_UNARY_OPS = {
    "Sqrt": 0x9F, "sqrt": 0x9F, "Floor": 0x9C, "floor": 0x9C,
    "Ceil": 0x9B, "ceil": 0x9B, "Trunc": 0x9D, "trunc": 0x9D,
}
_WASM_INTEGRAL_UNARY = {"floor", "ceil", "trunc"}
_WASM_UNARY_INTRINSICS = {
    "llvm.fabs.f64": 0x99, "llvm.ceil.f64": 0x9B, "llvm.floor.f64": 0x9C,
    "llvm.trunc.f64": 0x9D, "llvm.sqrt.f64": 0x9F,
    # rint in the default rounding mode is round-half-to-even: f64.nearest
    # exactly (tensor ``round`` lowers to llvm.rint since 2026-11-05).
    "llvm.rint.f64": 0x9E, "llvm.roundeven.f64": 0x9E,
}
#: libm functions WebAssembly has no instruction for.  They are imported
#: from the host under module ``math`` with this arity, the same way the
#: LLVM lane declares them and links libm: the host supplies ``Math.pow``
#: and friends.  Parity note for acceptance: a host's ``Math.exp`` is not
#: bit-identical to MSVC's libm in the last ulp; the Meowdoku law reaches
#: none of these at run time (they sit in dead opcode arms of
#: ``unary_double`` / ``binary_value``), so its parity stays exact.
_MATH_IMPORTS = {
    "pow": 2, "exp": 1, "log": 1, "sin": 1, "cos": 1, "tan": 1,
    "asin": 1, "acos": 1, "atan": 1, "atan2": 2, "sinh": 1, "cosh": 1, "tanh": 1,
    "asinh": 1, "acosh": 1, "atanh": 1,
}
#: SSA binary ops with no Wasm instruction that the LLVM lane spells as a libm
#: call (``ssa_llvm_backend``: ``Atan2`` -> ``@atan2``): here the host import.
_SSA_MATH_IMPORT_OPS = {"Atan2": "atan2", "atan2": "atan2"}
_PTR_DTYPES = frozenset({"ptr", "pointer"})
_OP_DROP = 0x1A
_OP_LOCAL_TEE = 0x22
_OP_F64_ABS = 0x99
_OP_F64_TRUNC = 0x9D
_OP_F64_ADD = 0xA0
_OP_F64_SUB = 0xA1
_OP_F64_GE = 0x66
_OP_F64_EQ = 0x61
_OP_F64_COPYSIGN = 0xA6
_OP_I32_OR = 0x72
_OP_I32_AND_ = 0x71


def emit_ssa_module_to_wasm_core(
    module: IRModule, function_name: str, *, entry_name: str | None = None,
) -> WasmCoreArtifact:
    """Emit ``function_name`` and its call closure as one wasm module.

    Modelled on the LLVM lane's ``_emit_repository_call_module`` and the C
    module lane: EVERY reachable SSA function becomes its own wasm function,
    a ``Call`` is a ``call``, and a callee with several ``Ret`` blocks
    returns from each (``binary_value`` is a 23-arm opcode switch).  The
    internal convention is LLVM's: a formal something addresses through is
    an i32 byte offset into linear memory, a scalar travels by value, and a
    function's ``Ret`` values are its wasm results (a span result is its
    pointer).  Only the exported entry keeps the host ABI the shells speak:
    buffers and scalars in, one output buffer per returned value, no wasm
    results.  Scalar externs are exact Wasm instructions where one exists
    (``llvm.fabs.f64`` ...) and ``math`` imports otherwise.
    """

    from .ir_identities import precision_backend_shortfalls
    from .ssa_llvm_backend import (
        python_int_unchecked_refusal,
    )
    from .identity_concordance import identity_book
    from .ssa_record_return_state import posted_ssa_value_fact
    from .ir_literals import decode_llvm_scalar_literal
    from .ssa_storage_requirements import (
        is_compiler_owned_storage, module_storage_requirements,
    )
    from .wasm_binary import WasmFunction, WasmImport, uleb

    name = str(entry_name or function_name)
    root: Function = module.functions[function_name]
    book = identity_book(module)
    storage_requirements = module_storage_requirements(module)
    root_returns = tuple(
        instruction.args
        for block in root.blocks.values()
        for instruction in block.instrs
        if str(instruction.op) in {"Ret", "ret", "Return", "return"}
    )
    returned_values = root_returns[-1] if root_returns else ()
    shortfalls: list[WasmEmissionShortfall] = []
    reachable = [function_name]
    externs: set[str] = set()
    for owner_name in reachable:
        owner = module.functions.get(owner_name)
        if owner is None:
            continue
        for block in owner.blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) in _SSA_MATH_IMPORT_OPS and len(instruction.args) == 2:
                    # an SSA ``atan2`` op: spelled as the host's import
                    externs.add(_SSA_MATH_IMPORT_OPS[str(instruction.op)])
                if instruction.op in ("Call", "call"):
                    callee_name = str(instruction.attributes.get("callee") or "")
                    if callee_name in module.functions:
                        if callee_name not in reachable:
                            reachable.append(callee_name)
                    else:
                        externs.add(callee_name)
    shortfalls.extend(
        WasmEmissionShortfall(
            "precision_section",
            "backend cannot honour precision obligations "
            + repr(item["missing"]) + f" in {item['function']}",
        )
        for item in precision_backend_shortfalls(module, "wasm", reachable)
    )

    # Which values are addresses.  A root formal's posted physical span is a
    # buffer contract; a ``ptr`` formal or a non-scalar formal of any
    # function is one too; GetElementPtr/Store bases are; and pointer-ness
    # closes through Call actual/formal edges in both directions.  Identities
    # are scoped by function because several functions reuse small ids.
    pointer_ids: set[tuple[str, int]] = set()
    for formal in root.args:
        fact = posted_ssa_value_fact(root, int(formal.id), book=book)
        if fact is not None and fact.storage == "span" and int(fact.rank or 0) > 0:
            pointer_ids.add((root.name, int(formal.id)))
    for callee_name in reachable:
        function = module.functions.get(callee_name)
        if function is None:
            continue
        for formal in function.args:
            if (str(formal.dtype or "").casefold() in _PTR_DTYPES
                    or tuple(getattr(formal, "shape", ()) or ())):
                pointer_ids.add((function.name, int(formal.id)))
        for block in function.blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) == "GetElementPtr" and instruction.args:
                    pointer_ids.add((function.name, int(instruction.args[0].id)))
                elif str(instruction.op) == "Store" and len(instruction.args) >= 2:
                    pointer_ids.add((function.name, int(instruction.args[1].id)))
    changed = True
    while changed:
        changed = False
        for owner_name in reachable:
            owner = module.functions.get(owner_name)
            if owner is None:
                continue
            for block in owner.blocks.values():
                for instruction in block.instrs:
                    if instruction.op not in ("Call", "call"):
                        continue
                    callee = module.functions.get(
                        str(instruction.attributes.get("callee") or "")
                    )
                    if callee is None:
                        continue
                    for actual, formal in zip(instruction.args, callee.args):
                        identity = (owner.name, int(actual.id))
                        formal_identity = (callee.name, int(formal.id))
                        if formal_identity in pointer_ids and identity not in pointer_ids:
                            pointer_ids.add(identity)
                            changed = True
                        elif identity in pointer_ids and formal_identity not in pointer_ids:
                            pointer_ids.add(formal_identity)
                            changed = True

    # Receipt (2026-10-24, extra frontends/backends lane, Lane X2): every
    # integer dtype was an i32 here, so a Python ``int`` formal of 2**31 or
    # more wrapped on entry and took the wrong branch, silently (``a > b``
    # with a = 2**32 + 1 returned the other branch's value).  Python ``int``
    # is int64 (rule 7), so it is an i64 in Wasm; bool and addresses stay
    # i32.  Each local now carries its kind, and an operation whose operand
    # kinds the emitter cannot honour is a named shortfall, never a module
    # that fails validation at instantiation.
    def value_kind(value) -> str | None:
        dtype = str(value.dtype or "").casefold()
        if dtype in _I64_DTYPES:
            return "i64"
        if dtype in _I32_DTYPES or dtype in _PTR_DTYPES:
            return "i32"
        if dtype in _F64_DTYPES:
            return "f64"
        if dtype in _F32_DTYPES:
            return "f32"
        return None

    def is_pointer(function, value) -> bool:
        return (
            (function.name, int(value.id)) in pointer_ids
            or str(value.dtype or "").casefold() in _PTR_DTYPES
            or bool(tuple(getattr(value, "shape", ()) or ()))
        )

    # -- the exported entry's host ABI ---------------------------------------
    parameters: list[tuple[str, int]] = []
    parameter_types: list[str] = []
    compiler_frame_formals = []
    for formal in root.args:
        value_id = int(formal.id)
        if is_compiler_owned_storage(formal):
            compiler_frame_formals.append(formal)
            continue
        if (root.name, value_id) in pointer_ids:
            parameters.append(("buffer", value_id))
            parameter_types.append("i32")
            continue
        kind = value_kind(formal)
        if kind is None:
            shortfalls.append(WasmEmissionShortfall(
                "formal",
                f"%t{value_id} dtype {formal.dtype!r} has no Wasm value kind",
            ))
            kind = "f64"
        parameters.append((kind, value_id))
        parameter_types.append(kind)
    argument_ids = {int(value.id) for value in root.args}
    output_parameter_indices: dict[int, int] = {}
    output_dtypes: dict[int, str] = {}
    output_elements: dict[int, int] = {}
    for output in returned_values:
        value_id = int(output.id)
        if value_id in argument_ids and is_pointer(root, output):
            continue
        output_parameter_indices[value_id] = len(parameters)
        published = posted_ssa_value_fact(root, value_id, book=book)
        physical = "" if published is None else str(published.physical_dtype or "").casefold()
        output_dtypes[value_id] = (
            "float64" if physical in _F64_DTYPES
            else "int64" if physical in _I64_DTYPES
            else "float64" if value_kind(output) in (None, "f32", "f64") else "int64"
        )
        if tuple(output.shape or ()) and all(
            type(axis) is int and axis >= 0 for axis in output.shape
        ):
            from math import prod
            output_elements[value_id] = prod(output.shape)
        parameters.append(("buffer", value_id))
        parameter_types.append("i32")

    # -- internal signatures and the function index space --------------------
    # Imports first, then the entry, then every other reachable function in
    # closure order (``WasmFunction`` documents the rule).
    import_names = sorted(symbol for symbol in externs if symbol in _MATH_IMPORTS)
    import_index = {symbol: index for index, symbol in enumerate(import_names)}
    others = [fname for fname in reachable if fname != function_name]
    function_index = {function_name: len(import_names)}
    for position, fname in enumerate(others):
        function_index[fname] = len(import_names) + 1 + position

    def ret_kinds(function, values) -> tuple[str, ...]:
        kinds = []
        for value in values:
            if is_pointer(function, value):
                kinds.append("i32")
                continue
            kind = value_kind(value)
            if kind is None:
                shortfalls.append(WasmEmissionShortfall(
                    "Ret", f"{function.name}: %t{value.id} dtype "
                    f"{value.dtype!r} has no Wasm result kind",
                ))
                kind = "f64"
            kinds.append(kind)
        return tuple(kinds)

    signatures: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for fname in others:
        function = module.functions[fname]
        param_kinds = []
        for formal in function.args:
            if is_pointer(function, formal):
                param_kinds.append("i32")
                continue
            kind = value_kind(formal)
            if kind is None:
                shortfalls.append(WasmEmissionShortfall(
                    "formal", f"{fname}: %t{formal.id} dtype "
                    f"{formal.dtype!r} has no Wasm value kind",
                ))
                kind = "f64"
            param_kinds.append(kind)
        rets = [
            instruction.args
            for block in function.blocks.values()
            for instruction in block.instrs
            if str(instruction.op) in {"Ret", "ret", "Return", "return"}
        ]
        result_kinds = ret_kinds(function, rets[0]) if rets else ()
        for later in rets[1:]:
            if ret_kinds(function, later) != result_kinds:
                shortfalls.append(WasmEmissionShortfall(
                    "Ret", f"{fname}: Ret blocks disagree on result kinds",
                ))
                break
        signatures[fname] = (tuple(param_kinds), result_kinds)

    # -- shared emission state -----------------------------------------------
    builder = CodeBuilder("f64", parameter_count=len(parameters))
    precision_present = False
    aggregate_slots: dict[tuple[str, int], tuple[int, str, int]] = {}
    aggregate_end = 8

    class _Env:
        """Value id -> local index for ONE function body."""

        def __init__(self, bindings, function):
            self.slots = dict(bindings)
            self.function = function
            self.kinds = {}
            # A repository Call may return ``ssa.aggregate``.  The caller's
            # GetElementPtr carries the exact ``source_output_id`` (or the
            # ``aggregate_index`` position) for each member, so retain the
            # declared identities rather than reconstructing them.
            self.aggregate_members = {}
            self.aggregate_order = {}
            self.direct_pointer_values = {}
            self.aggregate_element_kind = {}
            self.pointer_element_kind = {}
            self.sequence_storage_address = {}

    def slot_for(env, value, kind: str):
        value_id = int(value.id)
        held = env.slots.get(value_id)
        if held is None:
            held = builder.declare_local(kind)
            env.slots[value_id] = held
            env.kinds[value_id] = kind
        elif env.kinds.get(value_id) != kind:
            shortfalls.append(WasmEmissionShortfall(
                "local",
                f"%t{value_id} is written as {kind} but its local is "
                f"{env.kinds.get(value_id)}",
            ))
        return held

    def push(env, value) -> str | None:
        """Push ``value``'s local and return its kind (None: unavailable)."""

        held = env.slots.get(int(value.id))
        if held is None:
            shortfalls.append(WasmEmissionShortfall(
                "operand", f"{env.function.name}: %t{value.id} is unavailable"
            ))
            return None
        builder.local_get(held)
        return env.kinds.get(int(value.id))

    def push_as(
        env, value, kind: str, operation: str, conversions=_CONVERSIONS,
    ) -> bool:
        """Push ``value`` converted to ``kind`` by the C lane's conversions."""

        held = push(env, value)
        if held is None:
            return False
        conversion = conversions.get((held, kind))
        if held != kind and conversion is None:
            shortfalls.append(WasmEmissionShortfall(
                operation,
                f"%t{value.id} is {held}; no exact conversion to {kind}",
            ))
            return False
        if held != kind:
            builder.raw(*conversion)
        return True

    # Studio batch inputs are authored float32 spans. Read their posted physical
    # dtype rather than treating every tensor as f64: both address stride and
    # the host packing contract must remain four bytes for these scene columns.
    def physical_kind(env, value):
        fact = posted_ssa_value_fact(env.function, int(value.id), book=book)
        if fact is None:
            return None
        spelling = str(fact.physical_dtype or "").casefold()
        if spelling in _F32_DTYPES:
            return "f32"
        if spelling in _F64_DTYPES:
            return "f64"
        if spelling in _I64_DTYPES:
            return "i64"
        if spelling in _I32_DTYPES:
            return "i32"
        return None

    def ensure_call_output_arena(env, instruction) -> None:
        """Materialize a booked temporary output argument in this call frame."""

        nonlocal aggregate_end
        output_position = instruction.attributes.get("ssa_output_argument")
        if type(output_position) is not int or not 0 <= output_position < len(instruction.args):
            return
        output = instruction.args[output_position]
        value_id = int(output.id)
        if value_id in env.slots:
            return
        descriptor = (
            (module.tensor_tables or {}).get(env.function.name)
        )
        descriptor = None if descriptor is None else descriptor.tensors.get(value_id)
        requirement = storage_requirements.get(env.function.name, {}).get(value_id)
        # The booked storage must be static and exact.  A ``temporary`` is
        # the function's own scratch; an ``output`` is the function's own
        # result span, which the LLVM lane receives as a result pointer and
        # this lane allocates in the function's static frame and returns.
        if (requirement is None or requirement.dynamic
                or requirement.element_count is None
                or tuple(output.shape or ()) != tuple(requirement.shape)
                or (descriptor is not None and (
                    descriptor.storage not in {"temporary", "output"}
                    or descriptor.metadata_state != "static"
                    or int(descriptor.data_value_id) != value_id
                    or tuple(descriptor.shape) != tuple(requirement.shape)
                    or str(descriptor.dtype).casefold()
                        != str(requirement.dtype).casefold()))):
            shortfalls.append(WasmEmissionShortfall(
                "Call", f"{env.function.name}: output %t{value_id} shaped "
                f"{tuple(output.shape or ())} lacks a booked static arena "
                f"(requirement {None if requirement is None else (requirement.element_count, tuple(requirement.shape))}, "
                f"descriptor {None if descriptor is None else descriptor.storage!r})",
            ))
            return
        spelling = str(requirement.dtype).casefold()
        element_kind = (
            "f64" if spelling in _F64_DTYPES
            else "i64" if spelling in _I64_DTYPES
            else "i32" if spelling in _I32_DTYPES else None
        )
        width = {"i32": 4, "i64": 8, "f64": 8}.get(element_kind)
        if width is None or (
                descriptor is not None
                and int(descriptor.byte_size or -1) != requirement.element_count * width):
            shortfalls.append(WasmEmissionShortfall(
                "Call", f"output %t{value_id} lacks exact element storage width",
            ))
            return
        start = (aggregate_end + width - 1) // width * width
        aggregate_end = start + max(1, requirement.element_count) * width
        builder.i32_const(start)
        builder.local_set(slot_for(env, output, "i32"))
        env.pointer_element_kind[value_id] = element_kind

    def emit_extern_call(env, instruction) -> bool:
        """A call to a function outside the module: an exact Wasm spelling
        of an LLVM scalar intrinsic, or a ``math`` import.  False when the
        callee is a module function."""

        symbol = str(instruction.attributes.get("callee") or "")
        if symbol in module.functions:
            return False
        args = instruction.args
        result = instruction.res
        opcode = _WASM_UNARY_INTRINSICS.get(symbol)
        if opcode is not None:
            if len(args) != 1 or result is None or value_kind(result) != "f64":
                shortfalls.append(WasmEmissionShortfall(
                    "Call", f"{symbol} lacks its scalar f64 signature",
                ))
                return True
            if push_as(env, args[0], "f64", "Call"):
                builder.raw(opcode)
                builder.local_set(slot_for(env, result, "f64"))
            return True
        if symbol.startswith("llvm.memcpy.") or symbol.startswith("llvm.memmove."):
            # Program: any repository kernel that copies a span (``index_set``,
            # ``slice_copy``: ``memcpy(out, in, n * sizeof(double))``).  The
            # intrinsic's (dst, src, byte length, volatile) is exactly the
            # bulk-memory ``memory.copy`` (dst, src, n) on the one linear
            # memory; non-overlapping memcpy and memmove coincide there.
            if len(args) != 4 or result is not None:
                shortfalls.append(WasmEmissionShortfall(
                    "Call", f"{symbol} lacks its (dst, src, length, volatile) signature",
                ))
                return True
            if (push_as(env, args[0], "i32", "Call")
                    and push_as(env, args[1], "i32", "Call")
                    and push_as(env, args[2], "i32", "Call", _INDEX_CONVERSIONS)):
                builder.raw(0xFC, 0x0A, 0x00, 0x00)  # memory.copy 0 0
            return True
        if symbol == "llvm.round.f64":
            # Round half away from zero, which f64.nearest (half to even)
            # is not: r = trunc(x); |x - r| >= 0.5 ? r + copysign(1, x) : r.
            if len(args) != 1 or result is None or value_kind(result) != "f64":
                shortfalls.append(WasmEmissionShortfall(
                    "Call", f"{symbol} lacks its scalar f64 signature",
                ))
                return True
            if not push_as(env, args[0], "f64", "Call"):
                return True
            x = builder.declare_local("f64")
            r = builder.declare_local("f64")
            builder.raw(_OP_LOCAL_TEE, *uleb(x)).raw(_OP_F64_TRUNC).local_set(r)
            builder.local_get(r).value_const(1.0).local_get(x)
            builder.raw(_OP_F64_COPYSIGN).raw(_OP_F64_ADD)       # val1
            builder.local_get(r)                                  # val2
            builder.local_get(x).local_get(r).raw(_OP_F64_SUB)
            builder.raw(_OP_F64_ABS).value_const(0.5).raw(_OP_F64_GE)
            builder.select()
            builder.local_set(slot_for(env, result, "f64"))
            return True
        if symbol in {"llvm.fcmp.ord", "llvm.fcmp.uno"}:
            # ord: neither is NaN; uno: either is.  x == x is false for NaN.
            if len(args) != 2 or result is None:
                shortfalls.append(WasmEmissionShortfall(
                    "Call", f"{symbol} lacks its two f64 operands",
                ))
                return True
            # push a; push a; eq; push b; push b; eq; and  (xor 1 for uno).
            if not (push_as(env, args[0], "f64", "Call")
                    and push_as(env, args[0], "f64", "Call")):
                return True
            builder.raw(_OP_F64_EQ)
            if not (push_as(env, args[1], "f64", "Call")
                    and push_as(env, args[1], "f64", "Call")):
                return True
            builder.raw(_OP_F64_EQ).raw(_OP_I32_AND_)
            if symbol == "llvm.fcmp.uno":
                builder.i32_const(1).raw(0x73)  # xor 1
            builder.local_set(slot_for(env, result, "i32"))
            return True
        arity = _MATH_IMPORTS.get(symbol)
        if arity is not None:
            if len(args) != arity or result is None or value_kind(result) != "f64":
                shortfalls.append(WasmEmissionShortfall(
                    "Call", f"{symbol} lacks its f64 x{arity} -> f64 signature",
                ))
                return True
            for argument in args:
                if not push_as(env, argument, "f64", "Call"):
                    return True
            builder.call(import_index[symbol])
            builder.local_set(slot_for(env, result, "f64"))
            return True
        shortfalls.append(WasmEmissionShortfall(
            "Call", f"callee {symbol!r} is absent",
        ))
        return True

    def emit_straight_instruction(env, instruction) -> None:
        """One non-control instruction into the current position."""

        nonlocal precision_present, aggregate_end
        operation = str(instruction.op)
        unchecked = python_int_unchecked_refusal(
            instruction, "core WASM i64", _INT64_EXACT,
        )
        if unchecked is not None:
            shortfalls.append(WasmEmissionShortfall(operation, unchecked))
            return
        if instruction.attributes.get("precision_section"):
            precision_present = True
        if operation == "string_token" and instruction.res is not None:
            # The source string token is the same signed 64-bit scalar
            # emitted by C and LLVM. Its separate glyph-storage operations
            # remain executable and must acquire their own memory proof.
            token = instruction.attributes.get("token")
            if type(token) is not int:
                shortfalls.append(WasmEmissionShortfall(
                    operation, "string token has no exact int64 payload",
                ))
                return
            builder.i64_const(token)
            builder.local_set(slot_for(env, instruction.res, "i64"))
            sequence_table = module.sequence_tables.get(env.function.name)
            descriptor = (None if sequence_table is None else
                          sequence_table.sequences.get(int(instruction.res.id)))
            glyphs = tuple(
                candidate for block in env.function.blocks.values()
                for candidate in block.instrs
                if candidate.op == "GetElementPtr"
                and candidate.attributes.get("binding") == "ssa_sequence_literal_glyph"
                and candidate.args
                and int(candidate.args[0].id) == int(instruction.res.id)
            )
            if glyphs:
                literal = instruction.attributes.get("text")
                indices = []
                for glyph in glyphs:
                    index_id = int(glyph.args[1].id) if len(glyph.args) == 2 else None
                    matches = tuple(
                        item for block in env.function.blocks.values()
                        for item in block.instrs
                        if item.op == "Const" and item.res is not None
                        and int(item.res.id) == index_id
                    )
                    indices.append(matches[0].attributes.get("value")
                                   if len(matches) == 1 else None)
                if (descriptor is None or not isinstance(literal, str)
                        or tuple(descriptor.column_value_ids) != (int(instruction.res.id),)
                        or tuple(descriptor.column_dtypes) != ("int64",)
                        or str(descriptor.capacity_policy.value) != "fixed"
                        or sorted(indices) != list(range(len(literal)))):
                    shortfalls.append(WasmEmissionShortfall(
                        operation, "glyph sequence lacks its fixed booked storage",
                    ))
                    return
                key = (env.function.name, int(instruction.res.id))
                stored = aggregate_slots.get(key)
                if stored is None:
                    start = (aggregate_end + 7) // 8 * 8
                    stored = (start, "i64", len(literal))
                    aggregate_slots[key] = stored
                    aggregate_end = start + max(1, len(literal)) * 8
                env.sequence_storage_address[int(instruction.res.id)] = stored[0]
            return
        if operation in {"NoneValue", "nonevalue"}:
            shortfalls.append(WasmEmissionShortfall(
                operation,
                "core Wasm has no untyped absence value; lower an optional "
                "to explicit presence and payload values before emission",
            ))
            return
        if operation == "Const":
            if "values" in instruction.attributes and instruction.res is not None:
                values = instruction.attributes["values"]
                kind = value_kind(instruction.res)
                if (not isinstance(values, (tuple, list)) or not values
                        or kind not in {"i32", "i64", "f64"}
                        or len(tuple(instruction.res.shape or ())) != 1
                        or int(instruction.res.shape[0]) != len(values)):
                    shortfalls.append(WasmEmissionShortfall(
                        operation, "typed aggregate Const lacks a flat SSA vector",
                    ))
                    return
                width = 4 if kind == "i32" else 8
                if any(type(item) not in (int, float) or
                       (kind != "f64" and
                        (type(item) is not int or item < -(1 << (8 * width - 1))
                         or item >= (1 << (8 * width - 1))))
                       for item in values):
                    shortfalls.append(WasmEmissionShortfall(
                        operation, "typed aggregate Const has no exact element layout",
                    ))
                    return
                key = (env.function.name, int(instruction.res.id))
                stored = aggregate_slots.get(key)
                if stored is None:
                    start = (aggregate_end + width - 1) // width * width
                    stored = (start, kind, len(values))
                    aggregate_slots[key] = stored
                    aggregate_end = start + width * len(values)
                elif stored[1:] != (kind, len(values)):
                    shortfalls.append(WasmEmissionShortfall(
                        operation, "aggregate Const changed its booked storage layout",
                    ))
                    return
                start = stored[0]
                for index, item in enumerate(values):
                    builder.i32_const(start + index * width)
                    if kind == "i32":
                        builder.i32_const(int(item)).i32_store_width(32)
                    elif kind == "i64":
                        builder.i64_const(int(item)).i64_store()
                    else:
                        builder.value_const(float(item)).store()
                builder.i32_const(start)
                builder.local_set(slot_for(env, instruction.res, "i32"))
                env.aggregate_element_kind[int(instruction.res.id)] = kind
                return
            if "llvm_literal" in instruction.attributes:
                try:
                    held = decode_llvm_scalar_literal(
                        instruction.attributes["llvm_literal"]
                    )
                except ValueError as error:
                    shortfalls.append(WasmEmissionShortfall(operation, str(error)))
                    return
            else:
                held = instruction.attributes.get(
                    "constant", instruction.attributes.get("value")
                )
            if not isinstance(held, (int, float)):
                shortfalls.append(WasmEmissionShortfall(
                    operation,
                    f"constant {type(held).__name__} has no scalar Wasm "
                    "value; lower the aggregate before emission",
                ))
                return
            kind = (
                value_kind(instruction.res) if instruction.res is not None
                else None
            ) or (
                "i32" if isinstance(held, bool)
                else "i64" if isinstance(held, int) else "f64"
            )
            if kind == "i64":
                builder.i64_const(int(held))
            elif kind == "i32":
                builder.i32_const(int(held))
            elif kind == "f32":
                builder.raw(0x43, *_struct.pack("<f", float(held)))
            else:
                builder.value_const(float(held))
            builder.local_set(slot_for(env, instruction.res, kind))
            return
        if operation == "GetElementPtr" and len(instruction.args) >= 2:
            aggregate = env.aggregate_members.get(int(instruction.args[0].id))
            if aggregate is not None:
                source_output_id = instruction.attributes.get("source_output_id")
                if source_output_id is None:
                    position = instruction.attributes.get("aggregate_index")
                    order = env.aggregate_order.get(int(instruction.args[0].id), ())
                    if type(position) is int and 0 <= position < len(order):
                        source_output_id = order[position]
                held = None if source_output_id is None else aggregate.get(int(source_output_id))
                if held is None:
                    shortfalls.append(WasmEmissionShortfall(
                        operation,
                        "aggregate has no declared source output "
                        f"%t{source_output_id}",
                    ))
                    return
                env.direct_pointer_values[int(instruction.res.id)] = held
                return
            if instruction.attributes.get("binding") == "ssa_sequence_literal_glyph":
                address = env.sequence_storage_address.get(int(instruction.args[0].id))
                if address is None or len(instruction.args) != 2:
                    shortfalls.append(WasmEmissionShortfall(
                        operation, "glyph address lacks booked local storage",
                    ))
                    return
                builder.i32_const(address)
                if not push_as(env, instruction.args[1], "i32", operation,
                               _INDEX_CONVERSIONS):
                    return
                builder.i32_const(3).raw(_OP_I32_SHL).raw(OP_I32_ADD)
                builder.local_set(slot_for(env, instruction.res, "i32"))
                env.pointer_element_kind[int(instruction.res.id)] = "i64"
                return
            if len(instruction.args) != 2:
                shortfalls.append(WasmEmissionShortfall(
                    operation, "address has no single flat index",
                ))
                return
            base_id = int(instruction.args[0].id)
            element_kind = (
                env.aggregate_element_kind.get(base_id)
                or env.pointer_element_kind.get(base_id)
                or physical_kind(env, instruction.args[0])
            )
            if element_kind not in {"i32", "i64", "f32", "f64"}:
                shortfalls.append(WasmEmissionShortfall(
                    operation, f"{env.function.name}: %t{base_id} has no proved element layout",
                ))
                return
            # Address scale follows the source buffer's booked physical
            # element width. Shape vectors are int32; tensor cells are f64.
            if not (push_as(env, instruction.args[0], "i32", operation)
                    and push_as(env, instruction.args[1], "i32", operation,
                                _INDEX_CONVERSIONS)):
                return
            builder.i32_const(2 if element_kind in {"i32", "f32"} else 3)
            builder.raw(_OP_I32_SHL).raw(OP_I32_ADD)
            builder.local_set(slot_for(env, instruction.res, "i32"))
            env.pointer_element_kind[int(instruction.res.id)] = element_kind
            return
        if operation == "Load" and instruction.args:
            direct = env.direct_pointer_values.get(int(instruction.args[0].id))
            if direct is not None:
                value_id, held_kind, element_kind = direct
                env.slots[int(instruction.res.id)] = value_id
                env.kinds[int(instruction.res.id)] = held_kind
                if held_kind == "i32" and element_kind is not None:
                    env.pointer_element_kind[int(instruction.res.id)] = element_kind
                return
            address_id = int(instruction.args[0].id)
            element_kind = env.pointer_element_kind.get(address_id)
            result_kind = physical_kind(env, instruction.res)
            result_fact = posted_ssa_value_fact(
                env.function, int(instruction.res.id), book=book,
            )
            imported_typed_load = (
                instruction.attributes.get("llvm_opcode") == "load"
                and value_kind(instruction.res) == element_kind
                and result_fact is not None
                and str(result_fact.dtype).casefold()
                    == str(instruction.res.dtype).casefold()
                and result_fact.storage is None
            )
            if (element_kind not in {"i32", "i64", "f32", "f64"}
                    or not (result_kind == element_kind or imported_typed_load)
                    or result_fact is None or result_fact.storage == "span"):
                shortfalls.append(WasmEmissionShortfall(
                    operation,
                    f"{env.function.name}: %t{instruction.res.id} lacks a "
                    "scalar load matching its proved pointer element",
                ))
                return
            if not push_as(env, instruction.args[0], "i32", operation):
                return
            if element_kind == "f32":
                builder.raw(0x2A, 2, 0)  # f32.load, four-byte natural alignment
            elif element_kind == "i32":
                builder.i32_load()
            elif element_kind == "i64":
                builder.i64_load()
            else:
                builder.load()
            builder.local_set(slot_for(env, instruction.res, element_kind))
            return
        if operation == "Store" and len(instruction.args) >= 2:
            element_kind = env.pointer_element_kind.get(int(instruction.args[1].id))
            if element_kind not in {"i32", "i64", "f32", "f64"}:
                shortfalls.append(WasmEmissionShortfall(
                    operation, "store target has no proved element layout",
                ))
                return
            if not (push_as(env, instruction.args[1], "i32", operation)
                    and push_as(env, instruction.args[0], element_kind, operation)):
                return
            if element_kind == "f32":
                builder.raw(0x38, 2, 0)  # f32.store, four-byte natural alignment
            elif element_kind == "i32":
                builder.i32_store_width(32)
            elif element_kind == "i64":
                builder.i64_store()
            else:
                builder.store()
            return
        if operation == "Neg" and instruction.args:
            source = instruction.args[0]
            kind = env.kinds.get(int(source.id))
            if kind in ("i32", "i64"):
                if kind == "i64":
                    builder.i64_const(0)
                else:
                    builder.i32_const(0)
                if push(env, source) is None:
                    return
                builder.raw(_OP_I64_SUB if kind == "i64" else _OP_I32_SUB)
            else:
                if push(env, source) is None:
                    return
                builder.raw(0x8C if kind == "f32" else 0x9A)  # f32/f64.neg
            builder.local_set(slot_for(env, instruction.res, kind or "f64"))
            return
        if operation == "Abs" and instruction.args and env.kinds.get(
                int(instruction.args[0].id)) not in ("i32", "i64"):
            # Float magnitude is one instruction (the integer Abs has no
            # spelling here yet and still refuses below, by name).
            if push(env, instruction.args[0]) is None:
                return
            kind = env.kinds.get(int(instruction.args[0].id))
            builder.raw(0x8B if kind == "f32" else _OP_F64_ABS)
            builder.local_set(slot_for(env, instruction.res, kind))
            return
        if operation in {"Cast", "CastLike"} and instruction.args:
            result_shape = tuple(getattr(instruction.res, "shape", ()) or ())
            if result_shape:
                # A SHAPED cast is the elementwise cast of a span into a
                # fresh array of the result's physical element type -- the
                # loop the C module lane and the LLVM lane both emit.  The
                # region-boundary conversions the compiler inserts
                # (``physical_region_input_conversion``) are this shape.
                source = instruction.args[0]
                source_shape = tuple(getattr(source, "shape", ()) or ())
                if (source_shape != result_shape or any(
                        type(extent) is not int or extent < 0
                        for extent in result_shape)):
                    shortfalls.append(WasmEmissionShortfall(
                        operation, f"{env.function.name}: array cast requires "
                        f"a matching static shape ({source_shape} -> {result_shape})",
                    ))
                    return
                source_kind = (
                    env.pointer_element_kind.get(int(source.id))
                    or physical_kind(env, source)
                )
                target_kind = physical_kind(env, instruction.res) or value_kind(instruction.res)
                if (source_kind not in {"i32", "i64", "f64"}
                        or target_kind not in {"i32", "i64", "f64"}
                        or env.kinds.get(int(source.id)) != "i32"):
                    shortfalls.append(WasmEmissionShortfall(
                        operation, f"{env.function.name}: array cast of "
                        f"%t{source.id} lacks proved span element kinds "
                        f"({source_kind} -> {target_kind})",
                    ))
                    return
                from math import prod
                count = prod(result_shape)
                width = 4 if target_kind == "i32" else 8
                start = (aggregate_end + width - 1) // width * width
                aggregate_end = start + max(1, count) * width
                destination = slot_for(env, instruction.res, "i32")
                builder.i32_const(start).local_set(destination)
                env.pointer_element_kind[int(instruction.res.id)] = target_kind
                target_is_bool = str(instruction.res.dtype).casefold() in {"bool", "i1"}
                source_is_bool = str(source.dtype).casefold() in {"bool", "i1"}
                counter = builder.declare_local("i32")
                builder.i32_const(0).local_set(counter)
                builder.block().loop()
                builder.local_get(counter).i32_const(count)
                builder.raw(OP_I32_GE_S).br_if(1)
                builder.local_get(destination)
                builder.local_get(counter).i32_const(2 if width == 4 else 3)
                builder.raw(_OP_I32_SHL).raw(OP_I32_ADD)
                builder.local_get(env.slots[int(source.id)])
                builder.local_get(counter).i32_const(2 if source_kind == "i32" else 3)
                builder.raw(_OP_I32_SHL).raw(OP_I32_ADD)
                if source_kind == "i32":
                    builder.i32_load()
                elif source_kind == "i64":
                    builder.i64_load()
                else:
                    builder.load()
                if target_is_bool:
                    if source_kind == "i64":
                        builder.i64_const(0).raw(_OP_I64_NE)
                    elif source_kind == "f64":
                        builder.value_const(0.0).raw(_OP_F64_NE)
                    else:
                        builder.i32_const(0).raw(_OP_I32_NE)
                elif source_kind != target_kind:
                    builder.raw(*{
                        ("i64", "f64"): (0xB9,), ("i32", "f64"): (0xB7,),
                        ("f64", "i64"): (0xB0,), ("f64", "i32"): (0xAA,),
                        ("i32", "i64"): (0xAD,) if source_is_bool else (0xAC,),
                        ("i64", "i32"): (0xA7,),
                    }[(source_kind, target_kind)])
                if target_kind == "i32":
                    builder.i32_store_width(32)
                elif target_kind == "i64":
                    builder.i64_store()
                else:
                    builder.store()
                builder.local_get(counter).i32_const(1)
                builder.raw(OP_I32_ADD).local_set(counter)
                builder.br(0).end().end()
                return
            kind = value_kind(instruction.res)
            if kind is None:
                shortfalls.append(WasmEmissionShortfall(
                    operation,
                    f"%t{instruction.res.id} dtype {instruction.res.dtype!r} "
                    "has no Wasm value kind",
                ))
                return
            if kind == "i32" and str(instruction.res.dtype).casefold() in {
                "bool", "i1",
            }:
                # bool(x) is x != 0, not a narrowing.
                held = push(env, instruction.args[0])
                if held is None:
                    return
                if held == "i64":
                    builder.i64_const(0).raw(_OP_I64_NE)
                elif held == "f64":
                    builder.value_const(0.0).raw(_OP_F64_NE)
                elif held == "f32":
                    builder.raw(0x43, 0, 0, 0, 0, 0x5C)  # f32.const 0; f32.ne
                else:
                    builder.i32_const(0).raw(_OP_I32_NE)
            elif not push_as(
                env, instruction.args[0], kind, operation, _CAST_CONVERSIONS,
            ):
                return
            builder.local_set(slot_for(env, instruction.res, kind))
            return
        if operation == "SExt":
            if (instruction.attributes.get("llvm_opcode") != "sext"
                    or len(instruction.args) != 1 or instruction.res is None
                    or value_kind(instruction.args[0]) != "i32"
                    or value_kind(instruction.res) != "i64"):
                shortfalls.append(WasmEmissionShortfall(
                    operation, "imported signed i32 to i64 proof is absent",
                ))
                return
            if push_as(env, instruction.args[0], "i32", operation):
                builder.raw(0xAC)  # i64.extend_i32_s
                builder.local_set(slot_for(env, instruction.res, "i64"))
            return
        if operation in {"UiToFp", "SiToFp", "FpToSi"}:
            casts = {
                ("UiToFp", "i32", "f64", "uitofp"): 0xB8,
                ("SiToFp", "i64", "f64", "sitofp"): 0xB9,
                ("SiToFp", "i32", "f64", "sitofp"): 0xB7,
                ("FpToSi", "f64", "i64", "fptosi"): 0xB0,
                ("FpToSi", "f64", "i32", "fptosi"): 0xAA,
            }
            if len(instruction.args) != 1 or instruction.res is None:
                shortfalls.append(WasmEmissionShortfall(
                    operation, "imported conversion lacks one source and result",
                ))
                return
            source_kind = value_kind(instruction.args[0])
            target_kind = value_kind(instruction.res)
            opcode = casts.get((operation, source_kind, target_kind,
                                instruction.attributes.get("llvm_opcode")))
            if opcode is None:
                shortfalls.append(WasmEmissionShortfall(
                    operation, "imported numeric conversion has no exact Wasm kind",
                ))
                return
            if push_as(env, instruction.args[0], source_kind, operation):
                builder.raw(opcode)
                builder.local_set(slot_for(env, instruction.res, target_kind))
            return
        if operation == "Mod" and instruction.attributes.get("llvm_opcode") == "srem":
            if (len(instruction.args) == 2 and instruction.res is not None
                    and all(value_kind(arg) == "i32" for arg in instruction.args)
                    and value_kind(instruction.res) == "i32"
                    and all(push_as(env, arg, "i32", operation)
                            for arg in instruction.args)):
                builder.raw(0x6F)  # i32.rem_s
                builder.local_set(slot_for(env, instruction.res, "i32"))
            else:
                shortfalls.append(WasmEmissionShortfall(
                    operation, "imported signed remainder has no i32 operands",
                ))
            return
        if operation == "LNot" and len(instruction.args)==1 and value_kind(instruction.args[0])=="i32":
            if push_as(env, instruction.args[0], "i32", operation):
                builder.raw(0x45).local_set(slot_for(env,instruction.res,"i32"))
            return
        if operation == "Select":
            if (len(instruction.args) != 3 or instruction.res is None
                    or value_kind(instruction.args[0]) != "i32"
                    or value_kind(instruction.args[1]) != value_kind(instruction.args[2])
                    or value_kind(instruction.res) != value_kind(instruction.args[1])):
                shortfalls.append(WasmEmissionShortfall(
                    operation, "imported select has no exact condition/value kinds",
                ))
                return
            kind = value_kind(instruction.res)
            if (push_as(env, instruction.args[1], kind, operation)
                    and push_as(env, instruction.args[2], kind, operation)
                    and push_as(env, instruction.args[0], "i32", operation)):
                builder.select().local_set(slot_for(env, instruction.res, kind))
            return
        if operation in {"LAnd", "LOr"}:
            if len(instruction.args)==2 and all(str(a.dtype) in {"bool", "i1"} for a in instruction.args):
                if all(push_as(env,a,"i32",operation) for a in instruction.args):
                    builder.raw(0x71 if operation=="LAnd" else 0x72).local_set(slot_for(env,instruction.res,"i32"))
            else: shortfalls.append(WasmEmissionShortfall(operation,"logical operands must be boolean i32"))
            return
        if operation in {"And", "Or", "Xor", "Shl", "AShr"}:
            bitops = {
                "i32": {"And": 0x71, "Or": 0x72, "Xor": 0x73,
                        "Shl": 0x74, "AShr": 0x75},
                "i64": {"And": 0x83, "Or": 0x84, "Xor": 0x85,
                        "Shl": 0x86, "AShr": 0x87},
            }
            kind = None if instruction.res is None else value_kind(instruction.res)
            if (len(instruction.args) != 2 or kind not in bitops
                    or instruction.attributes.get("llvm_opcode")
                        != operation.casefold()
                    or any(value_kind(arg) != kind for arg in instruction.args)):
                shortfalls.append(WasmEmissionShortfall(
                    operation, "imported bit operation lacks exact integer kinds",
                ))
                return
            if all(push_as(env, arg, kind, operation) for arg in instruction.args):
                builder.raw(bitops[kind][operation])
                builder.local_set(slot_for(env, instruction.res, kind))
            return
        if operation in _F64_OPS and len(instruction.args) == 2:
            kinds = [env.kinds.get(int(a.id)) for a in instruction.args]
            kind = value_kind(instruction.res) or (
                "f64" if "f64" in kinds else kinds[0] or "f64"
            )
            if (kind not in {"f32", "f64"} and operation == "Div"
                    and not (kind == "i32"
                             and instruction.attributes.get("llvm_opcode") == "sdiv"
                             and all(value_kind(arg) == "i32"
                                     for arg in instruction.args))):
                shortfalls.append(WasmEmissionShortfall(
                    operation, f"integer-valued Div into {kind}",
                ))
                return
            if not all(
                push_as(env, argument, kind, operation)
                for argument in instruction.args
            ):
                return
            builder.raw({
                "f32": _F32_OPS, "f64": _F64_OPS, "i64": _I64_OPS, "i32": _I32_OPS,
            }[kind][operation])
            builder.local_set(slot_for(env, instruction.res, kind))
            return
        if operation in _I32_COMPARISONS and len(instruction.args) == 2:
            kinds = {env.kinds.get(int(a.id)) for a in instruction.args}
            kind = (
                "f64" if "f64" in kinds or None in kinds
                else "f32" if "f32" in kinds else "i64" if "i64" in kinds else "i32"
            )
            if not all(
                push_as(env, argument, kind, operation)
                for argument in instruction.args
            ):
                return
            builder.raw({
                "f32": _F32_COMPARISONS, "f64": _F64_COMPARISONS, "i64": _I64_COMPARISONS,
                "i32": _I32_COMPARISONS,
            }[kind][operation])
            builder.local_set(slot_for(env, instruction.res, "i32"))
            return
        if operation in {"Sqrt", "sqrt"} and len(instruction.args)==1 and value_kind(instruction.args[0])=="f32" and value_kind(instruction.res)=="f32":
            if push_as(env,instruction.args[0],"f32",operation):
                builder.raw(0x91).local_set(slot_for(env,instruction.res,"f32"))
            return
        if operation in _WASM_UNARY_OPS and len(instruction.args) == 1 and instruction.res is not None:
            source_kind = value_kind(instruction.args[0])
            target_kind = value_kind(instruction.res) or "f64"
            if source_kind in ("i32", "i64") and operation.casefold() in _WASM_INTEGRAL_UNARY:
                if push_as(env, instruction.args[0], target_kind, operation):
                    builder.local_set(slot_for(env, instruction.res, target_kind))
                return
            if target_kind not in ("f64", "i64", "i32"):
                shortfalls.append(WasmEmissionShortfall(
                    operation, f"{env.function.name}: unary {operation} into {target_kind} has no exact Wasm kind",
                ))
                return
            if not push_as(env, instruction.args[0], "f64", operation):
                return
            builder.raw(_WASM_UNARY_OPS[operation])
            if target_kind == "i64":
                builder.raw(0xB0)                 # i64.trunc_f64_s (exact: the value is integral)
            elif target_kind == "i32":
                builder.raw(0xAA)                 # i32.trunc_f64_s
            builder.local_set(slot_for(env, instruction.res, target_kind))
            return
        if operation in _SSA_MATH_IMPORT_OPS and len(instruction.args) == 2 and instruction.res is not None:
            symbol = _SSA_MATH_IMPORT_OPS[operation]
            if (value_kind(instruction.res) or "f64") != "f64" or symbol not in import_index:
                shortfalls.append(WasmEmissionShortfall(
                    operation, f"{env.function.name}: {operation} lacks its f64 x2 -> f64 import"))
                return
            if all(push_as(env, argument, "f64", operation) for argument in instruction.args):
                builder.call(import_index[symbol])
                builder.local_set(slot_for(env, instruction.res, "f64"))
            return
        if operation in TENSOR_OPERATION_SCALAR_SPELLING or operation == "equal":
            shapes = tuple(
                tuple(getattr(argument, "shape", ()) or ())
                for argument in instruction.args
            )
            shortfalls.append(WasmEmissionShortfall(
                operation,
                f"{env.function.name}: bare tensor {operation} over {shapes} "
                "is not a linked kernel; the core spells only repository "
                "kernels and scalars (the C module lane refuses this "
                "instruction by name too)",
            ))
            return
        shortfalls.append(WasmEmissionShortfall(
            operation, f"{env.function.name}: no module-lane WASM spelling"
        ))

    def emit_call(env, instruction) -> None:
        """``call`` a module function with the LLVM-style internal ABI."""

        if emit_extern_call(env, instruction):
            return
        callee_name = str(instruction.attributes.get("callee") or "")
        callee = module.functions[callee_name]
        if len(instruction.args) != len(callee.args):
            shortfalls.append(WasmEmissionShortfall(
                "Call", f"callee {callee_name!r} actual/formal arity disagrees",
            ))
            return
        ensure_call_output_arena(env, instruction)
        param_kinds, result_kinds = signatures[callee_name]
        for actual, kind in zip(instruction.args, param_kinds):
            if not push_as(env, actual, kind, "Call", _INDEX_CONVERSIONS):
                return
        builder.call(function_index[callee_name])
        result = instruction.res
        if result is None:
            for _ in result_kinds:
                builder.raw(_OP_DROP)
            return
        returned = next((
            item.args
            for block in callee.blocks.values()
            for item in block.instrs
            if str(item.op) in {"Ret", "Return"}
        ), ())
        declared = tuple(
            int(value_id)
            for value_id in instruction.attributes.get("output_ids", ())
        )
        aggregate = (
            instruction.attributes.get("result_convention") == "ssa.aggregate"
            or str(result.dtype or "").casefold() == "ssa.aggregate"
            or bool(declared)
        )
        if aggregate:
            callee_ids = tuple(
                int(value_id) for value_id in
                instruction.attributes.get("callee_output_ids", declared)
            )
            every_return = [
                tuple(int(value.id) for value in item.args)
                for block in callee.blocks.values()
                for item in block.instrs
                if str(item.op) in {"Ret", "Return"}
            ]
            if (len(declared) != len(result_kinds) or not every_return
                    or any(ids != callee_ids for ids in every_return)):
                shortfalls.append(WasmEmissionShortfall(
                    "Call",
                    f"callee {callee_name!r} aggregate result disagrees with "
                    "its declared output_ids",
                ))
                return
            members = {}
            # Results come back top of stack last: bind them in reverse.
            for position in reversed(range(len(declared))):
                kind = result_kinds[position]
                held = builder.declare_local(kind)
                builder.local_set(held)
                element = None
                if kind == "i32" and position < len(returned):
                    element = _callee_element_kind(callee, returned[position])
                members[declared[position]] = (held, kind, element)
            env.aggregate_members[int(result.id)] = members
            env.aggregate_order[int(result.id)] = declared
            return
        output_position = instruction.attributes.get("ssa_output_argument")
        if (type(output_position) is int
                and 0 <= output_position < len(instruction.args)
                and int(instruction.args[output_position].id) == int(result.id)):
            # The result IS the output argument the caller already owns.
            for _ in result_kinds:
                builder.raw(_OP_DROP)
            return
        if len(result_kinds) != 1:
            shortfalls.append(WasmEmissionShortfall(
                "Call", f"callee {callee_name!r} returns {len(result_kinds)} "
                "values into one result",
            ))
            return
        kind = result_kinds[0]
        builder.local_set(slot_for(env, result, kind))
        if kind == "i32" and returned:
            element = _callee_element_kind(callee, returned[0])
            if element is not None:
                env.pointer_element_kind[int(result.id)] = element

    def _callee_element_kind(callee, value):
        fact = posted_ssa_value_fact(callee, int(value.id), book=book)
        if fact is None:
            return None
        spelling = str(fact.physical_dtype or "").casefold()
        if spelling in _F32_DTYPES:
            return "f32"
        if spelling in _F64_DTYPES:
            return "f64"
        if spelling in _I64_DTYPES:
            return "i64"
        if spelling in _I32_DTYPES:
            return "i32"
        return None

    def phi_kind(instruction) -> str:
        return value_kind(instruction.res) or next((
            value_kind(argument) for argument in instruction.args
            if argument is not None and value_kind(argument) is not None
        ), "f64")

    def edge_assignments(env, function, source_block: str, target_block: str) -> None:
        # Every Phi on an edge reads its incoming value before any Phi on
        # that edge is written: a parallel copy.  Writing them one by one let
        # ``j = phi(.., i)`` read the ``i`` the same edge had just advanced
        # (c125 returned 10 where Python returns 9).  All incoming values go
        # on the Wasm stack first; the locals are set from the top down.
        target = function.blocks.get(target_block)
        if target is None:
            return
        destinations = []
        for instruction in target.instrs:
            if str(instruction.op) != "Phi" or instruction.res is None:
                continue
            incoming = tuple(
                instruction.attributes.get("incoming_blocks") or ()
            )
            for position, origin in enumerate(incoming):
                if str(origin) == source_block and position < len(
                    instruction.args
                ):
                    destinations.append(slot_for(
                        env, instruction.res, phi_kind(instruction),
                    ))
                    push_as(
                        env, instruction.args[position],
                        phi_kind(instruction), "Phi",
                    )
        for destination in reversed(destinations):
            builder.local_set(destination)

    def emit_cfg(function, env, on_return) -> None:
        """The function's CFG through the classic label dispatcher: a loop
        over nested blocks selected by a label local, WebAssembly's
        structured spelling of an arbitrary branch.  ``on_return`` emits a
        ``Ret``'s values in the function's own convention."""

        order = list(function.blocks)
        label_local = builder.declare_local("i32")  # zero-initialised: entry
        # Phi results need their locals to exist before any predecessor
        # writes them, so walk them first.
        for block_name in order:
            for instruction in function.blocks[block_name].instrs:
                if str(instruction.op) == "Phi" and instruction.res is not None:
                    slot_for(env, instruction.res, phi_kind(instruction))
        count = len(order)
        builder.loop()
        for _ in range(count):
            builder.block()
        for index in range(count):
            builder.local_get(label_local).i32_const(index)
            builder.raw(_OP_I32_EQ).br_if(index)
        builder.raw(_OP_UNREACHABLE)
        for index, block_name in enumerate(order):
            builder.end()  # opens block ``index``'s straight-line code region
            depth_to_loop = count - 1 - index
            for instruction in function.blocks[block_name].instrs:
                operation = str(instruction.op)
                if operation == "Phi":
                    continue  # defined by edge assignments
                if operation in ("Call", "call"):
                    emit_call(env, instruction)
                    continue
                if operation == "Br":
                    target = str(instruction.attributes.get("target"))
                    if target not in function.blocks:
                        shortfalls.append(WasmEmissionShortfall(
                            "Br", f"{function.name} has unknown target {target!r}",
                        ))
                        continue
                    edge_assignments(env, function, block_name, target)
                    builder.i32_const(order.index(target))
                    builder.local_set(label_local)
                    builder.br(depth_to_loop)
                    continue
                if operation == "CondBr":
                    on_true = str(instruction.attributes.get("true_target"))
                    on_false = str(instruction.attributes.get("false_target"))
                    if on_true not in function.blocks or on_false not in function.blocks:
                        shortfalls.append(WasmEmissionShortfall(
                            "CondBr", f"{function.name} has unknown target",
                        ))
                        continue
                    condition = push(env, instruction.args[0])
                    if condition == "i64":
                        builder.i64_const(0).raw(_OP_I64_NE)
                    elif condition != "i32":
                        if condition is not None:
                            shortfalls.append(WasmEmissionShortfall(
                                operation,
                                f"condition %t{instruction.args[0].id} is "
                                f"{condition}",
                            ))
                        continue
                    builder.if_()
                    edge_assignments(env, function, block_name, on_true)
                    builder.i32_const(order.index(on_true))
                    builder.local_set(label_local)
                    builder.br(depth_to_loop + 1)
                    builder.else_()
                    edge_assignments(env, function, block_name, on_false)
                    builder.i32_const(order.index(on_false))
                    builder.local_set(label_local)
                    builder.br(depth_to_loop + 1)
                    builder.end()
                    continue
                if operation in ("Ret", "Return"):
                    on_return(instruction)
                    builder.raw(_OP_RETURN)
                    continue
                emit_straight_instruction(env, instruction)
        builder.end()  # the dispatcher loop
        # Every block branches or returns, so this point is never reached;
        # ``unreachable`` is stack-polymorphic, which lets a function with
        # results validate here.
        builder.raw(_OP_UNREACHABLE)

    # -- the exported entry --------------------------------------------------
    root_env = _Env({
        value_id: index for index, (_, value_id) in enumerate(parameters)
        if value_id in argument_ids and index not in output_parameter_indices.values()
    }, root)
    for parameter_index, (kind, value_id) in enumerate(parameters):
        if value_id not in argument_ids or parameter_index in output_parameter_indices.values():
            continue
        root_env.kinds[value_id] = "i32" if kind == "buffer" else kind
    for kind, value_id in parameters:
        if value_id in argument_ids and kind == "buffer":
            element = physical_kind(root_env, next(
                formal for formal in root.args if int(formal.id) == value_id
            ))
            if element is not None:
                root_env.pointer_element_kind[value_id] = element

    for formal in compiler_frame_formals:
        value_id = int(formal.id)
        requirement = storage_requirements.get(root.name, {}).get(value_id)
        count = (
            None if requirement is None or requirement.dynamic else
            requirement.element_count if requirement.element_count is not None
            else 1 if not requirement.shape else None
        )
        dtype = "" if requirement is None else str(requirement.dtype).casefold()
        width = 4 if dtype in _I32_DTYPES else 8 if dtype in (_I64_DTYPES | _F64_DTYPES) else None
        if count is None or width is None:
            shortfalls.append(WasmEmissionShortfall(
                "formal", f"compiler frame %t{value_id} has no static storage obligation",
            ))
            continue
        start = (aggregate_end + width - 1) // width * width
        aggregate_end = start + max(1, count) * width
        held = builder.declare_local("i32")
        builder.i32_const(start).local_set(held)
        root_env.slots[value_id] = held
        root_env.kinds[value_id] = "i32"
        root_env.pointer_element_kind[value_id] = (
            "i32" if width == 4 else "f64" if dtype in _F64_DTYPES else "i64"
        )

    def root_return(instruction) -> None:
        for returned in instruction.args:
            output_parameter = output_parameter_indices.get(int(returned.id))
            if output_parameter is None:
                continue
            if (root.name, int(returned.id)) in pointer_ids or returned.shape:
                result_fact = posted_ssa_value_fact(
                    root, int(returned.id), book=book,
                )
                count = output_elements.get(int(returned.id))
                source_slot = root_env.slots.get(int(returned.id))
                source_kind = root_env.kinds.get(int(returned.id))
                element_kind = (
                    None if result_fact is None else physical_kind(root_env, returned)
                )
                if (count is None or source_slot is None
                        or source_kind != "i32"
                        or result_fact is None
                        or result_fact.storage != "span"
                        or result_fact.rank != len(tuple(returned.shape or ()))
                        or element_kind not in {"f64", "i64"}):
                    shortfalls.append(WasmEmissionShortfall(
                        "Ret", "aggregate return has no exact span "
                        f"storage and static output count: %t{returned.id}",
                    ))
                    continue
                counter = builder.declare_local("i32")
                builder.i32_const(0).local_set(counter)
                builder.block().loop()
                builder.local_get(counter).i32_const(count)
                builder.raw(OP_I32_GE_S).br_if(1)
                builder.local_get(output_parameter)
                builder.local_get(counter).i32_const(3)
                builder.raw(_OP_I32_SHL).raw(OP_I32_ADD)
                builder.local_get(source_slot)
                builder.local_get(counter).i32_const(3)
                builder.raw(_OP_I32_SHL).raw(OP_I32_ADD)
                if element_kind == "i64":
                    builder.i64_load().i64_store()
                else:
                    builder.load().store()
                builder.local_get(counter).i32_const(1)
                builder.raw(OP_I32_ADD).local_set(counter)
                builder.br(0).end().end()
                continue
            builder.local_get(output_parameter)
            if output_dtypes[int(returned.id)] == "int64":
                # A signed GLSL int32 result uses signed extension. Bool still
                # uses the unsigned extension declared by the common ABI.
                conversions = _CONVERSIONS
                if str(returned.dtype).casefold() in {"int32", "i32"}:
                    conversions = {**_CONVERSIONS, ("i32", "i64"): (0xAC,)}
                if push_as(root_env, returned, "i64", "Ret", conversions):
                    builder.raw(_OP_I64_STORE, 3, 0)
            elif push_as(root_env, returned, "f64", "Ret"):
                builder.store()

    emit_cfg(root, root_env, root_return)
    root_builder = builder

    # -- every other reachable function --------------------------------------
    further: list = []
    for fname in others:
        function = module.functions[fname]
        param_kinds, result_kinds = signatures[fname]
        builder = CodeBuilder("f64", parameter_count=len(param_kinds))
        env = _Env({int(formal.id): index for index, formal in enumerate(function.args)}, function)
        for formal, kind in zip(function.args, param_kinds):
            env.kinds[int(formal.id)] = kind
            if kind == "i32":
                element = physical_kind(env, formal)
                if element is not None:
                    env.pointer_element_kind[int(formal.id)] = element

        def callee_return(instruction, env=env, result_kinds=result_kinds) -> None:
            if len(instruction.args) != len(result_kinds):
                shortfalls.append(WasmEmissionShortfall(
                    "Ret", f"{env.function.name}: Ret arity disagrees with its signature",
                ))
                return
            for value, kind in zip(instruction.args, result_kinds):
                if not push_as(env, value, kind, "Ret", _INDEX_CONVERSIONS):
                    return

        emit_cfg(function, env, callee_return)
        further.append(WasmFunction(
            name=fname, parameter_types=tuple(param_kinds), body=builder,
            result_types=tuple(result_kinds),
        ))
    builder = root_builder

    binary = b""
    imports = [
        WasmImport(
            module="math", field=symbol, kind="func",
            parameter_types=("f64",) * _MATH_IMPORTS[symbol],
            result_types=("f64",),
        )
        for symbol in import_names
    ]
    if not shortfalls:
        binary = build_module(
            function_name=name,
            parameter_types=parameter_types,
            body=root_builder,
            memory_pages=max(1, (aggregate_end + 65535) // 65536),
            imports=imports,
            functions=further,
        )
    # The host-facing tables and the book rows, as the C module lane posts
    # them: MODULE_TEXT and BUFFER_ORDER derive from the slot values' cells,
    # and the ProgramABI slot rows (which formal is which state field) are
    # posted once here and read by ``NativeSystem.slots``.
    buffer_order = tuple(value_id for kind, value_id in parameters if kind == "buffer")
    values_by_id = {int(formal.id): formal for formal in root.args}
    for value in returned_values:
        values_by_id.setdefault(int(value.id), value)
    buffer_dtypes = []
    buffer_shapes = []
    for value_id in buffer_order:
        value = values_by_id.get(value_id)
        dtype = output_dtypes.get(value_id)
        if dtype is None:
            physical = None if value is None else physical_kind(root_env, value)
            dtype = {"i64": "int64", "i32": "int32", "f32": "float32", "f64": "float64"}.get(physical, "float64")
        buffer_dtypes.append(dtype)
        buffer_shapes.append(tuple(getattr(value, "shape", ()) or ()))
    emission_record = None
    if not shortfalls:
        from .concordance_declarations import (
            ArtifactPart, Backend, EMISSION_WASM, VALUE_WITHOUT_IDENTITY_CELL,
        )
        from .emission_concordance import (
            ArtifactEmission, emission_book, post_artifact_part,
            post_program_abi_field_slots, value_cell,
        )

        emission = emission_book(module, "emit_ssa_module_to_wasm_core")
        slot_cells = tuple(value_cell(emission, root, value_id) for value_id in buffer_order)
        module_text = post_artifact_part(
            emission, name, Backend.WASM_BINARY, ArtifactPart.MODULE_TEXT,
            data=binary, sources=slot_cells, reason=VALUE_WITHOUT_IDENTITY_CELL,
            stage=EMISSION_WASM, module=module, what="emit_ssa_module_to_wasm_core",
        )
        buffer_order_cell = None if emission is None else post_artifact_part(
            emission, name, Backend.WASM_BINARY, ArtifactPart.BUFFER_ORDER,
            data=repr((buffer_order, tuple(buffer_dtypes), tuple(buffer_shapes))),
            location=buffer_order, sources=slot_cells,
            reason=VALUE_WITHOUT_IDENTITY_CELL, stage=EMISSION_WASM,
        )
        post_program_abi_field_slots(
            emission, root, name, Backend.WASM_BINARY, buffer_order=buffer_order,
            buffer_dtypes=buffer_dtypes, buffer_shapes=buffer_shapes,
            buffer_order_cell=buffer_order_cell, stage=EMISSION_WASM,
        )
        if emission is not None:
            emission_record = ArtifactEmission(
                emission, Backend.WASM_BINARY, module_text, buffer_order_cell,
                root=root,
            )
    api = None
    if not shortfalls:
        from .compiled_program_api import CompiledProgramAPI, describe_fortran_function

        source_names = {
            int(value_id): str(source)
            for source, value_id in tuple(root.metadata.get("parameter_names") or ())
        }
        for source, value_id in tuple(root.metadata.get("named_outputs") or ()):
            source_names.setdefault(int(value_id), str(source))
        entry_point = describe_fortran_function(
            name, root, outputs=returned_values, kind="control",
            source_names=source_names,
            note="every parameter is a byte offset into the module's exported memory",
        )
        api = CompiledProgramAPI(
            module=name, language="wasm", entry=name, entry_points=(entry_point,),
            metadata={"memory_export": "memory", "reserved_bytes": int(aggregate_end)},
        )
    return WasmCoreArtifact(
        name=name,
        binary=binary,
        parameters=tuple(parameters),
        shortfalls=tuple(shortfalls),
        output_order=tuple(int(value.id) for value in returned_values),
        precision_sections=precision_present,
        output_dtypes=tuple(sorted(output_dtypes.items())),
        output_elements=tuple(sorted(output_elements.items())),
        reserved_bytes=aggregate_end,
        imports=tuple(("math", symbol, _MATH_IMPORTS[symbol]) for symbol in import_names),
        functions=tuple(
            (item.name, tuple(item.parameter_types), tuple(item.result_types))
            for item in further
        ),
        buffer_dtypes=tuple(buffer_dtypes),
        buffer_shapes=tuple(buffer_shapes),
        emission=emission_record,
        api=api,
    )


_NODE_CORE_WORKER = r"""
import readline from 'node:readline';
let exports = null;
let held_args = [];
let entry = '';
const rl = readline.createInterface({ input: process.stdin });
const reply = (payload) => process.stdout.write(JSON.stringify(payload) + '\n');
rl.on('line', async (line) => {
  let msg;
  try { msg = JSON.parse(line); } catch (e) { reply({ok: false, error: 'bad json'}); return; }
  try {
    if (msg.cmd === 'init') {
      const bytes = Buffer.from(msg.module, 'base64');
      const math = {};
      for (const name of ['pow','exp','log','sin','cos','tan','asin','acos','atan','atan2',
                          'sinh','cosh','tanh','asinh','acosh','atanh']) math[name] = Math[name];
      const { instance } = await WebAssembly.instantiate(bytes, {math});
      exports = instance.exports;
      entry = msg.entry;
      const need = Math.ceil(msg.bytes / 65536);
      const have = exports.memory.buffer.byteLength / 65536;
      if (need > have) exports.memory.grow(need - have);
      for (const buffer of msg.buffers) {
        const data = Buffer.from(buffer.data, 'base64');
        new Uint8Array(exports.memory.buffer, buffer.off, data.length).set(data);
      }
      held_args = msg.args.map((value, index) =>
        msg.kinds[index] === 'i64' ? BigInt(value) : value);
      reply({ok: true});
    } else if (msg.cmd === 'run') {
      exports[entry](...held_args);
      reply({ok: true});
    } else if (msg.cmd === 'read') {
      const view = Buffer.from(exports.memory.buffer, msg.off, msg.len);
      reply({ok: true, data: view.toString('base64')});
    } else if (msg.cmd === 'exit') {
      process.exit(0);
    } else {
      reply({ok: false, error: 'unknown cmd'});
    }
  } catch (error) {
    reply({ok: false, error: String(error && error.stack || error)});
  }
});
"""


class _WasmCoreBuffers:
    """Read-through view of the worker's linear memory, keyed by value id.

    The arrays live inside the Node process; a lookup fetches the bytes at
    that moment, so a read AFTER ``run()`` sees what the kernel wrote --
    the same observable behaviour as the in-process lanes' shared arrays.
    """

    def __init__(self, execution, layout):
        self._execution = execution
        self._layout = layout  # value_id -> (offset, byte_length, dtype)

    def __contains__(self, value_id) -> bool:
        return int(value_id) in self._layout

    def __getitem__(self, value_id):
        import numpy as np

        offset, length, dtype = self._layout[int(value_id)]
        raw = self._execution.request({
            "cmd": "read", "off": offset, "len": length,
        })["data"]
        return np.frombuffer(
            _base64.b64decode(raw), dtype=dtype,
        ).copy()


class WasmCoreExecution:
    """One instantiated core inside a persistent Node worker."""

    def __init__(self, artifact: WasmCoreArtifact, feeds: Mapping[int, Any]):
        import numpy as np
        import shutil

        node = shutil.which("node")
        if node is None:
            raise RuntimeError(
                "node is required to execute WebAssembly cores"
            )
        layout: dict[int, tuple[int, int, str]] = {}
        buffers = []
        offset = max(8, int(artifact.reserved_bytes))
        offset = (offset + 7) // 8 * 8
        arguments: list[float | int | str] = []
        argument_kinds: list[str] = []
        output_dtypes = dict(artifact.output_dtypes)
        output_elements = dict(artifact.output_elements)
        buffer_dtypes = dict(zip(artifact.buffer_order, artifact.buffer_dtypes))
        for kind, value_id in artifact.parameters:
            fed = feeds.get(int(value_id))
            argument_kinds.append(kind)
            if kind == "buffer":
                dtype = buffer_dtypes.get(int(value_id), output_dtypes.get(int(value_id), "float64"))
                held = np.ascontiguousarray(
                    np.atleast_1d(np.asarray(
                        (np.zeros(output_elements[int(value_id)], dtype=dtype)
                         if fed is None and int(value_id) in output_elements
                         else 0 if fed is None else fed)
                    )), dtype=dtype,
                )
                layout[int(value_id)] = (offset, held.nbytes, dtype)
                buffers.append({
                    "off": offset,
                    "data": _base64.b64encode(held.tobytes()).decode(),
                })
                arguments.append(offset)
                offset += (held.nbytes + 7) // 8 * 8
            elif kind == "i64":
                # JSON numbers stop being exact at 2**53; the worker reads
                # the decimal text into a BigInt, which an i64 formal needs.
                arguments.append(str(int(0 if fed is None else fed)))
            elif kind == "i32":
                arguments.append(int(0 if fed is None else fed))
            else:
                arguments.append(float(0.0 if fed is None else fed))
        self._process = _subprocess.Popen(
            [node, "--input-type=module", "--eval", _NODE_CORE_WORKER],
            stdin=_subprocess.PIPE, stdout=_subprocess.PIPE,
            stderr=_subprocess.PIPE, text=True, encoding="utf-8",
        )
        self.buffers = _WasmCoreBuffers(self, layout)
        self.request({
            "cmd": "init",
            "module": _base64.b64encode(artifact.binary).decode(),
            "entry": artifact.name,
            "bytes": offset,
            "buffers": buffers,
            "args": arguments,
            "kinds": argument_kinds,
        })

    def request(self, payload: dict) -> dict:
        self._process.stdin.write(_json.dumps(payload) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            error = self._process.stderr.read()
            raise RuntimeError(f"wasm worker died: {error[-1500:]}")
        answer = _json.loads(line)
        if not answer.get("ok"):
            raise RuntimeError(
                f"wasm worker error: {answer.get('error')!r}"
            )
        return answer

    def run(self) -> "WasmCoreExecution":
        self.request({"cmd": "run"})
        return self

    def close(self) -> None:
        try:
            self.request({"cmd": "exit"})
        except Exception:
            pass
        self._process.terminate()

    def __del__(self):  # best-effort; the worker also dies with the parent
        try:
            self._process.terminate()
        except Exception:
            pass


def prepare_wasm_core_execution(
    artifact: WasmCoreArtifact, feeds: Mapping[int, Any],
) -> WasmCoreExecution:
    if not artifact.complete:
        raise ValueError("wasm core artifact has emission shortfalls")
    return WasmCoreExecution(artifact, feeds)


__all__ = [
    "SSAWasmArtifact",
    "WasmCoreArtifact",
    "WasmCoreExecution",
    "WasmEmissionShortfall",
    "emit_ssa_function_to_wasm",
    "emit_ssa_module_to_wasm_core",
    "prepare_wasm_core_execution",
]
