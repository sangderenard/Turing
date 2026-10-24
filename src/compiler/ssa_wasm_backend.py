"""Direct scalar repository-SSA to WebAssembly emission."""

from __future__ import annotations

from dataclasses import dataclass
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
    OP_I32_TRUNC_F64_S,
    build_module,
)


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
                or instruction.res.dtype
                or "float64"
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
_I32_OPS = {"Add": 0x6A, "Sub": 0x6B, "Mul": 0x6C, "Div": 0x6D}
_I32_COMPARISONS = {
    "Lt": 0x48, "Le": 0x4C, "Gt": 0x4A, "Ge": 0x4E, "Eq": 0x46, "Ne": 0x47,
}
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
# The C lane's implicit conversions: (double) of an integer and bool widened
# unsigned.  Narrowings are never implicit: a Cast truncates a double the
# way C does (i64.trunc_f64_s traps where C's is undefined), and only an
# address index is wrapped to i32.
_CONVERSIONS = {
    ("i64", "f64"): (0xB9,),  # f64.convert_i64_s
    ("i32", "f64"): (0xB7,),  # f64.convert_i32_s
    ("i32", "i64"): (0xAD,),  # i64.extend_i32_u
}
_CAST_CONVERSIONS = {**_CONVERSIONS, ("f64", "i64"): (0xB0,)}
_INDEX_CONVERSIONS = {**_CONVERSIONS, ("i64", "i32"): (0xA7,)}


@dataclass(frozen=True, slots=True)
class WasmCoreArtifact:
    """A whole repository-call module as one exported wasm function.

    ``parameters`` is the call ABI in formal order: ``("buffer", id)`` for an
    i32 byte-offset parameter, ``("i32", id)`` / ``("f64", id)`` for scalars
    passed by value. ``buffer_order`` lists the buffer-parameter value ids,
    which is what a feed dictionary keys.
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

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    @property
    def buffer_order(self) -> tuple[int, ...]:
        return tuple(
            value_id for kind, value_id in self.parameters
            if kind == "buffer"
        )


def emit_ssa_module_to_wasm_core(
    module: IRModule, function_name: str, *, entry_name: str | None = None,
) -> WasmCoreArtifact:
    """Emit ``function_name`` plus its inlined callees as one wasm function."""

    from .ir_identities import precision_backend_shortfalls

    name = str(entry_name or function_name)
    root: Function = module.functions[function_name]
    root_returns = tuple(
        instruction.args
        for block in root.blocks.values()
        for instruction in block.instrs
        if str(instruction.op) in {"Ret", "ret", "Return", "return"}
    )
    returned_values = root_returns[-1] if root_returns else ()
    shortfalls: list[WasmEmissionShortfall] = []
    reachable = [function_name]
    for block in root.blocks.values():
        for instruction in block.instrs:
            if instruction.op in ("Call", "call"):
                reachable.append(
                    str(instruction.attributes.get("callee") or "")
                )
    shortfalls.extend(
        WasmEmissionShortfall(
            "precision_section",
            "backend cannot honour precision obligations "
            + repr(item["missing"]) + f" in {item['function']}",
        )
        for item in precision_backend_shortfalls(module, "wasm", reachable)
    )

    # Which root formals are addressed through (see the C module lane for
    # the fixed-point version; here one call level suffices because the
    # region is the only callee and it is inlined).
    pointer_ids: set[int] = set()
    for callee_name in reachable:
        function = module.functions.get(callee_name)
        if function is None:
            continue
        for block in function.blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) == "GetElementPtr" and instruction.args:
                    pointer_ids.add(int(instruction.args[0].id))
                elif str(instruction.op) == "Store" and len(instruction.args) >= 2:
                    pointer_ids.add(int(instruction.args[1].id))
    for block in root.blocks.values():
        for instruction in block.instrs:
            if instruction.op not in ("Call", "call"):
                continue
            callee = module.functions.get(
                str(instruction.attributes.get("callee") or "")
            )
            if callee is None:
                continue
            for actual, formal in zip(instruction.args, callee.args):
                if int(formal.id) in pointer_ids:
                    pointer_ids.add(int(actual.id))

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
        if dtype in _I32_DTYPES:
            return "i32"
        if dtype in _F64_DTYPES:
            return "f64"
        return None

    parameters: list[tuple[str, int]] = []
    parameter_types: list[str] = []
    for formal in root.args:
        value_id = int(formal.id)
        if value_id in pointer_ids:
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
    for output in returned_values:
        value_id = int(output.id)
        if value_id in argument_ids:
            continue
        output_parameter_indices[value_id] = len(parameters)
        output_dtypes[value_id] = (
            "float64" if value_kind(output) in (None, "f64") else "int64"
        )
        parameters.append(("buffer", value_id))
        parameter_types.append("i32")

    builder = CodeBuilder("f64", parameter_count=len(parameters))
    precision_present = False

    class _Env:
        """Value id -> local index, with parameters pre-bound."""

        def __init__(self, bindings):
            self.slots = dict(bindings)
            self.kinds = {}
            # A repository Call may return ``ssa.aggregate``.  The wrapper's
            # GetElementPtr carries the exact ``source_output_id`` for each
            # member, so retain that identity rather than reconstructing it
            # from aggregate position after the callee is inlined.
            self.aggregate_members = {}
            self.direct_pointer_values = {}

    root_env = _Env({
        int(formal.id): index for index, formal in enumerate(root.args)
    })
    for kind, value_id in parameters[:len(root.args)]:
        root_env.kinds[value_id] = "i32" if kind == "buffer" else kind

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
                "operand", f"%t{value.id} is unavailable"
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

    def emit_straight_instruction(env, instruction) -> None:
        """One non-control instruction into the current position."""

        nonlocal precision_present
        operation = str(instruction.op)
        if instruction.attributes.get("precision_section"):
            precision_present = True
        if operation in {"NoneValue", "nonevalue"}:
            shortfalls.append(WasmEmissionShortfall(
                operation,
                "core Wasm has no untyped absence value; lower an optional "
                "to explicit presence and payload values before emission",
            ))
            return
        if operation == "Const":
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
            else:
                builder.value_const(float(held))
            builder.local_set(slot_for(env, instruction.res, kind))
            return
        if operation == "GetElementPtr" and len(instruction.args) >= 2:
            aggregate = env.aggregate_members.get(int(instruction.args[0].id))
            source_output_id = instruction.attributes.get("source_output_id")
            if aggregate is not None and source_output_id is not None:
                held = aggregate.get(int(source_output_id))
                if held is None:
                    shortfalls.append(WasmEmissionShortfall(
                        operation,
                        "aggregate has no declared source output "
                        f"%t{source_output_id}",
                    ))
                    return
                env.direct_pointer_values[int(instruction.res.id)] = held
                return
            # address = base_offset + (index << 3); limbs are f64.
            if not (push_as(env, instruction.args[0], "i32", operation)
                    and push_as(env, instruction.args[1], "i32", operation,
                                _INDEX_CONVERSIONS)):
                return
            builder.i32_const(3).raw(_OP_I32_SHL).raw(OP_I32_ADD)
            builder.local_set(slot_for(env, instruction.res, "i32"))
            return
        if operation == "Load" and instruction.args:
            direct = env.direct_pointer_values.get(int(instruction.args[0].id))
            if direct is not None:
                value_id, held_kind = direct
                env.slots[int(instruction.res.id)] = value_id
                env.kinds[int(instruction.res.id)] = held_kind
                return
            if value_kind(instruction.res) not in (None, "f64"):
                shortfalls.append(WasmEmissionShortfall(
                    operation,
                    f"%t{instruction.res.id} is {instruction.res.dtype}; "
                    "linear-memory cells here are f64",
                ))
                return
            if not push_as(env, instruction.args[0], "i32", operation):
                return
            builder.load()
            builder.local_set(slot_for(env, instruction.res, "f64"))
            return
        if operation == "Store" and len(instruction.args) >= 2:
            if not (push_as(env, instruction.args[1], "i32", operation)
                    and push_as(env, instruction.args[0], "f64", operation)):
                return
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
                builder.raw(0x9A)  # f64.neg
            builder.local_set(slot_for(env, instruction.res, kind or "f64"))
            return
        if operation in {"Cast", "CastLike"} and instruction.args:
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
                else:
                    builder.i32_const(0).raw(_OP_I32_NE)
            elif not push_as(
                env, instruction.args[0], kind, operation, _CAST_CONVERSIONS,
            ):
                return
            builder.local_set(slot_for(env, instruction.res, kind))
            return
        if operation in _F64_OPS and len(instruction.args) == 2:
            kinds = [env.kinds.get(int(a.id)) for a in instruction.args]
            kind = value_kind(instruction.res) or (
                "f64" if "f64" in kinds else kinds[0] or "f64"
            )
            if kind != "f64" and operation == "Div":
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
                "f64": _F64_OPS, "i64": _I64_OPS, "i32": _I32_OPS,
            }[kind][operation])
            builder.local_set(slot_for(env, instruction.res, kind))
            return
        if operation in _I32_COMPARISONS and len(instruction.args) == 2:
            kinds = {env.kinds.get(int(a.id)) for a in instruction.args}
            kind = (
                "f64" if "f64" in kinds or None in kinds
                else "i64" if "i64" in kinds else "i32"
            )
            if not all(
                push_as(env, argument, kind, operation)
                for argument in instruction.args
            ):
                return
            builder.raw({
                "f64": _F64_COMPARISONS, "i64": _I64_COMPARISONS,
                "i32": _I32_COMPARISONS,
            }[kind][operation])
            builder.local_set(slot_for(env, instruction.res, "i32"))
            return
        shortfalls.append(WasmEmissionShortfall(
            operation, "no module-lane WASM spelling"
        ))

    def inline_call(env, instruction) -> None:
        callee_name = str(instruction.attributes.get("callee") or "")
        callee = module.functions.get(callee_name)
        if callee is None or set(callee.blocks) != {"entry"}:
            shortfalls.append(WasmEmissionShortfall(
                "Call",
                f"callee {callee_name!r} is absent or not a single block",
            ))
            return
        # The callee's formals alias the caller's actual slots; everything
        # else the callee computes gets fresh locals in its own namespace.
        callee_env = _Env({})
        for actual, formal in zip(instruction.args, callee.args):
            held = env.slots.get(int(actual.id))
            if held is None:
                shortfalls.append(WasmEmissionShortfall(
                    "Call", f"actual %t{actual.id} is unavailable",
                ))
                return
            callee_env.slots[int(formal.id)] = held
            callee_env.kinds[int(formal.id)] = env.kinds.get(int(actual.id))
        for callee_instruction in callee.blocks["entry"].instrs:
            operation = str(callee_instruction.op)
            if operation in ("Ret", "Return"):
                continue
            if operation in ("Call", "call"):
                inline_call(callee_env, callee_instruction)
                continue
            emit_straight_instruction(callee_env, callee_instruction)
        if instruction.res is None:
            return
        returned = tuple(
            item.args
            for item in callee.blocks["entry"].instrs
            if str(item.op) in {"Ret", "Return"}
        )
        declared = tuple(
            int(value_id)
            for value_id in instruction.attributes.get("output_ids", ())
        )
        if not returned or len(declared) != len(returned[-1]):
            shortfalls.append(WasmEmissionShortfall(
                "Call",
                f"callee {callee_name!r} aggregate result disagrees with "
                "its declared output_ids",
            ))
            return
        returned_by_id = {
            int(value.id): (
                None if callee_env.slots.get(int(value.id)) is None
                else (
                    callee_env.slots[int(value.id)],
                    callee_env.kinds.get(int(value.id)),
                )
            )
            for value in returned[-1]
        }
        if set(returned_by_id) != set(declared) or any(
            held is None for held in returned_by_id.values()
        ):
            shortfalls.append(WasmEmissionShortfall(
                "Call",
                f"callee {callee_name!r} did not materialize its exact "
                "declared output identities",
            ))
            return
        env.aggregate_members[int(instruction.res.id)] = {
            value_id: returned_by_id[value_id] for value_id in declared
        }

    # -- the wrapper CFG through the label dispatcher -----------------------
    order = list(root.blocks)
    label_local = builder.declare_local("i32")  # zero-initialised: entry

    def edge_assignments(source_block: str, target_block: str) -> None:
        target = root.blocks.get(target_block)
        if target is None:
            return
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
                    destination = slot_for(
                        root_env, instruction.res, phi_kind(instruction),
                    )
                    if push_as(
                        root_env, instruction.args[position],
                        phi_kind(instruction), "Phi",
                    ):
                        builder.local_set(destination)

    def phi_kind(instruction) -> str:
        return value_kind(instruction.res) or next((
            value_kind(argument) for argument in instruction.args
            if argument is not None and value_kind(argument) is not None
        ), "f64")

    # Phi results need their locals to exist before any predecessor writes
    # them, so walk them first.
    for block_name in order:
        for instruction in root.blocks[block_name].instrs:
            if str(instruction.op) == "Phi" and instruction.res is not None:
                slot_for(root_env, instruction.res, phi_kind(instruction))

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
        for instruction in root.blocks[block_name].instrs:
            operation = str(instruction.op)
            if operation == "Phi":
                continue  # defined by edge assignments
            if operation in ("Call", "call"):
                inline_call(root_env, instruction)
                continue
            if operation == "Br":
                target = str(instruction.attributes.get("target"))
                edge_assignments(block_name, target)
                builder.i32_const(order.index(target))
                builder.local_set(label_local)
                builder.br(depth_to_loop)
                continue
            if operation == "CondBr":
                on_true = str(instruction.attributes.get("true_target"))
                on_false = str(instruction.attributes.get("false_target"))
                condition = push(root_env, instruction.args[0])
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
                edge_assignments(block_name, on_true)
                builder.i32_const(order.index(on_true))
                builder.local_set(label_local)
                builder.br(depth_to_loop + 1)
                builder.else_()
                edge_assignments(block_name, on_false)
                builder.i32_const(order.index(on_false))
                builder.local_set(label_local)
                builder.br(depth_to_loop + 1)
                builder.end()
                continue
            if operation in ("Ret", "Return"):
                for returned in instruction.args:
                    output_parameter = output_parameter_indices.get(
                        int(returned.id)
                    )
                    if output_parameter is None:
                        continue
                    if int(returned.id) in pointer_ids or returned.shape:
                        shortfalls.append(WasmEmissionShortfall(
                            "Ret",
                            "aggregate return requires an explicit output-arena "
                            f"copy: %t{returned.id}",
                        ))
                        continue
                    builder.local_get(output_parameter)
                    if output_dtypes[int(returned.id)] == "int64":
                        if push_as(root_env, returned, "i64", operation):
                            builder.raw(_OP_I64_STORE, 3, 0)
                    elif push_as(root_env, returned, "f64", operation):
                        builder.store()
                builder.raw(_OP_RETURN)
                continue
            emit_straight_instruction(root_env, instruction)
    builder.end()  # the dispatcher loop

    binary = b""
    if not shortfalls:
        binary = build_module(
            function_name=name,
            parameter_types=parameter_types,
            body=builder,
            memory_pages=1,
        )
    return WasmCoreArtifact(
        name=name,
        binary=binary,
        parameters=tuple(parameters),
        shortfalls=tuple(shortfalls),
        output_order=tuple(int(value.id) for value in returned_values),
        precision_sections=precision_present,
        output_dtypes=tuple(sorted(output_dtypes.items())),
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
      const { instance } = await WebAssembly.instantiate(bytes, {});
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
        offset = 8  # keep 0 unused so a zero offset is never a real buffer
        arguments: list[float | int | str] = []
        argument_kinds: list[str] = []
        output_dtypes = dict(artifact.output_dtypes)
        for kind, value_id in artifact.parameters:
            fed = feeds.get(int(value_id))
            argument_kinds.append(kind)
            if kind == "buffer":
                dtype = output_dtypes.get(int(value_id), "float64")
                held = np.ascontiguousarray(
                    np.atleast_1d(np.asarray(
                        0 if fed is None else fed
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
