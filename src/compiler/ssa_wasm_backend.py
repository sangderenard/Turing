"""Direct scalar repository-SSA to WebAssembly emission."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..transmogrifier.ssa import Function, IRModule
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


def emit_ssa_function_to_wasm(
    module: IRModule, function_name: str, *, entry_name: str | None = None,
    trig_solver: str = "lut", trig_epsilon: float | None = None,
) -> SSAWasmArtifact:
    """Emit one scalar SSA function as a WebAssembly module.

    ``trig_solver`` selects how a function WebAssembly has no instruction for
    is realised, mirroring ``LLVMTrigSolver`` in the LLVM lane: ``"lut"``
    bakes the sampled table into linear memory, ``"continuous"`` evaluates a
    reduced argument series in arithmetic alone and needs no data segment.
    Both are available deliberately -- a table costs memory and an interpolation,
    a series costs multiplies, and which is cheaper depends on the target.
    """
    from .work_contract import active_contract

    function: Function = module.functions[function_name]
    name = str(entry_name or function_name)
    contract = active_contract()
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

    def get(value_id: int) -> None:
        builder.local_get(locals_by_id[int(value_id)])

    for instruction in function.blocks["entry"].instrs:
        op = str(instruction.op)
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
        elif op == "Pow" and len(args) == 2:
            exponent = constants.get(args[1])
            if exponent == 2.0:
                get(args[0]); get(args[0]); builder.op("mul")
                wat_operation = f"local.get $t{args[0]} local.get $t{args[0]} f64.mul"
            elif exponent == -1.0:
                builder.value_const(1.0); get(args[0]); builder.op("div")
                wat_operation = f"f64.const 0x1.0000000000000p+0 local.get $t{args[0]} f64.div"
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


__all__ = ["SSAWasmArtifact", "WasmEmissionShortfall", "emit_ssa_function_to_wasm"]
