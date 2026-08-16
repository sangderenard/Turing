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
) -> SSAWasmArtifact:
    function: Function = module.functions[function_name]
    name = str(entry_name or function_name)
    input_names = tuple(function.metadata.get("argument_names", ()))
    output_names = tuple(function.metadata.get("output_names", ()))
    if set(function.blocks) != {"entry"}:
        return SSAWasmArtifact(
            name, "", b"", input_names, output_names, (), (),
            (WasmEmissionShortfall("control", "direct scalar WASM requires one entry block"),),
        )
    if len(input_names) != len(function.args):
        input_names = tuple(f"arg{index}" for index in range(len(function.args)))

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
    input_offsets = tuple(index * 8 for index in range(len(function.args)))
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
        elif op in {"Add", "Sub", "Mul", "Div", "Max", "Min"} and len(args) == 2:
            get(args[0]); get(args[1]); builder.op(op.lower())
            wat_operation = f"local.get $t{args[0]} local.get $t{args[1]} f64.{op.lower()}"
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
            elif exponent == -2.0:
                builder.value_const(1.0); get(args[0]); get(args[0]); builder.op("mul"); builder.op("div")
                wat_operation = (
                    "f64.const 0x1.0000000000000p+0 "
                    f"local.get $t{args[0]} local.get $t{args[0]} f64.mul f64.div"
                )
            elif exponent == 0.5:
                get(args[0])
                builder.op("sqrt")
                wat_operation = f"local.get $t{args[0]} f64.sqrt"
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
    output_offsets = tuple((len(function.args) + index) * 8 for index in range(len(outputs)))
    for value_id, offset in zip(outputs, output_offsets):
        builder.local_get(0).i32_const(offset).raw(OP_I32_ADD)
        get(value_id)
        builder.store()
        wat_lines.append(
            f"    local.get $io i32.const {offset} i32.add local.get $t{value_id} f64.store"
        )
    wat_lines.extend(("  )", ")", ""))
    binary = b"" if shortfalls else build_module(
        function_name=name,
        parameter_types=["i32"],
        body=builder,
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
