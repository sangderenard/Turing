"""Deterministic likeness between Turing SSA operations and LLVM SSA.

The LLVM sibling of :mod:`ssa_fortran_backend`'s operation tables, and
nothing more: one table entry per SSA operation, giving the exact LLVM
spelling that operation reduces to. ``{0}`` and ``{1}`` are the operand
registers, ``{out}`` the result register. Anything absent is reported as an
unsupported operation by consumers rather than guessed at -- the same
contract every other backend table in this repository carries.

Scalar operations are pure LLVM instructions. Tensor operations name the
authored C-kernel LLVM definitions (``c_backend_llvm_ssa.TRANSLATIONS``);
their likeness is the kernel symbol, and their calling convention is the
authored signature -- both already catalogued there. This module does not
re-state kernel bodies or signatures; it states which symbol each SSA tensor
operation is deterministically alike to.
"""

from __future__ import annotations


# --- scalar likeness: SSA opcode -> LLVM instruction template ---------------
# Floating (double) domain first; the integer columns join as the type
# vocabulary fills out, exactly as everywhere else in the matrix.
_BINARY: dict[str, str] = {
    "Add": "{out} = fadd double {0}, {1}",
    "Sub": "{out} = fsub double {0}, {1}",
    "Mul": "{out} = fmul double {0}, {1}",
    "Div": "{out} = fdiv double {0}, {1}",
    "Mod": "{out} = frem double {0}, {1}",
    "Pow": "{out} = call double @llvm.pow.f64(double {0}, double {1})",
    "FloorDiv": (
        "{out}.q = fdiv double {0}, {1}\n"
        "{out} = call double @llvm.floor.f64(double {out}.q)"
    ),
    "Eq": "{out} = fcmp oeq double {0}, {1}",
    "Ne": "{out} = fcmp one double {0}, {1}",
    "Lt": "{out} = fcmp olt double {0}, {1}",
    "Le": "{out} = fcmp ole double {0}, {1}",
    "Gt": "{out} = fcmp ogt double {0}, {1}",
    "Ge": "{out} = fcmp oge double {0}, {1}",
    # Unsigned bit-sequence comparisons (the Fortran table's blt/ble): the
    # operands are integer-typed values, compared as unsigned.
    "ULt": "{out} = icmp ult i64 {0}, {1}",
    "ULe": "{out} = icmp ule i64 {0}, {1}",
    "And": "{out} = and i1 {0}, {1}",
    "Or": "{out} = or i1 {0}, {1}",
    "Xor": "{out} = xor i1 {0}, {1}",
    "BitAnd": "{out} = and i64 {0}, {1}",
    "BitOr": "{out} = or i64 {0}, {1}",
    "BitXor": "{out} = xor i64 {0}, {1}",
    "Shl": "{out} = shl i64 {0}, {1}",
    "Shr": "{out} = lshr i64 {0}, {1}",
    "Min": "{out} = call double @llvm.minnum.f64(double {0}, double {1})",
    "Max": "{out} = call double @llvm.maxnum.f64(double {0}, double {1})",
}

_UNARY: dict[str, str] = {
    "Neg": "{out} = fneg double {0}",
    "Abs": "{out} = call double @llvm.fabs.f64(double {0})",
    "Sqrt": "{out} = call double @llvm.sqrt.f64(double {0})",
    "Exp": "{out} = call double @llvm.exp.f64(double {0})",
    "Log": "{out} = call double @llvm.log.f64(double {0})",
    "Sin": "{out} = call double @llvm.sin.f64(double {0})",
    "Cos": "{out} = call double @llvm.cos.f64(double {0})",
    "Floor": "{out} = call double @llvm.floor.f64(double {0})",
    "Ceil": "{out} = call double @llvm.ceil.f64(double {0})",
    "Trunc": "{out} = call double @llvm.trunc.f64(double {0})",
    "Round": "{out} = call double @llvm.round.f64(double {0})",
    "Not": "{out} = xor i1 {0}, true",
    "Invert": "{out} = xor i64 {0}, -1",
    "SIToFP": "{out} = sitofp i32 {0} to double",
    "FPToSI": "{out} = fptosi double {0} to i32",
}

# --- tensor likeness: SSA tensor operation -> authored kernel symbol --------
# The symbol is the deterministic likeness; body and signature live with the
# authored kernels in c_backend_llvm_ssa and are not restated here.
_TENSOR: dict[str, str] = {
    "add": "binary_double",
    "sub": "binary_double",
    "mul": "binary_double",
    "truediv": "binary_double",
    "pow": "binary_double",
    "mod": "binary_double",
    "floordiv": "binary_double",
    "eq": "binary_double",
    "ne": "binary_double",
    "lt": "binary_double",
    "le": "binary_double",
    "gt": "binary_double",
    "ge": "binary_double",
    "maximum": "binary_double",
    "minimum": "binary_double",
    "add_scalar": "binary_scalar_double",
    "abs": "unary_double",
    "neg": "unary_double",
    "sqrt": "unary_double",
    "exp": "unary_double",
    "log": "unary_double",
    "tanh": "unary_double",
    "sin": "unary_double",
    "cos": "unary_double",
    "sign": "sign_double",
    "matmul": "matmul_double",
    "transpose": "transpose_double",
    "swapaxes": "transpose_double",
    "permute": "transpose_double",
    "sum": "sum_double",
    "mean": "sum_double",          # flat mean = sum likeness + scalar Div
    "sum_dim": "reduce_dim_double",
    "prod": "reduce_dim_double",
    "min": "reduce_dim_double",
    "max": "reduce_dim_double",
    "any": "reduce_dim_double",
    "all": "reduce_dim_double",
    "cumsum": "cumsum_dim_double",
    "where": "where_double",
    "broadcast": "broadcast_double",
    "fill": "fill_double",
    "zeros": "fill_double",
    "ones": "fill_double",
    "full": "fill_double",
    "float": "cast_double_to_float_values",
    "double": "cast_double_to_float_values",
    "long": "cast_double_to_int_values",
    "int": "cast_double_to_int_values",
    "arange": "create_arange",
    "extent": "extent",            # runtime metadata read; each target's own
}

# --- shape-only operations: no runtime existence, alias in every target -----
_SHAPE_ONLY = frozenset({
    "reshape", "view", "flatten", "unsqueeze", "squeeze", "contiguous",
})


def supported_scalar_operations() -> frozenset[str]:
    return frozenset(_BINARY) | frozenset(_UNARY)


def supported_tensor_operations() -> frozenset[str]:
    return frozenset(_TENSOR) | _SHAPE_ONLY


def scalar_likeness(operation: str) -> str | None:
    return _BINARY.get(operation) or _UNARY.get(operation)


def tensor_likeness(operation: str) -> str | None:
    return _TENSOR.get(operation)


# --- the table wired as an emitter ------------------------------------------
#
# The emitter renders one SSA function through the likeness tables above:
# every instruction either has a table entry or becomes a named shortfall.
# Kernel calling conventions are parsed from the authored kernel definitions
# themselves (c_backend_llvm_ssa), never restated. Compilation is a separate
# step through an LLVM compiler ahead of time -- Zig's embedded clang, the
# toolchain this repository already builds C with. No JIT.

import ctypes as _ctypes
import re as _re
import struct as _struct
import subprocess as _subprocess
import tempfile as _tempfile
from dataclasses import dataclass as _dataclass, field as _field
from pathlib import Path as _Path
from typing import Any as _Any

from ..transmogrifier.ssa import IRModule as _IRModule


@_dataclass(frozen=True)
class LLVMEmissionShortfall:
    function: str
    operation: str
    reason: str


def _kernel_signature(symbol: str) -> tuple[str, tuple[str, ...]]:
    """(return type, argument types) parsed from the authored definition."""

    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        extract_llvm_function,
    )

    text = extract_llvm_function(symbol)
    match = _re.search(
        r"define\s+([\w<>\s\*]+?)\s*@" + _re.escape(symbol) + r"\((.*?)\)",
        text, _re.DOTALL,
    )
    if match is None:
        raise ValueError(f"authored kernel {symbol!r} has no parseable define")
    returns = match.group(1).strip()
    arguments = tuple(
        parameter.strip().split()[0]
        for parameter in match.group(2).split(",")
        if parameter.strip()
    )
    return returns, arguments


def _double_literal(value: _Any) -> str:
    bits = _struct.unpack(">Q", _struct.pack(">d", float(value)))[0]
    return f"0x{bits:016X}"


def _value_llvm_type(value: _Any) -> str:
    if tuple(
        (getattr(value, "accounting", {}) or {}).get(
            "ssa_aggregate_outputs", ()
        )
    ):
        return "ptr"
    dtype = str(getattr(value, "dtype", None) or "float64").lower()
    if dtype in {"bool", "i1"}:
        return "i1"
    if dtype in {"int", "int32", "i32"}:
        return "i32"
    if dtype in {"int64", "i64", "long"}:
        return "i64"
    if dtype == "opaque_ref":
        return "i64"
    return "double"


def _value_element_count(value: _Any) -> int:
    from math import prod

    aggregate = tuple(
        (getattr(value, "accounting", {}) or {}).get(
            "ssa_aggregate_outputs", ()
        )
    )
    if aggregate:
        return len(aggregate)
    shape = tuple(getattr(value, "shape", ()) or ())
    return max(1, int(prod(map(int, shape)))) if shape else 1


def _internal_call_closure(
    module: _IRModule, root: str,
) -> tuple[set[str], set[str]]:
    """Return repository functions and authored leaves reachable from root."""

    repository: set[str] = set()
    kernels: set[str] = set()
    pending = [str(root)]
    while pending:
        name = pending.pop()
        if name in repository or name not in module.functions:
            continue
        repository.add(name)
        for block in module.functions[name].blocks.values():
            for instruction in block.instrs:
                callee = instruction.attributes.get("callee")
                if callee is None:
                    continue
                symbol = str(callee)
                try:
                    _kernel_signature(symbol)
                except (KeyError, ValueError):
                    if symbol in module.functions:
                        pending.append(symbol)
                else:
                    kernels.add(symbol)
    return repository, kernels


def _emit_repository_call_module(
    module: _IRModule,
    function_name: str,
    *,
    entry_name: str,
    text_sink: bool,
) -> "LLVMFunctionArtifact":
    """Emit a repository-SSA call closure with a pointer-only internal ABI.

    Every internal function receives one pointer per SSA argument followed by
    one pointer per result.  Aggregate call results remain explicit arrays of
    pointers, so the repository's GetElementPtr/Load projections retain their
    meaning without flattening or call-site substitution.
    """

    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        extract_llvm_declaration,
        extract_llvm_function,
    )

    reachable, kernels_used = _internal_call_closure(module, function_name)
    shortfalls: list[LLVMEmissionShortfall] = []

    values_by_function: dict[str, dict[int, _Any]] = {}
    for name in reachable:
        function = module.functions[name]
        values: dict[int, _Any] = {int(value.id): value for value in function.args}
        for block in function.blocks.values():
            for instruction in block.instrs:
                for value in instruction.args:
                    current = values.get(int(value.id))
                    if current is None or (
                        not tuple(getattr(current, "shape", ()) or ())
                        and tuple(getattr(value, "shape", ()) or ())
                    ):
                        values[int(value.id)] = value
                if instruction.res is not None:
                    value = instruction.res
                    current = values.get(int(value.id))
                    if current is None or (
                        not tuple(getattr(current, "shape", ()) or ())
                        and tuple(getattr(value, "shape", ()) or ())
                    ):
                        values[int(value.id)] = value
        values_by_function[name] = values

    # Determine which aggregate projections are actually consumed.  This
    # shrinks planned-region ABIs and removes descriptor getters made dead by
    # structural specialization before target emission.
    aggregate_outputs: dict[str, list[int]] = {}
    aggregate_output_values: dict[str, dict[int, _Any]] = {}
    aggregate_positions: dict[tuple[str, int], tuple[int, ...]] = {}
    for caller_name in reachable:
        function = module.functions[caller_name]
        uses: dict[int, int] = {}
        instructions = [
            instruction
            for block in function.blocks.values()
            for instruction in block.instrs
        ]
        for instruction in instructions:
            for argument in instruction.args:
                uses[int(argument.id)] = uses.get(int(argument.id), 0) + 1
        for instruction in instructions:
            if (
                instruction.op not in {"Call", "call"}
                or instruction.res is None
                or instruction.attributes.get("result_convention")
                != "ssa.aggregate"
            ):
                continue
            callee = str(instruction.attributes.get("callee") or "")
            declared = tuple(map(int, instruction.attributes.get("output_ids", ())))
            live_positions: list[int] = []
            address_position: dict[int, int] = {}
            projected_values: dict[int, _Any] = {}
            for follower in instructions:
                if (
                    follower.op in {"GetElementPtr", "getelementptr"}
                    and follower.res is not None
                    and follower.args
                    and int(follower.args[0].id) == int(instruction.res.id)
                ):
                    position = follower.attributes.get("aggregate_index")
                    if position is not None:
                        address_position[int(follower.res.id)] = int(position)
                elif (
                    follower.op in {"Load", "load"}
                    and follower.res is not None
                    and follower.args
                    and int(follower.args[0].id) in address_position
                    and uses.get(int(follower.res.id), 0) > 0
                ):
                    projected_position = address_position[int(follower.args[0].id)]
                    live_positions.append(projected_position)
                    projected_values[projected_position] = follower.res
            selected = tuple(dict.fromkeys(live_positions))
            if not selected:
                selected = tuple(range(len(declared)))
            aggregate_positions[(caller_name, int(instruction.res.id))] = selected
            if callee in reachable and declared:
                output_ids = [declared[index] for index in selected]
                existing = aggregate_outputs.setdefault(callee, [])
                for value_id in output_ids:
                    if value_id not in existing:
                        existing.append(value_id)
                typed = aggregate_output_values.setdefault(callee, {})
                for position in selected:
                    if position in projected_values:
                        typed[declared[position]] = projected_values[position]

    function_outputs: dict[str, tuple[_Any, ...]] = {}
    for name in reachable:
        function = module.functions[name]
        returned = next((
            tuple(instruction.args)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"Ret", "ret", "Return", "return"}
        ), ())
        ids = aggregate_outputs.get(name)
        function_outputs[name] = tuple(
            (
                values_by_function[name][value_id]
                if value_id in values_by_function[name]
                else aggregate_output_values[name][value_id]
            )
            for value_id in ids
            if (
                value_id in values_by_function[name]
                or value_id in aggregate_output_values.get(name, {})
            )
        ) if ids is not None else returned

    # A source wrapper may return a callee's aggregate unchanged (the
    # canonical backward ``bw_matmul -> matmul_vjp`` shape).  Returning an
    # array of pointers to callee temporaries is not a valid native ABI: those
    # pointees die with the wrapper's stack frame.  Spell the wrapper as the
    # same multiple-output ABI as its callee, so its caller owns every output
    # buffer and no ephemeral pointer escapes.
    forwarded_aggregate_calls: dict[str, tuple[int, str]] = {}
    changed = True
    while changed:
        changed = False
        for name in reachable:
            function = module.functions[name]
            returned = next((
                tuple(instruction.args)
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.op in {"Ret", "ret", "Return", "return"}
            ), ())
            if len(returned) != 1 or not tuple(
                (returned[0].accounting or {}).get(
                    "ssa_aggregate_outputs", ()
                )
            ):
                continue
            producer = next((
                instruction
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.op in {"Call", "call"}
                and instruction.res is not None
                and int(instruction.res.id) == int(returned[0].id)
                and str(instruction.attributes.get("callee") or "")
                in reachable
            ), None)
            if producer is None:
                continue
            callee = str(producer.attributes["callee"])
            callee_outputs = function_outputs.get(callee, ())
            if len(callee_outputs) <= 1:
                continue
            forwarded_aggregate_calls[name] = (
                int(producer.res.id), callee,
            )
            if function_outputs.get(name) != callee_outputs:
                function_outputs[name] = callee_outputs
                changed = True

    internal_symbols = {
        name: "__ssa_" + _re.sub(r"[^A-Za-z0-9_$.-]", "_", name)
        for name in reachable
    }

    def literal(payload: _Any, llvm_type: str) -> str:
        if llvm_type == "double":
            return _double_literal(0.0 if payload is None else payload)
        if llvm_type == "i1":
            return "true" if bool(payload) else "false"
        return str(int(0 if payload is None else payload))

    emitted_functions: list[str] = []
    for name in sorted(reachable, key=lambda item: item != function_name):
        function = module.functions[name]
        outputs = function_outputs[name]
        parameters = [
            *(f"ptr %arg.{index}" for index in range(len(function.args))),
            *(f"ptr %out.{index}" for index in range(len(outputs))),
        ]
        body: list[str] = []
        entry_allocas: list[str] = []
        pointers: dict[int, str] = {
            int(value.id): f"%arg.{index}"
            for index, value in enumerate(function.args)
        }
        aggregate_members: dict[int, dict[int, str]] = {}
        address_members: dict[int, str] = {}
        address_slots: dict[int, str] = {}
        allocated: set[int] = set()
        output_pointer = {
            int(value.id): f"%out.{index}"
            for index, value in enumerate(outputs)
        }

        def pointer(value: _Any) -> str:
            value_id = int(value.id)
            known = pointers.get(value_id)
            if known is not None:
                return known
            known = output_pointer.get(value_id)
            if known is not None:
                pointers[value_id] = known
                return known
            llvm_type = _value_llvm_type(value)
            count = _value_element_count(value)
            register = f"%value.{value_id}"
            if value_id not in allocated:
                entry_allocas.append(
                    f"  {register} = alloca {llvm_type}, i64 {count}, align 8"
                )
                allocated.add(value_id)
            pointers[value_id] = register
            return register

        def load_as(value: _Any, wanted: str, tag: str) -> str:
            source_type = _value_llvm_type(value)
            loaded = f"%load.{tag}"
            body.append(
                f"  {loaded} = load {source_type}, ptr {pointer(value)}, align 8"
            )
            if source_type == wanted:
                return loaded
            converted = f"%convert.{tag}"
            if wanted == "double" and source_type in {"i1", "i32", "i64"}:
                opcode = "uitofp" if source_type == "i1" else "sitofp"
                body.append(
                    f"  {converted} = {opcode} {source_type} {loaded} to double"
                )
                return converted
            if wanted in {"i32", "i64"} and source_type == "double":
                body.append(
                    f"  {converted} = fptosi double {loaded} to {wanted}"
                )
                return converted
            return loaded

        scheduled_instructions = [
            (block_name, instruction)
            for block_name, block in function.blocks.items()
            for instruction in block.instrs
        ]
        projection_values: dict[int, dict[int, _Any]] = {}
        projection_addresses: dict[int, tuple[int, int]] = {}
        constant_values = {
            int(projected.res.id): (
                projected.attributes.get("constant")
                if projected.attributes.get("constant") is not None
                else projected.attributes.get("value")
            )
            for _block_name, projected in scheduled_instructions
            if projected.op == "Const" and projected.res is not None
        }
        for _block_name, projected in scheduled_instructions:
            if (
                projected.op in {"GetElementPtr", "getelementptr"}
                and projected.res is not None
                and projected.args
                and projected.attributes.get("aggregate_index") is not None
            ):
                projection_addresses[int(projected.res.id)] = (
                    int(projected.args[0].id),
                    int(projected.attributes["aggregate_index"]),
                )
            elif (
                projected.op in {"Load", "load"}
                and projected.res is not None
                and projected.args
                and int(projected.args[0].id) in projection_addresses
            ):
                aggregate_id, position = projection_addresses[
                    int(projected.args[0].id)
                ]
                projection_values.setdefault(aggregate_id, {})[position] = (
                    projected.res
                )

        def emit_return_values() -> None:
            for output_index, output in enumerate(outputs):
                source = pointers.get(int(output.id))
                if source is None:
                    tensor_table = getattr(module, "tensor_tables", {}).get(name)
                    descriptor = (
                        tensor_table.by_id(int(output.id))
                        if tensor_table is not None else None
                    )
                    if descriptor is not None:
                        source = pointers.get(int(descriptor.data_value_id))
                destination = f"%out.{output_index}"
                if source is None or source == destination:
                    continue
                llvm_type = _value_llvm_type(output)
                count = _value_element_count(output)
                if count == 1:
                    loaded = f"%return.load.{output_index}.{len(body)}"
                    body.append(
                        f"  {loaded} = load {llvm_type}, ptr {source}, align 8"
                    )
                    body.append(
                        f"  store {llvm_type} {loaded}, ptr {destination}, align 8"
                    )
                else:
                    body.append(
                        "  call void @llvm.memcpy.p0.p0.i64("
                        f"ptr {destination}, ptr {source}, i64 {count * 8}, i1 false)"
                    )
        active_block: str | None = None
        emitted_return = False
        for instruction_index, (block_name, instruction) in enumerate(
            scheduled_instructions
        ):
            if block_name != active_block:
                body.append(f"{block_name}:")
                active_block = block_name
            operation = str(instruction.op)
            result = instruction.res
            result_id = int(result.id) if result is not None else None
            tag = f"{instruction_index}.{result_id if result_id is not None else 'v'}"

            if operation in {"Const", "StaticRef"} and result is not None:
                if operation == "StaticRef":
                    payload = int(instruction.attributes["reference_handle"])
                else:
                    payload = instruction.attributes.get("constant")
                if payload is None and "values" in instruction.attributes:
                    payload = instruction.attributes.get("values")
                if payload is None and "value" in instruction.attributes:
                    payload = instruction.attributes.get("value")
                target = pointer(result)
                if isinstance(payload, (tuple, list)):
                    for index, item in enumerate(payload):
                        slot = f"%const.slot.{tag}.{index}"
                        body.append(
                            f"  {slot} = getelementptr i32, ptr {target}, i64 {index}"
                        )
                        body.append(f"  store i32 {int(item)}, ptr {slot}, align 4")
                else:
                    llvm_type = _value_llvm_type(result)
                    body.append(
                        f"  store {llvm_type} {literal(payload, llvm_type)}, ptr {target}, align 8"
                    )
                continue

            if operation in {"Phi", "phi"} and result is not None:
                incoming_blocks = tuple(
                    instruction.attributes.get("incoming_blocks") or ()
                )
                incoming = tuple(instruction.attributes.get("incoming") or ())
                if incoming:
                    incoming_blocks = tuple(str(item[0]) for item in incoming)
                    incoming_values = tuple(item[1] for item in incoming)
                else:
                    incoming_values = tuple(instruction.args)
                if len(incoming_blocks) != len(incoming_values):
                    shortfalls.append(LLVMEmissionShortfall(
                        name, operation,
                        "phi incoming blocks do not match incoming values",
                    ))
                    continue
                register = f"%phi.{result_id}"
                body.append(
                    f"  {register} = phi ptr "
                    + ", ".join(
                        f"[ {pointer(value)}, %{predecessor} ]"
                        for predecessor, value in zip(
                            incoming_blocks, incoming_values
                        )
                    )
                )
                pointers[result_id] = register
                continue

            if operation in {"Br", "br"}:
                target = str(instruction.attributes.get("target") or "")
                if target not in function.blocks:
                    shortfalls.append(LLVMEmissionShortfall(
                        name, operation, f"unknown branch target {target!r}",
                    ))
                    continue
                body.append(f"  br label %{target}")
                continue

            if operation in {"CondBr", "condbr"} and instruction.args:
                true_target = str(
                    instruction.attributes.get("true")
                    or instruction.attributes.get("true_target")
                    or ""
                )
                false_target = str(
                    instruction.attributes.get("false")
                    or instruction.attributes.get("false_target")
                    or ""
                )
                if true_target not in function.blocks or false_target not in function.blocks:
                    shortfalls.append(LLVMEmissionShortfall(
                        name, operation, "conditional branch has an unknown target",
                    ))
                    continue
                condition = load_as(instruction.args[0], "i1", f"{tag}.condition")
                body.append(
                    f"  br i1 {condition}, label %{true_target}, label %{false_target}"
                )
                continue

            if operation in {"Ret", "ret", "Return", "return"}:
                emit_return_values()
                body.append("  ret void")
                emitted_return = True
                continue

            if operation in _SHAPE_ONLY and result is not None and instruction.args:
                source_pointer = pointer(instruction.args[0])
                destination = output_pointer.get(result_id)
                if destination is not None and destination != source_pointer:
                    body.append(
                        "  call void @llvm.memcpy.p0.p0.i64("
                        f"ptr {destination}, ptr {source_pointer}, "
                        f"i64 {_value_element_count(result) * 8}, i1 false)"
                    )
                    pointers[result_id] = destination
                else:
                    pointers[result_id] = source_pointer
                continue

            if operation in {"GetElementPtr", "getelementptr"} and result is not None:
                base_id = int(instruction.args[0].id) if instruction.args else -1
                members = aggregate_members.get(base_id)
                position = instruction.attributes.get("aggregate_index")
                if position is None and len(instruction.args) > 1:
                    position = constant_values.get(int(instruction.args[1].id))
                if members is not None and position is not None and int(position) in members:
                    address_members[result_id] = members[int(position)]
                    continue
                if (
                    position is not None
                    and instruction.args
                ):
                    slot = f"%aggregate.slot.{tag}"
                    body.append(
                        f"  {slot} = getelementptr ptr, ptr {pointer(instruction.args[0])}, i64 {int(position)}"
                    )
                    address_slots[result_id] = slot
                    continue

            if operation in {"Load", "load"} and result is not None and instruction.args:
                member = address_members.get(int(instruction.args[0].id))
                if member is not None:
                    pointers[result_id] = member
                    continue
                slot = address_slots.get(int(instruction.args[0].id))
                if slot is not None:
                    if _value_llvm_type(result) == "i64":
                        loaded_value = f"%reference.load.{tag}"
                        body.append(
                            f"  {loaded_value} = load i64, ptr {slot}, align 8"
                        )
                        body.append(
                            f"  store i64 {loaded_value}, ptr {pointer(result)}, align 8"
                        )
                        continue
                    loaded_pointer = f"%aggregate.load.{tag}"
                    body.append(
                        f"  {loaded_pointer} = load ptr, ptr {slot}, align 8"
                    )
                    pointers[result_id] = loaded_pointer
                    continue

            if operation in {"Store", "store"} and len(instruction.args) == 2:
                source, address = instruction.args
                destination = address_slots.get(int(address.id), pointer(address))
                source_type = _value_llvm_type(source)
                loaded_value = f"%store.load.{tag}"
                body.append(
                    f"  {loaded_value} = load {source_type}, ptr {pointer(source)}, align 8"
                )
                body.append(
                    f"  store {source_type} {loaded_value}, ptr {destination}, align 8"
                )
                continue

            callee = instruction.attributes.get("callee")
            if callee is not None:
                symbol = str(callee)
                try:
                    returns, argument_types = _kernel_signature(symbol)
                except (KeyError, ValueError):
                    returns = ""
                    argument_types = ()
                if argument_types or returns:
                    kernels_used.add(symbol)
                    arguments = list(instruction.args)
                    output_argument = instruction.attributes.get("ssa_output_argument")
                    if output_argument is not None and len(arguments) < len(argument_types):
                        arguments.insert(int(output_argument), result)
                    rendered: list[str] = []
                    for position, (argument_type, argument) in enumerate(zip(argument_types, arguments)):
                        rendered.append(
                            f"ptr {pointer(argument)}"
                            if argument_type == "ptr"
                            else f"{argument_type} {load_as(argument, argument_type, f'{tag}.{position}')}"
                        )
                    if len(rendered) != len(argument_types):
                        shortfalls.append(LLVMEmissionShortfall(
                            name, symbol, "authored call arity does not match its definition",
                        ))
                        continue
                    joined = ", ".join(rendered)
                    if returns == "void":
                        body.append(f"  call void @{symbol}({joined})")
                    elif result is not None:
                        call_result = f"%call.{tag}"
                        body.append(f"  {call_result} = call {returns} @{symbol}({joined})")
                        body.append(
                            f"  store {returns} {call_result}, ptr {pointer(result)}, align 8"
                        )
                    continue

                if symbol in reachable:
                    callee_outputs = function_outputs[symbol]
                    declared_ids = tuple(map(
                        int, instruction.attributes.get("output_ids", ())
                    ))
                    selected = aggregate_positions.get(
                        (name, result_id), tuple(range(len(callee_outputs)))
                    )
                    projections = projection_values.get(result_id, {})
                    forwarded = forwarded_aggregate_calls.get(name)
                    if (
                        forwarded is not None
                        and int(forwarded[0]) == result_id
                        and str(forwarded[1]) == symbol
                    ):
                        result_ptrs = [
                            f"%out.{index}"
                            for index in range(len(callee_outputs))
                        ]
                    elif declared_ids:
                        result_ptrs = [
                            pointer(projections[position])
                            for position in selected
                            if position in projections
                        ]
                    else:
                        if len(callee_outputs) == 1 and result is not None:
                            result_ptrs = [pointer(result)]
                        else:
                            result_ptrs = []
                            for output_index, value in enumerate(callee_outputs):
                                llvm_type = _value_llvm_type(value)
                                count = _value_element_count(value)
                                temporary = f"%call.output.{tag}.{output_index}"
                                body.append(
                                    f"  {temporary} = alloca {llvm_type}, i64 {count}, align 8"
                                )
                                result_ptrs.append(temporary)
                    call_args = [pointer(argument) for argument in instruction.args]
                    if len(result_ptrs) != len(callee_outputs):
                        shortfalls.append(LLVMEmissionShortfall(
                            name, symbol,
                            "live aggregate projections do not match callee outputs",
                        ))
                        continue
                    body.append(
                        f"  call void @{internal_symbols[symbol]}("
                        + ", ".join(f"ptr {value}" for value in (*call_args, *result_ptrs))
                        + ")"
                    )
                    if result is not None:
                        if forwarded is not None and int(forwarded[0]) == result_id:
                            # The call already wrote the wrapper's public
                            # outputs.  Its aggregate result has no independent
                            # storage and must not be reconstructed from local
                            # pointers.
                            pass
                        elif len(callee_outputs) == 1 and not declared_ids:
                            pointers[result_id] = result_ptrs[0]
                        else:
                            aggregate_members[result_id] = {
                                original_position: result_ptrs[index]
                                for index, original_position in enumerate(selected)
                                if index < len(result_ptrs)
                            }
                            if not declared_ids:
                                aggregate = f"%aggregate.{tag}"
                                body.append(
                                    f"  {aggregate} = alloca ptr, i64 {len(result_ptrs)}, align 8"
                                )
                                for output_index, result_pointer in enumerate(result_ptrs):
                                    slot = f"%aggregate.output.slot.{tag}.{output_index}"
                                    body.append(
                                        f"  {slot} = getelementptr ptr, ptr {aggregate}, i64 {output_index}"
                                    )
                                    body.append(
                                        f"  store ptr {result_pointer}, ptr {slot}, align 8"
                                    )
                                pointers[result_id] = aggregate
                    continue

            template = scalar_likeness(operation)
            if template is not None and result is not None:
                result_type = _value_llvm_type(result)
                operand_type = (
                    _value_llvm_type(instruction.args[0])
                    if instruction.args else result_type
                )
                operands = [
                    load_as(argument, operand_type, f"{tag}.{position}")
                    for position, argument in enumerate(instruction.args)
                ]
                register = f"%scalar.{tag}"
                if operand_type in {"i1", "i32", "i64"}:
                    integer_binary = {
                        "Add": "add", "Sub": "sub", "Mul": "mul",
                        "Div": "sdiv", "Mod": "srem",
                        "And": "and", "Or": "or", "Xor": "xor",
                        "BitAnd": "and", "BitOr": "or", "BitXor": "xor",
                        "Shl": "shl", "Shr": "lshr",
                    }
                    integer_comparison = {
                        "Eq": "eq", "Ne": "ne", "Lt": "slt",
                        "Le": "sle", "Gt": "sgt", "Ge": "sge",
                        "ULt": "ult", "ULe": "ule",
                    }
                    if operation in integer_binary and len(operands) == 2:
                        body.append(
                            f"  {register} = {integer_binary[operation]} "
                            f"{operand_type} {operands[0]}, {operands[1]}"
                        )
                    elif operation in integer_comparison and len(operands) == 2:
                        body.append(
                            f"  {register} = icmp {integer_comparison[operation]} "
                            f"{operand_type} {operands[0]}, {operands[1]}"
                        )
                        result_type = "i1"
                    elif operation == "Neg" and len(operands) == 1:
                        body.append(
                            f"  {register} = sub {operand_type} 0, {operands[0]}"
                        )
                    elif operation == "Not" and len(operands) == 1:
                        body.append(
                            f"  {register} = xor {operand_type} {operands[0]}, 1"
                        )
                    else:
                        shortfalls.append(LLVMEmissionShortfall(
                            name, operation,
                            f"integer scalar operation has no LLVM emission for {operand_type}",
                        ))
                        continue
                else:
                    for rendered_line in template.format(
                        *operands, out=register
                    ).splitlines():
                        body.append(f"  {rendered_line}")
                    if operation in {"Eq", "Ne", "Lt", "Le", "Gt", "Ge"}:
                        result_type = "i1"
                body.append(
                    f"  store {result_type} {register}, ptr {pointer(result)}, align 8"
                )
                continue

            # Descriptor getters can be present as dead planned outputs after
            # call specialization. They are omitted only when no selected ABI
            # result or live instruction consumes them.
            if operation == "getattr" and result is not None and result_id not in output_pointer:
                continue
            shortfalls.append(LLVMEmissionShortfall(
                name, operation, "operation has no repository LLVM emission",
            ))

        if not emitted_return:
            emit_return_values()
            body.append("  ret void")
        if not any(line.endswith(":") for line in body):
            body.insert(0, "entry:")
        entry_label_index = next(
            (index for index, line in enumerate(body) if line.endswith(":")),
            0,
        )
        body[entry_label_index + 1:entry_label_index + 1] = entry_allocas
        emitted_functions.append("\n".join((
            f"define void @{internal_symbols[name]}({', '.join(parameters)}) {{",
            *body,
            "}",
        )))

    root = module.functions[function_name]
    root_outputs = function_outputs[function_name]
    public_values = [*root.args, *root_outputs]
    buffer_order: list[int] = []
    buffer_shapes: list[tuple[_Any, ...]] = []
    public_pointer: dict[int, str] = {}
    wrapper: list[str] = ["entry:"]
    for value in public_values:
        value_id = int(value.id)
        if value_id in public_pointer:
            continue
        slot = len(buffer_order)
        buffer_order.append(value_id)
        buffer_shapes.append(tuple(value.shape or ()))
        address = f"%public.addr.{slot}"
        loaded = f"%public.{slot}"
        wrapper.append(f"  {address} = getelementptr ptr, ptr %buffers, i64 {slot}")
        wrapper.append(f"  {loaded} = load ptr, ptr {address}, align 8")
        public_pointer[value_id] = loaded
    wrapper.append(
        f"  call void @{internal_symbols[function_name]}("
        + ", ".join(
            f"ptr {public_pointer[int(value.id)]}"
            for value in (*root.args, *root_outputs)
        )
        + ")"
    )
    wrapper.append("  ret void")

    definitions: dict[str, str] = {}
    declarations: dict[str, str] = {
        "llvm.memcpy.p0.p0.i64": (
            "declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)"
        )
    }
    unresolved: set[str] = set()
    pending_kernels = set(kernels_used)
    while pending_kernels:
        symbol = pending_kernels.pop()
        if symbol in definitions or symbol in declarations:
            continue
        try:
            definition = extract_llvm_function(symbol)
        except KeyError:
            try:
                declarations[symbol] = extract_llvm_declaration(symbol)
            except KeyError:
                unresolved.add(symbol)
                shortfalls.append(LLVMEmissionShortfall(
                    function_name, symbol,
                    "referenced LLVM symbol has no authored definition or declaration",
                ))
            continue
        definitions[symbol] = definition
        for dependency in _re.findall(r"@([A-Za-z_$.-][\w$.-]*)\s*\(", definition):
            if dependency != symbol:
                pending_kernels.add(dependency)

    llvm_ir = "\n\n".join(part for part in (
        f'source_filename = "turing.ssa-llvm.{entry_name}"',
        "\n".join(declarations[symbol] for symbol in sorted(declarations)),
        "\n\n".join(definitions[symbol] for symbol in sorted(definitions)),
        "\n\n".join(emitted_functions),
        "\n".join((
            f"define void @{entry_name}(ptr %buffers, ptr %extents) {{",
            *wrapper,
            "}",
        )),
    ) if part)
    return LLVMFunctionArtifact(
        name=entry_name,
        llvm_ir=llvm_ir + "\n",
        buffer_order=tuple(buffer_order),
        buffer_shapes=tuple(buffer_shapes),
        extent_order=(),
        shortfalls=tuple(shortfalls),
        needs_text_sink=bool(text_sink),
    )


@_dataclass
class LLVMFunctionArtifact:
    """One SSA function emitted through the likeness table."""

    name: str
    llvm_ir: str
    buffer_order: tuple[int, ...]
    buffer_shapes: tuple[tuple[_Any, ...], ...]
    extent_order: tuple[tuple[int, str, int | None], ...]
    shortfalls: tuple[LLVMEmissionShortfall, ...]
    needs_text_sink: bool = False
    library_path: _Path | None = None
    _entry: _Any = _field(default=None, repr=False)

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    def entry(self):
        if self.library_path is None:
            raise RuntimeError("artifact was not compiled")
        if self._entry is None:
            library = _ctypes.CDLL(str(self.library_path))
            function = getattr(library, self.name)
            function.restype = None
            function.argtypes = [
                _ctypes.POINTER(_ctypes.c_void_p),
                _ctypes.POINTER(_ctypes.c_int32),
            ]
            self._entry = function
        return self._entry


def emit_ssa_function_to_llvm(
    module: _IRModule, function_name: str, *, entry_name: str | None = None,
    text_sink: bool = False,
) -> LLVMFunctionArtifact:
    """Render one SSA function of table-covered instructions as LLVM IR.

    ``text_sink`` states the target's publication capability. A shell-class
    target links ``turing_stream_buffer.c`` and takes publications as calls
    into it; a bare native artifact has no sink, so publications are elided
    -- they are never load-bearing for the numerics, and the same SSA runs
    either way.
    """

    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        extract_llvm_declaration,
        extract_llvm_function,
    )

    repository_closure, _authored_leaves = _internal_call_closure(
        module, function_name
    )
    if (
        len(repository_closure) > 1
        or len(module.functions[function_name].blocks) > 1
    ):
        return _emit_repository_call_module(
            module,
            function_name,
            entry_name=entry_name or function_name,
            text_sink=text_sink,
        )

    function = module.functions[function_name]
    name = entry_name or function_name
    shortfalls: list[LLVMEmissionShortfall] = []
    lines: list[str] = []
    globals_out: list[str] = []
    buffer_ids: list[int] = []
    buffer_index: dict[int, int] = {}
    extent_order: list[tuple[int, str, int | None]] = []
    scalars: dict[int, tuple[str, str]] = {}   # value id -> (rendering, type)
    kernels_used: set[str] = set()
    publishes_text = False
    value_shapes: dict[int, tuple[_Any, ...]] = {
        int(argument.id): tuple(argument.shape or ())
        for argument in function.args
    }
    for block in function.blocks.values():
        for instruction in block.instrs:
            for argument in instruction.args:
                value_shapes.setdefault(
                    int(argument.id), tuple(argument.shape or ()),
                )
            if instruction.res is not None:
                value_shapes[int(instruction.res.id)] = tuple(
                    instruction.res.shape or ()
                )

    def buffer(value_id: int) -> str:
        # The instruction stream is already scheduled by the compiler; a
        # buffer pointer is loaded inline at its first use, in stream order,
        # never collected and reordered.
        value_id = int(value_id)
        if value_id not in buffer_index:
            index = len(buffer_ids)
            buffer_index[value_id] = index
            buffer_ids.append(value_id)
            lines.append(
                f"  %buffer.addr.{value_id} = getelementptr ptr, ptr %buffers, "
                f"i64 {index}"
            )
            lines.append(
                f"  %buffer.{value_id} = load ptr, ptr %buffer.addr.{value_id}, "
                "align 8"
            )
        return f"%buffer.{value_id}"

    def as_type(value_id: int, wanted: str, tag: str) -> str | None:
        known = scalars.get(int(value_id))
        if known is None:
            return None
        rendering, kind = known
        if kind == wanted:
            return rendering
        if wanted == "double" and kind == "i32":
            if not rendering.startswith("%"):
                return _double_literal(float(int(rendering)))
            register = f"%conv.{tag}"
            lines.append(f"  {register} = sitofp i32 {rendering} to double")
            return register
        if wanted == "i32" and kind == "double" and not rendering.startswith("%"):
            return str(int(float.fromhex(rendering)))
        return None

    for block in function.blocks.values():
        for instruction in block.instrs:
            operation = instruction.op
            result_id = int(instruction.res.id) if instruction.res is not None else None

            if operation == "extent":
                kind = str(instruction.attributes.get("extent_kind"))
                axis = instruction.attributes.get("axis")
                slot = len(extent_order)
                extent_order.append((
                    int(instruction.args[0].id), kind,
                    int(axis) if axis is not None else None,
                ))
                address = f"%extent.addr.{slot}"
                lines.append(
                    f"  {address} = getelementptr i32, ptr %extents, i64 {slot}"
                )
                if kind == "shape":
                    scalars[result_id] = (address, "ptr")
                else:
                    register = f"%extent.{slot}"
                    lines.append(f"  {register} = load i32, ptr {address}, align 4")
                    scalars[result_id] = (register, "i32")
                continue

            if operation == "Const":
                payload = instruction.attributes.get("constant")
                if payload is None:
                    payload = instruction.attributes.get("values")
                if payload is None and "value" in instruction.attributes:
                    payload = instruction.attributes.get("value")
                if isinstance(payload, (tuple, list)):
                    symbol = f"@const.vec.{result_id}"
                    elements = ", ".join(f"i32 {int(item)}" for item in payload)
                    globals_out.append(
                        f"{symbol} = private constant [{len(payload)} x i32] [{elements}]"
                    )
                    scalars[result_id] = (symbol, "ptr")
                elif isinstance(payload, float) and not payload.is_integer():
                    scalars[result_id] = (_double_literal(payload), "double")
                elif payload is not None:
                    scalars[result_id] = (str(int(payload)), "i32")
                else:
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, "Const", "constant without payload",
                    ))
                continue

            if operation == "StaticRef":
                scalars[result_id] = (
                    str(int(instruction.attributes["reference_handle"])),
                    "i64",
                )
                continue

            if operation in {"GetElementPtr", "getelementptr"} and (
                instruction.res is not None and len(instruction.args) >= 2
            ):
                base = instruction.args[0]
                base_value = scalars.get(int(base.id))
                base_pointer = (
                    base_value[0]
                    if base_value is not None and base_value[1] == "ptr"
                    else buffer(int(base.id))
                )
                index = as_type(
                    int(instruction.args[1].id), "i32", f"gep.{result_id}"
                )
                if index is None:
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, operation,
                        "address index is not an emitted integer scalar",
                    ))
                    continue
                address = f"%address.{result_id}"
                lines.append(
                    f"  {address} = getelementptr i64, ptr {base_pointer}, i32 {index}"
                )
                scalars[result_id] = (address, "ptr")
                continue

            if operation in {"Store", "store"} and len(instruction.args) == 2:
                source, address = instruction.args
                stored = scalars.get(int(source.id))
                destination = scalars.get(int(address.id))
                if stored is None or destination is None or destination[1] != "ptr":
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, operation,
                        "store source or destination address has no emitted producer",
                    ))
                    continue
                lines.append(
                    f"  store {stored[1]} {stored[0]}, ptr {destination[0]}, align 8"
                )
                continue

            if operation in {"Load", "load"} and (
                instruction.res is not None and len(instruction.args) == 1
            ):
                address = scalars.get(int(instruction.args[0].id))
                if address is None or address[1] != "ptr":
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, operation,
                        "load address has no emitted pointer producer",
                    ))
                    continue
                llvm_type = _value_llvm_type(instruction.res)
                register = f"%load.{result_id}"
                lines.append(
                    f"  {register} = load {llvm_type}, ptr {address[0]}, align 8"
                )
                scalars[result_id] = (register, llvm_type)
                continue

            if operation in {"Ret", "ret", "Return", "return"}:
                for position, argument in enumerate(instruction.args):
                    value_id = int(argument.id)
                    known = scalars.get(value_id)
                    if known is None or known[1] == "ptr":
                        # Tensor-producing kernels already write to this
                        # buffer. Calling buffer() here also exposes an output
                        # that otherwise had no downstream consumer.
                        buffer(value_id)
                        continue
                    rendering = as_type(
                        value_id, "double", f"return.{position}"
                    )
                    if rendering is None:
                        shortfalls.append(LLVMEmissionShortfall(
                            function_name, "return",
                            f"output %t{value_id} cannot render as double",
                        ))
                        continue
                    destination = buffer(value_id)
                    lines.append(
                        f"  store double {rendering}, ptr {destination}, align 8"
                    )
                continue

            callee = instruction.attributes.get("callee")
            if (
                operation == "stream_publish"
                or callee == "turing_stream_publish"
            ):
                if not text_sink:
                    continue        # no sink on this target: elide
                payload = instruction.args[0] if instruction.args else None
                if payload is None:
                    continue
                value_id = int(payload.id)
                known = scalars.get(value_id)
                if known is not None and known[1] == "double":
                    rendered = known[0]
                else:
                    pointer = buffer(value_id)
                    rendered = f"%publish.{value_id}.{len(lines)}"
                    lines.append(
                        f"  {rendered} = load double, ptr {pointer}, align 8"
                    )
                stream_id = int(instruction.attributes.get("stream_id", 0))
                final = 1 if instruction.attributes.get("final") else 0
                lines.append(
                    f"  call void @turing_stream_publish_double("
                    f"i32 {stream_id}, double {rendered}, i32 {final})"
                )
                publishes_text = True
                continue
            if callee is not None:
                symbol = str(callee)
                try:
                    returns, argument_types = _kernel_signature(symbol)
                except ValueError as error:
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, symbol, str(error),
                    ))
                    continue
                kernels_used.add(symbol)
                arguments = list(instruction.args)
                output_argument = instruction.attributes.get("ssa_output_argument")
                if output_argument is not None and len(arguments) < len(argument_types):
                    arguments.insert(int(output_argument), instruction.res)
                if len(arguments) != len(argument_types):
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, symbol,
                        f"call has {len(arguments)} operands for "
                        f"{len(argument_types)} parameters",
                    ))
                    continue
                rendered: list[str] = []
                trouble: str | None = None
                for position, (argument_type, argument) in enumerate(
                    zip(argument_types, arguments)
                ):
                    value_id = int(argument.id)
                    if argument_type == "ptr":
                        known = scalars.get(value_id)
                        if known is not None and known[1] == "ptr":
                            rendered.append(f"ptr {known[0]}")
                        else:
                            rendered.append(f"ptr {buffer(value_id)}")
                    else:
                        rendering = as_type(
                            value_id, argument_type,
                            f"{result_id}.{position}",
                        )
                        if rendering is None and argument_type == "double":
                            pointer = buffer(value_id)
                            register = f"%load.{value_id}.{len(lines)}"
                            lines.append(
                                f"  {register} = load double, ptr {pointer}, align 8"
                            )
                            rendering = register
                        if rendering is None:
                            trouble = (
                                f"operand %t{value_id} cannot render as "
                                f"{argument_type}"
                            )
                            break
                        rendered.append(f"{argument_type} {rendering}")
                if trouble is not None:
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, symbol, trouble,
                    ))
                    continue
                joined = ", ".join(rendered)
                if returns == "void":
                    lines.append(f"  call void @{symbol}({joined})")
                else:
                    register = f"%call.{result_id}"
                    lines.append(
                        f"  {register} = call {returns} @{symbol}({joined})"
                    )
                    scalars[result_id] = (register, returns)
                    destination = buffer(result_id)
                    lines.append(
                        f"  store {returns} {register}, ptr {destination}, align 8"
                    )
                continue

            template = scalar_likeness(str(operation))
            if template is not None:
                operands = []
                trouble = None
                for position, argument in enumerate(instruction.args):
                    rendering = as_type(
                        int(argument.id), "double", f"{result_id}.{position}"
                    )
                    if rendering is None:
                        trouble = (
                            f"scalar operand %t{int(argument.id)} unavailable"
                        )
                        break
                    operands.append(rendering)
                if trouble is not None:
                    shortfalls.append(LLVMEmissionShortfall(
                        function_name, str(operation), trouble,
                    ))
                    continue
                register = f"%scalar.{result_id}"
                for line in template.format(
                    *operands, out=register
                ).splitlines():
                    lines.append(f"  {line}")
                scalars[result_id] = (register, "double")
                continue

            shortfalls.append(LLVMEmissionShortfall(
                function_name, str(operation),
                "operation has no likeness-table entry",
            ))

    # Authored kernels can call other authored helpers as well as external
    # math/intrinsic symbols.  Carry their definition closure and exact
    # canonical declarations into this otherwise standalone module.
    definitions = {
        symbol: extract_llvm_function(symbol) for symbol in kernels_used
    }
    external_declarations: dict[str, str] = {}
    unresolved_symbols: set[str] = set()
    while True:
        dependency_text = "\n".join((*definitions.values(), *lines))
        referenced = set(_re.findall(
            r"@([A-Za-z_$.-][\w$.-]*)\s*\(", dependency_text,
        ))
        pending = referenced - set(definitions) - set(external_declarations)
        pending.discard("turing_stream_publish_double")
        pending -= unresolved_symbols
        if not pending:
            break
        for symbol in sorted(pending):
            try:
                definitions[symbol] = extract_llvm_function(symbol)
                continue
            except KeyError:
                pass
            try:
                external_declarations[symbol] = extract_llvm_declaration(symbol)
            except KeyError:
                unresolved_symbols.add(symbol)
                shortfalls.append(LLVMEmissionShortfall(
                    function_name, symbol,
                    "referenced LLVM symbol has no authored definition or declaration",
                ))

    kernel_texts = "\n\n".join(
        definitions[symbol] for symbol in sorted(definitions)
    )
    declarations = [
        external_declarations[symbol]
        for symbol in sorted(external_declarations)
    ]
    if publishes_text:
        # Resolved by linking turing_stream_buffer.c -- the shell-class sink.
        declarations.append(
            "declare void @turing_stream_publish_double(i32, double, i32)"
        )
    llvm_ir = "\n".join((
        f'source_filename = "turing.ssa-llvm.{name}"',
        *globals_out,
        *declarations,
        "",
        kernel_texts,
        "",
        f"define void @{name}(ptr %buffers, ptr %extents) {{",
        "entry:",
        *lines,
        "  ret void",
        "}",
        "",
    ))
    return LLVMFunctionArtifact(
        name=name,
        llvm_ir=llvm_ir,
        buffer_order=tuple(buffer_ids),
        buffer_shapes=tuple(value_shapes.get(value_id, ()) for value_id in buffer_ids),
        extent_order=tuple(extent_order),
        shortfalls=tuple(shortfalls),
        needs_text_sink=publishes_text,
    )


def compile_artifact(
    artifact: LLVMFunctionArtifact, *, directory: _Path | None = None,
) -> LLVMFunctionArtifact:
    """Build the emitted module with the LLVM compiler, ahead of time."""

    if not artifact.complete:
        raise ValueError(
            "artifact has shortfalls: "
            + "; ".join(s.reason for s in artifact.shortfalls[:5])
        )
    build_dir = _Path(directory) if directory is not None else _Path(
        _tempfile.mkdtemp(prefix=f"ssa_llvm_{artifact.name}_")
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    source = build_dir / f"{artifact.name}.ll"
    source.write_text(artifact.llvm_ir, encoding="utf-8")
    library = build_dir / f"{artifact.name}.dll"
    # Same LLVM toolchain resolution the C backend uses: the ziglang package
    # bundles clang, invoked through the interpreter, no PATH assumptions.
    import sys as _sys
    command = [_sys.executable, "-m", "ziglang", "cc", "-shared", "-O2",
               "-o", str(library), str(source)]
    if artifact.needs_text_sink:
        command.append(str(
            _Path(__file__).resolve().parents[1]
            / "common" / "tensors" / "accelerator_backends" / "c_backend"
            / "turing_stream_buffer.c"
        ))
    completed = _subprocess.run(
        command, capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0 or not library.is_file():
        raise RuntimeError(
            f"LLVM compile failed ({completed.returncode}):\n"
            + completed.stderr[-2000:]
        )
    artifact.library_path = library
    return artifact
