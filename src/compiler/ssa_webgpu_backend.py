"""Emit WebGPU WGSL compute shaders directly from Turing SSA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue


_BINARY: dict[str, str] = {
    "Add": "({0} + {1})", "Sub": "({0} - {1})",
    "Mul": "({0} * {1})", "Div": "({0} / {1})",
    "Mod": "({0} % {1})", "Pow": "pow({0}, {1})",
    "FloorDiv": "floor({0} / {1})",
    "Eq": "f32({0} == {1})", "Ne": "f32({0} != {1})",
    "Lt": "f32({0} < {1})", "Le": "f32({0} <= {1})",
    "Gt": "f32({0} > {1})", "Ge": "f32({0} >= {1})",
    "LAnd": "f32(bool({0}) && bool({1}))",
    "LOr": "f32(bool({0}) || bool({1}))",
    "And": "bitcast<f32>(bitcast<u32>({0}) & bitcast<u32>({1}))",
    "Or": "bitcast<f32>(bitcast<u32>({0}) | bitcast<u32>({1}))",
    "Xor": "bitcast<f32>(bitcast<u32>({0}) ^ bitcast<u32>({1}))",
    "Shl": "bitcast<f32>(bitcast<u32>({0}) << bitcast<u32>({1}))",
    "Shr": "bitcast<f32>(bitcast<u32>({0}) >> bitcast<u32>({1}))",
    "add": "({0} + {1})", "sub": "({0} - {1})",
    "mul": "({0} * {1})", "truediv": "({0} / {1})",
    "mod": "({0} % {1})", "pow": "pow({0}, {1})",
    "floordiv": "floor({0} / {1})",
    "maximum": "max({0}, {1})", "minimum": "min({0}, {1})",
    "equal": "f32({0} == {1})", "not_equal": "f32({0} != {1})",
    "less": "f32({0} < {1})", "less_equal": "f32({0} <= {1})",
    "greater": "f32({0} > {1})", "greater_equal": "f32({0} >= {1})",
    "logical_and": "f32(bool({0}) && bool({1}))",
    "logical_or": "f32(bool({0}) || bool({1}))",
    "bitand": "bitcast<f32>(bitcast<u32>({0}) & bitcast<u32>({1}))",
    "bitor": "bitcast<f32>(bitcast<u32>({0}) | bitcast<u32>({1}))",
    "bitxor": "bitcast<f32>(bitcast<u32>({0}) ^ bitcast<u32>({1}))",
}

_UNARY: dict[str, str] = {
    "Neg": "(-{0})", "Abs": "abs({0})", "LNot": "f32(!bool({0}))",
    "Not": "bitcast<f32>(~bitcast<u32>({0}))",
    "neg": "(-{0})", "abs": "abs({0})", "sqrt": "sqrt({0})",
    "exp": "exp({0})", "log": "log({0})", "sin": "sin({0})",
    "cos": "cos({0})", "tan": "tan({0})", "asin": "asin({0})",
    "acos": "acos({0})", "atan": "atan({0})", "sinh": "sinh({0})",
    "cosh": "cosh({0})", "tanh": "tanh({0})", "floor": "floor({0})",
    "ceil": "ceil({0})", "round": "round({0})", "trunc": "trunc({0})",
    "sign": "sign({0})", "logical_not": "f32(!bool({0}))",
    "invert": "bitcast<f32>(~bitcast<u32>({0}))", "copy": "{0}",
    "bool_to_float64": "f32({0})",
}

_REDUCTION: dict[str, str] = {}
_SHAPE_ONLY = frozenset({"reshape", "view"})
_DTYPE: dict[str, str] = {
    "float32": "f32", "float": "f32", "f32": "f32",
    "int32": "i32", "int": "i32", "i32": "i32",
    "uint32": "u32", "u32": "u32", "bool": "bool",
}
_FLOAT64 = frozenset({"float64", "double", "f64"})
_COMPARISON = {
    "Eq": "{0} == {1}", "Ne": "{0} != {1}",
    "Lt": "{0} < {1}", "Le": "{0} <= {1}",
    "Gt": "{0} > {1}", "Ge": "{0} >= {1}",
    "equal": "{0} == {1}", "not_equal": "{0} != {1}",
    "less": "{0} < {1}", "less_equal": "{0} <= {1}",
    "greater": "{0} > {1}", "greater_equal": "{0} >= {1}",
}


def supported_tensor_operations() -> frozenset[str]:
    from ..common.tensors.operator_catalog import CANONICAL_ABSTRACT_TENSOR_OPERATORS

    registered = frozenset(_BINARY) | frozenset(_UNARY) | _SHAPE_ONLY
    return frozenset(registered & CANONICAL_ABSTRACT_TENSOR_OPERATORS)


@dataclass(frozen=True)
class WGSLComputeLimits:
    max_workgroup_size: tuple[int, int, int] = (256, 256, 64)
    max_invocations_per_workgroup: int = 256
    max_workgroups_per_dimension: int = 65535


@dataclass(frozen=True)
class WGSLLaunchPlan:
    count: int
    workgroup_size: tuple[int, int, int]
    groups: tuple[int, int, int]
    limits: WGSLComputeLimits = WGSLComputeLimits()

    @property
    def skipped(self) -> bool:
        return self.count == 0


def plan_wgsl_launch(
    count: int, *, preferred_local_size: int = 256,
) -> WGSLLaunchPlan:
    count = int(count)
    if count < 0:
        raise ValueError("launch count cannot be negative")
    if preferred_local_size <= 0:
        raise ValueError("preferred local size must be positive")
    limits = WGSLComputeLimits()
    cap = min(
        preferred_local_size,
        limits.max_workgroup_size[0],
        limits.max_invocations_per_workgroup,
    )
    local = 1 << (int(cap).bit_length() - 1)
    if count:
        local = min(local, max(min(32, local), 1 << (count - 1).bit_length()))
    if count == 0:
        return WGSLLaunchPlan(0, (local, 1, 1), (0, 0, 0), limits)
    needed = (count + local - 1) // local
    group_x = min(needed, limits.max_workgroups_per_dimension)
    remaining = (needed + group_x - 1) // group_x
    group_y = min(remaining, limits.max_workgroups_per_dimension)
    group_z = (remaining + group_y - 1) // group_y
    if group_z > limits.max_workgroups_per_dimension:
        capacity = limits.max_workgroups_per_dimension ** 3 * local
        raise ValueError(
            f"launch count {count} exceeds one-dispatch capacity {capacity}"
        )
    return WGSLLaunchPlan(count, (local, 1, 1), (group_x, group_y, group_z), limits)


@dataclass(frozen=True)
class WGSLShortfall:
    function: str
    operation: str
    reason: str

    def format(self) -> str:
        return f"{self.function}: {self.operation}: {self.reason}"


@dataclass
class WGSLModule:
    name: str
    source: str
    complete: bool
    shortfalls: tuple[WGSLShortfall, ...]
    api: Any
    launch_plan: WGSLLaunchPlan
    io_layout: Any = None

    def write(self, directory: str | Path) -> Path:
        path = Path(directory) / f"{self.name}.wgsl"
        path.write_text(self.source, encoding="utf-8", newline="\n")
        if self.api is not None:
            self.api.write(path.with_suffix(".api.yaml"))
        return path


def _op(instr: Instr) -> str:
    return str(instr.attributes.get("tensor_operation") or instr.op)


def _name(value: SSAValue) -> str:
    return f"v_{value.id}"


def _literal(value: Any, dtype: str = "f32") -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if dtype == "u32":
        return f"{int(value)}u"
    if dtype == "i32":
        return f"{int(value)}i"
    text = repr(float(value))
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return f"{text}f"


class _FunctionEmitter:
    def __init__(
        self,
        function: Function,
        *,
        outputs: Sequence[SSAValue],
        loop_records: Sequence[Mapping[str, Any]],
        callees: Mapping[str, tuple[SSAValue, ...]] | None = None,
        store_outputs: bool = True,
    ):
        self.function = function
        self.outputs = tuple(outputs)
        self.loops = tuple(loop_records)
        self.callees = dict(callees or {})
        self.store_outputs = store_outputs
        self.shortfalls: list[WGSLShortfall] = []
        self.lines: list[str] = []
        self.declared: set[int] = set()
        self.aliases: dict[int, str] = {}
        self.output_bindings = {value.id: index for index, value in enumerate(outputs)}

    def fail(self, operation: str, reason: str) -> None:
        item = WGSLShortfall(self.function.name, operation, reason)
        if item not in self.shortfalls:
            self.shortfalls.append(item)

    def dtype(self, value: SSAValue) -> str | None:
        dtype = str(value.dtype or "float32").lower()
        if dtype in _FLOAT64:
            self.fail(f"value %{value.id}", "float64 has no WebGPU core equivalent")
            return None
        result = _DTYPE.get(dtype)
        if result is None and dtype not in {"ptr", "ssa.aggregate"}:
            self.fail(f"value %{value.id}", f"unsupported WGSL dtype {dtype!r}")
        return result

    def operand(self, value: SSAValue) -> str:
        return self.aliases.get(value.id, _name(value))

    def emit_assignment(self, instr: Instr, indent: str) -> None:
        operation = _op(instr)
        result = instr.res
        if result is None:
            self.fail(operation, "instruction has no result")
            return
        dtype = self.dtype(result)
        args = [self.operand(value) for value in instr.args]
        if "right_scalar" in instr.attributes:
            scalar = _literal(instr.attributes["right_scalar"], dtype)
            args = [*args, scalar]
            if instr.attributes.get("reverse"):
                args.reverse()
        expression: str | None = None
        if operation in _COMPARISON and len(args) == 2:
            comparison = _COMPARISON[operation].format(*args)
            expression = (
                comparison if dtype == "bool"
                else f"select(0.0f, 1.0f, {comparison})"
            )
        elif operation in _BINARY and len(args) == 2:
            expression = _BINARY[operation].format(*args)
        elif operation in _UNARY and len(args) == 1:
            expression = _UNARY[operation].format(*args)
        elif operation in _SHAPE_ONLY and len(args) == 1:
            expression = args[0]
        elif operation in {"Const", "const"}:
            constant = instr.attributes.get("constant")
            if constant is None:
                constant = instr.attributes.get("value")
            if constant is None and "values" in instr.attributes:
                values = tuple(instr.attributes["values"])
                if len(values) == 1:
                    constant = values[0]
            if constant is not None:
                expression = _literal(constant, dtype or "f32")
        elif operation in {"Call", "call"}:
            callee = str(instr.attributes.get("callee", ""))
            returns = self.callees.get(callee)
            if returns is None:
                self.fail(operation, f"unknown WGSL callee {callee!r}")
                return
            if len(returns) != 1:
                self.fail(operation, "multi-output numerical regions are not yet supported")
                return
            dtype = self.dtype(returns[0])
            expression = f"{callee}({', '.join(args)})"
        elif operation in {"GetElementPtr", "getelementptr"} and args:
            self.aliases[result.id] = args[0]
            return
        elif operation in {"Load", "load"} and args:
            expression = args[0]
        elif operation in {"Select", "select", "where"} and len(args) == 3:
            expression = f"select({args[2]}, {args[1]}, bool({args[0]}))"
        if expression is None:
            self.fail(operation, "unsupported SSA operation for WGSL")
            return
        if dtype is None:
            return
        keyword = "let" if result.id not in self.declared else ""
        declaration = f"{keyword} {_name(result)}: {dtype}" if keyword else _name(result)
        self.lines.append(f"{indent}{declaration} = {expression};")
        self.declared.add(result.id)

    def emit_block(self, block: BasicBlock, indent: str, *, omit_control: bool = False) -> None:
        for instr in block.instrs:
            operation = str(instr.op)
            if operation in {"Phi", "phi"}:
                continue
            if operation in {"Br", "br", "CondBr", "condbr", "Ret", "ret", "Return", "return"}:
                if not omit_control:
                    self.fail(operation, "unsupported control-flow shape for structured WGSL lowering")
                continue
            self.emit_assignment(instr, indent)

    def phi_assignment(self, phi: Instr, predecessor: str, indent: str) -> None:
        incoming = tuple(phi.attributes.get("incoming_blocks", ()))
        if predecessor not in incoming:
            self.fail("Phi", "phi incoming block does not match structured predecessor")
            return
        value = phi.args[incoming.index(predecessor)]
        self.lines.append(f"{indent}{_name(phi.res)} = {_name(value)};")

    def emit_loop(self, record: Mapping[str, Any], indent: str) -> set[str]:
        blocks = self.function.blocks
        names = {str(record[key]) for key in ("preheader", "header", "body", "latch", "exit") if key in record}
        if len(names) != 5 or not names <= set(blocks):
            self.fail("loop", "unsupported control-flow shape for structured WGSL lowering")
            return names
        preheader, header, body, latch = (
            blocks[str(record[key])] for key in ("preheader", "header", "body", "latch")
        )
        self.emit_block(preheader, indent, omit_control=True)
        phis = [instr for instr in header.instrs if str(instr.op) in {"Phi", "phi"}]
        for phi in phis:
            dtype = self.dtype(phi.res)
            incoming = tuple(phi.attributes.get("incoming_blocks", ()))
            if dtype is None or preheader.name not in incoming:
                continue
            initial = phi.args[incoming.index(preheader.name)]
            self.lines.append(f"{indent}var {_name(phi.res)}: {dtype} = {_name(initial)};")
            self.declared.add(phi.res.id)
        header_work = [instr for instr in header.instrs if str(instr.op) not in {"Phi", "phi", "CondBr", "condbr"}]
        condition_instr = next((instr for instr in header.instrs if str(instr.op) in {"CondBr", "condbr"}), None)
        if condition_instr is None or not condition_instr.args:
            self.fail("loop", "unsupported control-flow shape for structured WGSL lowering")
            return names
        self.lines.append(f"{indent}loop {{")
        for instr in header_work:
            self.emit_assignment(instr, indent + "  ")
        self.lines.append(f"{indent}  break if (!{self.operand(condition_instr.args[0])});")
        self.emit_block(body, indent + "  ", omit_control=True)
        self.lines.append(f"{indent}  continuing {{")
        self.emit_block(latch, indent + "    ", omit_control=True)
        for phi in phis:
            self.phi_assignment(phi, latch.name, indent + "    ")
        self.lines.append(f"{indent}  }}")
        self.lines.append(f"{indent}}}")
        return names

    def emit_diamond(self, record: Mapping[str, str], indent: str) -> set[str]:
        blocks = self.function.blocks
        source, passed, failed, merge = (
            blocks[record[key]] for key in ("source", "true", "false", "merge")
        )
        branch = source.instrs[-1]
        phis = [
            instruction for instruction in merge.instrs
            if str(instruction.op) in {"Phi", "phi"}
        ]
        self.emit_block(source, indent, omit_control=True)
        for phi in phis:
            dtype = self.dtype(phi.res)
            incoming = tuple(phi.attributes.get("incoming_blocks", ()))
            if dtype is None or incoming != (passed.name, failed.name):
                self.fail("Phi", "diamond merge phi does not match branch predecessors")
                continue
            self.lines.append(f"{indent}var {_name(phi.res)}: {dtype};")
            self.declared.add(phi.res.id)
        self.lines.append(f"{indent}if ({self.operand(branch.args[0])}) {{")
        self.emit_block(passed, indent + "  ", omit_control=True)
        for phi in phis:
            self.phi_assignment(phi, passed.name, indent + "  ")
        self.lines.append(f"{indent}}} else {{")
        self.emit_block(failed, indent + "  ", omit_control=True)
        for phi in phis:
            self.phi_assignment(phi, failed.name, indent + "  ")
        self.lines.append(f"{indent}}}")
        self.emit_block(merge, indent, omit_control=True)
        return {source.name, passed.name, failed.name, merge.name}

    def emit(self) -> tuple[str, tuple[WGSLShortfall, ...]]:
        for value in self.function.args:
            self.dtype(value)
            self.declared.add(value.id)
        loop_blocks: set[str] = set()
        for record in self.loops:
            loop_blocks |= self.emit_loop(record, "  ")
        if not self.loops:
            blocks = list(self.function.blocks.values())
            if len(blocks) == 1:
                self.emit_block(blocks[0], "  ", omit_control=True)
            else:
                diamond = _canonical_diamond_record(self.function)
                if diamond is None:
                    self.fail("control", "unsupported control-flow shape for structured WGSL lowering")
                else:
                    self.emit_diamond(diamond, "  ")
        else:
            for block in self.function.blocks.values():
                if block.name not in loop_blocks and block.instrs:
                    self.emit_block(block, "  ", omit_control=True)
        if self.store_outputs:
            for value_id, binding in self.output_bindings.items():
                self.lines.append(
                    f"  output_{binding}[linear_index] = "
                    f"{self.aliases.get(value_id, f'v_{value_id}')};"
                )
        elif len(self.outputs) == 1:
            self.lines.append(f"  return {self.operand(self.outputs[0])};")
        return "\n".join(self.lines), tuple(self.shortfalls)


def _loop_records(module: IRModule, function_name: str) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    tables = [module.recursion_table, module.functions[function_name].metadata.get("recursion_table", {})]
    for table in tables:
        for region in table.values():
            if str(region.get("function", function_name)) == function_name:
                records.extend(region.get("loops", ()))
    unique: list[Mapping[str, Any]] = []
    for record in records:
        if record not in unique:
            unique.append(record)
    if not unique:
        unique.extend(_canonical_loop_records(module.functions[function_name]))
    return tuple(unique)


def _branch_target(block: BasicBlock) -> str | None:
    if not block.instrs or str(block.instrs[-1].op) not in {"Br", "br"}:
        return None
    return str(block.instrs[-1].attributes.get("target", ""))


def _canonical_loop_records(function: Function) -> tuple[Mapping[str, str], ...]:
    """Recognize only the exact structured loop contract emitted by lower_loop."""
    records: list[Mapping[str, str]] = []
    blocks = function.blocks
    for header in blocks.values():
        phis = [item for item in header.instrs if str(item.op) in {"Phi", "phi"}]
        branch = next(
            (item for item in header.instrs if str(item.op) in {"CondBr", "condbr"}),
            None,
        )
        if not phis or branch is None:
            continue
        incoming = tuple(phis[0].attributes.get("incoming_blocks", ()))
        if len(incoming) != 2 or any(
            tuple(phi.attributes.get("incoming_blocks", ())) != incoming
            for phi in phis
        ):
            continue
        preheader_name, latch_name = map(str, incoming)
        body_name = str(
            branch.attributes.get("true_target") or branch.attributes.get("true") or ""
        )
        exit_name = str(
            branch.attributes.get("false_target") or branch.attributes.get("false") or ""
        )
        names = {preheader_name, header.name, body_name, latch_name, exit_name}
        if len(names) != 5 or not names <= set(blocks):
            continue
        if (
            _branch_target(blocks[preheader_name]) != header.name
            or _branch_target(blocks[body_name]) != latch_name
            or _branch_target(blocks[latch_name]) != header.name
        ):
            continue
        records.append({
            "preheader": preheader_name,
            "header": header.name,
            "body": body_name,
            "latch": latch_name,
            "exit": exit_name,
        })
    return tuple(records)


def _canonical_diamond_record(function: Function) -> Mapping[str, str] | None:
    blocks = function.blocks
    if len(blocks) != 4:
        return None
    source = next(
        (
            block for block in blocks.values()
            if block.instrs
            and str(block.instrs[-1].op) in {"CondBr", "condbr"}
        ),
        None,
    )
    if source is None or not source.instrs[-1].args:
        return None
    branch = source.instrs[-1]
    passed_name = str(
        branch.attributes.get("true_target") or branch.attributes.get("true") or ""
    )
    failed_name = str(
        branch.attributes.get("false_target") or branch.attributes.get("false") or ""
    )
    if passed_name not in blocks or failed_name not in blocks:
        return None
    passed_target = _branch_target(blocks[passed_name])
    failed_target = _branch_target(blocks[failed_name])
    if not passed_target or passed_target != failed_target or passed_target not in blocks:
        return None
    merge = blocks[passed_target]
    phis = [
        instruction for instruction in merge.instrs
        if str(instruction.op) in {"Phi", "phi"}
    ]
    if any(
        tuple(phi.attributes.get("incoming_blocks", ()))
        != (passed_name, failed_name)
        for phi in phis
    ):
        return None
    return {
        "source": source.name,
        "true": passed_name,
        "false": failed_name,
        "merge": merge.name,
    }


def emit_module(
    module: IRModule | Mapping[str, Function],
    *,
    name: str = "turing_ssa",
    outputs: Mapping[str, Sequence[SSAValue]] | None = None,
    count: int = 1,
    preferred_local_size: int = 256,
) -> WGSLModule:
    ir_module = module if isinstance(module, IRModule) else IRModule(dict(module))
    functions = ir_module.functions
    named_outputs = dict(outputs or {})
    launch_plan = plan_wgsl_launch(count, preferred_local_size=preferred_local_size)
    shortfalls: list[WGSLShortfall] = []
    requested_entries = [key for key in named_outputs if key in functions]
    if requested_entries:
        function_name = requested_entries[0]
    elif len(functions) == 1:
        function_name = next(iter(functions))
    else:
        function_name = next(iter(functions), "")
        shortfalls.append(WGSLShortfall(
            name, "module", "multiple SSA functions require a named output entry",
        ))
    if not function_name:
        raise ValueError("WGSL compute emission requires at least one SSA function")
    function = functions[function_name]
    output_values = tuple(named_outputs.get(function_name, ()))
    if not output_values:
        shortfalls.append(WGSLShortfall(
            function_name, "outputs", "compute module has no named output",
        ))
    function_returns = {
        current_name: tuple(
            instruction.args
            for block in current.blocks.values()
            for instruction in block.instrs
            if str(instruction.op) in {"Ret", "ret", "Return", "return"}
        )[-1]
        for current_name, current in functions.items()
        if any(
            str(instruction.op) in {"Ret", "ret", "Return", "return"}
            for block in current.blocks.values()
            for instruction in block.instrs
        )
    }
    called_names = {
        str(instruction.attributes.get("callee", ""))
        for block in function.blocks.values()
        for instruction in block.instrs
        if str(instruction.op) in {"Call", "call"}
        and "tensor_operation" not in instruction.attributes
    }
    callees = {
        callee: function_returns.get(callee, ())
        for callee in called_names
        if callee in functions
    }
    from .shader_stages import COMPUTE, BufferBinding, ShaderIOLayout

    bindings = []
    feed_bindings = []
    for binding, value in enumerate(function.args):
        dtype = _DTYPE.get(str(value.dtype or "float32").lower(), "f32")
        bindings.append(
            f"@group(0) @binding({binding}) var<storage, read> feed_{value.id}: array<{dtype}>;"
        )
        feed_bindings.append(BufferBinding(
            f"feed_{value.id}", "feed", dtype, binding, value_id=value.id,
        ))
    output_bindings = []
    for index, value in enumerate(output_values):
        binding = len(function.args) + index
        dtype = _DTYPE.get(str(value.dtype or "float32").lower(), "f32")
        bindings.append(
            f"@group(0) @binding({binding}) var<storage, read_write> output_{index}: array<{dtype}>;"
        )
        output_bindings.append(BufferBinding(
            f"output_{index}", "output", dtype, binding, value_id=value.id,
        ))
    io_layout = ShaderIOLayout(
        COMPUTE.name, feeds=tuple(feed_bindings), outputs=tuple(output_bindings),
    )
    helpers: list[str] = []
    for callee, returns in callees.items():
        helper = functions[callee]
        if len(returns) != 1:
            shortfalls.append(WGSLShortfall(
                callee, "return", "multi-output numerical regions are not yet supported",
            ))
            continue
        return_dtype = _DTYPE.get(str(returns[0].dtype or "float32").lower())
        parameter_types = [
            _DTYPE.get(str(value.dtype or "float32").lower())
            for value in helper.args
        ]
        if return_dtype is None or any(item is None for item in parameter_types):
            shortfalls.append(WGSLShortfall(
                callee, "signature", "unsupported numerical-region WGSL dtype",
            ))
            continue
        helper_emitter = _FunctionEmitter(
            helper,
            outputs=returns,
            loop_records=_loop_records(ir_module, callee),
            callees=callees,
            store_outputs=False,
        )
        helper_body, helper_shortfalls = helper_emitter.emit()
        shortfalls.extend(helper_shortfalls)
        parameters = ", ".join(
            f"{_name(value)}: {dtype}"
            for value, dtype in zip(helper.args, parameter_types)
        )
        helpers.append(
            f"fn {callee}({parameters}) -> {return_dtype} {{\n{helper_body}\n}}"
        )
    emitter = _FunctionEmitter(
        function, outputs=output_values,
        loop_records=_loop_records(ir_module, function_name),
        callees=callees,
    )
    for value in function.args:
        dtype = _DTYPE.get(str(value.dtype or "float32").lower(), "f32")
        emitter.lines.append(
            f"  let {_name(value)}: {dtype} = feed_{value.id}[linear_index];"
        )
    body, function_shortfalls = emitter.emit()
    shortfalls.extend(function_shortfalls)
    local_x, local_y, local_z = launch_plan.workgroup_size
    source = "\n".join([
        *bindings,
        "",
        *helpers,
        "" if helpers else "",
        f"@compute @workgroup_size({local_x}, {local_y}, {local_z})",
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) grid: vec3<u32>) {",
        f"  let linear_index: u32 = gid.x + gid.y * grid.x * {local_x}u + gid.z * grid.x * grid.y * {local_x * local_y}u;",
        f"  if (linear_index >= {int(count)}u) {{ return; }}",
        body,
        "}",
        "",
    ])
    from .compiled_program_api import CompiledProgramAPI, EntryPoint

    api = CompiledProgramAPI(
        module=name,
        language="wgsl",
        entry="main",
        entry_points=(EntryPoint("main", "main", "numerical"),),
        metadata={
            "execution_model": "compute",
            "storage": "WebGPU storage buffers",
            "workgroup_size": launch_plan.workgroup_size,
            "dispatch_workgroups": launch_plan.groups,
            "count": launch_plan.count,
            "stage": COMPUTE.name,
            "io_layout": io_layout.to_mapping(),
            "feed_bindings": [item.to_mapping() for item in feed_bindings],
            "outputs": [item.to_mapping() for item in output_bindings],
        },
    )
    return WGSLModule(
        name, source, not shortfalls, tuple(shortfalls), api, launch_plan, io_layout,
    )


__all__ = [
    "WGSLComputeLimits", "WGSLLaunchPlan", "WGSLModule", "WGSLShortfall",
    "emit_module", "plan_wgsl_launch", "supported_tensor_operations",
]