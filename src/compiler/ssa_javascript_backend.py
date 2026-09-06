"""Emit executable ECMAScript modules from repository SSA.

JavaScript is the browser host language, not a shader or assembly dialect.
This backend therefore prints the ordinary control program first: functions,
calls, memory accesses, and arbitrary SSA control-flow graphs.  GPU and Wasm
regions can later be attached to the same module boundary without making the
host program depend on either technology.

ECMAScript has no ``goto``.  Each SSA function is consequently emitted as a
small block dispatcher: ``block`` names the next basic block and a
``while/switch`` pair performs the transfer.  This is deliberately less
pretty than reconstructing selected loops and conditionals, but it is the
faithful generic template--irreducible graphs remain printable too.  Phi
nodes are evaluated into temporaries before any Phi destination is assigned,
which preserves parallel-copy semantics on loop back-edges.

The public entry accepts either an array in ``bufferOrder`` or an object keyed
by SSA value id.  Pointer-like formals receive their array/object unchanged;
scalar formals may be supplied directly or in a one-element typed array.  The
result is always an array in SSA ``Ret`` order.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Iterable, Mapping

from ..transmogrifier.ssa import Function, IRModule
from .class_emission_plan import plan_class_emission
from .compiled_program_api import CompiledProgramAPI, EntryPoint, Parameter
from .javascript_numeric_operators import (
    JAVASCRIPT_BITWISE_OPERATIONS,
    render_javascript_numeric_operation,
)
from .javascript_runtime_utilities import (
    javascript_utility_closure,
    render_javascript_utilities,
)
from .oop_schema import ClassSchema


@dataclass(frozen=True, slots=True)
class JavaScriptEmissionShortfall:
    operation: str
    reason: str
    function: str | None = None
    block: str | None = None

    def format(self) -> str:
        location = ""
        if self.function is not None:
            location = f" in {self.function}"
        if self.block is not None:
            location += f":{self.block}"
        return f"{self.operation}{location}: {self.reason}"


@dataclass(frozen=True, slots=True)
class JavaScriptModuleArtifact:
    name: str
    source: str
    entry: str | None
    buffer_order: tuple[int, ...]
    pointer_formals: tuple[int, ...]
    shortfalls: tuple[JavaScriptEmissionShortfall, ...]
    api: CompiledProgramAPI

    @property
    def complete(self) -> bool:
        return not self.shortfalls


_RESERVED = frozenset({
    "await", "break", "case", "catch", "class", "const", "continue",
    "debugger", "default", "delete", "do", "else", "export", "extends",
    "finally", "for", "function", "if", "import", "in", "instanceof",
    "let", "new", "return", "static", "super", "switch", "this", "throw",
    "try", "typeof", "var", "void", "while", "with", "yield",
})


def _symbol(value: str) -> str:
    held = re.sub(r"[^A-Za-z0-9_$]", "_", str(value))
    if not held or held[0].isdigit() or held in _RESERVED:
        held = "_" + held
    return held


def _literal(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "Number.NaN"
        if math.isinf(value):
            return "Number.POSITIVE_INFINITY" if value > 0 else "Number.NEGATIVE_INFINITY"
        if value == 0.0 and math.copysign(1.0, value) < 0:
            return "-0"
    if isinstance(value, tuple):
        value = list(value)
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return "undefined"


def _call_closure(
    module: IRModule,
    roots: tuple[str, ...],
    *,
    external_functions: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    pending, seen = list(map(str, roots)), []
    while pending:
        name = pending.pop()
        if name in seen or name not in module.functions:
            continue
        seen.append(name)
        if name in external_functions:
            continue
        for block in module.functions[name].blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) in {"Call", "call"}:
                    pending.append(str(instruction.attributes.get("callee") or ""))
    return tuple(seen)


def _pointer_formals(
    module: IRModule, reachable: tuple[str, ...],
) -> dict[str, set[int]]:
    pointers = {name: set() for name in reachable}
    for name in reachable:
        pointers[name].update(
            int(argument.id)
            for argument in module.functions[name].args
            if (argument.accounting or {}).get("program_abi_storage")
            == "scalar"
            and bool((argument.accounting or {}).get("program_abi_mutable"))
            and bool((argument.accounting or {}).get(
                "program_abi_field_written"
            ))
        )
        for block in module.functions[name].blocks.values():
            for instruction in block.instrs:
                operation = str(instruction.op)
                if operation in {"GetElementPtr", "getelementptr"} and instruction.args:
                    pointers[name].add(int(instruction.args[0].id))
                elif operation in {"Store", "store"} and len(instruction.args) >= 2:
                    pointers[name].add(int(instruction.args[1].id))
    changed = True
    while changed:
        changed = False
        for name in reachable:
            for block in module.functions[name].blocks.values():
                for instruction in block.instrs:
                    if str(instruction.op) not in {"Call", "call"}:
                        continue
                    callee = str(instruction.attributes.get("callee") or "")
                    target = module.functions.get(callee)
                    if target is None or callee not in pointers:
                        continue
                    for actual, formal in zip(instruction.args, target.args):
                        if int(formal.id) in pointers[callee] and int(actual.id) not in pointers[name]:
                            pointers[name].add(int(actual.id))
                            changed = True
    return pointers


def _region_output_contracts(module: IRModule) -> dict[str, tuple[int, ...] | str]:
    """Recover return records declared by aggregate call sites.

    Planned scalar regions commonly have no Ret instruction of their own;
    their caller owns the ordered ``output_ids`` contract.  Conflicting call
    sites are a compiler defect, not something an emitter may resolve by
    choosing one.
    """

    contracts: dict[str, tuple[int, ...] | str] = {}
    for function in module.functions.values():
        for block in function.blocks.values():
            for instruction in block.instrs:
                if str(instruction.op) not in {"Call", "call"}:
                    continue
                callee = str(instruction.attributes.get("callee") or "")
                declared = instruction.attributes.get("output_ids")
                if not callee or declared is None:
                    continue
                outputs = tuple(map(int, declared))
                existing = contracts.get(callee)
                if existing is None:
                    contracts[callee] = outputs
                elif isinstance(existing, tuple) and existing != outputs:
                    contracts[callee] = (
                        f"call sites disagree about {callee!r} outputs: "
                        f"{existing!r} != {outputs!r}"
                    )
    return contracts


def _source_names(function: Function) -> dict[int, str]:
    names: dict[int, str] = {}
    metadata = function.metadata or {}
    declared = tuple(metadata.get("argument_names") or ())
    if len(declared) == len(function.args):
        names.update(
            (int(value.id), str(name)) for value, name in zip(function.args, declared)
        )
    for key in ("parameter_names", "value_names"):
        for name, value_id in metadata.get(key, ()):
            names.setdefault(int(value_id), str(name))
    return names


def _javascript_performance_labels(
    module: IRModule,
    reachable: tuple[str, ...],
    async_functions: set[str],
    class_plan: Any,
) -> tuple[dict[str, Any], ...]:
    """Describe optimization intent without pretending to control a JS JIT."""

    method_owners = {
        str(method.function_name): {
            "class": definition.identity,
            "method": method.name,
        }
        for definition in class_plan.classes
        for method in definition.methods
        if method.function_name is not None
    }
    labels = []
    for name in reachable:
        function = module.functions[name]
        instructions = tuple(
            instruction
            for block in function.blocks.values()
            for instruction in block.instrs
        )
        call_count = sum(
            str(item.op) in {"Call", "call"} for item in instructions
        )
        branch_count = sum(
            str(item.op) in {"Br", "br", "CondBr", "condbr"}
            for item in instructions
        )
        allocation_risk = (
            "dynamic" if any(
                str(item.op).casefold() in {
                    "alloc", "allocate", "zeros", "ones", "full", "repeat",
                }
                for item in instructions
            ) else "none-observed"
        )
        explicit = dict((function.metadata or {}).get("performance", {}) or {})
        if "inline" in explicit:
            inline = str(explicit["inline"])
            basis = "authored-metadata"
        elif name in async_functions or len(instructions) > 64:
            inline, basis = "avoid", "emitter-structural-estimate"
        elif len(function.blocks) == 1 and len(instructions) <= 12 and call_count == 0:
            inline, basis = "prefer", "emitter-structural-estimate"
        else:
            inline, basis = "neutral", "emitter-structural-estimate"
        if inline not in {"prefer", "neutral", "avoid", "forbid"}:
            raise ValueError(
                f"function {name!r} has unsupported inline policy {inline!r}"
            )
        labels.append({
            "identity": f"function:{name}",
            "function": name,
            "role": "class-method" if name in method_owners else "function",
            **method_owners.get(name, {}),
            "inline": inline,
            "basis": basis,
            "hot_path": bool(explicit.get("hot_path", False)),
            "frequency": str(explicit.get("frequency", "unknown")),
            "instruction_count": len(instructions),
            "block_count": len(function.blocks),
            "call_count": call_count,
            "branch_count": branch_count,
            "async_boundary": name in async_functions,
            "allocation_risk": str(
                explicit.get("allocation_risk", allocation_risk)
            ),
        })
    return tuple(labels)


def _dtype_is_int64(value: Any) -> bool:
    return str(getattr(value, "dtype", "") or "").casefold() in {
        "int64", "i64", "uint64", "u64", "long", "opaque_ref",
    }


def _elementwise_numeric_expression(
    canonical: str | None,
    arguments: list[str],
    scalar_expression: str | None,
) -> str | None:
    """Lift one canonical scalar spelling over runtime array containers.

    Repository SSA keeps the numerical identity canonical while dynamic span
    extents live in the call frame.  JavaScript therefore cannot decide from
    the static dtype alone whether an ``Add`` is scalar or elementwise.  The
    same scalar spelling remains authoritative at each leaf; this wrapper only
    performs the shape-preserving traversal which ECMAScript does not provide.
    """

    if canonical is None or scalar_expression is None:
        return scalar_expression
    scalar_names = [f"a{index}" for index in range(len(arguments))]
    _resolved, leaf = render_javascript_numeric_operation(
        canonical, scalar_names,
    )
    if leaf is None:
        return scalar_expression
    return (
        "turingElementwise(("
        + ", ".join(scalar_names)
        + ") => "
        + leaf
        + ", "
        + ", ".join(arguments)
        + ")"
    )


def _non_dominating_uses(function: Function) -> tuple[tuple[str, int, str, int], ...]:
    """Return operand uses with no definition reaching their CFG position."""

    block_names = tuple(function.blocks)
    if not block_names:
        return ()
    entry = block_names[0]
    successors = {name: set() for name in block_names}
    for name, block in function.blocks.items():
        if not block.instrs:
            continue
        terminator = block.instrs[-1]
        attributes = terminator.attributes or {}
        if str(terminator.op) in {"Br", "br"}:
            successors[name].add(str(attributes.get("target")))
        elif str(terminator.op) in {"CondBr", "condbr"}:
            successors[name].update((
                str(attributes.get("true_target")),
                str(attributes.get("false_target")),
            ))
        successors[name].intersection_update(block_names)
    predecessors = {name: set() for name in block_names}
    for source, targets in successors.items():
        for target in targets:
            predecessors[target].add(source)
    reachable = {entry}
    pending = [entry]
    while pending:
        source = pending.pop()
        for target in successors[source] - reachable:
            reachable.add(target)
            pending.append(target)
    dominators = {
        name: ({name} if name == entry else set(reachable))
        for name in reachable
    }
    changed = True
    while changed:
        changed = False
        for name in reachable - {entry}:
            incoming = predecessors[name] & reachable
            common = (
                set.intersection(*(dominators[parent] for parent in incoming))
                if incoming else set()
            )
            updated = {name, *common}
            if updated != dominators[name]:
                dominators[name] = updated
                changed = True

    definitions: dict[int, list[tuple[str | None, int]]] = {}
    for argument in function.args:
        definitions.setdefault(int(argument.id), []).append((None, -1))
    for name, block in function.blocks.items():
        for index, instruction in enumerate(block.instrs):
            if instruction.res is not None:
                definitions.setdefault(int(instruction.res.id), []).append(
                    (name, index)
                )

    findings = []
    for name in reachable:
        block = function.blocks[name]
        for index, instruction in enumerate(block.instrs):
            incoming_blocks = tuple(map(
                str, (instruction.attributes or {}).get("incoming_blocks", ())
            ))
            for operand_index, operand in enumerate(instruction.args):
                value_id = int(operand.id)
                use_block = (
                    incoming_blocks[operand_index]
                    if str(instruction.op) in {"Phi", "phi"}
                    and operand_index < len(incoming_blocks)
                    else name
                )
                candidates = definitions.get(value_id, ())
                reaches = any(
                    definition_block is None
                    or (
                        definition_block in dominators.get(use_block, {use_block})
                        and (
                            definition_block != use_block
                            or definition_index < (
                                len(function.blocks[use_block].instrs)
                                if use_block != name else index
                            )
                        )
                    )
                    for definition_block, definition_index in candidates
                )
                if not reaches:
                    findings.append((name, index, str(instruction.op), value_id))
    return tuple(findings)


def emit_ssa_module_to_javascript(
    module: IRModule,
    function_name: str | None = None,
    *,
    entry_name: str | None = None,
    class_schemas: Iterable[ClassSchema] | Mapping[str, ClassSchema] | None = None,
    external_functions: Iterable[str] = (),
    runtime_utilities: Iterable[str] = (),
) -> JavaScriptModuleArtifact:
    """Print a repository-SSA call closure as one dependency-free ES module.

    ``external_functions`` names compiler-selected deployment boundaries.
    Their bodies remain owned by another emitted target (normally WGSL or
    Wasm); calls cross the generated module's ``runtime.call`` seam instead
    of duplicating that numerical program in JavaScript.  Async status is
    propagated through the ordinary SSA call graph, so host control remains
    authored SSA even when a device result must be joined before branching.
    """

    if function_name is not None and function_name not in module.functions:
        raise KeyError(f"SSA module has no function {function_name!r}")
    entry = None if function_name is None else _symbol(entry_name or function_name)
    external = frozenset(map(str, external_functions))
    utilities = javascript_utility_closure(runtime_utilities)
    unknown_external = external - set(module.functions)
    if unknown_external:
        raise KeyError(
            "SSA module has no external deployment function(s) "
            f"{sorted(unknown_external)!r}"
        )
    class_plan = plan_class_emission(module, schemas=class_schemas)
    class_method_roots = tuple(
        str(method.function_name)
        for definition in class_plan.classes
        for method in definition.methods
        if method.function_name is not None and method.body_available
    )
    roots = ((str(function_name),) if function_name is not None else ()) + class_method_roots
    reachable = _call_closure(
        module, roots, external_functions=external,
    )
    reachable = tuple(name for name in reachable if name not in external)
    async_functions = set()
    changed_async = True
    while changed_async:
        changed_async = False
        for name in reachable:
            if name in async_functions:
                continue
            calls = {
                str(instruction.attributes.get("callee") or "")
                for block in module.functions[name].blocks.values()
                for instruction in block.instrs
                if str(instruction.op) in {"Call", "call"}
                and instruction.attributes.get("tensor_operation") is None
            }
            if calls & (external | async_functions):
                async_functions.add(name)
                changed_async = True
    performance_labels = _javascript_performance_labels(
        module, reachable, async_functions, class_plan,
    )
    pointers = _pointer_formals(module, reachable)
    region_outputs = _region_output_contracts(module)
    shortfalls: list[JavaScriptEmissionShortfall] = []
    shortfalls.extend(
        JavaScriptEmissionShortfall(
            "class-plan", issue.format(), issue.class_identity, None,
        )
        for issue in class_plan.issues
        if issue.severity == "error"
    )
    for function_label in reachable:
        function = module.functions[function_label]
        metadata = function.metadata or {}
        diagnostics = dict(
            metadata.get(
                "unresolved_call_diagnostics", {}
            ) or {}
        )
        for callsite_id, diagnostic in diagnostics.items():
            shortfalls.append(JavaScriptEmissionShortfall(
                "source-call",
                "unresolved authored callsite "
                f"{callsite_id}: {diagnostic!r}",
                function_label,
                None,
            ))
        for value_id, operation, role in tuple(
            metadata.get("structural_output_shortfalls", ()) or ()
        ):
            shortfalls.append(JavaScriptEmissionShortfall(
                "source-value",
                "unlowered authored value "
                f"{value_id} ({operation}, {role})",
                function_label,
                None,
            ))
        for value_id, operation, roles in tuple(
            metadata.get("unresolved_required_source_values", ()) or ()
        ):
            shortfalls.append(JavaScriptEmissionShortfall(
                "source-value",
                "unresolved required authored value "
                f"{value_id} ({operation}, {roles!r})",
                function_label,
                None,
            ))
        for block_name, instruction_index, operation, value_id in (
            _non_dominating_uses(function)
        ):
            shortfalls.append(JavaScriptEmissionShortfall(
                operation,
                f"operand value {value_id} has no definition dominating "
                f"instruction {instruction_index}",
                function_label,
                block_name,
            ))
    definitions: list[str] = []
    function_symbols: dict[str, str] = {}

    def refuse(operation: str, reason: str, function: str, block: str) -> None:
        shortfalls.append(JavaScriptEmissionShortfall(operation, reason, function, block))

    for function_label in reachable:
        function = module.functions[function_label]
        function_symbol = "impl_" + _symbol(function_label)
        collision = next(
            (
                name for name, symbol in function_symbols.items()
                if symbol == function_symbol and name != function_label
            ),
            None,
        )
        if collision is not None:
            shortfalls.append(JavaScriptEmissionShortfall(
                "function-symbol",
                f"functions {collision!r} and {function_label!r} both "
                f"sanitize to {function_symbol!r}",
                function_label,
                None,
            ))
        function_symbols[function_label] = function_symbol
        arguments = ", ".join(f"t{int(value.id)}" for value in function.args)
        inout_span_formals = {
            int(value.id)
            for value in function.args
            if (value.accounting or {}).get("program_abi_storage") == "span"
            and bool((value.accounting or {}).get("program_abi_mutable"))
            and bool((value.accounting or {}).get(
                "program_abi_field_written"
            ))
        }
        inout_scalar_formals = {
            int(value.id)
            for value in function.args
            if (value.accounting or {}).get("program_abi_storage") == "scalar"
            and bool((value.accounting or {}).get("program_abi_mutable"))
            and bool((value.accounting or {}).get(
                "program_abi_field_written"
            ))
        }
        if function_label in async_functions:
            arguments = ", ".join(filter(None, (arguments, "turingRuntime")))
        produced = {
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        }
        available_ids = produced | {int(value.id) for value in function.args}
        missing_operands: set[int] = set()
        for block in function.blocks.values():
            for instruction in block.instrs:
                for value in instruction.args:
                    if int(value.id) not in available_ids:
                        missing_operands.add(int(value.id))
        for value_id in sorted(missing_operands):
            shortfalls.append(JavaScriptEmissionShortfall(
                "operand",
                f"%t{value_id} has no formal or producer",
                function_label,
                None,
            ))
        locals_ = sorted(produced - {int(value.id) for value in function.args})
        async_prefix = "async " if function_label in async_functions else ""
        lines = [f"{async_prefix}function {function_symbol}({arguments}) {{"]
        if locals_:
            lines.append("  let " + ", ".join(f"t{value_id}" for value_id in locals_) + ";")
        first_block = "entry" if "entry" in function.blocks else next(iter(function.blocks), None)
        if first_block is None:
            shortfalls.append(JavaScriptEmissionShortfall(
                "control", "function has no basic blocks", function_label, None,
            ))
            lines.extend(("  return [];", "}"))
            definitions.append("\n".join(lines))
            continue
        lines.extend((
            f"  let block = {_literal(first_block)};",
            "  let predecessor = null;",
            "  for (;;) {",
            "    switch (block) {",
        ))

        for block_name, block in function.blocks.items():
            lines.append(f"      case {_literal(block_name)}: {{")
            phis = [item for item in block.instrs if str(item.op) in {"Phi", "phi"}]
            for index, instruction in enumerate(phis):
                incoming = tuple(instruction.attributes.get("incoming_blocks") or ())
                if instruction.res is None or len(incoming) != len(instruction.args) or not incoming:
                    refuse("Phi", "incoming_blocks must match operands", function_label, block_name)
                    continue
                choices = " : ".join(
                    f"predecessor === {_literal(origin)} ? t{int(value.id)}"
                    for origin, value in zip(incoming, instruction.args)
                )
                lines.append(
                    f"        const phi_{index} = {choices} : turingBadPhi(" 
                    f"{_literal(function_label)}, {_literal(block_name)}, predecessor);"
                )
            for index, instruction in enumerate(phis):
                incoming = tuple(instruction.attributes.get("incoming_blocks") or ())
                if instruction.res is not None and len(incoming) == len(instruction.args) and incoming:
                    lines.append(f"        t{int(instruction.res.id)} = phi_{index};")

            terminated = False
            for instruction in block.instrs:
                operation = str(instruction.op)
                if operation in {"Phi", "phi"}:
                    continue
                args = [f"t{int(value.id)}" for value in instruction.args]
                result = None if instruction.res is None else f"t{int(instruction.res.id)}"
                attributes = instruction.attributes or {}

                if operation in {"Const", "const"}:
                    if result is None:
                        refuse(operation, "constant has no result", function_label, block_name)
                    elif "value" in attributes or "constant" in attributes:
                        value = attributes.get("value", attributes.get("constant"))
                        rendered = _literal(value)
                        if value is None:
                            refuse(
                                operation,
                                "None must use the explicit NoneValue operation",
                                function_label,
                                block_name,
                            )
                        elif rendered == "undefined":
                            refuse(operation, f"literal {value!r} is not JSON-compatible", function_label, block_name)
                        else:
                            lines.append(f"        {result} = {rendered};")
                    else:
                        refuse(operation, "constant carries no value", function_label, block_name)
                    continue
                if operation in {"NoneValue", "nonevalue"}:
                    if result is None:
                        refuse(operation, "None value has no result", function_label, block_name)
                    elif instruction.args or attributes:
                        refuse(
                            operation,
                            "NoneValue must carry no operands or attributes",
                            function_label,
                            block_name,
                        )
                    else:
                        lines.append(f"        {result} = null;")
                    continue
                if operation in {"string_token", "StringToken"}:
                    token = attributes.get("token")
                    if result is None or not isinstance(token, int):
                        refuse(
                            operation,
                            "string token needs an exact integer token and result",
                            function_label,
                            block_name,
                        )
                    else:
                        # Repository string identities are 64-bit values and
                        # routinely exceed Number.MAX_SAFE_INTEGER. BigInt is
                        # the only dependency-free ECMAScript spelling that
                        # preserves exact keyed-lookup/equality semantics.
                        lines.append(f"        {result} = {int(token)}n;")
                    continue
                if operation in {"Br", "br"}:
                    target = str(attributes.get("target") or "")
                    if target not in function.blocks:
                        refuse(operation, f"unknown target {target!r}", function_label, block_name)
                    lines.extend((
                        f"        predecessor = {_literal(block_name)};",
                        f"        block = {_literal(target)};",
                        "        continue;",
                    ))
                    terminated = True
                    continue
                if operation in {"CondBr", "condbr"}:
                    on_true = str(attributes.get("true_target") or "")
                    on_false = str(attributes.get("false_target") or "")
                    for target in (on_true, on_false):
                        if target not in function.blocks:
                            refuse(operation, f"unknown target {target!r}", function_label, block_name)
                    if len(args) != 1:
                        refuse(operation, "conditional branch needs one condition", function_label, block_name)
                    else:
                        lines.extend((
                            f"        predecessor = {_literal(block_name)};",
                            f"        block = Boolean({args[0]}) ? {_literal(on_true)} : {_literal(on_false)};",
                            "        continue;",
                        ))
                    terminated = True
                    continue
                if operation in {"Ret", "ret", "Return", "return"}:
                    lines.append("        return [" + ", ".join(args) + "];")
                    terminated = True
                    continue
                if operation in {"Call", "call"}:
                    tensor_operation = attributes.get("tensor_operation")
                    if tensor_operation is not None:
                        canonical, rendered = render_javascript_numeric_operation(
                            operation,
                            args,
                            tensor_operation=str(tensor_operation),
                        )
                        rendered = _elementwise_numeric_expression(
                            canonical, args, rendered,
                        )
                        if canonical is None:
                            refuse(
                                operation,
                                "tensor operation "
                                f"{tensor_operation!r} has no JavaScript numeric spelling",
                                function_label,
                                block_name,
                            )
                        elif rendered is None or result is None:
                            refuse(
                                operation,
                                f"tensor operation {canonical!r} has the wrong arity "
                                "or no result",
                                function_label,
                                block_name,
                            )
                        elif (
                            canonical in JAVASCRIPT_BITWISE_OPERATIONS
                            and any(_dtype_is_int64(value) for value in instruction.args)
                        ):
                            refuse(
                                operation,
                                "Number bitwise operators are 32-bit; int64 needs "
                                "a BigInt lowering",
                                function_label,
                                block_name,
                            )
                        else:
                            lines.append(f"        {result} = {rendered};")
                        continue
                    callee = str(attributes.get("callee") or "")
                    if callee in external:
                        if function_label not in async_functions:
                            refuse(operation, "external deployment needs async host control", function_label, block_name)
                            continue
                        descriptor = _literal({
                            "callee": callee,
                            "regionIndex": attributes.get("region_index"),
                            "feedIds": list(attributes.get("feed_ids") or ()),
                            "outputIds": list(attributes.get("output_ids") or ()),
                        })
                        call = (
                            f"await turingRuntime.call({descriptor}, ["
                            + ", ".join(args) + "])"
                        )
                        if result is None:
                            lines.append(f"        {call};")
                        elif attributes.get("result_convention") == "ssa.aggregate":
                            lines.append(f"        {result} = {call};")
                        else:
                            lines.append(f"        {result} = turingScalarResult({call});")
                        continue
                    if callee not in module.functions:
                        refuse(operation, f"call to unknown function {callee!r}", function_label, block_name)
                        continue
                    call_args = list(args)
                    if callee in async_functions:
                        call_args.append("turingRuntime")
                    call = f"impl_{_symbol(callee)}(" + ", ".join(call_args) + ")"
                    if callee in async_functions:
                        call = "await " + call
                    if result is None:
                        lines.append(f"        {call};")
                    elif attributes.get("result_convention") == "ssa.aggregate":
                        lines.append(f"        {result} = {call};")
                    else:
                        lines.append(f"        {result} = turingScalarResult({call});")
                    continue
                if operation in {"GetElementPtr", "getelementptr"}:
                    if result is None or not args:
                        refuse(operation, "address calculation needs a result and base", function_label, block_name)
                    else:
                        aggregate_index = attributes.get("aggregate_index")
                        if aggregate_index is not None:
                            lines.append(f"        {result} = turingPointer({args[0]}, {int(aggregate_index)});")
                        elif len(args) >= 2:
                            lines.append(
                                f"        {result} = turingPointer("
                                + ", ".join(args)
                                + ");"
                            )
                        else:
                            refuse(operation, "address calculation needs at least one index", function_label, block_name)
                    continue
                if operation in {"Load", "load"}:
                    if result is not None and len(args) == 1:
                        lines.append(f"        {result} = turingLoad({args[0]});")
                    else:
                        refuse(operation, "load needs one address and a result", function_label, block_name)
                    continue
                if operation in {"Store", "store"}:
                    if len(args) >= 2:
                        lines.append(f"        turingStore({args[1]}, {args[0]});")
                    else:
                        refuse(operation, "store needs value and address", function_label, block_name)
                    continue
                if operation in {"Cast", "CastLike", "cast_like"}:
                    if result is None or not args:
                        refuse(operation, "cast needs an operand and result", function_label, block_name)
                    else:
                        dtype = str(attributes.get("target_dtype") or instruction.res.dtype or "float64")
                        lines.append(f"        {result} = turingCast({args[0]}, {_literal(dtype)});")
                    continue
                if operation in {"Select", "select", "where"}:
                    if result is not None and len(args) == 3:
                        lines.append(f"        {result} = Boolean({args[0]}) ? {args[1]} : {args[2]};")
                    else:
                        refuse(operation, "select needs condition, true, false, and result", function_label, block_name)
                    continue
                if operation in {"GetAttr", "getattr"}:
                    attribute = attributes.get("attribute")
                    if result is not None and len(args) == 1 and attribute:
                        lines.append(f"        {result} = {args[0]}[{_literal(str(attribute))}];")
                    else:
                        refuse(operation, "getattr needs a named attribute", function_label, block_name)
                    continue
                if operation in {"SetAttr", "setattr"}:
                    attribute = attributes.get("attribute")
                    if len(args) >= 2 and attribute:
                        lines.append(f"        {args[0]}[{_literal(str(attribute))}] = {args[1]};")
                    else:
                        refuse(operation, "setattr needs receiver, value, and attribute", function_label, block_name)
                    continue

                if operation in {"Deploy", "Join"}:
                    # Scheduling permission/boundary only. Device work is
                    # represented by the compiler-selected external Call;
                    # serial JavaScript semantics contain no instruction at
                    # either marker (the same rule as LLVM and Fortran).
                    lines.append(f"        // {operation} deployment boundary")
                    continue

                canonical, rendered = render_javascript_numeric_operation(
                    operation, args,
                )
                rendered = _elementwise_numeric_expression(
                    canonical, args, rendered,
                )
                if operation == "ULt" and len(args) == 2:
                    rendered = f"(({args[0]} >>> 0) < ({args[1]} >>> 0))"
                elif operation == "ULe" and len(args) == 2:
                    rendered = f"(({args[0]} >>> 0) <= ({args[1]} >>> 0))"
                elif operation.casefold() == "fma" and len(args) == 3:
                    refuse(operation, "ECMAScript has no correctly-rounded fused multiply-add", function_label, block_name)
                    continue
                if (
                    canonical in JAVASCRIPT_BITWISE_OPERATIONS
                    and any(_dtype_is_int64(value) for value in instruction.args)
                ):
                    refuse(operation, "Number bitwise operators are 32-bit; int64 needs a BigInt lowering", function_label, block_name)
                    continue
                if rendered is None or result is None:
                    refuse(operation, "no JavaScript SSA spelling", function_label, block_name)
                    continue
                lines.append(f"        {result} = {rendered};")

            if not terminated:
                contract = region_outputs.get(function_label)
                if isinstance(contract, tuple) and block_name == next(reversed(function.blocks)):
                    lines.append(
                        "        return [" + ", ".join(
                            f"t{value_id}" for value_id in contract
                        ) + "];"
                    )
                else:
                    reason = (
                        contract
                        if isinstance(contract, str)
                        else "basic block has no terminator"
                    )
                    refuse("control", reason, function_label, block_name)
                    lines.append(
                        f"        throw new Error({_literal(function_label + ':' + block_name + ' has no terminator')});"
                    )
            lines.append("      }")
        lines.extend((
            "      default:",
            f"        throw new Error(`{function_symbol}: unknown block ${'{'}block{'}'}`);",
            "    }",
            "  }",
            "}",
        ))
        # A writable span field is a caller-owned container.  Assigning a new
        # JavaScript array to the local parameter would only rebind the local
        # name; native by-reference backends instead publish the new elements
        # into that exact field storage.  Preserve that ABI identity by
        # converting every SSA definition of such a formal into a checked
        # shape-preserving copy.
        for value_id in inout_span_formals:
            assignment = re.compile(
                rf"^(\s*)t{value_id} = (.*);$"
            )
            lines = [
                assignment.sub(
                    rf"\1turingCopyInto(t{value_id}, \2);", line,
                )
                for line in lines
            ]
        for value_id in inout_scalar_formals:
            assignment = re.compile(
                rf"^(\s*)t{value_id} = (.*);$"
            )
            lines = [
                assignment.sub(
                    rf"\1turingCopyScalarInto(t{value_id}, \2);", line,
                )
                for line in lines
            ]
        definitions.append("\n".join(lines))

    class_definitions: list[str] = []
    class_symbols: dict[str, str] = {}
    for definition in class_plan.classes:
        class_symbol = _symbol(definition.identity)
        collided = next(
            (
                identity for identity, symbol in class_symbols.items()
                if symbol == class_symbol and identity != str(definition.identity)
            ),
            None,
        )
        if collided is not None:
            shortfalls.append(JavaScriptEmissionShortfall(
                "class-symbol",
                f"identities {collided!r} and {definition.identity!r} both "
                f"sanitize to {class_symbol!r}",
                str(definition.identity),
                None,
            ))
        class_symbols[str(definition.identity)] = class_symbol
        fields = definition.fields
        field_layout = [
            {"name": str(field.name), "slot": int(field.slot)} for field in fields
        ]
        constructors = tuple(
            method for method in definition.methods if method.kind == "initializer"
        )
        constructor = constructors[0] if len(constructors) == 1 else None
        if len(constructors) > 1:
            shortfalls.append(JavaScriptEmissionShortfall(
                "class-constructor",
                "JavaScript has one constructor but the class plan has "
                f"{len(constructors)} initializers",
                definition.identity,
                None,
            ))
        if definition.bases:
            shortfalls.append(JavaScriptEmissionShortfall(
                "class-inheritance",
                "base classes are planned but JavaScript inheritance emission "
                "is not wired yet: " + repr(definition.bases),
                definition.identity,
                None,
            ))
        constructor_args = (
            [] if constructor is None
            else [_symbol(parameter.name) for parameter in constructor.parameters]
        )
        lines = [
            f"export class {class_symbol} {{",
            f"  static ssaIdentity = {_literal(str(definition.identity))};",
            "  static fieldLayout = Object.freeze(" + _literal(field_layout) + ");",
            f"  constructor({', '.join(constructor_args)}) {{",
        ]
        for field in fields:
            initial = _literal(field.initial) if field.has_initial else "undefined"
            if field.has_initial and initial == "undefined" and field.initial is not None:
                shortfalls.append(JavaScriptEmissionShortfall(
                    "class-field-initializer",
                    f"field {field.name!r} initializer is not a JavaScript literal",
                    definition.identity,
                    None,
                ))
            lines.append(f"    this[{_literal(str(field.name))}] = {initial};")
        if constructor is not None:
            if not constructor.body_available or constructor.function_name is None:
                lines.append(
                    "    throw new Error(" + _literal(
                        f"{definition.identity}.{constructor.name}: missing SSA body "
                        f"{constructor.function_name!r}"
                    ) + ");"
                )
            else:
                by_position = {
                    parameter.position: name
                    for parameter, name in zip(
                        constructor.parameters, constructor_args
                    )
                }
                if constructor.receiver_position is not None:
                    by_position[constructor.receiver_position] = "this"
                for receiver_field in constructor.receiver_fields:
                    by_position[receiver_field.formal_position] = (
                        "turingFieldStorage(this, "
                        f"{receiver_field.field_slot}, {receiver_field.offset})"
                    )
                call_args = ", ".join(
                    by_position[position] for position in sorted(by_position)
                )
                lines.append(
                    f"    impl_{_symbol(str(constructor.function_name))}({call_args});"
                )
        lines.append("  }")
        for method in definition.methods:
            if method is constructor:
                continue
            method_name = method.name
            arguments = [_symbol(parameter.name) for parameter in method.parameters]
            static_prefix = "static " if method.is_static else ""
            lines.append(
                f"  {static_prefix}[{_literal(method_name)}]({', '.join(arguments)}) {{"
            )
            if method.kind == "allocator":
                shortfalls.append(JavaScriptEmissionShortfall(
                    "class-allocator",
                    "Python-style __new__ allocation has no JavaScript class "
                    "mapping yet",
                    definition.identity,
                    None,
                ))
                lines.append(
                    "    throw new Error(" + _literal(
                        f"{definition.identity}.{method_name}: allocator emission unavailable"
                    ) + ");"
                )
            elif not method.body_available or method.function_name is None:
                lines.append(
                    "    throw new Error(" + _literal(
                        f"{definition.identity}.{method_name}: missing SSA body "
                        f"{method.function_name!r}"
                    ) + ");"
                )
            else:
                by_position = {
                    parameter.position: name
                    for parameter, name in zip(method.parameters, arguments)
                }
                if method.receiver_position is not None:
                    by_position[method.receiver_position] = "this"
                for receiver_field in method.receiver_fields:
                    by_position[receiver_field.formal_position] = (
                        "turingFieldStorage(this, "
                        f"{receiver_field.field_slot}, {receiver_field.offset})"
                    )
                call_args = ", ".join(
                    by_position[position] for position in sorted(by_position)
                )
                lines.append(
                    f"    return turingPublicResult(impl_{_symbol(str(method.function_name))}({call_args}));"
                )
            lines.append("  }")
        lines.append("}")
        class_definitions.append("\n".join(lines))

    root = None if function_name is None else module.functions[function_name]
    buffer_order = () if root is None else tuple(int(value.id) for value in root.args)
    root_pointers = () if root is None else tuple(
        sorted(pointers[str(function_name)] & set(buffer_order))
    )
    source_names = {} if root is None else _source_names(root)
    parameters = tuple(
        Parameter(
            name=f"t{value.id}",
            role="inout" if int(value.id) in root_pointers else "input",
            dtype=str(value.dtype or "float64"),
            c_type="JavaScript value",
            ctypes_name="n/a",
            passing="reference" if int(value.id) in root_pointers else "value",
            shape=tuple(value.shape or ()),
            source_name=source_names.get(int(value.id)),
        )
        for value in (() if root is None else root.args)
    )
    api = CompiledProgramAPI(
        module=entry or "javascript_classes",
        language="javascript",
        entry=entry,
        entry_points=(
            () if entry is None else (EntryPoint(entry, entry, "control", parameters),)
        ),
        metadata={
            "module_format": "esm",
            "execution_model": "ssa-block-dispatch",
            "buffer_order": buffer_order,
            "pointer_formals": root_pointers,
            "external_functions": tuple(sorted(external)),
            "runtime_utilities": tuple(
                utility.to_data() for utility in utilities
            ),
            "performance_labels": performance_labels,
            "return_convention": "array-in-ssa-ret-order",
            "class_table_schema": "turing.repository-ssa-class-table.v1",
            "class_table": tuple(
                {
                    **definition.to_mapping(),
                    "symbol": class_symbols[definition.identity],
                }
                for definition in class_plan.classes
            ),
            "class_emission_plan": class_plan.to_mapping(),
        },
    )
    abi_json = json.dumps({
        "schema": "turing-javascript-program-v1",
        "entry": entry,
        "bufferOrder": buffer_order,
        "pointerFormals": root_pointers,
        "returnConvention": "array-in-ssa-ret-order",
        "classes": tuple(
            {
                "identity": identity,
                "symbol": symbol,
            }
            for identity, symbol in class_symbols.items()
        ),
    }, separators=(",", ":"))
    pointer_slots = [value_id in root_pointers for value_id in buffer_order]
    scalar_pointer_ids = {
        int(value.id)
        for value in (() if root is None else root.args)
        if int(value.id) in root_pointers
        and (value.accounting or {}).get("program_abi_storage") == "scalar"
    }
    wrapper_arguments = ", ".join(
        (
            f"turingScalarPointerInput(ordered[{index}])"
            if value_id in scalar_pointer_ids
            else f"ordered[{index}]"
            if pointer
            else f"turingScalarInput(ordered[{index}])"
        )
        for index, (value_id, pointer) in enumerate(zip(
            buffer_order, pointer_slots,
        ))
    )
    utility_exports = "export const RUNTIME_UTILITIES = Object.freeze({" + ",".join(
        _literal(utility.identity) + ":Object.freeze({" + ",".join((
            f"identity:{_literal(utility.identity)}",
            f"contentKey:{_literal(utility.content_key)}",
            f"capability:{_literal(utility.capability)}",
            "exports:Object.freeze({" + ",".join(
                f"{_literal(name)}:{symbol}" for name, symbol in utility.exports
            ) + "})",
        )) + "})"
        for utility in utilities
    ) + "});"
    performance_json = json.dumps(
        performance_labels, ensure_ascii=False, separators=(",", ":"),
    )
    source = "\n\n".join((
        """// Generated by Turing's repository-SSA JavaScript backend.
// This module is dependency-free and does not require WebGPU or WebAssembly.
function turingMod(a, b) {
  const remainder = a % b;
  return remainder !== 0 && ((remainder < 0) !== (b < 0)) ? remainder + b : remainder;
}
function turingRoundEven(value) {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}
function turingPointer(container, ...indices) {
  if (container && container.__turingPointer === true) {
    return {__turingPointer: true, container: container.container, path: [...container.path, ...indices.map(Number)]};
  }
  return {__turingPointer: true, container, path: indices.map(Number)};
}
function turingResolvedIndex(container, index) {
  const layout = container?.constructor?.fieldLayout;
  if (Array.isArray(layout)) {
    const field = layout.find((candidate) => candidate.slot === Number(index));
    if (field) return field.name;
  }
  return index;
}
function turingFieldStorage(owner, slot, offset = 0) {
  const field = owner?.constructor?.fieldLayout?.find(
    (candidate) => candidate.slot === Number(slot)
  );
  if (!field) throw new Error(`object has no SSA field slot ${slot}`);
  const storage = Object.create(null);
  Object.defineProperty(storage, String(offset), {
    enumerable: true,
    get: () => owner[field.name],
    set: (value) => { owner[field.name] = value; }
  });
  return storage;
}
function turingLoad(pointer) {
  if (!pointer || pointer.__turingPointer !== true) return pointer;
  let value = pointer.container;
  for (let depth = 0; depth < pointer.path.length; depth += 1) {
    const index = pointer.path[depth];
    if (value === null || value === undefined) {
      throw new TypeError(
        `load path ${JSON.stringify(pointer.path)} reached ${String(value)} before index ${depth}`
      );
    }
    value = value[turingResolvedIndex(value, index)];
  }
  return value;
}
function turingStore(pointer, value) {
  if (!pointer || pointer.__turingPointer !== true || pointer.path.length === 0) {
    throw new TypeError("store destination is not an address");
  }
  let container = pointer.container;
  for (let depth = 0; depth < pointer.path.length - 1; depth += 1) {
    const index = pointer.path[depth];
    if (container === null || container === undefined) {
      throw new TypeError(
        `store path ${JSON.stringify(pointer.path)} reached ${String(container)} before index ${depth}`
      );
    }
    container = container[turingResolvedIndex(container, index)];
  }
  if (container === null || container === undefined) {
    throw new TypeError(
      `store path ${JSON.stringify(pointer.path)} has no destination container`
    );
  }
  const finalIndex = turingResolvedIndex(container, pointer.path.at(-1));
  container[finalIndex] = value;
}
function turingScalarInput(value) {
  return ArrayBuffer.isView(value) || Array.isArray(value) ? value[0] : value;
}
function turingScalarResult(values) {
  return Array.isArray(values) && values.length === 1 ? values[0] : values;
}
function turingScalarPointerInput(value) {
  if (value && value.__turingScalarPointer === true) return value;
  const backing = (
    ArrayBuffer.isView(value) || Array.isArray(value)
  ) ? value : null;
  let resident = backing === null ? value : backing[0];
  const box = {
    __turingScalarPointer: true,
    valueOf() { return this.value; },
    [Symbol.toPrimitive]() { return this.value; },
  };
  Object.defineProperty(box, "value", {
    enumerable: true,
    get: () => backing === null ? resident : backing[0],
    set: (next) => {
      if (backing === null) resident = next;
      else backing[0] = next;
    },
  });
  Object.defineProperty(box, "0", {
    enumerable: true,
    get: () => box.value,
    set: (next) => { box.value = next; },
  });
  return box;
}
function turingScalarValue(value) {
  return value && value.__turingScalarPointer === true ? value.value : value;
}
function turingArrayLike(value) {
  return Array.isArray(value) || (
    ArrayBuffer.isView(value) && !(value instanceof DataView)
  );
}
function turingElementwise(operation, ...values) {
  values = values.map(turingScalarValue);
  const containers = values.filter(turingArrayLike);
  if (containers.length === 0) return operation(...values);
  const length = containers[0].length;
  if (containers.some((value) => value.length !== length)) {
    throw new RangeError("elementwise operands have different extents");
  }
  const mapped = Array.from({length}, (_, index) => turingElementwise(
    operation,
    ...values.map((value) => turingArrayLike(value) ? value[index] : value),
  ));
  const prototype = containers[0];
  return ArrayBuffer.isView(prototype)
    ? new prototype.constructor(mapped)
    : mapped;
}
function turingCopyScalarInto(destination, source) {
  if (!destination || destination.__turingScalarPointer !== true) {
    throw new TypeError("scalar writeback destination is not a scalar pointer");
  }
  destination.value = turingScalarValue(source);
  return destination;
}
function turingCopyInto(destination, source) {
  if (!turingArrayLike(destination) || !turingArrayLike(source)) {
    throw new TypeError("span writeback needs array-like source and destination");
  }
  if (destination.length !== source.length) {
    throw new RangeError("span writeback changed the declared extent");
  }
  for (let index = 0; index < destination.length; index += 1) {
    if (turingArrayLike(destination[index]) || turingArrayLike(source[index])) {
      turingCopyInto(destination[index], source[index]);
    } else {
      destination[index] = source[index];
    }
  }
  return destination;
}
function turingPublicResult(values) {
  if (!Array.isArray(values)) return values;
  if (values.length === 0) return undefined;
  return values.length === 1 ? values[0] : values;
}
function turingExternalValue(value) {
  return value && value.__turingScalarPointer === true ? value.value : value;
}
function turingCast(value, dtype) {
  switch (String(dtype).toLowerCase()) {
    case "bool": case "i1": return Boolean(value);
    case "float32": case "f32": case "float": return Math.fround(value);
    case "uint32": case "u32": return value >>> 0;
    case "int": case "int32": case "i32": return value | 0;
    case "int64": case "i64": case "long": return Math.trunc(value);
    default: return Number(value);
  }
}
function turingBadPhi(fn, block, predecessor) {
  throw new Error(`${fn}:${block}: no Phi input for predecessor ${predecessor}`);
}""",
        render_javascript_utilities(utility.identity for utility in utilities),
        *definitions,
        *class_definitions,
        utility_exports,
        f"export const PERFORMANCE_LABELS = Object.freeze({performance_json});",
        f"export const PROGRAM_ABI = Object.freeze({abi_json});",
        "export const SSA_CLASSES = Object.freeze({" + ",".join(
            f"{_literal(identity)}:{symbol}"
            for identity, symbol in class_symbols.items()
        ) + "});",
        (
            "\n".join((
                f"export {'async ' if function_name in async_functions else ''}function {entry}(buffers, extents = [], runtime = null) {{",
                "  void extents;",
                "  const ordered = Array.isArray(buffers)",
                "    ? buffers",
                "    : PROGRAM_ABI.bufferOrder.map((id) => buffers[id] ?? buffers[String(id)]);",
                f"  if (ordered.length !== {len(buffer_order)}) throw new RangeError(\"expected {len(buffer_order)} buffers\");",
                f"  const result = {'await ' if function_name in async_functions else ''}impl_{_symbol(str(function_name))}({wrapper_arguments}{', runtime' if function_name in async_functions else ''});",
                "  return Array.isArray(result) ? result.map(turingExternalValue) : turingExternalValue(result);",
                "}",
                f"export default Object.freeze({{abi: PROGRAM_ABI, classes: SSA_CLASSES, runtimeUtilities: RUNTIME_UTILITIES, performance: PERFORMANCE_LABELS, run: {entry}}});",
                "",
            ))
            if entry is not None
            else "export default Object.freeze({abi: PROGRAM_ABI, classes: SSA_CLASSES, runtimeUtilities: RUNTIME_UTILITIES, performance: PERFORMANCE_LABELS});\n"
        ),
    ))
    return JavaScriptModuleArtifact(
        entry or "javascript_classes", source, entry, buffer_order, root_pointers,
        tuple(shortfalls), api,
    )


__all__ = [
    "JavaScriptEmissionShortfall",
    "JavaScriptModuleArtifact",
    "emit_ssa_module_to_javascript",
]
