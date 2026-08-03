"""Class-shaped coordination ABI for shared-memory WebAssembly methods.

A translated class is represented by three things: one memory, an ordered
inventory of callable methods, and field slots containing byte addresses in
that memory.  The browser runner and an ordinary Python process use the same
``ControlProgram`` schedule.  Only the final rendering differs: Python calls
the inventory object, while WebAssembly imports the inventory's methods and
loads their arguments from the resident slot table.

The first client is the segmented Mandelbrot deployment.  The contract is
deliberately general enough to be the receiving side of future OOP lowering;
there is no Mandelbrot-specific behavior in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .control_source import (
    CallBlock,
    ControlProgram,
    ControlExpression,
    ControlTarget,
    ControlUniform,
    LoopControlBlock,
    LoopBlock,
    ParallelDeployment,
    RegionCode,
    StateMachineTick,
    StatementBlock,
    WhileBlock,
    SequenceBlock,
    compile_python_shell,
    render_python_shell,
)
from .precompile_to_ssa import lower_control_program_to_ssa
from .wasm_binary import (
    OP_I32_AND,
    OP_I32_EQ,
    OP_I32_EQZ,
    OP_I32_LE_S,
    OP_I32_LT_S,
    OP_I32_GE_S,
    OP_I32_ADD,
    OP_I32_TRUNC_F64_S,
    OP_F64_CONVERT_I32_S,
    CodeBuilder,
    WasmImport,
    build_module,
)


def _scheduled_region(block: StatementBlock) -> int | None:
    if len(block.lines) != 1:
        return None
    line = str(block.lines[0])
    prefix = "__scheduled_region_"
    if not line.startswith(prefix) or not line.endswith("__"):
        return None
    return int(line[len(prefix):-2])


def build_browser_thread_plan(
    control: ControlProgram,
    region_methods: Mapping[int, int],
    *,
    region_extent_effects: Mapping[int, str] | None = None,
) -> dict | None:
    """Project lexical parallel tags into a browser deployment plan.

    The Wasm binary remains the serial semantic reference.  This optional
    plan lets the browser run proven-independent lanes on Web Workers and
    await the matching Join barrier. Element tiling is separately guarded by
    compiler-authored extent effects: a collective region must remain on the
    whole invocation until a partial-reduction Join exists. Control forms
    whose ordering cannot be represented exactly are left to the Wasm
    coordinator by returning ``None``.
    """

    found_deploy = False

    def project(block):
        nonlocal found_deploy
        if isinstance(block, StatementBlock):
            region = _scheduled_region(block)
            if region is None or int(region) not in region_methods:
                return None
            return {"kind": "call", "method": int(region_methods[int(region)])}
        if isinstance(block, SequenceBlock):
            children = []
            for child in block.blocks:
                projected = project(child)
                if projected is None:
                    return None
                children.append(projected)
            return {"kind": "sequence", "children": children}
        if isinstance(block, CallBlock):
            return project(block.callee)
        if isinstance(block, ParallelDeployment):
            lanes = []
            for lane in block.lanes:
                projected = project(lane)
                if projected is None:
                    return None
                lanes.append(projected)
            found_deploy = True
            return {
                "kind": "deploy",
                "scale": 1,
                "schedule_preference": block.schedule_preference,
                "join": {"mode": "barrier"},
                "lanes": lanes,
            }
        return None

    plan = project(control.root)
    if plan is None:
        return None
    if not found_deploy:
        linear_regions: list[int] = []

        def flatten(node) -> bool:
            if isinstance(node, StatementBlock):
                region = _scheduled_region(node)
                if region is None:
                    return False
                linear_regions.append(int(region))
                return True
            if isinstance(node, SequenceBlock):
                return all(flatten(child) for child in node.blocks)
            if isinstance(node, CallBlock):
                return flatten(node.callee)
            return False

        if not flatten(control.root):
            return None
        groups: dict[int, tuple[set[int], dict]] = {}
        occupied: set[int] = set()
        positions = {
            region: index for index, region in enumerate(linear_regions)
        }
        for deployment in control.deployment_regions:
            if deployment.join.mode.value != "barrier" or not deployment.lanes:
                continue
            lane_regions = [
                tuple(map(int, lane.region_indices))
                for lane in deployment.lanes
            ]
            members = {region for lane in lane_regions for region in lane}
            if not members or not members <= positions.keys() or members & occupied:
                continue
            member_positions = sorted(positions[region] for region in members)
            if member_positions != list(range(
                member_positions[0], member_positions[-1] + 1
            )):
                continue

            def lane_node(regions: tuple[int, ...]) -> dict:
                calls = [
                    {"kind": "call", "method": int(region_methods[region])}
                    for region in regions
                ]
                return calls[0] if len(calls) == 1 else {
                    "kind": "sequence", "children": calls
                }

            deploy = {
                "kind": "deploy",
                "region_id": int(deployment.region_id),
                "scale": int(deployment.scale),
                "schedule_preference": deployment.schedule_preference,
                "join": {"mode": "barrier"},
                "lanes": [lane_node(regions) for regions in lane_regions],
            }
            groups[member_positions[0]] = (members, deploy)
            occupied.update(members)
        if not groups:
            return None
        children = []
        index = 0
        while index < len(linear_regions):
            grouped = groups.get(index)
            if grouped is None:
                children.append({
                    "kind": "call",
                    "method": int(region_methods[linear_regions[index]]),
                })
                index += 1
                continue
            members, deploy = grouped
            children.append(deploy)
            index += len(members)
        plan = {"kind": "sequence", "children": children}
    extent_effects = {
        int(region): str(effect)
        for region, effect in (region_extent_effects or {}).items()
    }
    invalid_effects = set(extent_effects.values()) - {
        "pointwise", "collective", "global-state",
    }
    if invalid_effects:
        raise ValueError(
            "unknown WebAssembly extent effects: "
            + ", ".join(sorted(invalid_effects))
        )
    collective_methods = sorted({
        int(region_methods[region])
        for region, effect in extent_effects.items()
        if effect in {"collective", "global-state"}
        and region in region_methods
    })
    return {
        "abi": "turing.wasm-thread-deployment.v1",
        "tile_alignment": 8,
        "tiles_per_worker": 2,
        "extent_effect": (
            "collective" if collective_methods else "pointwise"
        ),
        "collective_methods": collective_methods,
        "root": plan,
    }


@dataclass(frozen=True)
class ClassFieldSlot:
    """One field whose resident value is a byte offset in shared memory."""

    index: int
    key: str


@dataclass(frozen=True)
class StorageRedirect:
    """One public storage identity resolved onto another resident slot."""

    identity: str
    storage: str


@dataclass(frozen=True)
class ClassMethodCard:
    """One method and the field slots bound to its pointer parameters."""

    index: int
    module: str
    entry: str
    input_slots: tuple[int, ...]
    output_slots: tuple[int, ...]

    @property
    def parameter_count(self) -> int:
        return 1 + len(self.input_slots) + len(self.output_slots)


@dataclass(frozen=True)
class ClassInventory:
    """Serializable class descriptor: fields plus ordered method cards."""

    fields: tuple[ClassFieldSlot, ...]
    methods: tuple[ClassMethodCard, ...]
    storage_redirects: tuple[StorageRedirect, ...] = ()

    def to_mapping(self) -> dict:
        return {
            "abi": "turing.class-memory-inventory.v1",
            "field_slots": [
                {"index": field.index, "key": field.key}
                for field in self.fields
            ],
            "storage_redirects": [
                {"identity": item.identity, "storage": item.storage}
                for item in self.storage_redirects
            ],
            "methods": [
                {
                    "index": method.index,
                    "module": method.module,
                    "entry": method.entry,
                    "input_slots": list(method.input_slots),
                    "output_slots": list(method.output_slots),
                    "parameter_count": method.parameter_count,
                }
                for method in self.methods
            ],
        }


@dataclass(frozen=True)
class WasmClassCoordinator:
    """Both renderings and the translated control evidence for one class."""

    name: str
    inventory: ClassInventory
    control: ControlProgram
    python_source: str
    binary: bytes
    wat: str
    ssa: object


def build_class_inventory(manifest: Mapping[str, object]) -> ClassInventory:
    """Derive field and method bindings from a class-graph manifest."""

    modules = tuple(manifest.get("modules", ()))
    source_of: dict[str, str] = {}
    for edge in manifest.get("edges", ()):
        source = edge["from"]
        target = edge["to"]
        source_of[f"{target['module']}::{target['input']}"] = (
            f"out::{source['module']}::{source['output']}"
        )
    for logical_name, targets in manifest.get("logical_inputs", {}).items():
        for module_name, input_name in targets:
            source_of[f"{module_name}::{input_name}"] = f"in::{logical_name}"

    keys = [f"in::{name}" for name in manifest.get("logical_inputs", {})]
    keys.extend(
        f"out::{module['name']}::{output_name}"
        for module in modules
        for output_name in module["outputs"]
    )
    if len(keys) != len(set(keys)):
        raise ValueError("class inventory field keys must be unique")
    redirects = {
        str(identity): str(storage)
        for identity, storage in dict(
            manifest.get("storage_redirects", {}) or {}
        ).items()
    }
    unknown = (set(redirects) | set(redirects.values())) - set(keys)
    if unknown:
        raise ValueError(
            "storage redirects name unknown identities: "
            + ", ".join(sorted(unknown))
        )

    def canonical(key: str) -> str:
        seen = []
        while key in redirects:
            if key in seen:
                raise ValueError("storage redirects contain a cycle")
            seen.append(key)
            key = redirects[key]
        return key

    canonical_keys = tuple(dict.fromkeys(canonical(key) for key in keys))
    canonical_index = {
        key: index for index, key in enumerate(canonical_keys)
    }
    field_index = {
        key: canonical_index[canonical(key)] for key in keys
    }

    methods = []
    for index, module in enumerate(modules):
        inputs = []
        for input_name in module["inputs"]:
            binding = source_of.get(f"{module['name']}::{input_name}")
            if binding is None:
                raise ValueError(
                    f"{module['name']} input {input_name} has no class field binding"
                )
            inputs.append(field_index[binding])
        outputs = tuple(
            field_index[f"out::{module['name']}::{output_name}"]
            for output_name in module["outputs"]
        )
        methods.append(ClassMethodCard(
            index=index,
            module=str(module["name"]),
            entry=str(module["entry"]),
            input_slots=tuple(inputs),
            output_slots=outputs,
        ))
    return ClassInventory(
        fields=tuple(
            ClassFieldSlot(index, key) for index, key in enumerate(canonical_keys)
        ),
        methods=tuple(methods),
        storage_redirects=tuple(
            StorageRedirect(identity, canonical(identity))
            for identity in keys
            if canonical(identity) != identity
        ),
    )


def build_coordinator_control(method_count: int) -> ControlProgram:
    """Plan an end-exclusive, latch-friendly range of method dispatches."""

    indices = tuple(range(int(method_count)))
    cases = tuple(
        (str(index), StatementBlock((f"__scheduled_region_{index}__",)))
        for index in indices
    )
    return ControlProgram(
        root=LoopBlock(
            "method_index", "start", "end", "1",
            StateMachineTick("method_index", cases),
        ),
        region_indices=indices,
        uniforms=(
            ControlUniform("start", 0, "int"),
            ControlUniform("end", 1, "int"),
        ),
    )


def _python_regions(inventory: ClassInventory) -> tuple[RegionCode, ...]:
    return tuple(
        RegionCode(
            method.index,
            ControlTarget.PYTHON,
            StatementBlock((
                f"inventory.call({method.index}, memory, count)",
            )),
        )
        for method in inventory.methods
    )


def compile_python_coordinator(inventory: ClassInventory):
    """Compile the same class schedule for an ordinary Python shell."""

    control = build_coordinator_control(len(inventory.methods))
    return compile_python_shell(
        control,
        _python_regions(inventory),
        function_name="coordinate_class_range",
        parameters=("memory", "inventory", "count", "start", "end"),
    )


def _render_wat(name: str, inventory: ClassInventory) -> str:
    lines = [
        f"(module ;; {name}",
        '  (import "env" "memory" (memory 1))',
    ]
    for method in inventory.methods:
        params = " ".join("(param i32)" for _ in range(method.parameter_count))
        lines.append(
            f'  (import "{method.module}" "{method.entry}" (func $method_{method.index} {params}))'
        )
    lines.extend((
        "  ;; run_range(count, field_slot_table, start, end); end is exclusive",
        '  (func (export "run_range") (param i32 i32 i32 i32)',
        "    ;; generated from the Python ControlProgram method schedule",
    ))
    for method in inventory.methods:
        slots = (*method.input_slots, *method.output_slots)
        lines.append(
            f"    ;; if start <= {method.index} < end: method {method.module}.{method.entry}"
        )
        lines.append(
            "    ;; args: count " + " ".join(f"field[{slot}]" for slot in slots)
        )
    lines.extend(("  )", ")", ""))
    return "\n".join(lines)


def emit_wasm_class_coordinator(
    inventory: ClassInventory,
    *,
    name: str = "class_coordinator",
) -> WasmClassCoordinator:
    """Translate a class schedule into one resident WebAssembly coordinator.

    JavaScript calls ``run_range`` once for an uninterrupted deployment.  A
    debugger or outer shell can instead issue adjacent ranges and hold a
    latch between them.  Individual method calls stay inside WebAssembly.
    """

    control = build_coordinator_control(len(inventory.methods))
    regions = _python_regions(inventory)
    python_source = render_python_shell(
        control,
        regions,
        function_name="coordinate_class_range",
        parameters=("memory", "inventory", "count", "start", "end"),
    )
    region_callees = {
        method.index: f"{method.module}.{method.entry}"
        for method in inventory.methods
    }
    # Field-slot IDs are the resident values visible at every method seam.
    # Offset them above the uniform IDs used by the schedule.
    region_signatures = {
        method.index: (
            tuple(2 + slot for slot in method.input_slots),
            tuple(2 + slot for slot in method.output_slots),
        )
        for method in inventory.methods
    }
    ssa, shortfalls = lower_control_program_to_ssa(
        control,
        function_name=name,
        first_value_id=2 + len(inventory.fields),
        region_callees=region_callees,
        region_signatures=region_signatures,
    )
    if shortfalls:
        details = "; ".join(item.reason for item in shortfalls)
        raise ValueError(f"coordinator control did not lower completely: {details}")

    imports = [
        WasmImport(
            module=method.module,
            field=method.entry,
            kind="func",
            parameter_types=("i32",) * method.parameter_count,
        )
        for method in inventory.methods
    ]
    imports.append(WasmImport(
        module="env", field="memory", kind="memory", memory_pages=1,
    ))
    # count, field_slot_table byte offset, start, end
    body = CodeBuilder(value_type="f64", parameter_count=4)
    for method in inventory.methods:
        body.local_get(2).i32_const(method.index).raw(OP_I32_LE_S)
        body.i32_const(method.index).local_get(3).raw(OP_I32_LT_S)
        body.raw(OP_I32_AND).if_()
        body.local_get(0)
        for slot in (*method.input_slots, *method.output_slots):
            body.local_get(1).i32_load(offset=slot * 4)
        body.call(method.index).end()
    binary = build_module(
        function_name="run_range",
        parameter_types=("i32", "i32", "i32", "i32"),
        body=body,
        imports=imports,
    )
    return WasmClassCoordinator(
        name=name,
        inventory=inventory,
        control=control,
        python_source=python_source,
        binary=binary,
        wat=_render_wat(name, inventory),
        ssa=ssa,
    )


def emit_wasm_control_coordinator(
    inventory: ClassInventory,
    control: ControlProgram,
    *,
    region_methods: Mapping[int, int],
    value_slots: Mapping[int, int] | None = None,
    region_signatures: Mapping[
        int, tuple[tuple[int, ...], tuple[int, ...]]
    ] | None = None,
    name: str = "control_coordinator",
) -> WasmClassCoordinator:
    """Emit the planner's real structured control around Wasm regions.

    Unlike ``emit_wasm_class_coordinator`` this does not invent a linear
    method-card range.  Region markers are invoked exactly where the supplied
    ``ControlProgram`` places them, including constant-bound lexical loops.
    Unsupported control is rejected instead of replaced by the discovery
    trace.
    """

    methods = {method.index: method for method in inventory.methods}
    imports = [
        WasmImport(
            module=method.module,
            field=method.entry,
            kind="func",
            parameter_types=("i32",) * method.parameter_count,
        )
        for method in inventory.methods
    ]
    imports.append(WasmImport(
        module="env", field="memory", kind="memory", memory_pages=1,
    ))
    # Preserve the established browser ABI: count, field-slot table, start,
    # end.  A real control kernel owns its whole schedule, so start/end are
    # intentionally not used to slice it into a different program.
    body = CodeBuilder(value_type="f64", parameter_count=4)
    value_slots = {int(key): int(value) for key, value in (value_slots or {}).items()}
    local_control: dict[str, int] = {}

    def call_region(region_index: int) -> None:
        method_index = region_methods.get(int(region_index))
        if method_index is None or method_index not in methods:
            raise ValueError(
                f"control region {region_index} has no WebAssembly method"
            )
        method = methods[method_index]
        body.local_get(0)
        for slot in (*method.input_slots, *method.output_slots):
            body.local_get(1).i32_load(offset=int(slot) * 4)
        body.call(int(method.index))

    def literal(text: str, role: str) -> int:
        try:
            return int(str(text), 10)
        except ValueError as error:
            raise ValueError(
                "WebAssembly control kernel needs a resident lowering for "
                f"dynamic {role} expression {text!r}"
            ) from error

    def load_resident_value(value_id: int) -> None:
        slot = value_slots.get(int(value_id))
        if slot is None:
            raise ValueError(
                f"WebAssembly control value {value_id} has no resident slot"
            )
        body.local_get(1).i32_load(offset=slot * 4).load()

    def load_i32_control(text: str, role: str) -> None:
        spelling = str(text).strip()
        if spelling in local_control:
            body.local_get(local_control[spelling])
            return
        if spelling.startswith("value_") and spelling[6:].isdigit():
            load_resident_value(int(spelling[6:]))
            body.raw(OP_I32_TRUNC_F64_S)
            return
        body.i32_const(literal(spelling, role))

    def load_predicate(value_id: int) -> None:
        load_resident_value(value_id)
        body.i32_const(0)
        body.raw(OP_F64_CONVERT_I32_S, 0x62)  # f64.ne

    def emit_control_expression(expression: ControlExpression) -> str:
        if expression.op == "value":
            load_resident_value(int(expression.value_id))
            return "f64"
        if expression.op == "const":
            if isinstance(expression.literal, bool):
                body.i32_const(int(expression.literal))
                return "i32"
            body.value_const(float(expression.literal))
            return "f64"
        operand_types = [
            emit_control_expression(operand)
            for operand in expression.operands
        ]
        if expression.op in {"item", "float", "int", "bool"}:
            return operand_types[0]
        if expression.op in {"add", "sub", "mul", "div"}:
            body.op(expression.op)
            return "f64"
        if expression.op == "neg":
            body.op("neg")
            return "f64"
        if expression.op in {"lt", "le", "gt", "ge", "eq", "ne"}:
            body.op(expression.op)
            return "i32"
        if expression.op in {"and", "or"}:
            body.raw(OP_I32_AND if expression.op == "and" else 0x72)
            return "i32"
        if expression.op == "not":
            body.raw(OP_I32_EQZ)
            return "i32"
        raise ValueError(
            f"unsupported WebAssembly control expression {expression.op!r}"
        )

    def emit_predicate(
        value_id: int,
        expression: ControlExpression | None,
    ) -> None:
        if expression is None:
            load_predicate(value_id)
            return
        value_type = emit_control_expression(expression)
        if value_type == "f64":
            body.i32_const(0).raw(OP_F64_CONVERT_I32_S, 0x62)

    def emit(block, *, break_depth: int | None = None, continue_depth: int | None = None) -> None:
        if isinstance(block, StatementBlock):
            region = _scheduled_region(block)
            if region is None:
                if any(str(line).strip() for line in block.lines):
                    raise ValueError(
                        "WebAssembly control kernel cannot execute raw "
                        f"control statements: {block.lines!r}"
                    )
                return
            call_region(region)
            return
        if isinstance(block, SequenceBlock):
            for child in block.blocks:
                emit(child, break_depth=break_depth, continue_depth=continue_depth)
            return
        if isinstance(block, LoopBlock):
            start = literal(block.start, "loop start")
            stop = literal(block.stop, "loop stop")
            step = literal(block.step, "loop step")
            if step == 0:
                raise ValueError("WebAssembly control loop step cannot be zero")
            induction = body.declare_local("i32")
            previous = local_control.get(block.induction)
            local_control[block.induction] = induction
            body.i32_const(start).local_set(induction)
            body.block().loop()
            body.local_get(induction).i32_const(stop).raw(
                OP_I32_GE_S if step > 0 else OP_I32_LE_S
            ).br_if(1)
            emit(block.body, break_depth=1, continue_depth=0)
            body.local_get(induction).i32_const(step).raw(
                OP_I32_ADD
            ).local_set(induction).br(0).end().end()
            if previous is None:
                local_control.pop(block.induction, None)
            else:
                local_control[block.induction] = previous
            return
        if isinstance(block, WhileBlock):
            emit(block.condition, break_depth=break_depth, continue_depth=continue_depth)
            body.block().loop()
            emit_predicate(
                block.predicate_value_id, block.predicate_expression
            )
            body.raw(OP_I32_EQZ).br_if(1)
            emit(block.body, break_depth=1, continue_depth=0)
            emit(block.condition, break_depth=1, continue_depth=0)
            body.br(0).end().end()
            return
        if isinstance(block, LoopControlBlock):
            depth = break_depth if block.action == "break" else continue_depth
            if depth is None:
                raise ValueError(f"WebAssembly {block.action} appears outside a loop")
            if block.predicate_value_id is None:
                body.br(depth)
            else:
                emit_predicate(
                    block.predicate_value_id, block.predicate_expression
                )
                if not block.expect_true:
                    body.raw(OP_I32_EQZ)
                body.br_if(depth)
            return
        if isinstance(block, ParallelDeployment):
            # Lanes are independent by planner proof. The scalar coordinator
            # may execute them serially without changing their semantics.
            for lane in block.lanes:
                emit(lane, break_depth=break_depth, continue_depth=continue_depth)
            return
        if isinstance(block, CallBlock):
            emit(block.callee, break_depth=break_depth, continue_depth=continue_depth)
            return
        if isinstance(block, StateMachineTick):
            def emit_cases(index: int, nested_depth: int) -> None:
                if index >= len(block.cases):
                    if block.default is not None:
                        emit(
                            block.default,
                            break_depth=(None if break_depth is None else break_depth + nested_depth),
                            continue_depth=(None if continue_depth is None else continue_depth + nested_depth),
                        )
                    return
                case, case_body = block.cases[index]
                load_i32_control(block.state, "state-machine selector")
                body.i32_const(literal(case, "state-machine case")).raw(OP_I32_EQ).if_()
                emit(
                    case_body,
                    break_depth=(None if break_depth is None else break_depth + nested_depth + 1),
                    continue_depth=(None if continue_depth is None else continue_depth + nested_depth + 1),
                )
                if index + 1 < len(block.cases) or block.default is not None:
                    body.else_()
                    emit_cases(index + 1, nested_depth + 1)
                body.end()

            emit_cases(0, 0)
            return
        raise ValueError(
            "WebAssembly control kernel has no lowering for "
            f"{type(block).__name__}"
        )

    emit(control.root)
    binary = build_module(
        function_name="run_range",
        parameter_types=("i32", "i32", "i32", "i32"),
        body=body,
        imports=imports,
    )
    region_callees = {
        int(region): (
            f"{methods[int(method)].module}.{methods[int(method)].entry}"
        )
        for region, method in region_methods.items()
    }
    next_result_id = 2 + len(inventory.fields)
    lowering_signatures = {
        int(region): (
            tuple(map(int, feeds)), tuple(map(int, outputs))
        )
        for region, (feeds, outputs) in (region_signatures or {}).items()
    }
    if not lowering_signatures:
        for region, method_index in region_methods.items():
            method = methods[int(method_index)]
            output_ids = tuple(
                range(next_result_id, next_result_id + len(method.output_slots))
            )
            next_result_id += len(output_ids)
            lowering_signatures[int(region)] = (
                tuple(2 + slot for slot in method.input_slots),
                output_ids,
            )
    ssa, shortfalls = lower_control_program_to_ssa(
        control,
        function_name=name,
        first_value_id=next_result_id,
        region_callees=region_callees,
        region_signatures=lowering_signatures,
    )
    if shortfalls:
        raise ValueError(
            "WebAssembly control SSA did not lower completely: "
            + "; ".join(item.reason for item in shortfalls)
        )
    wat = (
        f"(module ;; {name}\n"
        "  ;; binary generated directly from the planner ControlProgram\n"
        "  ;; imports one function per retained numerical region\n"
        ")\n"
    )
    return WasmClassCoordinator(
        name=name,
        inventory=inventory,
        control=control,
        python_source="",
        binary=binary,
        wat=wat,
        ssa=ssa,
    )


__all__ = [
    "ClassFieldSlot",
    "ClassInventory",
    "build_browser_thread_plan",
    "ClassMethodCard",
    "StorageRedirect",
    "WasmClassCoordinator",
    "build_class_inventory",
    "build_coordinator_control",
    "compile_python_coordinator",
    "emit_wasm_class_coordinator",
    "emit_wasm_control_coordinator",
]
