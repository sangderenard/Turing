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
    ControlProgram,
    ControlTarget,
    ControlUniform,
    LoopBlock,
    RegionCode,
    StateMachineTick,
    StatementBlock,
    compile_python_shell,
    render_python_shell,
)
from .precompile_to_ssa import lower_control_program_to_ssa
from .wasm_binary import (
    OP_I32_AND,
    OP_I32_LE_S,
    OP_I32_LT_S,
    CodeBuilder,
    WasmImport,
    build_module,
)


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


__all__ = [
    "ClassFieldSlot",
    "ClassInventory",
    "ClassMethodCard",
    "StorageRedirect",
    "WasmClassCoordinator",
    "build_class_inventory",
    "build_coordinator_control",
    "compile_python_coordinator",
    "emit_wasm_class_coordinator",
]
