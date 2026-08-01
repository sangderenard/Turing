import json
import shutil
import subprocess

import pytest

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.wasm_class_coordinator import (
    build_class_inventory,
    compile_python_coordinator,
    emit_wasm_class_coordinator,
)
from src.compiler.wasm_class_modules import (
    build_embedded_class_graph,
    emit_class_modules,
    partition_reduced_program,
)


def _deployment():
    previous = 1
    steps = []
    for index in range(4):
        result = 100 + index
        steps.append(OpStep(
            index, "mul", [previous], {"right_scalar": 2.0}, result,
        ))
        previous = result
    program = FusedProgram(
        version=1,
        feeds={1},
        steps=steps,
        outputs={"result": previous},
        extras={"capture_feed_origins": {1: {"binding_name": "x"}}},
    )
    specs = partition_reduced_program(program, chunk_size=2, owner_name="kernel")
    modules = emit_class_modules(
        specs, dtype="float64", link_calls=False, shared_memory=True,
    )
    manifest = build_embedded_class_graph(
        specs, modules, program, entrypoint="kernel", embed_binaries=False,
    )
    return specs, modules, manifest


def test_python_shell_uses_the_same_range_and_inventory_contract():
    _specs, _modules, manifest = _deployment()
    inventory = build_class_inventory(manifest)
    calls = []

    class RuntimeInventory:
        def call(self, index, memory, count):
            calls.append((index, memory, count))

    memory = object()
    coordinate = compile_python_coordinator(inventory)
    coordinate(memory, RuntimeInventory(), 17, 1, 2)

    assert calls == [(1, memory, 17)]
    assert "for method_index in range(start, end, 1)" in coordinate.__compiled_shell_source__


def test_inventory_is_a_class_descriptor_not_a_host_schedule():
    specs, _modules, manifest = _deployment()
    mapping = build_class_inventory(manifest).to_mapping()

    assert mapping["abi"] == "turing.class-memory-inventory.v1"
    assert [field["key"] for field in mapping["field_slots"]] == [
        "in::x",
        f"out::{specs[0].module_name}::value_101",
        f"out::{specs[1].module_name}::result",
    ]
    assert [method["index"] for method in mapping["methods"]] == [0, 1]
    assert mapping["methods"][1]["input_slots"] == [1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_wasm_coordinator_calls_cards_internally_and_honors_latched_ranges(tmp_path):
    specs, modules, manifest = _deployment()
    inventory = build_class_inventory(manifest)
    coordinator = emit_wasm_class_coordinator(inventory)
    for spec in specs:
        (tmp_path / f"{spec.module_name}.wasm").write_bytes(
            modules[spec.index].binary
        )
    coordinator_path = tmp_path / "coordinator.wasm"
    coordinator_path.write_bytes(coordinator.binary)
    descriptor_path = tmp_path / "inventory.json"
    descriptor_path.write_text(json.dumps(inventory.to_mapping()), encoding="utf-8")
    script = tmp_path / "run.mjs"
    script.write_text(
        """
        import {readFileSync} from "node:fs";
        const [directory, descriptorPath, coordinatorPath] = process.argv.slice(2);
        const descriptor = JSON.parse(readFileSync(descriptorPath, "utf8"));
        const memory = new WebAssembly.Memory({initial: 1});
        const imports = {env: {memory}};
        for (const method of descriptor.methods) {
          const bytes = readFileSync(directory + "/" + method.module + ".wasm");
          const {instance} = await WebAssembly.instantiate(bytes, {env: {memory}});
          imports[method.module] ||= {};
          imports[method.module][method.entry] = instance.exports[method.entry];
        }
        const {instance: coordinator} = await WebAssembly.instantiate(
          readFileSync(coordinatorPath), imports
        );
        const count = 3;
        const inventoryOffset = 0;
        const offsets = [16, 40, 64];
        new Int32Array(memory.buffer, inventoryOffset, offsets.length).set(offsets);
        new Float64Array(memory.buffer, offsets[0], count).set([1, 2, 3]);
        coordinator.exports.run_range(count, inventoryOffset, 0, 1);
        const seam = Array.from(new Float64Array(memory.buffer, offsets[1], count));
        coordinator.exports.run_range(count, inventoryOffset, 1, 2);
        const result = Array.from(new Float64Array(memory.buffer, offsets[2], count));
        console.log(JSON.stringify({seam, result}));
        """,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(tmp_path), str(descriptor_path), str(coordinator_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload == {"seam": [4, 8, 12], "result": [16, 32, 48]}
