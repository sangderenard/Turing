import json
import shutil
import subprocess

import pytest

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.wasm_class_coordinator import (
    build_browser_thread_plan,
    build_class_inventory,
    compile_python_coordinator,
    emit_wasm_class_coordinator,
    emit_wasm_control_coordinator,
)
from src.compiler.wasm_class_modules import (
    build_embedded_class_graph,
    emit_class_modules,
    emit_control_region_modules,
    partition_reduced_program,
)
from src.compiler.control_source import (
    ControlDeploymentLane,
    ControlDeploymentRegion,
    ControlProgram,
    LoopControlBlock,
    LoopBlock,
    ParallelDeployment,
    SequenceBlock,
    StatementBlock,
    StateMachineTick,
    WhileBlock,
)
from src.compiler.deployment_frame import DeploymentJoin


def test_parallel_tags_become_scale_one_worker_deploy_and_join_plan():
    control = ControlProgram(
        SequenceBlock((
            StatementBlock(("__scheduled_region_0__",)),
            ParallelDeployment((
                StatementBlock(("__scheduled_region_1__",)),
                SequenceBlock((
                    StatementBlock(("__scheduled_region_2__",)),
                    StatementBlock(("__scheduled_region_3__",)),
                )),
            ), schedule_preference="asap"),
        )),
        region_indices=(0, 1, 2, 3),
    )

    plan = build_browser_thread_plan(
        control, {0: 10, 1: 11, 2: 12, 3: 13}
    )

    assert plan["abi"] == "turing.wasm-thread-deployment.v1"
    assert plan["tile_alignment"] == 8
    deploy = plan["root"]["children"][1]
    assert deploy == {
        "kind": "deploy",
        "scale": 1,
        "schedule_preference": "asap",
        "join": {"mode": "barrier"},
        "lanes": [
            {"kind": "call", "method": 11},
            {"kind": "sequence", "children": [
                {"kind": "call", "method": 12},
                {"kind": "call", "method": 13},
            ]},
        ],
    }


def test_nonlexical_control_keeps_wasm_coordinator_as_serial_fallback():
    control = ControlProgram(
        LoopBlock(
            "i", "0", "4", "1",
            StatementBlock(("__scheduled_region_0__",)),
        ),
        region_indices=(0,),
    )
    assert build_browser_thread_plan(control, {0: 0}) is None


def test_durable_parallel_table_groups_contiguous_unrolled_regions():
    control = ControlProgram(
        SequenceBlock(tuple(
            StatementBlock((f"__scheduled_region_{index}__",))
            for index in range(4)
        )),
        region_indices=(0, 1, 2, 3),
        deployment_regions=(ControlDeploymentRegion(
            region_id=7,
            kind="parallel_candidate",
            schedule="independent_lanes",
            schedule_preference="alap",
            lanes=(
                ControlDeploymentLane(0, region_indices=(1,)),
                ControlDeploymentLane(1, region_indices=(2,)),
            ),
            join=DeploymentJoin(),
        ),),
    )

    plan = build_browser_thread_plan(control, {index: index for index in range(4)})

    children = plan["root"]["children"]
    assert [child["kind"] for child in children] == ["call", "deploy", "call"]
    assert children[1]["region_id"] == 7
    assert children[1]["scale"] == 1
    assert [lane["method"] for lane in children[1]["lanes"]] == [1, 2]


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


def test_inventory_redirects_input_identity_to_the_exact_output_slot():
    specs, _modules, manifest = _deployment()
    output_key = f"out::{specs[-1].module_name}::result"
    manifest["storage_redirects"] = {"in::x": output_key}

    inventory = build_class_inventory(manifest)
    mapping = inventory.to_mapping()

    assert mapping["storage_redirects"] == [{
        "identity": "in::x", "storage": output_key,
    }]
    assert "in::x" not in {field["key"] for field in mapping["field_slots"]}
    assert inventory.methods[0].input_slots == inventory.methods[-1].output_slots


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


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_wasm_control_coordinator_runs_planner_loop_over_real_region(tmp_path):
    program = FusedProgram(
        version=1,
        feeds={1},
        steps=[OpStep(0, "mul", [1], {"right_scalar": 2.0}, 100)],
        outputs={"result": 100},
        extras={"capture_feed_origins": {1: {"binding_name": "x"}}},
    )
    control = ControlProgram(
        LoopBlock(
            "iteration_0", "0", "3", "1",
            StatementBlock(("__scheduled_region_0__",)),
        ),
        region_indices=(0,),
    )
    modules, manifest = emit_control_region_modules(
        control,
        {0: program},
        owner_name="loop_kernel",
        module_dir=".",
    )
    module_name = manifest["modules"][0]["name"]
    output_key = f"out::{module_name}::result"
    manifest["storage_redirects"] = {"in::x": output_key}
    inventory = build_class_inventory(manifest)
    coordinator = emit_wasm_control_coordinator(
        inventory,
        control,
        region_methods={0: 0},
        name="loop_control",
    )
    (tmp_path / f"{module_name}.wasm").write_bytes(modules[0].binary)
    (tmp_path / "coordinator.wasm").write_bytes(coordinator.binary)
    (tmp_path / "inventory.json").write_text(
        json.dumps(inventory.to_mapping()), encoding="utf-8"
    )
    script = tmp_path / "run-control.mjs"
    script.write_text(
        """
        import {readFileSync} from "node:fs";
        const directory = process.argv[2];
        const descriptor = JSON.parse(readFileSync(directory + "/inventory.json", "utf8"));
        const memory = new WebAssembly.Memory({initial: 1});
        const imports = {env: {memory}};
        for (const method of descriptor.methods) {
          const {instance} = await WebAssembly.instantiate(
            readFileSync(directory + "/" + method.module + ".wasm"),
            {env: {memory}}
          );
          imports[method.module] ||= {};
          imports[method.module][method.entry] = instance.exports[method.entry];
        }
        const {instance: coordinator} = await WebAssembly.instantiate(
          readFileSync(directory + "/coordinator.wasm"), imports
        );
        const count = 3;
        const inventoryOffset = 0;
        const dataOffset = 16;
        new Int32Array(memory.buffer, inventoryOffset, 1)[0] = dataOffset;
        new Float64Array(memory.buffer, dataOffset, count).set([1, 2, 3]);
        coordinator.exports.run_range(count, inventoryOffset, 0, 1);
        console.log(JSON.stringify(Array.from(
          new Float64Array(memory.buffer, dataOffset, count)
        )));
        """,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == [8, 16, 24]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_wasm_control_coordinator_runs_resident_condition_loop(tmp_path):
    condition_program = FusedProgram(
        version=1,
        feeds={1},
        steps=[OpStep(0, "less", [1], {"right_scalar": 8.0}, 10)],
        outputs={"predicate": 10},
        extras={"capture_feed_origins": {1: {"binding_name": "x"}}},
    )
    body_program = FusedProgram(
        version=1,
        feeds={1},
        steps=[OpStep(0, "mul", [1], {"right_scalar": 2.0}, 20)],
        outputs={"result": 20},
        extras={"capture_feed_origins": {1: {"binding_name": "x"}}},
    )
    control = ControlProgram(
        WhileBlock(
            predicate_value_id=10,
            condition=StatementBlock(("__scheduled_region_0__",)),
            body=StateMachineTick(
                state="value_10",
                cases=((
                    "1", StatementBlock(("__scheduled_region_1__",)),
                ),),
                default=LoopControlBlock("break"),
            ),
        ),
        region_indices=(0, 1),
    )
    programs = {0: condition_program, 1: body_program}
    modules, manifest = emit_control_region_modules(
        control, programs, owner_name="while_kernel", module_dir="."
    )
    body_entry = next(
        entry for entry in manifest["modules"] if entry["region_index"] == 1
    )
    body_output = f"out::{body_entry['name']}::result"
    manifest["storage_redirects"] = {"in::x": body_output}
    inventory = build_class_inventory(manifest)
    slots = {field.key: field.index for field in inventory.fields}
    value_slots = {
        int(value_id): slots[key]
        for value_id, key in manifest["value_bindings"].items()
        if key in slots
    }
    coordinator = emit_wasm_control_coordinator(
        inventory,
        control,
        region_methods={0: 0, 1: 1},
        value_slots=value_slots,
        region_signatures={
            region: (
                tuple(sorted(program.feeds)),
                tuple(program.outputs.values()),
            )
            for region, program in programs.items()
        },
        name="while_control",
    )
    for region, module in modules.items():
        entry = next(
            item for item in manifest["modules"]
            if item["region_index"] == region
        )
        (tmp_path / f"{entry['name']}.wasm").write_bytes(module.binary)
    (tmp_path / "coordinator.wasm").write_bytes(coordinator.binary)
    script = tmp_path / "run-while.mjs"
    script.write_text(
        """
        import {readFileSync} from "node:fs";
        const directory = process.argv[2];
        const modules = JSON.parse(process.argv[3]);
        const slots = JSON.parse(process.argv[4]);
        const memory = new WebAssembly.Memory({initial: 1});
        const imports = {env: {memory}};
        for (const item of modules) {
          const {instance} = await WebAssembly.instantiate(
            readFileSync(directory + "/" + item.name + ".wasm"),
            {env: {memory}}
          );
          imports[item.name] ||= {};
          imports[item.name][item.entry] = instance.exports[item.entry];
        }
        const {instance} = await WebAssembly.instantiate(
          readFileSync(directory + "/coordinator.wasm"), imports
        );
        const offsets = Array.from({length: slots.field_count}, (_, i) => 32 + i * 16);
        new Int32Array(memory.buffer, 0, offsets.length).set(offsets);
        new Float64Array(memory.buffer, offsets[slots.body], 1)[0] = 1;
        instance.exports.run_range(1, 0, 0, 0);
        console.log(new Float64Array(memory.buffer, offsets[slots.body], 1)[0]);
        """,
        encoding="utf-8",
    )
    module_args = [
        {"name": entry["name"], "entry": entry["entry"]}
        for entry in manifest["modules"]
    ]
    completed = subprocess.run(
        [
            "node", str(script), str(tmp_path), json.dumps(module_args),
            json.dumps({
                "body": slots[body_output],
                "field_count": len(inventory.fields),
            }),
        ],
        capture_output=True, text=True, check=True,
    )
    assert float(completed.stdout) == 8.0
