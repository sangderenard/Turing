"""End-to-end: a container field runs through the real class coordinator.

A class with a container field ``table`` gets a store region (table[gx][gy]=val)
and a read region (table[gx][gy]) sharing that one resident field. The coordinator
seeds the field with a heap map (capacity header) and the bump cursor -- exactly
what the JS ``layout()`` does -- then dispatches both regions. The read recovers
what the store wrote, proving the container store/read kernels, the heap map ABI,
and the coordinator's container seeding all agree end to end.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.control_source import (
    ControlProgram, SequenceBlock, StatementBlock,
)
from src.compiler.wasm_class_modules import emit_control_region_modules
from src.compiler.wasm_class_coordinator import (
    build_class_inventory, emit_wasm_class_coordinator,
)


def _store_read_class():
    store = FusedProgram(
        version=1, feeds={1, 2, 3, 4},
        steps=[OpStep(0, "IndexedStore", [1, 2, 3, 4],
                      {"source_type": "SubscriptStore"}, 5)],
        outputs={"tbl": 5}, meta={},
        extras={"capture_feed_origins": {
            1: {"binding_name": "table"}, 2: {"binding_name": "gx"},
            3: {"binding_name": "gy"}, 4: {"binding_name": "val"}}},
    )
    read = FusedProgram(
        version=1, feeds={11, 12, 13},
        steps=[OpStep(0, "Indexed", [11, 12], {"source_type": "Subscript"}, 14),
               OpStep(1, "Indexed", [14, 13], {"source_type": "Subscript"}, 15)],
        outputs={"result": 15}, meta={},
        extras={"capture_feed_origins": {
            11: {"binding_name": "table"}, 12: {"binding_name": "gx"},
            13: {"binding_name": "gy"}}},
    )
    control = ControlProgram(
        SequenceBlock((StatementBlock(("__scheduled_region_0__",)),
                       StatementBlock(("__scheduled_region_1__",)))),
        region_indices=(0, 1),
    )
    modules, manifest = emit_control_region_modules(
        control, {0: store, 1: read}, owner_name="cls",
        module_dir=".", dtype="float64",
    )
    inventory = build_class_inventory(manifest)
    coordinator = emit_wasm_class_coordinator(inventory, name="cc")
    return modules, manifest, inventory, coordinator


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_container_field_round_trips_through_the_coordinator(tmp_path):
    modules, manifest, inventory, coordinator = _store_read_class()
    inv = inventory.to_mapping()
    # Emit the two kernels and the coordinator; wire imports by kernel name.
    (tmp_path / "cc.wasm").write_bytes(coordinator.binary)
    kernels = {}
    for region, spec in zip((0, 1), manifest["modules"]):
        (tmp_path / f"{spec['kernel']}.wasm").write_bytes(modules[region].binary)
        kernels[spec["kernel"]] = spec["entry"]
    plan = {
        "fields": [f["key"] for f in inv["field_slots"]],
        "containerFields": inv["container_fields"],
        "methods": [
            {"kernel": m["kernel"], "entry": m["entry"],
             "input_slots": m["input_slots"], "output_slots": m["output_slots"]}
            for m in inv["methods"]
        ],
        "kernels": kernels,
        "heap": manifest["heap"],
        "sharedStatic": manifest["shared_static_bytes"],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    script = tmp_path / "run.mjs"
    script.write_text(
        r"""
        import {readFileSync} from "node:fs";
        const plan = JSON.parse(readFileSync(process.argv[2]));
        const dir = process.argv[3];
        const memory = new WebAssembly.Memory({initial: 4});
        const load = async (name) =>
          (await WebAssembly.instantiate(readFileSync(dir + "/" + name + ".wasm"),
            {env: {memory}})).instance;
        // Instantiate each unique kernel once, then the coordinator importing them.
        const kinst = {};
        for (const k of Object.keys(plan.kernels)) kinst[k] = await load(k);
        const imports = {env: {memory}};
        for (const m of plan.methods) {
          imports[m.kernel] = imports[m.kernel] || {};
          imports[m.kernel][m.entry] = kinst[m.kernel].exports[m.entry];
        }
        const cc = (await WebAssembly.instantiate(
          readFileSync(dir + "/cc.wasm"), imports)).instance;

        // --- Replicate the coordinator's layout() seeding. ---
        const elementBytes = 8, count = 1;
        const fields = plan.fields;
        const containerFields = new Set(plan.containerFields.map(Number));
        const heap = plan.heap;
        const reserved = Number(heap.reserved_bytes);
        const inventoryOffset = Math.max(
          reserved, Math.ceil(Number(plan.sharedStatic || 0) / 4) * 4);
        let cursor = inventoryOffset + fields.length * 4;
        cursor = Math.ceil(cursor / elementBytes) * elementBytes;
        const fieldOffsets = fields.map((f, i) => {
          if (containerFields.has(i)) {
            const base = cursor;
            cursor = Math.ceil((base + Number(heap.map_block_bytes)) / elementBytes) * elementBytes;
            return base;
          }
          const off = cursor; cursor += count * elementBytes; return off;
        });
        new Int32Array(memory.buffer, inventoryOffset, fields.length).set(fieldOffsets);
        for (let i = 0; i < fields.length; i++) {
          if (!containerFields.has(i)) continue;
          new Uint8Array(memory.buffer, fieldOffsets[i], Number(heap.map_block_bytes)).fill(0);
          new Int32Array(memory.buffer, fieldOffsets[i], 1)[0] = Number(heap.map_capacity);
        }
        new Int32Array(memory.buffer, Number(heap.cursor_addr), 1)[0] = cursor;

        // Set the scalar inputs gx, gy, val (raw i64 bits in their field cells).
        const slotOf = key => fieldOffsets[fields.indexOf(key)];
        const i64 = new BigInt64Array(memory.buffer);
        const gx = 0x401000n, gy = 0x20n, val = 0x1234567890n;
        i64[slotOf("in::gx") / 8] = gx;
        i64[slotOf("in::gy") / 8] = gy;
        i64[slotOf("in::val") / 8] = val;

        // Run region 0 (store) then region 1 (read).
        cc.exports.run_range(count, inventoryOffset, 0, 2);

        const resultKey = fields.find(k => k.endsWith("::result"));
        const out = new BigInt64Array(memory.buffer)[slotOf(resultKey) / 8];
        console.log(out.toString());
        """,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(tmp_path / "plan.json"), str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    assert completed.stdout.strip() == str(0x1234567890), (
        completed.stdout + completed.stderr
    )
