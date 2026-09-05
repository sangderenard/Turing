"""The coordinator imports each unique KERNEL once and dispatches every
INVOCATION (region) to it with that region's own field slots.

Two byte-identical ``index_set`` regions collapse to one kernel file (see
``test_region_kernel_dedup``). This test carries that dedup through to the
resident coordinator: its WebAssembly binary imports a single kernel function
(not one per region), yet running it still mutates each region's *distinct*
buffer at that region's *distinct* slots. Kernel = shared bytes; invocation =
per-region slot binding. A program repeating one operation over N regions
imports O(unique kernels), not O(N), functions.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from src.common.tensors.fused_ir import FusedProgram, OpStep
from src.compiler.control_source import ControlProgram, StatementBlock
from src.compiler.wasm_class_modules import emit_control_region_modules
from src.compiler.wasm_class_coordinator import (
    build_class_inventory,
    emit_wasm_class_coordinator,
)


def _region(data_id, value_id, name):
    """``self.data[2] = value`` -- an in-place aliased scatter."""
    return FusedProgram(
        version=1, feeds={data_id, value_id},
        steps=[OpStep(0, "index_set", [data_id, value_id], {"slices": 2},
                      data_id + 900)],
        outputs={name: data_id + 900},
        extras={"capture_feed_origins": {
            data_id: {"binding_name": f"d{data_id}"},
            value_id: {"binding_name": f"v{value_id}"}}},
    )


def _two_region_coordinator():
    control = ControlProgram(
        StatementBlock(("__scheduled_region_0__", "__scheduled_region_1__")),
        region_indices=(0, 1),
    )
    modules, manifest = emit_control_region_modules(
        control, {0: _region(1, 2, "dataA"), 1: _region(11, 22, "dataB")},
        owner_name="buf", module_dir=".", dtype="int64",
    )
    inventory = build_class_inventory(manifest)
    coordinator = emit_wasm_class_coordinator(inventory, name="cc")
    return modules, manifest, inventory, coordinator


def _func_import_count(binary: bytes) -> int:
    """Count function imports in a WebAssembly binary's import section."""
    assert binary[:4] == b"\x00asm"
    pos = 8
    count = 0

    def uleb(p):
        result = shift = 0
        while True:
            byte = binary[p]
            p += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result, p
            shift += 7

    while pos < len(binary):
        section_id = binary[pos]
        pos += 1
        size, pos = uleb(pos)
        end = pos + size
        if section_id == 2:  # import section
            n, pos = uleb(pos)
            for _ in range(n):
                mlen, pos = uleb(pos)
                pos += mlen
                flen, pos = uleb(pos)
                pos += flen
                kind = binary[pos]
                pos += 1
                if kind == 0x00:  # func
                    count += 1
                    _, pos = uleb(pos)  # type index
                elif kind == 0x02:  # memory
                    flags = binary[pos]
                    pos += 1
                    _, pos = uleb(pos)  # min
                    if flags & 0x01:
                        _, pos = uleb(pos)  # max
                else:  # table/global -- not emitted here
                    raise AssertionError(f"unexpected import kind {kind}")
        pos = end
    return count


def test_two_invocations_share_one_kernel_import():
    _modules, manifest, inventory, coordinator = _two_region_coordinator()
    assert manifest["unique_kernels"] == 1
    assert len(inventory.methods) == 2
    # Both invocations reference the same kernel; the binary imports it once.
    assert inventory.methods[0].import_module == inventory.methods[1].import_module
    assert _func_import_count(coordinator.binary) == 1


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_deduped_coordinator_dispatches_each_invocation_to_its_own_slots(tmp_path):
    modules, manifest, inventory, coordinator = _two_region_coordinator()
    kernel_name = inventory.methods[0].import_module
    # Both regions are byte-identical, so either module's binary IS the kernel.
    assert modules[0].binary == modules[1].binary

    coord_wasm = tmp_path / "cc.wasm"
    coord_wasm.write_bytes(coordinator.binary)
    kernel_wasm = tmp_path / "kernel.wasm"
    kernel_wasm.write_bytes(modules[0].binary)

    entry = inventory.methods[0].entry
    script = tmp_path / "run.mjs"
    script.write_text(
        f"""
        import {{readFileSync}} from "node:fs";
        const memory = new WebAssembly.Memory({{initial: 2}});
        const kbytes = readFileSync(process.argv[2]);
        const kmod = await WebAssembly.instantiate(kbytes, {{env: {{memory}}}});
        // The coordinator imports exactly one kernel, keyed by kernel name.
        const imports = {{env: {{memory}}}};
        imports["{kernel_name}"] = {{"{entry}": kmod.instance.exports["{entry}"]}};
        const cbytes = readFileSync(process.argv[3]);
        const coord = await WebAssembly.instantiate(cbytes, imports);
        const mem = new BigInt64Array(memory.buffer);
        const i32 = new Int32Array(memory.buffer);
        // Field-slot table at byte 0: byte offsets of each field's buffer.
        // fields: 0=dataA 1=valueA 2=dataB 3=valueB
        i32[0]=64; i32[1]=128; i32[2]=192; i32[3]=256;
        [10n,20n,30n,40n].forEach((v,i)=>mem[8+i]=v);   // dataA  @ byte 64
        [99n,99n,99n,99n].forEach((v,i)=>mem[16+i]=v);  // valueA @ byte 128
        [10n,20n,30n,40n].forEach((v,i)=>mem[24+i]=v);  // dataB  @ byte 192
        [88n,88n,88n,88n].forEach((v,i)=>mem[32+i]=v);  // valueB @ byte 256
        coord.instance.exports.run_range(4, 0, 0, 2);
        const a=[],b=[];
        for (let i=0;i<4;i++) a.push(Number(mem[8+i]));
        for (let i=0;i<4;i++) b.push(Number(mem[24+i]));
        console.log(JSON.stringify([a,b]));
        """,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(kernel_wasm), str(coord_wasm)],
        capture_output=True, text=True, check=True,
    )
    # Each invocation mutated only its own buffer, at index 2, with its value.
    assert completed.stdout.strip().endswith("[[10,20,99,40],[10,20,88,40]]"), (
        completed.stdout + completed.stderr
    )
