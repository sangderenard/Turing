"""A shapeless nested-container IndexedStore lowers to the heap map ABI.

Region 115 of the decoder build is exactly this: ``table[gx][gy] = value`` where
the target is a dict/list container (unbounded RVA keys, no tensor shape). The
emitter routes such a pure store to a dedicated kernel that reads the scalar
operands once and mutates a heap open-addressing map -- no per-cell array walk.
A shaped store still takes the strided tensor scatter (asserted elsewhere).

The coordinator (a later step) seeds the container field with a map header and
the bump cursor; here the harness seeds them to verify the kernel itself.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from src.common.tensors.fused_ir import FusedProgram, OpStep, Meta
from src.compiler.fused_program_wasm_backend import emit_wasm_module
from src.compiler.wasm_container import DEFAULT_MAP_CAPACITY, HEAP_CURSOR_ADDR


def _region_115_program():
    return FusedProgram(
        version=1, feeds={10, 11, 12, 13},
        steps=[OpStep(0, "IndexedStore", [10, 11, 12, 13],
                      {"source_type": "SubscriptStore"}, 14)],
        outputs={"value_136": 14}, meta={},
        extras={"capture_feed_origins": {
            10: {"binding_name": "data"}, 11: {"binding_name": "i"},
            12: {"binding_name": "j"}, 13: {"binding_name": "value"}}},
    )


def test_shapeless_2index_store_emits_a_complete_container_kernel():
    module = emit_wasm_module(_region_115_program(), name="r115", dtype="float64")
    assert module.complete, module.shortfall_report()
    assert module.parameters == ("$count", "$data", "$i", "$j", "$value", "$out0")
    assert module.binary and module.binary[:4] == b"\x00asm"


def test_single_subscript_shapeless_store_stays_on_the_strided_path():
    # A single shapeless subscript is ambiguous (a 1-D buffer scatter looks the
    # same in region meta as a single dict store), so it is NOT routed to the
    # container map -- it keeps the strided lowering. The container path is only
    # taken for the unambiguous two-subscript nested case.
    program = FusedProgram(
        version=1, feeds={0, 1, 2},
        steps=[OpStep(0, "IndexedStore", [0, 1, 2],
                      {"source_type": "SubscriptStore"}, 3)],
        outputs={"d": 3}, meta={},
        extras={"capture_feed_origins": {
            0: {"binding_name": "data"}, 1: {"binding_name": "key"},
            2: {"binding_name": "value"}}},
    )
    module = emit_wasm_module(program, name="r1", dtype="float64")
    assert module.complete, module.shortfall_report()
    # It is the strided scatter, not the container map: no placeholder WAT, and
    # it emits a real per-cell store loop.
    assert "container store lowered" not in module.source
    assert "f64.store" in module.source


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_container_store_kernel_runs_the_nested_map_insert(tmp_path):
    module = emit_wasm_module(_region_115_program(), name="r115", dtype="float64")
    wasm = tmp_path / "r115.wasm"
    wasm.write_bytes(module.binary)
    # Memory plan (bytes): cursor cell at HEAP_CURSOR_ADDR; a top-level map at
    # TABLE seeded with a small capacity header; scalar operand cells; the heap
    # (for the autovivified inner map of DEFAULT_MAP_CAPACITY) starts at HEAP.
    TABLE, TOP_CAP = 512, 8
    KEY0, KEY1, VALUE, OUT = 64, 128, 192, 256
    HEAP = 4096
    script = tmp_path / "run.mjs"
    script.write_text(
        f"""
        import {{readFileSync}} from "node:fs";
        const mod = await WebAssembly.instantiate(readFileSync(process.argv[2]), {{}});
        const {{run, memory}} = mod.instance.exports;
        const i32 = new Int32Array(memory.buffer);
        const i64 = new BigInt64Array(memory.buffer);
        i32[{HEAP_CURSOR_ADDR} / 4] = {HEAP};   // bump cursor
        i32[{TABLE} / 4] = {TOP_CAP};           // top-level map capacity header
        const k0 = 0x401000n, k1 = 0x20n, v = 12345n;
        i64[{KEY0} / 8] = k0; i64[{KEY1} / 8] = k1; i64[{VALUE} / 8] = v;
        run(1, {TABLE}, {KEY0}, {KEY1}, {VALUE}, {OUT});
        // Mirror the kernel's linear-probe lookup to read the value back.
        function mapGet(base, key, cap) {{
          let slot = Number(BigInt.asUintN(64, key) % BigInt(cap));
          for (let n = 0; n < cap; n++) {{
            const addr = base + 8 + slot * 24;
            const state = i64[addr / 8];
            if (state === 0n) return null;
            if (i64[addr / 8 + 1] === BigInt.asIntN(64, key)) return i64[addr / 8 + 2];
            slot = (slot + 1) % cap;
          }}
          return null;
        }}
        const child = mapGet({TABLE}, k0, {TOP_CAP});
        const value = child === null ? null
          : mapGet(Number(child), k1, {DEFAULT_MAP_CAPACITY});
        console.log(JSON.stringify({{
          childHandle: child === null ? null : Number(child),
          value: value === null ? null : Number(value),
          heap: HEAP,
        }}));
        """.replace("HEAP", str(HEAP)),
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(wasm)], capture_output=True, text=True, check=True,
    )
    import json
    out = json.loads(completed.stdout)
    # The inner map was autovivified from the heap and holds the stored value.
    assert out["childHandle"] == HEAP, out
    assert out["value"] == 12345, out


def test_string_keyed_single_dict_store_lowers_to_a_map():
    # d['metadata'] = value -- region 116. The string key materialises as a
    # tensor_from_list constant; a string key is unambiguously a dict, so the
    # single-subscript store takes the container path (string hashed to an i64).
    program = FusedProgram(
        version=1, feeds={100, 102},
        steps=[OpStep(0, "tensor_from_list", [], {"values": "metadata"}, 101),
               OpStep(1, "IndexedStore", [100, 101, 102],
                      {"source_type": "SubscriptStore"}, 103)],
        outputs={"value_112": 103}, meta={},
        extras={"capture_feed_origins": {
            100: {"binding_name": "data"}, 102: {"binding_name": "value"}}},
    )
    module = emit_wasm_module(program, name="r116", dtype="float64")
    assert module.complete, module.shortfall_report()
    # The string constant is consumed as an immediate key, not a tensor param.
    assert module.parameters == ("$count", "$data", "$value", "$out0")
    assert "container store lowered" in module.source


def _nested_read_program():
    # value = table[gx][gy]: two Indexed gathers rooted at a container feed.
    return FusedProgram(
        version=1, feeds={20, 21, 22},
        steps=[OpStep(0, "Indexed", [20, 21], {"source_type": "Subscript"}, 30),
               OpStep(1, "Indexed", [30, 22], {"source_type": "Subscript"}, 31)],
        outputs={"value": 31}, meta={},
        extras={"capture_feed_origins": {
            20: {"binding_name": "table"}, 21: {"binding_name": "gx"},
            22: {"binding_name": "gy"}}},
    )


def test_nested_container_read_lowers():
    module = emit_wasm_module(_nested_read_program(), name="rd", dtype="float64")
    assert module.complete, module.shortfall_report()
    assert module.parameters == ("$count", "$table", "$gx", "$gy", "$out0")
    assert "container read lowered" in module.source


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_container_store_then_read_round_trips(tmp_path):
    # Store table[k0][k1]=v with the store kernel, then read it back with the
    # read kernel over the same linear memory: the two heap-map kernels agree on
    # layout and hashing.
    store = emit_wasm_module(_region_115_program(), name="st", dtype="float64")
    read = emit_wasm_module(_nested_read_program(), name="rd", dtype="float64")
    (tmp_path / "st.wasm").write_bytes(store.binary)
    (tmp_path / "rd.wasm").write_bytes(read.binary)
    script = tmp_path / "run.mjs"
    script.write_text(
        """
        import {readFileSync} from "node:fs";
        const st = await WebAssembly.instantiate(readFileSync(process.argv[2]), {});
        const sMem = st.instance.exports.memory;
        const s32 = new Int32Array(sMem.buffer), s64 = new BigInt64Array(sMem.buffer);
        const CURSOR=0, TABLE=512, TOP_CAP=8, K0=64, K1=128, VAL=192, OUT=256, HEAP=4096;
        s32[CURSOR/4]=HEAP; s32[TABLE/4]=TOP_CAP;
        const k0=0x401000n, k1=0x20n, v=98765n;
        s64[K0/8]=k0; s64[K1/8]=k1; s64[VAL/8]=v;
        st.instance.exports.run(1, TABLE, K0, K1, VAL, OUT);
        // Move the store's linear memory into the read instance and look it up.
        const rd = await WebAssembly.instantiate(readFileSync(process.argv[3]), {});
        const rMem = rd.instance.exports.memory;
        if (rMem.buffer.byteLength < sMem.buffer.byteLength)
          rMem.grow((sMem.buffer.byteLength - rMem.buffer.byteLength + 65535) >> 16);
        new Uint8Array(rMem.buffer).set(new Uint8Array(sMem.buffer));
        const r64 = new BigInt64Array(rMem.buffer);
        const RK0=64, RK1=128, ROUT=256;
        r64[RK0/8]=k0; r64[RK1/8]=k1;
        rd.instance.exports.run(1, TABLE, RK0, RK1, ROUT);
        console.log(new BigInt64Array(rMem.buffer)[ROUT/8].toString());
        """,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(tmp_path/"st.wasm"), str(tmp_path/"rd.wasm")],
        capture_output=True, text=True, check=True,
    )
    assert completed.stdout.strip() == "98765", completed.stdout + completed.stderr
