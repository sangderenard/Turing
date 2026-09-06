"""The heap + nested-container ABI: ``table[gx][gy] = value`` with a bump heap.

This is the substrate for lowering the decoder's shapeless nested-container
stores (x86 opcode maps, gray-code tables) that a shaped-tensor scatter cannot
address. A two-level table autovivifies rows from a bump heap and stores by
direct integer indexing. Verified in Node: distinct rows get distinct heap
blocks, a re-touched row is reused (not reallocated), and each cell holds its
own value.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from src.compiler.wasm_binary import CodeBuilder, build_module
from src.compiler.wasm_container import (
    emit_map_get,
    emit_map_new,
    emit_map_set,
    emit_nested_table_store,
)


_HEAP_CURSOR_ADDR = 0
_TABLE_BASE = 64
_HEAP_BASE = 1024
_NCOLS = 8


def _build_store_module() -> bytes:
    """``run(table_base, gx, gy, value)`` performs one nested-table store."""
    body = CodeBuilder(value_type="i64", parameter_count=4)
    row_ptr = body.declare_local("i32")
    alloc_tmp = body.declare_local("i32")
    emit_nested_table_store(
        body,
        table_base_local=0, gx_local=1, gy_local=2, value_local=3,
        ncols=_NCOLS, heap_cursor_addr=_HEAP_CURSOR_ADDR,
        row_ptr_local=row_ptr, alloc_tmp_local=alloc_tmp,
    )
    return build_module(
        function_name="run",
        parameter_types=["i32", "i32", "i32", "i64"],
        body=body,
        memory_pages=1,
    )


def test_nested_table_store_module_is_valid_wasm():
    binary = _build_store_module()
    assert binary[:4] == b"\x00asm"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_nested_table_store_autovivifies_and_indexes(tmp_path):
    binary = _build_store_module()
    wasm = tmp_path / "container.wasm"
    wasm.write_bytes(binary)
    script = tmp_path / "run.mjs"
    script.write_text(
        f"""
        import {{readFileSync}} from "node:fs";
        const mod = await WebAssembly.instantiate(readFileSync(process.argv[2]), {{}});
        const {{run, memory}} = mod.instance.exports;
        const i32 = new Int32Array(memory.buffer);
        const i64 = new BigInt64Array(memory.buffer);
        // Initialise the bump cursor; the table block starts zeroed.
        i32[{_HEAP_CURSOR_ADDR} / 4] = {_HEAP_BASE};
        run(  {_TABLE_BASE}, 2, 3, 77n);   // autovivify row 2
        run(  {_TABLE_BASE}, 2, 5, 88n);   // reuse row 2
        run(  {_TABLE_BASE}, 4, 1, 99n);   // autovivify row 4
        const handle2 = i32[({_TABLE_BASE} + 2 * 4) / 4];
        const handle4 = i32[({_TABLE_BASE} + 4 * 4) / 4];
        const handle1 = i32[({_TABLE_BASE} + 1 * 4) / 4];
        const cursor  = i32[{_HEAP_CURSOR_ADDR} / 4];
        const out = {{
          handle2, handle4, handle1, cursor,
          row2_3: Number(i64[(handle2 + 3 * 8) / 8]),
          row2_5: Number(i64[(handle2 + 5 * 8) / 8]),
          row4_1: Number(i64[(handle4 + 1 * 8) / 8]),
          rows_distinct: handle2 !== handle4,
        }};
        console.log(JSON.stringify(out));
        """,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(wasm)], capture_output=True, text=True, check=True,
    )
    import json
    out = json.loads(completed.stdout)
    assert out["handle2"] == _HEAP_BASE, out                     # first row
    assert out["handle4"] == _HEAP_BASE + _NCOLS * 8, out        # second row
    assert out["handle1"] == 0, out                             # untouched key
    assert out["cursor"] == _HEAP_BASE + 2 * _NCOLS * 8, out     # two rows allocated
    assert out["rows_distinct"] is True, out
    assert out["row2_3"] == 77, out
    assert out["row2_5"] == 88, out                             # row 2 reused
    assert out["row4_1"] == 99, out


def _build_map_set_get_module() -> bytes:
    """``run(map_base, key, value, out_addr)``: set then read back to out_addr.

    The map header (capacity) is placed by the harness; slots start zero.
    """
    body = CodeBuilder(value_type="i64", parameter_count=4)
    cap = body.declare_local("i32")
    slot = body.declare_local("i32")
    addr = body.declare_local("i32")
    result = body.declare_local("i64")
    emit_map_set(body, map_base_local=0, key_local=1, value_local=2,
                 cap_local=cap, slot_local=slot, addr_local=addr)
    emit_map_get(body, map_base_local=0, key_local=1,
                 cap_local=cap, slot_local=slot, addr_local=addr, result_local=result)
    body.local_get(3).local_get(result).i64_store()  # *out_addr = result
    return build_module(
        function_name="run",
        parameter_types=["i32", "i64", "i64", "i32"],
        body=body, memory_pages=1,
    )


def test_map_module_is_valid_wasm():
    assert _build_map_set_get_module()[:4] == b"\x00asm"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_map_set_get_overwrite_and_probe(tmp_path):
    binary = _build_map_set_get_module()
    wasm = tmp_path / "map.wasm"
    wasm.write_bytes(binary)
    # A capacity-4 map at byte 64; colliding keys 1,5,9 (all %4==1) force probing.
    # out cell at byte 32.
    script = tmp_path / "run.mjs"
    script.write_text(
        """
        import {readFileSync} from "node:fs";
        const mod = await WebAssembly.instantiate(readFileSync(process.argv[2]), {});
        const {run, memory} = mod.instance.exports;
        const i32 = new Int32Array(memory.buffer);
        const MAP = 64, CAP = 4, OUT = 32;
        i32[MAP / 4] = CAP;                 // capacity header; slots stay zero
        const results = {};
        function setget(key, value) {
          run(MAP, BigInt(key), BigInt(value), OUT);
          const i64 = new BigInt64Array(memory.buffer);
          return Number(i64[OUT / 8]);
        }
        results.k1  = setget(1, 100);       // slot 1
        results.k5  = setget(5, 500);       // collides -> slot 2
        results.k9  = setget(9, 900);       // collides -> slot 3
        results.k1b = setget(1, 111);       // overwrite existing key 1
        // Re-read all three after the overwrite to prove probing is stable.
        const i64 = new BigInt64Array(memory.buffer);
        // read via get-only by setting same value back is destructive; instead
        // scan the slots directly: slot i at MAP+8 + i*24, [state,key,value].
        const slots = [];
        for (let i = 0; i < CAP; i++) {
          const base = (MAP + 8 + i * 24) / 8;
          slots.push([Number(i64[base]), Number(i64[base + 1]), Number(i64[base + 2])]);
        }
        console.log(JSON.stringify({results, slots}));
        """,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(wasm)], capture_output=True, text=True, check=True,
    )
    import json
    out = json.loads(completed.stdout)
    r = out["results"]
    assert r["k1"] == 100 and r["k5"] == 500 and r["k9"] == 900, out
    assert r["k1b"] == 111, out                      # overwrite returns new value
    slots = out["slots"]
    # slot 1 = key 1 (overwritten to 111), slot 2 = key 5, slot 3 = key 9.
    assert slots[1] == [1, 1, 111], slots
    assert slots[2] == [1, 5, 500], slots
    assert slots[3] == [1, 9, 900], slots
    assert slots[0][0] == 0, slots                   # slot 0 never occupied


def _build_nested_map_module(capacity: int, heap_cursor_addr: int) -> bytes:
    """``run(table_base, gx, gy, value, out_addr)``: nested-map store then read
    back table[gx][gy] into out_addr. Both levels are open-addressing maps."""
    from src.compiler.wasm_container import emit_nested_map_store
    body = CodeBuilder(value_type="i64", parameter_count=5)
    child = body.declare_local("i32")
    child_val = body.declare_local("i64")
    cap = body.declare_local("i32")
    slot = body.declare_local("i32")
    addr = body.declare_local("i32")
    result = body.declare_local("i64")
    child2 = body.declare_local("i32")
    emit_nested_map_store(
        body, table_base_local=0, gx_local=1, gy_local=2, value_local=3,
        capacity=capacity, heap_cursor_addr=heap_cursor_addr,
        child_local=child, child_val_local=child_val,
        cap_local=cap, slot_local=slot, addr_local=addr,
    )
    # read back: child2 = table.get(gx); result = child2.get(gy)
    emit_map_get(body, map_base_local=0, key_local=1, cap_local=cap,
                 slot_local=slot, addr_local=addr, result_local=child_val)
    body.local_get(child_val).raw(0xA7).local_set(child2)  # i32.wrap_i64
    emit_map_get(body, map_base_local=child2, key_local=2, cap_local=cap,
                 slot_local=slot, addr_local=addr, result_local=result)
    body.local_get(4).local_get(result).i64_store()
    return build_module(
        function_name="run",
        parameter_types=["i32", "i64", "i64", "i64", "i32"],
        body=body, memory_pages=2,
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_nested_map_store_with_arbitrary_rva_keys(tmp_path):
    CAP, CURSOR, TABLE, HEAP, OUT = 8, 0, 64, 4096, 32
    binary = _build_nested_map_module(CAP, CURSOR)
    wasm = tmp_path / "nested_map.wasm"
    wasm.write_bytes(binary)
    script = tmp_path / "run.mjs"
    script.write_text(
        f"""
        import {{readFileSync}} from "node:fs";
        const mod = await WebAssembly.instantiate(readFileSync(process.argv[2]), {{}});
        const {{run, memory}} = mod.instance.exports;
        const i32 = new Int32Array(memory.buffer);
        i32[{CURSOR} / 4] = {HEAP};        // bump cursor
        i32[{TABLE} / 4] = {CAP};          // top-level map capacity header
        function store(gx, gy, v) {{
          run({TABLE}, BigInt(gx), BigInt(gy), BigInt(v), {OUT});
          const i64 = new BigInt64Array(memory.buffer);
          return Number(i64[{OUT} / 8]);
        }}
        // RVA-scale keys: unbounded, would overflow any dense array.
        const a = store(0x401000, 0x10, 111);   // autovivify row 0x401000
        const b = store(0x401000, 0x20, 222);    // reuse that row, new col
        const c = store(0x402000, 0x10, 333);    // different row, same col key
        console.log(JSON.stringify({{a, b, c}}));
        """,
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(script), str(wasm)], capture_output=True, text=True, check=True,
    )
    import json
    out = json.loads(completed.stdout)
    # Each read-back returns exactly what was stored, across autovivified rows.
    assert out == {"a": 111, "b": 222, "c": 333}, out
