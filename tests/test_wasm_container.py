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
from src.compiler.wasm_container import emit_nested_table_store


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
