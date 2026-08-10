"""A heap + integer-keyed nested-container ABI for WebAssembly kernels.

The decoder's lookup tables (x86 opcode maps, gray-code tables) are captured as
Python dict/list-of-list containers with no tensor shape, so a store like
``table[gx][gy] = value`` cannot be flattened against a static stride the way a
shaped tensor scatter can (see ``fused_program_wasm_backend._store_dimension_strides``).
This module is the substrate that lets such stores lower for real: a bump heap
in linear memory plus an autovivifying two-level table.

Layout
------
A fixed **heap-cursor cell** (one i32 at ``heap_cursor_addr``) holds the next
free heap byte offset. ``bump_alloc(size)`` returns the current cursor and
advances it -- the same discipline ``amd64_machine_semantics`` uses for its
guest arena.

A **table** is a heap block of ``ncols`` i32 child handles (0 = empty). A
**row** is a heap block of ``ncols`` value cells (i64 by default). Addressing a
key is direct integer indexing (the decoder's keys are small integers: opcode
bytes 0..255, gray indices), so no hashing is needed; a missing row is
autovivified on first write, exactly like ``collections.defaultdict``.

The store ``table[gx][gy] = value`` is therefore:

    row = table[gx]
    if row == 0:           # autovivify
        row = bump_alloc(ncols * value_width)
        table[gx] = row
    row[gy] = value

Everything is emitted inline into the calling kernel's ``CodeBuilder`` -- region
kernels are single-function modules, so there are no helper functions to import.
"""
from __future__ import annotations

from .wasm_binary import CodeBuilder

# i32 arithmetic/compare opcodes (the numerical kernel value type may be i64 or
# f64; addresses are always i32, so these are emitted raw).
_I32_ADD = 0x6A
_I32_MUL = 0x6C
_I32_EQZ = 0x45


def emit_bump_alloc(
    builder: CodeBuilder,
    size: int,
    *,
    heap_cursor_addr: int,
    result_local: int,
) -> None:
    """Allocate ``size`` bytes from the bump heap; leave the offset in
    ``result_local`` (an i32 local). Reads the cursor, keeps it as the result,
    and writes ``cursor + size`` back.
    """

    # result_local = load(heap_cursor_addr)
    builder.i32_const(heap_cursor_addr).i32_load()
    builder.local_set(result_local)
    # store(heap_cursor_addr, result_local + size)
    builder.i32_const(heap_cursor_addr)
    builder.local_get(result_local).i32_const(int(size)).raw(_I32_ADD)
    builder.i32_store_width(32)


def emit_nested_table_store(
    builder: CodeBuilder,
    *,
    table_base_local: int,
    gx_local: int,
    gy_local: int,
    value_local: int,
    ncols: int,
    heap_cursor_addr: int,
    row_ptr_local: int,
    alloc_tmp_local: int,
    value_width: int = 8,
    handle_width: int = 4,
) -> None:
    """Emit ``table[gx][gy] = value`` with autovivification.

    ``table_base_local`` is an i32 local holding the table block's heap offset;
    ``gx_local``/``gy_local`` are i32 index locals; ``value_local`` holds the
    stored value in the kernel's value type. ``row_ptr_local`` and
    ``alloc_tmp_local`` are scratch i32 locals the caller has declared.
    """

    # row_ptr = i32.load(table_base + gx * handle_width)
    builder.local_get(table_base_local)
    builder.local_get(gx_local).i32_const(handle_width).raw(_I32_MUL).raw(_I32_ADD)
    builder.i32_load()
    builder.local_set(row_ptr_local)

    # if row_ptr == 0: autovivify a fresh row and link it into table[gx].
    builder.local_get(row_ptr_local).raw(_I32_EQZ)
    builder.if_()
    emit_bump_alloc(
        builder, ncols * value_width,
        heap_cursor_addr=heap_cursor_addr, result_local=alloc_tmp_local,
    )
    builder.local_get(alloc_tmp_local).local_set(row_ptr_local)
    # table[gx] = row_ptr   (i32.store at table_base + gx*handle_width)
    builder.local_get(table_base_local)
    builder.local_get(gx_local).i32_const(handle_width).raw(_I32_MUL).raw(_I32_ADD)
    builder.local_get(row_ptr_local)
    builder.i32_store_width(32)
    builder.end()

    # row[gy] = value   (store at row_ptr + gy*value_width)
    builder.local_get(row_ptr_local)
    builder.local_get(gy_local).i32_const(value_width).raw(_I32_MUL).raw(_I32_ADD)
    builder.local_get(value_local)
    if value_width == 8:
        builder.i64_store()
    else:
        builder.i64_store_width(value_width * 8)
