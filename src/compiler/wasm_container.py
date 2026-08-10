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

# Shared coordinator<->kernel ABI. The coordinator reserves a 4-byte bump-cursor
# cell at this fixed offset (initialised to the first free heap byte) and every
# container kernel allocates against it. DEFAULT_MAP_CAPACITY is the fixed slot
# count each map is born with (no rehash yet -- a full map traps via the probe
# guard rather than looping).
HEAP_CURSOR_ADDR = 0
DEFAULT_MAP_CAPACITY = 1024

# i32 arithmetic/compare opcodes (the numerical kernel value type may be i64 or
# f64; addresses are always i32, so these are emitted raw).
_I32_ADD = 0x6A
_I32_MUL = 0x6C
_I32_EQZ = 0x45
_I32_REM_U = 0x6F
_I32_WRAP_I64 = 0xA7
_I32_GE_U = 0x4F
_UNREACHABLE = 0x00
# i64 opcodes for hashed keys/values (keys are arbitrary i64 -- e.g. RVAs).
_I64_EQZ = 0x50
_I64_EQ = 0x51
_I64_REM_U = 0x82
_I64_EXTEND_I32_U = 0xAD

# Open-addressing map block layout: [capacity:i32] then, from byte 8 (i64
# alignment), ``capacity`` slots of [state:i64, key:i64, value:i64] (24 bytes).
# state 0 = empty (fresh bump-heap memory is already zero), 1 = occupied.
_MAP_HEADER_BYTES = 8
_MAP_SLOT_BYTES = 24
_SLOT_STATE_OFF = 0
_SLOT_KEY_OFF = 8
_SLOT_VALUE_OFF = 16


def emit_map_new(
    builder: CodeBuilder,
    capacity: int,
    *,
    heap_cursor_addr: int,
    result_local: int,
) -> None:
    """Allocate an empty open-addressing map of ``capacity`` slots; leave its
    heap offset in ``result_local`` (i32). Slots are left zero (empty) by the
    bump heap; only the capacity header is written.
    """

    emit_bump_alloc(
        builder, _MAP_HEADER_BYTES + int(capacity) * _MAP_SLOT_BYTES,
        heap_cursor_addr=heap_cursor_addr, result_local=result_local,
    )
    builder.local_get(result_local).i32_const(int(capacity)).i32_store_width(32)


def _emit_probe_addr(builder: CodeBuilder, map_base_local: int, slot_local: int,
                     addr_local: int) -> None:
    """addr = map_base + header + slot * slot_bytes."""
    builder.local_get(map_base_local).i32_const(_MAP_HEADER_BYTES).raw(_I32_ADD)
    builder.local_get(slot_local).i32_const(_MAP_SLOT_BYTES).raw(_I32_MUL).raw(_I32_ADD)
    builder.local_set(addr_local)


def _emit_initial_slot(builder: CodeBuilder, key_local: int, cap_local: int,
                       slot_local: int) -> None:
    """slot = (u64)key % capacity, narrowed to i32."""
    builder.local_get(key_local)
    builder.local_get(cap_local).raw(_I64_EXTEND_I32_U)
    builder.raw(_I64_REM_U).raw(_I32_WRAP_I64)
    builder.local_set(slot_local)


def _emit_advance_slot(builder: CodeBuilder, slot_local: int, cap_local: int,
                       guard_local: int | None = None) -> None:
    """slot = (slot + 1) % capacity  (linear probe, wrapping).

    With ``guard_local`` (an i32 counter reset to 0 by the caller), trap via
    ``unreachable`` once the probe has visited ``capacity`` slots -- a full map
    with the key absent would otherwise loop forever. A trap is an honest,
    visible failure; capacity is a fixed-size limitation until rehashing lands.
    """
    if guard_local is not None:
        builder.local_get(guard_local).i32_const(1).raw(_I32_ADD).local_set(guard_local)
        builder.local_get(guard_local).local_get(cap_local).raw(_I32_GE_U)
        builder.if_()
        builder.raw(_UNREACHABLE)
        builder.end()
    builder.local_get(slot_local).i32_const(1).raw(_I32_ADD)
    builder.local_get(cap_local).raw(_I32_REM_U)
    builder.local_set(slot_local)


def emit_map_set(
    builder: CodeBuilder,
    *,
    map_base_local: int,
    key_local: int,
    value_local: int,
    cap_local: int,
    slot_local: int,
    addr_local: int,
    guard_local: int | None = None,
) -> None:
    """Emit ``map[key] = value`` by linear-probe open addressing. Overwrites an
    existing key or occupies the first empty slot. All the ``*_local`` args are
    caller-declared locals (key/value i64, the rest i32). ``guard_local`` (i32)
    traps on a full map instead of looping forever.
    """

    builder.local_get(map_base_local).i32_load().local_set(cap_local)
    if guard_local is not None:
        builder.i32_const(0).local_set(guard_local)
    _emit_initial_slot(builder, key_local, cap_local, slot_local)
    builder.block()          # depth 2 target: done
    builder.loop()           # depth 1: probe
    _emit_probe_addr(builder, map_base_local, slot_local, addr_local)
    # empty slot -> occupy (state=1, key, value), then leave the probe.
    builder.local_get(addr_local).i64_load(offset=_SLOT_STATE_OFF).raw(_I64_EQZ)
    builder.if_()
    builder.local_get(addr_local).i64_const(1).i64_store(offset=_SLOT_STATE_OFF)
    builder.local_get(addr_local).local_get(key_local).i64_store(offset=_SLOT_KEY_OFF)
    builder.local_get(addr_local).local_get(value_local).i64_store(offset=_SLOT_VALUE_OFF)
    builder.br(2)
    builder.end()
    # existing key -> overwrite value.
    builder.local_get(addr_local).i64_load(offset=_SLOT_KEY_OFF)
    builder.local_get(key_local).raw(_I64_EQ)
    builder.if_()
    builder.local_get(addr_local).local_get(value_local).i64_store(offset=_SLOT_VALUE_OFF)
    builder.br(2)
    builder.end()
    _emit_advance_slot(builder, slot_local, cap_local, guard_local)
    builder.br(0)            # continue probing
    builder.end()            # loop
    builder.end()            # block


def emit_map_get(
    builder: CodeBuilder,
    *,
    map_base_local: int,
    key_local: int,
    cap_local: int,
    slot_local: int,
    addr_local: int,
    result_local: int,
    guard_local: int | None = None,
) -> None:
    """Emit ``result = map.get(key, 0)`` by the same linear probe. A miss
    yields 0, which is an unused sentinel for handle values (heap offsets are
    always > 0). ``result_local`` is an i64 local. ``guard_local`` (i32) traps
    on a full map with the key absent instead of looping forever.
    """

    builder.local_get(map_base_local).i32_load().local_set(cap_local)
    if guard_local is not None:
        builder.i32_const(0).local_set(guard_local)
    _emit_initial_slot(builder, key_local, cap_local, slot_local)
    builder.block()          # done
    builder.loop()           # probe
    _emit_probe_addr(builder, map_base_local, slot_local, addr_local)
    # empty slot -> miss (result 0).
    builder.local_get(addr_local).i64_load(offset=_SLOT_STATE_OFF).raw(_I64_EQZ)
    builder.if_()
    builder.i64_const(0).local_set(result_local)
    builder.br(2)
    builder.end()
    # key match -> return its value.
    builder.local_get(addr_local).i64_load(offset=_SLOT_KEY_OFF)
    builder.local_get(key_local).raw(_I64_EQ)
    builder.if_()
    builder.local_get(addr_local).i64_load(offset=_SLOT_VALUE_OFF).local_set(result_local)
    builder.br(2)
    builder.end()
    _emit_advance_slot(builder, slot_local, cap_local, guard_local)
    builder.br(0)
    builder.end()            # loop
    builder.end()            # block


def emit_nested_map_store(
    builder: CodeBuilder,
    *,
    table_base_local: int,
    gx_local: int,
    gy_local: int,
    value_local: int,
    capacity: int,
    heap_cursor_addr: int,
    child_local: int,
    child_val_local: int,
    cap_local: int,
    slot_local: int,
    addr_local: int,
    guard_local: int | None = None,
) -> None:
    """Emit ``table[gx][gy] = value`` where both levels are open-addressing maps
    keyed by arbitrary i64 (the decoder's keys are RVAs/addresses, not bounded).
    The inner map is autovivified from the heap on first touch of ``gx``.

    ``gx_local``/``gy_local``/``value_local`` are i64 locals; ``child_local`` is
    an i32 (the child map's heap base); ``child_val_local`` is an i64 scratch
    (the handle as stored/read). The probe scratch (cap/slot/addr) is shared by
    the three map operations, which run sequentially.
    """

    # child = (i32) table.get(gx)
    emit_map_get(builder, map_base_local=table_base_local, key_local=gx_local,
                 cap_local=cap_local, slot_local=slot_local, addr_local=addr_local,
                 result_local=child_val_local, guard_local=guard_local)
    builder.local_get(child_val_local).raw(_I32_WRAP_I64).local_set(child_local)
    # if child == 0: autovivify an inner map and link table[gx] = child.
    builder.local_get(child_local).raw(_I32_EQZ)
    builder.if_()
    emit_map_new(builder, capacity, heap_cursor_addr=heap_cursor_addr,
                 result_local=child_local)
    builder.local_get(child_local).raw(_I64_EXTEND_I32_U).local_set(child_val_local)
    emit_map_set(builder, map_base_local=table_base_local, key_local=gx_local,
                 value_local=child_val_local, cap_local=cap_local,
                 slot_local=slot_local, addr_local=addr_local, guard_local=guard_local)
    builder.end()
    # child[gy] = value
    emit_map_set(builder, map_base_local=child_local, key_local=gy_local,
                 value_local=value_local, cap_local=cap_local,
                 slot_local=slot_local, addr_local=addr_local, guard_local=guard_local)


def emit_nested_map_get(
    builder: CodeBuilder,
    *,
    table_base_local: int,
    gx_local: int,
    gy_local: int,
    result_local: int,
    child_local: int,
    child_val_local: int,
    cap_local: int,
    slot_local: int,
    addr_local: int,
    guard_local: int | None = None,
) -> None:
    """Emit ``result = table[gx][gy]`` (0 on any miss). The read counterpart of
    ``emit_nested_map_store``. A missing outer key yields a 0 child handle, so
    the inner lookup is guarded -- probing an unallocated map (capacity 0) would
    divide by zero. All the ``*_local`` args are caller-declared locals.
    """

    builder.i64_const(0).local_set(result_local)  # default: miss
    emit_map_get(builder, map_base_local=table_base_local, key_local=gx_local,
                 cap_local=cap_local, slot_local=slot_local, addr_local=addr_local,
                 result_local=child_val_local, guard_local=guard_local)
    builder.local_get(child_val_local).raw(_I32_WRAP_I64).local_set(child_local)
    builder.local_get(child_local).raw(_I32_EQZ)
    builder.if_()          # child == 0: leave result at 0
    builder.else_()        # child present: result = child[gy]
    emit_map_get(builder, map_base_local=child_local, key_local=gy_local,
                 cap_local=cap_local, slot_local=slot_local, addr_local=addr_local,
                 result_local=result_local, guard_local=guard_local)
    builder.end()


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
