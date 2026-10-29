"""Lower SSA sequence/table policies into ordinary memory and control SSA.

This module contains no runtime collection adapter.  It specializes a compact
``SSASequenceDescriptor`` into functions made solely from the repository's
normal CFG and address vocabulary.  A list is a descriptor with no key
columns; a set is the same storage with a unique key; a dictionary is a
multi-column row with one or more unique key columns.

Fixed arenas retain their existing capacity/status semantics. Typed scalar
native pointer-cell arenas with an explicit heap owner also support checked
geometric reserve, insert, table store and extend. Other dynamic contracts
produce a typed shortfall rather than truncating or delegating to Python.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import prod

from ..transmogrifier.ssa import (
    BasicBlock,
    Function,
    Instr,
    SSASequenceCapacityPolicy,
    SSAChildTablePoolDescriptor,
    SSASequenceDescriptor,
    SSASequenceTable,
    SSAValue,
)
from .monotonic_ids import GLOBAL_MONOTONIC_IDS


class SSASequenceShortfallCode(str, Enum):
    DYNAMIC_GROWTH_UNAVAILABLE = "dynamic-growth-unavailable"
    READ_ONLY_DESTINATION = "read-only-destination"
    SHAPED_KEY_UNAVAILABLE = "shaped-key-unavailable"


@dataclass(frozen=True)
class SSASequenceLoweringShortfall:
    code: SSASequenceShortfallCode
    sequence_id: int
    operation: str
    reason: str


@dataclass(frozen=True)
class SSASequenceLowering:
    functions: tuple[Function, ...] = ()
    shortfalls: tuple[SSASequenceLoweringShortfall, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.shortfalls

    @property
    def entry(self) -> Function:
        """The helper the caller calls: the one no other helper here calls.

        A lowering may return its private dependencies alongside the entry
        (``lookup, append, extend``; a dynamic ``insert`` with its fixed
        insert and reserve).  Their order is not a contract -- the dynamic
        lowering listed its entry first while every caller took the last
        one, so ``xs.append`` called ``__reserve``.  The call edges say
        which function is the entry.
        """

        names = {function.name for function in self.functions}
        called = {
            str(instruction.attributes.get("callee"))
            for function in self.functions
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op == "Call"
            and str(instruction.attributes.get("callee")) in names
        }
        roots = [
            function for function in self.functions
            if function.name not in called
        ]
        if len(roots) != 1:
            raise ValueError(
                "sequence lowering has no unique entry helper: "
                f"{[function.name for function in roots]!r}"
            )
        return roots[0]


def schedule_joined_sequence_mutations(function: Function) -> int:
    """Place a list-of-bytes dual-view update after its source is complete.

    Sequence concatenation is itself lowered into resident reset/append-slice
    operations. A joined-list append that consumes that result must follow
    those operations in the same CFG compartment; source position alone is
    insufficient because the concatenation has no standalone control block.
    This pass moves only a proven local unit and never crosses a basic-block
    boundary.
    """

    moved = 0
    for block in function.blocks.values():
        index = 0
        while index < len(block.instrs):
            extend = block.instrs[index]
            attributes = extend.attributes or {}
            if (
                extend.op not in {"Call", "call"}
                or attributes.get("ssa_sequence_operation")
                != "extend_joined_bytes"
                or attributes.get("joined_source_sequence_id") is None
            ):
                index += 1
                continue
            source_sequence_id = int(
                attributes["joined_source_sequence_id"]
            )
            effect_id = attributes.get("source_effect_node_id")
            count_index = next((
                candidate
                for candidate in range(index - 1, -1, -1)
                if (
                    block.instrs[candidate].attributes.get(
                        "ssa_sequence_operation"
                    ) == "append_joined_count"
                    and block.instrs[candidate].attributes.get(
                        "source_effect_node_id"
                    ) == effect_id
                )
            ), None)
            if count_index is None:
                index += 1
                continue
            unit_results = {
                int(instruction.res.id)
                for instruction in block.instrs[count_index:index + 1]
                if instruction.res is not None
            }
            unit_end = index + 1
            while unit_end < len(block.instrs) and (
                block.instrs[unit_end].op in {"Store", "store"}
                and any(
                    int(argument.id) in unit_results
                    for argument in block.instrs[unit_end].args
                )
            ):
                unit_end += 1
            source_indices = [
                candidate
                for candidate, instruction in enumerate(block.instrs)
                if candidate >= unit_end
                and int(instruction.attributes.get("sequence_id", -1))
                == source_sequence_id
            ]
            if not source_indices:
                index = unit_end
                continue
            source_end = max(source_indices) + 1
            source_results = {
                int(block.instrs[candidate].res.id)
                for candidate in source_indices
                if block.instrs[candidate].res is not None
            }
            while source_end < len(block.instrs) and any(
                int(argument.id) in source_results
                for argument in block.instrs[source_end].args
            ):
                if block.instrs[source_end].res is not None:
                    source_results.add(int(block.instrs[source_end].res.id))
                source_end += 1
            unit = block.instrs[count_index:unit_end]
            del block.instrs[count_index:unit_end]
            insertion = source_end - len(unit)
            for instruction in unit:
                instruction.attributes["joined_sequence_scheduled_after_source"] = (
                    source_sequence_id
                )
            block.instrs[insertion:insertion] = unit
            moved += 1
            index = insertion + len(unit)
    if moved:
        function.metadata["scheduled_joined_sequence_mutations"] = moved
    return moved


class _Builder:
    def __init__(self, first_value_id: int) -> None:
        self.blocks: dict[str, BasicBlock] = {}

    def block(self, name: str) -> BasicBlock:
        block = BasicBlock(name)
        self.blocks[name] = block
        return block

    def fresh(
        self, dtype: str | None = None, shape: tuple[int, ...] = (),
    ) -> SSAValue:
        return SSAValue(
            GLOBAL_MONOTONIC_IDS.mint(), dtype=dtype, shape=tuple(shape),
        )

    def emit(
        self,
        block: BasicBlock,
        op: str,
        args: list[SSAValue],
        res: SSAValue | None = None,
        *,
        attributes: dict | None = None,
    ) -> Instr:
        instruction = Instr(op, args, res, attributes=attributes or {})
        block.instrs.append(instruction)
        return instruction

    def const(self, block: BasicBlock, literal: int, *, dtype: str = "int") -> SSAValue:
        result = self.fresh(dtype)
        self.emit(block, "Const", [], result, attributes={"value": int(literal)})
        return result

    def branch(self, block: BasicBlock, target: BasicBlock) -> None:
        self.emit(block, "Br", [], attributes={"target": target.name})
        block.successors.append(target.name)

    def cond(
        self,
        block: BasicBlock,
        condition: SSAValue,
        if_true: BasicBlock,
        if_false: BasicBlock,
    ) -> None:
        self.emit(
            block,
            "CondBr",
            [condition],
            attributes={
                "true_target": if_true.name,
                "false_target": if_false.name,
            },
        )
        block.successors.extend((if_true.name, if_false.name))


def _child_pool_identity_ids(
    pool: SSAChildTablePoolDescriptor | None,
) -> tuple[int, ...]:
    """Return every SSA identity reserved by a child-table descriptor."""

    if pool is None:
        return ()
    return tuple(int(value_id) for value_id in (
        *pool.column_value_ids,
        pool.length_value_id,
        pool.capacity_value_id,
        pool.row_stride_value_id,
        *((pool.shape_value_id,) if pool.shape_value_id is not None else ()),
        *((pool.rank_value_id,) if pool.rank_value_id is not None else ()),
        *((pool.shape_stride_value_id,) if pool.shape_stride_value_id is not None else ()),
        *((pool.status_value_id,) if pool.status_value_id is not None else ()),
        *((pool.live_flags_value_id,) if pool.live_flags_value_id is not None else ()),
    ))


def _sequence_descriptor_identity_ids(
    descriptor: SSASequenceDescriptor,
) -> tuple[int, ...]:
    """Return storage and compile-time identities owned by a descriptor.

    Some identities, notably status cells and the sequence's own identity, are
    not arguments of every generated helper.  They still occupy the enclosing
    program's deterministic SSA namespace and therefore must never be minted
    again for a helper-local value.
    """

    return tuple(int(value_id) for value_id in (
        descriptor.sequence_id,
        *descriptor.column_value_ids,
        descriptor.length_address_id,
        descriptor.capacity_value_id,
        *((descriptor.status_address_id,) if descriptor.status_address_id is not None else ()),
        *((descriptor.live_flags_value_id,) if descriptor.live_flags_value_id is not None else ()),
        *((descriptor.heap_owner_value_id,) if descriptor.heap_owner_value_id is not None else ()),
        *_child_pool_identity_ids(descriptor.child_table_pool),
    ))


def _first_fresh_identity(
    *descriptors: SSASequenceDescriptor,
    values: tuple[SSAValue, ...] = (),
    pools: tuple[SSAChildTablePoolDescriptor, ...] = (),
    minimum: int | None = None,
) -> int:
    """Choose the deterministic first ID above every identity in scope."""

    reserved = {
        *(value.id for value in values),
        *(
            value_id
            for descriptor in descriptors
            for value_id in _sequence_descriptor_identity_ids(descriptor)
        ),
        *(
            value_id
            for pool in pools
            for value_id in _child_pool_identity_ids(pool)
        ),
    }
    return max(max(reserved, default=-1) + 1, int(minimum or 0))


def _storage_values(
    descriptor: SSASequenceDescriptor,
) -> tuple[SSAValue, ...]:
    from ..transmogrifier.dtype_layout import layout_for
    values = [
        *(SSAValue(
            value_id, dtype="ptr" if descriptor.heap_owner_value_id is not None else dtype,
            accounting={"sequence_row_shape": tuple(shape), **({
                # Alloca publishes the shared layout's canonical name. Source
                # linking copies formal accounting by exact identity; an
                # opaque_ref spelling here otherwise displaces that allocator
                # contract's int64 layout despite being the same physical type.
                "heap_element_dtype": layout_for(dtype).name,
                "heap_owner_value_id": int(descriptor.heap_owner_value_id),
            } if descriptor.heap_owner_value_id is not None else {})},
        ) for value_id, dtype, shape in zip(
            descriptor.column_value_ids,
            descriptor.column_dtypes
            or ("unknown",) * len(descriptor.column_value_ids),
            descriptor.column_shapes
            or ((),) * len(descriptor.column_value_ids),
        )),
        # A sequence's length/capacity cells are shared ABI storage -- every
        # caller and every helper generated here must agree on one true
        # width for them, the same way column dtypes already do (see
        # _canonical_sequence_dtype in precompile_to_ssa.py). int64 matches
        # that width and matches a caller-supplied keyed instance field's
        # own length, which is declared int64 by the program ABI contract;
        # declaring these int32 here caused a real Fortran ABI mismatch
        # whenever a caller's actual value was wider than this hardcoded
        # default.
        SSAValue(descriptor.length_address_id, dtype="int64", shape=(1,)),
        SSAValue(descriptor.capacity_value_id, dtype="int64"),
    ]
    if descriptor.live_flags_value_id is not None:
        values.append(SSAValue(descriptor.live_flags_value_id, dtype="bool"))
    if descriptor.heap_owner_value_id is not None and descriptor.child_table_pool is not None:
        pool = descriptor.child_table_pool
        for value_id, dtype in (
            *zip(pool.column_value_ids, pool.column_dtypes),
            (pool.length_value_id, "int64"),
            *(() if pool.status_value_id is None else ((pool.status_value_id, "int"),)),
        ):
            values.append(SSAValue(value_id, "ptr", accounting={
                "heap_element_dtype": layout_for(dtype).name,
                "heap_owner_value_id": int(descriptor.heap_owner_value_id),
            }))
        values.extend((SSAValue(pool.capacity_value_id, "int64"),
                       SSAValue(pool.row_stride_value_id, "int64")))
    if descriptor.heap_owner_value_id is not None:
        # The owner travels with its heap cells.  Program: ``s += tokens``
        # where ``tokens`` is a DYNAMIC list (f"a@{n}").  Wrong record: the
        # extend helper read the source's heap cells, each naming its owner
        # (``heap_owner_value_id``), while the owner itself was no formal of
        # the helper, so the heap-context projection could not bind it.
        # Rule: a heap descriptor's storage is its cells AND their owner,
        # last, in every helper built from it.
        values.append(SSAValue(
            descriptor.heap_owner_value_id, "ptr", shape=(1,),
            accounting={"heap_owner": True},
        ))
    # Alias descriptors can intentionally name the same arena more than once.
    return tuple({value.id: value for value in values}.values())


def lower_sequence_allocate(
    descriptor: SSASequenceDescriptor, *, function_name: str | None = None,
) -> SSASequenceLowering:
    """Allocate descriptor columns at its runtime capacity, then clear length.

    This initializes EMPTY storage. It neither migrates existing rows nor
    implements growth. Columns are caller-supplied native pointer cells and
    remain valid until explicit owner release, including across helper returns.
    """
    from .ssa_heap_allocation import heap_allocate

    if descriptor.heap_owner_value_id is None:
        raise ValueError("sequence allocation requires an explicit heap owner")
    if not descriptor.writable or descriptor.live_flags_value_id is not None:
        raise ValueError("heap initialization requires writable scalar columns without live flags")
    pool = descriptor.child_table_pool
    if pool is not None and (pool.key_columns or pool.live_flags_value_id is not None or pool.shape_value_id is not None):
        raise ValueError("owned child pools require unkeyed scalar rows")
    if any(descriptor.column_shapes):
        raise ValueError("heap initialization of shaped rows requires checked capacity multiplication")
    storage = _storage_values(descriptor)
    values = {int(value.id): value for value in storage}
    owner = values[int(descriptor.heap_owner_value_id)]
    builder = _Builder(_first_fresh_identity(descriptor, values=storage))
    entry = builder.block("entry")
    capacity = values[int(descriptor.capacity_value_id)]
    carrier = hash_index_carrier(descriptor)
    # The zeroed carrier is an index of no rows (``M == 0 == length``).
    carrier_count = (None if carrier is None
                     else _hash_index_arena_count(builder, entry, capacity))
    for index, column_id in enumerate(descriptor.column_value_ids):
        entry.instrs.append(heap_allocate(
            carrier_count if index == carrier else capacity, owner, values[int(column_id)],
            element_dtype=descriptor.column_dtypes[index], zero_initialize=True))
    zero = builder.const(entry, 0, dtype="int64")
    if pool is not None:
        for value_id, dtype in (*zip(pool.column_value_ids, pool.column_dtypes),
                                (pool.length_value_id, "int64"),
                                *(() if pool.status_value_id is None else ((pool.status_value_id, "int"),))):
            entry.instrs.append(heap_allocate(zero, owner, values[value_id],
                element_dtype=dtype, zero_initialize=True))
        builder.emit(entry, "Store", [zero, values[pool.capacity_value_id]])
        builder.emit(entry, "Store", [zero, values[pool.row_stride_value_id]])
    slot = builder.fresh("ptr")
    builder.emit(entry, "GetElementPtr", [values[int(descriptor.length_address_id)], zero], slot)
    builder.emit(entry, "Store", [zero, slot])
    builder.emit(entry, "Ret", [])
    function = Function(function_name or f"ssa_sequence_{descriptor.sequence_id}_allocate",
                        [*storage], builder.blocks, metadata={
        "ssa_sequence_operation": "allocate", "sequence_id": descriptor.sequence_id,
        "sequence_array_argument_ids": _storage_array_ids(descriptor, storage),
    })
    return SSASequenceLowering((function,))


def _storage_array_ids(
    descriptor: SSASequenceDescriptor,
    storage: tuple[SSAValue, ...],
) -> tuple[int, ...]:
    """The storage formals a positional helper indexes as arrays.

    The capacity cell is a scalar formal (``_storage_values`` declares it
    with no shape, and ``allocate`` already leaves it out).  Program:
    ``xs = [5, n, 2]; xs.pop(0)``.  Wrong record: ``delete_at`` listed the
    capacity among its array arguments, so Fortran gave the caller's scalar
    capacity a shape-dynamic extent nothing could supply, and the call failed
    to run.  Rule: every storage formal except the capacity cell.
    """

    scalar_ids = {int(descriptor.capacity_value_id)}
    if descriptor.heap_owner_value_id is not None and descriptor.child_table_pool is not None:
        scalar_ids.update((descriptor.child_table_pool.capacity_value_id,
                           descriptor.child_table_pool.row_stride_value_id))
    return tuple(
        int(value.id) for value in storage
        if int(value.id) not in scalar_ids
    )


def _status_branch(
    builder: _Builder,
    block: BasicBlock,
    status: int,
    target: BasicBlock,
) -> SSAValue:
    value = builder.const(block, status)
    builder.branch(block, target)
    return value


# ---------------------------------------------------------------------------
# Hash index of a keyed heap table.
#
# Program: the string core's golden workload (``repeated_manager``), whose
# ``Vocabulary._candidates`` and ``._prefix`` dictionaries gain one key per
# minted token.  Wrong record: every lookup, setdefault and store scanned
# every row, so 10x the rounds cost ~80x the native time while Python's dict
# stayed linear (rounds 30000: Python 3.9 s, C 5.9 s, LLVM 6.3 s).  Rule: a
# keyed table whose arenas its own heap owner allocates (only
# ``lower_sequence_allocate`` and the two reserves do) carries an
# open-addressing index in its CARRIER key column's arena, past the
# ``capacity`` rows:
#
#   carrier[0 : C]            the rows (unchanged)
#   carrier[C]                M, the row count the index covers
#   carrier[C + 1 : 3C + 1]   2C slots; 0 is empty, otherwise row + 1
#
# The index is trusted only while ``M == length``; otherwise every helper
# scans exactly as before.  Reserve rebuilds it into the fresh zeroed arena,
# a table store appends the row it wrote, and every other helper store to the
# length cell sets ``M = -1`` (``_guard_hash_indexes``), so a row change the
# index did not see can never answer a lookup.  A hit always compares every
# key column, so the hash only chooses where to look.
# ---------------------------------------------------------------------------

_HASH_INDEX_KEY_DTYPES = frozenset({"int64", "uint64", "int32", "uint32"})
_HASH_INDEX_PRIME = 2147483647
_HASH_INDEX_MULTIPLIER = 1000003


def hash_index_carrier(descriptor: SSASequenceDescriptor) -> int | None:
    """The key column position whose arena carries the table's hash index.

    ``None`` keeps the incumbent linear scan: no heap owner (the arenas are
    caller- or host-sized), no keys, a key dtype the hash does not read
    exactly (floats, opaque references, bools), a shaped key, or no unaliased
    int64 key column to hold the slots.
    """

    if descriptor.heap_owner_value_id is None or not descriptor.key_columns:
        return None
    dtypes = tuple(map(str, descriptor.column_dtypes))
    if len(dtypes) != len(descriptor.column_value_ids):
        return None
    shapes = descriptor.column_shapes or ((),) * len(dtypes)
    if any(
        dtypes[column] not in _HASH_INDEX_KEY_DTYPES or shapes[column]
        for column in descriptor.key_columns
    ):
        return None
    ids = [int(value_id) for value_id in descriptor.column_value_ids]
    for column in sorted(descriptor.key_columns):
        if dtypes[column] == "int64" and ids.count(ids[column]) == 1:
            return int(column)
    return None


def _hash_index_capacity_limit(maximum: int) -> int:
    """The largest capacity whose carrier arena (3C + 1 int64) still fits."""

    return min(int(maximum), ((2**63 - 1 - 8) // 8 - 1) // 3)


def _hash_index_arena_count(builder, block, capacity):
    count = builder.fresh("int64")
    tripled = builder.fresh("int64")
    builder.emit(block, "Mul", [capacity, builder.const(block, 3, dtype="int64")], tripled)
    builder.emit(block, "Add", [tripled, builder.const(block, 1, dtype="int64")], count)
    return count


def _hash_index_slot(builder, block, keys, capacity):
    """Return ``(slot, slots)``: the first probe in ``[0, 2C)`` and ``2C``.

    ``keys`` are ``(column position, value)`` pairs.

    Each key component is reduced modulo a prime in its own dtype, so the
    residue is exact and below 2**31 before it joins the int64 accumulator;
    no product can leave int64.
    """

    zero = builder.const(block, 0, dtype="int64")
    # Hash in column order, whatever order a view lists its key columns in.
    keys = [key for _column, key in sorted(keys, key=lambda pair: pair[0])]
    prime = builder.const(block, _HASH_INDEX_PRIME, dtype="int64")
    multiplier = builder.const(block, _HASH_INDEX_MULTIPLIER, dtype="int64")
    accumulated = zero
    for key in keys:
        dtype = str(key.dtype)
        residue = builder.fresh(dtype)
        builder.emit(block, "Mod", [key, builder.const(block, _HASH_INDEX_PRIME, dtype=dtype)], residue)
        if dtype != "int64":
            widened = builder.fresh("int64")
            builder.emit(block, "Cast", [residue], widened, attributes={
                "target_dtype": "int64", "source_dtype": dtype,
                "binding": "ssa_sequence_hash_index_residue",
            })
            residue = widened
        scaled, summed, reduced = (builder.fresh("int64") for _ in range(3))
        builder.emit(block, "Mul", [accumulated, multiplier], scaled)
        builder.emit(block, "Add", [scaled, residue], summed)
        builder.emit(block, "Mod", [summed, prime], reduced)
        accumulated = reduced
    slots, raw, lifted, slot = (builder.fresh("int64") for _ in range(4))
    negative = builder.fresh("bool")
    builder.emit(block, "Add", [capacity, capacity], slots)
    builder.emit(block, "Mod", [accumulated, slots], raw)
    builder.emit(block, "Lt", [raw, zero], negative)
    builder.emit(block, "Add", [raw, slots], lifted)
    builder.emit(block, "Select", [negative, lifted, raw], slot)
    return slot, slots


def _hash_index_slot_address(builder, block, carrier, capacity, slot):
    """Address of slot ``slot``: element ``C + 1 + slot`` of the carrier."""

    base, position = builder.fresh("int64"), builder.fresh("int64")
    address = builder.fresh("ptr")
    builder.emit(block, "Add", [capacity, builder.const(block, 1, dtype="int64")], base)
    builder.emit(block, "Add", [base, slot], position)
    builder.emit(block, "GetElementPtr", [carrier, position], address)
    return address


def _hash_index_next_slot(builder, block, slot, slots):
    following, wrapped = builder.fresh("int64"), builder.fresh("int64")
    at_end = builder.fresh("bool")
    builder.emit(block, "Add", [slot, builder.const(block, 1, dtype="int64")], following)
    builder.emit(block, "Eq", [following, slots], at_end)
    builder.emit(block, "Select", [at_end, builder.const(block, 0, dtype="int64"), following], wrapped)
    return wrapped


def _hash_index_carrier_value(descriptor):
    carrier = hash_index_carrier(descriptor)
    return SSAValue(descriptor.column_value_ids[carrier], dtype="int64")


def _hash_index_valid(builder, block, descriptor, capacity, length, *, label):
    """Branch on ``C > 0 and M == length``; return ``(valid_block, invalid_block)``.

    The capacity test comes first: a capacity-0 table may have no arena.
    """

    carrier = _hash_index_carrier_value(descriptor)
    marker_block = builder.block(f"{label}_index_marker")
    valid_block = builder.block(f"{label}_index_valid")
    invalid_block = builder.block(f"{label}_index_invalid")
    allocated = builder.fresh("bool")
    builder.emit(block, "Gt", [capacity, builder.const(block, 0, dtype="int64")], allocated)
    builder.cond(block, allocated, marker_block, invalid_block)
    marker_address, marker = builder.fresh("ptr"), builder.fresh("int64")
    covers = builder.fresh("bool")
    builder.emit(marker_block, "GetElementPtr", [carrier, capacity], marker_address)
    builder.emit(marker_block, "Load", [marker_address], marker)
    builder.emit(marker_block, "Eq", [marker, length], covers)
    builder.cond(marker_block, covers, valid_block, invalid_block)
    return valid_block, invalid_block


def _hash_index_place(builder, block, descriptor, capacity, keys, row, *, label):
    """Store ``row + 1`` in the first empty slot of ``keys``' probe sequence.

    The caller guarantees an empty slot exists: at most C rows over 2C slots.
    Returns the block that continues after the placement.
    """

    carrier = _hash_index_carrier_value(descriptor)
    slot0, slots = _hash_index_slot(builder, block, list(zip(descriptor.key_columns, keys)), capacity)
    header = builder.block(f"{label}_place_header")
    latch = builder.block(f"{label}_place_latch")
    place = builder.block(f"{label}_place")
    builder.branch(block, header)
    slot, next_slot = builder.fresh("int64"), builder.fresh("int64")
    builder.emit(header, "Phi", [slot0, next_slot], slot, attributes={
        "incoming_blocks": (block.name, latch.name),
    })
    address = _hash_index_slot_address(builder, header, carrier, capacity, slot)
    occupant, empty = builder.fresh("int64"), builder.fresh("bool")
    builder.emit(header, "Load", [address], occupant)
    builder.emit(header, "Eq", [occupant, builder.const(header, 0, dtype="int64")], empty)
    builder.cond(header, empty, place, latch)
    wrapped = _hash_index_next_slot(builder, latch, slot, slots)
    builder.emit(latch, "Add", [wrapped, builder.const(latch, 0, dtype="int64")], next_slot)
    builder.branch(latch, header)
    entry = builder.fresh("int64")
    builder.emit(place, "Add", [row, builder.const(place, 1, dtype="int64")], entry)
    builder.emit(place, "Store", [entry, address])
    return place


def _hash_index_probe(builder, block, descriptor, capacity, queries, *, label,
                      found, absent, key_match):
    """Probe ``queries``; branch to ``found`` or ``absent``.

    Returns ``(row, probe_block)``: the matching row (valid in ``found``) and
    the block that branches to ``found``, for the caller's row Phi.
    """

    carrier = _hash_index_carrier_value(descriptor)
    slot0, slots = _hash_index_slot(builder, block, list(zip(descriptor.key_columns, queries)), capacity)
    header = builder.block(f"{label}_probe_header")
    body = builder.block(f"{label}_probe_body")
    latch = builder.block(f"{label}_probe_latch")
    builder.branch(block, header)
    slot, next_slot = builder.fresh("int64"), builder.fresh("int64")
    builder.emit(header, "Phi", [slot0, next_slot], slot, attributes={
        "incoming_blocks": (block.name, latch.name),
    })
    address = _hash_index_slot_address(builder, header, carrier, capacity, slot)
    occupant, empty = builder.fresh("int64"), builder.fresh("bool")
    builder.emit(header, "Load", [address], occupant)
    builder.emit(header, "Eq", [occupant, builder.const(header, 0, dtype="int64")], empty)
    builder.cond(header, empty, absent, body)
    row = builder.fresh("int64")
    builder.emit(body, "Sub", [occupant, builder.const(body, 1, dtype="int64")], row)
    builder.cond(body, key_match(body, row), found, latch)
    wrapped = _hash_index_next_slot(builder, latch, slot, slots)
    builder.emit(latch, "Add", [wrapped, builder.const(latch, 0, dtype="int64")], next_slot)
    builder.branch(latch, header)
    return row, body, header


def _key_row_match(builder, descriptor, key_dtypes, queries):
    """``key_match(block, row)``: every key column equals its query and the
    row is live -- the incumbent scan's comparison, for any row."""

    def key_match(block, row):
        matches = None
        for key_column, key_dtype, query in zip(
            descriptor.key_columns, key_dtypes, queries
        ):
            key_address = builder.fresh("ptr")
            existing_key = builder.fresh(key_dtype)
            component_match = builder.fresh("bool")
            builder.emit(block, "GetElementPtr", [
                SSAValue(descriptor.column_value_ids[key_column], dtype=key_dtype), row
            ], key_address)
            builder.emit(block, "Load", [key_address], existing_key)
            builder.emit(block, "Eq", [existing_key, query], component_match)
            if matches is None:
                matches = component_match
            else:
                combined = builder.fresh("bool")
                builder.emit(block, "LAnd", [matches, component_match], combined)
                matches = combined
        assert matches is not None
        if descriptor.live_flags_value_id is not None:
            live_address = builder.fresh("ptr")
            live = builder.fresh("bool")
            active_match = builder.fresh("bool")
            builder.emit(block, "GetElementPtr", [
                SSAValue(descriptor.live_flags_value_id, dtype="bool"), row
            ], live_address)
            builder.emit(block, "Load", [live_address], live)
            builder.emit(block, "LAnd", [matches, live], active_match)
            matches = active_match
        return matches

    return key_match


def _hash_index_rebuild(builder, block, descriptor, capacity, length, *, label):
    """Index rows ``[0, length)`` into a fresh zeroed carrier; set ``M``.

    Columns are read through their published cells, so call this only after
    the reserve stored the new arenas and ``capacity`` is the new extent.
    Returns the continuing block.
    """

    carrier = _hash_index_carrier_value(descriptor)
    key_dtypes = tuple(str(descriptor.column_dtypes[column]) for column in descriptor.key_columns)
    header = builder.block(f"{label}_rebuild_header")
    body = builder.block(f"{label}_rebuild_body")
    done = builder.block(f"{label}_rebuild_done")
    zero = builder.const(block, 0, dtype="int64")
    builder.branch(block, header)
    row, next_row = builder.fresh("int64"), builder.fresh("int64")
    more = builder.fresh("bool")
    phi = builder.emit(header, "Phi", [zero, next_row], row, attributes={
        "incoming_blocks": (block.name, None),
    })
    builder.emit(header, "Lt", [row, length], more)
    builder.cond(header, more, body, done)
    keys = []
    for column, dtype in zip(descriptor.key_columns, key_dtypes):
        address, key = builder.fresh("ptr"), builder.fresh(dtype)
        builder.emit(body, "GetElementPtr", [
            SSAValue(descriptor.column_value_ids[column], dtype=dtype), row
        ], address)
        builder.emit(body, "Load", [address], key)
        keys.append(key)
    placed = _hash_index_place(builder, body, descriptor, capacity, keys, row, label=label + "_rebuild")
    builder.emit(placed, "Add", [row, builder.const(placed, 1, dtype="int64")], next_row)
    builder.branch(placed, header)
    phi.attributes["incoming_blocks"] = (block.name, placed.name)
    marker_address = builder.fresh("ptr")
    builder.emit(done, "GetElementPtr", [carrier, capacity], marker_address)
    builder.emit(done, "Store", [length, marker_address])
    return done


def _length_cell_stores(function: Function, length_id: int):
    """``(block_name, position)`` of every Store into the length cell."""

    addresses = {int(length_id)}
    for block in function.blocks.values():
        for instruction in block.instrs:
            if (instruction.op == "GetElementPtr" and instruction.res is not None
                    and instruction.args and int(instruction.args[0].id) == int(length_id)):
                addresses.add(int(instruction.res.id))
    return [
        (block.name, position)
        for block in function.blocks.values()
        for position, instruction in enumerate(block.instrs)
        if instruction.op == "Store" and len(instruction.args) == 2
        and int(instruction.args[1].id) in addresses
        and not instruction.attributes.get("hash_index_guarded")
    ]


def _invalidate_hash_index_after_length_stores(function: Function, descriptor) -> None:
    """After each length store: ``if C > 0: M = -1`` (see the section note)."""

    if hash_index_carrier(descriptor) is None:
        return
    stores = _length_cell_stores(function, int(descriptor.length_address_id))
    if not stores:
        return
    carrier_id = int(descriptor.column_value_ids[hash_index_carrier(descriptor)])
    capacity_id = int(descriptor.capacity_value_id)
    formals = {int(value.id) for value in function.args}
    if carrier_id not in formals or capacity_id not in formals:
        raise ValueError(
            f"hash-indexed table {int(descriptor.sequence_id)}: helper "
            f"{function.name!r} stores its length without the carrier column "
            "and capacity it must invalidate"
        )
    builder = _Builder(0)
    builder.blocks = function.blocks
    carrier = SSAValue(carrier_id, dtype="int64")
    capacity = SSAValue(capacity_id, dtype="int64")
    counter = 0
    while stores:
        block_name, position = stores[0]
        block = function.blocks[block_name]
        store = block.instrs[position]
        store.attributes = {**store.attributes, "hash_index_guarded": True}
        tail_instructions = block.instrs[position + 1:]
        del block.instrs[position + 1:]
        counter += 1
        while f"{block_name}_index_guard_{counter}" in function.blocks:
            counter += 1
        invalidate = builder.block(f"{block_name}_index_invalidate_{counter}")
        tail = builder.block(f"{block_name}_index_guard_{counter}")
        tail.instrs = tail_instructions
        tail.successors = list(block.successors)
        block.successors = []
        allocated = builder.fresh("bool")
        builder.emit(block, "Gt", [capacity, builder.const(block, 0, dtype="int64")], allocated)
        builder.cond(block, allocated, invalidate, tail)
        marker_address = builder.fresh("ptr")
        builder.emit(invalidate, "GetElementPtr", [carrier, capacity], marker_address)
        builder.emit(invalidate, "Store", [builder.const(invalidate, -1, dtype="int64"), marker_address])
        builder.branch(invalidate, tail)
        for successor in tail.successors:
            for instruction in function.blocks[successor].instrs:
                if instruction.op != "Phi":
                    continue
                incoming = tuple(instruction.attributes.get("incoming_blocks", ()))
                if block_name in incoming:
                    instruction.attributes = {**instruction.attributes, "incoming_blocks": tuple(
                        tail.name if name == block_name else name for name in incoming
                    )}
        stores = _length_cell_stores(function, int(descriptor.length_address_id))


def _guard_hash_indexes(lowerer):
    """Apply the length-store invalidation to every helper a lowerer builds."""

    import functools
    import inspect

    signature = inspect.signature(lowerer)

    @functools.wraps(lowerer)
    def guarded(*args, **kwargs):
        lowering = lowerer(*args, **kwargs)
        if not isinstance(lowering, SSASequenceLowering):
            return lowering
        bound = signature.bind_partial(*args, **kwargs)
        descriptors = [value for value in bound.arguments.values()
                       if isinstance(value, SSASequenceDescriptor)]
        for descriptor in descriptors:
            if hash_index_carrier(descriptor) is None:
                continue
            for function in lowering.functions:
                _invalidate_hash_index_after_length_stores(function, descriptor)
        return lowering

    return guarded


def _unsupported_destination(
    descriptor: SSASequenceDescriptor, operation: str, *, allow_dynamic: bool = False,
) -> SSASequenceLowering | None:
    if not descriptor.writable:
        return SSASequenceLowering(shortfalls=(SSASequenceLoweringShortfall(
            SSASequenceShortfallCode.READ_ONLY_DESTINATION,
            descriptor.sequence_id,
            operation,
            "sequence insertion requires writable destination storage",
        ),))
    if descriptor.capacity_policy is SSASequenceCapacityPolicy.DYNAMIC and (
        not allow_dynamic or descriptor.heap_owner_value_id is None
        or descriptor.live_flags_value_id is not None
        or (descriptor.child_table_pool is not None and (
            descriptor.child_table_pool.key_columns
            or descriptor.child_table_pool.live_flags_value_id is not None
            or descriptor.child_table_pool.shape_value_id is not None
        )) or any(descriptor.column_shapes)
    ):
        return SSASequenceLowering(shortfalls=(SSASequenceLoweringShortfall(
            SSASequenceShortfallCode.DYNAMIC_GROWTH_UNAVAILABLE,
            descriptor.sequence_id,
            operation,
            "dynamic sequence growth requires typed scalar native pointer cells, "
            "an explicit heap owner, no live flags, and scalar unkeyed child pools; "
            "this operation must support arena replacement",
        ),))
    return None


def lower_sequence_reserve(
    descriptor: SSASequenceDescriptor, *, function_name: str | None = None,
) -> SSASequenceLowering:
    """Grow existing typed arenas and publish their pointer cells last.

    The exact owner retains retired arenas until explicit release. Storage
    CELL aliases remain stable; interior addresses are views of one arena
    version and must be reacquired after reserve, never silently rebound.
    Status 0 means unchanged, 1 grown, 2 invalid extent/storage contract.
    Length and existing rows are preserved, new rows are zero-initialized.
    """
    from .ssa_heap_allocation import heap_allocate, heap_copy
    from ..transmogrifier.dtype_layout import layout_for

    unsupported = _unsupported_destination(descriptor, "reserve", allow_dynamic=True)
    if unsupported is not None:
        return unsupported
    if descriptor.capacity_policy is not SSASequenceCapacityPolicy.DYNAMIC:
        raise ValueError("reserve requires a dynamic sequence capacity contract")
    if descriptor.child_table_pool is not None:
        return _lower_sequence_child_pool_reserve(descriptor, function_name=function_name)
    storage = _storage_values(descriptor)
    values = {int(value.id): value for value in storage}
    owner = values[int(descriptor.heap_owner_value_id)]
    builder = _Builder(_first_fresh_identity(descriptor, values=storage))
    required = builder.fresh("int64")
    capacity = values[int(descriptor.capacity_value_id)]
    entry = builder.block("entry")
    check = builder.block("capacity_check")
    grow = builder.block("growth_check")
    double = builder.block("double_capacity")
    capped = builder.block("maximum_capacity")
    candidate_block = builder.block("candidate_capacity")
    exact = builder.block("exact_capacity")
    geometric = builder.block("geometric_capacity")
    allocate = builder.block("allocate")
    unchanged = builder.block("unchanged")
    invalid = builder.block("invalid_extent")
    complete = builder.block("complete")
    result = builder.block("result")
    zero = builder.const(entry, 0, dtype="int64")
    length_slot, length = builder.fresh("ptr"), builder.fresh("int64")
    builder.emit(entry, "GetElementPtr", [values[descriptor.length_address_id], zero], length_slot)
    builder.emit(entry, "Load", [length_slot], length)
    maximum = min((2**63 - 1 - 8) // layout_for(dtype).byte_size
                  for dtype in descriptor.column_dtypes)
    carrier = hash_index_carrier(descriptor)
    carrier_id = None if carrier is None else descriptor.column_value_ids[carrier]
    if carrier is not None:
        maximum = _hash_index_capacity_limit(maximum)
    maximum_value = builder.const(entry, maximum, dtype="int64")
    predicates = []
    for op, left, right in (("Ge", required, zero), ("Ge", length, zero),
                            ("Le", length, capacity), ("Le", capacity, maximum_value),
                            ("Le", required, maximum_value)):
        predicate = builder.fresh("bool")
        builder.emit(entry, op, [left, right], predicate)
        predicates.append(predicate)
    valid = predicates[0]
    for predicate in predicates[1:]:
        both = builder.fresh("bool")
        builder.emit(entry, "LAnd", [valid, predicate], both)
        valid = both
    builder.cond(entry, valid, check, invalid)
    fits = builder.fresh("bool")
    builder.emit(check, "Le", [required, capacity], fits)
    builder.cond(check, fits, unchanged, grow)
    can_double = builder.fresh("bool")
    builder.emit(grow, "Le", [capacity, builder.const(grow, maximum // 2, dtype="int64")], can_double)
    builder.cond(grow, can_double, double, capped)
    doubled = builder.fresh("int64")
    builder.emit(double, "Mul", [capacity, builder.const(double, 2, dtype="int64")], doubled)
    builder.branch(double, candidate_block)
    builder.branch(capped, candidate_block)
    candidate = builder.fresh("int64")
    builder.emit(candidate_block, "Phi", [doubled, maximum_value], candidate,
                 attributes={"incoming_blocks": (double.name, capped.name)})
    needs_exact = builder.fresh("bool")
    builder.emit(candidate_block, "Gt", [required, candidate], needs_exact)
    builder.cond(candidate_block, needs_exact, exact, geometric)
    builder.branch(exact, allocate)
    builder.branch(geometric, allocate)
    new_capacity = builder.fresh("int64")
    builder.emit(allocate, "Phi", [required, candidate], new_capacity,
                 attributes={"incoming_blocks": (exact.name, geometric.name)})
    new_columns = {}
    carrier_count = (None if carrier is None
                     else _hash_index_arena_count(builder, allocate, new_capacity))
    for column_id, dtype in zip(descriptor.column_value_ids, descriptor.column_dtypes, strict=True):
        if column_id in new_columns:
            continue  # Exact same pointer cell, already allocated/copied.
        column = builder.fresh("ptr")
        allocate.instrs.append(heap_allocate(
            carrier_count if column_id == carrier_id else new_capacity, owner, column,
            element_dtype=dtype, zero_initialize=True))
        new_columns[column_id] = column
    for column_id, column in new_columns.items():
        allocate.instrs.append(heap_copy(length, values[column_id], column))
    # Only publish after every column has allocated and copied successfully.
    for column_id, column in new_columns.items():
        builder.emit(allocate, "Store", [column, values[column_id]])
    builder.emit(allocate, "Store", [new_capacity, capacity])
    if carrier is not None:
        rebuilt = _hash_index_rebuild(
            builder, allocate, descriptor, new_capacity, length, label="reserve",
        )
        builder.branch(rebuilt, complete)
    else:
        builder.branch(allocate, complete)
    outcomes = [(block, _status_branch(builder, block, status, result))
                for block, status in ((unchanged, 0), (complete, 1), (invalid, 2))]
    status = builder.fresh("int")
    builder.emit(result, "Phi", [value for _, value in outcomes], status,
                 attributes={"incoming_blocks": tuple(block.name for block, _ in outcomes)})
    builder.emit(result, "Ret", [status])
    function = Function(function_name or f"ssa_sequence_{descriptor.sequence_id}_reserve",
                        [*storage, required], builder.blocks, metadata={
        "ssa_sequence_operation": "reserve", "sequence_id": descriptor.sequence_id,
        "required_capacity_value_id": required.id,
        "heap_owner_value_id": owner.id,
        "sequence_array_argument_ids": (*descriptor.column_value_ids, descriptor.length_address_id, owner.id),
        "status_values": {"unchanged": 0, "grown": 1, "invalid_extent": 2},
        "growth_policy": "checked_geometric", "retired_arena_lifetime": "owner_release",
        "named_outputs": (("status", status.id),),
    })
    return SSASequenceLowering((function,))


def _lower_sequence_child_pool_reserve(descriptor, *, function_name):
    """Reserve both existing pool dimensions under the enclosing context.

    c640's constructor owns empty dictionaries, so setdefault reached error75
    at its first row.  A pool's capacity is flattened, while handles name
    parent rows and row_stride is each list's capacity.  Keep those identities
    and the incumbent checked geometric policy; move each live row to its new
    stride before publishing any replacement pointer cell.
    """
    from .ssa_heap_allocation import heap_allocate, heap_copy
    from ..transmogrifier.dtype_layout import layout_for

    pool = descriptor.child_table_pool
    storage = _storage_values(descriptor)
    values = {value.id: value for value in storage}
    owner = values[descriptor.heap_owner_value_id]
    builder = _Builder(_first_fresh_identity(descriptor, values=storage))
    required, required_stride = builder.fresh("int64"), builder.fresh("int64")
    entry, old_stride_check, old_limit_check, old_flat_check, check, dimensions, stride_check, product_check, allocate, row_header, row_body, item_header, item_body, item_latch, row_latch, publish, unchanged, invalid, result = (
        builder.block(name) for name in ("entry", "old_stride_check", "old_limit_check", "old_flat_check", "capacity_check", "dimensions", "stride_check", "product_check", "allocate", "row_header", "row_body", "item_header", "item_body", "item_latch", "row_latch", "publish", "unchanged", "invalid_extent", "result")
    )
    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")
    capacity = values[descriptor.capacity_value_id]
    stride = values[pool.row_stride_value_id]
    pool_capacity = values[pool.capacity_value_id]
    slot, length = builder.fresh("ptr"), builder.fresh("int64")
    builder.emit(entry, "GetElementPtr", [values[descriptor.length_address_id], zero], slot)
    builder.emit(entry, "Load", [slot], length)
    maximum = min((2**63 - 1 - 8) // layout_for(dtype).byte_size
                  for dtype in (*descriptor.column_dtypes, *pool.column_dtypes, "int64", "int"))
    carrier = hash_index_carrier(descriptor)
    carrier_id = None if carrier is None else descriptor.column_value_ids[carrier]
    if carrier is not None:
        maximum = _hash_index_capacity_limit(maximum)
    maximum_value = builder.const(entry, maximum, dtype="int64")
    valid = None
    for op, left, right in (("Ge", required, zero), ("Ge", required_stride, zero),
                            ("Ge", length, zero), ("Le", length, capacity),
                            ("Ge", stride, zero), ("Le", capacity, maximum_value),
                            ("Ge", pool_capacity, zero), ("Le", pool_capacity, maximum_value),
                            ("Le", stride, maximum_value), ("Le", required, maximum_value),
                            ("Le", required_stride, maximum_value)):
        predicate = builder.fresh("bool")
        builder.emit(entry, op, [left, right], predicate)
        if valid is None:
            valid = predicate
        else:
            both = builder.fresh("bool")
            builder.emit(entry, "LAnd", [valid, predicate], both)
            valid = both
    builder.cond(entry, valid, old_stride_check, invalid)
    old_positive_stride = builder.fresh("bool")
    builder.emit(old_stride_check, "Gt", [stride, zero], old_positive_stride)
    builder.cond(old_stride_check, old_positive_stride, old_limit_check, check)
    old_outer_limit, old_product_fits = builder.fresh("int64"), builder.fresh("bool")
    builder.emit(old_limit_check, "Div", [maximum_value, stride], old_outer_limit)
    builder.emit(old_limit_check, "Le", [capacity, old_outer_limit], old_product_fits)
    builder.cond(old_limit_check, old_product_fits, old_flat_check, invalid)
    old_flat_count, old_pool_fits = builder.fresh("int64"), builder.fresh("bool")
    builder.emit(old_flat_check, "Mul", [capacity, stride], old_flat_count)
    builder.emit(old_flat_check, "Le", [old_flat_count, pool_capacity], old_pool_fits)
    builder.cond(old_flat_check, old_pool_fits, check, invalid)
    outer_fits, stride_fits, fits = (builder.fresh("bool") for _ in range(3))
    builder.emit(check, "Le", [required, capacity], outer_fits)
    builder.emit(check, "Le", [required_stride, stride], stride_fits)
    builder.emit(check, "LAnd", [outer_fits, stride_fits], fits)
    builder.cond(check, fits, unchanged, dimensions)

    def geometric(block, current, requested, label):
        """The same checked doubling/required choice as scalar reserve."""
        double, capped, candidate_block, exact, grown, merged = (
            builder.block(label + suffix) for suffix in ("_double", "_capped", "_candidate", "_exact", "_grown", "_merged")
        )
        may_double = builder.fresh("bool")
        builder.emit(block, "Le", [current, builder.const(block, maximum // 2, dtype="int64")], may_double)
        builder.cond(block, may_double, double, capped)
        doubled = builder.fresh("int64")
        builder.emit(double, "Mul", [current, builder.const(double, 2, dtype="int64")], doubled)
        builder.branch(double, candidate_block)
        builder.branch(capped, candidate_block)
        candidate = builder.fresh("int64")
        builder.emit(candidate_block, "Phi", [doubled, maximum_value], candidate,
                     attributes={"incoming_blocks": (double.name, capped.name)})
        use_required = builder.fresh("bool")
        builder.emit(candidate_block, "Gt", [requested, candidate], use_required)
        builder.cond(candidate_block, use_required, exact, grown)
        builder.branch(exact, merged)
        builder.branch(grown, merged)
        proposed = builder.fresh("int64")
        builder.emit(merged, "Phi", [requested, candidate], proposed,
                     attributes={"incoming_blocks": (exact.name, grown.name)})
        # A dimension that already fits keeps its capacity: growing another
        # dimension does not consume geometric growth in this one.
        keep, expand, done = (builder.block(label + suffix) for suffix in ("_keep", "_expand", "_done"))
        already_fits = builder.fresh("bool")
        builder.emit(merged, "Le", [requested, current], already_fits)
        builder.cond(merged, already_fits, keep, expand)
        builder.branch(keep, done)
        builder.branch(expand, done)
        value = builder.fresh("int64")
        builder.emit(done, "Phi", [current, proposed], value,
                     attributes={"incoming_blocks": (keep.name, expand.name)})
        return done, value

    tail, new_capacity = geometric(dimensions, capacity, required, "outer")
    tail, new_stride = geometric(tail, stride, required_stride, "child")
    positive_stride = builder.fresh("bool")
    builder.emit(tail, "Gt", [new_stride, zero], positive_stride)
    builder.cond(tail, positive_stride, product_check, stride_check)
    builder.branch(stride_check, allocate)
    outer_limit, product_fits = builder.fresh("int64"), builder.fresh("bool")
    builder.emit(product_check, "Div", [maximum_value, new_stride], outer_limit)
    builder.emit(product_check, "Le", [new_capacity, outer_limit], product_fits)
    builder.cond(product_check, product_fits, allocate, invalid)
    flattened = builder.fresh("int64")
    builder.emit(allocate, "Mul", [new_capacity, new_stride], flattened)
    replacements = {}
    carrier_count = (None if carrier is None
                     else _hash_index_arena_count(builder, allocate, new_capacity))
    allocation_specs = (
        *((column, dtype, carrier_count if column == carrier_id else new_capacity, length)
          for column, dtype in zip(descriptor.column_value_ids, descriptor.column_dtypes)),
        (pool.length_value_id, "int64", new_capacity, length),
        *(() if pool.status_value_id is None else ((pool.status_value_id, "int", new_capacity, length),)),
        *((column, dtype, flattened, None) for column, dtype in zip(pool.column_value_ids, pool.column_dtypes)),
    )
    for column, dtype, count, copy_count in allocation_specs:
        fresh = builder.fresh("ptr")
        allocate.instrs.append(heap_allocate(count, owner, fresh, element_dtype=dtype, zero_initialize=True))
        replacements[column] = fresh
        if copy_count is not None:
            allocate.instrs.append(heap_copy(copy_count, values[column], fresh))
    builder.branch(allocate, row_header)
    row, row_next = builder.fresh("int64"), builder.fresh("int64")
    builder.emit(row_header, "Phi", [zero, row_next], row,
                 attributes={"incoming_blocks": (allocate.name, row_latch.name)})
    more_rows = builder.fresh("bool")
    builder.emit(row_header, "Lt", [row, length], more_rows)
    builder.cond(row_header, more_rows, row_body, publish)
    length_address, row_length = builder.fresh("ptr"), builder.fresh("int64")
    builder.emit(row_body, "GetElementPtr", [values[pool.length_value_id], row], length_address)
    builder.emit(row_body, "Load", [length_address], row_length)
    nonnegative, within_stride, row_valid = (builder.fresh("bool") for _ in range(3))
    builder.emit(row_body, "Ge", [row_length, zero], nonnegative)
    builder.emit(row_body, "Le", [row_length, stride], within_stride)
    builder.emit(row_body, "LAnd", [nonnegative, within_stride], row_valid)
    builder.cond(row_body, row_valid, item_header, invalid)
    item, item_next = builder.fresh("int64"), builder.fresh("int64")
    builder.emit(item_header, "Phi", [zero, item_next], item,
                 attributes={"incoming_blocks": (row_body.name, item_latch.name)})
    more_items = builder.fresh("bool")
    builder.emit(item_header, "Lt", [item, row_length], more_items)
    builder.cond(item_header, more_items, item_body, row_latch)
    old_base, new_base, old_index, new_index = (builder.fresh("int64") for _ in range(4))
    builder.emit(item_body, "Mul", [row, stride], old_base)
    builder.emit(item_body, "Mul", [row, new_stride], new_base)
    builder.emit(item_body, "Add", [old_base, item], old_index)
    builder.emit(item_body, "Add", [new_base, item], new_index)
    for column, dtype in zip(pool.column_value_ids, pool.column_dtypes):
        old_address, new_address, value = builder.fresh("ptr"), builder.fresh("ptr"), builder.fresh(dtype)
        builder.emit(item_body, "GetElementPtr", [values[column], old_index], old_address)
        builder.emit(item_body, "Load", [old_address], value)
        builder.emit(item_body, "GetElementPtr", [replacements[column], new_index], new_address)
        builder.emit(item_body, "Store", [value, new_address])
    builder.branch(item_body, item_latch)
    builder.emit(item_latch, "Add", [item, one], item_next)
    builder.branch(item_latch, item_header)
    builder.emit(row_latch, "Add", [row, one], row_next)
    builder.branch(row_latch, row_header)
    for column, fresh in replacements.items():
        builder.emit(publish, "Store", [fresh, values[column]])
    builder.emit(publish, "Store", [new_capacity, capacity])
    builder.emit(publish, "Store", [flattened, values[pool.capacity_value_id]])
    builder.emit(publish, "Store", [new_stride, stride])
    published = publish
    if carrier is not None:
        published = _hash_index_rebuild(
            builder, publish, descriptor, new_capacity, length, label="reserve",
        )
    outcomes = [(block, _status_branch(builder, block, status, result))
                for block, status in ((unchanged, 0), (published, 1), (invalid, 2))]
    status = builder.fresh("int")
    builder.emit(result, "Phi", [value for _, value in outcomes], status,
                 attributes={"incoming_blocks": tuple(block.name for block, _ in outcomes)})
    builder.emit(result, "Ret", [status])
    function = Function(function_name or f"ssa_sequence_{descriptor.sequence_id}_reserve",
                        [*storage, required, required_stride], builder.blocks, metadata={
        "ssa_sequence_operation": "reserve", "sequence_id": descriptor.sequence_id,
        "required_capacity_value_id": required.id,
        "required_child_capacity_value_id": required_stride.id,
        "heap_owner_value_id": owner.id,
        "sequence_array_argument_ids": _storage_array_ids(descriptor, storage),
        "growth_policy": "checked_geometric", "retired_arena_lifetime": "owner_release",
        "named_outputs": (("status", status.id),),
    })
    return SSASequenceLowering((function,))


def _lower_dynamic_sequence_mutation(descriptor, *, function_name, operation, first_value_id):
    """Use incumbent insertion/store, reserving only when it reports full."""
    is_store = operation == "table_store"
    name = function_name or f"ssa_sequence_{descriptor.sequence_id}_{'store' if is_store else 'insert'}"
    fixed_descriptor = replace(descriptor, capacity_policy=SSASequenceCapacityPolicy.FIXED)
    # A fill writes ``count`` rows from row 0 and an append-fill writes
    # ``count`` rows after the incumbent ones: their reserve asks for those
    # rows, not one more (the ``requested`` count is the fixed helper's last
    # formal).
    fills = {"fill": lower_sequence_fill, "append_fill": lower_sequence_append_fill}
    fixed = (lower_table_store(fixed_descriptor, function_name=name + "__fixed", first_value_id=first_value_id)
             if is_store else fills[operation](fixed_descriptor,
                 function_name=name + "__fixed", first_value_id=first_value_id,
                 raise_when_full=False)
             if operation in fills else lower_sequence_insert_at(fixed_descriptor,
                 function_name=name + "__fixed", first_value_id=first_value_id,
                 raise_when_full=False)
             if operation == "insert_at" else lower_sequence_insert(fixed_descriptor,
                 function_name=name + "__fixed", operation=operation, first_value_id=first_value_id,
                 raise_when_full=False))
    if not fixed.complete:
        return fixed
    insert = fixed.functions[0]
    reserve = lower_sequence_reserve(descriptor, function_name=name + "__reserve").functions[0]

    # Each helper owns its formal occurrences.  Registration rewrites a
    # helper's construction-local ids IN PLACE, scope by scope; an entry that
    # reused its fixed insert's formal objects had the row value re-keyed
    # under the entry's scope, leaving the fixed helper's own row without
    # its physical fields (``ssa-physical-unposted``).
    def own(value):
        return SSAValue(value.id, value.dtype, shape=tuple(value.shape or ()),
                        accounting=dict(value.accounting or {}))

    values = {value.id: own(value) for value in insert.args}
    owner = values[descriptor.heap_owner_value_id]
    builder = _Builder(_first_fresh_identity(descriptor, values=tuple(values.values())))
    entry, insert_entry, grow, retry, result = (builder.block(block_name)
        for block_name in ("entry", "insert", "grow", "retry", "result"))
    # Reject inconsistent caller extents before the unique scan can read a
    # row. Required zero validates without allocation or arena replacement.
    validation_zero = builder.const(entry, 0, dtype="int64")
    validation_values = {**values, owner.id: owner,
                         reserve.metadata["required_capacity_value_id"]: validation_zero}
    if descriptor.child_table_pool is not None:
        validation_values[reserve.metadata["required_child_capacity_value_id"]] = values[descriptor.child_table_pool.row_stride_value_id]
    validation_status = builder.fresh("int")
    builder.emit(entry, "Call", [validation_values[value.id] for value in reserve.args], validation_status,
                 attributes={"callee": reserve.name})
    valid = builder.fresh("bool")
    builder.emit(entry, "Lt", [validation_status, builder.const(entry, 2)], valid)
    builder.cond(entry, valid, insert_entry, result)

    def call_mutation(block, status):
        builder.emit(block, "Call", [values[value.id] for value in insert.args],
                     None if is_store else status, attributes={"callee": insert.name})
        if is_store:
            slot = builder.fresh("ptr")
            builder.emit(block, "GetElementPtr", [values[descriptor.status_address_id],
                builder.const(block, 0, dtype="int64")], slot)
            builder.emit(block, "Load", [slot], status)

    first_status = builder.fresh("int")
    call_mutation(insert_entry, first_status)
    full = builder.fresh("bool")
    builder.emit(insert_entry, "Eq", [first_status, builder.const(insert_entry, 2)], full)
    builder.cond(insert_entry, full, grow, result)
    zero = builder.const(grow, 0, dtype="int64")
    length_slot, length = (builder.fresh(dtype) for dtype in ("ptr", "int64"))
    builder.emit(grow, "GetElementPtr", [values[descriptor.length_address_id], zero], length_slot)
    builder.emit(grow, "Load", [length_slot], length)
    if operation == "fill":
        required = values[insert.args[-1].id]
    else:
        required = builder.fresh("int64")
        builder.emit(grow, "Add", [length, values[insert.args[-1].id]
                                   if operation == "append_fill"
                                   else builder.const(grow, 1, dtype="int64")], required)
    reserve_values = {**values, owner.id: owner,
                      reserve.metadata["required_capacity_value_id"]: required}
    if descriptor.child_table_pool is not None:
        reserve_values[reserve.metadata["required_child_capacity_value_id"]] = values[descriptor.child_table_pool.row_stride_value_id]
    reserve_status = builder.fresh("int")
    builder.emit(grow, "Call", [reserve_values[value.id] for value in reserve.args], reserve_status,
                 attributes={"callee": reserve.name})
    succeeded = builder.fresh("bool")
    builder.emit(grow, "Lt", [reserve_status, builder.const(grow, 2)], succeeded)
    builder.cond(grow, succeeded, retry, result)
    retry_status = builder.fresh("int")
    call_mutation(retry, retry_status)
    builder.branch(retry, result)
    status = builder.fresh("int")
    builder.emit(result, "Phi", [validation_status, first_status, reserve_status, retry_status], status,
                 attributes={"incoming_blocks": (entry.name, insert_entry.name, grow.name, retry.name)})
    if is_store:
        slot = builder.fresh("ptr")
        builder.emit(result, "GetElementPtr", [values[descriptor.status_address_id],
            builder.const(result, 0, dtype="int64")], slot)
        builder.emit(result, "Store", [status, slot])
    builder.emit(result, "Ret", [] if is_store else [status])
    function = Function(name, [values[value.id] for value in insert.args],
                        builder.blocks, metadata={
        **insert.metadata, "fixed_capacity": False, "heap_owner_value_id": owner.id,
        "sequence_array_argument_ids": (*insert.metadata["sequence_array_argument_ids"], owner.id),
        "named_outputs": () if is_store else (("status", status.id),),
    })
    return SSASequenceLowering((function, insert, reserve))


def _raise_fixed_capacity_exhausted(builder, block, operation):
    """A FIXED arena's capacity is a proof; exhausting it means the proof
    was wrong.  Program: a callee list whose capacity cell stayed 0 (Lane E
    E10, c303); every append reported "full" into a status nobody reads
    and the callee summed zeros.  Rule (takeover rule 10): a fixed-capacity
    write that finds no room raises validation error 75, the same as a
    fixed fill.  Only the DYNAMIC protocol's insert step reports "full"
    without raising: its caller reserves and retries."""
    builder.emit(block, "Call", [], attributes={
        "callee": "turing_validation_error",
        "error_code": 75,
        "source_linked": True,
        "ssa_sequence_operation": operation,
        "failure": "fixed_capacity_exhausted",
    })


def lower_sequence_insert(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    operation: str = "insert",
    first_value_id: int | None = None,
    raise_when_full: bool = True,
) -> SSASequenceLowering:
    """Build fixed-capacity row insertion with optional linear key dedup.

    Return status is 0 when an existing unique row absorbs the insertion, 1
    when a row is written, and 2 when fixed capacity is exhausted.
    """

    unsupported = _unsupported_destination(descriptor, operation, allow_dynamic=True)
    if unsupported is not None:
        return unsupported
    if descriptor.capacity_policy is SSASequenceCapacityPolicy.DYNAMIC:
        return _lower_dynamic_sequence_mutation(descriptor, function_name=function_name,
            operation=operation, first_value_id=first_value_id)
    column_shapes = descriptor.column_shapes or tuple(
        () for _column in descriptor.column_value_ids
    )
    if any(column_shapes[int(column)] for column in descriptor.key_columns):
        return SSASequenceLowering(shortfalls=(SSASequenceLoweringShortfall(
            SSASequenceShortfallCode.SHAPED_KEY_UNAVAILABLE,
            descriptor.sequence_id,
            operation,
            "unique insertion requires an elementwise equality contract for "
            "a shaped key column",
        ),))

    storage = _storage_values(descriptor)
    builder = _Builder(_first_fresh_identity(
        descriptor, values=storage, minimum=first_value_id
    ))
    row_values = tuple(
        builder.fresh(dtype, tuple(shape))
        for dtype, shape in zip(
            descriptor.column_dtypes
            or ("unknown",) * len(descriptor.column_value_ids),
            column_shapes,
        )
    )
    columns = tuple(SSAValue(value_id, dtype=value.dtype) for value_id, value in zip(
        descriptor.column_value_ids, row_values
    ))
    # Untouched LLVM insert stored double 1.0 into this int64 length cell
    # because the operand view called it merely "ptr". Derive the operand
    # from the same storage declaration as the helper's formal, not a weaker
    # pointer spelling which loses the cell's element type.
    length_address = next(value for value in storage if value.id == descriptor.length_address_id)
    capacity = SSAValue(descriptor.capacity_value_id, dtype="int64")
    live_flags = (
        None
        if descriptor.live_flags_value_id is None
        else SSAValue(descriptor.live_flags_value_id, dtype="bool")
    )

    entry = builder.block("entry")
    capacity_check = builder.block("capacity_check")
    write = builder.block("write")
    full = builder.block("capacity_exhausted")
    inserted = builder.block("inserted")
    result_block = builder.block("result")
    status_values: list[tuple[BasicBlock, SSAValue]] = []
    current_length = builder.fresh("int64")
    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")
    length_slot = builder.fresh("ptr")
    builder.emit(
        entry, "GetElementPtr", [length_address, zero], length_slot
    )
    builder.emit(entry, "Load", [length_slot], current_length)

    if descriptor.key_columns:
        scan_header = builder.block("unique_scan_header")
        scan_body = builder.block("unique_scan_body")
        scan_latch = builder.block("unique_scan_latch")
        duplicate = builder.block("duplicate")
        builder.branch(entry, scan_header)
        scan_index = builder.fresh("int64")
        next_scan_index = builder.fresh(scan_index.dtype)
        builder.emit(
            scan_header,
            "Phi",
            [zero, next_scan_index],
            scan_index,
            attributes={"incoming_blocks": (entry.name, scan_latch.name)},
        )
        scan_continues = builder.fresh("bool")
        builder.emit(scan_header, "Lt", [scan_index, current_length], scan_continues)
        builder.cond(scan_header, scan_continues, scan_body, capacity_check)

        matches: SSAValue | None = None
        for key_column in descriptor.key_columns:
            address = builder.fresh("ptr")
            existing = builder.fresh(row_values[key_column].dtype)
            equal = builder.fresh("bool")
            builder.emit(
                scan_body,
                "GetElementPtr",
                [columns[key_column], scan_index],
                address,
            )
            builder.emit(scan_body, "Load", [address], existing)
            builder.emit(scan_body, "Eq", [existing, row_values[key_column]], equal)
            if matches is None:
                matches = equal
            else:
                both = builder.fresh("bool")
                builder.emit(scan_body, "LAnd", [matches, equal], both)
                matches = both
        assert matches is not None
        if live_flags is not None:
            live_address = builder.fresh("ptr")
            live = builder.fresh("bool")
            active_match = builder.fresh("bool")
            builder.emit(
                scan_body, "GetElementPtr", [live_flags, scan_index], live_address
            )
            builder.emit(scan_body, "Load", [live_address], live)
            builder.emit(scan_body, "LAnd", [matches, live], active_match)
            matches = active_match
        builder.cond(scan_body, matches, duplicate, scan_latch)
        builder.emit(scan_latch, "Add", [scan_index, one], next_scan_index)
        builder.branch(scan_latch, scan_header)
        status_values.append((
            duplicate,
            _status_branch(builder, duplicate, 0, result_block),
        ))
    else:
        builder.branch(entry, capacity_check)

    has_capacity = builder.fresh("bool")
    builder.emit(capacity_check, "Lt", [current_length, capacity], has_capacity)
    builder.cond(capacity_check, has_capacity, write, full)
    for column, value, row_shape in zip(
        columns, row_values, column_shapes, strict=True,
    ):
        row_width = prod(row_shape) if row_shape else 1
        row_offset = current_length
        if row_width != 1:
            width = builder.const(write, row_width)
            row_offset = builder.fresh("int64")
            builder.emit(write, "Mul", [current_length, width], row_offset)
        for element_index in range(row_width):
            destination_index = row_offset
            if element_index:
                destination_index = builder.fresh("int64")
                builder.emit(
                    write, "Add",
                    [row_offset, builder.const(write, element_index)],
                    destination_index,
                )
            destination_address = builder.fresh("ptr")
            builder.emit(
                write, "GetElementPtr",
                [column, destination_index], destination_address,
            )
            stored = value
            # A singleton shaped row is still an array. Store its element,
            # not its address/array value (append((..., (3.0,)))): Fortran
            # rejected the rank mismatch, while scalar-only tests missed it.
            if row_shape:
                source_address = builder.fresh("ptr")
                stored = builder.fresh(value.dtype)
                builder.emit(
                    write, "GetElementPtr",
                    [value, builder.const(write, element_index)],
                    source_address,
                )
                builder.emit(write, "Load", [source_address], stored)
            builder.emit(write, "Store", [stored, destination_address])
    if live_flags is not None:
        live_address = builder.fresh("ptr")
        builder.emit(write, "GetElementPtr", [live_flags, current_length], live_address)
        builder.emit(write, "Store", [one, live_address])
    next_length = builder.fresh("int64")
    builder.emit(write, "Add", [current_length, one], next_length)
    builder.emit(write, "Store", [next_length, length_slot])
    builder.branch(write, inserted)
    status_values.append((
        inserted,
        _status_branch(builder, inserted, 1, result_block),
    ))
    if raise_when_full:
        _raise_fixed_capacity_exhausted(builder, full, operation)
    status_values.append((
        full,
        _status_branch(builder, full, 2, result_block),
    ))
    status_result = builder.fresh("int")
    builder.emit(
        result_block,
        "Phi",
        [value for _block, value in status_values],
        status_result,
        attributes={
            "incoming_blocks": tuple(block.name for block, _value in status_values)
        },
    )
    builder.emit(result_block, "Ret", [status_result])

    name = function_name or f"ssa_sequence_{descriptor.sequence_id}_insert"
    function = Function(
        name,
        [*storage, *row_values],
        builder.blocks,
        metadata={
            "ssa_sequence_operation": operation,
            "sequence_id": descriptor.sequence_id,
            "key_columns": descriptor.key_columns,
            "allows_duplicates": descriptor.allows_duplicates,
            "sequence_row_argument_ids": tuple(value.id for value in row_values),
            "fixed_capacity": True,
            "status_values": {"duplicate": 0, "inserted": 1, "capacity_exhausted": 2},
            "sequence_array_argument_ids": tuple((
                *map(int, descriptor.column_value_ids),
                int(descriptor.length_address_id),
                *(
                    (int(descriptor.live_flags_value_id),)
                    if descriptor.live_flags_value_id is not None else ()
                ),
            )),
            "named_outputs": (("status", int(status_result.id)),),
        },
    )
    return SSASequenceLowering((function,))


def lower_sequence_append(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Lower source ``append`` through the shared destination insertion policy."""

    return lower_sequence_insert(
        descriptor,
        function_name=function_name,
        operation="append",
        first_value_id=first_value_id,
    )


def lower_record_sequence_append_with_child_copy(
    destination: SSASequenceDescriptor,
    source: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Append one outer row while snapshotting its single child sequence.

    The outer row's child-table column stores the destination row index.  The
    source sequence is copied into that row's fixed-stride slice before the
    outer length is published, so a later mutation of the returned record does
    not rewrite rows already appended to the destination.
    """

    unsupported = _unsupported_destination(destination, "append_child_copy")
    if unsupported is not None:
        return unsupported
    pool = destination.child_table_pool
    if pool is None:
        raise ValueError("child-copy append requires a destination child pool")
    if destination.key_columns:
        raise ValueError("child-copy append does not support outer key dedup")
    if source.child_table_pool is not None:
        raise ValueError("child-copy append requires a leaf source sequence")
    if (
        len(pool.column_value_ids) != len(source.column_value_ids)
        or tuple(pool.column_dtypes) != tuple(source.column_dtypes)
        or tuple(pool.key_columns) != tuple(source.key_columns)
    ):
        raise ValueError("child-copy append requires matching child row storage")
    if not pool.writable:
        raise ValueError("child-copy append requires a writable child pool")

    destination_storage = _storage_values(destination)
    pool_storage = (
        *(SSAValue(value_id, dtype=dtype) for value_id, dtype in zip(
            pool.column_value_ids, pool.column_dtypes
        )),
        SSAValue(pool.length_value_id, dtype="int64"),
        SSAValue(pool.capacity_value_id, dtype="int64"),
        SSAValue(pool.row_stride_value_id, dtype="int64"),
        *((SSAValue(pool.status_value_id, dtype="int"),)
          if pool.status_value_id is not None else ()),
        *((SSAValue(pool.live_flags_value_id, dtype="bool"),)
          if pool.live_flags_value_id is not None else ()),
    )
    source_storage = _storage_values(source)
    storage = tuple({
        int(value.id): value
        for value in (*destination_storage, *pool_storage, *source_storage)
    }.values())
    builder = _Builder(_first_fresh_identity(
        destination, source, values=storage, pools=(pool,),
        minimum=first_value_id,
    ))
    handle_column = int(pool.handle_column)
    row_values = tuple(
        builder.fresh(dtype)
        for index, dtype in enumerate(destination.column_dtypes)
        if index != handle_column
    )
    destination_columns = tuple(
        SSAValue(value_id, dtype=dtype)
        for value_id, dtype in zip(
            destination.column_value_ids, destination.column_dtypes
        )
    )
    child_columns = tuple(
        SSAValue(value_id, dtype=dtype)
        for value_id, dtype in zip(pool.column_value_ids, pool.column_dtypes)
    )
    source_columns = tuple(
        SSAValue(value_id, dtype=dtype)
        for value_id, dtype in zip(
            source.column_value_ids, source.column_dtypes
        )
    )

    entry = builder.block("entry")
    copy_header = builder.block("copy_header")
    copy_body = builder.block("copy_body")
    copy_latch = builder.block("copy_latch")
    publish_child = builder.block("publish_child")
    full = builder.block("capacity_exhausted")
    inserted = builder.block("inserted")
    result_block = builder.block("result")
    # N4 child-copy helpers seeded an int64 loop Phi with the builder's
    # default int32 zero. Index/length arithmetic is int64 on both edges;
    # status and live-flag stores retain their separate int/bool contracts.
    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")
    status_zero = builder.const(entry, 0)
    live_one = builder.const(entry, True, dtype="bool")

    outer_length_slot = builder.fresh("ptr")
    outer_length = builder.fresh("int64")
    builder.emit(entry, "GetElementPtr", [
        SSAValue(destination.length_address_id, dtype="int64", shape=(1,)),
        zero,
    ], outer_length_slot)
    builder.emit(entry, "Load", [outer_length_slot], outer_length)
    outer_has_capacity = builder.fresh("bool")
    builder.emit(entry, "Lt", [
        outer_length,
        SSAValue(destination.capacity_value_id, dtype="int64"),
    ], outer_has_capacity)

    source_length_slot = builder.fresh("ptr")
    source_length = builder.fresh("int64")
    builder.emit(entry, "GetElementPtr", [
        SSAValue(source.length_address_id, dtype="int64", shape=(1,)), zero,
    ], source_length_slot)
    builder.emit(entry, "Load", [source_length_slot], source_length)
    child_fits_stride = builder.fresh("bool")
    builder.emit(entry, "Le", [
        source_length, SSAValue(pool.row_stride_value_id, dtype="int64"),
    ], child_fits_stride)
    next_outer_length = builder.fresh("int64")
    row_end = builder.fresh("int64")
    builder.emit(entry, "Add", [outer_length, one], next_outer_length)
    builder.emit(entry, "Mul", [
        next_outer_length, SSAValue(pool.row_stride_value_id, dtype="int64"),
    ], row_end)
    child_pool_fits = builder.fresh("bool")
    builder.emit(entry, "Le", [
        row_end, SSAValue(pool.capacity_value_id, dtype="int64"),
    ], child_pool_fits)
    capacity_ok = builder.fresh("bool")
    all_capacity_ok = builder.fresh("bool")
    builder.emit(entry, "LAnd", [
        outer_has_capacity, child_fits_stride,
    ], capacity_ok)
    builder.emit(entry, "LAnd", [
        capacity_ok, child_pool_fits,
    ], all_capacity_ok)
    builder.cond(entry, all_capacity_ok, copy_header, full)

    copy_index = builder.fresh("int64")
    next_copy_index = builder.fresh("int64")
    builder.emit(copy_header, "Phi", [zero, next_copy_index], copy_index,
                 attributes={"incoming_blocks": (entry.name, copy_latch.name)})
    copy_continues = builder.fresh("bool")
    builder.emit(copy_header, "Lt", [
        copy_index, source_length,
    ], copy_continues)
    builder.cond(copy_header, copy_continues, copy_body, publish_child)

    row_offset = builder.fresh("int64")
    destination_index = builder.fresh("int64")
    builder.emit(copy_body, "Mul", [
        outer_length, SSAValue(pool.row_stride_value_id, dtype="int64"),
    ], row_offset)
    builder.emit(copy_body, "Add", [
        row_offset, copy_index,
    ], destination_index)
    for source_column, child_column in zip(source_columns, child_columns):
        source_address = builder.fresh("ptr")
        source_value = builder.fresh(source_column.dtype)
        destination_address = builder.fresh("ptr")
        builder.emit(copy_body, "GetElementPtr", [
            source_column, copy_index,
        ], source_address)
        builder.emit(copy_body, "Load", [source_address], source_value)
        builder.emit(copy_body, "GetElementPtr", [
            child_column, destination_index,
        ], destination_address)
        builder.emit(copy_body, "Store", [
            source_value, destination_address,
        ])
    if pool.live_flags_value_id is not None:
        child_live_address = builder.fresh("ptr")
        child_live_value = live_one
        if source.live_flags_value_id is not None:
            source_live_address = builder.fresh("ptr")
            child_live_value = builder.fresh("bool")
            builder.emit(copy_body, "GetElementPtr", [
                SSAValue(source.live_flags_value_id, dtype="bool"), copy_index,
            ], source_live_address)
            builder.emit(copy_body, "Load", [
                source_live_address,
            ], child_live_value)
        builder.emit(copy_body, "GetElementPtr", [
            SSAValue(pool.live_flags_value_id, dtype="bool"),
            destination_index,
        ], child_live_address)
        builder.emit(copy_body, "Store", [
            child_live_value, child_live_address,
        ])
    builder.branch(copy_body, copy_latch)
    builder.emit(copy_latch, "Add", [copy_index, one], next_copy_index)
    builder.branch(copy_latch, copy_header)

    child_length_slot = builder.fresh("ptr")
    builder.emit(publish_child, "GetElementPtr", [
        SSAValue(pool.length_value_id, dtype="int64"), outer_length,
    ], child_length_slot)
    builder.emit(publish_child, "Store", [source_length, child_length_slot])
    if pool.status_value_id is not None:
        child_status_slot = builder.fresh("ptr")
        builder.emit(publish_child, "GetElementPtr", [
            SSAValue(pool.status_value_id, dtype="int"), outer_length,
        ], child_status_slot)
        builder.emit(publish_child, "Store", [status_zero, child_status_slot])

    row_value_iter = iter(row_values)
    for index, column in enumerate(destination_columns):
        value = outer_length if index == handle_column else next(row_value_iter)
        address = builder.fresh("ptr")
        builder.emit(publish_child, "GetElementPtr", [
            column, outer_length,
        ], address)
        builder.emit(publish_child, "Store", [value, address])
    if destination.live_flags_value_id is not None:
        live_address = builder.fresh("ptr")
        builder.emit(publish_child, "GetElementPtr", [
            SSAValue(destination.live_flags_value_id, dtype="bool"),
            outer_length,
        ], live_address)
        builder.emit(publish_child, "Store", [live_one, live_address])
    builder.emit(publish_child, "Store", [
        next_outer_length, outer_length_slot,
    ])
    builder.branch(publish_child, inserted)

    inserted_status = _status_branch(builder, inserted, 1, result_block)
    _raise_fixed_capacity_exhausted(builder, full, "append_child_copy")
    full_status = _status_branch(builder, full, 2, result_block)
    status_result = builder.fresh("int")
    builder.emit(result_block, "Phi", [inserted_status, full_status],
                 status_result,
                 attributes={"incoming_blocks": (inserted.name, full.name)})
    builder.emit(result_block, "Ret", [status_result])

    name = function_name or (
        f"ssa_sequence_{destination.sequence_id}_append_child_"
        f"{source.sequence_id}"
    )
    function = Function(
        name,
        [*storage, *row_values],
        builder.blocks,
        metadata={
            "ssa_sequence_operation": "append_child_copy",
            "sequence_id": int(destination.sequence_id),
            "child_source_sequence_id": int(source.sequence_id),
            "child_handle_column": handle_column,
            "fixed_capacity": True,
            "status_values": {"inserted": 1, "capacity_exhausted": 2},
            "named_outputs": (("status", int(status_result.id)),),
        },
    )
    return SSASequenceLowering((function,))


def lower_sequence_add(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Lower source ``add`` through the same storage, retaining key dedup."""

    return lower_sequence_insert(
        descriptor,
        function_name=function_name,
        operation="add",
        first_value_id=first_value_id,
    )


def lower_sequence_sort(
    descriptor: SSASequenceDescriptor,
    scratch: SSASequenceDescriptor,
    order: SSASequenceDescriptor,
    *,
    key_column: int = 0,
    key_mapping: SSASequenceDescriptor | None = None,
    reverse: bool = False,
    function_name: str | None = None,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Lower a stable, batch-wide bottom-up merge sort into repository SSA.

    The sort orders a permutation, not the rows, and every phase is a flat
    loop whose iterations are independent of one another:

    * keys: one pass records ``order[i] = i`` and, for a mapping key, looks
      each row's key up ONCE (``n`` lookups, never one per comparison);
    * merge passes (``log2 n`` of them): every element of the pass finds its
      output position in one step, ``run start + own offset + the count of
      the opposite run that must precede it``, by a fixed-trip binary search
      that selects instead of branching; a left element precedes equal right
      elements, which keeps the sort stable (``reverse`` keeps source order
      among equals too);
    * rows move twice, whole: a gather into ``scratch`` by the final
      permutation and a copy back.

    ``order`` is caller-owned workspace: columns ``(a, b[, key])`` where
    ``a``/``b`` are int64 permutation buffers and ``key`` (present exactly
    when ``key_mapping`` is) holds the mapped key by source index.
    ``scratch`` has the row layout of ``descriptor``.

    Status 1 means sorted and status 2 means a workspace is too small.  The
    helper neither calls a host sort nor hides an algorithm in a backend.
    """

    if any(
        sequence.capacity_policy is SSASequenceCapacityPolicy.DYNAMIC
        for sequence in (descriptor, scratch, order)
    ):
        return _lower_dynamic_sequence_sort(
            descriptor, scratch, order,
            key_column=key_column, key_mapping=key_mapping, reverse=reverse,
            function_name=function_name, first_value_id=first_value_id,
        )
    unsupported = _unsupported_destination(descriptor, "sort")
    if unsupported is not None:
        return unsupported
    for workspace, label in ((scratch, "sort_scratch"), (order, "sort_order")):
        unsupported = _unsupported_destination(workspace, label)
        if unsupported is not None:
            return unsupported
    sequence_ids = {descriptor.sequence_id, scratch.sequence_id, order.sequence_id}
    if len(sequence_ids) != 3:
        raise ValueError("sequence sort requires distinct scratch and order storage")
    if (
        tuple(descriptor.column_dtypes) != tuple(scratch.column_dtypes)
        or tuple(descriptor.column_shapes) != tuple(scratch.column_shapes)
        or len(descriptor.column_value_ids) != len(scratch.column_value_ids)
    ):
        raise ValueError("sequence sort scratch must match the row layout")
    for sequence in (descriptor, scratch, order):
        if (
            sequence.key_columns
            or sequence.live_flags_value_id is not None
            or sequence.child_table_pool is not None
            or any(sequence.column_shapes)
        ):
            raise ValueError(
                "sequence sort currently requires dense scalar row columns"
            )
    if not 0 <= int(key_column) < len(descriptor.column_value_ids):
        raise ValueError("sequence sort key column is outside the row layout")
    key_value_dtype = descriptor.column_dtypes[int(key_column)]
    if key_mapping is not None:
        mapping_values = tuple(
            column for column in range(len(key_mapping.column_value_ids))
            if column not in key_mapping.key_columns
        )
        if len(key_mapping.key_columns) != 1 or len(mapping_values) != 1:
            raise ValueError(
                "sequence sort key mapping requires one key and one value column"
            )
        mapping_key = int(key_mapping.key_columns[0])
        if key_mapping.column_dtypes[mapping_key] != key_value_dtype:
            raise ValueError(
                "sequence sort row key and mapping key dtypes must match: "
                f"sequence {descriptor.sequence_id} key column "
                f"{int(key_column)} is {key_value_dtype!r} but mapping "
                f"{key_mapping.sequence_id} key column {mapping_key} is "
                f"{key_mapping.column_dtypes[mapping_key]!r}"
            )
        if key_mapping.status_address_id is None:
            raise ValueError("sequence sort key mapping requires a status cell")
        key_value_dtype = key_mapping.column_dtypes[mapping_values[0]]
    expected_order = (
        ("int64", "int64") if key_mapping is None
        else ("int64", "int64", key_value_dtype)
    )
    if tuple(order.column_dtypes) != expected_order:
        raise ValueError(
            "sequence sort order workspace must have columns "
            f"{expected_order!r}, got {tuple(order.column_dtypes)!r}"
        )

    destination_storage = _storage_values(descriptor)
    scratch_storage = _storage_values(scratch)
    order_storage = _storage_values(order)
    key_storage = (
        (*_storage_values(key_mapping), SSAValue(
            int(key_mapping.status_address_id), dtype="int", shape=(1,),
        ))
        if key_mapping is not None else ()
    )
    arguments = tuple({
        int(value.id): value
        for value in (
            *destination_storage, *scratch_storage, *order_storage,
            *key_storage,
        )
    }.values())
    values = {int(value.id): value for value in arguments}
    builder = _Builder(_first_fresh_identity(
        descriptor, scratch, order,
        *((key_mapping,) if key_mapping is not None else ()),
        values=arguments, minimum=first_value_id,
    ))
    destination_columns = tuple(
        values[int(value_id)] for value_id in descriptor.column_value_ids
    )
    scratch_columns = tuple(
        values[int(value_id)] for value_id in scratch.column_value_ids
    )
    order_columns = tuple(
        values[int(value_id)] for value_id in order.column_value_ids
    )
    permutation_in, permutation_out = order_columns[0], order_columns[1]
    mapped_keys = order_columns[2] if key_mapping is not None else None

    entry = builder.block("entry")
    size_check = builder.block("size_check")
    key_header = builder.block("key_header")
    key_body = builder.block("key_body")
    pass_header = builder.block("pass_header")
    flat_header = builder.block("flat_header")
    flat_body = builder.block("flat_body")
    search_header = builder.block("search_header")
    search_body = builder.block("search_body")
    search_done = builder.block("search_done")
    copy_header = builder.block("copy_header")
    copy_body = builder.block("copy_body")
    pass_latch = builder.block("pass_latch")
    gather_header = builder.block("gather_header")
    gather_body = builder.block("gather_body")
    restore_header = builder.block("restore_header")
    restore_body = builder.block("restore_body")
    success = builder.block("success")
    workspace_small = builder.block("workspace_too_small")
    result = builder.block("result")

    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")
    two = builder.const(entry, 2, dtype="int64")
    length_slot = builder.fresh("ptr")
    length = builder.fresh("int64")
    builder.emit(entry, "GetElementPtr", [
        values[int(descriptor.length_address_id)], zero,
    ], length_slot)
    builder.emit(entry, "Load", [length_slot], length)
    scratch_fits = builder.fresh("bool")
    order_fits = builder.fresh("bool")
    workspace_fits = builder.fresh("bool")
    builder.emit(entry, "Ge", [
        values[int(scratch.capacity_value_id)], length,
    ], scratch_fits)
    builder.emit(entry, "Ge", [
        values[int(order.capacity_value_id)], length,
    ], order_fits)
    builder.emit(entry, "LAnd", [scratch_fits, order_fits], workspace_fits)
    builder.cond(entry, workspace_fits, size_check, workspace_small)

    needs_sort = builder.fresh("bool")
    builder.emit(size_check, "Gt", [length, one], needs_sort)
    builder.cond(size_check, needs_sort, key_header, success)

    name = function_name or f"ssa_sequence_{descriptor.sequence_id}_sort"
    key_lookup = None
    if key_mapping is not None:
        key_lookup = lower_table_lookup(
            key_mapping,
            function_name=f"{name}__key_lookup",
            first_value_id=GLOBAL_MONOTONIC_IDS.peek(),
            required=True,
        )

    def element_pointer(
        block: BasicBlock, array: SSAValue, index: SSAValue,
    ) -> SSAValue:
        pointer = builder.fresh("ptr")
        builder.emit(block, "GetElementPtr", [array, index], pointer)
        return pointer

    def load_at(
        block: BasicBlock, array: SSAValue, index: SSAValue, dtype: str,
    ) -> SSAValue:
        value = builder.fresh(dtype)
        builder.emit(
            block, "Load", [element_pointer(block, array, index)], value,
        )
        return value

    def source_key(block: BasicBlock, source_index: SSAValue) -> SSAValue:
        """The key of the row that started at ``source_index``."""
        if mapped_keys is not None:
            return load_at(block, mapped_keys, source_index, key_value_dtype)
        return load_at(
            block, destination_columns[int(key_column)], source_index,
            key_value_dtype,
        )

    def copy_row(
        block: BasicBlock,
        source_columns: tuple[SSAValue, ...],
        source_index: SSAValue,
        target_columns: tuple[SSAValue, ...],
        target_index: SSAValue,
    ) -> None:
        for source, target, dtype in zip(
            source_columns, target_columns, descriptor.column_dtypes,
        ):
            value = load_at(block, source, source_index, dtype)
            builder.emit(block, "Store", [
                value, element_pointer(block, target, target_index),
            ])

    # -- keys: one flat pass --------------------------------------------
    key_index = builder.fresh("int64")
    next_key_index = builder.fresh("int64")
    builder.emit(key_header, "Phi", [zero, next_key_index], key_index,
                 attributes={"incoming_blocks": (
                     size_check.name, key_body.name,
                 )})
    keys_remain = builder.fresh("bool")
    builder.emit(key_header, "Lt", [key_index, length], keys_remain)
    builder.cond(key_header, keys_remain, key_body, pass_header)
    builder.emit(key_body, "Store", [
        key_index, element_pointer(key_body, permutation_in, key_index),
    ])
    if key_mapping is not None:
        row_key = load_at(
            key_body, destination_columns[int(key_column)], key_index,
            descriptor.column_dtypes[int(key_column)],
        )
        mapped = builder.fresh(key_value_dtype)
        builder.emit(
            key_body, "Call", [*key_storage, row_key], mapped,
            attributes={
                "callee": key_lookup.functions[-1].name,
                "source_linked": True,
                "ssa_sequence_operation": "sort_key_lookup",
                "sequence_id": int(key_mapping.sequence_id),
            },
        )
        builder.emit(key_body, "Store", [
            mapped, element_pointer(key_body, mapped_keys, key_index),
        ])
    builder.emit(key_body, "Add", [key_index, one], next_key_index)
    builder.branch(key_body, key_header)

    # -- merge passes ---------------------------------------------------
    width = builder.fresh("int64")
    next_width = builder.fresh("int64")
    steps = builder.fresh("int64")
    next_steps = builder.fresh("int64")
    builder.emit(pass_header, "Phi", [one, next_width], width,
                 attributes={"incoming_blocks": (
                     key_header.name, pass_latch.name,
                 )})
    builder.emit(pass_header, "Phi", [one, next_steps], steps,
                 attributes={"incoming_blocks": (
                     key_header.name, pass_latch.name,
                 )})
    another_pass = builder.fresh("bool")
    builder.emit(pass_header, "Lt", [width, length], another_pass)
    builder.cond(pass_header, another_pass, flat_header, gather_header)

    element = builder.fresh("int64")
    next_element = builder.fresh("int64")
    builder.emit(flat_header, "Phi", [zero, next_element], element,
                 attributes={"incoming_blocks": (
                     pass_header.name, search_done.name,
                 )})
    elements_remain = builder.fresh("bool")
    builder.emit(flat_header, "Lt", [element, length], elements_remain)
    builder.cond(flat_header, elements_remain, flat_body, copy_header)

    pair_width = builder.fresh("int64")
    run_number = builder.fresh("int64")
    run_start = builder.fresh("int64")
    mid_raw = builder.fresh("int64")
    mid_inside = builder.fresh("bool")
    run_mid = builder.fresh("int64")
    end_raw = builder.fresh("int64")
    end_inside = builder.fresh("bool")
    run_end = builder.fresh("int64")
    in_left = builder.fresh("bool")
    left_offset = builder.fresh("int64")
    right_offset = builder.fresh("int64")
    own_offset = builder.fresh("int64")
    opposite_base = builder.fresh("int64")
    left_count = builder.fresh("int64")
    right_count = builder.fresh("int64")
    opposite_length = builder.fresh("int64")
    builder.emit(flat_body, "Add", [width, width], pair_width)
    builder.emit(flat_body, "FloorDiv", [element, pair_width], run_number)
    builder.emit(flat_body, "Mul", [run_number, pair_width], run_start)
    builder.emit(flat_body, "Add", [run_start, width], mid_raw)
    builder.emit(flat_body, "Lt", [mid_raw, length], mid_inside)
    builder.emit(flat_body, "Select", [mid_inside, mid_raw, length], run_mid)
    builder.emit(flat_body, "Add", [run_start, pair_width], end_raw)
    builder.emit(flat_body, "Lt", [end_raw, length], end_inside)
    builder.emit(flat_body, "Select", [end_inside, end_raw, length], run_end)
    builder.emit(flat_body, "Lt", [element, run_mid], in_left)
    builder.emit(flat_body, "Sub", [element, run_start], left_offset)
    builder.emit(flat_body, "Sub", [element, run_mid], right_offset)
    builder.emit(flat_body, "Select", [in_left, left_offset, right_offset],
                 own_offset)
    builder.emit(flat_body, "Select", [in_left, run_mid, run_start],
                 opposite_base)
    builder.emit(flat_body, "Sub", [run_end, run_mid], right_count)
    builder.emit(flat_body, "Sub", [run_mid, run_start], left_count)
    builder.emit(flat_body, "Select", [in_left, right_count, left_count],
                 opposite_length)
    source_index = load_at(flat_body, permutation_in, element, "int64")
    element_key = source_key(flat_body, source_index)
    builder.branch(flat_body, search_header)

    # Fixed-trip binary search over the opposite run: ``position`` ends as
    # the number of its elements that must precede this one.
    position = builder.fresh("int64")
    upper = builder.fresh("int64")
    step = builder.fresh("int64")
    next_position = builder.fresh("int64")
    next_upper = builder.fresh("int64")
    next_step = builder.fresh("int64")
    builder.emit(search_header, "Phi", [zero, next_position], position,
                 attributes={"incoming_blocks": (
                     flat_body.name, search_body.name,
                 )})
    builder.emit(search_header, "Phi", [opposite_length, next_upper], upper,
                 attributes={"incoming_blocks": (
                     flat_body.name, search_body.name,
                 )})
    builder.emit(search_header, "Phi", [zero, next_step], step,
                 attributes={"incoming_blocks": (
                     flat_body.name, search_body.name,
                 )})
    searching = builder.fresh("bool")
    builder.emit(search_header, "Lt", [step, steps], searching)
    builder.cond(search_header, searching, search_body, search_done)

    active = builder.fresh("bool")
    span = builder.fresh("int64")
    midpoint = builder.fresh("int64")
    probe_raw = builder.fresh("int64")
    probe = builder.fresh("int64")
    builder.emit(search_body, "Lt", [position, upper], active)
    builder.emit(search_body, "Add", [position, upper], span)
    builder.emit(search_body, "FloorDiv", [span, two], midpoint)
    builder.emit(search_body, "Add", [opposite_base, midpoint], probe_raw)
    builder.emit(search_body, "Select", [active, probe_raw, zero], probe)
    probe_source = load_at(search_body, permutation_in, probe, "int64")
    probe_key = source_key(search_body, probe_source)
    left_precedes = builder.fresh("bool")
    right_precedes = builder.fresh("bool")
    precedes = builder.fresh("bool")
    # A left element yields to opposite elements strictly before it; a right
    # element to the opposite elements before it or equal: equals keep order.
    builder.emit(
        search_body, "Gt" if reverse else "Lt",
        [probe_key, element_key], left_precedes,
    )
    builder.emit(
        search_body, "Ge" if reverse else "Le",
        [probe_key, element_key], right_precedes,
    )
    builder.emit(search_body, "Select", [in_left, left_precedes, right_precedes],
                 precedes)
    take = builder.fresh("bool")
    after_midpoint = builder.fresh("int64")
    shrunk = builder.fresh("int64")
    builder.emit(search_body, "LAnd", [active, precedes], take)
    builder.emit(search_body, "Add", [midpoint, one], after_midpoint)
    builder.emit(search_body, "Select", [take, after_midpoint, position],
                 next_position)
    builder.emit(search_body, "Select", [active, midpoint, upper], shrunk)
    builder.emit(search_body, "Select", [precedes, upper, shrunk], next_upper)
    builder.emit(search_body, "Add", [step, one], next_step)
    builder.branch(search_body, search_header)

    run_offset = builder.fresh("int64")
    output_index = builder.fresh("int64")
    builder.emit(search_done, "Add", [run_start, own_offset], run_offset)
    builder.emit(search_done, "Add", [run_offset, position], output_index)
    builder.emit(search_done, "Store", [
        source_index, element_pointer(search_done, permutation_out, output_index),
    ])
    builder.emit(search_done, "Add", [element, one], next_element)
    builder.branch(search_done, flat_header)

    copy_index = builder.fresh("int64")
    next_copy_index = builder.fresh("int64")
    builder.emit(copy_header, "Phi", [zero, next_copy_index], copy_index,
                 attributes={"incoming_blocks": (
                     flat_header.name, copy_body.name,
                 )})
    copies_remain = builder.fresh("bool")
    builder.emit(copy_header, "Lt", [copy_index, length], copies_remain)
    builder.cond(copy_header, copies_remain, copy_body, pass_latch)
    moved = load_at(copy_body, permutation_out, copy_index, "int64")
    builder.emit(copy_body, "Store", [
        moved, element_pointer(copy_body, permutation_in, copy_index),
    ])
    builder.emit(copy_body, "Add", [copy_index, one], next_copy_index)
    builder.branch(copy_body, copy_header)

    builder.emit(pass_latch, "Add", [width, width], next_width)
    builder.emit(pass_latch, "Add", [steps, one], next_steps)
    builder.branch(pass_latch, pass_header)

    # -- rows move twice, whole -----------------------------------------
    gather_index = builder.fresh("int64")
    next_gather_index = builder.fresh("int64")
    builder.emit(gather_header, "Phi", [zero, next_gather_index], gather_index,
                 attributes={"incoming_blocks": (
                     pass_header.name, gather_body.name,
                 )})
    gathers_remain = builder.fresh("bool")
    builder.emit(gather_header, "Lt", [gather_index, length], gathers_remain)
    builder.cond(gather_header, gathers_remain, gather_body, restore_header)
    gathered_source = load_at(gather_body, permutation_in, gather_index, "int64")
    copy_row(
        gather_body, destination_columns, gathered_source,
        scratch_columns, gather_index,
    )
    builder.emit(gather_body, "Add", [gather_index, one], next_gather_index)
    builder.branch(gather_body, gather_header)

    restore_index = builder.fresh("int64")
    next_restore_index = builder.fresh("int64")
    builder.emit(restore_header, "Phi", [zero, next_restore_index], restore_index,
                 attributes={"incoming_blocks": (
                     gather_header.name, restore_body.name,
                 )})
    restores_remain = builder.fresh("bool")
    builder.emit(restore_header, "Lt", [restore_index, length], restores_remain)
    builder.cond(restore_header, restores_remain, restore_body, success)
    copy_row(
        restore_body, scratch_columns, restore_index,
        destination_columns, restore_index,
    )
    builder.emit(restore_body, "Add", [restore_index, one], next_restore_index)
    builder.branch(restore_body, restore_header)

    success_status = _status_branch(builder, success, 1, result)
    workspace_status = _status_branch(builder, workspace_small, 2, result)
    status = builder.fresh("int")
    builder.emit(
        result, "Phi", [success_status, workspace_status], status,
        attributes={"incoming_blocks": (success.name, workspace_small.name)},
    )
    builder.emit(result, "Ret", [status])

    function = Function(name, list(arguments), builder.blocks, metadata={
        "ssa_sequence_operation": "sort",
        "sequence_id": int(descriptor.sequence_id),
        "scratch_sequence_id": int(scratch.sequence_id),
        "order_sequence_id": int(order.sequence_id),
        "key_column": int(key_column),
        "key_mapping_sequence_id": (
            None if key_mapping is None else int(key_mapping.sequence_id)
        ),
        "reverse": bool(reverse),
        "stable": True,
        "algorithm": "batch_rank_merge_sort",
        "status_values": {"sorted": 1, "workspace_too_small": 2},
        "sequence_array_argument_ids": tuple(value.id for value in arguments),
        "named_outputs": (("status", int(status.id)),),
    })
    return SSASequenceLowering((
        *((key_lookup.functions) if key_lookup is not None else ()), function,
    ))


def _lower_dynamic_sequence_sort(
    descriptor, scratch, order, *, key_column, key_mapping, reverse,
    function_name, first_value_id,
):
    """Reserve DYNAMIC sort workspaces to the rows being sorted, then sort.

    Program: ``xs = [5, n, 2, 9]; ys = sorted(xs)`` (scratch c120).  Wrong
    record: the result arena and the scratch/order workspaces were
    caller-provided root formals whose capacities no caller can know; the
    harness passed 0, the copy failed its capacity check, and the sort's
    status overwrote the failure.  Rule: a sort over heap-owned storage
    reserves each DYNAMIC workspace to the destination's length (the
    existing checked reserve) and runs the fixed sort on the same cells.
    The destination itself is never grown by a sort.
    """

    name = function_name or f"ssa_sequence_{descriptor.sequence_id}_sort"
    sequences = (descriptor, scratch, order)
    for sequence, label in zip(sequences, ("sort", "sort_scratch", "sort_order")):
        unsupported = _unsupported_destination(sequence, label, allow_dynamic=True)
        if unsupported is not None:
            return unsupported
    fixed = lower_sequence_sort(
        *(replace(sequence, capacity_policy=SSASequenceCapacityPolicy.FIXED)
          for sequence in sequences),
        key_column=key_column, key_mapping=key_mapping, reverse=reverse,
        function_name=name + "__fixed", first_value_id=first_value_id,
    )
    if not fixed.complete:
        return fixed
    sort = fixed.entry
    reserves = tuple(
        lower_sequence_reserve(
            workspace, function_name=f"{name}__reserve_{label}",
        ).functions[0]
        for workspace, label in ((scratch, "scratch"), (order, "order"))
        if workspace.capacity_policy is SSASequenceCapacityPolicy.DYNAMIC
    )

    def own(value):
        return SSAValue(value.id, value.dtype, shape=tuple(value.shape or ()),
                        accounting=dict(value.accounting or {}))

    values = {int(value.id): own(value) for value in sort.args}
    for reserve in reserves:
        for value in reserve.args:
            if int(value.id) != int(reserve.metadata["required_capacity_value_id"]):
                values.setdefault(int(value.id), own(value))
    builder = _Builder(_first_fresh_identity(
        *sequences, values=tuple(values.values()),
    ))
    entry = builder.block("entry")
    result = builder.block("result")
    zero = builder.const(entry, 0, dtype="int64")
    length_slot, length = builder.fresh("ptr"), builder.fresh("int64")
    builder.emit(entry, "GetElementPtr",
                 [values[int(descriptor.length_address_id)], zero], length_slot)
    builder.emit(entry, "Load", [length_slot], length)
    current = entry
    outcomes = []
    for reserve in reserves:
        reserve_values = {
            **values, int(reserve.metadata["required_capacity_value_id"]): length,
        }
        reserve_status = builder.fresh("int")
        builder.emit(current, "Call",
                     [reserve_values[int(value.id)] for value in reserve.args],
                     reserve_status, attributes={"callee": reserve.name})
        reserved = builder.fresh("bool")
        builder.emit(current, "Lt", [reserve_status, builder.const(current, 2)], reserved)
        following = builder.block(f"reserved_{reserve.name.rsplit('_', 1)[-1]}")
        builder.cond(current, reserved, following, result)
        outcomes.append((current, reserve_status))
        current = following
    sort_status = builder.fresh("int")
    builder.emit(current, "Call", [values[int(value.id)] for value in sort.args],
                 sort_status, attributes={"callee": sort.name})
    builder.branch(current, result)
    outcomes.append((current, sort_status))
    status = builder.fresh("int")
    builder.emit(result, "Phi", [value for _block, value in outcomes], status,
                 attributes={"incoming_blocks": tuple(block.name for block, _ in outcomes)})
    builder.emit(result, "Ret", [status])
    arguments = list(values.values())
    function = Function(name, arguments, builder.blocks, metadata={
        **sort.metadata, "fixed_capacity": False,
        "status_values": {"sorted": 1, "workspace_too_small": 2},
        "sequence_array_argument_ids": tuple(int(value.id) for value in arguments),
        "named_outputs": (("status", int(status.id)),),
    })
    return SSASequenceLowering((function, *fixed.functions, *reserves))


def lower_sequence_fill(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
    raise_when_full: bool = True,
) -> SSASequenceLowering:
    """Fill a caller arena with one repeated value and publish its length.

    The count and every row index are int64, the width of the length and
    capacity cells they are compared with and stored into
    (``_storage_values``).  Program ``xs = [2] * (n % 4 + 1)`` (fuzzer
    p3_54): a 32-bit count formal met the caller's int64 count ("storage
    types are immutable") and was stored into the int64 length cell.
    """

    unsupported = _unsupported_destination(descriptor, "fill", allow_dynamic=True)
    if unsupported is not None:
        return unsupported
    if descriptor.capacity_policy is SSASequenceCapacityPolicy.DYNAMIC:
        return _lower_dynamic_sequence_mutation(descriptor, function_name=function_name,
            operation="fill", first_value_id=first_value_id)
    if len(descriptor.column_value_ids) != 1 or descriptor.key_columns:
        raise ValueError("sequence fill requires one duplicate-policy column")
    storage = _storage_values(descriptor)
    builder = _Builder(_first_fresh_identity(
        descriptor, values=storage, minimum=first_value_id
    ))
    value = builder.fresh(descriptor.column_dtypes[0])
    requested = builder.fresh("int64")
    entry = builder.block("entry")
    header = builder.block("fill_header")
    body = builder.block("fill_body")
    latch = builder.block("fill_latch")
    complete = builder.block("complete")
    full = builder.block("capacity_exhausted")
    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")
    within_capacity = builder.fresh("bool")
    builder.emit(entry, "Le", [requested, SSAValue(
        descriptor.capacity_value_id, dtype="int64"
    )], within_capacity)
    builder.cond(entry, within_capacity, header, full)
    index = builder.fresh("int64")
    next_index = builder.fresh("int64")
    builder.emit(header, "Phi", [zero, next_index], index, attributes={
        "incoming_blocks": (entry.name, latch.name)
    })
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, requested], continues)
    builder.cond(header, continues, body, complete)
    address = builder.fresh("ptr")
    builder.emit(body, "GetElementPtr", [SSAValue(
        descriptor.column_value_ids[0], dtype=descriptor.column_dtypes[0]
    ), index], address)
    builder.emit(body, "Store", [value, address])
    builder.branch(body, latch)
    builder.emit(latch, "Add", [index, one], next_index)
    builder.branch(latch, header)
    length_slot = builder.fresh("ptr")
    builder.emit(complete, "GetElementPtr", [SSAValue(
        descriptor.length_address_id, dtype="int64", shape=(1,)
    ), zero], length_slot)
    builder.emit(complete, "Store", [requested, length_slot])
    status_ok = builder.const(complete, 1)
    builder.emit(complete, "Ret", [status_ok])
    # A fixed arena's capacity is a proof; a fill that exceeds it means the
    # proof was wrong.  Program ``xs = [2] * (n % 4 + 1)`` (fuzzer p3_54):
    # the arena took the literal's one-row extent, the fill reported "full"
    # into a status nobody reads and ``len(xs)`` was 0.  Rule: exhausting
    # a fixed fill raises (validation error 75), never a silent short list.
    # The DYNAMIC protocol's insert step (``raise_when_full=False``) reports
    # "full" instead: its caller grows the arena and retries (scratch c_k1,
    # ``[5] * (n % 4 + 1)``, raised 75 on its first insert).
    if raise_when_full:
        builder.emit(full, "Call", [], attributes={
            "callee": "turing_validation_error",
            "error_code": 75,
            "source_linked": True,
            "ssa_sequence_operation": "fill",
            "failure": "fixed_capacity_exhausted",
        })
    status_full = builder.const(full, 2)
    builder.emit(full, "Ret", [status_full])
    function = Function(
        function_name or f"ssa_sequence_{descriptor.sequence_id}_fill",
        [*storage, value, requested],
        builder.blocks,
        metadata={
            "ssa_sequence_operation": "fill",
            "sequence_id": descriptor.sequence_id,
            "status_values": {"filled": 1, "capacity_exhausted": 2},
            "sequence_array_argument_ids": (
                descriptor.column_value_ids[0], descriptor.length_address_id
            ),
            "named_outputs": (("status", int(status_ok.id)),),
        },
    )
    return SSASequenceLowering((function,))


def lower_sequence_append_fill(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
    raise_when_full: bool = True,
) -> SSASequenceLowering:
    """Append one repeated value without discarding resident rows."""

    unsupported = _unsupported_destination(descriptor, "append_fill", allow_dynamic=True)
    if unsupported is not None:
        return unsupported
    if descriptor.capacity_policy is SSASequenceCapacityPolicy.DYNAMIC:
        return _lower_dynamic_sequence_mutation(descriptor, function_name=function_name,
            operation="append_fill", first_value_id=first_value_id)
    if len(descriptor.column_value_ids) != 1 or descriptor.key_columns:
        raise ValueError(
            "sequence append-fill requires one duplicate-policy column"
        )
    storage = _storage_values(descriptor)
    builder = _Builder(_first_fresh_identity(
        descriptor, values=storage, minimum=first_value_id
    ))
    value = builder.fresh(descriptor.column_dtypes[0])
    requested = builder.fresh("int64")
    entry = builder.block("entry")
    header = builder.block("append_fill_header")
    body = builder.block("append_fill_body")
    latch = builder.block("append_fill_latch")
    complete = builder.block("complete")
    full = builder.block("capacity_exhausted")
    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")
    length_address = builder.fresh("ptr")
    old_length = builder.fresh("int64")
    new_length = builder.fresh("int64")
    builder.emit(entry, "GetElementPtr", [SSAValue(
        descriptor.length_address_id, dtype="int64", shape=(1,)
    ), zero], length_address)
    builder.emit(entry, "Load", [length_address], old_length)
    builder.emit(entry, "Add", [old_length, requested], new_length)
    within_capacity = builder.fresh("bool")
    builder.emit(entry, "Le", [new_length, SSAValue(
        descriptor.capacity_value_id, dtype="int64"
    )], within_capacity)
    builder.cond(entry, within_capacity, header, full)
    index = builder.fresh("int64")
    next_index = builder.fresh("int64")
    builder.emit(header, "Phi", [zero, next_index], index, attributes={
        "incoming_blocks": (entry.name, latch.name)
    })
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, requested], continues)
    builder.cond(header, continues, body, complete)
    destination_index = builder.fresh("int64")
    address = builder.fresh("ptr")
    builder.emit(body, "Add", [old_length, index], destination_index)
    builder.emit(body, "GetElementPtr", [SSAValue(
        descriptor.column_value_ids[0], dtype=descriptor.column_dtypes[0]
    ), destination_index], address)
    builder.emit(body, "Store", [value, address])
    builder.branch(body, latch)
    builder.emit(latch, "Add", [index, one], next_index)
    builder.branch(latch, header)
    builder.emit(complete, "Store", [new_length, length_address])
    status_ok = builder.const(complete, 1)
    builder.emit(complete, "Ret", [status_ok])
    # A fixed arena's capacity is a proof; a fill that exceeds it means the
    # proof was wrong.  Program ``xs = [2] * (n % 4 + 1)`` (fuzzer p3_54):
    # the arena took the literal's one-row extent, the fill reported "full"
    # into a status nobody reads and ``len(xs)`` was 0.  Rule: exhausting
    # a fixed fill raises (validation error 75), never a silent short list.
    # The DYNAMIC protocol's insert step (``raise_when_full=False``) reports
    # "full" instead: its caller grows the arena and retries (scratch c_k1,
    # ``[5] * (n % 4 + 1)``, raised 75 on its first insert).
    if raise_when_full:
        builder.emit(full, "Call", [], attributes={
            "callee": "turing_validation_error",
            "error_code": 75,
            "source_linked": True,
            "ssa_sequence_operation": "fill",
            "failure": "fixed_capacity_exhausted",
        })
    status_full = builder.const(full, 2)
    builder.emit(full, "Ret", [status_full])
    function = Function(
        function_name
        or f"ssa_sequence_{descriptor.sequence_id}_append_fill",
        [*storage, value, requested],
        builder.blocks,
        metadata={
            "ssa_sequence_operation": "append_fill",
            "sequence_id": descriptor.sequence_id,
            "status_values": {"appended": 1, "capacity_exhausted": 2},
            "sequence_array_argument_ids": (
                descriptor.column_value_ids[0], descriptor.length_address_id
            ),
            "named_outputs": (("status", int(status_ok.id)),),
        },
    )
    return SSASequenceLowering((function,))


def lower_sequence_append_slice(
    destination: SSASequenceDescriptor,
    source: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Append a unit-stride source slice to resident destination storage.

    Bounds are normalized exactly like a positive-step Python slice: negative
    bounds are relative to the source length and both bounds are clipped into
    ``[0, source_length]``.  The slice is copied directly between caller-owned
    arenas; no temporary sequence or runtime iterator is materialized.
    """

    # Program: ``values=[]; for i in range(n): values.append(i); selected=values[:end]``.
    # Wrong record: the slice is a fresh DYNAMIC sequence with a proven owner,
    # yet append_slice refused before copying any rows because it had only a
    # fixed-capacity branch. Rule: reserve ``old_length + clipped_count``
    # through the descriptor's existing checked owner protocol before copying,
    # and publish the new length only after the copy completes.
    unsupported = _unsupported_destination(
        destination, "append_slice", allow_dynamic=True,
    )
    if unsupported is not None:
        return unsupported
    if (
        len(destination.column_value_ids) != 1
        or len(source.column_value_ids) != 1
        or destination.key_columns
    ):
        raise ValueError(
            "sequence append-slice requires one duplicate-policy column"
        )

    if destination.capacity_policy is SSASequenceCapacityPolicy.DYNAMIC and (
        destination.child_table_pool is not None
        or destination.live_flags_value_id is not None
        or any(destination.column_shapes)
    ):
        return SSASequenceLowering(shortfalls=(SSASequenceLoweringShortfall(
            SSASequenceShortfallCode.DYNAMIC_GROWTH_UNAVAILABLE,
            destination.sequence_id,
            "append_slice",
            "dynamic append-slice requires dense scalar rows without live "
            "flags or child pools",
        ),))

    destination_storage = _storage_values(destination)
    source_storage = _storage_values(source)
    all_storage = tuple({
        value.id: value
        for value in (*destination_storage, *source_storage)
    }.values())
    name = function_name or (
        f"ssa_sequence_{destination.sequence_id}_append_slice_"
        f"{source.sequence_id}"
    )
    reserve_function = None
    if destination.capacity_policy is SSASequenceCapacityPolicy.DYNAMIC:
        reserve_lowering = lower_sequence_reserve(
            destination, function_name=name + "__reserve",
        )
        if not reserve_lowering.complete:
            return reserve_lowering
        reserve_function = reserve_lowering.functions[0]
        reserve_ids = {
            int(value.id)
            for value in reserve_function.args
        } | {
            int(instruction.res.id)
            for block in reserve_function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        }
        first_value_id = max(
            int(first_value_id or 0), max(reserve_ids, default=-1) + 1,
        )
    builder = _Builder(_first_fresh_identity(
        destination, source, values=all_storage, minimum=first_value_id
    ))
    lower = builder.fresh("int64")
    upper = builder.fresh("int64")
    entry = builder.block("entry")
    lower_negative = builder.block("lower_negative")
    lower_negative_zero = builder.block("lower_negative_zero")
    lower_negative_value = builder.block("lower_negative_value")
    lower_nonnegative = builder.block("lower_nonnegative")
    lower_nonnegative_length = builder.block("lower_nonnegative_length")
    lower_nonnegative_value = builder.block("lower_nonnegative_value")
    lower_merge = builder.block("lower_merge")
    upper_negative = builder.block("upper_negative")
    upper_negative_zero = builder.block("upper_negative_zero")
    upper_negative_value = builder.block("upper_negative_value")
    upper_nonnegative = builder.block("upper_nonnegative")
    upper_nonnegative_length = builder.block("upper_nonnegative_length")
    upper_nonnegative_value = builder.block("upper_nonnegative_value")
    bounds_merge = builder.block("bounds_merge")
    empty = builder.block("empty")
    capacity = builder.block("capacity")
    reserve_block = (
        builder.block("append_slice_reserve")
        if reserve_function is not None else None
    )
    header = builder.block("append_slice_header")
    body = builder.block("append_slice_body")
    latch = builder.block("append_slice_latch")
    complete = builder.block("complete")
    full = builder.block("capacity_exhausted")
    result = builder.block("result")

    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")
    source_length_slot = builder.fresh("ptr")
    source_length = builder.fresh("int64")
    destination_length_slot = builder.fresh("ptr")
    old_length = builder.fresh("int64")
    builder.emit(entry, "GetElementPtr", [SSAValue(
        source.length_address_id, dtype="int64", shape=(1,)
    ), zero], source_length_slot)
    builder.emit(entry, "Load", [source_length_slot], source_length)
    builder.emit(entry, "GetElementPtr", [SSAValue(
        destination.length_address_id, dtype="int64", shape=(1,)
    ), zero], destination_length_slot)
    builder.emit(entry, "Load", [destination_length_slot], old_length)
    lower_is_negative = builder.fresh("bool")
    builder.emit(entry, "Lt", [lower, zero], lower_is_negative)
    builder.cond(entry, lower_is_negative, lower_negative, lower_nonnegative)

    relative_lower = builder.fresh("int64")
    builder.emit(lower_negative, "Add", [source_length, lower], relative_lower)
    relative_lower_is_negative = builder.fresh("bool")
    builder.emit(
        lower_negative, "Lt", [relative_lower, zero],
        relative_lower_is_negative,
    )
    builder.cond(
        lower_negative, relative_lower_is_negative,
        lower_negative_zero, lower_negative_value,
    )
    builder.branch(lower_negative_zero, lower_merge)
    builder.branch(lower_negative_value, lower_merge)
    lower_exceeds_length = builder.fresh("bool")
    builder.emit(
        lower_nonnegative, "Gt", [lower, source_length],
        lower_exceeds_length,
    )
    builder.cond(
        lower_nonnegative, lower_exceeds_length,
        lower_nonnegative_length, lower_nonnegative_value,
    )
    builder.branch(lower_nonnegative_length, lower_merge)
    builder.branch(lower_nonnegative_value, lower_merge)
    normalized_lower = builder.fresh("int64")
    builder.emit(
        lower_merge,
        "Phi",
        [zero, relative_lower, source_length, lower],
        normalized_lower,
        attributes={"incoming_blocks": (
            lower_negative_zero.name,
            lower_negative_value.name,
            lower_nonnegative_length.name,
            lower_nonnegative_value.name,
        )},
    )
    upper_is_negative = builder.fresh("bool")
    builder.emit(lower_merge, "Lt", [upper, zero], upper_is_negative)
    builder.cond(
        lower_merge, upper_is_negative, upper_negative, upper_nonnegative
    )

    relative_upper = builder.fresh("int64")
    builder.emit(upper_negative, "Add", [source_length, upper], relative_upper)
    relative_upper_is_negative = builder.fresh("bool")
    builder.emit(
        upper_negative, "Lt", [relative_upper, zero],
        relative_upper_is_negative,
    )
    builder.cond(
        upper_negative, relative_upper_is_negative,
        upper_negative_zero, upper_negative_value,
    )
    builder.branch(upper_negative_zero, bounds_merge)
    builder.branch(upper_negative_value, bounds_merge)
    upper_exceeds_length = builder.fresh("bool")
    builder.emit(
        upper_nonnegative, "Gt", [upper, source_length],
        upper_exceeds_length,
    )
    builder.cond(
        upper_nonnegative, upper_exceeds_length,
        upper_nonnegative_length, upper_nonnegative_value,
    )
    builder.branch(upper_nonnegative_length, bounds_merge)
    builder.branch(upper_nonnegative_value, bounds_merge)
    normalized_upper = builder.fresh("int64")
    builder.emit(
        bounds_merge,
        "Phi",
        [zero, relative_upper, source_length, upper],
        normalized_upper,
        attributes={"incoming_blocks": (
            upper_negative_zero.name,
            upper_negative_value.name,
            upper_nonnegative_length.name,
            upper_nonnegative_value.name,
        )},
    )
    nonempty = builder.fresh("bool")
    builder.emit(
        bounds_merge, "Gt", [normalized_upper, normalized_lower], nonempty
    )
    builder.cond(bounds_merge, nonempty, capacity, empty)
    empty_status = _status_branch(builder, empty, 1, result)

    requested = builder.fresh("int64")
    new_length = builder.fresh("int64")
    builder.emit(capacity, "Sub", [normalized_upper, normalized_lower], requested)
    builder.emit(capacity, "Add", [old_length, requested], new_length)
    if reserve_function is not None:
        storage_by_id = {int(value.id): value for value in destination_storage}
        reserve_status = builder.fresh("int")
        builder.emit(reserve_block, "Call", [
            *(storage_by_id[int(value.id)] for value in reserve_function.args[:-1]),
            new_length,
        ], reserve_status, attributes={
            "callee": reserve_function.name,
            "source_linked": True,
        })
        reserve_succeeded = builder.fresh("bool")
        builder.emit(reserve_block, "Lt", [
            reserve_status, builder.const(reserve_block, 2),
        ], reserve_succeeded)
        builder.cond(reserve_block, reserve_succeeded, header, full)
        builder.branch(capacity, reserve_block)
        capacity_predecessor = reserve_block.name
    else:
        within_capacity = builder.fresh("bool")
        builder.emit(capacity, "Le", [new_length, SSAValue(
            destination.capacity_value_id, dtype="int64"
        )], within_capacity)
        builder.cond(capacity, within_capacity, header, full)
        capacity_predecessor = capacity.name
    index = builder.fresh("int64")
    next_index = builder.fresh("int64")
    builder.emit(
        header, "Phi", [zero, next_index], index,
        attributes={"incoming_blocks": (capacity_predecessor, latch.name)},
    )
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, requested], continues)
    builder.cond(header, continues, body, complete)
    source_index = builder.fresh("int64")
    source_address = builder.fresh("ptr")
    value = builder.fresh(source.column_dtypes[0])
    destination_index = builder.fresh("int64")
    destination_address = builder.fresh("ptr")
    builder.emit(body, "Add", [normalized_lower, index], source_index)
    builder.emit(body, "GetElementPtr", [SSAValue(
        source.column_value_ids[0], dtype=source.column_dtypes[0]
    ), source_index], source_address)
    builder.emit(body, "Load", [source_address], value)
    builder.emit(body, "Add", [old_length, index], destination_index)
    builder.emit(body, "GetElementPtr", [SSAValue(
        destination.column_value_ids[0], dtype=destination.column_dtypes[0]
    ), destination_index], destination_address)
    builder.emit(body, "Store", [value, destination_address])
    builder.branch(body, latch)
    builder.emit(latch, "Add", [index, one], next_index)
    builder.branch(latch, header)
    builder.emit(complete, "Store", [new_length, destination_length_slot])
    complete_status = _status_branch(builder, complete, 1, result)
    if reserve_function is None:
        _raise_fixed_capacity_exhausted(builder, full, "append_slice")
    full_status = _status_branch(builder, full, 2, result)
    status = builder.fresh("int")
    builder.emit(
        result,
        "Phi",
        [empty_status, complete_status, full_status],
        status,
        attributes={"incoming_blocks": (
            empty.name, complete.name, full.name
        )},
    )
    builder.emit(result, "Ret", [status])
    function = Function(
        name,
        [*all_storage, lower, upper],
        builder.blocks,
        metadata={
            "ssa_sequence_operation": "append_slice",
            "destination_sequence_id": destination.sequence_id,
            "source_sequence_id": source.sequence_id,
            "slice_step": 1,
            "sequence_array_argument_ids": tuple((
                int(destination.column_value_ids[0]),
                int(destination.length_address_id),
                int(source.column_value_ids[0]),
                int(source.length_address_id),
                *((int(destination.heap_owner_value_id),)
                  if destination.heap_owner_value_id is not None else ()),
                *((int(source.heap_owner_value_id),)
                  if source.heap_owner_value_id is not None else ()),
            )),
            "named_outputs": (("status", int(status.id)),),
        },
    )
    return SSASequenceLowering((
        function,
        *((reserve_function,) if reserve_function is not None else ()),
    ))


def lower_sequence_pack_bits(
    destination: SSASequenceDescriptor,
    source: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Pack a 0/1 byte sequence into little-endian fixed-width words."""

    unsupported = _unsupported_destination(destination, "pack_bits")
    if unsupported is not None:
        return unsupported
    if (
        len(destination.column_value_ids) != 1
        or len(source.column_value_ids) != 1
        or destination.key_columns
    ):
        raise ValueError("bit packing requires one duplicate-policy column")
    storage = tuple({
        value.id: value
        for value in (*_storage_values(destination), *_storage_values(source))
    }.values())
    builder = _Builder(_first_fresh_identity(
        destination, source, values=storage, minimum=first_value_id
    ))
    width = builder.fresh("int")
    entry = builder.block("entry")
    outer_header = builder.block("word_header")
    outer_body = builder.block("word_body")
    inner_header = builder.block("bit_header")
    inner_body = builder.block("bit_body")
    inner_latch = builder.block("bit_latch")
    store_word = builder.block("store_word")
    outer_latch = builder.block("word_latch")
    complete = builder.block("complete")
    full = builder.block("capacity_exhausted")
    invalid = builder.block("invalid_width")
    result = builder.block("result")
    zero = builder.const(entry, 0)
    one = builder.const(entry, 1)
    source_length_slot = builder.fresh("ptr")
    source_length = builder.fresh("int")
    destination_length_slot = builder.fresh("ptr")
    builder.emit(entry, "GetElementPtr", [SSAValue(
        source.length_address_id, dtype="int64", shape=(1,)
    ), zero], source_length_slot)
    builder.emit(entry, "Load", [source_length_slot], source_length)
    builder.emit(entry, "GetElementPtr", [SSAValue(
        destination.length_address_id, dtype="int64", shape=(1,)
    ), zero], destination_length_slot)
    width_valid = builder.fresh("bool")
    builder.emit(entry, "Gt", [width, zero], width_valid)
    capacity_check = builder.block("capacity_check")
    builder.cond(entry, width_valid, capacity_check, invalid)
    rounded = builder.fresh("int")
    word_count = builder.fresh("int")
    builder.emit(capacity_check, "Sub", [width, one], rounded)
    rounded_length = builder.fresh("int")
    builder.emit(capacity_check, "Add", [source_length, rounded], rounded_length)
    builder.emit(capacity_check, "FloorDiv", [rounded_length, width], word_count)
    fits = builder.fresh("bool")
    builder.emit(capacity_check, "Le", [word_count, SSAValue(
        destination.capacity_value_id, dtype="int64"
    )], fits)
    builder.cond(capacity_check, fits, outer_header, full)

    word_index = builder.fresh("int")
    next_word = builder.fresh("int")
    builder.emit(
        outer_header, "Phi", [zero, next_word], word_index,
        attributes={"incoming_blocks": (capacity_check.name, outer_latch.name)},
    )
    words_continue = builder.fresh("bool")
    builder.emit(outer_header, "Lt", [word_index, word_count], words_continue)
    builder.cond(outer_header, words_continue, outer_body, complete)
    source_start = builder.fresh("int")
    builder.emit(outer_body, "Mul", [word_index, width], source_start)
    builder.branch(outer_body, inner_header)

    bit_index = builder.fresh("int")
    next_bit = builder.fresh("int")
    accumulated = builder.fresh(destination.column_dtypes[0])
    next_accumulated = builder.fresh(destination.column_dtypes[0])
    builder.emit(
        inner_header, "Phi", [zero, next_bit], bit_index,
        attributes={"incoming_blocks": (outer_body.name, inner_latch.name)},
    )
    builder.emit(
        inner_header, "Phi", [zero, next_accumulated], accumulated,
        attributes={"incoming_blocks": (outer_body.name, inner_latch.name)},
    )
    source_index = builder.fresh("int")
    builder.emit(inner_header, "Add", [source_start, bit_index], source_index)
    under_width = builder.fresh("bool")
    under_length = builder.fresh("bool")
    continues = builder.fresh("bool")
    builder.emit(inner_header, "Lt", [bit_index, width], under_width)
    builder.emit(inner_header, "Lt", [source_index, source_length], under_length)
    builder.emit(inner_header, "LAnd", [under_width, under_length], continues)
    builder.cond(inner_header, continues, inner_body, store_word)
    source_address = builder.fresh("ptr")
    bit = builder.fresh(source.column_dtypes[0])
    shifted = builder.fresh(destination.column_dtypes[0])
    builder.emit(inner_body, "GetElementPtr", [SSAValue(
        source.column_value_ids[0], dtype=source.column_dtypes[0]
    ), source_index], source_address)
    builder.emit(inner_body, "Load", [source_address], bit)
    builder.emit(inner_body, "Shl", [bit, bit_index], shifted)
    builder.emit(inner_body, "Or", [accumulated, shifted], next_accumulated)
    builder.branch(inner_body, inner_latch)
    builder.emit(inner_latch, "Add", [bit_index, one], next_bit)
    builder.branch(inner_latch, inner_header)
    destination_address = builder.fresh("ptr")
    builder.emit(store_word, "GetElementPtr", [SSAValue(
        destination.column_value_ids[0], dtype=destination.column_dtypes[0]
    ), word_index], destination_address)
    builder.emit(store_word, "Store", [accumulated, destination_address])
    builder.branch(store_word, outer_latch)
    builder.emit(outer_latch, "Add", [word_index, one], next_word)
    builder.branch(outer_latch, outer_header)
    builder.emit(complete, "Store", [word_count, destination_length_slot])
    ok = _status_branch(builder, complete, 1, result)
    _raise_fixed_capacity_exhausted(builder, full, "pack_bits")
    full_status = _status_branch(builder, full, 2, result)
    invalid_status = _status_branch(builder, invalid, 3, result)
    status = builder.fresh("int")
    builder.emit(
        result, "Phi", [ok, full_status, invalid_status], status,
        attributes={"incoming_blocks": (complete.name, full.name, invalid.name)},
    )
    builder.emit(result, "Ret", [status])
    name = function_name or (
        f"ssa_sequence_{destination.sequence_id}_pack_bits_{source.sequence_id}"
    )
    return SSASequenceLowering((Function(
        name, [*storage, width], builder.blocks,
        metadata={
            "ssa_sequence_operation": "pack_bits",
            "destination_sequence_id": destination.sequence_id,
            "source_sequence_id": source.sequence_id,
            "sequence_array_argument_ids": tuple((
                int(destination.column_value_ids[0]),
                int(destination.length_address_id),
                int(source.column_value_ids[0]),
                int(source.length_address_id),
            )),
            "named_outputs": (("status", int(status.id)),),
        },
    ),))


def lower_sequence_prepend(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Insert one scalar at index zero by shifting resident rows right."""

    unsupported = _unsupported_destination(descriptor, "prepend")
    if unsupported is not None:
        return unsupported
    if len(descriptor.column_value_ids) != 1 or descriptor.key_columns:
        raise ValueError("sequence prepend requires one duplicate-policy column")
    storage = _storage_values(descriptor)
    builder = _Builder(_first_fresh_identity(
        descriptor, values=storage, minimum=first_value_id
    ))
    value = builder.fresh(descriptor.column_dtypes[0])
    entry = builder.block("entry")
    header = builder.block("shift_header")
    body = builder.block("shift_body")
    latch = builder.block("shift_latch")
    store_prefix = builder.block("store_prefix")
    full = builder.block("capacity_exhausted")
    result = builder.block("result")
    zero = builder.const(entry, 0)
    one = builder.const(entry, 1)
    length_slot = builder.fresh("ptr")
    old_length = builder.fresh("int")
    new_length = builder.fresh("int")
    builder.emit(entry, "GetElementPtr", [SSAValue(
        descriptor.length_address_id, dtype="int64", shape=(1,)
    ), zero], length_slot)
    builder.emit(entry, "Load", [length_slot], old_length)
    builder.emit(entry, "Add", [old_length, one], new_length)
    fits = builder.fresh("bool")
    builder.emit(entry, "Le", [new_length, SSAValue(
        descriptor.capacity_value_id, dtype="int64"
    )], fits)
    builder.cond(entry, fits, header, full)
    index = builder.fresh("int")
    previous_index = builder.fresh("int")
    builder.emit(
        header, "Phi", [old_length, previous_index], index,
        attributes={"incoming_blocks": (entry.name, latch.name)},
    )
    continues = builder.fresh("bool")
    builder.emit(header, "Gt", [index, zero], continues)
    builder.cond(header, continues, body, store_prefix)
    source_index = builder.fresh("int")
    source_address = builder.fresh("ptr")
    existing = builder.fresh(descriptor.column_dtypes[0])
    destination_address = builder.fresh("ptr")
    builder.emit(body, "Sub", [index, one], source_index)
    builder.emit(body, "GetElementPtr", [SSAValue(
        descriptor.column_value_ids[0], dtype=descriptor.column_dtypes[0]
    ), source_index], source_address)
    builder.emit(body, "Load", [source_address], existing)
    builder.emit(body, "GetElementPtr", [SSAValue(
        descriptor.column_value_ids[0], dtype=descriptor.column_dtypes[0]
    ), index], destination_address)
    builder.emit(body, "Store", [existing, destination_address])
    builder.branch(body, latch)
    builder.emit(latch, "Sub", [index, one], previous_index)
    builder.branch(latch, header)
    prefix_address = builder.fresh("ptr")
    builder.emit(store_prefix, "GetElementPtr", [SSAValue(
        descriptor.column_value_ids[0], dtype=descriptor.column_dtypes[0]
    ), zero], prefix_address)
    builder.emit(store_prefix, "Store", [value, prefix_address])
    builder.emit(store_prefix, "Store", [new_length, length_slot])
    ok = _status_branch(builder, store_prefix, 1, result)
    _raise_fixed_capacity_exhausted(builder, full, "prepend")
    full_status = _status_branch(builder, full, 2, result)
    status = builder.fresh("int")
    builder.emit(
        result, "Phi", [ok, full_status], status,
        attributes={"incoming_blocks": (store_prefix.name, full.name)},
    )
    builder.emit(result, "Ret", [status])
    name = function_name or f"ssa_sequence_{descriptor.sequence_id}_prepend"
    return SSASequenceLowering((Function(
        name, [*storage, value], builder.blocks,
        metadata={
            "ssa_sequence_operation": "prepend",
            "sequence_id": descriptor.sequence_id,
            "sequence_array_argument_ids": (
                int(descriptor.column_value_ids[0]),
                int(descriptor.length_address_id),
            ),
            "named_outputs": (("status", int(status.id)),),
        },
    ),))


def _positional_row_columns(
    descriptor: SSASequenceDescriptor,
) -> tuple[SSAValue, ...]:
    """The row columns as element views, as ``lower_sequence_insert`` spells
    them (a heap column's cell is resolved by its storage formal)."""

    return tuple(
        SSAValue(int(value_id), dtype=str(dtype))
        for value_id, dtype in zip(
            descriptor.column_value_ids,
            descriptor.column_dtypes
            or ("unknown",) * len(descriptor.column_value_ids),
        )
    )


def _positional_unsupported(
    descriptor: SSASequenceDescriptor, operation: str,
) -> SSASequenceLowering | None:
    unsupported = _unsupported_destination(
        descriptor, operation, allow_dynamic=True,
    )
    if unsupported is not None:
        return unsupported
    if (
        descriptor.key_columns
        or descriptor.live_flags_value_id is not None
        or descriptor.child_table_pool is not None
        or any(descriptor.column_shapes or ())
    ):
        return SSASequenceLowering(shortfalls=(SSASequenceLoweringShortfall(
            SSASequenceShortfallCode.DYNAMIC_GROWTH_UNAVAILABLE,
            descriptor.sequence_id,
            operation,
            "positional list operations require dense scalar duplicate-"
            "policy rows (no keys, live flags, child pools or shaped rows)",
        ),))
    return None


def _move_row(
    builder: _Builder, block: BasicBlock, columns: tuple[SSAValue, ...],
    source_index: SSAValue, destination_index: SSAValue,
) -> None:
    for column in columns:
        source_address = builder.fresh("ptr")
        destination_address = builder.fresh("ptr")
        moved = builder.fresh(column.dtype)
        builder.emit(block, "GetElementPtr", [column, source_index], source_address)
        builder.emit(block, "Load", [source_address], moved)
        builder.emit(block, "GetElementPtr", [column, destination_index], destination_address)
        builder.emit(block, "Store", [moved, destination_address])


def lower_sequence_delete_at(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Remove row ``index`` by shifting every later row left by one.

    ``list.pop(i)`` reads row ``i`` first and then calls this; ``del xs[i]``
    is this alone.  ``index`` is already normalized by the caller (Python's
    negative-index rule).  Status 1 means removed; status 2 means ``index``
    names no row (Python's IndexError), and nothing is changed.  Removing a
    row never grows storage, so a DYNAMIC arena uses its cells as they are.
    """

    unsupported = _positional_unsupported(descriptor, "delete_at")
    if unsupported is not None:
        return unsupported
    storage = _storage_values(descriptor)
    builder = _Builder(_first_fresh_identity(
        descriptor, values=storage, minimum=first_value_id,
    ))
    index = builder.fresh("int64")
    columns = _positional_row_columns(descriptor)
    length_address = next(
        value for value in storage if value.id == descriptor.length_address_id
    )
    entry = builder.block("entry")
    header = builder.block("shift_header")
    body = builder.block("shift_body")
    done = builder.block("removed")
    outside = builder.block("index_outside")
    result = builder.block("result")
    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")
    length_slot, length = builder.fresh("ptr"), builder.fresh("int64")
    builder.emit(entry, "GetElementPtr", [length_address, zero], length_slot)
    builder.emit(entry, "Load", [length_slot], length)
    at_least_zero, below_length, inside = (
        builder.fresh("bool"), builder.fresh("bool"), builder.fresh("bool"),
    )
    builder.emit(entry, "Ge", [index, zero], at_least_zero)
    builder.emit(entry, "Lt", [index, length], below_length)
    builder.emit(entry, "LAnd", [at_least_zero, below_length], inside)
    last = builder.fresh("int64")
    builder.emit(entry, "Sub", [length, one], last)
    builder.cond(entry, inside, header, outside)
    position, following = builder.fresh("int64"), builder.fresh("int64")
    builder.emit(
        header, "Phi", [index, following], position,
        attributes={"incoming_blocks": (entry.name, body.name)},
    )
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [position, last], continues)
    builder.cond(header, continues, body, done)
    builder.emit(body, "Add", [position, one], following)
    _move_row(builder, body, columns, following, position)
    builder.branch(body, header)
    builder.emit(done, "Store", [last, length_slot])
    removed = _status_branch(builder, done, 1, result)
    rejected = _status_branch(builder, outside, 2, result)
    status = builder.fresh("int")
    builder.emit(
        result, "Phi", [removed, rejected], status,
        attributes={"incoming_blocks": (done.name, outside.name)},
    )
    builder.emit(result, "Ret", [status])
    name = function_name or f"ssa_sequence_{descriptor.sequence_id}_delete_at"
    return SSASequenceLowering((Function(
        name, [*storage, index], builder.blocks,
        metadata={
            "ssa_sequence_operation": "delete_at",
            "sequence_id": descriptor.sequence_id,
            "sequence_array_argument_ids": _storage_array_ids(descriptor, storage),
            "status_values": {"removed": 1, "index_outside": 2},
            "named_outputs": (("status", int(status.id)),),
        },
    ),))


def lower_sequence_insert_at(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
    raise_when_full: bool = True,
) -> SSASequenceLowering:
    """``list.insert(index, row)``: Python's index rule, then a right shift.

    A negative ``index`` counts from the end, and the result is clamped to
    ``[0, length]`` (inserting past either end appends or prepends, as
    Python does).  Status 1 means inserted; status 2 means fixed capacity is
    exhausted and nothing changed.  A DYNAMIC arena reserves and retries
    through ``_lower_dynamic_sequence_mutation``.
    """

    unsupported = _positional_unsupported(descriptor, "insert_at")
    if unsupported is not None:
        return unsupported
    if descriptor.capacity_policy is SSASequenceCapacityPolicy.DYNAMIC:
        return _lower_dynamic_sequence_mutation(
            descriptor, function_name=function_name, operation="insert_at",
            first_value_id=first_value_id,
        )
    storage = _storage_values(descriptor)
    builder = _Builder(_first_fresh_identity(
        descriptor, values=storage, minimum=first_value_id,
    ))
    columns = _positional_row_columns(descriptor)
    index = builder.fresh("int64")
    row_values = tuple(builder.fresh(column.dtype) for column in columns)
    length_address = next(
        value for value in storage if value.id == descriptor.length_address_id
    )
    capacity = SSAValue(descriptor.capacity_value_id, dtype="int64")
    entry = builder.block("entry")
    negative = builder.block("negative_index")
    nonnegative = builder.block("normalized_low")
    clamp_high = builder.block("clamp_high")
    header = builder.block("shift_header")
    body = builder.block("shift_body")
    write = builder.block("write")
    full = builder.block("capacity_exhausted")
    result = builder.block("result")
    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")
    length_slot, length = builder.fresh("ptr"), builder.fresh("int64")
    builder.emit(entry, "GetElementPtr", [length_address, zero], length_slot)
    builder.emit(entry, "Load", [length_slot], length)
    new_length, fits = builder.fresh("int64"), builder.fresh("bool")
    builder.emit(entry, "Add", [length, one], new_length)
    builder.emit(entry, "Le", [new_length, capacity], fits)
    is_negative = builder.fresh("bool")
    builder.emit(entry, "Lt", [index, zero], is_negative)
    builder.cond(entry, is_negative, negative, nonnegative)
    from_end, from_end_clamped = builder.fresh("int64"), builder.fresh("int64")
    builder.emit(negative, "Add", [index, length], from_end)
    builder.emit(negative, "Max", [from_end, zero], from_end_clamped)
    builder.branch(negative, nonnegative)
    low = builder.fresh("int64")
    builder.emit(
        nonnegative, "Phi", [from_end_clamped, index], low,
        attributes={"incoming_blocks": (negative.name, entry.name)},
    )
    builder.branch(nonnegative, clamp_high)
    target = builder.fresh("int64")
    builder.emit(clamp_high, "Min", [low, length], target)
    builder.cond(clamp_high, fits, header, full)
    position, previous = builder.fresh("int64"), builder.fresh("int64")
    builder.emit(
        header, "Phi", [length, previous], position,
        attributes={"incoming_blocks": (clamp_high.name, body.name)},
    )
    continues = builder.fresh("bool")
    builder.emit(header, "Gt", [position, target], continues)
    builder.cond(header, continues, body, write)
    builder.emit(body, "Sub", [position, one], previous)
    _move_row(builder, body, columns, previous, position)
    builder.branch(body, header)
    for column, value in zip(columns, row_values):
        address = builder.fresh("ptr")
        builder.emit(write, "GetElementPtr", [column, target], address)
        builder.emit(write, "Store", [value, address])
    builder.emit(write, "Store", [new_length, length_slot])
    inserted = _status_branch(builder, write, 1, result)
    if raise_when_full:
        _raise_fixed_capacity_exhausted(builder, full, "insert_at")
    exhausted = _status_branch(builder, full, 2, result)
    status = builder.fresh("int")
    builder.emit(
        result, "Phi", [inserted, exhausted], status,
        attributes={"incoming_blocks": (write.name, full.name)},
    )
    builder.emit(result, "Ret", [status])
    name = function_name or f"ssa_sequence_{descriptor.sequence_id}_insert_at"
    return SSASequenceLowering((Function(
        name, [*storage, index, *row_values], builder.blocks,
        metadata={
            "ssa_sequence_operation": "insert_at",
            "sequence_id": descriptor.sequence_id,
            "sequence_array_argument_ids": _storage_array_ids(descriptor, storage),
            "status_values": {"inserted": 1, "capacity_exhausted": 2},
            "named_outputs": (("status", int(status.id)),),
        },
    ),))


def lower_sequence_reverse(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """``list.reverse()``: swap row ``i`` with row ``length - 1 - i``.

    Status 1 always: reversing never changes the length or the storage.
    """

    unsupported = _positional_unsupported(descriptor, "reverse")
    if unsupported is not None:
        return unsupported
    storage = _storage_values(descriptor)
    builder = _Builder(_first_fresh_identity(
        descriptor, values=storage, minimum=first_value_id,
    ))
    columns = _positional_row_columns(descriptor)
    length_address = next(
        value for value in storage if value.id == descriptor.length_address_id
    )
    entry = builder.block("entry")
    header = builder.block("swap_header")
    body = builder.block("swap_body")
    done = builder.block("reversed")
    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")
    length_slot, length = builder.fresh("ptr"), builder.fresh("int64")
    builder.emit(entry, "GetElementPtr", [length_address, zero], length_slot)
    builder.emit(entry, "Load", [length_slot], length)
    last = builder.fresh("int64")
    builder.emit(entry, "Sub", [length, one], last)
    builder.branch(entry, header)
    low, high = builder.fresh("int64"), builder.fresh("int64")
    next_low, next_high = builder.fresh("int64"), builder.fresh("int64")
    builder.emit(
        header, "Phi", [zero, next_low], low,
        attributes={"incoming_blocks": (entry.name, body.name)},
    )
    builder.emit(
        header, "Phi", [last, next_high], high,
        attributes={"incoming_blocks": (entry.name, body.name)},
    )
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [low, high], continues)
    builder.cond(header, continues, body, done)
    for column in columns:
        low_address, high_address = builder.fresh("ptr"), builder.fresh("ptr")
        low_row, high_row = builder.fresh(column.dtype), builder.fresh(column.dtype)
        builder.emit(body, "GetElementPtr", [column, low], low_address)
        builder.emit(body, "GetElementPtr", [column, high], high_address)
        builder.emit(body, "Load", [low_address], low_row)
        builder.emit(body, "Load", [high_address], high_row)
        builder.emit(body, "Store", [high_row, low_address])
        builder.emit(body, "Store", [low_row, high_address])
    builder.emit(body, "Add", [low, one], next_low)
    builder.emit(body, "Sub", [high, one], next_high)
    builder.branch(body, header)
    status = builder.const(done, 1)
    builder.emit(done, "Ret", [status])
    name = function_name or f"ssa_sequence_{descriptor.sequence_id}_reverse"
    return SSASequenceLowering((Function(
        name, list(storage), builder.blocks,
        metadata={
            "ssa_sequence_operation": "reverse",
            "sequence_id": descriptor.sequence_id,
            "sequence_array_argument_ids": _storage_array_ids(descriptor, storage),
            "status_values": {"reversed": 1},
            "named_outputs": (("status", int(status.id)),),
        },
    ),))


def lower_sequence_prepend_packed_bytes(
    destination: SSASequenceDescriptor,
    source: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Prepend one scalar and a packed little-endian byte sequence."""

    unsupported = _unsupported_destination(destination, "prepend_packed_bytes")
    if unsupported is not None:
        return unsupported
    if (
        len(destination.column_value_ids) != 1
        or len(source.column_value_ids) != 1
        or destination.key_columns
    ):
        raise ValueError(
            "packed-byte prepend requires one duplicate-policy column"
        )
    storage = tuple({
        value.id: value
        for value in (*_storage_values(destination), *_storage_values(source))
    }.values())
    builder = _Builder(_first_fresh_identity(
        destination, source, values=storage, minimum=first_value_id
    ))
    prefix = builder.fresh(destination.column_dtypes[0])
    byte_width = builder.fresh("int")
    entry = builder.block("entry")
    capacity_check = builder.block("capacity_check")
    shift_header = builder.block("shift_header")
    shift_body = builder.block("shift_body")
    shift_latch = builder.block("shift_latch")
    prefix_store = builder.block("prefix_store")
    word_header = builder.block("word_header")
    word_body = builder.block("word_body")
    byte_header = builder.block("byte_header")
    byte_body = builder.block("byte_body")
    byte_latch = builder.block("byte_latch")
    word_store = builder.block("word_store")
    word_latch = builder.block("word_latch")
    complete = builder.block("complete")
    full = builder.block("capacity_exhausted")
    invalid = builder.block("invalid_width")
    result = builder.block("result")
    zero = builder.const(entry, 0)
    one = builder.const(entry, 1)
    eight = builder.const(entry, 8)
    source_length_slot = builder.fresh("ptr")
    source_length = builder.fresh("int")
    destination_length_slot = builder.fresh("ptr")
    old_length = builder.fresh("int")
    builder.emit(entry, "GetElementPtr", [SSAValue(
        source.length_address_id, dtype="int64", shape=(1,)
    ), zero], source_length_slot)
    builder.emit(entry, "Load", [source_length_slot], source_length)
    builder.emit(entry, "GetElementPtr", [SSAValue(
        destination.length_address_id, dtype="int64", shape=(1,)
    ), zero], destination_length_slot)
    builder.emit(entry, "Load", [destination_length_slot], old_length)
    valid = builder.fresh("bool")
    builder.emit(entry, "Gt", [byte_width, zero], valid)
    builder.cond(entry, valid, capacity_check, invalid)
    rounded_width = builder.fresh("int")
    rounded_length = builder.fresh("int")
    word_count = builder.fresh("int")
    prefix_count = builder.fresh("int")
    new_length = builder.fresh("int")
    builder.emit(capacity_check, "Sub", [byte_width, one], rounded_width)
    builder.emit(
        capacity_check, "Add", [source_length, rounded_width], rounded_length
    )
    builder.emit(
        capacity_check, "FloorDiv", [rounded_length, byte_width], word_count
    )
    builder.emit(capacity_check, "Add", [word_count, one], prefix_count)
    builder.emit(capacity_check, "Add", [old_length, prefix_count], new_length)
    fits = builder.fresh("bool")
    builder.emit(capacity_check, "Le", [new_length, SSAValue(
        destination.capacity_value_id, dtype="int64"
    )], fits)
    builder.cond(capacity_check, fits, shift_header, full)

    shift_index = builder.fresh("int")
    previous_shift = builder.fresh("int")
    builder.emit(
        shift_header, "Phi", [old_length, previous_shift], shift_index,
        attributes={"incoming_blocks": (capacity_check.name, shift_latch.name)},
    )
    shifting = builder.fresh("bool")
    builder.emit(shift_header, "Gt", [shift_index, zero], shifting)
    builder.cond(shift_header, shifting, shift_body, prefix_store)
    old_index = builder.fresh("int")
    shifted_index = builder.fresh("int")
    old_address = builder.fresh("ptr")
    old_value = builder.fresh(destination.column_dtypes[0])
    shifted_address = builder.fresh("ptr")
    builder.emit(shift_body, "Sub", [shift_index, one], old_index)
    builder.emit(shift_body, "Add", [old_index, prefix_count], shifted_index)
    builder.emit(shift_body, "GetElementPtr", [SSAValue(
        destination.column_value_ids[0], dtype=destination.column_dtypes[0]
    ), old_index], old_address)
    builder.emit(shift_body, "Load", [old_address], old_value)
    builder.emit(shift_body, "GetElementPtr", [SSAValue(
        destination.column_value_ids[0], dtype=destination.column_dtypes[0]
    ), shifted_index], shifted_address)
    builder.emit(shift_body, "Store", [old_value, shifted_address])
    builder.branch(shift_body, shift_latch)
    builder.emit(shift_latch, "Sub", [shift_index, one], previous_shift)
    builder.branch(shift_latch, shift_header)
    prefix_address = builder.fresh("ptr")
    builder.emit(prefix_store, "GetElementPtr", [SSAValue(
        destination.column_value_ids[0], dtype=destination.column_dtypes[0]
    ), zero], prefix_address)
    builder.emit(prefix_store, "Store", [prefix, prefix_address])
    builder.branch(prefix_store, word_header)

    word_index = builder.fresh("int")
    next_word = builder.fresh("int")
    builder.emit(
        word_header, "Phi", [zero, next_word], word_index,
        attributes={"incoming_blocks": (prefix_store.name, word_latch.name)},
    )
    words_continue = builder.fresh("bool")
    builder.emit(word_header, "Lt", [word_index, word_count], words_continue)
    builder.cond(word_header, words_continue, word_body, complete)
    source_start = builder.fresh("int")
    builder.emit(word_body, "Mul", [word_index, byte_width], source_start)
    builder.branch(word_body, byte_header)
    byte_index = builder.fresh("int")
    next_byte = builder.fresh("int")
    accumulated = builder.fresh(destination.column_dtypes[0])
    next_accumulated = builder.fresh(destination.column_dtypes[0])
    builder.emit(
        byte_header, "Phi", [zero, next_byte], byte_index,
        attributes={"incoming_blocks": (word_body.name, byte_latch.name)},
    )
    builder.emit(
        byte_header, "Phi", [zero, next_accumulated], accumulated,
        attributes={"incoming_blocks": (word_body.name, byte_latch.name)},
    )
    source_index = builder.fresh("int")
    builder.emit(byte_header, "Add", [source_start, byte_index], source_index)
    under_width = builder.fresh("bool")
    under_length = builder.fresh("bool")
    bytes_continue = builder.fresh("bool")
    builder.emit(byte_header, "Lt", [byte_index, byte_width], under_width)
    builder.emit(byte_header, "Lt", [source_index, source_length], under_length)
    builder.emit(byte_header, "LAnd", [under_width, under_length], bytes_continue)
    builder.cond(byte_header, bytes_continue, byte_body, word_store)
    source_address = builder.fresh("ptr")
    byte = builder.fresh(source.column_dtypes[0])
    shift_amount = builder.fresh("int")
    shifted = builder.fresh(destination.column_dtypes[0])
    builder.emit(byte_body, "GetElementPtr", [SSAValue(
        source.column_value_ids[0], dtype=source.column_dtypes[0]
    ), source_index], source_address)
    builder.emit(byte_body, "Load", [source_address], byte)
    builder.emit(byte_body, "Mul", [byte_index, eight], shift_amount)
    builder.emit(byte_body, "Shl", [byte, shift_amount], shifted)
    builder.emit(byte_body, "Or", [accumulated, shifted], next_accumulated)
    builder.branch(byte_body, byte_latch)
    builder.emit(byte_latch, "Add", [byte_index, one], next_byte)
    builder.branch(byte_latch, byte_header)
    destination_index = builder.fresh("int")
    destination_address = builder.fresh("ptr")
    builder.emit(word_store, "Add", [word_index, one], destination_index)
    builder.emit(word_store, "GetElementPtr", [SSAValue(
        destination.column_value_ids[0], dtype=destination.column_dtypes[0]
    ), destination_index], destination_address)
    builder.emit(word_store, "Store", [accumulated, destination_address])
    builder.branch(word_store, word_latch)
    builder.emit(word_latch, "Add", [word_index, one], next_word)
    builder.branch(word_latch, word_header)
    builder.emit(complete, "Store", [new_length, destination_length_slot])
    ok = _status_branch(builder, complete, 1, result)
    _raise_fixed_capacity_exhausted(builder, full, "prepend_packed_bytes")
    full_status = _status_branch(builder, full, 2, result)
    invalid_status = _status_branch(builder, invalid, 3, result)
    status = builder.fresh("int")
    builder.emit(
        result, "Phi", [ok, full_status, invalid_status], status,
        attributes={"incoming_blocks": (complete.name, full.name, invalid.name)},
    )
    builder.emit(result, "Ret", [status])
    name = function_name or (
        f"ssa_sequence_{destination.sequence_id}_prepend_packed_"
        f"{source.sequence_id}"
    )
    return SSASequenceLowering((Function(
        name, [*storage, prefix, byte_width], builder.blocks,
        metadata={
            "ssa_sequence_operation": "prepend_packed_bytes",
            "destination_sequence_id": destination.sequence_id,
            "source_sequence_id": source.sequence_id,
            "sequence_array_argument_ids": tuple((
                int(destination.column_value_ids[0]),
                int(destination.length_address_id),
                int(source.column_value_ids[0]),
                int(source.length_address_id),
            )),
            "named_outputs": (("status", int(status.id)),),
        },
    ),))


def lower_sequence_contains(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Build a resident membership scan returning one boolean value.

    Keyed tables compare their declared key tuple.  A one-column duplicate
    sequence compares its sole value column, which is Python's membership
    contract for strings, bytes, tuples, and lists.  Wider unkeyed rows have
    no scalar-query contract and remain inadmissible.
    """

    membership_columns = tuple(descriptor.key_columns) or (
        (0,) if len(descriptor.column_value_ids) == 1 else ()
    )
    if not membership_columns:
        raise ValueError(
            "sequence membership requires keyed columns or one value column"
        )
    storage = _storage_values(descriptor)
    builder = _Builder(_first_fresh_identity(
        descriptor, values=storage, minimum=first_value_id
    ))
    key_dtypes = tuple(
        descriptor.column_dtypes[column] for column in membership_columns
    )
    queries = tuple(builder.fresh(dtype) for dtype in key_dtypes)
    entry = builder.block("entry")
    header = builder.block("contains_header")
    body = builder.block("contains_body")
    latch = builder.block("contains_latch")
    found = builder.block("found")
    absent = builder.block("absent")
    result_block = builder.block("result")
    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")
    length_slot = builder.fresh("ptr")
    length = builder.fresh("int64")
    builder.emit(
        entry,
        "GetElementPtr",
        [SSAValue(descriptor.length_address_id, dtype="int64", shape=(1,)), zero],
        length_slot,
    )
    builder.emit(entry, "Load", [length_slot], length)
    builder.branch(entry, header)
    index = builder.fresh("int64")
    next_index = builder.fresh("int64")
    builder.emit(
        header,
        "Phi",
        [zero, next_index],
        index,
        attributes={"incoming_blocks": (entry.name, latch.name)},
    )
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, length], continues)
    builder.cond(header, continues, body, absent)
    matches = None
    for key_column, key_dtype, query in zip(
        membership_columns, key_dtypes, queries
    ):
        address = builder.fresh("ptr")
        existing = builder.fresh(key_dtype)
        component_match = builder.fresh("bool")
        builder.emit(
            body,
            "GetElementPtr",
            [SSAValue(descriptor.column_value_ids[key_column], dtype=key_dtype), index],
            address,
        )
        builder.emit(body, "Load", [address], existing)
        builder.emit(body, "Eq", [existing, query], component_match)
        if matches is None:
            matches = component_match
        else:
            combined = builder.fresh("bool")
            builder.emit(body, "LAnd", [matches, component_match], combined)
            matches = combined
    assert matches is not None
    if descriptor.live_flags_value_id is not None:
        live_address = builder.fresh("ptr")
        live = builder.fresh("bool")
        active_match = builder.fresh("bool")
        builder.emit(
            body,
            "GetElementPtr",
            [SSAValue(descriptor.live_flags_value_id, dtype="bool"), index],
            live_address,
        )
        builder.emit(body, "Load", [live_address], live)
        builder.emit(body, "LAnd", [matches, live], active_match)
        matches = active_match
    builder.cond(body, matches, found, latch)
    builder.emit(latch, "Add", [index, one], next_index)
    builder.branch(latch, header)
    true_value = builder.const(found, True, dtype="bool")
    builder.branch(found, result_block)
    false_value = builder.const(absent, False, dtype="bool")
    builder.branch(absent, result_block)
    result = builder.fresh("bool")
    builder.emit(
        result_block,
        "Phi",
        [true_value, false_value],
        result,
        attributes={"incoming_blocks": (found.name, absent.name)},
    )
    builder.emit(result_block, "Ret", [result])
    name = function_name or f"ssa_sequence_{descriptor.sequence_id}_contains"
    function = Function(
        name,
        [*storage, *queries],
        builder.blocks,
        metadata={
            "ssa_sequence_operation": "contains",
            "sequence_id": descriptor.sequence_id,
            "key_columns": descriptor.key_columns,
            "membership_columns": membership_columns,
            "sequence_array_argument_ids": tuple((
                *map(int, descriptor.column_value_ids),
                int(descriptor.length_address_id),
                *((int(descriptor.live_flags_value_id),) if descriptor.live_flags_value_id is not None else ()),
            )),
            "named_outputs": (("contains", int(result.id)),),
        },
    )
    return SSASequenceLowering((function,))


def lower_sequence_live_length(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Count the live rows of a table that retains deleted rows.

    ``del d[k]`` clears a live flag and leaves the physical row resident, so
    the length cell counts rows ever written, not Python's ``len(d)``.
    """

    if descriptor.live_flags_value_id is None:
        raise ValueError("live length requires caller-visible live flags")
    storage = _storage_values(descriptor)
    builder = _Builder(_first_fresh_identity(
        descriptor, values=storage, minimum=first_value_id
    ))
    entry = builder.block("entry")
    header = builder.block("count_header")
    body = builder.block("count_body")
    result_block = builder.block("result")
    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")
    length_slot = builder.fresh("ptr")
    length = builder.fresh("int64")
    builder.emit(
        entry,
        "GetElementPtr",
        [SSAValue(descriptor.length_address_id, dtype="int64", shape=(1,)), zero],
        length_slot,
    )
    builder.emit(entry, "Load", [length_slot], length)
    builder.branch(entry, header)
    index = builder.fresh("int64")
    count = builder.fresh("int64")
    next_index = builder.fresh("int64")
    next_count = builder.fresh("int64")
    builder.emit(
        header, "Phi", [zero, next_index], index,
        attributes={"incoming_blocks": (entry.name, body.name)},
    )
    builder.emit(
        header, "Phi", [zero, next_count], count,
        attributes={"incoming_blocks": (entry.name, body.name)},
    )
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, length], continues)
    builder.cond(header, continues, body, result_block)
    live_address = builder.fresh("ptr")
    live = builder.fresh("bool")
    live_count = builder.fresh("int64")
    builder.emit(
        body,
        "GetElementPtr",
        [SSAValue(descriptor.live_flags_value_id, dtype="bool"), index],
        live_address,
    )
    builder.emit(body, "Load", [live_address], live)
    builder.emit(body, "Select", [live, one, zero], live_count)
    builder.emit(body, "Add", [count, live_count], next_count)
    builder.emit(body, "Add", [index, one], next_index)
    builder.branch(body, header)
    builder.emit(result_block, "Ret", [count])
    name = function_name or f"ssa_sequence_{descriptor.sequence_id}_live_length"
    function = Function(
        name,
        list(storage),
        builder.blocks,
        metadata={
            "ssa_sequence_operation": "live_length",
            "sequence_id": descriptor.sequence_id,
            "sequence_array_argument_ids": tuple((
                *map(int, descriptor.column_value_ids),
                int(descriptor.length_address_id),
                int(descriptor.live_flags_value_id),
            )),
            "named_outputs": (("length", int(count.id)),),
        },
    )
    return SSASequenceLowering((function,))


def lower_sequence_equal(
    left: SSASequenceDescriptor,
    right: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Compare two resident sequences by complete logical contents.

    This is the native, collision-free equality used by resident strings: the
    lengths must agree and every stored glyph/component in every column must
    agree in order.  A content hash may be a useful accelerator, but it is not
    evidence of equality and therefore is deliberately absent from the truth
    path here.

    Tombstoned keyed tables need a logical-row walk rather than this dense-row
    operation.  They are rejected instead of accidentally comparing physical
    arena debris.  Strings and formatted glyph sequences are dense duplicate
    sequences and take this path directly.
    """

    if left.live_flags_value_id is not None or right.live_flags_value_id is not None:
        raise ValueError("sequence equality requires dense resident sequences")
    left_dtypes = tuple(left.column_dtypes) or (
        "unknown",
    ) * len(left.column_value_ids)
    right_dtypes = tuple(right.column_dtypes) or (
        "unknown",
    ) * len(right.column_value_ids)
    left_shapes = tuple(left.column_shapes) or (
        (),
    ) * len(left.column_value_ids)
    right_shapes = tuple(right.column_shapes) or (
        (),
    ) * len(right.column_value_ids)
    if not left.column_value_ids or (
        len(left.column_value_ids) != len(right.column_value_ids)
        or left_dtypes != right_dtypes
        or left_shapes != right_shapes
    ):
        raise ValueError(
            "sequence equality requires identical non-empty row schemas"
        )

    storage = tuple({
        value.id: value
        for value in (*_storage_values(left), *_storage_values(right))
    }.values())
    builder = _Builder(_first_fresh_identity(
        left, right, values=storage, minimum=first_value_id,
    ))
    entry = builder.block("entry")
    header = builder.block("equal_header")
    body = builder.block("equal_body")
    latch = builder.block("equal_latch")
    equal = builder.block("equal")
    unequal = builder.block("unequal")
    result_block = builder.block("result")
    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")

    def load_length(descriptor: SSASequenceDescriptor):
        slot = builder.fresh("ptr")
        length = builder.fresh("int64")
        builder.emit(entry, "GetElementPtr", [SSAValue(
            descriptor.length_address_id, dtype="int64", shape=(1,),
        ), zero], slot)
        builder.emit(entry, "Load", [slot], length)
        return length

    left_length = load_length(left)
    right_length = load_length(right)
    same_length = builder.fresh("bool")
    builder.emit(entry, "Eq", [left_length, right_length], same_length)
    builder.cond(entry, same_length, header, unequal)

    index = builder.fresh("int64")
    next_index = builder.fresh("int64")
    builder.emit(
        header, "Phi", [zero, next_index], index,
        attributes={"incoming_blocks": (entry.name, latch.name)},
    )
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, left_length], continues)
    builder.cond(header, continues, body, equal)

    row_matches = None
    for left_column, right_column, dtype in zip(
        left.column_value_ids, right.column_value_ids, left_dtypes,
    ):
        left_address = builder.fresh("ptr")
        right_address = builder.fresh("ptr")
        left_value = builder.fresh(dtype)
        right_value = builder.fresh(dtype)
        component_match = builder.fresh("bool")
        builder.emit(body, "GetElementPtr", [
            SSAValue(left_column, dtype=dtype), index,
        ], left_address)
        builder.emit(body, "GetElementPtr", [
            SSAValue(right_column, dtype=dtype), index,
        ], right_address)
        builder.emit(body, "Load", [left_address], left_value)
        builder.emit(body, "Load", [right_address], right_value)
        builder.emit(body, "Eq", [left_value, right_value], component_match)
        if row_matches is None:
            row_matches = component_match
        else:
            combined = builder.fresh("bool")
            builder.emit(body, "LAnd", [row_matches, component_match], combined)
            row_matches = combined
    assert row_matches is not None
    builder.cond(body, row_matches, latch, unequal)
    builder.emit(latch, "Add", [index, one], next_index)
    builder.branch(latch, header)

    true_value = builder.const(equal, True, dtype="bool")
    builder.branch(equal, result_block)
    false_value = builder.const(unequal, False, dtype="bool")
    builder.branch(unequal, result_block)
    result = builder.fresh("bool")
    builder.emit(
        result_block, "Phi", [true_value, false_value], result,
        attributes={"incoming_blocks": (equal.name, unequal.name)},
    )
    builder.emit(result_block, "Ret", [result])
    name = function_name or (
        f"ssa_sequence_{left.sequence_id}_equal_{right.sequence_id}"
    )
    function = Function(name, list(storage), builder.blocks, metadata={
        "ssa_sequence_operation": "equal",
        "left_sequence_id": int(left.sequence_id),
        "right_sequence_id": int(right.sequence_id),
        "sequence_array_argument_ids": tuple(value.id for value in storage),
        "named_outputs": (("equal", int(result.id)),),
    })
    return SSASequenceLowering((function,))


def lower_table_lookup(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    default_parameter: bool = False,
    required: bool = False,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Build key lookup and return every physical value column.

    ``default_parameter`` appends one trailing argument of the value column's
    dtype and returns it from the absent branch -- the ``d.get(key, default)``
    contract -- instead of the absent branch's literal zero.
    """

    value_columns = tuple(
        column for column in range(len(descriptor.column_value_ids))
        if column not in descriptor.key_columns
    )
    if not descriptor.key_columns or not value_columns:
        raise ValueError("table lookup requires key and value columns")
    if default_parameter and len(value_columns) != 1:
        raise ValueError(
            "table lookup defaults require exactly one value column"
        )
    if descriptor.status_address_id is None:
        raise ValueError("table lookup requires a caller-visible status cell")
    storage = _storage_values(descriptor)
    status_arena = SSAValue(descriptor.status_address_id, dtype="int", shape=(1,))
    builder = _Builder(_first_fresh_identity(
        descriptor, values=(*storage, status_arena), minimum=first_value_id
    ))
    key_dtypes = tuple(
        descriptor.column_dtypes[column] for column in descriptor.key_columns
    )
    value_dtypes = tuple(
        descriptor.column_dtypes[column] for column in value_columns
    )
    queries = tuple(builder.fresh(dtype) for dtype in key_dtypes)
    entry = builder.block("entry")
    header = builder.block("lookup_header")
    body = builder.block("lookup_body")
    latch = builder.block("lookup_latch")
    found = builder.block("found")
    absent = builder.block("absent")
    result_block = builder.block("result")
    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")
    length_slot = builder.fresh("ptr")
    length = builder.fresh("int64")
    status_slot = builder.fresh("ptr")
    builder.emit(entry, "GetElementPtr", [
        SSAValue(descriptor.length_address_id, dtype="int64", shape=(1,)), zero
    ], length_slot)
    builder.emit(entry, "Load", [length_slot], length)
    builder.emit(entry, "GetElementPtr", [status_arena, zero], status_slot)
    key_match = _key_row_match(builder, descriptor, key_dtypes, queries)
    found_rows: list[tuple[SSAValue, BasicBlock]] = []
    scan = entry
    if hash_index_carrier(descriptor) is not None:
        capacity = SSAValue(descriptor.capacity_value_id, dtype="int64")
        indexed, scan = _hash_index_valid(
            builder, entry, descriptor, capacity, length, label="lookup",
        )
        probed_row, probe_body, _probe_header = _hash_index_probe(
            builder, indexed, descriptor, capacity, queries, label="lookup",
            found=found, absent=absent, key_match=key_match,
        )
        found_rows.append((probed_row, probe_body))
    builder.branch(scan, header)
    index = builder.fresh("int64")
    next_index = builder.fresh("int64")
    builder.emit(header, "Phi", [zero, next_index], index, attributes={
        "incoming_blocks": (scan.name, latch.name)
    })
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, length], continues)
    builder.cond(header, continues, body, absent)
    builder.cond(body, key_match(body, index), found, latch)
    builder.emit(latch, "Add", [index, one], next_index)
    builder.branch(latch, header)
    found_rows.append((index, body))
    if len(found_rows) > 1:
        row = builder.fresh("int64")
        builder.emit(found, "Phi", [value for value, _ in found_rows], row, attributes={
            "incoming_blocks": tuple(block.name for _, block in found_rows),
        })
    else:
        row = index
    found_values = []
    for value_column, value_dtype in zip(value_columns, value_dtypes):
        value_address = builder.fresh("ptr")
        found_value = builder.fresh(value_dtype)
        builder.emit(found, "GetElementPtr", [
            SSAValue(
                descriptor.column_value_ids[value_column], dtype=value_dtype,
            ), row,
        ], value_address)
        builder.emit(found, "Load", [value_address], found_value)
        found_values.append(found_value)
    builder.emit(found, "Store", [one, status_slot])
    builder.branch(found, result_block)
    if default_parameter:
        missing_values = (builder.fresh(value_dtypes[0]),)
        default_argument = missing_values[0]
    else:
        missing_values = tuple(
            builder.fresh(value_dtype) for value_dtype in value_dtypes
        )
        default_argument = None
        for missing_value in missing_values:
            builder.emit(
                absent, "Const", [], missing_value, attributes={"value": 0},
            )
    builder.emit(absent, "Store", [zero, status_slot])
    if required:
        builder.emit(absent, "Call", [], attributes={
            "callee": "turing_validation_error",
            "error_code": 84,
            "source_linked": True,
            "ssa_sequence_operation": "lookup",
            "failure": "required_key_absent",
        })
    builder.branch(absent, result_block)
    results = []
    for found_value, missing_value, value_dtype in zip(
        found_values, missing_values, value_dtypes,
    ):
        result = builder.fresh(value_dtype)
        builder.emit(
            result_block, "Phi", [found_value, missing_value], result,
            attributes={"incoming_blocks": (found.name, absent.name)},
        )
        results.append(result)
    builder.emit(result_block, "Ret", list(results))
    name = function_name or f"ssa_sequence_{descriptor.sequence_id}_lookup"
    function = Function(
        name,
        [
            *storage, status_arena, *queries,
            *((default_argument,) if default_argument is not None else ()),
        ],
        builder.blocks,
        metadata={
            "ssa_sequence_operation": "lookup",
            "sequence_id": descriptor.sequence_id,
            "key_columns": descriptor.key_columns,
            "sequence_array_argument_ids": tuple((
                *map(int, descriptor.column_value_ids),
                int(descriptor.length_address_id),
                int(descriptor.status_address_id),
                *((int(descriptor.live_flags_value_id),) if descriptor.live_flags_value_id is not None else ()),
            )),
            "named_outputs": tuple(
                (
                    "value" if len(results) == 1 else f"value_{index}",
                    int(result.id),
                )
                for index, result in enumerate(results)
            ),
            "required": bool(required),
        },
    )
    return SSASequenceLowering((function,))


def lower_glyph_token_copy(
    destination: SSASequenceDescriptor,
    vocabulary: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
) -> SSASequenceLowering:
    """Copy one StringTable token's glyph row through keyed resident lookup.

    ``vocabulary`` is the incumbent three-column table keyed by
    ``(string_token, ordinal)`` with one int64 value column.  Ordinal ``-1``
    stores the exact glyph count; ordinals ``0..count-1`` store glyph tokens.
    Unknown tokens return status 0, fixed-capacity exhaustion returns 2, and
    a complete copy returns 1.  The helper is ordinary CFG/call SSA composed
    solely from the existing table-lookup and sequence-append operations.
    """

    if (
        len(destination.column_value_ids) != 1
        or destination.key_columns
        or tuple(destination.column_dtypes) != ("int64",)
    ):
        raise ValueError("glyph destination must be one int64 sequence column")
    if (
        len(vocabulary.column_value_ids) != 3
        or tuple(vocabulary.key_columns) != (0, 1)
        or tuple(vocabulary.column_dtypes) != ("int64", "int64", "int64")
        or vocabulary.status_address_id is None
    ):
        raise ValueError(
            "glyph vocabulary must be (token, ordinal)->glyph int64 table"
        )
    lookup = lower_table_lookup(vocabulary).functions[0]
    append = lower_sequence_append(destination).functions[0]
    destination_storage = _storage_values(destination)
    vocabulary_storage = _storage_values(vocabulary)
    vocabulary_status = SSAValue(
        int(vocabulary.status_address_id), dtype="int", shape=(1,),
    )
    arguments = tuple({
        int(value.id): value
        for value in (
            *destination_storage,
            *vocabulary_storage,
            vocabulary_status,
        )
    }.values())
    builder = _Builder(_first_fresh_identity(
        destination, vocabulary,
        values=(*arguments,),
    ))
    token = builder.fresh("int64")
    entry = builder.block("entry")
    known = builder.block("known")
    header = builder.block("glyph_header")
    body = builder.block("glyph_body")
    append_block = builder.block("glyph_append")
    latch = builder.block("glyph_latch")
    complete = builder.block("complete")
    unknown = builder.block("unknown_token")
    exhausted = builder.block("capacity_exhausted")
    result_block = builder.block("result")
    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")
    two = builder.const(entry, 2, dtype="int")
    minus_one = builder.const(entry, -1, dtype="int64")

    destination_length = next(
        value for value in destination_storage
        if int(value.id) == int(destination.length_address_id)
    )
    destination_length_slot = builder.fresh("ptr")
    builder.emit(entry, "GetElementPtr", [
        destination_length, zero,
    ], destination_length_slot)
    builder.emit(entry, "Store", [zero, destination_length_slot])

    lookup_storage = [
        *vocabulary_storage, vocabulary_status,
    ]
    length = builder.fresh("int64")
    builder.emit(entry, "Call", [
        *lookup_storage, token, minus_one,
    ], length, attributes={
        "callee": lookup.name,
        "source_linked": True,
        "ssa_sequence_operation": "lookup",
        "sequence_id": int(vocabulary.sequence_id),
    })
    status_slot = builder.fresh("ptr")
    lookup_status = builder.fresh("int")
    builder.emit(entry, "GetElementPtr", [
        vocabulary_status, zero,
    ], status_slot)
    builder.emit(entry, "Load", [status_slot], lookup_status)
    found = builder.fresh("bool")
    builder.emit(entry, "Eq", [lookup_status, one], found)
    builder.cond(entry, found, known, unknown)
    builder.branch(known, header)

    ordinal = builder.fresh("int64")
    next_ordinal = builder.fresh("int64")
    builder.emit(header, "Phi", [zero, next_ordinal], ordinal, attributes={
        "incoming_blocks": (known.name, latch.name),
    })
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [ordinal, length], continues)
    builder.cond(header, continues, body, complete)
    glyph = builder.fresh("int64")
    builder.emit(body, "Call", [
        *lookup_storage, token, ordinal,
    ], glyph, attributes={
        "callee": lookup.name,
        "source_linked": True,
        "ssa_sequence_operation": "lookup",
        "sequence_id": int(vocabulary.sequence_id),
    })
    glyph_status = builder.fresh("int")
    builder.emit(body, "Load", [status_slot], glyph_status)
    glyph_found = builder.fresh("bool")
    builder.emit(body, "Eq", [glyph_status, one], glyph_found)
    builder.cond(body, glyph_found, append_block, unknown)
    append_status = builder.fresh("int")
    builder.emit(append_block, "Call", [
        *destination_storage, glyph,
    ], append_status, attributes={
        "callee": append.name,
        "source_linked": True,
        "ssa_sequence_operation": "append",
        "sequence_id": int(destination.sequence_id),
    })
    full = builder.fresh("bool")
    builder.emit(append_block, "Eq", [append_status, two], full)
    builder.cond(append_block, full, exhausted, latch)
    builder.emit(latch, "Add", [ordinal, one], next_ordinal)
    builder.branch(latch, header)
    success_value = builder.const(complete, 1)
    builder.branch(complete, result_block)
    builder.emit(unknown, "Call", [], attributes={
        "callee": "turing_validation_error",
        "error_code": 83,
        "source_linked": True,
        "ssa_sequence_operation": "glyph_token_lookup_copy",
        "failure": "unknown_string_token",
    })
    unknown_value = builder.const(unknown, 0)
    builder.branch(unknown, result_block)
    exhausted_value = builder.const(exhausted, 2)
    builder.branch(exhausted, result_block)
    result = builder.fresh("int")
    builder.emit(result_block, "Phi", [
        success_value, unknown_value, exhausted_value,
    ], result, attributes={
        "incoming_blocks": (
            complete.name, unknown.name, exhausted.name,
        ),
    })
    builder.emit(result_block, "Ret", [result])
    name = function_name or (
        f"ssa_sequence_{destination.sequence_id}_glyph_token_copy_"
        f"{vocabulary.sequence_id}"
    )
    function = Function(name, [*arguments, token], builder.blocks, metadata={
        "ssa_sequence_operation": "glyph_token_lookup_copy",
        "sequence_id": int(destination.sequence_id),
        "vocabulary_sequence_id": int(vocabulary.sequence_id),
        "sequence_array_argument_ids": tuple(
            int(value.id) for value in arguments
        ),
        "named_outputs": (("status", int(result.id)),),
        "status_values": {
            "unknown_token": 0, "complete": 1,
            "capacity_exhausted": 2,
        },
    })
    # Dependency helpers precede the public composition entry, matching every
    # other sequence lowering consumed by ``_register_sequence_lowering``.
    return SSASequenceLowering((lookup, append, function))


def lower_table_store(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Build update-existing-or-insert for a fixed keyed row table."""

    unsupported = _unsupported_destination(descriptor, "table_store", allow_dynamic=True)
    if unsupported is not None:
        return unsupported
    if descriptor.capacity_policy is SSASequenceCapacityPolicy.DYNAMIC:
        return _lower_dynamic_sequence_mutation(descriptor, function_name=function_name,
            operation="table_store", first_value_id=first_value_id)
    value_columns = tuple(
        column for column in range(len(descriptor.column_value_ids))
        if column not in descriptor.key_columns
    )
    if not descriptor.key_columns or not value_columns:
        raise ValueError("table store requires key and value columns")
    if descriptor.status_address_id is None:
        raise ValueError("table store requires a caller-visible status cell")
    storage = _storage_values(descriptor)
    status_arena = SSAValue(descriptor.status_address_id, dtype="int", shape=(1,))
    builder = _Builder(_first_fresh_identity(
        descriptor, values=(*storage, status_arena), minimum=first_value_id
    ))
    key_dtypes = tuple(
        descriptor.column_dtypes[column] for column in descriptor.key_columns
    )
    value_dtypes = tuple(
        descriptor.column_dtypes[column] for column in value_columns
    )
    queries = tuple(builder.fresh(dtype) for dtype in key_dtypes)
    new_values = tuple(builder.fresh(dtype) for dtype in value_dtypes)
    entry = builder.block("entry")
    header = builder.block("store_header")
    body = builder.block("store_body")
    latch = builder.block("store_latch")
    update = builder.block("update")
    capacity_check = builder.block("capacity_check")
    insert = builder.block("insert")
    full = builder.block("capacity_exhausted")
    compact_header = builder.block("compact_header")
    compact_body = builder.block("compact_body")
    compact_copy = builder.block("compact_copy")
    compact_latch = builder.block("compact_latch")
    compact_done = builder.block("compact_done")
    insert_compacted = builder.block("insert_compacted")
    full_status = builder.block("capacity_exhausted_status")
    complete = builder.block("complete")
    # All sequence extent cells are declared int64. Keep their scan and
    # increment values int64 too; source call-type settlement otherwise
    # widens the latch Add but leaves an incompatible i32 header Phi.
    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")
    two = builder.const(entry, 2)
    three = builder.const(entry, 3)
    length_slot = builder.fresh("ptr")
    length = builder.fresh("int64")
    status_slot = builder.fresh("ptr")
    builder.emit(entry, "GetElementPtr", [
        SSAValue(descriptor.length_address_id, dtype="int64", shape=(1,)), zero
    ], length_slot)
    builder.emit(entry, "Load", [length_slot], length)
    builder.emit(entry, "GetElementPtr", [status_arena, zero], status_slot)
    key_match = _key_row_match(builder, descriptor, key_dtypes, queries)
    indexed_table = hash_index_carrier(descriptor) is not None
    capacity = SSAValue(descriptor.capacity_value_id, dtype="int64")
    found_rows: list[tuple[SSAValue, BasicBlock]] = []
    scan = entry
    if indexed_table:
        indexed, scan = _hash_index_valid(
            builder, entry, descriptor, capacity, length, label="store",
        )
        probed_row, probe_body, _probe_header = _hash_index_probe(
            builder, indexed, descriptor, capacity, queries, label="store",
            found=update, absent=capacity_check, key_match=key_match,
        )
        found_rows.append((probed_row, probe_body))
    builder.branch(scan, header)
    index = builder.fresh("int64")
    next_index = builder.fresh("int64")
    builder.emit(header, "Phi", [zero, next_index], index, attributes={
        "incoming_blocks": (scan.name, latch.name)
    })
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, length], continues)
    builder.cond(header, continues, body, capacity_check)
    builder.cond(body, key_match(body, index), update, latch)
    builder.emit(latch, "Add", [index, one], next_index)
    builder.branch(latch, header)
    found_rows.append((index, body))
    if len(found_rows) > 1:
        row = builder.fresh("int64")
        builder.emit(update, "Phi", [value for value, _ in found_rows], row, attributes={
            "incoming_blocks": tuple(block.name for _, block in found_rows),
        })
    else:
        row = index
    for value_column, value_dtype, new_value in zip(
        value_columns, value_dtypes, new_values, strict=True,
    ):
        update_address = builder.fresh("ptr")
        builder.emit(update, "GetElementPtr", [
            SSAValue(
                descriptor.column_value_ids[value_column],
                dtype=value_dtype,
            ),
            row,
        ], update_address)
        builder.emit(update, "Store", [new_value, update_address])
    builder.emit(update, "Store", [three, status_slot])
    builder.branch(update, complete)
    has_capacity = builder.fresh("bool")
    builder.emit(capacity_check, "Lt", [
        length, SSAValue(descriptor.capacity_value_id, dtype="int64")
    ], has_capacity)
    builder.cond(capacity_check, has_capacity, insert, full)
    inserted_columns = tuple(
        (descriptor.column_value_ids[column], dtype, query)
        for column, dtype, query in zip(
            descriptor.key_columns, key_dtypes, queries
        )
    ) + tuple(
        (descriptor.column_value_ids[column], dtype, value)
        for column, dtype, value in zip(
            value_columns, value_dtypes, new_values, strict=True,
        )
    )

    def emit_insert(block, slot):
        # ``slot < C`` on both insert paths, so the carrier arena exists:
        # an index that covered every row before this one covers this too.
        covered = None
        if indexed_table:
            marker_address, marker = builder.fresh("ptr"), builder.fresh("int64")
            covered = builder.fresh("bool")
            builder.emit(block, "GetElementPtr", [_hash_index_carrier_value(descriptor), capacity], marker_address)
            builder.emit(block, "Load", [marker_address], marker)
            builder.emit(block, "Eq", [marker, slot], covered)
        for column_id, dtype, value in inserted_columns:
            address = builder.fresh("ptr")
            builder.emit(block, "GetElementPtr", [
                SSAValue(column_id, dtype=dtype), slot
            ], address)
            builder.emit(block, "Store", [value, address])
        if descriptor.live_flags_value_id is not None:
            live_address = builder.fresh("ptr")
            builder.emit(block, "GetElementPtr", [
                SSAValue(descriptor.live_flags_value_id, dtype="bool"), slot
            ], live_address)
            builder.emit(block, "Store", [one, live_address])
        next_length = builder.fresh("int64")
        builder.emit(block, "Add", [slot, one], next_length)
        builder.emit(block, "Store", [next_length, length_slot])
        builder.emit(block, "Store", [one, status_slot])
        if covered is None:
            builder.branch(block, complete)
            return
        # The length store above is guarded (``M = -1``); the covered path
        # then indexes the new row and restores ``M`` to the new length.
        maintain = builder.block(f"{block.name}_index_append")
        builder.cond(block, covered, maintain, complete)
        placed = _hash_index_place(
            builder, maintain, descriptor, capacity, queries, slot,
            label=f"{block.name}_index",
        )
        marker_address = builder.fresh("ptr")
        builder.emit(placed, "GetElementPtr", [_hash_index_carrier_value(descriptor), capacity], marker_address)
        builder.emit(placed, "Store", [next_length, marker_address])
        builder.branch(placed, complete)

    emit_insert(insert, length)
    # Program: ``d = {2: 5}; for i in range(n): d[1] = i; del d[1]`` (corpus
    # c713).  Wrong record: a delete clears a live flag and keeps the row,
    # and a store only appends, so every pass spent a fresh row against a
    # bound that counts keys; the table reported "full" (code 75) with one
    # live key.  Rule: a full table whose rows include dead ones closes up
    # its live rows in order (row order is insertion order, so a re-stored
    # key still lands last) and stores into the room that frees; with no
    # dead row it is full.
    if descriptor.live_flags_value_id is None:
        builder.branch(full, full_status)
    else:
        builder.branch(full, compact_header)
        row_index = builder.fresh("int64")
        next_row = builder.fresh("int64")
        write_index = builder.fresh("int64")
        next_write = builder.fresh("int64")
        builder.emit(compact_header, "Phi", [zero, next_row], row_index, attributes={
            "incoming_blocks": (full.name, compact_latch.name)
        })
        builder.emit(compact_header, "Phi", [zero, next_write], write_index, attributes={
            "incoming_blocks": (full.name, compact_latch.name)
        })
        scanning = builder.fresh("bool")
        builder.emit(compact_header, "Lt", [row_index, length], scanning)
        builder.cond(compact_header, scanning, compact_body, compact_done)
        row_live_address = builder.fresh("ptr")
        row_live = builder.fresh("bool")
        builder.emit(compact_body, "GetElementPtr", [
            SSAValue(descriptor.live_flags_value_id, dtype="bool"), row_index
        ], row_live_address)
        builder.emit(compact_body, "Load", [row_live_address], row_live)
        builder.cond(compact_body, row_live, compact_copy, compact_latch)
        for column in range(len(descriptor.column_value_ids)):
            column_dtype = descriptor.column_dtypes[column]
            source_address = builder.fresh("ptr")
            target_address = builder.fresh("ptr")
            moved = builder.fresh(column_dtype)
            column_value = SSAValue(
                descriptor.column_value_ids[column], dtype=column_dtype,
            )
            builder.emit(compact_copy, "GetElementPtr", [column_value, row_index], source_address)
            builder.emit(compact_copy, "Load", [source_address], moved)
            builder.emit(compact_copy, "GetElementPtr", [column_value, write_index], target_address)
            builder.emit(compact_copy, "Store", [moved, target_address])
        moved_live_address = builder.fresh("ptr")
        builder.emit(compact_copy, "GetElementPtr", [
            SSAValue(descriptor.live_flags_value_id, dtype="bool"), write_index
        ], moved_live_address)
        builder.emit(compact_copy, "Store", [one, moved_live_address])
        copied_write = builder.fresh("int64")
        builder.emit(compact_copy, "Add", [write_index, one], copied_write)
        builder.branch(compact_copy, compact_latch)
        builder.emit(compact_latch, "Phi", [write_index, copied_write], next_write, attributes={
            "incoming_blocks": (compact_body.name, compact_copy.name)
        })
        builder.emit(compact_latch, "Add", [row_index, one], next_row)
        builder.branch(compact_latch, compact_header)
        builder.emit(compact_done, "Store", [write_index, length_slot])
        compacted_room = builder.fresh("bool")
        builder.emit(compact_done, "Lt", [
            write_index, SSAValue(descriptor.capacity_value_id, dtype="int64")
        ], compacted_room)
        builder.cond(compact_done, compacted_room, insert_compacted, full_status)
        emit_insert(insert_compacted, write_index)
    builder.emit(full_status, "Store", [two, status_slot])
    builder.branch(full_status, complete)
    builder.emit(complete, "Ret", [])
    name = function_name or f"ssa_sequence_{descriptor.sequence_id}_store"
    function = Function(
        name,
        [*storage, status_arena, *queries, *new_values],
        builder.blocks,
        metadata={
            "ssa_sequence_operation": "table_store",
            "sequence_id": descriptor.sequence_id,
            "key_columns": descriptor.key_columns,
            "status_values": {"inserted": 1, "capacity_exhausted": 2, "updated": 3},
            "sequence_array_argument_ids": tuple((
                *map(int, descriptor.column_value_ids),
                int(descriptor.length_address_id),
                int(descriptor.status_address_id),
                *((int(descriptor.live_flags_value_id),) if descriptor.live_flags_value_id is not None else ()),
            )),
        },
    )
    return SSASequenceLowering((function,))


def lower_table_delete(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Build key deletion by clearing the table's caller-owned live flag.

    Physical rows remain resident so outstanding row indices are never shifted.
    Status is 4 when a live row was deleted and 0 when the key was absent.
    """

    unsupported = _unsupported_destination(descriptor, "table_delete")
    if unsupported is not None:
        return unsupported
    value_columns = tuple(
        column for column in range(len(descriptor.column_value_ids))
        if column not in descriptor.key_columns
    )
    if not descriptor.key_columns or len(value_columns) != 1:
        raise ValueError("table delete requires key columns and one value column")
    if descriptor.status_address_id is None:
        raise ValueError("table delete requires a caller-visible status cell")
    if descriptor.live_flags_value_id is None:
        raise ValueError("table delete requires caller-visible live flags")
    storage = _storage_values(descriptor)
    status_arena = SSAValue(descriptor.status_address_id, dtype="int", shape=(1,))
    builder = _Builder(_first_fresh_identity(
        descriptor, values=(*storage, status_arena), minimum=first_value_id
    ))
    key_dtypes = tuple(
        descriptor.column_dtypes[column] for column in descriptor.key_columns
    )
    queries = tuple(builder.fresh(dtype) for dtype in key_dtypes)
    entry = builder.block("entry")
    header = builder.block("delete_header")
    body = builder.block("delete_body")
    latch = builder.block("delete_latch")
    remove = builder.block("delete_found")
    absent = builder.block("delete_absent")
    complete = builder.block("complete")
    zero = builder.const(entry, 0)
    deleted = builder.const(entry, 4)
    # Program: ``del d[2]; len(d)`` (c338, c406).  Wrong record: the row
    # counter was an ``int`` phi whose latch increment the type linker
    # settles to the int64 length it counts against, so LLVM refused the
    # phi.  Rule: a row counter over an int64 length is int64, as in every
    # other scan.
    row_zero = builder.const(entry, 0, dtype="int64")
    row_one = builder.const(entry, 1, dtype="int64")
    length_slot = builder.fresh("ptr")
    length = builder.fresh("int64")
    status_slot = builder.fresh("ptr")
    builder.emit(entry, "GetElementPtr", [
        SSAValue(descriptor.length_address_id, dtype="int64", shape=(1,)), zero
    ], length_slot)
    builder.emit(entry, "Load", [length_slot], length)
    builder.emit(entry, "GetElementPtr", [status_arena, zero], status_slot)
    builder.branch(entry, header)
    index = builder.fresh("int64")
    next_index = builder.fresh("int64")
    builder.emit(header, "Phi", [row_zero, next_index], index, attributes={
        "incoming_blocks": (entry.name, latch.name)
    })
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, length], continues)
    builder.cond(header, continues, body, absent)
    matches = None
    live_address = builder.fresh("ptr")
    live = builder.fresh("bool")
    active_match = builder.fresh("bool")
    for key_column, key_dtype, query in zip(
        descriptor.key_columns, key_dtypes, queries
    ):
        key_address = builder.fresh("ptr")
        existing_key = builder.fresh(key_dtype)
        component_match = builder.fresh("bool")
        builder.emit(body, "GetElementPtr", [
            SSAValue(descriptor.column_value_ids[key_column], dtype=key_dtype), index
        ], key_address)
        builder.emit(body, "Load", [key_address], existing_key)
        builder.emit(body, "Eq", [existing_key, query], component_match)
        if matches is None:
            matches = component_match
        else:
            combined = builder.fresh("bool")
            builder.emit(body, "LAnd", [matches, component_match], combined)
            matches = combined
    assert matches is not None
    builder.emit(body, "GetElementPtr", [
        SSAValue(descriptor.live_flags_value_id, dtype="bool"), index
    ], live_address)
    builder.emit(body, "Load", [live_address], live)
    builder.emit(body, "LAnd", [matches, live], active_match)
    builder.cond(body, active_match, remove, latch)
    builder.emit(latch, "Add", [index, row_one], next_index)
    builder.branch(latch, header)
    builder.emit(remove, "Store", [zero, live_address])
    builder.emit(remove, "Store", [deleted, status_slot])
    builder.branch(remove, complete)
    builder.emit(absent, "Store", [zero, status_slot])
    builder.branch(absent, complete)
    builder.emit(complete, "Ret", [])
    name = function_name or f"ssa_sequence_{descriptor.sequence_id}_delete"
    function = Function(
        name,
        [*storage, status_arena, *queries],
        builder.blocks,
        metadata={
            "ssa_sequence_operation": "table_delete",
            "sequence_id": descriptor.sequence_id,
            "key_columns": descriptor.key_columns,
            "status_values": {"missing": 0, "deleted": 4},
            "sequence_array_argument_ids": tuple((
                *map(int, descriptor.column_value_ids),
                int(descriptor.length_address_id),
                int(descriptor.status_address_id),
                int(descriptor.live_flags_value_id),
            )),
        },
    )
    return SSASequenceLowering((function,))


def lower_table_delete_first(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Delete the first live row, matching ``del table[next(iter(table))]``."""

    unsupported = _unsupported_destination(descriptor, "table_delete_first")
    if unsupported is not None:
        return unsupported
    if descriptor.status_address_id is None or descriptor.live_flags_value_id is None:
        raise ValueError("first-row deletion requires status and live arenas")
    storage = _storage_values(descriptor)
    status_arena = SSAValue(descriptor.status_address_id, dtype="int", shape=(1,))
    builder = _Builder(_first_fresh_identity(
        descriptor, values=(*storage, status_arena), minimum=first_value_id
    ))
    entry = builder.block("entry")
    header = builder.block("delete_header")
    body = builder.block("delete_body")
    latch = builder.block("delete_latch")
    remove = builder.block("delete_found")
    absent = builder.block("delete_absent")
    complete = builder.block("complete")
    zero = builder.const(entry, 0)
    deleted = builder.const(entry, 4)
    # Program: ``del d[2]; len(d)`` (c338, c406).  Wrong record: the row
    # counter was an ``int`` phi whose latch increment the type linker
    # settles to the int64 length it counts against, so LLVM refused the
    # phi.  Rule: a row counter over an int64 length is int64, as in every
    # other scan.
    row_zero = builder.const(entry, 0, dtype="int64")
    row_one = builder.const(entry, 1, dtype="int64")
    length_slot = builder.fresh("ptr")
    length = builder.fresh("int64")
    status_slot = builder.fresh("ptr")
    builder.emit(entry, "GetElementPtr", [SSAValue(
        descriptor.length_address_id, dtype="int64", shape=(1,)
    ), zero], length_slot)
    builder.emit(entry, "Load", [length_slot], length)
    builder.emit(entry, "GetElementPtr", [status_arena, zero], status_slot)
    builder.branch(entry, header)
    index = builder.fresh("int64")
    next_index = builder.fresh("int64")
    builder.emit(header, "Phi", [row_zero, next_index], index, attributes={
        "incoming_blocks": (entry.name, latch.name)
    })
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, length], continues)
    builder.cond(header, continues, body, absent)
    live_address = builder.fresh("ptr")
    live = builder.fresh("bool")
    builder.emit(body, "GetElementPtr", [SSAValue(
        descriptor.live_flags_value_id, dtype="bool"
    ), index], live_address)
    builder.emit(body, "Load", [live_address], live)
    builder.cond(body, live, remove, latch)
    builder.emit(latch, "Add", [index, row_one], next_index)
    builder.branch(latch, header)
    builder.emit(remove, "Store", [zero, live_address])
    builder.emit(remove, "Store", [deleted, status_slot])
    builder.branch(remove, complete)
    builder.emit(absent, "Store", [zero, status_slot])
    builder.branch(absent, complete)
    builder.emit(complete, "Ret", [])
    function = Function(
        function_name or f"ssa_sequence_{descriptor.sequence_id}_delete_first",
        [*storage, status_arena],
        builder.blocks,
        metadata={
            "ssa_sequence_operation": "table_delete_first",
            "sequence_id": descriptor.sequence_id,
            "status_values": {"missing": 0, "deleted": 4},
            "sequence_array_argument_ids": (
                *descriptor.column_value_ids,
                descriptor.length_address_id,
                descriptor.status_address_id,
                descriptor.live_flags_value_id,
            ),
        },
    )
    return SSASequenceLowering((function,))


def lower_child_table_delete(
    pool: SSAChildTablePoolDescriptor,
    *,
    function_name: str,
    first_value_id: int | None = None,
) -> SSASequenceLowering:
    """Clear one key in the child slice selected by an explicit handle."""

    if pool.live_flags_value_id is None or pool.status_value_id is None:
        raise ValueError("child table deletion requires live/status arenas")
    storage_ids = (
        *pool.column_value_ids,
        pool.length_value_id,
        pool.capacity_value_id,
        pool.row_stride_value_id,
        pool.status_value_id,
        pool.live_flags_value_id,
    )
    storage = [SSAValue(int(value_id)) for value_id in storage_ids]
    builder = _Builder(_first_fresh_identity(
        pools=(pool,), minimum=first_value_id
    ))
    handle = builder.fresh("int")
    query = builder.fresh(pool.column_dtypes[0] if pool.column_dtypes else "unknown")
    entry = builder.block("entry")
    header = builder.block("delete_header")
    body = builder.block("delete_body")
    latch = builder.block("delete_latch")
    remove = builder.block("delete_found")
    absent = builder.block("delete_absent")
    complete = builder.block("complete")
    zero = builder.const(entry, 0)
    one = builder.const(entry, 1)
    deleted = builder.const(entry, 4)
    length_address = builder.fresh("ptr")
    length = builder.fresh("int")
    base = builder.fresh("int")
    status_address = builder.fresh("ptr")
    builder.emit(entry, "GetElementPtr", [
        SSAValue(pool.length_value_id, dtype="int"), handle
    ], length_address)
    builder.emit(entry, "Load", [length_address], length)
    builder.emit(entry, "Mul", [
        handle, SSAValue(pool.row_stride_value_id, dtype="int")
    ], base)
    builder.emit(entry, "GetElementPtr", [
        SSAValue(pool.status_value_id, dtype="int"), handle
    ], status_address)
    builder.branch(entry, header)
    index = builder.fresh("int")
    next_index = builder.fresh("int")
    builder.emit(header, "Phi", [zero, next_index], index, attributes={
        "incoming_blocks": (entry.name, latch.name)
    })
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, length], continues)
    builder.cond(header, continues, body, absent)
    offset = builder.fresh("int")
    key_address = builder.fresh("ptr")
    existing_key = builder.fresh(query.dtype)
    matches = builder.fresh("bool")
    live_address = builder.fresh("ptr")
    live = builder.fresh("bool")
    active = builder.fresh("bool")
    builder.emit(body, "Add", [base, index], offset)
    builder.emit(body, "GetElementPtr", [
        SSAValue(pool.column_value_ids[0], dtype=query.dtype), offset
    ], key_address)
    builder.emit(body, "Load", [key_address], existing_key)
    builder.emit(body, "Eq", [existing_key, query], matches)
    builder.emit(body, "GetElementPtr", [
        SSAValue(pool.live_flags_value_id, dtype="bool"), offset
    ], live_address)
    builder.emit(body, "Load", [live_address], live)
    builder.emit(body, "LAnd", [matches, live], active)
    builder.cond(body, active, remove, latch)
    builder.emit(latch, "Add", [index, one], next_index)
    builder.branch(latch, header)
    builder.emit(remove, "Store", [zero, live_address])
    builder.emit(remove, "Store", [deleted, status_address])
    builder.branch(remove, complete)
    builder.emit(absent, "Store", [zero, status_address])
    builder.branch(absent, complete)
    builder.emit(complete, "Ret", [])
    return SSASequenceLowering((Function(
        function_name,
        [*storage, handle, query],
        builder.blocks,
        metadata={
            "ssa_sequence_operation": "child_table_delete",
            "status_values": {"missing": 0, "deleted": 4},
            "sequence_array_argument_ids": tuple(map(int, storage_ids)),
        },
    ),))


def lower_sequence_replace(
    destination: SSASequenceDescriptor,
    source: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
    column_value_maps: dict[int, tuple[tuple[int, int], ...]] | None = None,
) -> SSASequenceLowering:
    """Copy a compatible resident arena, publishing its length last.

    Capacity failure leaves the destination intact. Identical storage is
    valid: source length is captured before writing, and rows retain their
    positions. This operation preserves keyed rows rather than reinserting
    them under a potentially different uniqueness policy.
    """
    unsupported = _unsupported_destination(destination, "replace", allow_dynamic=True)
    if unsupported is not None:
        return unsupported
    if (len(destination.column_value_ids) != len(source.column_value_ids)
            or destination.column_dtypes != source.column_dtypes
            or destination.key_columns != source.key_columns
            or (destination.live_flags_value_id is None) != (source.live_flags_value_id is None)):
        raise ValueError("sequence replacement requires matching row storage and key policy")
    if destination.child_table_pool is not None or source.child_table_pool is not None:
        raise ValueError("nested sequence replacement requires child ownership lowering")
    dst_columns = list(zip(destination.column_value_ids, destination.column_dtypes
                           or ("unknown",) * len(destination.column_value_ids)))
    src_columns = list(zip(source.column_value_ids, source.column_dtypes
                           or ("unknown",) * len(source.column_value_ids)))
    if destination.live_flags_value_id is not None:
        dst_columns.append((destination.live_flags_value_id, "bool"))
        src_columns.append((source.live_flags_value_id, "bool"))
    if any(dst == other for index, (dst, _) in enumerate(dst_columns)
           for other_index, (other, _) in enumerate(src_columns) if index != other_index):
        raise ValueError("sequence replacement cannot overlap different columns")
    storage = tuple({value.id: value for value in (
        *_storage_values(destination), *_storage_values(source)
    )}.values())
    builder = _Builder(_first_fresh_identity(destination, source, values=storage, minimum=first_value_id))
    entry = builder.block("entry")
    header = builder.block("replace_header")
    body = builder.block("replace_body")
    latch = builder.block("replace_latch")
    complete = builder.block("complete")
    exhausted = builder.block("capacity_exhausted")
    result = builder.block("result")
    # Replacement's length, index and backedge are int64 ABI cells. The
    # default int seed emitted i32, so each N4 replacement helper joined
    # i32 zero with an i64 backedge in an i64 Phi. Seed the same width at
    # its producer; do not reinterpret or repair that merge in a backend.
    zero = builder.const(entry, 0, dtype="int64")
    one = builder.const(entry, 1, dtype="int64")
    source_address = builder.fresh("ptr")
    length = builder.fresh("int64")
    builder.emit(entry, "GetElementPtr", [SSAValue(source.length_address_id, dtype="int64"), zero], source_address)
    builder.emit(entry, "Load", [source_address], length)
    nonnegative, source_fits, destination_fits = (builder.fresh("bool") for _ in range(3))
    builder.emit(entry, "Ge", [length, zero], nonnegative)
    builder.emit(entry, "Le", [length, SSAValue(source.capacity_value_id, dtype="int64")], source_fits)
    valid_source, fits = builder.fresh("bool"), builder.fresh("bool")
    builder.emit(entry, "And", [nonnegative, source_fits], valid_source)
    reserve_function = None
    shared_arena = tuple(map(int, destination.column_value_ids)) == tuple(
        map(int, source.column_value_ids)
    )
    if shared_arena:
        # The destination is a view of the source's own arena (an optional
        # glyph payload aliased to its present arm).  The rows are already
        # in place (the copy loop reads and writes the same addresses) and
        # replacement publishes the length.  Neither a reserve nor the
        # view's capacity cell applies: reserving would reallocate the
        # shared arena against the view's capacity and copy only its rows.
        builder.cond(entry, valid_source, header, exhausted)
    elif destination.capacity_policy is SSASequenceCapacityPolicy.DYNAMIC:
        # Program: ``detail = None; if c: detail = f'scope={t!r}'`` -- the
        # optional glyph payload is replaced from a DYNAMIC f-string arena.
        # Rule: a DYNAMIC destination grows through its own checked reserve
        # (the helper insert uses) to the source length before any row is
        # written; reserve status 2 is capacity exhaustion, and the
        # destination is left intact.
        reserve_function = lower_sequence_reserve(
            destination,
            function_name=(
                function_name
                or f"ssa_sequence_{destination.sequence_id}_replace_{source.sequence_id}"
            ) + "__reserve",
        ).functions[0]
        reserve_block = builder.block("replace_reserve")
        builder.cond(entry, valid_source, reserve_block, exhausted)
        storage_by_id = {int(value.id): value for value in storage}
        reserve_status = builder.fresh("int")
        builder.emit(reserve_block, "Call", [
            *(storage_by_id[int(value.id)] for value in reserve_function.args[:-1]),
            length,
        ], reserve_status, attributes={
            "callee": reserve_function.name, "source_linked": True,
        })
        reserve_invalid = builder.fresh("bool")
        builder.emit(reserve_block, "Eq", [
            reserve_status, builder.const(reserve_block, 2),
        ], reserve_invalid)
        builder.cond(reserve_block, reserve_invalid, exhausted, header)
    else:
        builder.emit(entry, "Le", [length, SSAValue(destination.capacity_value_id, dtype="int64")], destination_fits)
        builder.emit(entry, "And", [valid_source, destination_fits], fits)
        builder.cond(entry, fits, header, exhausted)
    index, next_index = builder.fresh("int64"), builder.fresh("int64")
    builder.emit(header, "Phi", [zero, next_index], index,
                 attributes={"incoming_blocks": (
                     entry.name if reserve_function is None else "replace_reserve",
                     latch.name,
                 )})
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, length], continues)
    builder.cond(header, continues, body, complete)
    column_value_maps = dict(column_value_maps or {})
    for column_index, ((dst, dtype), (src, _)) in enumerate(zip(
        dst_columns, src_columns
    )):
        src_address, dst_address = builder.fresh("ptr"), builder.fresh("ptr")
        value = builder.fresh(dtype)
        builder.emit(body, "GetElementPtr", [SSAValue(src, dtype=dtype), index], src_address)
        builder.emit(body, "Load", [src_address], value)
        value_map = tuple(column_value_maps.get(int(column_index), ()))
        if value_map:
            remapped = builder.const(body, int(value_map[0][1]), dtype=dtype)
            for source_token, target_token in value_map[1:]:
                matches = builder.fresh("bool")
                selected = builder.fresh(dtype)
                builder.emit(body, "Eq", [
                    value, builder.const(body, int(source_token), dtype=dtype),
                ], matches)
                builder.emit(body, "Select", [
                    matches,
                    builder.const(body, int(target_token), dtype=dtype),
                    remapped,
                ], selected, attributes={
                    "binding": "sequence_column_value_remap",
                    "column": int(column_index),
                })
                remapped = selected
            value = remapped
        builder.emit(body, "GetElementPtr", [SSAValue(dst, dtype=dtype), index], dst_address)
        builder.emit(body, "Store", [value, dst_address])
    builder.branch(body, latch)
    builder.emit(latch, "Add", [index, one], next_index)
    builder.branch(latch, header)
    dst_length_address = builder.fresh("ptr")
    builder.emit(complete, "GetElementPtr", [SSAValue(destination.length_address_id, dtype="int64"), zero], dst_length_address)
    builder.emit(complete, "Store", [length, dst_length_address])
    success = _status_branch(builder, complete, 1, result)
    failure = _status_branch(builder, exhausted, 2, result)
    status = builder.fresh("int")
    builder.emit(result, "Phi", [success, failure], status,
                 attributes={"incoming_blocks": (complete.name, exhausted.name)})
    builder.emit(result, "Ret", [status])
    function = Function(function_name or f"ssa_sequence_{destination.sequence_id}_replace_{source.sequence_id}",
                        list(storage), builder.blocks, metadata={
        "ssa_sequence_operation": "replace",
        "destination_sequence_id": destination.sequence_id,
        "source_sequence_id": source.sequence_id,
        "sequence_column_value_maps": tuple(sorted(column_value_maps.items())),
        "sequence_array_argument_ids": tuple(dict.fromkeys((
            *(column for column, _ in dst_columns), destination.length_address_id,
            *(column for column, _ in src_columns), source.length_address_id,
        ))),
        "named_outputs": (("status", int(status.id)),),
    })
    if reserve_function is not None:
        return SSASequenceLowering((function, reserve_function))
    return SSASequenceLowering((function,))


def lower_sequence_extend(
    destination: SSASequenceDescriptor,
    source: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    first_value_id: int | None = None,
    column_value_maps: dict[int, tuple[tuple[int, int], ...]] | None = None,
) -> SSASequenceLowering:
    """Lower extend to source iteration plus the destination's insert policy."""

    unsupported = _unsupported_destination(destination, "extend", allow_dynamic=True)
    if unsupported is not None:
        return unsupported
    if len(destination.column_value_ids) != len(source.column_value_ids):
        raise ValueError("sequence extend requires matching row widths")
    from ..transmogrifier.dtype_layout import layout_for
    source_layouts = tuple(map(layout_for, source.column_dtypes))
    destination_layouts = tuple(map(layout_for, destination.column_dtypes))
    for source_layout, destination_layout in zip(source_layouts, destination_layouts, strict=True):
        if source_layout == destination_layout:
            continue
        # c634: array("Q").extend(list[int]) copies rows; it does not
        # identify the source arena with the array's physical storage. The
        # existing insert loop must convert each loaded row before insertion.
        if (source_layout.kind not in {"bool", "int", "uint", "float"}
                or destination_layout.kind not in {"int", "uint", "float"}
                or (source_layout.kind == "float" and destination_layout.kind != "float")
                or any(destination.column_shapes) or any(source.column_shapes)):
            raise ValueError("sequence extend has no declared scalar row conversion: "
                             f"{source_layout.name} -> {destination_layout.name}")
    if destination.heap_owner_value_id is not None or source.heap_owner_value_id is not None:
        if any(destination.column_shapes) or any(source.column_shapes):
            raise ValueError("heap sequence extend requires scalar row columns")

    # Two specialized owners can both have local sequence75. Extending each
    # must not install one global ssa_sequence_75_insert with incompatible
    # length/capacity/owner identities. The supplied operation identity owns
    # its nested insert, just as it owns its reserve helpers.
    insert_name = (f"{function_name}__insert" if function_name is not None
                   else f"ssa_sequence_{destination.sequence_id}_insert")
    insert_lowering = lower_sequence_insert(
        destination,
        function_name=insert_name,
        first_value_id=first_value_id,
    )
    if not insert_lowering.complete:
        return insert_lowering

    destination_storage = _storage_values(destination)
    source_storage = _storage_values(source)
    all_storage = tuple({value.id: value for value in (
        *destination_storage, *source_storage
    )}.values())
    insert_function = insert_lowering.entry
    insert_ids = {
        value.id
        for function in insert_lowering.functions
        for value in (
            *function.args,
            *(
                instruction.res
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            ),
        )
    }
    builder = _Builder(_first_fresh_identity(
        destination,
        source,
        values=all_storage,
        minimum=max(insert_ids, default=(first_value_id or -1) - 1) + 1,
    ))
    entry = builder.block("entry")
    header = builder.block("extend_header")
    body = builder.block("extend_body")
    call_insert = builder.block("extend_insert")
    latch = builder.block("extend_latch")
    complete = builder.block("complete")
    exhausted = builder.block("capacity_exhausted")
    result_block = builder.block("result")
    # Lengths are int64 cells (``_storage_values``); the walk over them is
    # int64 too.  An ``int`` index loaded half the length cell and LLVM
    # refused the index Phi whose latch increment settled i64 (f"a@{n}"'s
    # glyph extend).
    source_length = builder.fresh("int64")
    zero = builder.const(entry, 0, dtype="int64")
    source_length_slot = builder.fresh("ptr")
    builder.emit(
        entry,
        "GetElementPtr",
        [SSAValue(source.length_address_id, dtype="ptr"), zero],
        source_length_slot,
    )
    builder.emit(
        entry,
        "Load",
        [source_length_slot],
        source_length,
    )
    one = builder.const(entry, 1, dtype="int64")
    full_status = builder.const(entry, 2)
    builder.branch(entry, header)
    index = builder.fresh("int64")
    next_index = builder.fresh("int64")
    builder.emit(
        header,
        "Phi",
        [zero, next_index],
        index,
        attributes={"incoming_blocks": (entry.name, latch.name)},
    )
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, source_length], continues)
    builder.cond(header, continues, body, complete)

    column_value_maps = dict(column_value_maps or {})
    row_values: list[SSAValue] = []
    for column_index, (column_id, dtype) in enumerate(zip(
        source.column_value_ids,
        source.column_dtypes or ("unknown",) * len(source.column_value_ids),
    )):
        address = builder.fresh("ptr")
        value = builder.fresh(dtype)
        builder.emit(body, "GetElementPtr", [SSAValue(column_id, dtype=dtype), index], address)
        builder.emit(body, "Load", [address], value)
        value_map = tuple(column_value_maps.get(int(column_index), ()))
        if value_map:
            remapped = builder.const(body, int(value_map[0][1]), dtype=dtype)
            for source_token, target_token in value_map[1:]:
                matches = builder.fresh("bool")
                selected = builder.fresh(dtype)
                builder.emit(body, "Eq", [
                    value, builder.const(body, int(source_token), dtype=dtype),
                ], matches)
                builder.emit(body, "Select", [
                    matches,
                    builder.const(body, int(target_token), dtype=dtype),
                    remapped,
                ], selected, attributes={
                    "binding": "sequence_column_value_remap",
                    "column": int(column_index),
                })
                remapped = selected
            value = remapped
        row_values.append(value)
    if source.live_flags_value_id is not None:
        live_address = builder.fresh("ptr")
        live = builder.fresh("bool")
        builder.emit(
            body,
            "GetElementPtr",
            [SSAValue(source.live_flags_value_id, dtype="bool"), index],
            live_address,
        )
        builder.emit(body, "Load", [live_address], live)
        builder.cond(body, live, call_insert, latch)
    else:
        builder.branch(body, call_insert)

    # Use the same checked integer-row conversion contract as append. Bounds
    # are compared in the loaded source's domain; a bound outside that domain
    # cannot fail and must not itself be encoded in a too-narrow constant.
    insert_block = call_insert
    converted_rows = []
    for value, source_layout, destination_layout in zip(
        row_values, source_layouts, destination_layouts, strict=True,
    ):
        if source_layout == destination_layout:
            converted_rows.append(value)
            continue
        if destination_layout.kind in {"int", "uint"}:
            source_bits, target_bits = 8 * source_layout.byte_size, 8 * destination_layout.byte_size
            source_min = (0 if source_layout.kind in {"uint", "bool"} else -(1 << (source_bits - 1)))
            source_max = ((1 << source_bits) - 1 if source_layout.kind == "uint" else
                          1 if source_layout.kind == "bool" else (1 << (source_bits - 1)) - 1)
            target_min = (0 if destination_layout.kind == "uint" else -(1 << (target_bits - 1)))
            target_max = ((1 << target_bits) - 1 if destination_layout.kind == "uint" else
                          (1 << (target_bits - 1)) - 1)
            for operation, bound, needed in (
                ("Ge", target_min, target_min > source_min),
                ("Le", target_max, target_max < source_max),
            ):
                if not needed:
                    continue
                valid = builder.fresh("bool")
                builder.emit(insert_block, operation,
                             [value, builder.const(insert_block, bound, dtype=source_layout.name)], valid)
                accepted = builder.block(f"row_{value.id}_{operation}_accepted")
                invalid = builder.block(f"row_{value.id}_{operation}_invalid")
                builder.cond(insert_block, valid, accepted, invalid)
                builder.emit(invalid, "Call", [], attributes={
                    "callee": "turing_validation_error", "error_code": 74,
                    "binding": f"ssa_sequence_{destination_layout.name}_range",
                    "source_linked": True,
                })
                builder.emit(invalid, "Ret", [builder.const(invalid, 3)])
                insert_block = accepted
        converted = builder.fresh(destination_layout.name)
        builder.emit(insert_block, "Cast", [value], converted, attributes={
            "target_dtype": destination_layout.name,
            "binding": (f"ssa_sequence_{destination_layout.name}_range"
                        if destination_layout.kind in {"int", "uint"} else
                        f"ssa_sequence_{destination_layout.name}_conversion"),
            "source_dtype": source_layout.name,
        })
        converted_rows.append(converted)
    row_values = converted_rows
    insert_status = builder.fresh("int")
    insert_values = {value.id: value for value in all_storage}
    insert_values.update(zip(insert_function.metadata["sequence_row_argument_ids"], row_values))
    builder.emit(
        insert_block,
        "Call",
        [insert_values[value.id] for value in insert_function.args],
        insert_status,
        attributes={"callee": insert_name, "source_linked": True},
    )
    is_full = builder.fresh("bool")
    builder.emit(insert_block, "Eq", [insert_status, full_status], is_full)
    builder.cond(insert_block, is_full, exhausted, latch)
    builder.emit(latch, "Add", [index, one], next_index)
    builder.branch(latch, header)
    completed_status = _status_branch(builder, complete, 1, result_block)
    exhausted_status = _status_branch(builder, exhausted, 2, result_block)
    status_result = builder.fresh("int")
    builder.emit(
        result_block,
        "Phi",
        [completed_status, exhausted_status],
        status_result,
        attributes={"incoming_blocks": (complete.name, exhausted.name)},
    )
    builder.emit(result_block, "Ret", [status_result])

    name = function_name or (
        f"ssa_sequence_{destination.sequence_id}_extend_"
        f"{source.sequence_id}"
    )
    function = Function(
        name,
        list(all_storage),
        builder.blocks,
        metadata={
            "ssa_sequence_operation": "extend",
            "destination_sequence_id": destination.sequence_id,
            "source_sequence_id": source.sequence_id,
            "destination_key_columns": destination.key_columns,
            "destination_allows_duplicates": destination.allows_duplicates,
            "insert_callee": insert_name,
            "sequence_column_value_maps": tuple(sorted(column_value_maps.items())),
            "sequence_array_argument_ids": tuple((
                *map(int, destination.column_value_ids),
                int(destination.length_address_id),
                *map(int, source.column_value_ids),
                int(source.length_address_id),
                *(
                    (int(destination.live_flags_value_id),)
                    if destination.live_flags_value_id is not None else ()
                ),
                *(
                    (int(source.live_flags_value_id),)
                    if source.live_flags_value_id is not None else ()
                ),
            )),
            "named_outputs": (("status", int(status_result.id)),),
        },
    )
    return SSASequenceLowering((*insert_lowering.functions, function))


def lower_sequence_aggregate_constants(
    functions: dict[str, Function],
    sequence_tables: dict[str, SSASequenceTable],
) -> None:
    """Replace structural collection literals with their proven SSA arenas.

    An empty list/set constructor is metadata plus storage allocation, not a
    scalar literal a numerical backend can print.  Once a sequence descriptor
    proves that value is mutable row storage, remove only its empty aggregate
    ``Const`` and expose the same SSA value as an arena argument.  Non-empty
    aggregates normally remain untouched.

    A compile-time table is the other exact case: its ``literal_table`` setup
    has already emitted ordinary per-column stores into a frame arena.  The
    original Python ``dict``/``set`` value is then structural provenance, not
    a second runtime value.  Replace that aggregate ``Const`` with the same
    explicit arena argument used by other compiler-local sequences, but only
    when the same function contains the marked literal-table initialization
    for that exact arena identity.  The emitted runtime binding allocates the
    frame; the retained stores populate it on every invocation.
    """

    # Value ids are numbered per function, so an arena's declared dtype is
    # keyed by (function, id); two descriptors of one function that declare
    # different dtypes for one arena are a disagreement, never first-wins.
    arena_dtypes: dict[tuple[str, int], str | None] = {}
    for table_name, table in sequence_tables.items():
        for descriptor in table.sequences.values():
            if not descriptor.column_value_ids:
                continue
            declared = (
                str(descriptor.column_dtypes[0])
                if descriptor.column_dtypes
                and str(descriptor.column_dtypes[0]) not in {"", "unknown"}
                else None
            )
            key = (str(table_name), int(descriptor.column_value_ids[0]))
            earlier = arena_dtypes.get(key)
            if key in arena_dtypes and None not in (earlier, declared) and earlier != declared:
                raise ValueError(
                    f"lower_sequence_aggregate_constants: sequence arena "
                    f"%t{key[1]} of {key[0]} is declared {earlier!r} and "
                    f"{declared!r} by two sequence descriptors"
                )
            arena_dtypes[key] = declared if declared is not None else earlier
    if not arena_dtypes:
        return

    def arena_dtype(function_name: str, value: SSAValue) -> str:
        # The arena's element type is proven by the Const's own annotation or
        # by the sequence descriptor's declared first column; nothing else.
        # 'unknown' is the absence of a dtype, not a dtype.
        own = str(value.dtype or "")
        proven = (
            own if own not in {"", "unknown"}
            else arena_dtypes.get((function_name, int(value.id)))
        )
        if not proven:
            raise ValueError(
                f"lower_sequence_aggregate_constants: sequence arena value "
                f"%t{int(value.id)} has no proven element dtype (neither the "
                f"aggregate Const nor its sequence descriptor declares one)"
            )
        return str(proven)

    for function in functions.values():
        arena_ids = {
            value_id for name, value_id in arena_dtypes
            if name == function.name
        }
        locally_initialized_arena_ids = {
            int(instruction.args[0].id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in {"GetElementPtr", "getelementptr"}
            and instruction.args
            and instruction.attributes.get("binding")
            in {"ssa_sequence_literal_table", "ssa_sequence_literal_bytes"}
            and int(instruction.args[0].id) in arena_ids
        }
        promoted: dict[int, SSAValue] = {}
        for block in function.blocks.values():
            rewritten = []
            for instruction in block.instrs:
                literal = instruction.attributes.get(
                    "value", instruction.attributes.get("constant")
                )
                if (
                    instruction.op in {"Const", "const"}
                    and instruction.res is not None
                    and int(instruction.res.id) in arena_ids
                    and isinstance(literal, (list, tuple, set, dict))
                ):
                    value_id = int(instruction.res.id)
                    if value_id in locally_initialized_arena_ids:
                        # The marked stores are the executable value. Keeping
                        # the authored aggregate as a scalar Const asks every
                        # backend to print a Python container as one literal.
                        promoted[value_id] = SSAValue(
                            value_id,
                            dtype=arena_dtype(function.name, instruction.res),
                            accounting={
                                "sequence_arena": True,
                                "compile_time_initialized": True,
                            },
                        )
                        continue
                    if len(literal) != 0:
                        rewritten.append(instruction)
                        continue
                    promoted[int(instruction.res.id)] = SSAValue(
                        int(instruction.res.id),
                        dtype=arena_dtype(function.name, instruction.res),
                        accounting={"sequence_arena": True},
                    )
                    continue
                rewritten.append(instruction)
            block.instrs = rewritten
        existing = {int(argument.id) for argument in function.args}
        function.args.extend(
            value for value_id, value in promoted.items()
            if value_id not in existing
        )
        if promoted:
            function.metadata["sequence_aggregate_inputs"] = tuple(
                sorted(promoted)
            )


# Every helper a lowerer builds over a hash-indexed table invalidates the
# index after its length stores (see ``hash_index_carrier``).  Wrapping the
# module's lowerers once means a new lowerer cannot forget it; nested calls
# resolve these globals, and the guard is idempotent per store.
for _lowerer_name, _lowerer in tuple(globals().items()):
    if _lowerer_name.startswith("lower_") and callable(_lowerer):
        globals()[_lowerer_name] = _guard_hash_indexes(_lowerer)
del _lowerer_name, _lowerer


__all__ = [
    "SSASequenceLowering",
    "SSASequenceLoweringShortfall",
    "SSASequenceShortfallCode",
    "lower_sequence_add",
    "lower_sequence_fill",
    "lower_sequence_aggregate_constants",
    "lower_sequence_append",
    "lower_record_sequence_append_with_child_copy",
    "lower_sequence_contains",
    "lower_sequence_extend",
    "lower_sequence_insert",
    "schedule_joined_sequence_mutations",
    "lower_table_lookup",
    "lower_sequence_live_length",
    "lower_table_delete",
    "lower_table_delete_first",
    "lower_child_table_delete",
    "lower_table_store",
]
