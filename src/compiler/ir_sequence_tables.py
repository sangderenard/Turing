"""Lower SSA sequence/table policies into ordinary memory and control SSA.

This module contains no runtime collection adapter.  It specializes a compact
``SSASequenceDescriptor`` into functions made solely from the repository's
normal CFG and address vocabulary.  A list is a descriptor with no key
columns; a set is the same storage with a unique key; a dictionary is a
multi-column row with one or more unique key columns.

The first executable foundation deliberately supports fixed-capacity arenas.
Dynamic allocation and arena replacement require an allocation ABI which the
repository SSA does not yet state, so requesting dynamic growth produces a
typed shortfall instead of silently truncating or delegating to Python.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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


class SSASequenceShortfallCode(str, Enum):
    DYNAMIC_GROWTH_UNAVAILABLE = "dynamic-growth-unavailable"
    READ_ONLY_DESTINATION = "read-only-destination"


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


class _Builder:
    def __init__(self, first_value_id: int) -> None:
        self.next_value_id = int(first_value_id)
        self.blocks: dict[str, BasicBlock] = {}

    def block(self, name: str) -> BasicBlock:
        block = BasicBlock(name)
        self.blocks[name] = block
        return block

    def fresh(self, dtype: str | None = None) -> SSAValue:
        value = SSAValue(self.next_value_id, dtype=dtype)
        self.next_value_id += 1
        return value

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

    def const(self, block: BasicBlock, literal: int) -> SSAValue:
        result = self.fresh("int")
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


def _storage_values(
    descriptor: SSASequenceDescriptor,
) -> tuple[SSAValue, ...]:
    values = [
        *(SSAValue(value_id, dtype=dtype) for value_id, dtype in zip(
            descriptor.column_value_ids,
            descriptor.column_dtypes
            or ("unknown",) * len(descriptor.column_value_ids),
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
    # Alias descriptors can intentionally name the same arena more than once.
    return tuple({value.id: value for value in values}.values())


def _status_branch(
    builder: _Builder,
    block: BasicBlock,
    status: int,
    target: BasicBlock,
) -> SSAValue:
    value = builder.const(block, status)
    builder.branch(block, target)
    return value


def _unsupported_destination(
    descriptor: SSASequenceDescriptor, operation: str
) -> SSASequenceLowering | None:
    if not descriptor.writable:
        return SSASequenceLowering(shortfalls=(SSASequenceLoweringShortfall(
            SSASequenceShortfallCode.READ_ONLY_DESTINATION,
            descriptor.sequence_id,
            operation,
            "sequence insertion requires writable destination storage",
        ),))
    if descriptor.capacity_policy is SSASequenceCapacityPolicy.DYNAMIC:
        return SSASequenceLowering(shortfalls=(SSASequenceLoweringShortfall(
            SSASequenceShortfallCode.DYNAMIC_GROWTH_UNAVAILABLE,
            descriptor.sequence_id,
            operation,
            "dynamic sequence growth requires an explicit allocation and arena-replacement ABI",
        ),))
    return None


def lower_sequence_insert(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    operation: str = "insert",
) -> SSASequenceLowering:
    """Build fixed-capacity row insertion with optional linear key dedup.

    Return status is 0 when an existing unique row absorbs the insertion, 1
    when a row is written, and 2 when fixed capacity is exhausted.
    """

    unsupported = _unsupported_destination(descriptor, operation)
    if unsupported is not None:
        return unsupported

    storage = _storage_values(descriptor)
    first_id = max((value.id for value in storage), default=-1) + 1
    builder = _Builder(first_id)
    row_values = tuple(
        builder.fresh(dtype)
        for dtype in descriptor.column_dtypes
        or ("unknown",) * len(descriptor.column_value_ids)
    )
    columns = tuple(SSAValue(value_id, dtype=value.dtype) for value_id, value in zip(
        descriptor.column_value_ids, row_values
    ))
    length_address = SSAValue(descriptor.length_address_id, dtype="ptr")
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
    current_length = builder.fresh("int")
    zero = builder.const(entry, 0)
    one = builder.const(entry, 1)
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
        scan_index = builder.fresh("int")
        next_scan_index = builder.fresh("int")
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
    for column, value in zip(columns, row_values):
        address = builder.fresh("ptr")
        builder.emit(write, "GetElementPtr", [column, current_length], address)
        builder.emit(write, "Store", [value, address])
    if live_flags is not None:
        live_address = builder.fresh("ptr")
        builder.emit(write, "GetElementPtr", [live_flags, current_length], live_address)
        builder.emit(write, "Store", [one, live_address])
    next_length = builder.fresh("int")
    builder.emit(write, "Add", [current_length, one], next_length)
    builder.emit(write, "Store", [next_length, length_slot])
    builder.branch(write, inserted)
    status_values.append((
        inserted,
        _status_branch(builder, inserted, 1, result_block),
    ))
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
) -> SSASequenceLowering:
    """Lower source ``append`` through the shared destination insertion policy."""

    return lower_sequence_insert(
        descriptor,
        function_name=function_name,
        operation="append",
    )


def lower_sequence_add(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
) -> SSASequenceLowering:
    """Lower source ``add`` through the same storage, retaining key dedup."""

    return lower_sequence_insert(
        descriptor,
        function_name=function_name,
        operation="add",
    )


def lower_sequence_fill(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
) -> SSASequenceLowering:
    """Fill a caller arena with one repeated value and publish its length."""

    unsupported = _unsupported_destination(descriptor, "fill")
    if unsupported is not None:
        return unsupported
    if len(descriptor.column_value_ids) != 1 or descriptor.key_columns:
        raise ValueError("sequence fill requires one duplicate-policy column")
    storage = _storage_values(descriptor)
    builder = _Builder(max(value.id for value in storage) + 1)
    value = builder.fresh(descriptor.column_dtypes[0])
    requested = builder.fresh("int")
    entry = builder.block("entry")
    header = builder.block("fill_header")
    body = builder.block("fill_body")
    latch = builder.block("fill_latch")
    complete = builder.block("complete")
    full = builder.block("capacity_exhausted")
    zero = builder.const(entry, 0)
    one = builder.const(entry, 1)
    within_capacity = builder.fresh("bool")
    builder.emit(entry, "Le", [requested, SSAValue(
        descriptor.capacity_value_id, dtype="int64"
    )], within_capacity)
    builder.cond(entry, within_capacity, header, full)
    index = builder.fresh("int")
    next_index = builder.fresh("int")
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
) -> SSASequenceLowering:
    """Append one repeated value without discarding resident rows."""

    unsupported = _unsupported_destination(descriptor, "append_fill")
    if unsupported is not None:
        return unsupported
    if len(descriptor.column_value_ids) != 1 or descriptor.key_columns:
        raise ValueError(
            "sequence append-fill requires one duplicate-policy column"
        )
    storage = _storage_values(descriptor)
    builder = _Builder(max(value.id for value in storage) + 1)
    value = builder.fresh(descriptor.column_dtypes[0])
    requested = builder.fresh("int")
    entry = builder.block("entry")
    header = builder.block("append_fill_header")
    body = builder.block("append_fill_body")
    latch = builder.block("append_fill_latch")
    complete = builder.block("complete")
    full = builder.block("capacity_exhausted")
    zero = builder.const(entry, 0)
    one = builder.const(entry, 1)
    length_address = builder.fresh("ptr")
    old_length = builder.fresh("int")
    new_length = builder.fresh("int")
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
    index = builder.fresh("int")
    next_index = builder.fresh("int")
    builder.emit(header, "Phi", [zero, next_index], index, attributes={
        "incoming_blocks": (entry.name, latch.name)
    })
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, requested], continues)
    builder.cond(header, continues, body, complete)
    destination_index = builder.fresh("int")
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
) -> SSASequenceLowering:
    """Append a unit-stride source slice to resident destination storage.

    Bounds are normalized exactly like a positive-step Python slice: negative
    bounds are relative to the source length and both bounds are clipped into
    ``[0, source_length]``.  The slice is copied directly between caller-owned
    arenas; no temporary sequence or runtime iterator is materialized.
    """

    unsupported = _unsupported_destination(destination, "append_slice")
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

    destination_storage = _storage_values(destination)
    source_storage = _storage_values(source)
    all_storage = tuple({
        value.id: value
        for value in (*destination_storage, *source_storage)
    }.values())
    builder = _Builder(max(value.id for value in all_storage) + 1)
    lower = builder.fresh("int")
    upper = builder.fresh("int")
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
    header = builder.block("append_slice_header")
    body = builder.block("append_slice_body")
    latch = builder.block("append_slice_latch")
    complete = builder.block("complete")
    full = builder.block("capacity_exhausted")
    result = builder.block("result")

    zero = builder.const(entry, 0)
    one = builder.const(entry, 1)
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
    lower_is_negative = builder.fresh("bool")
    builder.emit(entry, "Lt", [lower, zero], lower_is_negative)
    builder.cond(entry, lower_is_negative, lower_negative, lower_nonnegative)

    relative_lower = builder.fresh("int")
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
    normalized_lower = builder.fresh("int")
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

    relative_upper = builder.fresh("int")
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
    normalized_upper = builder.fresh("int")
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

    requested = builder.fresh("int")
    new_length = builder.fresh("int")
    builder.emit(capacity, "Sub", [normalized_upper, normalized_lower], requested)
    builder.emit(capacity, "Add", [old_length, requested], new_length)
    within_capacity = builder.fresh("bool")
    builder.emit(capacity, "Le", [new_length, SSAValue(
        destination.capacity_value_id, dtype="int64"
    )], within_capacity)
    builder.cond(capacity, within_capacity, header, full)
    index = builder.fresh("int")
    next_index = builder.fresh("int")
    builder.emit(
        header, "Phi", [zero, next_index], index,
        attributes={"incoming_blocks": (capacity.name, latch.name)},
    )
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, requested], continues)
    builder.cond(header, continues, body, complete)
    source_index = builder.fresh("int")
    source_address = builder.fresh("ptr")
    value = builder.fresh(source.column_dtypes[0])
    destination_index = builder.fresh("int")
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
    name = function_name or (
        f"ssa_sequence_{destination.sequence_id}_append_slice_"
        f"{source.sequence_id}"
    )
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
            )),
            "named_outputs": (("status", int(status.id)),),
        },
    )
    return SSASequenceLowering((function,))


def lower_sequence_pack_bits(
    destination: SSASequenceDescriptor,
    source: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
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
    builder = _Builder(max(value.id for value in storage) + 1)
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
) -> SSASequenceLowering:
    """Insert one scalar at index zero by shifting resident rows right."""

    unsupported = _unsupported_destination(descriptor, "prepend")
    if unsupported is not None:
        return unsupported
    if len(descriptor.column_value_ids) != 1 or descriptor.key_columns:
        raise ValueError("sequence prepend requires one duplicate-policy column")
    storage = _storage_values(descriptor)
    builder = _Builder(max(value.id for value in storage) + 1)
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


def lower_sequence_prepend_packed_bytes(
    destination: SSASequenceDescriptor,
    source: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
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
    builder = _Builder(max(value.id for value in storage) + 1)
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
) -> SSASequenceLowering:
    """Build a key-column membership scan returning one boolean value."""

    if not descriptor.key_columns:
        raise ValueError("sequence membership requires at least one key column")
    storage = _storage_values(descriptor)
    builder = _Builder(max((value.id for value in storage), default=-1) + 1)
    key_dtypes = tuple(
        descriptor.column_dtypes[column] for column in descriptor.key_columns
    )
    queries = tuple(builder.fresh(dtype) for dtype in key_dtypes)
    entry = builder.block("entry")
    header = builder.block("contains_header")
    body = builder.block("contains_body")
    latch = builder.block("contains_latch")
    found = builder.block("found")
    absent = builder.block("absent")
    result_block = builder.block("result")
    zero = builder.const(entry, 0)
    one = builder.const(entry, 1)
    length_slot = builder.fresh("ptr")
    length = builder.fresh("int")
    builder.emit(
        entry,
        "GetElementPtr",
        [SSAValue(descriptor.length_address_id, dtype="int64", shape=(1,)), zero],
        length_slot,
    )
    builder.emit(entry, "Load", [length_slot], length)
    builder.branch(entry, header)
    index = builder.fresh("int")
    next_index = builder.fresh("int")
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
        descriptor.key_columns, key_dtypes, queries
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
    true_value = builder.const(found, 1)
    builder.branch(found, result_block)
    false_value = builder.const(absent, 0)
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
            "sequence_array_argument_ids": tuple((
                *map(int, descriptor.column_value_ids),
                int(descriptor.length_address_id),
                *((int(descriptor.live_flags_value_id),) if descriptor.live_flags_value_id is not None else ()),
            )),
            "named_outputs": (("contains", int(result.id)),),
        },
    )
    return SSASequenceLowering((function,))


def lower_table_lookup(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
    default_parameter: bool = False,
) -> SSASequenceLowering:
    """Build key lookup for a two-column table and publish found status.

    ``default_parameter`` appends one trailing argument of the value column's
    dtype and returns it from the absent branch -- the ``d.get(key, default)``
    contract -- instead of the absent branch's literal zero.
    """

    value_columns = tuple(
        column for column in range(len(descriptor.column_value_ids))
        if column not in descriptor.key_columns
    )
    if not descriptor.key_columns or len(value_columns) != 1:
        raise ValueError("table lookup requires key columns and one value column")
    if descriptor.status_address_id is None:
        raise ValueError("table lookup requires a caller-visible status cell")
    storage = _storage_values(descriptor)
    status_arena = SSAValue(descriptor.status_address_id, dtype="int", shape=(1,))
    builder = _Builder(max((*[value.id for value in storage], status_arena.id)) + 1)
    key_dtypes = tuple(
        descriptor.column_dtypes[column] for column in descriptor.key_columns
    )
    value_column = value_columns[0]
    value_dtype = descriptor.column_dtypes[value_column]
    queries = tuple(builder.fresh(dtype) for dtype in key_dtypes)
    entry = builder.block("entry")
    header = builder.block("lookup_header")
    body = builder.block("lookup_body")
    latch = builder.block("lookup_latch")
    found = builder.block("found")
    absent = builder.block("absent")
    result_block = builder.block("result")
    zero = builder.const(entry, 0)
    one = builder.const(entry, 1)
    length_slot = builder.fresh("ptr")
    length = builder.fresh("int")
    status_slot = builder.fresh("ptr")
    builder.emit(entry, "GetElementPtr", [
        SSAValue(descriptor.length_address_id, dtype="int64", shape=(1,)), zero
    ], length_slot)
    builder.emit(entry, "Load", [length_slot], length)
    builder.emit(entry, "GetElementPtr", [status_arena, zero], status_slot)
    builder.branch(entry, header)
    index = builder.fresh("int")
    next_index = builder.fresh("int")
    builder.emit(header, "Phi", [zero, next_index], index, attributes={
        "incoming_blocks": (entry.name, latch.name)
    })
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, length], continues)
    builder.cond(header, continues, body, absent)
    matches = None
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
    if descriptor.live_flags_value_id is not None:
        live_address = builder.fresh("ptr")
        live = builder.fresh("bool")
        active_match = builder.fresh("bool")
        builder.emit(body, "GetElementPtr", [
            SSAValue(descriptor.live_flags_value_id, dtype="bool"), index
        ], live_address)
        builder.emit(body, "Load", [live_address], live)
        builder.emit(body, "LAnd", [matches, live], active_match)
        matches = active_match
    builder.cond(body, matches, found, latch)
    builder.emit(latch, "Add", [index, one], next_index)
    builder.branch(latch, header)
    value_address = builder.fresh("ptr")
    found_value = builder.fresh(value_dtype)
    builder.emit(found, "GetElementPtr", [
        SSAValue(descriptor.column_value_ids[value_column], dtype=value_dtype), index
    ], value_address)
    builder.emit(found, "Load", [value_address], found_value)
    builder.emit(found, "Store", [one, status_slot])
    builder.branch(found, result_block)
    if default_parameter:
        missing_value = builder.fresh(value_dtype)
        default_argument = missing_value
    else:
        missing_value = builder.fresh(value_dtype)
        default_argument = None
        builder.emit(
            absent, "Const", [], missing_value, attributes={"value": 0},
        )
    builder.emit(absent, "Store", [zero, status_slot])
    builder.branch(absent, result_block)
    result = builder.fresh(value_dtype)
    builder.emit(result_block, "Phi", [found_value, missing_value], result, attributes={
        "incoming_blocks": (found.name, absent.name)
    })
    builder.emit(result_block, "Ret", [result])
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
            "named_outputs": (("value", int(result.id)),),
        },
    )
    return SSASequenceLowering((function,))


def lower_table_store(
    descriptor: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
) -> SSASequenceLowering:
    """Build update-existing-or-insert for a two-column fixed table."""

    unsupported = _unsupported_destination(descriptor, "table_store")
    if unsupported is not None:
        return unsupported
    value_columns = tuple(
        column for column in range(len(descriptor.column_value_ids))
        if column not in descriptor.key_columns
    )
    if not descriptor.key_columns or len(value_columns) != 1:
        raise ValueError("table store requires key columns and one value column")
    if descriptor.status_address_id is None:
        raise ValueError("table store requires a caller-visible status cell")
    storage = _storage_values(descriptor)
    status_arena = SSAValue(descriptor.status_address_id, dtype="int", shape=(1,))
    builder = _Builder(max((*[value.id for value in storage], status_arena.id)) + 1)
    key_dtypes = tuple(
        descriptor.column_dtypes[column] for column in descriptor.key_columns
    )
    value_column = value_columns[0]
    value_dtype = descriptor.column_dtypes[value_column]
    queries = tuple(builder.fresh(dtype) for dtype in key_dtypes)
    new_value = builder.fresh(value_dtype)
    entry = builder.block("entry")
    header = builder.block("store_header")
    body = builder.block("store_body")
    latch = builder.block("store_latch")
    update = builder.block("update")
    capacity_check = builder.block("capacity_check")
    insert = builder.block("insert")
    full = builder.block("capacity_exhausted")
    complete = builder.block("complete")
    zero = builder.const(entry, 0)
    one = builder.const(entry, 1)
    two = builder.const(entry, 2)
    three = builder.const(entry, 3)
    length_slot = builder.fresh("ptr")
    length = builder.fresh("int")
    status_slot = builder.fresh("ptr")
    builder.emit(entry, "GetElementPtr", [
        SSAValue(descriptor.length_address_id, dtype="int64", shape=(1,)), zero
    ], length_slot)
    builder.emit(entry, "Load", [length_slot], length)
    builder.emit(entry, "GetElementPtr", [status_arena, zero], status_slot)
    builder.branch(entry, header)
    index = builder.fresh("int")
    next_index = builder.fresh("int")
    builder.emit(header, "Phi", [zero, next_index], index, attributes={
        "incoming_blocks": (entry.name, latch.name)
    })
    continues = builder.fresh("bool")
    builder.emit(header, "Lt", [index, length], continues)
    builder.cond(header, continues, body, capacity_check)
    matches = None
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
    if descriptor.live_flags_value_id is not None:
        live_address = builder.fresh("ptr")
        live = builder.fresh("bool")
        active_match = builder.fresh("bool")
        builder.emit(body, "GetElementPtr", [
            SSAValue(descriptor.live_flags_value_id, dtype="bool"), index
        ], live_address)
        builder.emit(body, "Load", [live_address], live)
        builder.emit(body, "LAnd", [matches, live], active_match)
        matches = active_match
    builder.cond(body, matches, update, latch)
    builder.emit(latch, "Add", [index, one], next_index)
    builder.branch(latch, header)
    update_address = builder.fresh("ptr")
    builder.emit(update, "GetElementPtr", [
        SSAValue(descriptor.column_value_ids[value_column], dtype=value_dtype), index
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
    ) + ((descriptor.column_value_ids[value_column], value_dtype, new_value),)
    for column_id, dtype, value in inserted_columns:
        address = builder.fresh("ptr")
        builder.emit(insert, "GetElementPtr", [
            SSAValue(column_id, dtype=dtype), length
        ], address)
        builder.emit(insert, "Store", [value, address])
    if descriptor.live_flags_value_id is not None:
        live_address = builder.fresh("ptr")
        builder.emit(insert, "GetElementPtr", [
            SSAValue(descriptor.live_flags_value_id, dtype="bool"), length
        ], live_address)
        builder.emit(insert, "Store", [one, live_address])
    next_length = builder.fresh("int")
    builder.emit(insert, "Add", [length, one], next_length)
    builder.emit(insert, "Store", [next_length, length_slot])
    builder.emit(insert, "Store", [one, status_slot])
    builder.branch(insert, complete)
    builder.emit(full, "Store", [two, status_slot])
    builder.branch(full, complete)
    builder.emit(complete, "Ret", [])
    name = function_name or f"ssa_sequence_{descriptor.sequence_id}_store"
    function = Function(
        name,
        [*storage, status_arena, *queries, new_value],
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
    builder = _Builder(max((*[value.id for value in storage], status_arena.id)) + 1)
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
    one = builder.const(entry, 1)
    deleted = builder.const(entry, 4)
    length_slot = builder.fresh("ptr")
    length = builder.fresh("int")
    status_slot = builder.fresh("ptr")
    builder.emit(entry, "GetElementPtr", [
        SSAValue(descriptor.length_address_id, dtype="int64", shape=(1,)), zero
    ], length_slot)
    builder.emit(entry, "Load", [length_slot], length)
    builder.emit(entry, "GetElementPtr", [status_arena, zero], status_slot)
    builder.branch(entry, header)
    index = builder.fresh("int")
    next_index = builder.fresh("int")
    builder.emit(header, "Phi", [zero, next_index], index, attributes={
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
    builder.emit(latch, "Add", [index, one], next_index)
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
) -> SSASequenceLowering:
    """Delete the first live row, matching ``del table[next(iter(table))]``."""

    unsupported = _unsupported_destination(descriptor, "table_delete_first")
    if unsupported is not None:
        return unsupported
    if descriptor.status_address_id is None or descriptor.live_flags_value_id is None:
        raise ValueError("first-row deletion requires status and live arenas")
    storage = _storage_values(descriptor)
    status_arena = SSAValue(descriptor.status_address_id, dtype="int", shape=(1,))
    builder = _Builder(max((*[value.id for value in storage], status_arena.id)) + 1)
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
    length_slot = builder.fresh("ptr")
    length = builder.fresh("int")
    status_slot = builder.fresh("ptr")
    builder.emit(entry, "GetElementPtr", [SSAValue(
        descriptor.length_address_id, dtype="int64", shape=(1,)
    ), zero], length_slot)
    builder.emit(entry, "Load", [length_slot], length)
    builder.emit(entry, "GetElementPtr", [status_arena, zero], status_slot)
    builder.branch(entry, header)
    index = builder.fresh("int")
    next_index = builder.fresh("int")
    builder.emit(header, "Phi", [zero, next_index], index, attributes={
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
    builder.emit(latch, "Add", [index, one], next_index)
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
    builder = _Builder(max(storage_ids) + 1)
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


def lower_sequence_extend(
    destination: SSASequenceDescriptor,
    source: SSASequenceDescriptor,
    *,
    function_name: str | None = None,
) -> SSASequenceLowering:
    """Lower extend to source iteration plus the destination's insert policy."""

    unsupported = _unsupported_destination(destination, "extend")
    if unsupported is not None:
        return unsupported
    if len(destination.column_value_ids) != len(source.column_value_ids):
        raise ValueError("sequence extend requires matching row widths")

    insert_name = f"ssa_sequence_{destination.sequence_id}_insert"
    insert_lowering = lower_sequence_insert(destination, function_name=insert_name)
    if not insert_lowering.complete:
        return insert_lowering

    destination_storage = _storage_values(destination)
    source_storage = _storage_values(source)
    all_storage = tuple({value.id: value for value in (
        *destination_storage, *source_storage
    )}.values())
    builder = _Builder(max((value.id for value in all_storage), default=-1) + 1)
    entry = builder.block("entry")
    header = builder.block("extend_header")
    body = builder.block("extend_body")
    call_insert = builder.block("extend_insert")
    latch = builder.block("extend_latch")
    complete = builder.block("complete")
    exhausted = builder.block("capacity_exhausted")
    result_block = builder.block("result")
    source_length = builder.fresh("int")
    zero = builder.const(entry, 0)
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
    one = builder.const(entry, 1)
    full_status = builder.const(entry, 2)
    builder.branch(entry, header)
    index = builder.fresh("int")
    next_index = builder.fresh("int")
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

    row_values: list[SSAValue] = []
    for column_id, dtype in zip(
        source.column_value_ids,
        source.column_dtypes or ("unknown",) * len(source.column_value_ids),
    ):
        address = builder.fresh("ptr")
        value = builder.fresh(dtype)
        builder.emit(body, "GetElementPtr", [SSAValue(column_id, dtype=dtype), index], address)
        builder.emit(body, "Load", [address], value)
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

    insert_status = builder.fresh("int")
    builder.emit(
        call_insert,
        "Call",
        [*destination_storage, *row_values],
        insert_status,
        attributes={"callee": insert_name, "source_linked": True},
    )
    is_full = builder.fresh("bool")
    builder.emit(call_insert, "Eq", [insert_status, full_status], is_full)
    builder.cond(call_insert, is_full, exhausted, latch)
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
    return SSASequenceLowering((insert_lowering.functions[0], function))


def lower_sequence_aggregate_constants(
    functions: dict[str, Function],
    sequence_tables: dict[str, SSASequenceTable],
) -> None:
    """Replace proven empty collection literals with caller-provided arenas.

    An empty list/set constructor is metadata plus storage allocation, not a
    scalar literal a numerical backend can print.  Once a sequence descriptor
    proves that value is mutable row storage, remove only its empty aggregate
    ``Const`` and expose the same SSA value as an arena argument.  Non-empty
    aggregates and values without a sequence descriptor are untouched.
    """

    arena_ids = {
        int(descriptor.column_value_ids[0])
        for table in sequence_tables.values()
        for descriptor in table.sequences.values()
        if descriptor.column_value_ids
    }
    if not arena_ids:
        return
    for function in functions.values():
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
                    and len(literal) == 0
                ):
                    promoted[int(instruction.res.id)] = SSAValue(
                        int(instruction.res.id),
                        dtype=instruction.res.dtype or "unknown",
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


__all__ = [
    "SSASequenceLowering",
    "SSASequenceLoweringShortfall",
    "SSASequenceShortfallCode",
    "lower_sequence_add",
    "lower_sequence_fill",
    "lower_sequence_aggregate_constants",
    "lower_sequence_append",
    "lower_sequence_contains",
    "lower_sequence_extend",
    "lower_sequence_insert",
    "lower_table_lookup",
    "lower_table_delete",
    "lower_table_delete_first",
    "lower_child_table_delete",
    "lower_table_store",
]
