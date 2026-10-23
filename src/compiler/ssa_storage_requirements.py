"""Backend-neutral storage capacity recovered from repository SSA.

An SSA id is a logical identity, while a compiled frame needs a physical byte
capacity.  Linked programs often retain several shaped *views* of one storage
id; choosing the empty canonical formal (or one arbitrary view) allocates a
single element and corrupts every native backend alike.  This module derives
one conservative allocation requirement from the evidence already carried by
SSA occurrences and the canonical ``SSATensorTable``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping
import re

from ..transmogrifier.dtype_layout import DEFAULT_DTYPE
from .ssa_aggregate_abi import _constant_integer


@dataclass(frozen=True, slots=True)
class SSAStorageRequirement:
    value_id: int
    dtype: str
    shape: tuple[int, ...]
    element_count: int | None
    views: tuple[tuple[int, ...], ...]
    dynamic: bool


def is_compiler_owned_storage(value: Any) -> bool:
    """Whether a root-frame value is workspace rather than authored ABI."""

    accounting = dict(getattr(value, "accounting", None) or {})
    if (
        accounting.get("program_abi_parameter")
        or accounting.get("caller_provided_workspace")
    ):
        return False
    return any(accounting.get(key) not in {None, ""} for key in (
        "linked_call_frame_storage",
        "returned_record_storage",
        "compiler_frame_storage",
    ))


def is_structural_abi_value(function: Any, value: Any) -> bool:
    """Whether a root formal is a descriptor rather than physical storage:
    a declared record parameter, an aggregate, or a keyed ProgramABI slot."""

    metadata = getattr(function, "metadata", None) or {}
    records = dict(metadata.get("parameter_record_abi") or {})
    record_parameter_ids = {
        int(value_id)
        for parameter_name, value_id in metadata.get("parameter_names", ())
        if str(parameter_name) in records
    }
    accounting = dict(getattr(value, "accounting", None) or {})
    return (
        int(value.id) in record_parameter_ids
        or str(getattr(value, "dtype", "") or "").casefold() == "ssa.aggregate"
        or accounting.get("program_abi_storage") == "keyed"
    )


def _static_shape(value: Any) -> tuple[int, ...]:
    shape = tuple(getattr(value, "shape", ()) or ())
    if not shape or any(not isinstance(item, int) or item < 0 for item in shape):
        return ()
    return tuple(map(int, shape))


def module_storage_requirements(
    module: Any,
) -> Mapping[str, Mapping[int, SSAStorageRequirement]]:
    """Recover storage requirements and propagate them across call bindings."""

    functions = getattr(module, "functions", module) or {}
    values: dict[str, dict[int, list[Any]]] = {
        str(name): {} for name in functions
    }
    shapes: dict[str, dict[int, set[tuple[int, ...]]]] = {
        str(name): {} for name in functions
    }
    dynamic_ids: dict[str, set[int]] = {str(name): set() for name in functions}

    def observe(function_name: str, value: Any) -> None:
        if value is None or not hasattr(value, "id"):
            return
        value_id = int(value.id)
        values[function_name].setdefault(value_id, []).append(value)
        shape = _static_shape(value)
        if shape:
            shapes[function_name].setdefault(value_id, set()).add(shape)

    for function_name, function in functions.items():
        function_name = str(function_name)
        integer_constants: dict[int, int] = {}
        address_indices: dict[int, list[tuple[int, ...] | None]] = {}
        for value in getattr(function, "args", ()):
            observe(function_name, value)
        for block in getattr(function, "blocks", {}).values():
            for instruction in getattr(block, "instrs", ()):
                for value in getattr(instruction, "args", ()):
                    observe(function_name, value)
                observe(function_name, getattr(instruction, "res", None))
                if (
                    str(instruction.op).casefold() in {"const", "constant"}
                    and instruction.res is not None
                ):
                    attributes = getattr(instruction, "attributes", {}) or {}
                    constant = _constant_integer(instruction)
                    if constant is not None:
                        integer_constants[int(instruction.res.id)] = constant
                    else:
                        literal = attributes.get("llvm_literal")
                        match = (
                            re.fullmatch(
                                r"i(?:1|8|16|32|64)\s+([-+]?\d+)",
                                literal.strip(),
                            )
                            if isinstance(literal, str) else None
                        )
                        if match is not None:
                            integer_constants[int(instruction.res.id)] = int(
                                match.group(1)
                            )
                if (
                    str(instruction.op).casefold() == "getelementptr"
                    and len(getattr(instruction, "args", ())) > 1
                ):
                    base_id = int(instruction.args[0].id)
                    indices = tuple(
                        integer_constants.get(int(index.id))
                        for index in instruction.args[1:]
                    )
                    address_indices.setdefault(base_id, []).append(
                        tuple(map(int, indices))
                        if all(index is not None for index in indices)
                        else None
                    )
        # When every address operation over one otherwise-rankless base uses
        # compile-time indices, those accesses prove the minimum physical span
        # the function can touch.  This is storage evidence, not a guessed
        # semantic tensor shape, and it propagates to the owning caller below.
        for value_id, accesses in address_indices.items():
            if not accesses or any(access is None for access in accesses):
                continue
            ranks = {len(access) for access in accesses if access is not None}
            if len(ranks) != 1 or next(iter(ranks)) < 1:
                continue
            rank = next(iter(ranks))
            shape = tuple(
                max(access[axis] for access in accesses if access is not None) + 1
                for axis in range(rank)
            )
            if all(extent > 0 for extent in shape):
                shapes[function_name].setdefault(value_id, set()).add(shape)

    tensor_tables = dict(getattr(module, "tensor_tables", {}) or {})
    for function_name, table in (getattr(module, "sequence_tables", {}) or {}).items():
        function = functions.get(function_name)
        if function is None:
            continue
        # Program: ``sum(i * k for i in range(6))``.  Wrong record: the
        # control builder publishes each static producer loop's extent as
        # ``sequence_static_capacity_bounds`` (the generator arena holds at
        # most 6 rows), but only ``static_mapping_capacity_bounds`` was read,
        # so the arena had no element count and emission refused it as
        # dynamic growth.  Rule: both records state an upper bound on the
        # rows of one resident sequence; each bounds its column storage.
        static_bounds = (
            *dict(function.metadata.get("static_mapping_capacity_bounds", {})).items(),
            *tuple(function.metadata.get("sequence_static_capacity_bounds", ())),
        )
        for sequence_id, count in static_bounds:
            descriptor = table.sequences.get(int(sequence_id))
            if descriptor is None:
                continue
            row_shapes = descriptor.column_shapes or tuple(
                () for _column in descriptor.column_value_ids
            )
            for column_id, row_shape in zip(
                descriptor.column_value_ids, row_shapes, strict=True,
            ):
                shapes[function_name].setdefault(int(column_id), set()).add(
                    (max(1, int(count)), *tuple(row_shape))
                )
            # Program: ``d = {1: n, 2: 3}; del d[1]; len(d)``.  Wrong
            # record: the live-flag arena (one flag per row) had no bound,
            # so it was allocated one element and the second row's flag was
            # written past it.  Rule: live flags are bounded like a column.
            if descriptor.live_flags_value_id is not None:
                shapes[function_name].setdefault(
                    int(descriptor.live_flags_value_id), set()
                ).add((max(1, int(count)),))
    for function_name, table in tensor_tables.items():
        function_name = str(function_name)
        if function_name not in functions:
            continue
        for descriptor in getattr(table, "tensors", {}).values():
            value_id = int(descriptor.data_value_id)
            if descriptor.metadata_state == "static" and tuple(descriptor.shape):
                shapes[function_name].setdefault(value_id, set()).add(
                    tuple(map(int, descriptor.shape))
                )
            elif descriptor.metadata_state != "static":
                dynamic_ids[function_name].add(value_id)

    for function_name, function in functions.items():
        function_name = str(function_name)
        for item in (getattr(function, "metadata", {}) or {}).get(
            "storage_formals", ()
        ):
            shape = tuple(map(int, item.get("shape") or ()))
            if shape:
                shapes[function_name].setdefault(
                    int(item["value_id"]), set()
                ).add(shape)

    # Formal/actual bindings describe the same physical span. Aggregate
    # projections similarly bind a callee output slot to caller storage.
    changed = True
    while changed:
        changed = False
        for caller_name, caller in functions.items():
            caller_name = str(caller_name)
            sequence_table = (getattr(module, "sequence_tables", {}) or {}).get(caller_name)
            for sequence_id, source_id in caller.metadata.get("sequence_capacity_sources", ()):
                descriptor = None if sequence_table is None else sequence_table.sequences.get(int(sequence_id))
                if descriptor is None:
                    continue
                # A retained comprehension emits at most one row per input
                # element. Filtering/deduplication can only reduce that count.
                outer_bounds = {
                    max(1, shape[0])
                    for shape in shapes[caller_name].get(int(source_id), ())
                    if shape
                }
                row_shapes = descriptor.column_shapes or tuple(
                    () for _column in descriptor.column_value_ids
                )
                for column_id, row_shape in zip(
                    descriptor.column_value_ids, row_shapes, strict=True,
                ):
                    bounds = {
                        (outer, *tuple(row_shape)) for outer in outer_bounds
                    }
                    column_shapes = shapes[caller_name].setdefault(int(column_id), set())
                    if not bounds.issubset(column_shapes):
                        column_shapes.update(bounds)
                        changed = True
                if descriptor.live_flags_value_id is not None:
                    bounds = {(outer,) for outer in outer_bounds}
                    flag_shapes = shapes[caller_name].setdefault(
                        int(descriptor.live_flags_value_id), set()
                    )
                    if not bounds.issubset(flag_shapes):
                        flag_shapes.update(bounds)
                        changed = True
            # Program: ``count_true([True, False, True])`` where the callee
            # loops ``for b in bs: if b: ...`` (c197; lane E).  Wrong
            # record: the callee's list formal keeps live flags, its column
            # was bounded to three rows, but the flags were bounded only by
            # the static and comprehension bounds above, so the flag formal
            # and the caller actual minted for it had no element count and
            # the Fortran entry asked for an extent no feed can supply.
            # Rule: a live-flag arena holds one flag per row of its
            # sequence, so each bound on the first column's rows bounds it.
            for descriptor in (
                () if sequence_table is None
                else sequence_table.sequences.values()
            ):
                if (
                    descriptor.live_flags_value_id is None
                    or not descriptor.column_value_ids
                ):
                    continue
                row_rank = len(tuple(
                    (descriptor.column_shapes or ((),))[0]
                ))
                bounds = {
                    (shape[0],)
                    for shape in shapes[caller_name].get(
                        int(descriptor.column_value_ids[0]), ()
                    )
                    if len(shape) == row_rank + 1
                }
                flag_shapes = shapes[caller_name].setdefault(
                    int(descriptor.live_flags_value_id), set()
                )
                if not bounds.issubset(flag_shapes):
                    flag_shapes.update(bounds)
                    changed = True
            for block in getattr(caller, "blocks", {}).values():
                instructions = tuple(getattr(block, "instrs", ()))
                for call in instructions:
                    if call.op not in {"Call", "call"}:
                        continue
                    callee_name = str(call.attributes.get("callee") or "")
                    callee = functions.get(callee_name)
                    if callee is None:
                        continue
                    for actual, formal in zip(call.args, callee.args):
                        actual_shapes = shapes[caller_name].setdefault(
                            int(actual.id), set()
                        )
                        formal_shapes = shapes[callee_name].setdefault(
                            int(formal.id), set()
                        )
                        union = actual_shapes | formal_shapes
                        if union != actual_shapes:
                            actual_shapes.update(union)
                            changed = True
                        if union != formal_shapes:
                            formal_shapes.update(union)
                            changed = True
                    named_outputs = tuple(
                        value_id
                        for _name, value_id in (
                            getattr(callee, "metadata", {}) or {}
                        ).get("named_outputs", ())
                    )
                    if not named_outputs or call.res is None:
                        continue
                    projected: dict[int, int] = {}
                    addresses: dict[int, int] = {}
                    for candidate in instructions:
                        if (
                            candidate.op in {"GetElementPtr", "getelementptr"}
                            and candidate.res is not None
                            and candidate.args
                            and int(candidate.args[0].id) == int(call.res.id)
                            and candidate.attributes.get("aggregate_index") is not None
                        ):
                            addresses[int(candidate.res.id)] = int(
                                candidate.attributes["aggregate_index"]
                            )
                        elif (
                            candidate.op in {"Load", "load"}
                            and candidate.res is not None
                            and candidate.args
                            and int(candidate.args[0].id) in addresses
                        ):
                            projected[addresses[int(candidate.args[0].id)]] = int(
                                candidate.res.id
                            )
                    for index, callee_value_id in enumerate(named_outputs):
                        caller_value_id = projected.get(index)
                        if caller_value_id is None:
                            continue
                        caller_shapes = shapes[caller_name].setdefault(
                            caller_value_id, set()
                        )
                        callee_shapes = shapes[callee_name].setdefault(
                            int(callee_value_id), set()
                        )
                        union = caller_shapes | callee_shapes
                        if union != caller_shapes:
                            caller_shapes.update(union)
                            changed = True
                        if union != callee_shapes:
                            callee_shapes.update(union)
                            changed = True

    result: dict[str, dict[int, SSAStorageRequirement]] = {}
    for function_name, function_values in values.items():
        requirements: dict[int, SSAStorageRequirement] = {}
        table = tensor_tables.get(function_name)
        for value_id, occurrences in function_values.items():
            ordered_views = tuple(sorted(
                shapes[function_name].get(value_id, ()),
                key=lambda item: (prod(item), item),
            ))
            shape = ordered_views[-1] if ordered_views else ()
            # A value with an exact static descriptor in ITS OWN function
            # has exactly that storage.  The cross-call view union above is
            # evidence for shapeless values only; letting its largest member
            # override an exact descriptor gave a (8,4,2,3) temporary the
            # storage of an unrelated (3,800,1600) leaf, and every multi-axis
            # address into it then had no extent origin.
            # ``SSATensorTable.by_id`` is keyed by TENSOR id; the value-keyed
            # reader is ``by_data_value``.  Reading ``by_id(value_id)`` here
            # handed ``catalog_clamp``'s count Const (value 2, shape ``()``)
            # the descriptor of tensor 2 (data value 6, shape ``(4,)``), and
            # the Fortran lane then refused the ``binary_scalar_double`` call
            # whose ``n`` operand it saw as a four-element array.  A value's
            # own storage is the allocation-owning static descriptor over it;
            # a view shares the owner's data value with its own shape and
            # ``owns_allocation=False``, so it is not the storage.
            own_storage = (
                set()
                if table is None
                else {
                    tuple(map(int, own.shape))
                    for own in table.by_data_value(int(value_id))
                    if own.metadata_state == "static"
                    and tuple(own.shape)
                    and own.owns_allocation
                }
            )
            if len(own_storage) > 1:
                raise ValueError(
                    f"{function_name}: value {value_id} has "
                    f"{len(own_storage)} allocation-owning static tensor "
                    f"descriptors with different shapes "
                    f"{sorted(own_storage)}"
                )
            if own_storage:
                shape = next(iter(own_storage))
                if shape not in ordered_views:
                    ordered_views = (*ordered_views, shape)
            dtype = next((
                str(value.dtype)
                for value in occurrences
                if getattr(value, "dtype", None)
            ), DEFAULT_DTYPE)
            requirements[value_id] = SSAStorageRequirement(
                value_id=value_id,
                dtype=dtype,
                shape=shape,
                element_count=(prod(shape) if shape else None),
                views=ordered_views,
                dynamic=(
                    not shape and value_id in dynamic_ids[function_name]
                ),
            )
        result[function_name] = requirements
    return result


def function_storage_requirements(
    module: Any, function_name: str,
) -> Mapping[int, SSAStorageRequirement]:
    """Return the propagated requirements for one function."""

    return module_storage_requirements(module)[str(function_name)]


__all__ = [
    "SSAStorageRequirement", "function_storage_requirements",
    "is_compiler_owned_storage", "module_storage_requirements",
]
