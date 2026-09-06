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
    if accounting.get("program_abi_parameter"):
        return False
    return any(accounting.get(key) not in {None, ""} for key in (
        "linked_call_frame_storage",
        "returned_record_storage",
        "compiler_frame_storage",
    ))


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
                    constant = attributes.get("constant", attributes.get("value"))
                    if isinstance(constant, (bool, int, float)):
                        integer_constants[int(instruction.res.id)] = int(constant)
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
            own = None if table is None else table.by_id(int(value_id))
            if (
                own is not None
                and own.metadata_state == "static"
                and tuple(own.shape)
            ):
                shape = tuple(map(int, own.shape))
                if shape not in ordered_views:
                    ordered_views = (*ordered_views, shape)
            dtype = next((
                str(value.dtype)
                for value in occurrences
                if getattr(value, "dtype", None)
            ), "float64")
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
