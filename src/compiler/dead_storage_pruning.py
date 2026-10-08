"""Dead storage pruning: drop unused private sequences and root private formals, strip dead cell bookkeeping (relocated from fortran_c_shell)"""

from __future__ import annotations
from typing import (
    Any,
    Iterable,
    Mapping,
)


def _drop_unused_private_sequences(function: Any, sequence_table: Any) -> int:
    """Remove compiler-owned sequence arenas with no semantic consumer.

    Lowering initializes every local sequence length defensively. That store
    alone must not make an otherwise unused list into public native storage:
    it is scaffolding for a sequence operation, not an observation. Authored
    parameters/outputs and any non-initialization SSA use keep the descriptor.
    """

    if sequence_table is None:
        return 0
    public_ids = {
        int(value_id)
        for _name, value_id in (
            getattr(function, "metadata", {}) or {}
        ).get("named_outputs", ())
    }
    public_ids.update(
        int(value_id)
        for _name, value_id in (
            getattr(function, "metadata", {}) or {}
        ).get("parameter_names", ())
    )
    public_ids.update(
        int(argument.id)
        for argument in getattr(function, "args", ())
        if (argument.accounting or {}).get("program_abi_parameter")
    )
    removed_ids: set[int] = set()
    removed_addresses: set[int] = set()
    removed_sequences: list[int] = []
    instructions = tuple(
        instruction
        for block in getattr(function, "blocks", {}).values()
        for instruction in getattr(block, "instrs", ())
    )
    for sequence_id, descriptor in tuple(sequence_table.sequences.items()):
        resident_ids = {
            int(descriptor.sequence_id),
            *map(int, descriptor.column_value_ids),
            int(descriptor.length_address_id),
            int(descriptor.capacity_value_id),
            *((int(descriptor.status_address_id),)
              if descriptor.status_address_id is not None else ()),
            *((int(descriptor.live_flags_value_id),)
              if descriptor.live_flags_value_id is not None else ()),
        }
        if resident_ids & public_ids:
            continue
        init_addresses = {
            int(instruction.res.id)
            for instruction in instructions
            if (
                instruction.op in {"GetElementPtr", "getelementptr"}
                and instruction.res is not None
                and instruction.args
                and int(instruction.args[0].id)
                == int(descriptor.length_address_id)
                and instruction.attributes.get("binding")
                == "ssa_local_sequence_length"
            )
        }
        meaningful = False
        for instruction in instructions:
            if (
                instruction.res is not None
                and int(instruction.res.id) in init_addresses
            ):
                continue
            if (
                instruction.op in {"Store", "store"}
                and any(int(value.id) in init_addresses for value in instruction.args)
            ):
                continue
            referenced = {
                *(int(value.id) for value in instruction.args),
                *((int(instruction.res.id),) if instruction.res is not None else ()),
            }
            if referenced & resident_ids:
                meaningful = True
                break
        if meaningful:
            continue
        removed_sequences.append(int(sequence_id))
        removed_ids.update(resident_ids)
        removed_addresses.update(init_addresses)
    if not removed_sequences:
        return 0
    function.args = [
        argument for argument in function.args
        if int(argument.id) not in removed_ids
    ]
    for block in function.blocks.values():
        block.instrs = [
            instruction
            for instruction in block.instrs
            if not (
                instruction.res is not None
                and int(instruction.res.id) in removed_addresses
            )
            if not (
                instruction.op in {"Store", "store"}
                and any(
                    int(value.id) in removed_addresses
                    for value in instruction.args
                )
            )
        ]
    for sequence_id in removed_sequences:
        sequence_table.sequences.pop(sequence_id, None)
    metadata = dict(getattr(function, "metadata", {}) or {})
    if "sequence_array_argument_ids" in metadata:
        metadata["sequence_array_argument_ids"] = tuple(
            value_id
            for value_id in metadata["sequence_array_argument_ids"]
            if int(value_id) not in removed_ids
        )
    if "sequence_value_names" in metadata:
        metadata["sequence_value_names"] = tuple(
            item for item in metadata["sequence_value_names"]
            if int(item[0]) not in removed_sequences
        )
    function.metadata = metadata
    return len(removed_sequences)


def _drop_unused_root_private_formals(function: Any) -> int:
    """Erase late compiler frame formals that the root never references."""

    metadata = dict(getattr(function, "metadata", {}) or {})
    protected = {
        int(value_id)
        for _name, value_id in (
            *metadata.get("parameter_names", ()),
            *metadata.get("named_outputs", ()),
        )
    }
    used = {
        int(value.id)
        for block in function.blocks.values()
        for instruction in block.instrs
        for value in (
            *instruction.args,
            *((instruction.res,) if instruction.res is not None else ()),
        )
    }
    locally_defined = {
        int(instruction.res.id)
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.res is not None
    }
    removable = {
        int(argument.id)
        for argument in function.args
        if (
            int(argument.id) not in used | protected
            or (
                int(argument.id) in locally_defined
                and (argument.accounting or {}).get("ssa_call_result_from")
            )
        )
        and not (argument.accounting or {}).get("program_abi_parameter")
    }
    if removable:
        function.args = [
            argument for argument in function.args
            if int(argument.id) not in removable
        ]
        if "sequence_array_argument_ids" in metadata:
            metadata["sequence_array_argument_ids"] = tuple(
                value_id
                for value_id in metadata["sequence_array_argument_ids"]
                if int(value_id) not in removable
            )
        function.metadata = metadata
    return len(removable)


def _drop_formals_and_call_operands(
    all_functions: Mapping[str, Any],
    function: Any,
    dropped_positions: Iterable[int],
) -> None:
    """Drop formals by position and the matching operand at every call site.

    A formal exists only together with the operand every caller feeds it.
    Dropping the formal alone leaves each call site one operand too long,
    and the public-span origin walk skips calls whose arity disagrees --
    silently severing every span reached through this function for every
    caller above it.  ``callee_input_ids`` receipts are trimmed in step.
    """

    dropped = set(map(int, dropped_positions))
    if not dropped:
        return
    original_arity = len(function.args)
    function.args = [
        value
        for position, value in enumerate(function.args)
        if position not in dropped
    ]
    function_symbol = next(
        (
            candidate_symbol
            for candidate_symbol, candidate in all_functions.items()
            if candidate is function
        ),
        None,
    )
    if function_symbol is None:
        return
    for caller in all_functions.values():
        for block in caller.blocks.values():
            for instruction in block.instrs:
                if (
                    instruction.op != "Call"
                    or str(
                        instruction.attributes.get("callee") or ""
                    ) != function_symbol
                    or len(instruction.args) != original_arity
                ):
                    continue
                instruction.args = [
                    argument
                    for position, argument in enumerate(instruction.args)
                    if position not in dropped
                ]
                declared = instruction.attributes.get("callee_input_ids")
                if declared is not None:
                    instruction.attributes["callee_input_ids"] = tuple(
                        value_id for position, value_id in enumerate(declared)
                        if position not in dropped
                    )


def _strip_dead_cell_bookkeeping(function: Any, members: set[int]) -> set[int]:
    """Remove the bookkeeping of storage cells nothing reads anymore.

    ``members`` are cell ids (a retired sequence's handle, columns, length,
    capacity, status).  Their bookkeeping is the address derivation over a
    cell (``GetElementPtr``/casts, ``ssa_local_sequence_length``) and the
    stores through those addresses (``ssa_local_sequence_initialize``,
    ``ssa_sequence_clear``).  A store touches the derived address, not the
    cell, so the closure runs forward over address derivation first; then
    every instruction that touches a member and whose result feeds nothing
    else is removed, to a fixed point, so the derivation goes once its
    store is gone.  A ``Load`` whose result is consumed is a real read and
    keeps the cell alive: it is left in place and the caller keeps the
    formal.  Returns the enlarged member set.
    """

    members = set(map(int, members))
    address_ops = {"GetElementPtr", "gep", "Cast", "cast_like", "BitCast"}
    grown = True
    while grown:
        grown = False
        for block in function.blocks.values():
            for instruction in block.instrs:
                if instruction.res is None or instruction.op not in address_ops:
                    continue
                if any(int(argument.id) in members for argument in instruction.args):
                    if int(instruction.res.id) not in members:
                        members.add(int(instruction.res.id))
                        grown = True
    changed = True
    while changed:
        changed = False
        consumers: dict[int, int] = {}
        for block in function.blocks.values():
            for instruction in block.instrs:
                for argument in instruction.args:
                    consumers[int(argument.id)] = (
                        consumers.get(int(argument.id), 0) + 1
                    )
        for block in function.blocks.values():
            kept = []
            for instruction in block.instrs:
                touches = any(
                    int(argument.id) in members for argument in instruction.args
                )
                result_id = (
                    None if instruction.res is None else int(instruction.res.id)
                )
                unused_result = (
                    result_id is None or consumers.get(result_id, 0) == 0
                )
                if (
                    touches
                    and unused_result
                    and instruction.op not in {
                        "Call", "call", "Ret", "ret", "Br", "CondBr",
                    }
                ):
                    if result_id is not None:
                        members.add(result_id)
                    changed = True
                    continue
                kept.append(instruction)
            block.instrs = kept
    return members
