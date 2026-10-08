"""Alias retirement: retire concordant sequence aliases, concorded record identity aliases and dead planning aliases; prune dead local sequences (relocated from fortran_c_shell)"""

from __future__ import annotations
from typing import (
    Any,
    Mapping,
)
from dataclasses import replace

from .sequence_storage_binding import _bind_sequence_storage_members
from .function_alias_concordance import (
    _concordant_function_aliases,
    _publish_concordant_function_aliases,
)
from .dead_storage_pruning import (
    _drop_formals_and_call_operands,
    _strip_dead_cell_bookkeeping,
)


def _prune_dead_local_sequences(
    all_functions: Mapping[str, Any],
    all_sequence_tables: Mapping[str, Any],
    all_record_tables: Mapping[str, Any] | None = None,
) -> int:
    """Remove sequence descriptors whose storage the function never has.

    Structural specialization can select away the producer of a local
    sequence -- ``channels = metrics.error_channels or {}`` keeps the field
    and drops the ``{}`` -- while the descriptor minted for that literal
    survives, together with the length/status cells it was given as leased
    frame formals and the entry ``clear`` store into them.  Nothing ever
    defines the descriptor's handle or columns, so nothing can read the
    sequence; the leftover is a formal every caller must lease storage for
    and a store into it, i.e. exactly the "no caller can know what to pass"
    signature (identity_concordance: ``descriptor-member-unknown``).

    A descriptor is dead when neither its handle nor any column is a formal
    or defined by an instruction of the function.  Its bookkeeping is
    removed (any instruction touching only its cells whose result is
    otherwise unused), the descriptor is unregistered, and member formals
    left without a consumer are dropped with their call operands.  Returns
    the number of descriptors removed.
    """

    removed = _retire_concordant_sequence_aliases(
        all_functions, all_sequence_tables, all_record_tables or {},
    )
    for symbol, function in all_functions.items():
        table = all_sequence_tables.get(str(symbol))
        if table is None or not table.sequences:
            continue
        formal_ids = {int(value.id) for value in function.args}
        defined_ids = {
            int(instruction.res.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        }
        consumed_ids = {
            int(argument.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        }
        for sequence_id, descriptor in tuple(table.sequences.items()):
            handle = int(descriptor.sequence_id)
            columns = tuple(map(int, descriptor.column_value_ids))
            # A definition nothing reads does not make the sequence readable.
            # woodshop run_superstep: seq3's handle was an unconsumed entry
            # ``NoneValue`` (the ``None`` default of an optional sequence);
            # it kept the descriptor alive here, later DCE removed it, and
            # the descriptor was left naming cells nothing defines.
            if any(
                value_id in formal_ids
                or (value_id in defined_ids and value_id in consumed_ids)
                for value_id in (handle, *columns)
            ):
                continue
            members = {
                handle, *columns,
                int(descriptor.length_address_id),
                int(descriptor.capacity_value_id),
                *((int(descriptor.status_address_id),)
                  if descriptor.status_address_id is not None else ()),
                *((int(descriptor.live_flags_value_id),)
                  if descriptor.live_flags_value_id is not None else ()),
            }
            members = _strip_dead_cell_bookkeeping(function, members)
            still_consumed = {
                int(argument.id)
                for block in function.blocks.values()
                for instruction in block.instrs
                for argument in instruction.args
            }
            # Stamp every RETAINED member before the descriptor goes.
            #
            # Below, a member formal that nothing consumes is dropped -- but
            # one that is still consumed is kept, and once the descriptor is
            # unregistered nothing can say what that formal was.  The pass
            # that builds ``storage_formals`` learns a cell's identity by
            # walking the surviving descriptors, so an orphaned member is
            # never stamped, never enters the ledger, and arrives at the
            # full-native contract as a formal with no accounting at all --
            # "no caller can know what to pass" -- even though its caller
            # does lease it a slot.
            #
            # This is the last moment the identity exists, so record it
            # here, with the same keys the descriptor walk would have
            # written. Measured on the two-piece llvm_dt_system product:
            # exactly two such formals, both sequence length-address cells.
            ordered_members = (
                *map(int, descriptor.column_value_ids),
                int(descriptor.length_address_id),
                int(descriptor.capacity_value_id),
                *((int(descriptor.status_address_id),)
                  if descriptor.status_address_id is not None else ()),
                *((int(descriptor.live_flags_value_id),)
                  if descriptor.live_flags_value_id is not None else ()),
            )
            member_positions = {
                value_id: index
                for index, value_id in enumerate(ordered_members)
            }
            for argument in function.args:
                value_id = int(argument.id)
                if (
                    value_id not in members
                    or value_id not in still_consumed
                ):
                    continue
                accounting = dict(argument.accounting or {})
                if any(accounting.get(key) not in {None, ""} for key in (
                    "program_abi_parameter",
                    "linked_call_frame_storage",
                    "returned_record_storage",
                    "compiler_frame_storage",
                )):
                    continue
                argument.accounting = {
                    **accounting,
                    "compiler_frame_storage": str(function.name),
                    "compiler_frame_sequence_id": int(sequence_id),
                    # -1 marks the descriptor handle itself, which the
                    # descriptor walk never indexes as a member.
                    "compiler_frame_member": int(
                        member_positions.get(value_id, -1)
                    ),
                    "compiler_frame_orphaned_descriptor": True,
                }
            del table.sequences[sequence_id]
            removed += 1
            _drop_formals_and_call_operands(
                all_functions, function, [
                    position
                    for position, value in enumerate(function.args)
                    if int(value.id) in members
                    and int(value.id) not in still_consumed
                ],
            )
            formal_ids = {int(value.id) for value in function.args}
    return removed


def _retire_concordant_sequence_aliases(
    all_functions: Mapping[str, Any],
    all_sequence_tables: Mapping[str, Any],
    all_record_tables: Mapping[str, Any],
) -> int:
    """Retire duplicate descriptors proved to name one physical sequence.

    Linked record arguments can first materialize a provisional sequence
    descriptor and later prove, member by member, that it is the caller's
    resident descriptor.  The planning concordance owns that exact proof.
    Once every descriptor member resolves to the resident descriptor and the
    non-identity contract is identical, keeping both rows makes one physical
    member appear owned by two sequences.  Remove only that fully concordant
    duplicate and advance record fields that still name its handle.
    """

    from ..transmogrifier.ssa import (
        SSARecordDescriptor, SSARecordFieldDescriptor,
    )

    retired = 0
    for symbol, function in all_functions.items():
        sequence_table = all_sequence_tables.get(str(symbol))
        if sequence_table is None or not sequence_table.sequences:
            continue
        aliases = _concordant_function_aliases(function)
        physical_ids = {
            *(int(value.id) for value in function.args),
            *(
                int(instruction.res.id)
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            ),
        }
        replacements: dict[int, int] = {}
        private_members: dict[int, set[int]] = {}

        def resident(value_id: int) -> int:
            return int(aliases.get(int(value_id), int(value_id)))

        for sequence_id, descriptor in tuple(sequence_table.sequences.items()):
            sequence_id = int(sequence_id)
            resident_id = resident(sequence_id)
            if resident_id == sequence_id:
                continue
            incumbent = sequence_table.by_id(resident_id)
            if incumbent is None:
                continue
            resolved = replace(
                descriptor,
                sequence_id=resident_id,
                column_value_ids=tuple(
                    resident(value_id)
                    for value_id in descriptor.column_value_ids
                ),
                # Once the exact handle and every arena column agree, the
                # resident descriptor owns the extent/status cells.  They
                # are compiler frame bookkeeping, not independent sequence
                # identity; bind them by their declared descriptor roles.
                length_address_id=int(incumbent.length_address_id),
                capacity_value_id=int(incumbent.capacity_value_id),
                status_address_id=incumbent.status_address_id,
                live_flags_value_id=incumbent.live_flags_value_id,
            )
            if resolved != incumbent or sequence_id in physical_ids:
                continue
            stale_private = {
                sequence_id,
                int(descriptor.length_address_id),
                int(descriptor.capacity_value_id),
                *((int(descriptor.status_address_id),)
                  if descriptor.status_address_id is not None else ()),
                *((int(descriptor.live_flags_value_id),)
                  if descriptor.live_flags_value_id is not None else ()),
            }
            # Prove the stale cells are only removable bookkeeping before
            # mutating the real function.  A call/return use means this is
            # not a duplicate descriptor and must remain visible.
            from copy import deepcopy
            trial = deepcopy(function)
            _strip_dead_cell_bookkeeping(trial, stale_private)
            if any(
                int(argument.id) in stale_private
                for block in trial.blocks.values()
                for instruction in block.instrs
                for argument in instruction.args
            ):
                continue
            replacements[sequence_id] = resident_id
            private_members[sequence_id] = stale_private

        if not replacements:
            continue

        record_table = all_record_tables.get(str(symbol))
        if record_table is not None:
            for record_id, descriptor in tuple(record_table.records.items()):
                fields = tuple(
                    SSARecordFieldDescriptor(
                        field.name,
                        field.storage,
                        storage_identity=field.storage_identity,
                        value_ids=field.value_ids,
                        sequence_id=(
                            None if field.sequence_id is None
                            else replacements.get(
                                int(field.sequence_id), int(field.sequence_id),
                            )
                        ),
                        record_id=field.record_id,
                        offset=field.offset,
                        dtype=field.dtype,
                        writable=field.writable,
                    )
                    for field in descriptor.fields
                )
                if fields != tuple(descriptor.fields):
                    record_table.records[int(record_id)] = SSARecordDescriptor(
                        int(record_id), str(descriptor.identity), fields,
                        descriptor.instance_pool,
                    )

        for stale_id, resident_id in sorted(replacements.items()):
            member_bindings: dict[int, int] = {}
            _bind_sequence_storage_members(
                member_bindings,
                sequence_table.by_id(int(stale_id)),
                sequence_table.by_id(int(resident_id)),
            )
            _publish_concordant_function_aliases(
                function,
                {
                    int(alias): int(target)
                    for alias, target in member_bindings.items()
                    if int(alias) != int(target)
                },
                provenance="exact_sequence_descriptor_member_role",
            )
            stale_private = private_members[int(stale_id)]
            _strip_dead_cell_bookkeeping(function, stale_private)
            still_consumed = {
                int(argument.id)
                for block in function.blocks.values()
                for instruction in block.instrs
                for argument in instruction.args
            }
            _drop_formals_and_call_operands(
                all_functions, function, [
                    position
                    for position, value in enumerate(function.args)
                    if int(value.id) in stale_private
                    and int(value.id) not in still_consumed
                ],
            )
            for argument in function.args:
                accounting = dict(argument.accounting or {})
                if accounting.get("compiler_frame_sequence_id") == int(stale_id):
                    argument.accounting = {
                        **accounting,
                        "compiler_frame_sequence_id": int(resident_id),
                    }
            del sequence_table.sequences[int(stale_id)]
            function.metadata.setdefault(
                "retired_concordant_sequence_descriptors", []
            ).append((
                int(stale_id), int(resident_id),
                "exact_memberwise_planning_concordance",
            ))
            retired += 1
    return retired


def _retire_concorded_record_identity_aliases(
    all_functions: Mapping[str, Any],
    all_record_tables: Mapping[str, Any],
    identity_book: Any,
) -> int:
    """Close planning-only record aliases after output identity is published.

    Aggregate record ids describe compile-time structure; they are not SSA
    storage.  Planning may temporarily alias two such ids while fields are
    linked.  Once the output page independently maps both ids to the same
    final record identity, tombstone that planning edge instead of retaining
    it as a physical ``value_aliases`` claim in the finished module.
    """

    planning_page = identity_book.page("planning_value_concordance")
    output_page = identity_book.page("output_identity_concordance")
    retired = 0
    for symbol, function in all_functions.items():
        record_table = all_record_tables.get(str(symbol))
        if record_table is None:
            continue
        record_ids = set(map(int, record_table.records))
        physical_ids = {
            *(int(value.id) for value in function.args),
            *(
                int(instruction.res.id)
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            ),
        }
        aliases = dict(function.metadata.get("value_aliases", {}) or {})
        removable: list[tuple[int, int, int]] = []
        for alias_id, target_id in aliases.items():
            alias_id, target_id = int(alias_id), int(target_id)
            if (
                alias_id not in record_ids
                or target_id not in record_ids
                or alias_id in physical_ids
                or target_id in physical_ids
            ):
                continue
            alias_output = output_page.latest((str(symbol), alias_id))
            target_output = output_page.latest((str(symbol), target_id))
            if alias_output is None or target_output is None:
                continue
            alias_output = output_page.resolve_alias(str(symbol), alias_id)
            target_output = output_page.resolve_alias(str(symbol), target_id)
            if int(alias_output) != int(target_output):
                continue
            # Do not cut an interior planning path.  A terminal structural
            # edge is safe to close; dependent aliases must first advance at
            # their own writer.
            if any(
                int(other_alias) != alias_id
                and int(other_target) == alias_id
                for other_alias, other_target in aliases.items()
            ):
                continue
            removable.append((alias_id, target_id, int(alias_output)))
        if not removable:
            continue
        for alias_id, target_id, output_id in removable:
            aliases.pop(alias_id, None)
            del planning_page.mapping(str(symbol))[alias_id]
            function.metadata.setdefault(
                "retired_record_identity_aliases", []
            ).append((
                alias_id, target_id, output_id,
                "shared_output_identity_concordance",
            ))
            retired += 1
        function.metadata["value_aliases"] = aliases
    return retired


def _retire_dead_planning_aliases(
    all_functions: Mapping[str, Any],
    all_tensor_tables: Mapping[str, Any],
    all_sequence_tables: Mapping[str, Any],
    all_record_tables: Mapping[str, Any],
    identity_book: Any,
) -> int:
    """Tombstone terminal planning edges absent from the finished program.

    Aggregate legalization can retire both a source projection occurrence and
    its provisional target after record descriptors have advanced to emitted
    output slots.  Such an edge is useful planning history but is no longer a
    physical alias.  Close it only when neither endpoint survives as SSA or as
    any tensor/record/sequence descriptor identity, and no other alias depends
    on it.
    """

    planning_page = identity_book.page("planning_value_concordance")
    retired = 0
    for symbol, function in all_functions.items():
        physical_ids = {
            *(int(value.id) for value in function.args),
            *(
                int(instruction.res.id)
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            ),
        }
        structural_ids: set[int] = set()

        def collect_descriptor_ids(value: Any, key: str = "") -> None:
            if isinstance(value, Mapping):
                for child_key, child in value.items():
                    collect_descriptor_ids(child, str(child_key))
                return
            if isinstance(value, (tuple, list)):
                for child in value:
                    collect_descriptor_ids(child, key)
                return
            if (
                value is not None
                and not isinstance(value, bool)
                and (key.endswith("_id") or key.endswith("_ids"))
            ):
                structural_ids.add(int(value))

        for table, attribute in (
            (all_tensor_tables.get(str(symbol)), "tensors"),
            (all_sequence_tables.get(str(symbol)), "sequences"),
            (all_record_tables.get(str(symbol)), "records"),
        ):
            if table is None:
                continue
            for descriptor in getattr(table, attribute).values():
                collect_descriptor_ids(descriptor.to_mapping())
        aliases = dict(function.metadata.get("value_aliases", {}) or {})
        removable = [
            (int(alias_id), int(target_id))
            for alias_id, target_id in aliases.items()
            if int(alias_id) not in physical_ids
            and int(target_id) not in physical_ids
            and int(alias_id) not in structural_ids
            and int(target_id) not in structural_ids
            and not any(
                int(other_alias) != int(alias_id)
                and int(other_target) == int(alias_id)
                for other_alias, other_target in aliases.items()
            )
        ]
        if not removable:
            continue
        for alias_id, target_id in removable:
            aliases.pop(alias_id, None)
            del planning_page.mapping(str(symbol))[alias_id]
            function.metadata.setdefault(
                "retired_dead_planning_aliases", []
            ).append((
                alias_id, target_id,
                "absent_from_final_ssa_and_descriptors",
            ))
            retired += 1
        function.metadata["value_aliases"] = aliases
    return retired
