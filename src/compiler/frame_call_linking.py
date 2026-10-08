"""Frame call linking: linked caller members, frame leases, post-aggregate record result reconciliation, propagated frame tails, unused callee formal pruning (relocated from fortran_c_shell)"""

from __future__ import annotations
from typing import (
    Any,
    Collection,
    Mapping,
)
from collections import Counter
from dataclasses import replace

from .frame_identity_book import (
    _argument_binding_fact,
    _frame_cells,
    _frame_mint,
    _frame_post,
    _frame_table_cell,
    _frame_value_cell,
    _result_storage_lease_cell,
)
from .function_alias_concordance import _concordant_function_aliases
from .frame_storage_roles import (
    _linked_frame_physical_shape,
    _linked_frame_storage_role,
    _resolve_repeated_aggregate_output_positions,
)


def _sequence_member_role(descriptor: Any, value_id: int) -> tuple | None:
    """The role ``value_id`` plays in one sequence descriptor, or None."""

    value_id = int(value_id)
    if int(descriptor.sequence_id) == value_id:
        return ("handle",)
    for position, column in enumerate(descriptor.column_value_ids):
        if int(column) == value_id:
            return ("column", position)
    for attribute in (
        "length_address_id", "capacity_value_id",
        "status_address_id", "live_flags_value_id",
    ):
        member = getattr(descriptor, attribute, None)
        if member is not None and int(member) == value_id:
            return (attribute,)
    return None


def _sequence_member_by_role(descriptor: Any, role: tuple) -> int | None:
    if role[0] == "handle":
        return int(descriptor.sequence_id)
    if role[0] == "column":
        columns = tuple(descriptor.column_value_ids)
        return int(columns[role[1]]) if role[1] < len(columns) else None
    member = getattr(descriptor, role[0], None)
    return None if member is None else int(member)


def _linked_caller_member(
    caller_symbol: str,
    record: Any,
    callee_id: int,
    callee_records: Any,
    caller_records: Any,
    callee_sequences: Any,
    caller_sequences: Any,
    caller_formal_ids: Collection[int] = (),
    linked_sequence_members: set[int] | None = None,
) -> int | None:
    """The caller value a callee record member is, by recorded ids only.

    The callee formal's identity was written when it was minted: the record
    descriptor (or that field's sequence descriptor) names it as one member
    of one declared field.  The call's ``argument_bindings`` -- or, for a
    nested record, ``call_record_pair_concordance`` -- names the caller
    record bound to that callee record, and the caller record's same declared
    field names the caller member.  Nothing is matched by parameter spelling,
    position among formals, dtype or shape.  Returns None when the callee
    formal is no member of a record bound at this call (it is then frame
    storage leased for exactly this callee formal).  Two chains naming
    different caller values, or a bound caller record without the declared
    field, raise: the links disagree and nothing may choose between them.
    """

    from .concordance_declarations import (
        CALL_RECORD_PAIR, FRAME_LINK, LINKED_CALLER_MEMBER,
        MEMBER_NOT_BOUND_AT_CALL, RECORD_DESCRIPTOR, RECORD_FIELD_DECOMPOSITION,
        RECORD_MEMBER, SEQUENCE_MEMBER,
    )
    from .identity_concordance import Mode, Unresolved, current_identity_book
    from ..transmogrifier.ssa import SSARecordFieldStorage

    callee_id = int(callee_id)
    caller_formal_ids = set(map(int, caller_formal_ids))
    bound = {
        int(callee_value): int(caller_value)
        for caller_value, callee_value in record.argument_bindings
    }
    book = current_identity_book()
    pair_page = book.page("call_record_pair_concordance")
    resolved: set[int] = set()
    sequence_members: set[int] = set()
    # The cells this decision reads, so the ``linked_caller_member`` row
    # derives from them: the callee formal's membership cells first.
    source_cells: list = list(_frame_cells(
        _frame_table_cell(callee_records, callee_id, RECORD_MEMBER),
        _frame_table_cell(callee_sequences, callee_id, SEQUENCE_MEMBER),
    ))

    def record_decision(member: int | None) -> int | None:
        """Write the decision as the book's row for (caller, callsite,
        callee, formal): the caller value, or Unresolved(MEMBER_NOT_BOUND_AT_CALL)
        when the formal is a member of no record bound at this call.  REVISE:
        a later round may bind a record this round could not."""

        if member is not None:
            source_cells.extend(_frame_cells(
                _frame_table_cell(caller_records, member, RECORD_MEMBER),
                _frame_table_cell(caller_sequences, member, SEQUENCE_MEMBER),
            ))
        cells = _frame_cells(*source_cells)
        _frame_post(
            LINKED_CALLER_MEMBER,
            (
                str(caller_symbol), record.callsite_id,
                str(record.callee_symbol), callee_id,
            ),
            (
                int(member) if member is not None
                else Unresolved(MEMBER_NOT_BOUND_AT_CALL, read=cells)
            ),
            stage=FRAME_LINK, cells=cells, mode=Mode.REVISE,
            reason=MEMBER_NOT_BOUND_AT_CALL,
        )
        return member
    # The callee formal's memberships, read from the book: ``record_member``
    # names the record fields whose values it is, ``sequence_member`` the
    # sequence descriptors it belongs to, whose handles ``record_member``
    # names as sequence fields.
    memberships: list[tuple[int, str, tuple]] = []
    if callee_records is not None:
        for record_id, field_name, _identity, role in (
            callee_records.member_claims(callee_id)
        ):
            if role[0] == "slot":
                memberships.append((int(record_id), field_name, ("value", 0)))
            elif role[0] == "value":
                memberships.append((int(record_id), field_name, role))
        if callee_sequences is not None:
            for sequence_id, role in callee_sequences.member_claims(callee_id):
                for record_id, field_name, _identity, field_role in (
                    callee_records.member_claims(sequence_id)
                ):
                    if field_role == ("sequence",):
                        memberships.append((
                            int(record_id), field_name, ("sequence", role),
                        ))
    # A sequence field's ``value_ids`` are a flattened physical inventory of
    # the same members named more precisely by ``sequence_member``.  Prefer
    # the role-labelled sequence claims.  Treating the positional inventory
    # as an independent field contract loses the distinction between shared
    # columns/extents and private operational cells such as status.
    sequence_fields = {
        (record_id, field_name)
        for record_id, field_name, member in memberships
        if member[0] == "sequence"
    }
    memberships = [
        (record_id, field_name, member)
        for record_id, field_name, member in memberships
        if not (
            member[0] == "value"
            and (record_id, field_name) in sequence_fields
        )
    ]
    for record_id, field_name, member in dict.fromkeys(memberships):
        descriptor = callee_records.records.get(record_id)
        field = None if descriptor is None else next((
            candidate for candidate in descriptor.fields
            if candidate.name == field_name
        ), None)
        if field is None:
            raise ValueError(
                f"{caller_symbol} callsite {record.callsite_id}: book claims "
                f"callee {record.callee_symbol} formal {callee_id} is member "
                f"{member!r} of field {field_name!r} of record {record_id}, "
                f"which the record_descriptor page does not hold"
            )
        caller_record_id = bound.get(record_id)
        if caller_record_id is None:
            pair_row = (
                str(caller_symbol), int(record.callsite_id), record_id,
            )
            caller_record_id = pair_page.latest(pair_row)
            if caller_record_id is not None:
                source_cells.extend(_frame_cells(
                    book.latest_ref(CALL_RECORD_PAIR, pair_row),
                ))
        if caller_record_id is None:
            continue
        caller_descriptor = getattr(caller_records, "records", {}).get(
            int(caller_record_id)
        )
        if caller_descriptor is None:
            continue
        source_cells.extend(_frame_cells(
            _frame_table_cell(callee_records, record_id, RECORD_DESCRIPTOR),
            _frame_table_cell(
                caller_records, caller_record_id, RECORD_DESCRIPTOR,
            ),
        ))
        caller_field = next((
            candidate for candidate in caller_descriptor.fields
            if candidate.storage_identity == field.storage_identity
        ), None)
        if caller_field is None and member[0] == "sequence":
            # ProgramABI keyed fields are deliberately represented by their
            # physical ``length``/``keys``/``values`` fields, not by a second
            # opaque parent field.  Materialization records the exact logical
            # field -> member-role decomposition on the identity book.  Follow
            # that graph receipt here, so a callee's local sequence view binds
            # to the caller's resident parts without matching names, positions,
            # dtypes, or SSA accounting.  A role absent from an existing
            # receipt is callee-private operational storage (normally status).
            decomposition_row = (
                caller_records.owner,
                int(caller_record_id),
                str(field.storage_identity),
            )
            decomposition = book.page(
                "record_field_decomposition"
            ).latest(decomposition_row)
            if decomposition is not None:
                source_cells.extend(_frame_cells(book.latest_ref(
                    RECORD_FIELD_DECOMPOSITION, decomposition_row,
                )))
                caller_member = dict(decomposition).get(tuple(member[1]))
                if caller_member is not None:
                    resolved.add(int(caller_member))
                    sequence_members.add(int(caller_member))
                continue
        if caller_field is None:
            raise ValueError(
                f"{caller_symbol} callsite {record.callsite_id}: callee "
                f"{record.callee_symbol} formal {callee_id} is member "
                f"{member!r} of declared field {field.storage_identity!r} "
                f"of record {record_id}, bound to caller record "
                f"{caller_record_id}, which has no such field"
            )
        if member[0] == "value" and (
            field.storage is SSARecordFieldStorage.SCALAR
        ):
            # A scalar field's value_ids are the SSA versions of ONE
            # slot (versioned writes).  The callee formal is its input
            # slot; the caller's input slot is the one version it
            # receives as a formal.
            caller_inputs = [
                int(value_id) for value_id in caller_field.value_ids
                if int(value_id) in caller_formal_ids
            ]
            if not caller_inputs and len(caller_field.value_ids) == 1:
                # A caller-local record: its field's only version is
                # the slot.
                caller_inputs = [int(caller_field.value_ids[0])]
            if len(caller_inputs) != 1:
                raise ValueError(
                    f"{caller_symbol} callsite {record.callsite_id}: "
                    f"scalar field {field.storage_identity!r} of caller "
                    f"record {caller_record_id} has input slots "
                    f"{caller_inputs!r} among versions "
                    f"{tuple(caller_field.value_ids)!r}"
                )
            resolved.add(caller_inputs[0])
            continue
        if member[0] == "value":
            caller_values = tuple(map(int, caller_field.value_ids))
            if len(caller_values) != len(field.value_ids):
                raise ValueError(
                    f"{caller_symbol} callsite {record.callsite_id}: "
                    f"declared field {field.storage_identity!r} has "
                    f"{len(field.value_ids)} members in callee record "
                    f"{record_id} and {len(caller_values)} in caller "
                    f"record {caller_record_id}: callee {field!r}; "
                    f"caller {caller_field!r}"
                )
            resolved.add(caller_values[member[1]])
            continue
        caller_sequence = (
            None if caller_field.sequence_id is None
            or caller_sequences is None
            else caller_sequences.by_id(int(caller_field.sequence_id))
        )
        caller_member = (
            None if caller_sequence is None
            else _sequence_member_by_role(caller_sequence, member[1])
        )
        if caller_member is None:
            raise ValueError(
                f"{caller_symbol} callsite {record.callsite_id}: callee "
                f"{record.callee_symbol} formal {callee_id} is sequence "
                f"member {member[1]!r} of {field.storage_identity!r}; the "
                f"bound caller record {caller_record_id} names no such "
                f"member"
            )
        resolved.add(int(caller_member))
        sequence_members.add(int(caller_member))
    if len(resolved) > 1:
        raise ValueError(
            f"{caller_symbol} callsite {record.callsite_id}: callee "
            f"{record.callee_symbol} formal {callee_id} links to caller "
            f"values {sorted(resolved)!r}"
        )
    if resolved and linked_sequence_members is not None:
        linked_sequence_members.update(resolved & sequence_members)
    return record_decision(next(iter(resolved), None))


def _linked_caller_member_cell(
    caller_symbol: Any, callsite_id: Any, callee_symbol: Any, formal_id: Any,
) -> Any:
    """The ``linked_caller_member`` cell ``_linked_caller_member`` wrote for
    one callee formal at one call, or None."""

    from .concordance_declarations import LINKED_CALLER_MEMBER
    from .identity_concordance import current_identity_book

    return current_identity_book().latest_ref(
        LINKED_CALLER_MEMBER,
        (str(caller_symbol), callsite_id, str(callee_symbol), int(formal_id)),
    )


def _restore_linked_sequence_member(
    caller: Any, values: dict, member_id: int, callee_formal: Any,
    callee_symbol: str, callsite_id: int,
) -> Any:
    """Materialize the caller's descriptor-named sequence member by its id.

    The caller's sequence descriptor already owns this exact storage
    identity; it is only not yet a physical value.  Restore that identity
    (the ``restored_caller_storage`` rule of frame-tail completion) rather
    than leasing a replacement, which would hand the callee storage the
    caller's record never sees.
    """

    from ..transmogrifier.ssa import SSAValue

    slot = SSAValue(
        int(member_id),
        dtype=callee_formal.dtype,
        shape=tuple(callee_formal.shape or ()),
        device=callee_formal.device,
        accounting={
            "linked_call_frame_storage": str(callee_symbol),
            "callsite_id": int(callsite_id),
            "propagated_formal_id": int(callee_formal.id),
            "restored_linked_sequence_member": True,
        },
    )
    caller.args.append(slot)
    values[int(member_id)] = slot
    return slot


def _lease_source(formal: Any) -> Any:
    """The callee formal a lease is shaped after, minus the callee's names.

    A lease is identified by its ``frame_lease_link`` row (callsite, callee,
    callee formal id).  The callee's ``program_abi_*`` accounting names the
    callee's record; copied onto caller storage it claimed a caller ABI
    field the caller does not have (identity_concordance
    conflicting-storage-claims).
    """

    return replace(formal, accounting={
        key: value for key, value in dict(formal.accounting or {}).items()
        if not str(key).startswith("program_abi_")
    })


def _link_frame_lease(
    caller_symbol: str, slot_id: int, callsite_id: Any,
    callee_symbol: str, callee_id: int, *,
    caller_function: Any = None, callee_function: Any = None,
    sources: Any = (),
) -> Any:
    """Write the one identity of a leased caller slot; it never changes.

    ``frame_lease_link`` row ``(caller, slot)`` -> ``(callsite, callee,
    formal)``, CONCORD, DERIVED(the slot's ``ssa_value`` cell, the callee
    formal's cell, and ``sources``: the ledger decision cell when the caller
    is a ``propose`` site).
    """

    from .concordance_declarations import FRAME_LEASE, FRAME_LINK

    return _frame_post(
        FRAME_LEASE,
        (str(caller_symbol), int(slot_id)),
        (callsite_id, str(callee_symbol), int(callee_id)),
        stage=FRAME_LINK,
        cells=(
            _frame_value_cell(caller_function, slot_id),
            _frame_value_cell(callee_function, callee_id),
            *tuple(sources or ()),
        ),
    )


def _linked_frame_storage_owner(
    formal: Any,
    storage_identity: str | None,
    callee_symbol: str,
    callee_id: int,
) -> tuple[str, Any]:
    """Name a frame slot by its strongest stable physical identity.

    Propagated nested-call formals can carry complete ProgramABI field
    accounting before a record descriptor includes their local SSA id.  Using
    the numeric id as fallback in that state invents a second owner for the
    same field and forces an irreversible split.  The declared record field,
    optional role, dtype, and shape are the cross-call identity; receiver
    parameter spelling is deliberately absent.
    """

    accounting = {} if formal is None else dict(formal.accounting or {})
    record = accounting.get("program_abi_record")
    field = accounting.get("program_abi_field")
    role = _linked_frame_storage_role(accounting)
    if (
        record is not None
        and field is not None
        and accounting.get("program_abi_storage") == "scalar"
        and not _linked_frame_physical_shape(formal)
    ):
        return (
            "program_abi_field",
            (
                str(record), str(field), role,
                None if formal is None else formal.dtype,
                _linked_frame_physical_shape(formal),
            ),
        )
    if storage_identity is not None:
        return "record", str(storage_identity)
    return "value", (str(callee_symbol), int(callee_id))


def _reconcile_post_aggregate_record_results(module: Any) -> tuple[dict, ...]:
    """Carry legalized result positions into records and later call frames."""

    receipts = []
    for caller_symbol, caller in module.functions.items():
        caller_table = module.record_tables.get(str(caller_symbol))
        records = list(module.call_table.get(str(caller_symbol), ()))
        if caller_table is None or not records:
            continue
        record_by_site = {int(record.callsite_id): record for record in records}
        aliases = _concordant_function_aliases(
            caller, include_output_identities=True,
        )

        for block in caller.blocks.values():
            for call in block.instrs:
                attributes = call.attributes or {}
                if call.op not in {"Call", "call"}:
                    continue
                native_ids = tuple(
                    int(row[0])
                    for row in attributes.get("native_result_contract", ())
                )
                if not native_ids:
                    continue
                position_to_output = dict(zip(
                    map(int, attributes.get("output_positions", ())),
                    map(int, attributes.get("output_ids", ())),
                ))
                for row in attributes.get(
                    "aggregate_output_passthrough_bindings", ()
                ):
                    position_to_output.setdefault(int(row[0]), int(row[3]))
                position_to_output, duplicate_receipts = (
                    _resolve_repeated_aggregate_output_positions(
                        native_ids, position_to_output,
                    )
                )
                record = record_by_site.get(int(attributes.get(
                    "plan_callsite_id", -1
                )))
                callee_symbol = str(attributes.get("callee") or "")
                callee = module.functions.get(callee_symbol)
                callee_table = module.record_tables.get(callee_symbol)
                if record is None or callee is None or callee_table is None:
                    continue
                return_layouts = dict(callee.metadata.get(
                    "record_return_layouts", ()
                ))
                for callee_record_id, caller_record_id in record.result_bindings:
                    callee_descriptor = callee_table.records.get(
                        int(callee_record_id)
                    )
                    layout = tuple(map(int, return_layouts.get(
                        int(callee_record_id), ()
                    )))
                    callee_layout = tuple(
                        int(value_id)
                        for field in (
                            () if callee_descriptor is None
                            else callee_descriptor.fields
                        )
                        for value_id in field.value_ids
                    )
                    if not layout or layout != callee_layout:
                        continue
                    starts = tuple(
                        start
                        for start in range(len(native_ids) - len(layout) + 1)
                        if native_ids[start:start + len(layout)] == layout
                    )
                    if len(starts) != 1:
                        continue
                    start = starts[0]
                    if not all(
                        start + offset in position_to_output
                        for offset in range(len(layout))
                    ):
                        continue
                    mapped_by_storage = {}
                    cursor = 0
                    for field in callee_descriptor.fields:
                        width = len(field.value_ids)
                        mapped_by_storage[str(field.storage_identity)] = tuple(
                            int(position_to_output[start + cursor + offset])
                            for offset in range(width)
                        )
                        cursor += width
                    target_record_ids = []
                    current = int(caller_record_id)
                    seen = set()
                    while current not in seen:
                        seen.add(current)
                        if current in caller_table.records:
                            target_record_ids.append(current)
                        if current not in aliases:
                            break
                        current = int(aliases[current])
                    for target_record_id in target_record_ids:
                        descriptor = caller_table.records[target_record_id]
                        fields = tuple(
                            replace(
                                field,
                                value_ids=mapped_by_storage.get(
                                    str(field.storage_identity),
                                    tuple(map(int, field.value_ids)),
                                ),
                            )
                            for field in descriptor.fields
                        )
                        if fields == tuple(descriptor.fields):
                            continue
                        caller_table.records[target_record_id] = replace(
                            descriptor, fields=fields,
                        )
                        receipt = {
                            "function": str(caller_symbol),
                            "callsite_id": int(record.callsite_id),
                            "callee_record_id": int(callee_record_id),
                            "caller_record_id": int(target_record_id),
                            "positions": tuple(sorted(position_to_output.items())),
                            "duplicate_positions": duplicate_receipts,
                            "priority": "exact_legalized_aggregate_position",
                            "tie_policy": "incumbent",
                        }
                        receipts.append(receipt)

        rebuilt = []
        for record in records:
            callee_table = module.record_tables.get(str(record.callee_symbol))
            exact = {}
            if callee_table is not None:
                for caller_record_id, callee_record_id in record.argument_bindings:
                    caller_descriptor = caller_table.records.get(
                        int(caller_record_id)
                    )
                    callee_descriptor = callee_table.records.get(
                        int(callee_record_id)
                    )
                    if (
                        caller_descriptor is None
                        or callee_descriptor is None
                        or caller_descriptor.identity != callee_descriptor.identity
                    ):
                        continue
                    caller_fields = {
                        str(field.storage_identity): field
                        for field in caller_descriptor.fields
                    }
                    for callee_field in callee_descriptor.fields:
                        caller_field = caller_fields.get(str(
                            callee_field.storage_identity
                        ))
                        if (
                            caller_field is None
                            or len(callee_field.value_ids)
                            != len(caller_field.value_ids)
                        ):
                            continue
                        for callee_id, caller_id in zip(
                            map(int, callee_field.value_ids),
                            map(int, caller_field.value_ids),
                            strict=True,
                        ):
                            exact.setdefault(callee_id, caller_id)
            frame_bindings = tuple(
                (
                    int(callee_id), str(kind),
                    exact.get(int(callee_id), source)
                    if str(kind) in {
                        "caller_storage", "caller_value", "caller_alias",
                    }
                    else source,
                )
                for callee_id, kind, source in record.frame_bindings
            )
            if frame_bindings != tuple(record.frame_bindings):
                receipts.append({
                    "function": str(caller_symbol),
                    "callsite_id": int(record.callsite_id),
                    "frame_rebindings": tuple(
                        (int(old[0]), int(old[2]), int(new[2]))
                        for old, new in zip(
                            record.frame_bindings, frame_bindings, strict=True
                        )
                        if old != new
                    ),
                    "priority": "exact_post_aggregate_record_field",
                    "tie_policy": "incumbent",
                })
                record = replace(record, frame_bindings=frame_bindings)
            rebuilt.append(record)
        module.call_table[str(caller_symbol)] = tuple(rebuilt)
    if receipts:
        module.metadata["post_aggregate_record_result_receipts"] = tuple(receipts)
    return tuple(receipts)


def _resident_call_inputs(call: Any, caller_values: Mapping[int, Any]) -> dict[int, Any]:
    """Keep exact formal-to-resident bindings through late frame expansion.

    A frame record names the authored pre-loop value; the scheduled call may
    already read its current Phi. Reconstructing every argument from that
    record discards ordinary SSA flow. Only missing/nonresident bindings need
    recovery from the frame record.
    """
    declared = tuple(call.attributes.get("callee_input_ids", ()))
    if len(declared) != len(call.args) or len(set(declared)) != len(declared):
        return {}
    return {
        int(formal): caller_values[int(actual.id)]
        for formal, actual in zip(declared, call.args)
        if int(actual.id) in caller_values
    }


def _complete_propagated_frame_tails(functions: Mapping[str, Any]) -> int:
    """Complete every call whose callee's signature grew a frame-storage tail.

    Linking a callee to ITS callees can append propagated frame-storage
    formals to its signature after a call into it was already built.
    Such a call is not malformed; it is incomplete by exactly the
    appended tail.  Complete it the way the linker propagates every
    unbound callee formal: the caller receives one storage slot per
    appended formal (stamped ``propagated_formal_id``, the identity
    `_call_argument_identity` reads for it) and passes it through.
    Completing a call appends the same tail to the CALLER's own signature,
    so its callers are now short by that tail too: iterate to a fixed
    point, never a single pass in dictionary order (tick ->
    tire_recurrence -> balloon_tire was left two arguments short exactly
    that way).

    This ran only inside `_prune_unused_callee_formals`, at the very end
    of lowering -- after `_harmonize_call_argument_shapes` had already
    seen the grown callee and reported the tail as an "unidentified
    remainder" that "must already agree in count" (diagnosed against the
    real managed-tire compile: `_no_exchange_observed` gained one
    `linked_call_frame_storage` formal for its own `_scalar` call the
    moment its result became live, and `_apply_energy_sidechain`'s call
    into it was one operand short).  It is the same technique
    `_propagate_record_field_demand` already runs early for record
    fields; run it early for frame storage too.  Returns the number of
    slots appended.
    """

    from ..transmogrifier.ssa import SSAValue
    from .concordance_declarations import (
        ARGUMENT_BINDING, FORMAL_UNBOUND_AT_TAIL, FRAME_TAIL, FRAME_TAIL_SLOT,
        FRAME_TAIL_STAGE,
    )
    from .identity_concordance import Ref, Unresolved, current_identity_book

    def calls_into(callee_name: str) -> list[tuple[Any, Any]]:
        return [
            (caller, instruction)
            for caller in functions.values()
            for block in caller.blocks.values()
            for instruction in block.instrs
            if (
                instruction.op in {"Call", "call"}
                and str(instruction.attributes.get("callee") or "")
                == str(callee_name)
            )
        ]

    appended = 0
    completed_any = True
    while completed_any:
        completed_any = False
        for callee_name, callee in functions.items():
            for owner, call in calls_into(callee_name):
                declared = call.attributes.get("callee_input_ids")
                if declared is not None:
                    if len(declared) != len(call.args) or len(set(declared)) != len(declared):
                        raise ValueError(
                            f"invalid native input binding receipt for {owner.name} -> {callee_name}: "
                            f"receipt={len(declared)}, operands={len(call.args)}, "
                            f"unique_formals={len(set(declared))}"
                        )
                    bound = dict(zip(declared, call.args))
                    live = [formal for formal in callee.args if int(formal.id) in bound]
                    call.args = [bound[int(formal.id)] for formal in live]
                    call.attributes["callee_input_ids"] = tuple(int(formal.id) for formal in live)
                    tail = [formal for formal in callee.args if int(formal.id) not in bound]
                else:
                    tail = callee.args[len(call.args):]
                if len(call.args) >= len(callee.args):
                    continue
                book = current_identity_book()
                callsite_id = call.attributes.get("plan_callsite_id")
                # The binding each tail formal has at this callsite, read
                # through the book's latest cell (never ``cells.get``).  A
                # binding the walk recorded as Unresolved(STORAGE_MINTED_FROM_ABSENCE)
                # still leased storage: that storage is read from
                # ``result_storage_binding`` and restored as caller storage.
                exact_bindings: dict[int, Any] = {}
                binding_cells: dict[int, tuple] = {}
                for formal in tail:
                    if callsite_id is None:
                        continue
                    fact = _argument_binding_fact(
                        book, callee_name, formal.id, int(callsite_id),
                    )
                    binding_cells[int(formal.id)] = _frame_cells(
                        book.latest_ref(ARGUMENT_BINDING, (
                            str(callee_name), int(formal.id), int(callsite_id),
                        )),
                    )
                    if isinstance(fact, Unresolved):
                        lease = _result_storage_lease_cell(
                            owner.name, int(callsite_id), int(formal.id),
                        )
                        storage_cell = (
                            None if lease is None
                            else book.pages[lease.page.name].latest(lease.row)
                        )
                        if isinstance(storage_cell, Ref):
                            fact = ("caller_storage", int(storage_cell.row[1]))
                            binding_cells[int(formal.id)] += (lease,)
                        else:
                            fact = None
                    exact_bindings[int(formal.id)] = fact
                if not all(
                    (formal.accounting or {}).get("linked_call_frame_storage")
                    or (formal.accounting or {}).get("compiler_frame_storage")
                    or (formal.accounting or {}).get("returned_record_storage")
                    or (
                        isinstance(exact_bindings.get(int(formal.id)), tuple)
                        and exact_bindings[int(formal.id)][0]
                        == "caller_storage"
                        and isinstance(exact_bindings[int(formal.id)][1], int)
                    )
                    for formal in tail
                ):
                    continue
                owner_values = {
                    int(value.id): value
                    for value in owner.args
                }
                owner_values.update({
                    int(instruction.res.id): instruction.res
                    for block in owner.blocks.values()
                    for instruction in block.instrs
                    if instruction.res is not None
                })
                for formal in tail:
                    formal_accounting = dict(formal.accounting or {})
                    exact_binding = exact_bindings.get(int(formal.id))
                    if (
                        isinstance(exact_binding, tuple)
                        and len(exact_binding) == 2
                        and exact_binding[0] == "caller_storage"
                        and isinstance(exact_binding[1], int)
                    ):
                        bound_from_concordance = True
                        source_id = int(exact_binding[1])
                        slot = owner_values.get(source_id)
                        if slot is None:
                            # The frame-binding transaction already minted
                            # this exact caller-storage identity, but a later
                            # structural cleanup may have removed the then-
                            # unreferenced formal before the callee acquired
                            # its final tail. Restore the concorded identity;
                            # minting a replacement here would fork one slot
                            # into two private records again.
                            slot = SSAValue(
                                source_id,
                                dtype=formal.dtype,
                                shape=tuple(formal.shape or ()),
                                device=formal.device,
                                accounting={
                                    "linked_call_frame_storage": str(
                                        callee_name
                                    ),
                                    "callsite_id": callsite_id,
                                    "propagated_formal_id": int(formal.id),
                                    "restored_argument_binding": True,
                                },
                            )
                            owner.args.append(slot)
                            owner_values[source_id] = slot
                            storage_kind = (
                                "argument_binding:restored_caller_storage"
                            )
                        else:
                            storage_kind = "argument_binding:caller_storage"
                    else:
                        bound_from_concordance = False
                        # The slot is NOVEL(FRAME_TAIL_SLOT, callee formal
                        # cell) on the owner's ``ssa_value`` page.
                        slot = SSAValue(
                            _frame_mint(
                                owner, FRAME_TAIL_SLOT,
                                (_frame_value_cell(callee, formal.id),),
                                dtype=formal.dtype,
                                shape=tuple(formal.shape or ()),
                                stage=FRAME_TAIL_STAGE,
                            ),
                            dtype=formal.dtype,
                            shape=tuple(formal.shape or ()),
                            device=formal.device,
                            accounting={
                                "linked_call_frame_storage": str(callee_name),
                                "callsite_id": callsite_id,
                                "propagated_formal_id": int(formal.id),
                            },
                        )
                        storage_kind = (
                            "returned_record_storage"
                            if formal_accounting.get("returned_record_storage")
                            else "compiler_frame_storage"
                            if formal_accounting.get("compiler_frame_storage")
                            else "linked_call_frame_storage"
                        )
                    # A callee's returned-record output slots are still
                    # caller-provided frame storage.  If that physical return
                    # surface grows after an outer call was linked, propagate
                    # it through the caller by the exact callee formal id and
                    # commit the transition before mutating either signature.
                    # This is the same transaction as ordinary linked frame
                    # storage, not a positional repair.
                    #
                    # The row derives from the ``argument_binding`` cell it
                    # read and the slot's ``ssa_value`` cell; a tail completed
                    # for a formal no binding row binds is recorded with the
                    # reason FORMAL_UNBOUND_AT_TAIL instead of as a bare fact.
                    page = book.page(
                        "propagated_frame_tail_concordance"
                    )
                    row = (
                        str(owner.name),
                        call.attributes.get("plan_callsite_id"),
                        str(callee_name), int(formal.id),
                    )
                    claim = (int(slot.id), storage_kind)
                    incumbent = page.latest(row)
                    if incumbent is None:
                        _frame_post(
                            FRAME_TAIL, row, claim, stage=FRAME_TAIL_STAGE,
                            cells=(
                                *binding_cells.get(int(formal.id), ()),
                                _frame_value_cell(owner, slot.id),
                                _frame_value_cell(callee, formal.id),
                            ) if bound_from_concordance else (),
                            reason=FORMAL_UNBOUND_AT_TAIL,
                        )
                    elif incumbent != claim:
                        raise ValueError(
                            "propagated frame-tail concordance disagreement: "
                            f"row={row!r}, incumbent={incumbent!r}, "
                            f"candidate={claim!r}"
                        )
                    if not bound_from_concordance:
                        owner.args.append(slot)
                        owner_values[int(slot.id)] = slot
                    call.args.append(slot)
                    if declared is not None:
                        call.attributes["callee_input_ids"] += (int(formal.id),)
                    completed_any = True
                    appended += 1
                if declared is not None:
                    bound = dict(zip(call.attributes["callee_input_ids"], call.args))
                    call.args = [bound[int(formal.id)] for formal in callee.args]
                    call.attributes["callee_input_ids"] = tuple(int(formal.id) for formal in callee.args)
    return appended


def _live_sequence_member_ids(function: Any, sequence_table: Any) -> set[int]:
    """Every storage cell named by a live sequence descriptor of ``function``.

    A descriptor is live when its handle or a column is a formal, or is
    defined and read by an instruction (the rule `_prune_dead_local_sequences`
    retires by).  Its length/capacity/status cells are storage of that
    sequence even when no instruction of this body happens to read them:
    pruning one leaves the descriptor naming a cell nothing defines
    (identity_concordance ``descriptor-member-unknown``; woodshop
    step_with_dt_control_used lost seq35/160/365/368 capacity this way).
    """

    sequences = getattr(sequence_table, "sequences", None) or {}
    if not sequences:
        return set()
    formal_ids = {int(value.id) for value in function.args}
    defined_ids = set()
    consumed_ids = set()
    for block in function.blocks.values():
        for instruction in block.instrs:
            if instruction.res is not None:
                defined_ids.add(int(instruction.res.id))
            consumed_ids.update(int(argument.id) for argument in instruction.args)
    members: set[int] = set()
    for descriptor in sequences.values():
        anchors = (
            int(descriptor.sequence_id), *map(int, descriptor.column_value_ids),
        )
        if not any(
            value_id in formal_ids
            or (value_id in defined_ids and value_id in consumed_ids)
            for value_id in anchors
        ):
            continue
        members.update(anchors)
        members.update(int(value_id) for value_id in (
            descriptor.length_address_id, descriptor.capacity_value_id,
            descriptor.status_address_id, descriptor.live_flags_value_id,
        ) if value_id is not None)
    return members


def _prune_unused_callee_formals(
    functions: Mapping[str, Any],
    call_records: Mapping[str, list[Any]] | None = None,
    sequence_tables: Mapping[str, Any] | None = None,
) -> int:
    """Prune to a fixed point: a formal whose only use was forwarding to a
    callee formal that one pass removes becomes unused itself, so a chain of
    conceptual record handles dies level by level from the bottom up."""

    total = 0
    while True:
        removed = _prune_unused_callee_formals_once(
            functions, call_records, sequence_tables,
        )
        total += removed
        if not removed:
            return total


def _reindex_frame_aliased_call_results(
    functions: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Resolve a frame-aliased result against the final callee signature.

    ``ssa_output_argument`` is an emitted-ABI position, not an identity.
    Record construction and dead conceptual-formal pruning can move that
    position after source-call linking. ``ssa_output_formal_id`` is the exact
    callee-local identity established by the linker; derive the final position
    from it at the completed-module seam and leave a receipt on the call.
    """

    receipts = []
    for caller_name, caller in functions.items():
        for block_name, block in caller.blocks.items():
            for instruction in block.instrs:
                if (
                    instruction.op not in {"Call", "call"}
                    or not instruction.attributes.get(
                        "result_aliases_frame"
                    )
                ):
                    continue
                callee_name = str(
                    instruction.attributes.get("callee") or ""
                )
                callee = functions.get(callee_name)
                formal_id = instruction.attributes.get(
                    "ssa_output_formal_id"
                )
                if callee is None or formal_id is None:
                    continue
                positions = tuple(
                    index for index, formal in enumerate(callee.args)
                    if int(formal.id) == int(formal_id)
                )
                if len(positions) != 1:
                    raise ValueError(
                        "frame-aliased result formal is absent or duplicated "
                        "in the final callee signature: "
                        f"caller={caller_name!r}, callee={callee_name!r}, "
                        f"formal={int(formal_id)}, positions={positions!r}"
                    )
                old_position = instruction.attributes.get(
                    "ssa_output_argument"
                )
                new_position = int(positions[0])
                instruction.attributes["ssa_output_argument"] = new_position
                if old_position == new_position:
                    continue
                receipt = {
                    "caller": str(caller_name),
                    "block": str(block_name),
                    "callee": callee_name,
                    "formal_id": int(formal_id),
                    "old_position": int(old_position),
                    "new_position": new_position,
                    "reason": "final_callee_signature",
                }
                instruction.attributes[
                    "ssa_output_argument_reindexed"
                ] = receipt
                receipts.append(receipt)
    return tuple(receipts)


def _reconcile_singleton_destructured_call_results(
    functions: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Make the final linked Call define its exact singleton projection.

    Source-call frame settlement is a fixed point.  A later round can rebuild
    a linked call from its original PlanCall result even after the
    destructuring concordance has proved that a one-member native return is
    the authored ``temporary[0]`` target.  At the completed-module seam the
    receipt is authoritative: move the Call definition to that exact target
    and retire only its compiler-owned placeholder formal.
    """

    reconciled = []
    for caller_name, caller in functions.items():
        for receipt in caller.metadata.get(
            "singleton_call_result_concordance", ()
        ):
            temporary_id = int(receipt["temporary_id"])
            projected_id = int(receipt["projected_result_id"])
            callsite_id = int(receipt["callsite_id"])
            callee_name = str(receipt["callee"])
            calls = tuple(
                (block_name, instruction)
                for block_name, block in caller.blocks.items()
                for instruction in block.instrs
                if instruction.op in {"Call", "call"}
                and str(instruction.attributes.get("callee") or "")
                == callee_name
                and int(instruction.attributes.get(
                    "plan_callsite_id", -1
                )) == callsite_id
            )
            if len(calls) != 1:
                continue
            block_name, instruction = calls[0]
            if (
                instruction.res is None
                or int(instruction.res.id) != temporary_id
            ):
                continue
            exact_values = {
                id(value): value
                for value in (
                    *caller.args,
                    *(
                        argument
                        for block in caller.blocks.values()
                        for candidate in block.instrs
                        for argument in candidate.args
                    ),
                )
                if int(value.id) == projected_id
            }
            projected = (
                next(iter(exact_values.values()))
                if len(exact_values) == 1
                else type(instruction.res)(
                    projected_id,
                    dtype=instruction.res.dtype,
                    shape=instruction.res.shape,
                    device=instruction.res.device,
                    accounting=dict(instruction.res.accounting or {}),
                )
            )
            projected.dtype = instruction.res.dtype
            projected.shape = tuple(instruction.res.shape or ())
            projected.device = instruction.res.device
            projected.accounting = {
                **dict(projected.accounting or {}),
                **dict(instruction.res.accounting or {}),
            }
            instruction.res = projected
            row = {
                "caller": str(caller_name),
                "block": str(block_name),
                "callee": callee_name,
                "callsite_id": callsite_id,
                "temporary_id": temporary_id,
                "projected_result_id": projected_id,
                "reason": "final_singleton_destructuring_concordance",
            }
            instruction.attributes[
                "singleton_result_reconciled"
            ] = row
            reconciled.append(row)
    return tuple(reconciled)


def _prune_unused_callee_formals_once(
    functions: Mapping[str, Any],
    call_records: Mapping[str, list[Any]] | None = None,
    sequence_tables: Mapping[str, Any] | None = None,
) -> int:
    """Rewrite dead callee formals and every call operand as one transaction.

    Region construction initially publishes its dependency frontier. Later
    constant folding and DCE can make some captures dead, but deleting only a
    formal or only an actual leaves the repository call ABI malformed. This
    pass owns both sides and updates the source-call receipt at the same seam.
    Authored/public parameters and declared storage remain stable even when an
    implementation happens not to read them.
    """

    from ..transmogrifier.ssa import SSAValue

    records = call_records or {}
    removed_total = 0
    def calls_into(callee_name: str) -> list[tuple[Any, Any]]:
        return [
            (caller, instruction)
            for caller in functions.values()
            for block in caller.blocks.values()
            for instruction in block.instrs
            if (
                instruction.op in {"Call", "call"}
                and str(instruction.attributes.get("callee") or "")
                == str(callee_name)
            )
        ]

    _complete_propagated_frame_tails(functions)
    for callee_name, callee in functions.items():
        caller_calls = calls_into(callee_name)
        callers = [instruction for _caller, instruction in caller_calls]
        if not callers:
            continue
        metadata = dict(getattr(callee, "metadata", {}) or {})
        protected = {
            int(value_id)
            for _name, value_id in (
                *metadata.get("parameter_names", ()),
                *metadata.get("named_outputs", ()),
            )
        }
        declared_storage_ids = {
            int(item["value_id"])
            for item in metadata.get("storage_formals", ())
            if item.get("value_id") is not None
        }
        protected.update(declared_storage_ids)
        sequence_member_ids = set(map(
            int, metadata.get("sequence_array_argument_ids", ())
        ))
        sequence_member_ids.update(_live_sequence_member_ids(
            callee, (sequence_tables or {}).get(str(callee_name)),
        ))
        protected.update(sequence_member_ids)
        record_parameter_names = set(map(
            str, dict(metadata.get("parameter_record_abi") or {}),
        ))
        record_parameter_ids = {
            int(value_id)
            for name, value_id in metadata.get("parameter_names", ())
            if str(name) in record_parameter_names
        }

        def conceptual_record_formal(formal: Any) -> bool:
            accounting = dict(formal.accounting or {})
            declared_parameter = accounting.get("program_abi_parameter")
            # A record container (declared ``storage: record`` with no field of
            # its own) is correlation: only its leaves cross the native ABI.
            return (
                not tuple(formal.shape or ())
                and accounting.get("program_abi_field") is None
                and accounting.get("program_abi_storage") in {None, "record"}
                and (
                    int(formal.id) in record_parameter_ids
                    or (
                        declared_parameter is not None
                        and str(declared_parameter) in record_parameter_names
                    )
                )
            )
        referenced = {
            int(value.id)
            for block in callee.blocks.values()
            for instruction in block.instrs
            for value in (
                *instruction.args,
                *((instruction.res,) if instruction.res is not None else ()),
            )
        }
        local_definition_counts = Counter(
            int(instruction.res.id)
            for block in callee.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        )

        def superseded_provisional_formal(formal: Any) -> bool:
            """Whether completed SSA now defines an unaccounted placeholder.

            Control construction can encounter a later region/tensor result
            before its producer exists and temporarily expose that identity as
            a formal.  Once the completed function contains exactly one real
            definition, that placeholder is no longer part of the ABI.  Real
            parameters and every declared storage kind remain protected.
            """

            value_id = int(formal.id)
            accounting = dict(formal.accounting or {})
            return (
                local_definition_counts[value_id] == 1
                and value_id not in declared_storage_ids
                and value_id not in record_parameter_ids
                and accounting.get("program_abi_parameter") is None
                and not any(
                    accounting.get(key) not in {None, ""}
                    for key in (
                        "linked_call_frame_storage",
                        "returned_record_storage",
                        "compiler_frame_storage",
                    )
                )
            )

        removable_indices = tuple(
            index
            for index, formal in enumerate(callee.args)
            if (
                int(formal.id) not in referenced
                or superseded_provisional_formal(formal)
            )
            and (
                superseded_provisional_formal(formal)
                or int(formal.id) not in protected
                or (formal.accounting or {}).get("unbound_variant_source_id")
                is not None
                or conceptual_record_formal(formal)
            )
            and (
                not (formal.accounting or {}).get("program_abi_parameter")
                or conceptual_record_formal(formal)
            )
            and not any(
                (formal.accounting or {}).get(key) not in {None, ""}
                for key in (
                    "linked_call_frame_storage",
                    "returned_record_storage",
                    "compiler_frame_storage",
                )
            )
        )
        if not removable_indices:
            continue
        original_arity = len(callee.args)
        removed_index_set = set(removable_indices)
        keep_indices = tuple(
            index for index in range(original_arity)
            if index not in removed_index_set
        )
        from .identity_concordance import current_identity_book

        pruned_page = current_identity_book().page(
            "pruned_callee_formal_concordance"
        )
        removed_ids = {
            int(callee.args[index].id) for index in removable_indices
            if pruned_page.concord(
                (str(callee_name), int(callee.args[index].id)), "unused",
            ) == "unused"
        }
        for call in callers:
            old_output_position = call.attributes.get(
                "ssa_output_argument"
            )
            output_formal_id = call.attributes.get(
                "ssa_output_formal_id"
            )
            if (
                output_formal_id is None
                and old_output_position is not None
                and 0 <= int(old_output_position) < original_arity
            ):
                output_formal_id = int(
                    callee.args[int(old_output_position)].id
                )
            if len(call.args) < original_arity:
                declared_inputs = tuple(
                    call.attributes.get("callee_input_ids") or ()
                )
                if len(declared_inputs) == len(call.args):
                    bound_ids = set(map(int, declared_inputs))
                    missing_formals = tuple(
                        formal for formal in callee.args
                        if int(formal.id) not in bound_ids
                    )
                else:
                    missing_formals = tuple(callee.args[len(call.args):])
                raise ValueError(
                    "call has fewer operands than the callee signature while "
                    f"pruning {callee_name!r}: operands={len(call.args)} "
                    f"formals={original_arity}; missing="
                    f"{tuple((int(formal.id), dict(formal.accounting or {})) for formal in missing_formals)!r}; "
                    f"callsite={call.attributes.get('plan_callsite_id')!r}; "
                    f"callee_input_ids={declared_inputs!r}"
                )
            call.args = [
                argument for index, argument in enumerate(call.args)
                if index >= original_arity or index in keep_indices
            ]
            for attribute_name in (
                "feed_ids", "feed_shapes", "feed_dtypes", "input_ids", "callee_input_ids",
            ):
                values = call.attributes.get(attribute_name)
                if values is None or len(tuple(values)) != original_arity:
                    continue
                call.attributes[attribute_name] = tuple(
                    values[index] for index in keep_indices
                )
            if output_formal_id is not None:
                output_positions = tuple(
                    new_index
                    for new_index, old_index in enumerate(keep_indices)
                    if int(callee.args[old_index].id)
                    == int(output_formal_id)
                )
                if len(output_positions) != 1:
                    raise ValueError(
                        "pruning removed or duplicated the exact frame-"
                        "aliased result formal: "
                        f"callee={callee_name!r}, "
                        f"formal={int(output_formal_id)}, "
                        f"positions={output_positions!r}"
                    )
                new_output_position = int(output_positions[0])
                call.attributes["ssa_output_formal_id"] = int(
                    output_formal_id
                )
                call.attributes["ssa_output_argument"] = (
                    new_output_position
                )
                if old_output_position != new_output_position:
                    call.attributes[
                        "ssa_output_argument_reindexed"
                    ] = (
                        int(old_output_position), new_output_position,
                        int(output_formal_id),
                        "pruned_callee_formal_concordance",
                    )
        callee.args = [callee.args[index] for index in keep_indices]
        integral = dict(metadata.get("source_region_integral") or {})
        if integral.get("capture_value_ids") is not None:
            integral["capture_value_ids"] = tuple(
                value_id for value_id in integral["capture_value_ids"]
                if int(value_id) not in removed_ids
            )
            metadata["source_region_integral"] = integral
        if "scalar_variant_argument_ids" in metadata:
            metadata["scalar_variant_argument_ids"] = tuple(
                value_id for value_id in metadata["scalar_variant_argument_ids"]
                if int(value_id) not in removed_ids
            )
        if "sequence_array_argument_ids" in metadata:
            metadata["sequence_array_argument_ids"] = tuple(
                value_id for value_id in metadata["sequence_array_argument_ids"]
                if int(value_id) not in removed_ids
            )
        callee.metadata = metadata
        for caller_records in records.values():
            for record_index, record in enumerate(tuple(caller_records)):
                if str(record.callee_symbol or "") != str(callee_name):
                    continue
                caller_records[record_index] = replace(
                    record,
                    argument_bindings=tuple(
                        binding for binding in record.argument_bindings
                        if int(binding[1]) not in removed_ids
                    ),
                    callee_storage_value_ids=tuple(
                        value_id for value_id in record.callee_storage_value_ids
                        if int(value_id) not in removed_ids
                    ),
                    frame_bindings=tuple(
                        binding for binding in record.frame_bindings
                        if int(binding[0]) not in removed_ids
                    ),
                    unresolved_frame_value_ids=tuple(
                        value_id for value_id in record.unresolved_frame_value_ids
                        if int(value_id) not in removed_ids
                    ),
                )
        removed_total += len(removable_indices)
    return removed_total


def _is_dead_conceptual_record_argument(
    argument: Any,
    record_ids: Collection[int],
    consumed_ids: Collection[int],
) -> bool:
    """Distinguish record correlation from physical record storage.

    Late scalar type settlement can give a conceptual record handle the dtype
    of one of its projected fields.  Dtype is therefore not identity evidence.
    ProgramABI field/storage accounting is: a physical field carries one of
    those durable markers, while the aggregate handle does not.
    """

    value_id = int(argument.id)
    accounting = dict(argument.accounting or {})
    physical = any(
        accounting.get(key) not in {None, ""}
        for key in (
            "program_abi_field",
            "program_abi_storage",
            "linked_call_frame_storage",
            "returned_record_storage",
            "compiler_frame_storage",
        )
    )
    return (
        value_id in set(map(int, record_ids))
        and value_id not in set(map(int, consumed_ids))
        and not tuple(argument.shape or ())
        and not physical
    )
