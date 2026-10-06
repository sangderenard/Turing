"""Linked-frame storage roles: forwarded aggregates, span storage, dead entry-field aliases (relocated from fortran_c_shell)."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def _retain_forwarded_aggregate_storage(call: Any, callee: Any) -> bool:
    """A return of the exact input frame does not define that frame again."""

    if call.attributes.get("result_convention") != "ssa.aggregate":
        return False
    outputs = tuple(call.attributes.get("output_ids", ()))
    selected = tuple(call.attributes.get("callee_output_ids", ()))
    if not outputs or len(outputs) != len(selected):
        return False
    formals = tuple(int(value.id) for value in callee.args)
    returned = tuple(
        int(value.id) for block in callee.blocks.values()
        for instruction in block.instrs if instruction.op == "Ret"
        for value in instruction.args
    )
    if not returned or any(value_id not in formals for value_id in returned):
        return False
    declared = tuple(call.attributes.get("callee_input_ids", formals))
    if len(declared) != len(call.args) or len(set(declared)) != len(declared):
        return False
    bound = dict(zip(declared, call.args))
    if any(callee_id not in formals or callee_id not in bound
           or int(bound[callee_id].id) != int(caller_id)
           for callee_id, caller_id in zip(selected, outputs)):
        return False
    call.attributes["forwarded_output_bindings"] = tuple(zip(selected, outputs))
    for key in ("result_convention", "output_ids", "output_positions", "output_slots", "callee_output_ids"):
        call.attributes.pop(key, None)
    call.res = None
    return True


def _resolve_repeated_aggregate_output_positions(
    native_ids: Sequence[int],
    position_to_output: Mapping[int, int],
) -> tuple[dict[int, int], tuple[tuple[int, int, int, str, str], ...]]:
    """Recover logical result positions omitted by physical deduplication.

    An aggregate result may use the same SSA value in several record fields.
    Legalization emits that value once and records only its retained position.
    The missing logical positions have exact identity evidence: they carry the
    same callee result id.  Resolve them to the first emitted physical output;
    an equal-priority later occurrence cannot displace that incumbent.
    """

    native_ids = tuple(map(int, native_ids))
    resolved = {
        int(position): int(output_id)
        for position, output_id in position_to_output.items()
    }
    output_by_identity = {}
    receipts = []
    for position, callee_output_id in enumerate(native_ids):
        emitted = resolved.get(position)
        incumbent = output_by_identity.get(callee_output_id)
        if incumbent is None and emitted is not None:
            output_by_identity[callee_output_id] = emitted
            continue
        if incumbent is None:
            continue
        if emitted == incumbent:
            continue
        resolved[position] = incumbent
        receipts.append((
            position, callee_output_id, incumbent,
            "exact_repeated_callee_result_identity",
            "incumbent_on_equal_priority",
        ))
    return resolved, tuple(receipts)


def _linked_frame_physical_shape(formal: Any) -> tuple[int, ...]:
    """Return the declared physical shape for linked ProgramABI storage."""

    if formal is None:
        return ()
    accounting = dict(formal.accounting or {})
    rank = accounting.get("program_abi_rank")
    if (
        accounting.get("program_abi_storage") == "scalar"
        and rank in (None, 0)
    ):
        return ()
    return tuple(formal.shape or ())


def _same_declared_span_storage(left: Any, right: Any) -> bool:
    """Recognize two SSA occurrences of one declared span arena.

    Scalar value ids are versions and must remain exact. A ProgramABI span's
    nonempty storage identity names its arena across call-frame occurrences;
    equal schema and arity therefore prove that two ids are views of the same
    physical field.
    """

    from ..transmogrifier.ssa import SSARecordFieldStorage

    return (
        left.storage is SSARecordFieldStorage.SPAN
        and right.storage is SSARecordFieldStorage.SPAN
        and bool(left.storage_identity)
        and left.storage_identity == right.storage_identity
        and left.name == right.name
        and left.record_id == right.record_id
        and left.offset == right.offset
        and left.dtype == right.dtype
        and len(left.value_ids) == len(right.value_ids)
    )


def _linked_frame_storage_role(
    accounting: Mapping[str, Any],
) -> str | None:
    """Return the physical role of one declared frame field.

    Optional presence is a separate Boolean slot and is always marked
    explicitly.  The ordinary scalar field is the payload slot, including
    read-only propagated copies made before optional lowering stamps the
    redundant ``program_abi_optional_payload`` marker.  Treating an absent
    marker as a third role splits one declared field across linked frames.
    """

    if accounting.get("program_abi_optional_presence"):
        return "presence"
    if (
        accounting.get("program_abi_record") is not None
        and accounting.get("program_abi_field") is not None
        and accounting.get("program_abi_storage") == "scalar"
    ):
        return "payload"
    return None


def _preferred_linked_field_candidates(candidates: Iterable[Any]) -> list[Any]:
    """Select the incumbent physical resident for one declared scalar field.

    A writable resident owns mutable storage more strongly than a read-only
    forwarding copy.  Otherwise an authored/non-callsite resident outranks a
    compiler-generated call-frame copy.  Original argument order breaks an
    equal-priority tie, so the incumbent remains stable. Aggregate storage can
    legitimately expose several physical members and retains every candidate.
    """

    candidates = list(candidates)
    if not candidates:
        return []
    if any(
        (value.accounting or {}).get("program_abi_storage") != "scalar"
        for value in candidates
    ):
        return candidates

    def priority(value: Any) -> tuple[int, int]:
        accounting = dict(value.accounting or {})
        return (
            int(bool(accounting.get("program_abi_field_written"))),
            int(accounting.get("callsite_id") is None),
        )

    best = max(map(priority, candidates))
    return [next(value for value in candidates if priority(value) == best)]


def _prune_dead_entry_field_aliases(
    functions: Mapping[str, Any],
    call_records: Mapping[str, Iterable[Any]],
) -> int:
    """Remove superseded generated field slots from top-level signatures.

    Internal callees need transactional call-signature pruning.  A function
    with no incoming repository calls has no caller signature to rewrite, so a
    dead generated duplicate can be removed directly once both instructions
    and call-record receipts have stopped naming it.  Authored residents and
    distinct parameters remain untouched.
    """

    called = {
        str(instruction.attributes.get("callee") or "")
        for function in functions.values()
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op in {"Call", "call"}
    }
    removed = 0
    for function_name, function in functions.items():
        if str(function_name) in called:
            continue
        referenced = {
            int(argument.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            for argument in instruction.args
        }
        receipt_sources = {
            int(source)
            for record in call_records.get(str(function_name), ())
            for _callee_id, kind, source in record.frame_bindings
            if str(kind) in {
                "caller_storage", "caller_value", "caller_alias",
            }
        }
        groups: dict[tuple[Any, ...], list[Any]] = {}
        for argument in function.args:
            accounting = dict(argument.accounting or {})
            record = accounting.get("program_abi_record")
            parameter = accounting.get("program_abi_parameter")
            field = accounting.get("program_abi_field")
            if (
                record is None or parameter is None or field is None
                or accounting.get("program_abi_storage") != "scalar"
            ):
                continue
            key = (
                str(record), str(parameter), str(field),
                _linked_frame_storage_role(accounting),
                argument.dtype, _linked_frame_physical_shape(argument),
            )
            groups.setdefault(key, []).append(argument)

        removable = set()
        # A specialized record identity call can disappear after its declared
        # span fields have been forwarded. The linker-created receiver then has
        # no physical use; unlike an authored parameter, it is not an ABI slot.
        authored = {int(value_id) for _name, value_id in (
            *function.metadata.get("parameter_names", ()),
            *function.metadata.get("named_outputs", ()),
        )}
        removable.update(
            int(argument.id) for argument in function.args
            if (argument.accounting or {}).get("linked_method_receiver_storage")
            and not (argument.accounting or {}).get("program_abi_parameter")
            and int(argument.id) not in referenced | receipt_sources | authored
        )
        # An authored record parameter whose declared fields are formals of
        # their own crosses the boundary as those fields; its aggregate id is
        # correlation, not a slot, once nothing consumes it.
        parameters_with_fields = {
            str((argument.accounting or {}).get("program_abi_parameter"))
            for argument in function.args
            if (argument.accounting or {}).get("program_abi_field")
        }
        from .identity_concordance import current_identity_book

        handle_page = current_identity_book().page(
            "entry_record_handle_concordance"
        )
        removable.update(
            int(value_id)
            for name, value_id in function.metadata.get("parameter_names", ())
            if str(name) in parameters_with_fields
            and int(value_id) not in referenced | receipt_sources
            and not any(
                (argument.accounting or {}).get(key) not in {None, ""}
                for argument in function.args
                if int(argument.id) == int(value_id)
                for key in (
                    "program_abi_field", "program_abi_storage",
                    "linked_call_frame_storage", "returned_record_storage",
                    "compiler_frame_storage",
                )
            )
            and handle_page.concord(
                (str(function_name), str(name)), "fields_only",
            ) == "fields_only"
        )
        # A top-level function has no incoming call transaction to prune an
        # identity-call placeholder from its signature. Once that generated
        # formal is unreferenced, unauthored, and owns no declared storage, it
        # has no native ABI meaning and no caller could supply it.
        removable.update(
            int(argument.id) for argument in function.args
            if int(argument.id) not in referenced | receipt_sources | authored
            and not any(
                (argument.accounting or {}).get(key) not in {None, ""}
                for key in (
                    "program_abi_parameter",
                    "program_abi_field",
                    "program_abi_storage",
                    "linked_call_frame_storage",
                    "returned_record_storage",
                    "compiler_frame_storage",
                )
            )
        )
        for candidates in groups.values():
            if len(candidates) < 2:
                continue
            incumbent = _preferred_linked_field_candidates(candidates)[0]
            for candidate in candidates:
                accounting = dict(candidate.accounting or {})
                candidate_id = int(candidate.id)
                generated = (
                    accounting.get("callsite_id") is not None
                    or "linked_call_frame_storage" in accounting
                )
                if (
                    candidate is not incumbent
                    and generated
                    and candidate_id not in referenced
                    and candidate_id not in receipt_sources
                ):
                    removable.add(candidate_id)
        if not removable:
            continue
        function.args = [
            argument for argument in function.args
            if int(argument.id) not in removable
        ]
        function.metadata.setdefault(
            "pruned_dead_entry_field_aliases", ()
        )
        function.metadata["pruned_dead_entry_field_aliases"] = (
            *function.metadata["pruned_dead_entry_field_aliases"],
            *tuple(sorted(removable)),
        )
        removed += len(removable)
    return removed
