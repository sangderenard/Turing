"""Explicit native representation for optional scalar function results."""

from __future__ import annotations

from typing import Any

from ..transmogrifier.ssa import Instr, SSAValue


_RETURN_OPS = {"Ret", "ret", "Return", "return"}
_PHI_OPS = {"Phi", "phi"}
_NONE_OPS = {"NoneValue"}
_EQ_OPS = {"Eq", "eq", "Equal", "equal"}
_NE_OPS = {"Ne", "ne", "NotEqual", "not_equal"}


def _functions(module: Any) -> dict[str, Any]:
    functions = getattr(module, "functions", module)
    return dict(functions or {})

def _insert_before_terminator(block: Any, instructions: list[Instr]) -> None:
    index = len(block.instrs)
    if index and block.instrs[-1].op in {
        "Br", "br", "Branch", "branch", "CondBr", "condbr", *_RETURN_OPS,
    }:
        index -= 1
    block.instrs[index:index] = instructions


def _record_covered_return_receipts(function: Any) -> tuple[dict[str, Any], ...]:
    """The per-position receipts of a function whose ONE output is an
    aggregate of varying arity (RETURN_OUTPUT_ARMS), in position order.

    Read the aggregate expansion and each payload/presence relationship from
    the owning book. The metadata receipts are diagnostics, not an alternate
    source of arity, positions or payload types.
    """

    from .structural_concordance import aggregate_return_layouts, optional_return_members

    # Program: a function with an optional tuple return and a planned region
    # (a loop body cut out of it).  The region's shape scope IS the parent's
    # scope string, so aggregate_return_layouts(region) reached the PARENT's
    # row (scope, 16) -- a row for a value the region never returns (its Ret
    # is [4]) -- and the call to the region was treated as record-covered
    # (1 payload projection for 2 receipts).  Rule: a layout row belongs to
    # this function only if its output id is an argument of one of this
    # function's own Ret instructions, either the output itself or (once the
    # return is physically expanded) every member of that output's layout;
    # rows for other outputs are not read.
    returned = {
        int(argument.id)
        for block in function.blocks.values()
        for instruction in block.instrs
        if instruction.op in _RETURN_OPS
        for argument in instruction.args
    }
    layouts = {
        output_id: layout
        for output_id, layout in aggregate_return_layouts(function).items()
        if output_id in returned
        or (layout and set(map(int, layout)) <= returned)
    }
    if len(layouts) != 1:
        return ()
    output_id = next(iter(layouts))
    members = optional_return_members(function, output_id)
    if not members or (len(members) == 1 and members[0][0] == output_id):
        return ()
    return tuple({
        "function": str(function.name), "return_output_id": output_id,
        "return_position": position, "payload_value_id": payload_id,
        "presence_value_id": presence_id, "dtype": payload.dtype,
        "shape": payload.shape,
    } for position, (payload_id, presence_id, payload) in enumerate(members))


def _lower_record_covered_call(
    caller_name: str, caller: Any, block: Any, index: int, call: Any,
    position_receipts: tuple[dict[str, Any], ...],
    producers: dict[int, Any], receipts: list[dict[str, Any]],
) -> int:
    """Caller side of a record-covered optional aggregate return.

    The callee returns ``(p0, q0, p1, q1, ...)``.  The linker bound the
    caller's projection ``k`` to payload ``k`` (``call.attributes
    ["output_ids"]``, loaded at aggregate index ``k``); the physical return
    now interleaves presence, so payload ``k`` loads at ``2k`` and a fresh
    presence value loads at ``2k + 1``.  The call's own result is the
    optional aggregate: ``is None`` tests of it read presence ``0`` (the
    kind OPTIONAL_AGGREGATE record says every arm is absent or full width,
    so one presence speaks for all positions).
    """

    from .concordance_declarations import (
        ADDRESS, CONTROL_CONST, OPTIONAL_PRESENCE, RECORD_RETURN_REPAIR,
    )
    from .ssa_record_return_state import mint_ssa_value

    payload_ids = tuple(map(int, call.attributes.get("output_ids", ())))
    # A record-covered call IS lowered completely or not at all.  For the
    # program ``r = f(...); r[0]; cells[r]`` a skipped lowering leaves the
    # raw call aggregate as ONE lookup query (a silent miscompile); the
    # receipts say the call is record-covered, so a shortfall is refused here.
    callee = str(call.attributes.get("callee") or "")
    positions = tuple(int(r["return_position"]) for r in position_receipts)

    def refuse(missing: str) -> ValueError:
        return ValueError(
            f"record-covered optional call %{int(call.res.id)} of "
            f"{callee!r} in {caller_name!r} cannot be lowered: {missing}; "
            f"the callee's optional_return_contracts receipts cover "
            f"positions {positions} of return output "
            f"%{position_receipts[0]['return_output_id']}, the call binds "
            f"output_ids {payload_ids}"
        )

    if len(payload_ids) != len(position_receipts):
        raise refuse(
            f"the call binds {len(payload_ids)} payload projections for "
            f"{len(position_receipts)} receipts"
        )
    payloads = []
    for payload_id in payload_ids:
        load = producers.get(payload_id)
        gep = None if load is None else producers.get(int(load.args[0].id))
        const = None if gep is None else producers.get(int(gep.args[1].id))
        if load is None or gep is None or const is None:
            raise refuse(
                f"projection of payload %{payload_id} lacks its "
                + ", ".join(
                    name for name, part in (
                        ("Load", load), ("GetElementPtr", gep),
                        ("Const", const),
                    ) if part is None
                )
            )
        payloads.append((load, gep, const))
    aggregate = call.res
    presences = []
    projection: list[Instr] = []
    for position, ((load, gep, const), receipt) in enumerate(
        zip(payloads, position_receipts, strict=True)
    ):
        payload = load.res
        payload.dtype = str(receipt["dtype"])
        payload.shape = tuple(receipt["shape"])
        payload.accounting = {
            **dict(payload.accounting or {}), "ssa_optional_payload": True,
        }
        slot = 2 * position
        for instruction in (load, gep):
            instruction.attributes["aggregate_index"] = slot
        const.attributes["value"] = slot
        presence = SSAValue(
            mint_ssa_value(
                caller, OPTIONAL_PRESENCE, (payload,),
                dtype="bool", stage=RECORD_RETURN_REPAIR,
            ),
            dtype="bool",
            accounting={
                "ssa_optional_presence": True,
                "ssa_optional_payload_id": int(payload.id),
            },
        )
        presences.append(presence)
        position_value = SSAValue(mint_ssa_value(
            caller, CONTROL_CONST, (aggregate,), dtype="int",
            stage=RECORD_RETURN_REPAIR,
        ), dtype="int")
        address = SSAValue(mint_ssa_value(
            caller, ADDRESS, (aggregate, position_value),
            dtype="ptr", stage=RECORD_RETURN_REPAIR,
        ), dtype="ptr")
        projection.extend((
            Instr("Const", [], position_value,
                  attributes={"value": slot + 1}),
            Instr("GetElementPtr", [aggregate, position_value], address,
                  attributes={"aggregate_index": slot + 1}),
            Instr("Load", [address], presence, attributes={
                "aggregate_index": slot + 1,
                "source_output_id": int(presence.id),
                "ssa_optional_result": True,
            }),
        ))
    interleaved = tuple(
        item
        for payload_presence in zip(
            (load.res for load, _g, _c in payloads), presences, strict=True,
        )
        for item in payload_presence
    )
    aggregate.dtype = "ssa.aggregate"
    aggregate.shape = (len(interleaved),)
    aggregate.accounting = {
        **dict(aggregate.accounting or {}),
        "ssa_aggregate_outputs": interleaved,
        "ssa_optional_result": True,
    }
    call.attributes.update({
        "output_ids": tuple(int(value.id) for value in interleaved),
        "callee_output_ids": tuple(
            int(item) for receipt in position_receipts
            for item in (receipt["payload_value_id"],
                         receipt["presence_value_id"])
        ),
        "native_result_contract": tuple(
            item for receipt in position_receipts for item in (
                (int(receipt["payload_value_id"]), str(receipt["dtype"]),
                 tuple(receipt["shape"])),
                (int(receipt["presence_value_id"]), "bool", ()),
            )
        ),
        "ssa_optional_result": True,
    })
    block.instrs[index + 1:index + 1] = projection
    presence = presences[0]
    payload = payloads[0][0].res
    for consumer_block in caller.blocks.values():
        for consumer in consumer_block.instrs:
            if consumer.op not in (_EQ_OPS | _NE_OPS):
                continue
            if not any(int(value.id) == int(aggregate.id)
                       for value in consumer.args):
                continue
            other = tuple(
                value for value in consumer.args
                if int(value.id) != int(aggregate.id)
            )
            if len(other) != 1:
                continue
            none_producer = producers.get(int(other[0].id))
            if none_producer is None or none_producer.op not in _NONE_OPS:
                continue
            consumer.args = [presence]
            consumer.op = "LNot" if consumer.op in _EQ_OPS else "Cast"
            consumer.attributes.update({
                "target_dtype": "bool",
                "optional_presence_test": True,
                "payload_value_id": int(payload.id),
                "presence_value_id": int(presence.id),
                "priority": "exact_optional_call_result",
                "tie_policy": "incumbent",
            })
    for position, presence_value in enumerate(presences):
        receipts.append({
            **position_receipts[position],
            "caller": str(caller_name),
            "callsite_id": call.attributes.get("plan_callsite_id"),
            "caller_payload_value_id": int(payloads[position][0].res.id),
            "caller_presence_value_id": int(presence_value.id),
            "aggregate_value_id": int(aggregate.id),
        })
    return index + 1 + len(projection)


def lower_optional_scalar_returns(module: Any) -> tuple[dict[str, Any], ...]:
    """Split every proven scalar-or-None return into payload and presence.

    The semantic result identity remains the payload id. Its physical return
    layout becomes ``(payload, presence)``. Linked callers receive the pair as
    an ordinary aggregate and exact identity comparisons with ``None`` read the
    Boolean presence result. Unsupported or ambiguous shapes remain unchanged,
    so the existing optional-merge checker continues to reject them.
    """

    functions = _functions(module)
    contracts: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    from .concordance_declarations import (
        ADDRESS, CONTROL_CONST, OPTIONAL_AGGREGATE,
        OPTIONAL_INACTIVE_PAYLOAD, OPTIONAL_PRESENCE, RECORD_RETURN_REPAIR,
        OPTIONAL_VALUE_STRUCTURE, STRUCTURAL_CONCORDANCE,
    )
    from .identity_concordance import current_identity_book, Derived, Mode
    from .ssa_record_return_state import (
        accepted_ssa_component,
        mint_ssa_value, function_scope_of, identity_cells, revise_ssa_value,
    )
    # Program: bootstrap unit 35 (IdentityPage._virtual_fact) composing the
    # accepted IdentityPage._resolved (unit 33).  Wrong record: this pass
    # lowered _resolved's record-covered call to _fork_of_scope a second
    # time; its output_ids were already the interleaved (payload, presence)
    # pairs, so 4 projections met 2 receipts.  Rule: an accepted component
    # is an immutable source -- no loop here writes into one.
    from .structural_concordance import (
        aggregate_return_layouts, publish_aggregate_return_layout,
        optional_structure,
    )

    # Call-frame and returned-record reconciliation can replace an early
    # ProgramABI argument object with another occurrence of the same settled
    # SSA identity.  The record descriptor is the final physical owner, so
    # restamp its optional payload/presence pair on every surviving local
    # occurrence before optional controls and native signatures are emitted.
    # This is identity concordance, not inference: ambiguous/missing pairs are
    # left untouched.
    record_presence_receipts: list[dict[str, Any]] = []
    record_tables = getattr(module, "record_tables", {}) or {}
    from .concordance_declarations import SSA_VALUE
    from .ssa_record_return_state import (
        record_descriptor_cell, record_member_cell,
    )
    for function_name, function in functions.items():
        if accepted_ssa_component(function):
            continue  # immutable source; its unit lowered it
        table = record_tables.get(function_name)
        if table is None:
            continue
        occurrences: dict[int, list[SSAValue]] = {}
        for value in (
            *function.args,
            *(
                item
                for block in function.blocks.values()
                for instruction in block.instrs
                for item in (
                    *instruction.args,
                    *((instruction.res,) if instruction.res is not None else ()),
                )
            ),
        ):
            bucket = occurrences.setdefault(int(value.id), [])
            if not any(value is incumbent for incumbent in bucket):
                bucket.append(value)
        for record_id, record in table.records.items():
            fields = {field.name: field for field in record.fields}
            for name, presence_field in fields.items():
                if not name.endswith(".__present"):
                    continue
                payload_name = name.removesuffix(".__present")
                payload_field = fields.get(payload_name)
                if (
                    payload_field is None
                    or len(payload_field.value_ids) != 1
                    or len(presence_field.value_ids) != 1
                    or str(presence_field.dtype or "").casefold() != "bool"
                ):
                    continue
                payload_id = int(payload_field.value_ids[0])
                presence_id = int(presence_field.value_ids[0])
                payload_values = occurrences.get(payload_id, ())
                presence_values = occurrences.get(presence_id, ())
                if not payload_values or not presence_values:
                    continue
                # Rule: post first; accounting is a view.  The pair's physical
                # records are revised onto their ``ssa_value`` rows DERIVED
                # from the record descriptor's cell and each member's own
                # ``record_member`` cell; the accounting stamped below is set
                # FROM the posted facts.
                _pair_sources = (
                    record_descriptor_cell(table, record_id),
                    record_member_cell(table, payload_id),
                    record_member_cell(table, presence_id),
                )
                _payload_cell = revise_ssa_value(
                    function, payload_id, dtype=payload_values[0].dtype,
                    shape=payload_values[0].shape,
                    stage=STRUCTURAL_CONCORDANCE, sources=_pair_sources,
                    physical={
                        "storage": "scalar", "arena": False,
                        "device": payload_values[0].device,
                    },
                )
                _presence_cell = revise_ssa_value(
                    function, presence_id, dtype="bool", shape=(),
                    stage=STRUCTURAL_CONCORDANCE, sources=_pair_sources,
                    physical={
                        "physical_dtype": "bool", "storage": "scalar",
                        "rank": 0, "arena": False,
                        "device": presence_values[0].device,
                    },
                )
                from .frame_identity_book import (
                    _frame_accounting_from_posted, _frame_physical_fields,
                )

                for payload in payload_values:
                    accounting = dict(payload.accounting or {})
                    leased_from = accounting.pop(
                        "linked_call_frame_storage", None,
                    )
                    payload.accounting = {
                        **accounting,
                        "program_abi_record": str(record.identity),
                        "program_abi_field": str(payload_name),
                        "program_abi_optional_payload": True,
                        "ssa_optional_presence_id": presence_id,
                        **({
                            "record_descriptor_reconciled_frame_storage": (
                                str(leased_from)
                            ),
                            "record_descriptor_storage_priority": (
                                "exact_record_descriptor"
                            ),
                        } if leased_from is not None else {}),
                    }
                    payload.accounting = _frame_accounting_from_posted(
                        payload.accounting,
                        _frame_physical_fields(function, payload_id),
                    )
                for presence in presence_values:
                    if _presence_cell is not None:
                        # The row was revised to the scalar Boolean above;
                        # the occurrence follows it.
                        presence.dtype = "bool"
                        presence.shape = ()
                    accounting = dict(presence.accounting or {})
                    leased_from = accounting.pop(
                        "linked_call_frame_storage", None,
                    )
                    accounting = {
                        **accounting,
                        "program_abi_record": str(record.identity),
                        "program_abi_field": str(payload_name),
                        "program_abi_optional_presence": True,
                        "program_abi_optional_present_when": True,
                        "ssa_optional_payload_id": payload_id,
                        **({
                            "record_descriptor_reconciled_frame_storage": (
                                str(leased_from)
                            ),
                            "record_descriptor_storage_priority": (
                                "exact_record_descriptor"
                            ),
                        } if leased_from is not None else {}),
                    }
                    # The descriptor confirms the physical Boolean type, but
                    # an earlier ProgramABI lowering is the stronger origin
                    # receipt.  Equal facts keep their incumbent provenance.
                    accounting.setdefault(
                        "physical_dtype_provenance",
                        "record_descriptor_optional_presence",
                    )
                    accounting.setdefault(
                        "physical_dtype_tie_policy", "incumbent",
                    )
                    presence.accounting = _frame_accounting_from_posted(
                        accounting,
                        _frame_physical_fields(function, presence_id),
                    )
                record_presence_receipts.append({
                    "function": str(function_name),
                    "record": str(record.identity),
                    "field": str(payload_name),
                    "payload_value_id": payload_id,
                    "presence_value_id": presence_id,
                    "priority": "exact_record_descriptor",
                    "tie_policy": "incumbent",
                })

    if record_presence_receipts:
        metadata = getattr(module, "metadata", None)
        if metadata is None:
            module.metadata = {}
            metadata = module.metadata
        metadata["optional_record_presence_concordance"] = tuple(
            record_presence_receipts
        )

    for function_name, function in functions.items():
        if accepted_ssa_component(function):
            continue  # immutable source; its unit lowered it
        producers = {
            int(instruction.res.id): instruction
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        }
        presence_by_payload: dict[int, SSAValue] = {}
        # Linked record fields have an explicit physical presence member.
        # Read the function-local descriptor, not copied accounting IDs from
        # the callee's namespace, when lowering tests of a returned field.
        record_table = (getattr(module, "record_tables", {}) or {}).get(function_name)
        if record_table is not None:
            values = {int(value.id): value for value in function.args}
            values.update({value_id: instruction.res for value_id, instruction in producers.items()})
            candidates: dict[int, set[int]] = {}
            for record in record_table.records.values():
                fields = {field.name: field for field in record.fields}
                for name, field in fields.items():
                    presence_field = fields.get(f"{name}.__present")
                    if (len(field.value_ids) == 1 and presence_field is not None
                            and len(presence_field.value_ids) == 1):
                        candidates.setdefault(int(field.value_ids[0]), set()).add(
                            int(presence_field.value_ids[0])
                        )
            for payload_id, presence_ids in candidates.items():
                if len(presence_ids) != 1:
                    continue
                presence_id = next(iter(presence_ids))
                if presence_id in values and values[presence_id].dtype == "bool":
                    presence_by_payload[payload_id] = values[presence_id]
        from .ssa_reachability import semantic_output_ids_of

        returned_ids = {
            int(argument.id)
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op in _RETURN_OPS
            for argument in instruction.args
        }
        lowered_one = True
        while lowered_one:
            lowered_one = False
            for _block_name, block in function.blocks.items():
                for instruction in tuple(block.instrs):
                    if (
                        instruction.op not in _PHI_OPS
                        or instruction.res is None
                        or int(instruction.res.id) in presence_by_payload
                        or str(instruction.attributes.get("binding") or "")
                        in {"optional_value_presence", "optional_return_presence"}
                    ):
                        continue
                    absent = tuple(
                        str(value.dtype or "").casefold() == "none"
                        or (
                            int(value.id) in producers
                            and producers[int(value.id)].op in _NONE_OPS
                        )
                        for value in instruction.args
                    )
                    if not any(absent) or all(absent):
                        continue
                    payloads = tuple(
                        value for value, is_absent in zip(
                            instruction.args, absent, strict=True,
                        ) if not is_absent
                    )
                    signatures = {
                        (str(value.dtype or ""), tuple(value.shape or ()), value.device)
                        for value in payloads
                    }
                    # Scalar optionals have one physical payload shape. Span
                    # and record optionals also require ownership and remain
                    # visible to the existing hard gate.
                    if len(signatures) != 1 or any(value.shape for value in payloads):
                        continue
                    dtype, shape, device = next(iter(signatures))
                    if not dtype or dtype.casefold() in {"none", "unknown"}:
                        continue
                    incoming_blocks = tuple(instruction.attributes.get(
                        "incoming_blocks", (),
                    ))
                    if len(incoming_blocks) != len(instruction.args):
                        continue

                    payload_args: list[SSAValue] = []
                    presence_args: list[SSAValue] = []
                    for incoming, value, is_absent in zip(
                        incoming_blocks, instruction.args, absent, strict=True,
                    ):
                        predecessor = function.blocks.get(str(incoming))
                        if predecessor is None:
                            break
                        edge_instructions: list[Instr] = []
                        if is_absent:
                            payload = SSAValue(
                                mint_ssa_value(
                                    function, OPTIONAL_INACTIVE_PAYLOAD,
                                    (instruction.res, value), dtype=dtype,
                                    shape=shape, stage=RECORD_RETURN_REPAIR,
                                ),
                                dtype=dtype,
                                shape=shape,
                                device=device,
                                accounting={"optional_inactive_payload": True},
                            )
                            edge_instructions.append(Instr(
                                "Const", [], payload,
                                attributes={
                                    "value": (
                                        False if dtype.casefold() == "bool" else 0
                                    ),
                                    "optional_inactive_payload": True,
                                },
                            ))
                        else:
                            payload = value
                        present = presence_by_payload.get(int(value.id))
                        if present is None:
                            present = SSAValue(
                                mint_ssa_value(
                                    function, OPTIONAL_PRESENCE,
                                    (instruction.res, value, payload),
                                    dtype="bool", stage=RECORD_RETURN_REPAIR,
                                ),
                                dtype="bool",
                                accounting={"ssa_optional_presence_edge": True},
                            )
                            edge_instructions.append(Instr(
                                "Const", [], present,
                                attributes={
                                    "value": not is_absent,
                                    "ssa_optional_presence_edge": True,
                                },
                            ))
                        _insert_before_terminator(
                            predecessor, edge_instructions
                        )
                        payload_args.append(payload)
                        presence_args.append(present)
                    else:
                        # Program: ``if flag > 0: r = row + 1 else: r = None;
                        # return r`` (and the bootstrap's single-exit guard
                        # normalization, which writes exactly that).  Wrong
                        # record: the merge's binding was
                        # ``conditional_result``, so the Phi was split but the
                        # Ret kept only the payload -- the export said "int"
                        # where Python returns None.  Rule: a Phi is a return
                        # merge when it is this function's semantic output and
                        # its own Ret returns it, whatever binding built it.
                        is_return = str(instruction.attributes.get(
                            "binding"
                        ) or "") == "return_merge" or (
                            int(instruction.res.id) in returned_ids
                            and int(instruction.res.id) in map(
                                int, semantic_output_ids_of(function),
                            )
                        )
                        presence = SSAValue(
                            mint_ssa_value(
                                function, OPTIONAL_PRESENCE,
                                (instruction.res, *presence_args),
                                dtype="bool", stage=RECORD_RETURN_REPAIR,
                            ),
                            dtype="bool",
                            accounting={
                                "ssa_optional_presence": True,
                                "ssa_optional_payload_id": int(instruction.res.id),
                            },
                        )
                        instruction.res.accounting = {
                            **dict(instruction.res.accounting or {}),
                            "ssa_optional_payload": True,
                            "ssa_optional_presence_id": int(presence.id),
                        }
                        presence_phi = Instr(
                            "Phi", presence_args, presence,
                            attributes={
                                "incoming_blocks": incoming_blocks,
                                "binding": (
                                    "optional_return_presence" if is_return
                                    else "optional_value_presence"
                                ),
                                "payload_value_id": int(instruction.res.id),
                            },
                        )
                        live_index = block.instrs.index(instruction)
                        block.instrs.insert(live_index + 1, presence_phi)
                        presence_by_payload[int(instruction.res.id)] = presence
                        structure_book = current_identity_book()
                        structure_scope = function_scope_of(function)
                        revise_ssa_value(
                            function, int(instruction.res.id), dtype=dtype, shape=shape,
                            stage=STRUCTURAL_CONCORDANCE,
                            sources=identity_cells(function, *payload_args, presence),
                            book=structure_book,
                        )
                        structure_book.post(
                            OPTIONAL_VALUE_STRUCTURE,
                            (structure_scope, int(instruction.res.id)),
                            (int(instruction.res.id), int(presence.id)),
                            stage=STRUCTURAL_CONCORDANCE,
                            provenance=Derived(identity_cells(function, instruction.res, presence)),
                            mode=Mode.CONCORD,
                        )
                        instruction.args = payload_args
                        instruction.res.dtype = dtype
                        instruction.res.shape = shape
                        instruction.res.device = device
                        receipt = {
                            "function": str(function_name),
                            "payload_value_id": int(instruction.res.id),
                            "presence_value_id": int(presence.id),
                            "dtype": dtype,
                            "shape": shape,
                            "priority": (
                                "exact_mixed_return_phi" if is_return
                                else "exact_mixed_scalar_phi"
                            ),
                            "tie_policy": "incumbent",
                        }
                        receipts.append(receipt)
                        if is_return:
                            layouts = aggregate_return_layouts(function)
                            if "return_arm_absent" in instruction.attributes:
                                # A ``return_output_arms`` member Phi: the
                                # semantic output's own layout entry names
                                # this payload as one leaf; its presence
                                # follows it, so N positions give
                                # ``(p0, q0, p1, q1, ...)``.  No per-Phi
                                # layout and no whole-function scalar
                                # contract: the contract is per output and
                                # per position.
                                payload_id = int(instruction.res.id)
                                for output_id, leaves in tuple(
                                    layouts.items()
                                ):
                                    if payload_id not in map(int, leaves):
                                        continue
                                    layouts[output_id] = tuple(
                                        item
                                        for leaf in map(int, leaves)
                                        for item in (
                                            (leaf, int(presence.id))
                                            if leaf == payload_id else (leaf,)
                                        )
                                    )
                                    receipt = {
                                        **receipt,
                                        "return_output_id": int(output_id),
                                        "return_position": int(
                                            instruction.attributes[
                                                "return_slot_index"
                                            ]
                                        ),
                                    }
                                function.metadata[
                                    "optional_return_contracts"
                                ] = (
                                    *tuple(function.metadata.get(
                                        "optional_return_contracts", (),
                                    )),
                                    receipt,
                                )
                            else:
                                layouts[int(instruction.res.id)] = (
                                    int(instruction.res.id), int(presence.id),
                                )
                                contracts[str(function_name)] = receipt
                            optional_cell = structure_book.current_ref(
                                OPTIONAL_VALUE_STRUCTURE,
                                (structure_scope, int(instruction.res.id)),
                            )
                            for output_id, layout in layouts.items():
                                if int(instruction.res.id) in layout:
                                    publish_aggregate_return_layout(
                                        function, output_id, layout,
                                        sources=(optional_cell,), stage=STRUCTURAL_CONCORDANCE,
                                        book=structure_book,
                                    )
                            for candidate_block in function.blocks.values():
                                for terminator in candidate_block.instrs:
                                    if terminator.op not in _RETURN_OPS:
                                        continue
                                    terminator.args = [
                                        item
                                        for value in terminator.args
                                        for item in (
                                            (instruction.res, presence)
                                            if int(value.id) == int(instruction.res.id)
                                            else (value,)
                                        )
                                    ]
                        lowered_one = True
                        break
                if lowered_one:
                    break

        # Identity tests of a local optional Phi consume its local presence
        # value just like tests of an optional Call result do below.
        for block in function.blocks.values():
            for consumer in block.instrs:
                if consumer.op not in (_EQ_OPS | _NE_OPS):
                    continue
                payload = next((
                    value for value in consumer.args
                    if int(value.id) in presence_by_payload
                ), None)
                if payload is None:
                    continue
                other = tuple(
                    value for value in consumer.args
                    if value is not payload
                )
                if len(other) != 1:
                    continue
                none_producer = producers.get(int(other[0].id))
                if none_producer is None or none_producer.op not in _NONE_OPS:
                    continue
                presence = presence_by_payload[int(payload.id)]
                consumer.args = [presence]
                consumer.op = "LNot" if consumer.op in _EQ_OPS else "Cast"
                consumer.attributes.update({
                    "target_dtype": "bool",
                    "optional_presence_test": True,
                    "payload_value_id": int(payload.id),
                    "presence_value_id": int(presence.id),
                    "priority": "exact_optional_phi",
                    "tie_policy": "incumbent",
                })

    # Close optional contracts through exact forwarding functions before
    # rewriting callsites.  A wrapper such as ``return maybe(...)`` has no
    # local mixed Phi, but it does not erase the callee's presence bit.  Its
    # physical return is the same payload/presence pair under wrapper-local
    # ids.  Without this closure, a caller's ``is not None`` sees the absent
    # payload (zero) as an ordinary scalar and can fold the test to true.
    forwarded_presence: dict[tuple[str, int], SSAValue] = {}
    forwarding_changed = True
    while forwarding_changed:
        forwarding_changed = False
        for function_name, function in functions.items():
            if accepted_ssa_component(function):
                continue  # immutable source; its unit lowered it
            if str(function_name) in contracts:
                continue
            producers = {
                int(instruction.res.id): instruction
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.res is not None
            }
            returns = tuple(
                instruction
                for block in function.blocks.values()
                for instruction in block.instrs
                if instruction.op in _RETURN_OPS
            )
            if not returns or any(len(ret.args) != 1 for ret in returns):
                continue
            payload_ids = {int(ret.args[0].id) for ret in returns}
            if len(payload_ids) != 1:
                continue
            payload_id = next(iter(payload_ids))
            producer = producers.get(payload_id)
            if producer is None or producer.op not in {"Call", "call"}:
                continue
            callee_contract = contracts.get(str(
                producer.attributes.get("callee") or ""
            ))
            if callee_contract is None:
                continue
            callee = functions[str(producer.attributes.get("callee") or "")]
            from .structural_concordance import function_optional_structure
            from .ssa_record_return_state import publish_ssa_value_type

            callee_structure, callee_cell = function_optional_structure(
                callee, int(callee_contract["payload_value_id"]),
            )
            payload_structure = dict(callee_structure.children)["payload"]
            payload = returns[0].args[0]
            publish_ssa_value_type(
                function, payload, dtype=payload_structure.dtype,
                shape=payload_structure.shape, stage=STRUCTURAL_CONCORDANCE,
                sources=(callee_cell,),
            )
            payload.accounting = {
                **dict(payload.accounting or {}),
                "ssa_optional_payload": True,
            }
            presence = SSAValue(
                mint_ssa_value(
                    function, OPTIONAL_PRESENCE, (payload,),
                    dtype="bool", stage=RECORD_RETURN_REPAIR,
                ),
                dtype="bool",
                accounting={
                    "ssa_optional_presence": True,
                    "ssa_optional_payload_id": payload_id,
                    "ssa_optional_forwarded_return": True,
                },
            )
            structure_book = current_identity_book()
            optional_cell = structure_book.post(
                OPTIONAL_VALUE_STRUCTURE, (function_scope_of(function), payload_id),
                (payload_id, int(presence.id)), stage=STRUCTURAL_CONCORDANCE,
                provenance=Derived((callee_cell, *identity_cells(function, payload, presence))),
                mode=Mode.CONCORD,
            )
            publish_aggregate_return_layout(
                function, payload_id, (payload_id, int(presence.id)),
                sources=(optional_cell,), stage=STRUCTURAL_CONCORDANCE,
                book=structure_book,
            )
            for ret in returns:
                ret.args = [ret.args[0], presence]
            receipt = {
                "function": str(function_name),
                "payload_value_id": payload_id,
                "presence_value_id": int(presence.id),
                "dtype": payload_structure.dtype,
                "shape": payload_structure.shape,
                "priority": "exact_forwarded_optional_return",
                "tie_policy": "incumbent",
                "forwarded_callee": str(
                    producer.attributes.get("callee") or ""
                ),
            }
            contracts[str(function_name)] = receipt
            forwarded_presence[(str(function_name), payload_id)] = presence
            receipts.append(receipt)
            forwarding_changed = True

    for caller_name, caller in functions.items():
        if accepted_ssa_component(caller):
            continue  # immutable source; its unit lowered it
        producers = {
            int(instruction.res.id): instruction
            for block in caller.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        }
        for block in caller.blocks.values():
            index = 0
            while index < len(block.instrs):
                call = block.instrs[index]
                contract = contracts.get(str(call.attributes.get("callee") or ""))
                if (
                    contract is None
                    and call.op in {"Call", "call"}
                    and call.res is not None
                ):
                    callee_function = functions.get(
                        str(call.attributes.get("callee") or "")
                    )
                    position_receipts = (
                        () if callee_function is None
                        else _record_covered_return_receipts(callee_function)
                    )
                    if position_receipts:
                        index = _lower_record_covered_call(
                            caller_name, caller, block, index, call,
                            position_receipts, producers, receipts,
                        )
                        continue
                if (
                    call.op not in {"Call", "call"}
                    or call.res is None
                    or contract is None
                ):
                    index += 1
                    continue
                payload = call.res
                payload.dtype = str(contract["dtype"])
                payload.shape = tuple(contract["shape"])
                payload.accounting = {
                    **dict(payload.accounting or {}),
                    "ssa_optional_payload": True,
                }
                presence = forwarded_presence.get(
                    (str(caller_name), int(payload.id))
                )
                if presence is None:
                    presence = SSAValue(
                        mint_ssa_value(
                            caller, OPTIONAL_PRESENCE, (payload,),
                            dtype="bool", stage=RECORD_RETURN_REPAIR,
                        ),
                        dtype="bool",
                        accounting={
                            "ssa_optional_presence": True,
                            "ssa_optional_payload_id": int(payload.id),
                        },
                    )
                aggregate = SSAValue(
                    mint_ssa_value(
                        caller, OPTIONAL_AGGREGATE, (payload, presence),
                        dtype="ssa.aggregate", shape=(2,),
                        stage=RECORD_RETURN_REPAIR,
                    ),
                    dtype="ssa.aggregate",
                    shape=(2,),
                    accounting={
                        "ssa_aggregate_outputs": (payload, presence),
                        "ssa_optional_result": True,
                    },
                )
                projection: list[Instr] = []
                for position, output in enumerate((payload, presence)):
                    position_value = SSAValue(mint_ssa_value(
                        caller, CONTROL_CONST, (aggregate,), dtype="int",
                        stage=RECORD_RETURN_REPAIR,
                    ), dtype="int")
                    address = SSAValue(mint_ssa_value(
                        caller, ADDRESS, (aggregate, position_value),
                        dtype="ptr", stage=RECORD_RETURN_REPAIR,
                    ), dtype="ptr")
                    projection.extend((
                        Instr("Const", [], position_value,
                              attributes={"value": position}),
                        Instr("GetElementPtr", [aggregate, position_value], address,
                              attributes={"aggregate_index": position}),
                        Instr("Load", [address], output, attributes={
                            "aggregate_index": position,
                            "source_output_id": int(output.id),
                            "ssa_optional_result": True,
                        }),
                    ))
                call.res = aggregate
                call.attributes.update({
                    "result_convention": "ssa.aggregate",
                    "output_ids": (int(payload.id), int(presence.id)),
                    "callee_output_ids": (
                        int(contract["payload_value_id"]),
                        int(contract["presence_value_id"]),
                    ),
                    "native_result_contract": (
                        (int(contract["payload_value_id"]),
                         str(contract["dtype"]), tuple(contract["shape"])),
                        (int(contract["presence_value_id"]), "bool", ()),
                    ),
                    "ssa_optional_result": True,
                })
                block.instrs[index + 1:index + 1] = projection
                index += 1 + len(projection)

                for consumer_block in caller.blocks.values():
                    for consumer in consumer_block.instrs:
                        if consumer.op not in (_EQ_OPS | _NE_OPS):
                            continue
                        if not any(int(value.id) == int(payload.id)
                                   for value in consumer.args):
                            continue
                        other = tuple(
                            value for value in consumer.args
                            if int(value.id) != int(payload.id)
                        )
                        if len(other) != 1:
                            continue
                        none_producer = producers.get(int(other[0].id))
                        if none_producer is None or none_producer.op not in _NONE_OPS:
                            continue
                        consumer.args = [presence]
                        consumer.op = (
                            "LNot" if consumer.op in _EQ_OPS else "Cast"
                        )
                        consumer.attributes.update({
                            "target_dtype": "bool",
                            "optional_presence_test": True,
                            "payload_value_id": int(payload.id),
                            "presence_value_id": int(presence.id),
                            "priority": "exact_optional_call_result",
                            "tie_policy": "incumbent",
                        })
                receipts.append({
                    **contract,
                    "caller": str(caller_name),
                    "callsite_id": call.attributes.get("plan_callsite_id"),
                    "caller_payload_value_id": int(payload.id),
                    "caller_presence_value_id": int(presence.id),
                    "aggregate_value_id": int(aggregate.id),
                })
                index += 1

    if receipts:
        metadata = getattr(module, "metadata", None)
        if metadata is None:
            module.metadata = {}
            metadata = module.metadata
        metadata["optional_scalar_return_receipts"] = tuple(receipts)
    return tuple(receipts)


__all__ = ["lower_optional_scalar_returns"]
