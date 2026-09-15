"""Resolve an immutable scalar field version for a physical return edge."""

import networkx as nx

from .monotonic_ids import GLOBAL_MONOTONIC_IDS


def normalize_declared_scalar_record_shapes(module):
    """Enforce the physical shape promised by scalar record descriptors.

    Result-type propagation can refine a late call output after its record
    storage was first materialized.  The descriptor is the stronger physical
    ABI evidence: a field declared as scalar occupies one scalar slot even if
    the source value temporarily carries a singleton tensor shape.  Apply the
    invariant at the completed-module seam and retain the prior shape as
    provenance.  Repeated or equal evidence keeps the incumbent scalar form.
    """

    receipts = []
    for symbol, function in module.functions.items():
        table = module.record_tables.get(str(symbol))
        if table is None:
            continue
        values = {
            int(value.id): value for value in function.args
        }
        values.update({
            int(instruction.res.id): instruction.res
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.res is not None
        })
        fields_by_value = {}
        for record_id, descriptor in table.records.items():
            for field in descriptor.fields:
                if str(getattr(field.storage, "value", field.storage)) != "scalar":
                    continue
                for value_id in field.value_ids:
                    fields_by_value.setdefault(int(value_id), []).append((
                        int(record_id), str(field.name),
                        str(field.storage_identity),
                    ))
        for value_id, owners in fields_by_value.items():
            value = values.get(value_id)
            prior_shape = () if value is None else tuple(value.shape or ())
            if value is None or not prior_shape:
                continue
            value.shape = ()
            value.accounting = {
                **dict(value.accounting or {}),
                "record_scalar_shape_normalization": True,
                "record_scalar_prior_shape": prior_shape,
                "record_scalar_shape_priority": "declared_record_storage",
                "record_scalar_shape_tie_policy": "incumbent",
            }
            receipts.append({
                "function": str(symbol),
                "value_id": int(value_id),
                "prior_shape": prior_shape,
                "record_fields": tuple(owners),
                "priority": "declared_record_storage",
                "tie_policy": "incumbent",
            })
    return tuple(receipts)


def publish_inout_scalar_return_snapshots(module):
    """Give returned snapshots the latest unambiguous in/out write version.

    An authored scalar record field may be both a callee formal and the source
    of a returned value.  Region projection lowering versions the write while
    the resident formal keeps the caller-owned address.  Returning the formal
    directly makes the native ABI treat the result as another in/out alias;
    later mutation of the resident field then also changes what should have
    been a value snapshot.  Replace that return position with the unique
    latest dominating write version.  Ambiguous maxima retain the incumbent.
    """
    from dataclasses import replace

    receipts = []
    returned_layout_updates = {}
    returned_values_by_symbol = {}
    for symbol, function in module.functions.items():
        block_names = tuple(function.blocks)
        if not block_names:
            continue
        entry = "entry" if "entry" in function.blocks else block_names[0]
        cfg = nx.DiGraph()
        cfg.add_nodes_from(block_names)
        for name, block in function.blocks.items():
            cfg.add_edges_from(
                (name, successor) for successor in block.successors
                if successor in function.blocks
            )
        dominators = nx.immediate_dominators(cfg, entry)
        dominators[entry] = entry

        def block_dominates(owner, target):
            current = target
            while current in dominators:
                if current == owner:
                    return True
                parent = dominators[current]
                if parent == current:
                    break
                current = parent
            return False

        locations = {}
        writes = {}
        for block_name, block in function.blocks.items():
            for index, instruction in enumerate(block.instrs):
                if instruction.res is None:
                    continue
                locations[int(instruction.res.id)] = (block_name, index)
                accounting = instruction.res.accounting or {}
                source_id = accounting.get("source_value_id")
                if (
                    accounting.get("ssa_inout_write_version")
                    and source_id is not None
                    and not tuple(instruction.res.shape or ())
                ):
                    writes.setdefault(int(source_id), []).append(
                        instruction.res
                    )
        formals = {
            int(value.id): value for value in function.args
            if not tuple(value.shape or ())
        }

        def dominates(left, right_block, right_index):
            owner, index = locations[int(left.id)]
            return block_dominates(owner, right_block) and (
                owner != right_block or index < right_index
            )

        def later(left, right):
            """Whether left is strictly later than right on every path."""
            left_block, left_index = locations[int(left.id)]
            right_block, right_index = locations[int(right.id)]
            return block_dominates(right_block, left_block) and (
                right_block != left_block or right_index < left_index
            )

        old_and_new_returns = []
        for block_name, block in function.blocks.items():
            for return_index, operation in enumerate(block.instrs):
                if operation.op not in {"Ret", "ret", "Return", "return"}:
                    continue
                old_arguments = tuple(operation.args)
                arguments = list(old_arguments)
                for position, incumbent in enumerate(old_arguments):
                    source_id = int(incumbent.id)
                    if source_id not in formals:
                        continue
                    candidates = [
                        value for value in writes.get(source_id, ())
                        if value.dtype == incumbent.dtype
                        and dominates(value, block_name, return_index)
                    ]
                    maxima = [
                        value for value in candidates
                        if not any(
                            value is not other and later(other, value)
                            for other in candidates
                        )
                    ]
                    if len(maxima) != 1:
                        continue
                    selected = maxima[0]
                    arguments[position] = selected
                    receipts.append({
                        "function": str(symbol),
                        "block": str(block_name),
                        "return_position": int(position),
                        "source_value_id": source_id,
                        "snapshot_value_id": int(selected.id),
                        "priority": "exact_dominating_inout_write",
                        "replaced_priority": "resident_formal",
                        "tie_policy": "incumbent",
                    })
                if arguments != list(old_arguments):
                    operation.args = arguments
                    old_and_new_returns.append((
                        tuple(int(value.id) for value in old_arguments),
                        tuple(int(value.id) for value in arguments),
                    ))
        if not old_and_new_returns:
            continue
        # Full-native source functions have one settled return layout.  If
        # several returns disagree, leave derived record metadata untouched;
        # the ordinary return merge must settle that ambiguity first.
        new_returns = {new for _old, new in old_and_new_returns}
        if len(new_returns) == 1:
            returned_values_by_symbol[str(symbol)] = next(iter(new_returns))
        layouts = []
        table = module.record_tables.get(symbol)
        for record_id, layout in function.metadata.get(
            "record_return_layouts", ()
        ):
            updated = tuple(map(int, layout))
            for old_return, new_return in old_and_new_returns:
                starts = [
                    start for start in range(len(old_return) - len(updated) + 1)
                    if old_return[start:start + len(updated)] == updated
                ]
                if len(starts) == 1:
                    start = starts[0]
                    updated = new_return[start:start + len(updated)]
            layouts.append((int(record_id), updated))
            if table is None or int(record_id) not in table.records:
                continue
            record = table.records[int(record_id)]
            cursor = 0
            fields = []
            for field in record.fields:
                width = len(field.value_ids)
                field_ids = updated[cursor:cursor + width]
                cursor += width
                fields.append(replace(field, value_ids=tuple(field_ids)))
            table.records[int(record_id)] = replace(
                record, fields=tuple(fields)
            )
        if layouts:
            function.metadata["record_return_layouts"] = tuple(layouts)
            returned_layout_updates[str(symbol)] = tuple(layouts)

    for caller in module.functions.values():
        for block in caller.blocks.values():
            for operation in block.instrs:
                if operation.op not in {"Call", "call"}:
                    continue
                callee = str((operation.attributes or {}).get("callee") or "")
                returned = returned_values_by_symbol.get(callee)
                if returned is None:
                    continue
                attributes = operation.attributes or {}
                callee_ids = tuple(map(int, attributes.get(
                    "callee_output_ids", ()
                )))
                if len(callee_ids) == len(returned):
                    attributes["callee_output_ids"] = returned
                contract = tuple(attributes.get("native_result_contract", ()))
                if len(contract) == len(returned):
                    attributes["native_result_contract"] = tuple(
                        (returned[index], *tuple(item)[1:])
                        for index, item in enumerate(contract)
                    )
    if receipts:
        module.metadata["inout_scalar_return_snapshot_receipts"] = tuple(
            receipts
        )
    return len(receipts)


def reconcile_forwarded_record_results(module):
    """Rebind identity-return records after their input frame settles.

    ``forwarded_output_bindings`` is created when a call returns only its
    formal inputs.  Later record-frame reconciliation may replace those call
    inputs with stronger physical residents.  Carry the same formal-to-actual
    mapping into the returned descriptor and dominated consumers; otherwise a
    stale synthetic output survives even though the call has no output port
    capable of defining it.
    """
    from dataclasses import replace

    receipts = []
    for symbol, function in module.functions.items():
        block_names = tuple(function.blocks)
        if not block_names:
            continue
        entry = "entry" if "entry" in function.blocks else block_names[0]
        cfg = nx.DiGraph()
        cfg.add_nodes_from(block_names)
        for name, block in function.blocks.items():
            cfg.add_edges_from(
                (name, successor) for successor in block.successors
                if successor in function.blocks
            )
        immediate = nx.immediate_dominators(cfg, entry)
        immediate[entry] = entry

        def block_dominates(owner, target):
            current = target
            while current in immediate:
                if current == owner:
                    return True
                parent = immediate[current]
                if parent == current:
                    break
                current = parent
            return False

        call_records = {
            int(record.callsite_id): record
            for record in module.call_table.get(str(symbol), ())
        }
        candidates = {}
        table = module.record_tables.get(str(symbol))
        for block_name, block in function.blocks.items():
            for instruction_index, operation in enumerate(block.instrs):
                attributes = operation.attributes or {}
                forwarded = tuple(attributes.get(
                    "forwarded_output_bindings", ()
                ))
                if operation.op not in {"Call", "call"} or not forwarded:
                    continue
                callee_ids = tuple(map(int, attributes.get(
                    "callee_input_ids", ()
                )))
                if len(callee_ids) != len(operation.args):
                    continue
                actual_by_formal = dict(zip(callee_ids, operation.args))
                replacements = {}
                updated_bindings = []
                for callee_id, incumbent_id in forwarded:
                    actual = actual_by_formal.get(int(callee_id))
                    if actual is None:
                        updated_bindings.append((
                            int(callee_id), int(incumbent_id),
                        ))
                        continue
                    updated_bindings.append((int(callee_id), int(actual.id)))
                    if int(actual.id) != int(incumbent_id):
                        replacements.setdefault(int(incumbent_id), actual)
                if not replacements:
                    continue
                attributes["forwarded_output_bindings"] = tuple(
                    updated_bindings
                )
                site = int(attributes.get("plan_callsite_id", -1))
                record = call_records.get(site)
                result_record_ids = {
                    int(caller_id)
                    for _callee_id, caller_id in (
                        () if record is None else record.result_bindings
                    )
                }
                if table is not None:
                    for record_id in result_record_ids:
                        descriptor = table.records.get(record_id)
                        if descriptor is None:
                            continue
                        fields = tuple(
                            replace(field, value_ids=tuple(
                                int(replacements.get(
                                    int(value_id), value_id
                                ).id) if int(value_id) in replacements
                                else int(value_id)
                                for value_id in field.value_ids
                            ))
                            for field in descriptor.fields
                        )
                        if fields != descriptor.fields:
                            table.records[record_id] = replace(
                                descriptor, fields=fields
                            )
                for incumbent_id, actual in replacements.items():
                    candidates.setdefault(int(incumbent_id), []).append((
                        str(block_name), int(instruction_index), actual,
                        int(site), tuple(sorted(result_record_ids)),
                    ))

        def call_precedes(candidate, use_block, use_index):
            owner, index, _actual, _site, _records = candidate
            return block_dominates(owner, use_block) and (
                owner != use_block or index < use_index
            )

        def later(left, right):
            left_block, left_index, *_ = left
            right_block, right_index, *_ = right
            return block_dominates(right_block, left_block) and (
                right_block != left_block or right_index < left_index
            )

        for block_name, block in function.blocks.items():
            for instruction_index, operation in enumerate(block.instrs):
                arguments = list(operation.args)
                for position, incumbent in enumerate(tuple(arguments)):
                    eligible = [
                        candidate
                        for candidate in candidates.get(int(incumbent.id), ())
                        if call_precedes(
                            candidate, str(block_name), int(instruction_index)
                        )
                    ]
                    maxima = [
                        candidate for candidate in eligible
                        if not any(
                            candidate is not other and later(other, candidate)
                            for other in eligible
                        )
                    ]
                    if len(maxima) != 1:
                        continue
                    owner, _index, actual, site, record_ids = maxima[0]
                    arguments[position] = actual
                    receipts.append({
                        "function": str(symbol),
                        "callsite_id": int(site),
                        "call_block": str(owner),
                        "consumer_block": str(block_name),
                        "consumer_operation": str(operation.op),
                        "consumer_position": int(position),
                        "incumbent_value_id": int(incumbent.id),
                        "forwarded_value_id": int(actual.id),
                        "result_record_ids": record_ids,
                        "priority": "exact_settled_forwarded_input",
                        "replaced_priority": "provisional_forwarded_output",
                        "tie_policy": "incumbent",
                    })
                operation.args = arguments
    if receipts:
        module.metadata["forwarded_record_result_receipts"] = tuple(receipts)
    return len(receipts)


def reconcile_conditional_phi_continuations(module):
    """Carry settled conditional values into dominated continuation uses.

    Conditional lowering can schedule a later numerical region before it has
    replaced the region's captured arm value with the Phi that joins that arm.
    Value ids are not sufficient here: independently lowered projections may
    deliberately retain the same source id.  Match the exact SSAValue object,
    select the unique latest dominating conditional Phi, and leave an
    incumbent untouched when candidates are incomparable. An explicitly owned
    loop update is a snapshot of its selected producer, not a stale capture of
    a later conditional version.
    """
    receipts = []
    for symbol, function in module.functions.items():
        block_names = tuple(function.blocks)
        if not block_names:
            continue
        entry = "entry" if "entry" in function.blocks else block_names[0]
        cfg = nx.DiGraph()
        cfg.add_nodes_from(block_names)
        for block_name, block in function.blocks.items():
            cfg.add_edges_from(
                (block_name, successor)
                for successor in block.successors
                if successor in function.blocks
            )
        reachable = set(nx.descendants(cfg, entry)) | {entry}
        if not reachable:
            continue
        immediate = nx.immediate_dominators(cfg.subgraph(reachable), entry)
        immediate[entry] = entry

        def dominates(owner, target):
            if owner not in immediate or target not in immediate:
                return False
            current = target
            while True:
                if current == owner:
                    return True
                parent = immediate[current]
                if parent == current:
                    return False
                current = parent

        # An arm is keyed by Python identity, not by source-derived value id.
        candidates = {}
        definitions = {}
        for block_name, block in function.blocks.items():
            for instruction_index, operation in enumerate(block.instrs):
                if operation.res is not None:
                    definitions.setdefault(int(operation.res.id), []).append((
                        str(block_name), int(instruction_index), operation.res,
                    ))
                attributes = operation.attributes or {}
                if (
                    operation.op != "Phi"
                    or operation.res is None
                    or attributes.get("binding") != "conditional_carried"
                ):
                    continue
                candidate = (
                    str(block_name), int(instruction_index), operation.res,
                )
                for arm in operation.args:
                    if arm is operation.res:
                        continue
                    candidates.setdefault(id(arm), []).append(candidate)

        def candidate_available(candidate, target_block, use_index, phi_edge):
            owner, instruction_index, _result = candidate
            if not dominates(owner, target_block):
                return False
            if owner != target_block or phi_edge:
                return True
            return instruction_index < use_index

        def strictly_later(left, right):
            """Return whether left is strictly later than right."""
            left_block, left_index, _ = left
            right_block, right_index, _ = right
            if left_block == right_block:
                return left_index > right_index
            return dominates(right_block, left_block)

        for block_name, block in function.blocks.items():
            for instruction_index, operation in enumerate(block.instrs):
                arguments = list(operation.args)
                incoming = tuple((operation.attributes or {}).get(
                    "incoming_blocks", ()
                )) if operation.op == "Phi" else ()
                for position, incumbent in enumerate(tuple(arguments)):
                    target_block = (
                        str(incoming[position])
                        if position < len(incoming)
                        else str(block_name)
                    )
                    phi_edge = position < len(incoming)
                    owned_definitions = definitions.get(int(incumbent.id), ())
                    if (
                        phi_edge
                        and (operation.attributes or {}).get("binding")
                        == "loop_carried"
                        and (operation.attributes or {}).get("updated_value_id")
                        == int(incumbent.id)
                        and dominates(str(block_name), target_block)
                        and len(owned_definitions) == 1
                        and owned_definitions[0][2] is incumbent
                        and candidate_available(
                            owned_definitions[0], target_block,
                            int(instruction_index), True,
                        )
                    ):
                        # Dominance proves availability, not assignment to
                        # this logical loop binding. The raw update may also
                        # feed a conditional that controls another recurrence.
                        # Retain the exact producer chosen by local lowering.
                        receipt = {
                            "function": str(symbol),
                            "consumer_block": str(block_name),
                            "consumer_value_id": int(operation.res.id),
                            "consumer_position": int(position),
                            "updated_value_id": int(incumbent.id),
                            "producer_block": owned_definitions[0][0],
                            "priority": "exact_loop_carried_update",
                            "tie_policy": "incumbent",
                        }
                        prior = tuple(function.metadata.get(
                            "retained_loop_update_receipts", (),
                        ))
                        if receipt not in prior:
                            function.metadata["retained_loop_update_receipts"] = (
                                *prior, receipt,
                            )
                        continue
                    current = incumbent
                    chain = []
                    seen = {id(current)}
                    while True:
                        eligible = [
                            candidate
                            for candidate in candidates.get(id(current), ())
                            if candidate_available(
                                candidate,
                                target_block,
                                int(instruction_index),
                                phi_edge,
                            )
                        ]
                        maxima = [
                            candidate for candidate in eligible
                            if not any(
                                candidate is not other
                                and strictly_later(other, candidate)
                                for other in eligible
                            )
                        ]
                        if len(maxima) != 1:
                            break
                        selected = maxima[0]
                        replacement = selected[2]
                        if id(replacement) in seen:
                            break
                        chain.append(selected)
                        seen.add(id(replacement))
                        current = replacement
                    if current is incumbent:
                        continue
                    arguments[position] = current
                    receipts.append({
                        "function": str(symbol),
                        "consumer_block": str(block_name),
                        "consumer_operation": str(operation.op),
                        "consumer_position": int(position),
                        "incumbent_value_id": int(incumbent.id),
                        "continued_value_id": int(current.id),
                        "phi_blocks": tuple(item[0] for item in chain),
                        "priority": "unique_dominating_conditional_phi",
                        "tie_policy": "incumbent",
                    })
                operation.args = arguments
    if receipts:
        module.metadata["conditional_phi_continuation_receipts"] = tuple(
            receipts
        )
    return len(receipts)


def freshen_redefined_ssa_objects(module):
    """Give every repeated definition of one SSAValue object a fresh value.

    Some conditional construction paths reuse the exact carried-value object
    as a later branch projection result.  The two definitions then cannot be
    distinguished even by object-aware consumers.  Keep the first definition
    as the incumbent, clone each later definition, and rebind precisely the
    uses dominated by that later definition (including individual Phi edges).
    """
    from dataclasses import replace

    receipts = []
    for symbol, function in module.functions.items():
        block_names = tuple(function.blocks)
        if not block_names:
            continue
        entry = "entry" if "entry" in function.blocks else block_names[0]
        cfg = nx.DiGraph()
        cfg.add_nodes_from(block_names)
        for block_name, block in function.blocks.items():
            cfg.add_edges_from(
                (block_name, successor)
                for successor in block.successors
                if successor in function.blocks
            )
        reachable = set(nx.descendants(cfg, entry)) | {entry}
        immediate = nx.immediate_dominators(cfg.subgraph(reachable), entry)
        immediate[entry] = entry

        def dominates(owner, target):
            if owner not in immediate or target not in immediate:
                return False
            current = target
            while True:
                if current == owner:
                    return True
                parent = immediate[current]
                if parent == current:
                    return False
                current = parent

        definitions = []
        for block_name, block in function.blocks.items():
            for instruction_index, operation in enumerate(block.instrs):
                if operation.res is not None:
                    definitions.append((
                        str(block_name), int(instruction_index), operation,
                    ))
        first_by_object = {}
        for owner, definition_index, definition in definitions:
            original = definition.res
            object_key = id(original)
            incumbent = first_by_object.setdefault(
                object_key, (owner, definition_index, definition)
            )
            if incumbent[2] is definition:
                continue
            accounting = dict(original.accounting or {})
            accounting.update({
                "ssa_redefinition_freshened": True,
                "source_value_id": int(original.id),
            })
            fresh = replace(
                original,
                id=GLOBAL_MONOTONIC_IDS.mint(),
                accounting=accounting,
            )
            definition.res = fresh

            for use_block, block in function.blocks.items():
                for use_index, operation in enumerate(block.instrs):
                    if operation is definition:
                        continue
                    incoming = tuple((operation.attributes or {}).get(
                        "incoming_blocks", ()
                    )) if operation.op == "Phi" else ()
                    arguments = list(operation.args)
                    for position, argument in enumerate(tuple(arguments)):
                        if argument is not original:
                            continue
                        target = (
                            str(incoming[position])
                            if position < len(incoming)
                            else str(use_block)
                        )
                        phi_edge = position < len(incoming)
                        if not dominates(owner, target):
                            continue
                        if (
                            owner == target
                            and not phi_edge
                            and definition_index >= use_index
                        ):
                            continue
                        arguments[position] = fresh
                    operation.args = arguments
            receipts.append({
                "function": str(symbol),
                "definition_block": str(owner),
                "definition_operation": str(definition.op),
                "incumbent_definition_block": str(incumbent[0]),
                "source_value_id": int(original.id),
                "fresh_value_id": int(fresh.id),
                "priority": "later_definition_requires_unique_identity",
                "tie_policy": "incumbent_first_definition",
            })
    if receipts:
        module.metadata["redefined_ssa_object_receipts"] = tuple(receipts)
    return len(receipts)


def reconcile_nondominating_identity_cast_results(module):
    """Replace escaped branch-local no-op cast results with their call input.

    A planned region may return a schema-normalizing ``Cast`` whose input and
    result already have the same physical type.  If a correlated later guard
    reuses that projected result, ordinary SSA dominance cannot see the source
    correlation.  Prove the returned slot from the callee body, trace the exact
    caller aggregate projection, and substitute the dominating actual input
    only at uses the projection itself cannot dominate.
    """
    receipts = []

    def physical_identity(left, right):
        return (
            str(left.dtype) == str(right.dtype)
            and tuple(left.shape or ()) == tuple(right.shape or ())
            and getattr(left, "device", None) == getattr(right, "device", None)
        )

    # callee -> returned slot -> formal argument position
    identities = {}
    for callee_name, callee in module.functions.items():
        formal_positions = {id(value): index for index, value in enumerate(callee.args)}
        definitions = {}
        returns = []
        for block in callee.blocks.values():
            for operation in block.instrs:
                if operation.res is not None:
                    definitions.setdefault(id(operation.res), []).append(operation)
                if operation.op == "Ret":
                    returns.append(operation)
        if not returns:
            continue
        width = len(returns[0].args)
        if any(len(operation.args) != width for operation in returns):
            continue
        proven = {}
        for slot in range(width):
            positions = []
            for operation in returns:
                returned = operation.args[slot]
                defining = definitions.get(id(returned), ())
                if len(defining) != 1:
                    positions = []
                    break
                cast = defining[0]
                if cast.op.casefold() != "cast" or len(cast.args) != 1:
                    positions = []
                    break
                source = cast.args[0]
                position = formal_positions.get(id(source))
                if position is None or not physical_identity(source, returned):
                    positions = []
                    break
                positions.append(position)
            if positions and len(set(positions)) == 1:
                proven[slot] = positions[0]
        if proven:
            identities[str(callee_name)] = proven

    for symbol, function in module.functions.items():
        block_names = tuple(function.blocks)
        if not block_names:
            continue
        entry = "entry" if "entry" in function.blocks else block_names[0]
        cfg = nx.DiGraph()
        cfg.add_nodes_from(block_names)
        for block_name, block in function.blocks.items():
            cfg.add_edges_from(
                (block_name, successor)
                for successor in block.successors
                if successor in function.blocks
            )
        reachable = set(nx.descendants(cfg, entry)) | {entry}
        immediate = nx.immediate_dominators(cfg.subgraph(reachable), entry)
        immediate[entry] = entry

        def block_dominates(owner, target):
            if owner not in immediate or target not in immediate:
                return False
            current = target
            while True:
                if current == owner:
                    return True
                parent = immediate[current]
                if parent == current:
                    return False
                current = parent

        definitions = {}
        for block_name, block in function.blocks.items():
            for instruction_index, operation in enumerate(block.instrs):
                if operation.res is not None:
                    definitions.setdefault(id(operation.res), []).append((
                        str(block_name), int(instruction_index), operation,
                    ))
        formals = {id(value) for value in function.args}

        def value_dominates(value, target_block, use_index, phi_edge):
            if id(value) in formals:
                return target_block in reachable
            owners = definitions.get(id(value), ())
            if len(owners) != 1:
                return False
            owner, definition_index, _operation = owners[0]
            if not block_dominates(owner, target_block):
                return False
            if owner != target_block or phi_edge:
                return True
            return definition_index < use_index

        projections = {}
        for call_block, block in function.blocks.items():
            for call_index, call in enumerate(block.instrs):
                if call.op not in {"Call", "call"} or call.res is None:
                    continue
                attributes = call.attributes or {}
                callee_name = str(attributes.get("callee") or "")
                slots = identities.get(callee_name)
                output_ids = tuple(map(int, attributes.get("output_ids", ())))
                if not slots or not output_ids:
                    continue
                for slot, formal_position in slots.items():
                    if slot >= len(output_ids) or formal_position >= len(call.args):
                        continue
                    output_id = output_ids[slot]
                    actual = call.args[formal_position]
                    for _load_block, _load_index, load in (
                        item
                        for items in definitions.values()
                        for item in items
                    ):
                        load_attributes = load.attributes or {}
                        if (
                            load.op != "Load"
                            or load.res is None
                            or int(load_attributes.get("source_output_id", -1))
                            != output_id
                            or len(load.args) != 1
                        ):
                            continue
                        pointer_definitions = definitions.get(id(load.args[0]), ())
                        if len(pointer_definitions) != 1:
                            continue
                        pointer = pointer_definitions[0][2]
                        if (
                            pointer.op != "GetElementPtr"
                            or not pointer.args
                            or pointer.args[0] is not call.res
                        ):
                            continue
                        projections.setdefault(id(load.res), []).append((
                            load.res, actual, str(call_block), int(call_index),
                            callee_name, int(slot),
                        ))

        for block_name, block in function.blocks.items():
            for instruction_index, operation in enumerate(block.instrs):
                arguments = list(operation.args)
                incoming = tuple((operation.attributes or {}).get(
                    "incoming_blocks", ()
                )) if operation.op == "Phi" else ()
                for position, incumbent in enumerate(tuple(arguments)):
                    candidates = projections.get(id(incumbent), ())
                    if len(candidates) != 1:
                        continue
                    projected, actual, call_block, _call_index, callee, slot = (
                        candidates[0]
                    )
                    target_block = (
                        str(incoming[position])
                        if position < len(incoming)
                        else str(block_name)
                    )
                    phi_edge = position < len(incoming)
                    if value_dominates(
                        projected, target_block, int(instruction_index), phi_edge
                    ):
                        continue
                    if not value_dominates(
                        actual, target_block, int(instruction_index), phi_edge
                    ):
                        continue
                    arguments[position] = actual
                    receipts.append({
                        "function": str(symbol),
                        "consumer_block": str(block_name),
                        "consumer_operation": str(operation.op),
                        "consumer_position": int(position),
                        "callee": str(callee),
                        "return_slot": int(slot),
                        "incumbent_value_id": int(incumbent.id),
                        "forwarded_value_id": int(actual.id),
                        "call_block": str(call_block),
                        "priority": "exact_physical_identity_cast_result",
                        "tie_policy": "incumbent",
                    })
                operation.args = arguments
    if receipts:
        module.metadata["identity_cast_result_receipts"] = tuple(receipts)
    return len(receipts)


def repair_non_dominating_return_phi_inputs(function):
    """Recompute an exact pure return expression on its physical edge.

    Structured source guards can prove that a branch-local numerical region
    executed whenever a later synthesized return edge is selected. Repository
    SSA deliberately does not rely on that path correlation: every Phi input
    must have an ordinary dominating definition. When the return receipt names
    the same source value as a non-dominating planned-region projection, clone
    only that pure expression slice onto the return edge and select the fresh
    result. Existing dominating inputs are incumbents and are never replaced.

    The accepted clone is a strictly stronger placement (edge-local versus
    path-correlated), and the resulting operand dominates its edge, so a second
    pass makes no change.
    """
    from ..transmogrifier.ssa import Instr, SSAValue

    block_names = tuple(function.blocks)
    if not block_names:
        return ()
    entry = "entry" if "entry" in function.blocks else block_names[0]
    predecessors = {name: set() for name in block_names}
    for name, block in function.blocks.items():
        for successor in block.successors:
            if successor in predecessors:
                predecessors[successor].add(name)
    dominators = {
        name: ({name} if name == entry else set(block_names))
        for name in block_names
    }
    changed = True
    while changed:
        changed = False
        for name in block_names:
            if name == entry:
                continue
            incoming = predecessors[name]
            common = (
                set.intersection(*(dominators[parent] for parent in incoming))
                if incoming else set()
            )
            updated = {name} | common
            if updated != dominators[name]:
                dominators[name] = updated
                changed = True

    definitions = {}
    for block_name, block in function.blocks.items():
        for index, instruction in enumerate(block.instrs):
            if instruction.res is not None:
                value_id = int(instruction.res.id)
                definitions.setdefault(value_id, []).append(
                    (block_name, index, instruction)
                )
    formal_ids = {int(value.id) for value in function.args}

    def definition_dominates(value_id, edge_name, insertion_index):
        value_id = int(value_id)
        if value_id in formal_ids:
            return True
        locations = definitions.get(value_id, ())
        if len(locations) != 1:
            return False
        owner, index, _instruction = locations[0]
        return owner in dominators[edge_name] and (
            owner != edge_name or index < insertion_index
        )

    def cloneable(instruction):
        attributes = instruction.attributes or {}
        operation = str(instruction.op)
        if operation == "Call":
            return (
                attributes.get("region_index") is not None
                and attributes.get("result_convention") == "ssa.aggregate"
            )
        if operation == "GetElementPtr":
            return (
                attributes.get("region_index") is not None
                and attributes.get("source_output_id") is not None
            )
        if operation == "Load":
            return (
                attributes.get("region_index") is not None
                or attributes.get("binding") == "ssa_sequence_length"
            )
        if operation == "Cast":
            return attributes.get("binding") == "ssa_sequence_length"
        return operation in {"Const", "NoneValue"}

    receipts = []
    for exit_block in function.blocks.values():
        for phi in exit_block.instrs:
            attributes = phi.attributes or {}
            if (
                phi.op != "Phi"
                or attributes.get("binding") != "return_merge"
                or phi.res is None
            ):
                continue
            incoming_blocks = tuple(attributes.get("incoming_blocks", ()))
            if len(incoming_blocks) != len(phi.args):
                continue
            slot = attributes.get("return_slot_index")
            if not isinstance(slot, int):
                continue
            for operand_index, (edge_name, operand) in enumerate(zip(
                incoming_blocks, tuple(phi.args)
            )):
                edge = function.blocks.get(str(edge_name))
                if edge is None or not edge.instrs:
                    continue
                insertion_index = len(edge.instrs) - 1
                if definition_dominates(
                    int(operand.id), str(edge_name), insertion_index
                ):
                    continue
                source_slots = tuple(
                    (edge.instrs[-1].attributes or {}).get(
                        "return_source_value_ids", ()
                    )
                )
                if (
                    slot < 0
                    or slot >= len(source_slots)
                    or int(source_slots[slot]) != int(operand.id)
                ):
                    continue
                root_definitions = definitions.get(int(operand.id), ())
                if len(root_definitions) != 1:
                    continue
                source_block, _source_index, root = root_definitions[0]
                if not (
                    root.op == "Load"
                    and (root.attributes or {}).get("region_index") is not None
                    and int((root.attributes or {}).get(
                        "source_output_id", -1
                    )) == int(operand.id)
                ):
                    continue

                planned = []
                memo = {}
                failed = False

                def materialize(value):
                    nonlocal failed
                    value_id = int(value.id)
                    if definition_dominates(
                        value_id, str(edge_name), insertion_index
                    ):
                        return value
                    if value_id in memo:
                        return memo[value_id]
                    locations = definitions.get(value_id, ())
                    if len(locations) != 1:
                        failed = True
                        return value
                    _owner, _index, producer = locations[0]
                    if not cloneable(producer):
                        failed = True
                        return value
                    cloned_arguments = [
                        materialize(argument) for argument in producer.args
                    ]
                    if failed or producer.res is None:
                        failed = True
                        return value
                    cloned_result = SSAValue(
                        GLOBAL_MONOTONIC_IDS.mint(),
                        dtype=producer.res.dtype,
                        shape=producer.res.shape,
                        device=producer.res.device,
                        accounting={
                            **dict(producer.res.accounting or {}),
                            "return_edge_recomputed_from": int(
                                producer.res.id
                            ),
                            "return_edge_recomputation": True,
                        },
                    )
                    cloned_attributes = dict(producer.attributes or {})
                    cloned_attributes.update({
                        "return_edge_recomputation": True,
                        "return_edge_source_value_id": int(producer.res.id),
                    })
                    cloned = Instr(
                        producer.op,
                        cloned_arguments,
                        cloned_result,
                        arg_roles=list(producer.arg_roles),
                        attributes=cloned_attributes,
                        source_span=producer.source_span,
                    )
                    planned.append(cloned)
                    memo[value_id] = cloned_result
                    return cloned_result

                replacement = materialize(operand)
                if failed or replacement is operand or not planned:
                    continue
                edge.instrs[insertion_index:insertion_index] = planned
                phi.args[operand_index] = replacement
                for offset, instruction in enumerate(planned):
                    definitions[int(instruction.res.id)] = [(
                        str(edge_name), insertion_index + offset, instruction,
                    )]
                receipts.append({
                    "edge": str(edge_name),
                    "return_slot_index": int(slot),
                    "source_value_id": int(operand.id),
                    "physical_value_id": int(replacement.id),
                    "source_block": str(source_block),
                    "operation_count": len(planned),
                    "priority": "edge_local_definition",
                    "replaced_priority": "path_correlated_definition",
                    "tie_policy": "incumbent",
                })
    if receipts:
        function.metadata["return_edge_recomputations"] = tuple((
            *function.metadata.get("return_edge_recomputations", ()),
            *receipts,
        ))
    return tuple(receipts)


def publish_scalar_record_return_fields(module):
    """Publish checked return versions after call signatures and CFG settle.

    Preserve physical return identities and recover initial storage from the
    record table, so repeated publication is idempotent.
    """
    from ..transmogrifier.ssa import Instr, SSAValue

    changes = 0
    for symbol, function in module.functions.items():
        receipts = function.metadata.get('record_return_state_receipts', ())
        table = module.record_tables.get(symbol)
        if not receipts or table is None:
            continue
        graph = nx.DiGraph()
        graph.graph.update(
            return_slot_values={span: slots for span, slots, states in receipts},
            return_record_field_states={span: states for span, slots, states in receipts},
        )
        lookup = scalar_return_field_versions(function, graph, module.functions)
        values = {int(value.id): value for value in function.args}
        for block in function.blocks.values():
            for operation in block.instrs:
                values.update((int(value.id), value) for value in operation.args)
                if operation.res is not None:
                    values[int(operation.res.id)] = operation.res
        conversions = {
            (name, int(op.args[0].id), op.res.dtype): op.res
            for name, block in function.blocks.items() for op in block.instrs
            if op.op == 'Cast' and op.args and op.res is not None
            and (op.attributes or {}).get('record_return_field_conversion')
        }
        for block in function.blocks.values():
            for operation in list(block.instrs):
                attrs = operation.attributes or {}
                if operation.op != 'Phi' or not attrs.get('record_return_scalar'):
                    continue
                receivers = attrs.get('record_return_receivers', ())
                predecessors = attrs.get('incoming_blocks', ())
                slot = attrs.get('return_slot_index')
                if not (len(receivers) == len(predecessors) == len(operation.args)):
                    continue
                arguments = list(operation.args)
                for index, (receiver, predecessor) in enumerate(zip(receivers, predecessors)):
                    edge = function.blocks.get(predecessor)
                    if edge is None or not edge.instrs:
                        continue
                    slots = (edge.instrs[-1].attributes or {}).get('return_source_value_ids', ())
                    if not isinstance(slot, int) or not 0 <= slot < len(slots):
                        continue
                    source = table.records.get(slots[slot])
                    physical = table.records.get(receiver)
                    if (source is None or physical is None or source.identity != physical.identity
                            or source.fields != physical.fields):
                        continue
                    field = next((item for item in physical.fields
                                  if item.name == attrs.get('record_field')), None)
                    if field is None or len(field.value_ids) != 1:
                        continue
                    fallback = values.get(int(field.value_ids[0]))
                    if fallback is None:
                        continue
                    selected = lookup(slots[slot], field.name, predecessor, fallback,
                                      alias_receivers=(receiver,))
                    if selected.dtype != fallback.dtype:
                        key = (predecessor, int(selected.id), fallback.dtype)
                        converted = conversions.get(key)
                        if converted is None:
                            converted = SSAValue(
                                GLOBAL_MONOTONIC_IDS.mint(),
                                dtype=fallback.dtype,
                            )
                            edge.instrs.insert(-1, Instr('Cast', [selected], converted, attributes={
                                'record_return_field_conversion': field.name,
                                'source_field_value_id': int(selected.id),
                            }))
                            conversions[key] = converted
                        selected = converted
                    arguments[index] = selected
                if [int(value.id) for value in arguments] != [int(value.id) for value in operation.args]:
                    operation.args = arguments
                    attrs['initial_value_id'] = int(arguments[0].id)
                    changes += 1
    return changes


def scalar_return_field_versions(function, source_graph, functions=None):
    """Return a conservative lookup for recorded scalar return state.

    Only an unambiguous source receipt for the exact receiver is eligible.
    Missing/contradictory receipts retain the existing field; this is not a
    complete mutable-record lowering. In particular, it does not select
    dictionary handles, infer call aliases, or synthesize loop-header state.
    """
    receipts = source_graph.graph.get('return_record_field_states') or {}
    if not receipts:
        return lambda receiver, field, predecessor, fallback, *, alias_receivers=(): fallback
    function.metadata['record_return_state_receipts'] = tuple(
        (span, tuple((source_graph.graph.get('return_slot_values') or {}).get(span, ())), tuple(states))
        for span, states in receipts.items()
    )
    formal_ids = {int(value.id) for value in function.args}
    definitions = {}
    instructions = {}
    for name, block in function.blocks.items():
        for instruction in block.instrs:
            if instruction.res is not None:
                definitions.setdefault(int(instruction.res.id), []).append((name, instruction.res))
                instructions[int(instruction.res.id)] = instruction
    cfg = nx.DiGraph()
    cfg.add_nodes_from(function.blocks)
    for name, block in function.blocks.items():
        if not block.instrs:
            continue
        terminal = block.instrs[-1]
        attrs = terminal.attributes or {}
        targets = ((attrs.get('target'),) if terminal.op == 'Br' else
                   (attrs.get('true_target'), attrs.get('false_target'))
                   if terminal.op == 'CondBr' else ())
        cfg.add_edges_from((name, target) for target in targets if target in function.blocks)
    entry = 'entry' if 'entry' in function.blocks else next(iter(function.blocks), None)
    dominators = nx.immediate_dominators(cfg, entry) if entry is not None else {}
    if entry is not None:
        # The repository's NetworkX adapter omits the root self-entry.
        dominators[entry] = entry

    def argument_readonly(callee_name, position, visiting):
        callee = (functions or {}).get(callee_name)
        key = (callee_name, position)
        if callee is None or key in visiting or position >= len(callee.args):
            return False
        formal = callee.args[position]
        if (formal.accounting or {}).get('program_abi_field_written'):
            return False
        operations = [op for body in callee.blocks.values() for op in body.instrs]
        aliases = {int(formal.id)}
        changed = True
        while changed:
            changed = False
            for op in operations:
                if (op.res is not None and op.op in {'GetElementPtr', 'BitCast', 'Cast', 'Identity'}
                        and any(int(arg.id) in aliases for arg in op.args)
                        and int(op.res.id) not in aliases):
                    aliases.add(int(op.res.id))
                    changed = True
        for op in operations:
            if op.res is not None and int(op.res.id) == int(formal.id):
                return False
            if op.op == 'Store':
                if len(op.args) != 2 or int(op.args[1].id) in aliases:
                    return False
            elif 'store' in op.op.lower() or 'atomic' in op.op.lower():
                if any(int(arg.id) in aliases for arg in op.args):
                    return False
            elif op.op == 'Call':
                target = (op.attributes or {}).get('callee')
                target_function = (functions or {}).get(target)
                for index, arg in enumerate(op.args):
                    if int(arg.id) not in aliases:
                        continue
                    if (target_function is None or len(op.args) != len(target_function.args)
                            or not argument_readonly(target, index, visiting | {key})):
                        return False
        return True

    def boolean_phi_tree(value, visiting):
        if value.shape:
            return False
        if value.dtype == 'bool':
            return True
        value_id = int(value.id)
        instruction = instructions.get(value_id)
        if (value_id in visiting or len(definitions.get(value_id, ())) != 1
                or instruction is None or instruction.op != 'Phi'
                or (instruction.attributes or {}).get('binding') != 'conditional_carried'
                or not instruction.args):
            return False
        return all(boolean_phi_tree(argument, visiting | {value_id})
                   for argument in instruction.args)

    def lookup(receiver, field, predecessor, fallback, *, alias_receivers=()):
        block = function.blocks.get(predecessor)
        if block is None or not block.instrs:
            return fallback
        slots = (block.instrs[-1].attributes or {}).get('return_source_value_ids')
        if slots is None:
            return fallback
        sites = [span for span, values in
                 (source_graph.graph.get('return_slot_values') or {}).items()
                 if tuple(values) == tuple(slots)]
        # Equal return slot identities can occur at different authored sites.
        # Missing field state at even one such site is not a proof.
        states = [dict(((int(r), str(f)), int(v)) for r, f, v in receipts.get(span, ()))
                  for span in sites]
        key = (int(receiver), str(field))
        if not states or any(key not in state for state in states):
            return fallback
        candidates = {state[key] for state in states}
        if len(candidates) != 1:
            return fallback
        candidates = definitions.get(next(iter(candidates)), ())
        if len(candidates) != 1:
            return fallback
        owner, value = candidates[0]
        definition = instructions[int(value.id)]
        if not (definition.op == 'Const' or (
                definition.op == 'Phi'
                and (definition.attributes or {}).get('binding') == 'conditional_carried')):
            return fallback
        if (int(value.id) in formal_ids or value.shape
                or (value.dtype != fallback.dtype and not (
                    fallback.dtype == 'bool' and boolean_phi_tree(value, set())))):
            return fallback
        current = predecessor
        while current in dominators:
            if current == owner:
                # A later effect through this field's storage invalidates
                # the receipt. Follow explicit pointer/value aliases
                # conservatively; do not infer that such a call is read-only.
                aliases = {int(fallback.id), int(value.id)}
                # A record receiver is not an alias of each of its fields.
                # Whole-record calls remain barriers, but projections of
                # unrelated fields must not taint the selected scalar slot.
                record_aliases = {int(receiver), *map(int, alias_receivers)}
                changed = True
                while changed:
                    changed = False
                    for operation in instructions.values():
                        if (operation.op in {'GetElementPtr', 'BitCast', 'Cast', 'Identity'}
                                and any(int(arg.id) in aliases for arg in operation.args)
                                and int(operation.res.id) not in aliases):
                            aliases.add(int(operation.res.id))
                            changed = True
                # Re-executing the definition kills the previous iteration's
                # version. Do not treat earlier effects in the next iteration
                # as intervening writes to the version reaching this return.
                tail_cfg = cfg.copy()
                tail_cfg.remove_edges_from(tuple(tail_cfg.in_edges(owner)))
                relevant = (nx.descendants(tail_cfg, owner) | {owner}) & (
                    nx.ancestors(tail_cfg, predecessor) | {predecessor})
                for block_name in relevant:
                    operations = function.blocks[block_name].instrs
                    if block_name == owner:
                        operations = operations[operations.index(definition) + 1:]
                    for operation in operations:
                        writes_slot = (operation.res is not None
                                       and int(operation.res.id) == int(fallback.id))
                        touches_alias = any(
                            int(arg.id) in aliases | record_aliases or
                            (arg.accounting or {}).get('ssa_storage_alias') in aliases | record_aliases
                            for arg in operation.args
                        )
                        if touches_alias and operation.op == 'Call':
                            target = (operation.attributes or {}).get('callee')
                            target_function = (functions or {}).get(target)
                            if (target_function is None or len(operation.args) != len(target_function.args)
                                    or any(not argument_readonly(target, index, set())
                                           for index, arg in enumerate(operation.args)
                                           if int(arg.id) in aliases | record_aliases or
                                           (arg.accounting or {}).get('ssa_storage_alias') in aliases | record_aliases)):
                                return fallback
                        store_target = (operation.args[1:] if operation.op == 'Store'
                                        and len(operation.args) == 2 else operation.args)
                        if writes_slot or (('store' in operation.op.lower() or 'atomic' in operation.op.lower())
                                           and any(int(arg.id) in aliases | record_aliases for arg in store_target)):
                            return fallback
                return value
            parent = dominators[current]
            if parent == current:
                break
            current = parent
        return fallback

    return lookup
