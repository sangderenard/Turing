"""Producer-backed result contracts with explicit revision and conflict rules."""
from .transformation_priority import TransformationLedger, TransformationRule


def _type(value):
    return (value.dtype, tuple(value.shape), value.device)


def _recorded_type(function, value):
    from .structural_concordance import function_value_structure

    structure, _cell = function_value_structure(function, int(value.id))
    return (structure.dtype, structure.shape, value.device)


def _publish_type(function, value, winner, decision):
    """Commit the retained decision before changing its resident SSA view."""
    from .concordance_declarations import FRAME_LINK
    from .ssa_record_return_state import publish_ssa_value_type

    changed = _type(value) != winner
    publish_ssa_value_type(
        function, value, dtype=winner[0], shape=winner[1],
        stage=FRAME_LINK, sources=(decision,),
    )
    value.device = winner[2]
    return changed


def _storage_type(value):
    # Logical Boolean/integer tensors can deliberately use double-valued
    # repository buffers. Absence, however, never becomes a numeric payload.
    dtype = value.dtype if value.dtype == 'none' else (value.accounting or {}).get('physical_dtype') or value.dtype
    return (dtype, tuple(value.shape), value.device)


def equivalent_physical_layout(left, right):
    """Contiguous views can differ in shape while sharing the same buffer ABI."""
    if left[0] != right[0] or (left[2] is not None and right[2] is not None and left[2] != right[2]):
        return False
    if left[1] == right[1]:
        return True
    from math import prod
    from operator import index
    try:
        shapes = [tuple(index(dimension) for dimension in shape) for shape in (left[1], right[1])]
    except (TypeError, ValueError):
        return False
    return all(dimension >= 0 for shape in shapes for dimension in shape) and prod(shapes[0]) == prod(shapes[1])


def _value_cells(*pairs):
    """The ``ssa_value`` cells of ``(function, value)`` pairs that the book
    holds (the proof of a proposal was read from these values)."""
    from .structural_concordance import function_value_structure

    cells = []
    for function, value in pairs:
        _structure, cell = function_value_structure(function, int(value.id))
        if cell not in cells:
            cells.append(cell)
    return tuple(cells)


def settle_call_result_types(functions, emit_outputs, function_values):
    from .ssa_record_return_state import accepted_ssa_component

    ledger = TransformationLedger((
        TransformationRule('provisional', 1),
        TransformationRule('callee_output', 2),
        TransformationRule('physical_storage', 3),
        TransformationRule('addressed_element', 3),
    ), {'provisional': ('callee_output', 'physical_storage', 'addressed_element'),
        'callee_output': ('physical_storage', 'addressed_element')}, concordant_ties=True)
    for function in functions.values():
        # Accepted dependency bodies retain the book that proved their
        # internal values.  They are immutable sources for this unit's call
        # edges, never targets of a second settlement on the ambient book.
        # Posting provisional decisions for them here produced raw rows for
        # every imported planned-region value because their exact cells live
        # on the accepted component book.
        if accepted_ssa_component(function):
            continue
        for value in function_values(function).values():
            accounting = value.accounting or {}
            recorded = _recorded_type(function, value)
            physical = (accounting.get('physical_dtype') or accounting.get('program_abi_storage')) and recorded[0] not in {None, '', 'unknown'}
            ledger.propose((function.name, int(value.id)),
                'physical_storage' if physical else 'provisional',
                ('initial', recorded), after=recorded,
                sources=_value_cells((function, value)))
            _publish_type(function, value, recorded,
                          ledger.decision_cell((function.name, int(value.id))))

    def bindings():
        for caller in functions.values():
            if accepted_ssa_component(caller):
                continue
            values = function_values(caller)
            instructions = [item for block in caller.blocks.values() for item in block.instrs]
            # Repeated region calls can retain semantic output_ids while the
            # loop's next-iteration Load owns a fresh physical result identity.
            # Follow the actual aggregate/pointer objects to settle every
            # projection instead of typing only the first semantic slot.
            projections = {}
            pointer_origins = {}
            for item in instructions:
                if item.op == 'GetElementPtr' and item.res is not None and item.args:
                    position = (item.attributes or {}).get('source_output_id')
                    if position is not None:
                        position = ('source', int(position))
                    if position is None:
                        position = (item.attributes or {}).get('aggregate_index')
                    if position is not None:
                        pointer_origins[id(item.res)] = (id(item.args[0]), position)
            for item in instructions:
                if item.op == 'Load' and item.res is not None and item.args:
                    origin = pointer_origins.get(id(item.args[0]))
                    if origin is not None:
                        projections.setdefault(origin[0], []).append((origin[1], item.res))
            for block in caller.blocks.values():
                for instruction in block.instrs:
                    if (instruction.op not in {'Call', 'call'}
                            or instruction.attributes.get('result_convention') != 'ssa.aggregate'):
                        if instruction.op in {'Call', 'call'} and instruction.res is not None:
                            callee = functions.get(str(instruction.attributes.get('callee', '')))
                            if callee is not None:
                                outputs = tuple(emit_outputs(callee.name, callee))
                                if len(outputs) == 1:
                                    yield caller, instruction.res, callee, outputs[0]
                        continue
                    callee = functions.get(str(instruction.attributes.get('callee', '')))
                    ids = tuple(map(int, instruction.attributes.get('output_ids', ())))
                    if callee is None or not ids:
                        continue
                    outputs = tuple(emit_outputs(callee.name, callee))
                    positions = tuple(map(
                        int,
                        instruction.attributes.get("output_positions", ()),
                    ))
                    selected = tuple(map(int, instruction.attributes.get('callee_output_ids', ())))
                    if positions:
                        # The linker correlates caller aggregate members with
                        # the callee's physical Ret ABI by position.  That is
                        # stronger than numeric identity: caller/callee SSA
                        # ids are function-local, and a callee output may be
                        # collision-freshened after its source-level result
                        # binding was recorded.  Looking up the old semantic
                        # ids then skips type settlement entirely (a matmul
                        # weight gradient whose id matched a helper-local
                        # integer constant was exposed as a scalar).  Preserve
                        # the exact positional contract the linker proved.
                        if (
                            len(positions) != len(ids)
                            or any(
                                position < 0 or position >= len(outputs)
                                for position in positions
                            )
                        ):
                            continue
                        outputs = tuple(outputs[position] for position in positions)
                    elif selected:
                        by_id = {int(value.id): value for value in outputs}
                        if len(selected) != len(ids) or any(value_id not in by_id for value_id in selected):
                            continue
                        outputs = tuple(by_id[value_id] for value_id in selected)
                    elif len(ids) != len(outputs):
                        continue
                    for value_id, source in zip(ids, outputs):
                        if value_id in values:
                            yield caller, values[value_id], callee, source
                    for position, target in projections.get(id(instruction.res), ()):
                        if isinstance(position, tuple):
                            # source_output_id names the caller's semantic
                            # output. Original aggregate indices may have gaps
                            # after dead output removal; the call's explicit
                            # caller/callee correspondence is authoritative.
                            candidates = [source for value_id, source in zip(ids, outputs)
                                          if value_id == position[1]]
                            if len(candidates) != 1:
                                continue
                            source = candidates[0]
                        elif isinstance(position, int) and 0 <= position < len(outputs):
                            source = outputs[position]
                        else:
                            continue
                        yield caller, target, callee, source

        # A planned numerical region is a separate repository function, but
        # its ``source_region_integral`` records each capture as the owner's
        # exact source value identity.  Call-result linking can settle that
        # owner value after the region was carved.  Carry the settled physical
        # type across this recorded capture edge so the region formal and its
        # consumers do not retain the provisional float default.
        for region in functions.values():
            if accepted_ssa_component(region):
                continue
            integral = dict(
                (region.metadata or {}).get("source_region_integral") or {}
            )
            owner = functions.get(str(integral.get("owner") or ""))
            if owner is None:
                continue
            capture_ids = set(map(
                int, integral.get("capture_value_ids") or ()
            ))
            if not capture_ids:
                continue
            owner_values = function_values(owner)
            for target in region.args:
                value_id = int(target.id)
                source = owner_values.get(value_id)
                if value_id in capture_ids and source is not None:
                    yield region, target, owner, source

    exact_values = set()
    changed = True
    rounds = 0
    while changed:
        changed = False
        rounds += 1
        # Indexed's element type and the call result that publishes it are
        # one connected type constraint, not two unrelated late passes.
        # The compiler heap micro-test settled its int64 base after region
        # carving but still returned float64 views of those integer bytes.
        # Resolve this exact address edge here, then let the existing call
        # edges carry the strengthened proof into the caller in this same
        # fixed point. Raw opaque-pointer Loads keep their declared type.
        for owner in functions.values():
            if accepted_ssa_component(owner):
                continue
            instructions = [item for block in owner.blocks.values() for item in block.instrs]
            addresses = {
                int(item.res.id): item.args[0]
                for item in instructions
                if item.op == 'GetElementPtr' and item.res is not None and item.args
            }
            for item in instructions:
                if (item.op != 'Load' or item.res is None or not item.args
                        or item.attributes.get('lowered_from') not in {'Indexed', 'gather'}):
                    continue
                base = addresses.get(int(item.args[0].id))
                if base is None:
                    continue
                base_type = _recorded_type(owner, base)
                if base_type[0] in {None, '', 'unknown', 'ptr', 'ssa.aggregate'}:
                    continue
                target = item.res
                proposal = (base_type[0], _recorded_type(owner, target)[1], base_type[2])
                identity = (owner.name, int(target.id))
                source_cells = _value_cells((owner, base), (owner, target))
                ledger.propose(
                    identity, 'addressed_element',
                    (owner.name, int(base.id), int(item.args[0].id), proposal),
                    before=_type(target), after=proposal,
                    sources=source_cells, revision_source=source_cells[0],
                )
                winner = ledger.incumbent_target(identity)
                if _publish_type(owner, target, winner, ledger.decision_cell(identity)):
                    changed = True
                physical = (base.accounting or {}).get('physical_dtype')
                if physical:
                    target.accounting = {**dict(target.accounting or {}),
                        'physical_dtype': physical,
                        'physical_dtype_provenance': ('addressed_element', owner.name, int(base.id))}
        for caller, target, callee, source in bindings():
            # Imported bodies remain immutable. Their exact producer proof
            # crosses through the book's digest-qualified import edge; it is
            # now usable evidence for this caller, not a foreign local Ref.
            source_type = _recorded_type(callee, source)
            if source_type[0] in {None, '', 'unknown'} and not source_type[1] and source_type[2] is None:
                continue
            identity = (caller.name, int(target.id))
            source_priority = ledger.incumbent_priority((callee.name, int(source.id))) or 1
            rule = 'physical_storage' if source_priority == 3 else 'callee_output'
            previous_priority = ledger.incumbent_priority(identity)
            source_cells = _value_cells((callee, source), (caller, target))
            accepted = ledger.propose(identity, rule,
                (callee.name, int(source.id), source_type), before=_recorded_type(caller, target), after=source_type,
                sources=source_cells + tuple(
                    cell for cell in (
                        ledger.decision_cell((callee.name, int(source.id))),
                    ) if cell is not None
                ), revision_source=source_cells[0])
            winner = ledger.incumbent_target(identity)
            if _publish_type(caller, target, winner, ledger.decision_cell(identity)):
                changed = True
            if ledger.incumbent_priority(identity) != previous_priority:
                changed = True
            if accepted:
                target.accounting = {**dict(target.accounting or {}),
                    'ssa_call_result_from': (callee.name, int(source.id))}
                physical_dtype = (source.accounting or {}).get('physical_dtype')
                if physical_dtype and previous_priority != 3:
                    target.accounting['physical_dtype'] = physical_dtype
                    target.accounting['physical_dtype_provenance'] = (
                        'callee_output', callee.name, int(source.id))
            elif winner == source_type:
                target.accounting = dict(target.accounting or {})
                target.accounting.setdefault('ssa_call_result_from', (callee.name, int(source.id)))
            # A rejected physical incumbent need not have a callee origin.
            # Only accepted/equivalent result bindings authorize downstream
            # consumers to propagate that origin.
            if (accepted or winner == source_type) and (target.accounting or {}).get('ssa_call_result_from'):
                exact_values.add(id(target))

    conflicts = tuple(dict(caller=caller.name, value_id=int(target.id),
        retained=_type(target), retained_storage=_storage_type(target),
        callee=callee.name, callee_value_id=int(source.id),
        proposed=_type(source), proposed_storage=_storage_type(source))
        for caller, target, callee, source in bindings()
        if source.dtype not in {None, '', 'unknown'} and not equivalent_physical_layout(_storage_type(source), _storage_type(target)))
    return exact_values, ledger.events, conflicts, rounds
