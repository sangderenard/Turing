"""Producer-backed result contracts with explicit revision and conflict rules."""
from .transformation_priority import TransformationLedger, TransformationRule


def _type(value):
    # The proof tuple: (dtype, shape, device, physical_dtype).  The resident
    # physical_dtype is its accounting view.
    return (value.dtype, tuple(value.shape), value.device,
            (value.accounting or {}).get('physical_dtype'))


def _recorded_type(function, value):
    from .structural_concordance import function_value_structure

    structure, _cell = function_value_structure(function, int(value.id))
    # The fourth member is the POSTED physical dtype (None when no producer
    # posted one): the proof reads the row, not the accounting.
    return (structure.dtype, structure.shape, value.device,
            structure.physical_dtype)


def _posted_physical(function, value):
    """The storage/rank/arena POSTED on ``value``'s ``ssa_value`` row (the
    fields a producer declared; absent ones are omitted).

    Program: call-result settlement.  Wrong record: ``_publish_type`` posted
    only the winner's ``physical_dtype``, so a call result or formal whose
    winning proof was a callee's posted ``span`` of rank 3 was revised with
    no storage or rank.  Rule: the winning proof's source row is read for its
    whole posted physical set; nothing is read from accounting."""
    from .structural_concordance import function_value_structure

    structure, _cell = function_value_structure(function, int(value.id))
    fields = {}
    if structure.declared_storage is not None:
        fields['storage'] = structure.declared_storage
    if structure.rank is not None:
        fields['rank'] = structure.rank
    if structure.arena:
        fields['arena'] = True
    return fields


def _row_physical(function, value):
    """The physical fields recorded on ``value``'s own row ({} for none)."""
    from .frame_identity_book import _frame_physical_fields

    fields = _frame_physical_fields(function, int(value.id))
    return dict(fields) if isinstance(fields, dict) else {}


def _publish_type(function, value, winner, decision, carried=None):
    """Commit the retained decision before changing its resident SSA view.

    Rule: post first; accounting is a view.  The whole winner -- dtype, shape
    and, when the winning proof carries one, the physical dtype -- is revised
    onto the value's row; the resident ``physical_dtype`` accounting is then
    set FROM that posted winner.  A winner with no posted physical dtype
    leaves the resident accounting untouched.

    ``carried`` is the posted storage/rank/arena of the source row of the
    ACCEPTED proposal (``_posted_physical``): a partial ``physical`` post
    merges over the recorded fact, so a field the winner's source did not
    post leaves the value's own posted field untouched."""
    from .concordance_declarations import FRAME_LINK, KEEP_PHYSICAL
    from .ssa_record_return_state import publish_ssa_value_type

    resident = _type(value)
    changed = (
        resident[:3] != winner[:3]
        or (winner[3] is not None and resident[3] != winner[3])
    )
    physical = {**({} if winner[3] is None
                   else {"physical_dtype": winner[3]}), **(carried or {})}
    publish_ssa_value_type(
        function, value, dtype=winner[0], shape=winner[1],
        stage=FRAME_LINK, sources=(decision,),
        physical=physical or KEEP_PHYSICAL,
    )
    value.device = winner[2]
    # Accounting is a view of the row just published.
    from .frame_identity_book import (
        _frame_accounting_from_posted, _frame_physical_fields,
    )
    value.accounting = _frame_accounting_from_posted(
        value.accounting, _frame_physical_fields(function, int(value.id)),
    )
    return changed

def _recorded_storage_type(function, value):
    recorded = _recorded_type(function, value)
    dtype = recorded[3] or recorded[0]
    return (dtype, recorded[1], recorded[2])

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


def _refresh_scope_occurrences(functions):
    """Make every occurrence of a settled identity a view of its row.

    Program: ``any([x > n for x in xs])`` beside a mutating call
    (c244/c269/c283).  Wrong record: a planned region and its control
    function share one ``ssa_value`` scope, so the region's publication of
    the list's settled ``(3,)`` revised the row both functions read, while
    ``publish_ssa_value_type`` refreshed only the publishing function's own
    occurrences; the control function kept the provisional ``()`` and the
    final concordance refused the disagreement.  Rule: a published type is
    the identity's, so each same-scope function's occurrences take the row's
    dtype and shape (the rule ``publish_ssa_value_type`` states)."""
    from .concordance_declarations import SSA_VALUE, SSAValueFact
    from .identity_concordance import current_identity_book
    from .ssa_record_return_state import accepted_ssa_component, function_scope_of

    book = current_identity_book()
    for function in functions.values():
        if accepted_ssa_component(function):
            continue
        scope = function_scope_of(function)
        occurrences = [
            *function.args,
            *(
                value
                for block in function.blocks.values()
                for instruction in block.instrs
                for value in (*instruction.args, instruction.res)
                if value is not None
            ),
        ]
        for value in occurrences:
            cell = book.latest_ref(SSA_VALUE, (scope, int(value.id)))
            if cell is None:
                continue
            fact = book.page(SSA_VALUE).cell(cell.row, cell.column)
            if (
                isinstance(fact, SSAValueFact)
                and fact.dtype not in (None, "", "unknown")
                and (value.dtype, tuple(value.shape or ()))
                != (fact.dtype, tuple(fact.shape or ()))
                and value.dtype == fact.dtype
            ):
                value.shape = tuple(fact.shape or ())


def settle_call_result_types(functions, emit_outputs, function_values):
    from .concordance_declarations import PLANNING_VALUE
    from .identity_concordance import current_identity_book
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
            physical_fields = _row_physical(function, value)
            # The generated VJP's bw_matmul call result 81 has a posted
            # rank-2 span, but its shape is still unresolved ().  Storage
            # alone cannot make that empty shape a complete rank-3 type
            # decision: the exact unbroadcast result later supplies (2, 2).
            # An unstated rank does not contradict an exact scalar physical
            # dtype (the bool incumbent test).  Only a POSTED positive rank
            # proves that an empty shape is still missing extents.
            complete_shape = (
                bool(recorded[1])
                or physical_fields.get('rank') in {None, 0}
            )
            physical = (
                bool(accounting.get('physical_dtype') or accounting.get('program_abi_storage'))
                and recorded[0] not in {None, '', 'unknown'}
                and complete_shape
            )
            identity = (function.name, int(value.id))
            sources = _value_cells((function, value))

            ledger.propose(
                identity,
                'provisional',
                ('existing_concordance_state', recorded),
                after=recorded,
                sources=sources,
            )

            # Do not republish the resident state here.
            # It is already the concordance-backed state of this value.

    capture_planning_cells = {}
    # The posted storage/rank/arena of each identity's accepted callee-output
    # proof, read from the source row it was accepted from.  Kept OUT of the
    # proof tuple: equal-priority proposals that differ only in a posted
    # field would otherwise raise TransformationConflict.
    carried_physical = {}

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
                    # A per-arm region call (``union_arm_dispatch``) types
                    # its own loads; the joined id is the merge of every
                    # arm's, not this callee's output.
                    for value_id, source in zip(ids, outputs):
                        if value_id in values and not instruction.attributes.get(
                            'union_arm_dispatch'
                        ):
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
            # Program: a returned record's field read (``m.err``) captured
            # into a region.  ``capture_value_ids`` were cut with the region
            # (the GetAttr projection id, e.g. 17); the record-projection
            # repair later rebinds that projection to the returned-record
            # resident (e.g. ...018) and posts PLANNING_VALUE
            # (owner scope, projection id) -> resident.  Wrong record: the
            # metadata capture id is no longer a value of the owner, so
            # ``owner_values.get(capture_id)`` found nothing and the formal kept
            # its cut-time dtype while the actual was the settled bool/int.
            # Rule: the capture id is resolved through the posted planning
            # value row; no row means the capture id is the resident.
            book = current_identity_book()
            planning_page = book.page(PLANNING_VALUE)
            for target in region.args:
                value_id = int(target.id)
                if value_id not in capture_ids:
                    continue
                row = (str(owner.name), value_id)
                resident_id = planning_page.latest(row)
                source = owner_values.get(
                    value_id if resident_id is None else int(resident_id)
                )
                if source is None:
                    continue
                planning_cell = book.latest_ref(PLANNING_VALUE, row)
                if planning_cell is not None:
                    capture_planning_cells[id(target)] = (planning_cell,)
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
                proposal = (base_type[0], _recorded_type(owner, target)[1],
                            base_type[2], base_type[3])
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
                # Rule: accounting is a view of the posted winner.  Program:
                # an addressed element whose winner posted no physical dtype.
                # Wrong record: the base's accounting dtype was copied onto
                # the target with no row behind it.  Rule: only the target's
                # posted row is mirrored.
                posted_target = _row_physical(owner, target)
                if posted_target.get('physical_dtype'):
                    target.accounting = {**dict(target.accounting or {}),
                        'physical_dtype': posted_target['physical_dtype'],
                        'physical_dtype_provenance': ('addressed_element', owner.name, int(base.id))}
        for caller, target, callee, source in bindings():
            # Imported bodies remain immutable. Their exact producer proof
            # crosses through the book's digest-qualified import edge; it is
            # now usable evidence for this caller, not a foreign local Ref.
            source_type = _recorded_type(callee, source)
            if source_type[0] in {None, '', 'unknown'} and not source_type[1] and source_type[2] is None:
                continue
            source_cells = _value_cells((callee, source), (caller, target))
            source_physical = _posted_physical(callee, source)
            if _publish_type(caller, target, source_type, source_cells[0],
                             source_physical):
                changed = True
            target.accounting = {**dict(target.accounting or {}),
                'ssa_call_result_from': (callee.name, int(source.id))}
            # Rule: accounting is a view of the target's posted row; a
            # source's accounting dtype with no row behind it is not
            # copied onto the target.
            physical_dtype = _row_physical(caller, target).get('physical_dtype')
            if physical_dtype is not None:
                target.accounting['physical_dtype'] = physical_dtype
                target.accounting['physical_dtype_provenance'] = (
                    'callee_output', callee.name, int(source.id))
            exact_values.add(id(target))

    _refresh_scope_occurrences(functions)
    conflicts = tuple(dict(caller=caller.name, value_id=int(target.id),
        retained=_type(target)[:3], retained_storage=_storage_type(target),
        retained_recorded=_recorded_type(caller, target),
        retained_physical=_row_physical(caller, target),
        callee=callee.name, callee_value_id=int(source.id),
        proposed=_type(source)[:3], proposed_storage=_storage_type(source),
        proposed_recorded=_recorded_type(callee, source),
        proposed_physical=_row_physical(callee, source))
        for caller, target, callee, source in bindings()
        if _recorded_type(callee, source)[0] not in {None, '', 'unknown'}
        and not equivalent_physical_layout(
            _recorded_storage_type(callee, source),
            _recorded_storage_type(caller, target),
        )
        and not (
            _recorded_type(callee, source)[:3]
            == _recorded_type(caller, target)[:3]
            and _recorded_type(callee, source)[3] is None
        ))
    return exact_values, ledger.events, conflicts, rounds
