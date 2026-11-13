"""Revalidate an invocation's borrowed glyph view; never grant owned escape."""
from .concordance_declarations import (
    BORROWED_RECORD_SPAN_ROLE, CANONICAL_VALUE, CELL_SET, CONTROL_BLOCK, ADDRESS,
    CONTROL_VALUE_BINDING, BINDING_WITHDRAWN, PHI_LOOP_HEADER,
    PROCESS_GRAPH_OPERANDS, ControlBlockKind,
    BindingKind, ControlBinding,
    ALIAS_APPLICATION, PLANNING_VALUE, SEQUENCE_DESCRIPTOR,
    DESCRIPTOR_CELL, FORMAL_ACTUAL_OCCURRENCE, REGION_CALL_RESULT,
    SEQUENCE_STORAGE_LIFETIME, SSA_VALUE,
)
from .source_borrowed_record_spans import (
    _fact, _validate_borrowed_record_string_view, borrowed_record_string_source,
)


def borrowed_record_string_invocation_sources(module, function_name, source_cell, result_cell):
    """Use only role identities captured by this exact root Call's mint.

    The caller still owes complete source mutation/helper and native-use
    closure. This proves the borrowed source, not the destination or transfer.
    """
    try:
        return _validate(module, str(function_name), source_cell, result_cell)
    except (ValueError, KeyError, AttributeError, TypeError, IndexError):
        return None


def _validate(module, name, source, result):
    from .ssa_record_return_state import function_scope_of
    from .source_owned_record_writes import _ssa_history, _call_occurrence_sources
    from .source_sequence_lifetime import _validate as lifetime_fact
    from ..transmogrifier.ssa import SSAChildTablePoolDescriptor
    from .source_loop_induction_identity import native_loop_induction_sources

    function, table = module.functions[name], module.sequence_tables[name]
    book, scope = table.book, function_scope_of(function)
    if dict(getattr(module, 'metadata', {}) or {}).get('identity_book') is not book:
        return None
    if (source.page != CANONICAL_VALUE or result.page != SSA_VALUE
            or result.row[0] != scope
            or book.latest_ref(source.page, source.row) != source
            or book.latest_ref(result.page, result.row) != result):
        return None
    definitions = {}
    for block in function.blocks.values():
        for op in block.instrs:
            if op.res is not None:
                definitions.setdefault(int(op.res.id), []).append(op)
    calls = definitions.get(int(result.row[-1]), ())
    if len(calls) != 1 or calls[0].op != 'Call':
        return None
    call = calls[0]
    positions = [(str(label), index) for label, block in function.blocks.items()
                 for index, operation in enumerate(block.instrs) if operation is call]
    if len(positions) != 1:
        return None
    label, index = positions[0]
    history = _ssa_history(book, result)
    births = [book.mint_of(cell) for cell in history if book.mint_of(cell) is not None]
    if len(births) != 1 or births[0][0] != REGION_CALL_RESULT or len(births[0][1]) != 1:
        return None
    bundle = births[0][1][0]
    if bundle.page != CELL_SET:
        return None
    captured = {parent for parent, _ in book.edges_into(bundle)}
    if set(_fact(book, bundle)) != {parent.key for parent in captured}:
        return None
    lexical_scope, source_id = source.row
    provenance = borrowed_record_string_source(book, lexical_scope, source_id)
    if provenance is None or provenance[2] != source:
        return None
    # This particular invocation must carry the authored argument/effect,
    # rather than merely happen to share a pool with another source call.
    mutations = []
    for control in captured:
        if control.page != CONTROL_BLOCK:
            continue
        for mutation in _fact(book, control).sequence_mutations:
            target, operation, arguments, effect, policy, kind, publication = mutation
            if (operation, arguments, policy, kind) == ('append', (source,), 'duplicates', 'value'):
                if effect in captured:
                    mutations.append((control, mutation))
    if len(mutations) != 1:
        return None
    control, mutation = mutations[0]
    callee = module.functions.get((call.attributes or {}).get('callee'))
    if callee is None or len(callee.args) != len(call.args):
        return None
    actuals, occurrence_proof, rewritten = [], set(), []
    for position, (formal, actual) in enumerate(zip(callee.args, call.args)):
        row = (callee.name, int(formal.id), name, label, index, position, str(scope))
        occurrence = book.latest_ref(FORMAL_ACTUAL_OCCURRENCE, row)
        current = book.latest_ref(SSA_VALUE, (scope, int(actual.id)))
        if (occurrence is None or current is None or _fact(book, occurrence) != int(actual.id)
                or current not in {p for p, _ in book.edges_into(occurrence)}):
            return None
        original_id = int(actual.id)
        application = book.latest_ref(ALIAS_APPLICATION, (name, label, index, position))
        if application is not None:
            applied = _fact(book, application)
            if (len(applied) == 5 and applied[0] != int(actual.id)
                    and applied[1:] == (int(actual.id), int(actual.id), 'resident_dominates_use', label)):
                original_id = int(applied[0])
        original = book.latest_ref(SSA_VALUE, (scope, original_id))
        if original is None:
            return None
        if original_id != int(actual.id):
            rewritten.append((position, original, current))
        occurrence_proof.update((occurrence, current))
        actuals.append(original)
    invocation = _call_occurrence_sources(book, result, actuals, control, mutation[3], callee)
    if invocation is None:
        return None
    occurrence_proof.update(invocation)
    parent = provenance[1]
    view = table.sequences.get(int(source_id))
    retired = view is None
    if retired:
        # No arbitrary descriptor-history search: every changed actual must
        # cite this exact original descriptor in its planning equivalence.
        retirement = book.latest_ref(SEQUENCE_DESCRIPTOR, (table.owner, int(source_id)))
        if retirement is None or _fact(book, retirement) is not None or len(rewritten) != 3:
            return None
        prior = {p for p, _ in book.edges_into(retirement)}
        if len(prior) != 1:
            return None
        descriptor, = prior
        if descriptor.page != SEQUENCE_DESCRIPTOR or descriptor.row != retirement.row:
            return None
        versions = sorted(c for r, c in book.page(SEQUENCE_DESCRIPTOR).cells if r == descriptor.row)
        if versions[-2:] != [descriptor.column, retirement.column]:
            return None
        for _, original, _ in rewritten:
            planning = book.latest_ref(PLANNING_VALUE, (name, int(original.row[-1])))
            if planning is None or descriptor not in {p for p, _ in book.edges_into(planning)}:
                return None
        occurrence_proof.update((descriptor, retirement))
        view = _fact(book, descriptor)
    if not {view.column_value_ids[0], view.length_address_id, view.capacity_value_id} <= {
            int(arg.row[-1]) for arg in actuals}:
        return None
    for original in actuals:
        if not _ssa_history(book, original).intersection(captured):
            return None
    resident_view = None
    if rewritten:
        remap = _exact_view_alias(module, name, source, provenance, view, call, label,
                                  index, actuals, rewritten, definitions, descriptor if retired else None)
        if remap is None:
            return None
        resident_view, alias_proof = remap
        occurrence_proof.update(alias_proof)
    descriptors = dict(table.sequences)
    if retired:
        if resident_view is None:
            return None
        descriptors[int(source_id)] = view
    # Decode role-bearing DESCRIPTOR_CELL births from this invocation only.
    # No global search or role labels from accounting can supply a missing cell.
    groups = {}
    for cell in captured:
        if cell.page != SSA_VALUE or cell.row[0] != scope:
            continue
        minted = [book.mint_of(c) for c in _ssa_history(book, cell)
                  if book.mint_of(c) is not None]
        if len(minted) != 1 or minted[0][0] != DESCRIPTOR_CELL or len(minted[0][1]) != 1:
            continue
        operands = minted[0][1][0]
        if operands.page != CELL_SET:
            continue
        causes = {p for p, _ in book.edges_into(operands)}
        roles = [p for p in causes if p.page == BORROWED_RECORD_SPAN_ROLE]
        if not roles:
            continue
        if len(roles) != 1 or set(_fact(book, operands)) != {p.key for p in causes}:
            return None
        role, = roles
        if role not in captured or role.row[0] != scope:
            return None
        values = groups.setdefault(role.row[1], {}).setdefault(role.row[2], set())
        values.add(int(cell.row[-1]))
    matches = []
    for original, roles in groups.items():
        if set(roles) != {'column', 'lengths', 'row_stride', 'capacity'} or any(len(v) != 1 for v in roles.values()):
            continue
        ids = {role: next(iter(values)) for role, values in roles.items()}
        retired_recipe = None
        if retired:
            logicals = [c for c in captured if c.page == CONTROL_VALUE_BINDING
                        and c.row == (scope, int(source_id))]
            rows = [c for c in captured if c.page == CONTROL_VALUE_BINDING
                    and c.row == (scope, int(provenance[4][5]))]
            if len(logicals) != 1 or len(rows) != 1:
                continue
            logical, = logicals
            handles = [p for p, _ in book.edges_into(logical) if p.page == SSA_VALUE
                       and p != _fact(book, logical).ssa_value]
            if len(handles) != 1:
                continue
            handle = int(handles[0].row[-1])
            index_cell = _fact(book, rows[0]).ssa_value
            operands = _mint_ids(book, scope, view.column_value_ids[0], ADDRESS)
            if operands is None or ids['column'] not in operands or len(operands) != 2:
                continue
            offset, = operands - {ids['column']}
            root = descriptors.get(int(parent.row[-1]))
            if root is None:
                continue
            arena = root.column_value_ids[int(provenance[4][1])]
            addresses = [int(c.row[-1]) for c in captured if c.page == SSA_VALUE and c.row[0] == scope
                         and _mint_ids(book, scope, c.row[-1], ADDRESS) == {arena, int(index_cell.row[-1])}]
            if len(addresses) != 1:
                continue
            retired_recipe = (int(index_cell.row[-1]), offset, addresses[0])
        else:
            length_ops = definitions.get(int(view.length_address_id), ())
            if (len(length_ops) != 1 or length_ops[0].op != 'GetElementPtr'
                    or len(length_ops[0].args) != 2 or length_ops[0].args[0].id != ids['lengths']):
                continue
            handle = int(length_ops[0].args[1].id)
            loads = definitions.get(handle, ())
            if len(loads) != 1 or loads[0].op != 'Load' or len(loads[0].args) != 1:
                continue
            addresses = definitions.get(int(loads[0].args[0].id), ())
            if len(addresses) != 1 or addresses[0].op != 'GetElementPtr' or len(addresses[0].args) != 2:
                continue
            index_cell = book.latest_ref(SSA_VALUE, (scope, int(addresses[0].args[1].id)))
        row_source = book.latest_ref(CANONICAL_VALUE, (lexical_scope, int(provenance[4][5])))
        if index_cell is None or row_source is None:
            continue
        mint = book.mint_of(index_cell)
        if mint is None or mint[0] != PHI_LOOP_HEADER or len(mint[1]) != 1:
            continue
        loops = [p for p, _ in book.edges_into(mint[1][0]) if p.page == CANONICAL_VALUE]
        if len(loops) != 1:
            continue
        loop, = loops
        loop_control = book.latest_ref(CONTROL_BLOCK, (lexical_scope, ControlBlockKind.LOOP, loop))
        loop_operands = book.latest_ref(PROCESS_GRAPH_OPERANDS, loop.row)
        if loop_control is None or loop_operands is None:
            continue
        control_fact = _fact(book, loop_control)
        iterables = tuple(p for p, role in _fact(book, loop_operands) if role in {'iterable', 'iter'})
        if (iterables != (parent,) or not control_fact.extra
                or control_fact.extra[0] != provenance[3]):
            continue
        scoped = native_loop_induction_sources(module, name, index_cell, loop, label)
        if scoped is None:
            continue
        scoped = (*scoped, loop_control, loop_operands)
        # The existing loop writer withdraws with next_induction (the exact
        # latch arm), not the header value whose binding is leaving scope.
        phi, = definitions[int(index_cell.row[-1])]
        exit_cause = book.latest_ref(SSA_VALUE, (scope, int(phi.args[1].id)))
        if exit_cause is None:
            continue
        bindings = {}
        withdrawal_proof = set()
        for canonical in (source, row_source):
            row = (scope, int(canonical.row[-1]))
            candidates = [c for c in captured if c.page == CONTROL_VALUE_BINDING and c.row == row]
            if len(candidates) != 1:
                break
            old, = candidates
            latest = book.latest_ref(CONTROL_VALUE_BINDING, row)
            if latest != old:
                fact = _fact(book, latest)
                old_value = _fact(book, old).ssa_value
                causes = {p for p, _ in book.edges_into(latest)}
                versions = sorted(c for r, c in book.page(CONTROL_VALUE_BINDING).cells if r == row)
                if versions[-2:] != [old.column, latest.column]:
                    break
                if isinstance(fact, ControlBinding) and fact.kind is BindingKind.RESTORED:
                    if len(versions) < 3:
                        break
                    from .identity_concordance import Ref
                    previous = Ref(CONTROL_VALUE_BINDING, row, versions[-3])
                    previous_fact = _fact(book, previous)
                    if (not isinstance(previous_fact, ControlBinding)
                            or previous_fact.ssa_value != fact.ssa_value
                            or causes != {canonical, loop, exit_cause, fact.ssa_value}):
                        break
                    withdrawal_proof.add(previous)
                else:
                    expected = {canonical, loop, exit_cause, old_value}
                    if (getattr(fact, 'reason', None) != BINDING_WITHDRAWN
                            or causes != expected or set(fact.read) != expected):
                        break
                withdrawal_proof.update((latest, exit_cause))
            bindings[int(canonical.row[-1])] = old
        else:
            pass
        if len(bindings) != len({int(source.row[-1]), int(row_source.row[-1])}):
            continue
        pool = SSAChildTablePoolDescriptor(
            handle_column=int(provenance[4][1]), column_value_ids=(ids['column'],),
            length_value_id=ids['lengths'], capacity_value_id=ids['capacity'],
            row_stride_value_id=ids['row_stride'], column_dtypes=(str(provenance[4][4]),),
            key_columns=(), writable=False)
        lifetime = book.latest_ref(SEQUENCE_STORAGE_LIFETIME, (scope, int(parent.row[-1])))
        if lifetime is None:
            continue
        contract = lifetime_fact(book, lifetime).contract_cell
        parts = _validate_borrowed_record_string_view(
            book, scope, lexical_scope, source_id, descriptors, pool, handle,
            function.blocks, contract, binding_cells=bindings,
            resident_view=resident_view, retired_recipe=retired_recipe, proof_parts=True)
        if parts is None:
            continue
        authored_proof, proof = parts
        # Only the authored recipe must have been captured by this invocation.
        # Live resident recipe evidence is separately authorized by the exact
        # planning/application triple above, never mistaken for captured proof.
        if any(not (_ssa_history(book, c).intersection(captured) if c.page == SSA_VALUE
                    else c in captured) for c in authored_proof):
            continue
        matches.append(set(proof) | set(scoped) | withdrawal_proof)
    if len(matches) != 1:
        return None
    return tuple(sorted(matches[0] | history | occurrence_proof | {bundle}, key=repr))


def _exact_view_alias(module, name, source, provenance, view, call, label, index,
                      original_actuals, rewritten, definitions, retired_descriptor=None):
    """One complete field-view substitution, not arbitrary operand equivalence."""
    from .identity_concordance import CorrelationTable
    from .ssa_record_return_state import function_scope_of
    function, table = module.functions[name], module.sequence_tables[name]
    book, scope = table.book, function_scope_of(function)
    old_ids = (view.column_value_ids[0], view.length_address_id, view.capacity_value_id)
    positions = tuple(i for i, c in enumerate(original_actuals) if int(c.row[-1]) in old_ids)
    if (len(positions) != 3 or tuple(int(original_actuals[i].row[-1]) for i in positions) != old_ids
            or tuple(i for i, _, _ in rewritten) != positions):
        return None
    new_ids = tuple(int(call.args[i].id) for i in positions)
    candidates = [d for d in table.sequences.values()
                  if (d.column_value_ids == (new_ids[0],) and d.length_address_id == new_ids[1]
                      and d.capacity_value_id == new_ids[2])]
    if len(candidates) != 1:
        return None
    resident, = candidates
    resident_source = book.latest_ref(CANONICAL_VALUE, (source.row[0], int(resident.sequence_id)))
    resident_provenance = borrowed_record_string_source(book, source.row[0], resident.sequence_id)
    if (resident_provenance is None or resident_provenance[2] != resident_source
            or resident_provenance[0] != provenance[0] or resident_provenance[1] != provenance[1]
            or resident_provenance[3:5] != provenance[3:5]):
        return None
    row_source = book.latest_ref(CANONICAL_VALUE, (source.row[0], int(provenance[4][5])))
    old_descriptor = retired_descriptor or book.latest_ref(SEQUENCE_DESCRIPTOR, (table.owner, int(view.sequence_id)))
    new_descriptor = book.latest_ref(SEQUENCE_DESCRIPTOR, (table.owner, int(resident.sequence_id)))
    if (old_descriptor is None or new_descriptor is None or row_source is None
            or _fact(book, old_descriptor) != view or _fact(book, new_descriptor) != resident):
        return None
    proof = {old_descriptor, new_descriptor, row_source, resident_source}
    dominators = CorrelationTable._dominators(function)
    for position, original, actual in rewritten:
        old_id, new_id = int(original.row[-1]), int(actual.row[-1])
        application = book.latest_ref(ALIAS_APPLICATION, (name, label, index, position))
        planning = book.latest_ref(PLANNING_VALUE, (name, old_id))
        if application is None or planning is None:
            return None
        app_sources = {p for p, _ in book.edges_into(application)}
        old_sources = [p for p in app_sources if p.page == SSA_VALUE and p.row == original.row]
        new_sources = [p for p in app_sources if p.page == SSA_VALUE and p.row == actual.row]
        if len(old_sources) != 1 or len(new_sources) != 1:
            return None
        old_source, = old_sources
        new_source, = new_sources
        old_revision = _exact_helper_type_revision(module, name, old_source, original)
        new_revision = _exact_helper_type_revision(module, name, new_source, actual)
        if (old_revision is None or new_revision is None
                or _fact(book, application) != (old_id, new_id, new_id, 'resident_dominates_use', label)
                or _fact(book, planning) != new_id
                or app_sources != {old_source, new_source}
                or {p for p, _ in book.edges_into(planning)} != {
                    source, row_source, old_descriptor, new_descriptor, old_source, new_source}):
            return None
        proof.update((*old_revision, *new_revision))
        sites = [(str(b), j) for b, block in function.blocks.items()
                 for j, op in enumerate(block.instrs) if op.res is not None and int(op.res.id) == new_id]
        if len(sites) != 1:
            return None
        owner, offset = sites[0]
        if owner not in dominators.get(label, set()) or (owner == label and offset >= index):
            return None
        proof.update((application, planning, original, actual))
    return resident, proof


def _mint_ids(book, scope, value_id, transform):
    from .source_owned_record_writes import _ssa_history
    cell = book.latest_ref(SSA_VALUE, (scope, int(value_id)))
    if cell is None:
        return None
    mints = [book.mint_of(c) for c in _ssa_history(book, cell) if book.mint_of(c) is not None]
    if len(mints) != 1 or mints[0][0] != transform or len(mints[0][1]) != 1:
        return None
    bundle, = mints[0][1]
    if bundle.page != CELL_SET:
        return None
    causes = {p for p, _ in book.edges_into(bundle)}
    if set(_fact(book, bundle)) != {p.key for p in causes}:
        return None
    if any(p.page != SSA_VALUE or p.row[0] != scope for p in causes):
        return None
    return {int(p.row[-1]) for p in causes}


def _exact_helper_type_revision(module, name, old, current):
    """The existing helper element-type normalization, not arbitrary history."""
    from .ssa_record_return_state import function_scope_of
    function = module.functions[name]
    book = module.sequence_tables[name].book
    if old == current:
        return (old,)
    if old.page != SSA_VALUE or current.page != SSA_VALUE or old.row != current.row:
        return None
    versions = sorted(c for r, c in book.page(SSA_VALUE).cells if r == current.row)
    if versions[-2:] != [old.column, current.column]:
        return None
    edges = tuple(book.edges_into(current))
    if any(stage.name != 'sequence_helper_value_types' for _, stage in edges):
        return None
    causes = {p for p, _ in edges}
    if old not in causes or len(causes) != 2:
        return None
    formal_cell, = causes - {old}
    if formal_cell.page != SSA_VALUE or formal_cell.row[0] == current.row[0]:
        return None
    previous, latest, formal_fact = (_fact(book, c) for c in (old, current, formal_cell))
    fields = ('shape', 'origin', 'physical_dtype', 'storage', 'rank', 'arena', 'device')
    if any(getattr(previous, key) != getattr(latest, key) for key in fields):
        return None
    if latest.dtype != formal_fact.dtype or tuple(formal_fact.shape) != (1,):
        return None
    witnesses = []
    for label, block in function.blocks.items():
        for index, call in enumerate(block.instrs):
            if call.op != 'Call':
                continue
            callee = module.functions.get((call.attributes or {}).get('callee'))
            if callee is None or len(callee.args) != len(call.args):
                continue
            for position, (formal, actual) in enumerate(zip(callee.args, call.args)):
                if int(actual.id) != int(current.row[-1]):
                    continue
                cell = book.latest_ref(SSA_VALUE, (function_scope_of(callee), int(formal.id)))
                if cell != formal_cell or tuple(formal.shape) != (1,) or formal.dtype != latest.dtype:
                    continue
                row = (callee.name, int(formal.id), name, str(label), index, position, str(current.row[0]))
                occurrence = book.latest_ref(FORMAL_ACTUAL_OCCURRENCE, row)
                if (occurrence is not None and _fact(book, occurrence) == int(actual.id)
                        and current in {p for p, _ in book.edges_into(occurrence)}):
                    witnesses.append(occurrence)
    return (old, current, formal_cell, *witnesses) if witnesses else None
