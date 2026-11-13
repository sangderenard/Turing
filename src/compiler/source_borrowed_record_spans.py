"""Exact borrowed record string views, not transferable owned sequences.

The source survey reserves a child layout; it does not authorize a native
copy. The latter additionally needs the signature lifetime and live view
geometry, which cannot be inferred from an int64 column or a missing owner.
"""
from .concordance_declarations import (
    CANONICAL_VALUE, CONTROL_BLOCK, CONTROL_PROGRAM, SOURCE_VALUE_CLASS,
)


def _fact(book, cell):
    return book.page(cell.page).cell(cell.row, cell.column)


def exact_string_classes(book, source):
    from .source_bytes_concat_class import _class_matches_operand
    if source is None:
        return ()
    classes = tuple(book.latest_ref(SOURCE_VALUE_CLASS, row)
                    for row in book.page(SOURCE_VALUE_CLASS).rows()
                    if _class_matches_operand(book, book.latest_ref(SOURCE_VALUE_CLASS, row), source))
    return classes if classes and all(_fact(book, cell)[0] == 'builtins.str' for cell in classes) else ()


def borrowed_record_string_source(book, scope, value_id):
    """Return the unique current direct-input field projection and classes."""
    source = book.latest_ref(CANONICAL_VALUE, (scope, int(value_id)))
    if source is None:
        return None
    matches = []
    for row in book.page(CONTROL_PROGRAM).rows():
        if row[0] != scope:
            continue
        control = book.latest_ref(CONTROL_PROGRAM, row)
        for parent, target, induction, projection in _fact(book, control).projected_iterable_bindings:
            if target != source or not isinstance(projection, tuple):
                continue
            if (len(projection) != 7 or projection[0] != 'record_span'
                    or projection[2] is not None or parent.page != CANONICAL_VALUE
                    or parent.row[0] != scope or not induction):
                continue
            matches.append((control, parent, source, induction, projection))
    if len(matches) != 1:
        return None
    classes = exact_string_classes(book, source)
    if not classes:
        return None
    return (*matches[0], classes)


def borrowed_record_string_layouts(graph, sequence_declarations):
    """Predeclare only a local list whose recorded writes are exact views.

    Unsupported writes leave the whole destination unselected. In particular
    this does not change sets, dictionaries, incoming string lists or copies.
    """
    from .identity_concordance import current_identity_book
    book = current_identity_book()
    scope = graph.graph.get('lexical_read_scope')
    if scope is None:
        return {}
    return borrowed_record_string_layouts_from_book(book, scope, sequence_declarations)


def _borrowed_string_comprehension_publication(book, scope, control, mutation):
    """Admit only the existing singleton retained-list publication recipe.

    This surveys storage, not execution: the original mutation and its
    predicate are never rewritten. Generator/set/dict and expanded records
    remain outside this narrow route.
    """
    from .concordance_declarations import NATIVE_OBJECT_SOURCE
    from .source_sequence_lifetime import _ast_kind, _local_birth
    target, operation, arguments, effect, policy, kind, publication = mutation
    if ((operation, policy, kind) != ('append', 'duplicates', 'value')
            or len(arguments) != 1 or publication is None
            or publication.page != NATIVE_OBJECT_SOURCE
            or len(publication.row) != 3 or publication.row[0] != scope
            or publication.row[-1] != 'publication'
            or book.latest_ref(publication.page, publication.row) != publication):
        return False
    birth = _local_birth(book, target)
    if len(birth) != 4:
        return False
    born_control, born_publication, materializer, loop = birth
    if (born_control != control or born_publication != publication or loop != effect
            or materializer.row != publication.row[:2]
            or _ast_kind(book, materializer) != 'value:ListComp'):
        return False
    argument, = arguments
    if argument.page != CANONICAL_VALUE or argument.row[0] != scope:
        return False
    # loop_composer publishes these exact two singleton mappings. Do not
    # reinterpret an expanded record, guessed field, or unrelated source ID.
    expected = (('projected_argument_value_ids', (int(argument.row[-1]),)),
                ('source_argument_value_ids', (int(argument.row[-1]),)))
    if _fact(book, publication) != expected:
        return False
    if {parent for parent, _ in book.edges_into(publication)} != {materializer, argument}:
        return False
    if not {target, effect, publication, argument} <= {
            parent for parent, _ in book.edges_into(control)}:
        return False
    return True


def borrowed_record_string_layouts_from_book(book, scope, sequence_declarations):
    """The same exact control survey at descriptor use, without cached IDs."""
    from .source_sequence_lifetime import _ast_kind
    declarations = {int(sid) for sid, policy, width, writable in sequence_declarations
                    if policy == 'duplicates' and int(width) == 1 and writable}
    writes = {}
    for row in book.page(CONTROL_BLOCK).rows():
        if row[0] != scope:
            continue
        control = book.latest_ref(CONTROL_BLOCK, row)
        for mutation in _fact(book, control).sequence_mutations:
            target, operation, arguments, effect, policy, kind, publication = mutation
            if target.page != CANONICAL_VALUE or target.row[0] != scope:
                continue
            sid = int(target.row[-1])
            if sid in declarations:
                writes.setdefault(sid, []).append((control, mutation))
    layouts = {}
    for sid, entries in writes.items():
        target = entries[0][1][0]
        plain_list = _ast_kind(book, target) == 'value:List'
        if not plain_list and (len(entries) != 1 or not
                _borrowed_string_comprehension_publication(book, scope, *entries[0])):
            continue
        dtypes = []
        for control, mutation in entries:
            target, operation, arguments, effect, policy, kind, publication = mutation
            if plain_list and operation == 'clear' and not arguments:
                continue
            if (operation != 'append' or policy != 'duplicates' or kind != 'value'
                    or (plain_list and publication is not None) or len(arguments) != 1):
                break
            argument, = arguments
            if argument.page != CANONICAL_VALUE or argument.row[0] != scope:
                break
            source = borrowed_record_string_source(book, scope, int(argument.row[-1]))
            if source is None or source[2] != argument:
                break
            dtypes.append(str(source[4][4]))
        else:
            if dtypes and len(set(dtypes)) == 1 and dtypes[0] not in {'unknown', 'None', ''}:
                layouts[sid] = (0, (dtypes[0],), (), True)
    return layouts


def publish_borrowed_record_span_role(builder, target_id, role, projection, handle_arena, contract):
    """Post an exact input field component before its descriptor cell birth."""
    from .concordance_declarations import BORROWED_RECORD_SPAN_ROLE, BorrowedRecordSpanRole
    from .identity_concordance import Derived, Mode
    if role not in {'column', 'lengths', 'row_stride', 'capacity'}:
        return None
    book = builder._book()
    source = borrowed_record_string_source(book, builder.lexical_read_scope, target_id)
    if source is None or source[4] != projection:
        return None
    control, parent, canonical, induction, declared, classes = source
    causes = (control, parent, canonical, *classes, builder._value_cell(handle_arena), contract)
    if None in causes:
        return None
    return book.post(BORROWED_RECORD_SPAN_ROLE, (builder._scope(), canonical, role),
                     BorrowedRecordSpanRole(control, parent, declared),
                     stage=builder._stage(), provenance=Derived(causes), mode=Mode.CONCORD)


def borrowed_record_string_view(builder, value_id):
    """Validate a live readonly field view back to independent input storage.

    This is not an owned-return receipt. It establishes that a destination
    created in this invocation cannot be the source of the borrowed storage.
    """
    from .concordance_declarations import (
        CELL_SET, DESCRIPTOR_CELL, SEQUENCE_STORAGE_LIFETIME, SSA_VALUE,
        SequenceLifetimeKind, CONTROL_VALUE_BINDING, BindingKind,
        BORROWED_RECORD_SPAN_ROLE, ADDRESS, CONTROL_EXPRESSION, CONTROL_CAST,
    )
    from .source_sequence_lifetime import _validate
    book, scope = builder._book(), builder._scope()
    source = borrowed_record_string_source(book, builder.lexical_read_scope, value_id)
    if source is None:
        return None
    control, parent, canonical, induction, projection, classes = source
    parent_id = int(parent.row[-1])
    lifetime_cell = book.latest_ref(SEQUENCE_STORAGE_LIFETIME, (scope, parent_id))
    if lifetime_cell is None:
        return None
    lifetime = _validate(book, lifetime_cell)
    root = builder.sequence_descriptors.get(parent_id)
    view = builder.sequence_descriptors.get(int(value_id))
    selection = builder.child_table_selections.get(int(value_id))
    pool = getattr(builder, 'borrowed_record_field_pools', {}).get((parent_id, projection[3]))
    if (lifetime.kind is not SequenceLifetimeKind.BORROWED or lifetime.source_cell != parent
            or root is None or view is None or pool is None or selection is None
            or selection[0] != pool or pool.writable or view.writable
            or pool.key_columns or any(value is not None for value in (
                pool.shape_value_id, pool.rank_value_id, pool.shape_stride_value_id,
                pool.status_value_id, pool.live_flags_value_id))
            or view.heap_owner_value_id is not None or view.child_table_pool is not None
            or view.key_columns or view.live_flags_value_id is not None
            or view.status_address_id is not None or view.column_shapes != ((),)
            or tuple(c.row[-1] for c in lifetime.column_cells) != root.column_value_ids
            or int(projection[1]) != pool.handle_column
            or pool.column_dtypes != (str(projection[4]),)
            or view.column_dtypes != pool.column_dtypes):
        return None
    handle_arena = root.column_value_ids[pool.handle_column]
    contract = builder._sequence_contract_cell(parent_id)
    expected_base = {book.latest_ref(SSA_VALUE, (scope, handle_arena)), contract}
    proof = {control, parent, canonical, lifetime_cell, *classes, *expected_base}
    pool_source = None
    role_members = ((pool.column_value_ids[0], 'column'), (pool.length_value_id, 'lengths'),
                    (pool.row_stride_value_id, 'row_stride'), (pool.capacity_value_id, 'capacity'))
    if len(pool.column_value_ids) != 1 or len({member for member, _ in role_members}) != 4:
        return None
    # Every field pool member was minted from this particular input arena,
    # not merely labelled with the same parameter name in accounting.
    for member, expected_role in role_members:
        current = book.latest_ref(SSA_VALUE, (scope, member))
        mint = book.mint_of(current)
        if not mint or mint[0] != DESCRIPTOR_CELL or len(mint[1]) != 1:
            return None
        bundle = mint[1][0]
        if bundle.page != CELL_SET:
            return None
        causes = {p for p, _ in book.edges_into(bundle)}
        roles = tuple(cell for cell in causes if cell.page == BORROWED_RECORD_SPAN_ROLE)
        if len(roles) != 1:
            return None
        role_cell, = roles
        if (book.latest_ref(BORROWED_RECORD_SPAN_ROLE, role_cell.row) != role_cell
                or role_cell.row[0] != scope or role_cell.row[2] != expected_role):
            return None
        source_cells = causes - expected_base - {role_cell}
        if (not expected_base <= causes or len(source_cells) != 1
                or set(_fact(book, bundle)) != {p.key for p in causes}):
            return None
        original_source, = source_cells
        if original_source.page != CANONICAL_VALUE or original_source.row[0] != builder.lexical_read_scope:
            return None
        original = borrowed_record_string_source(book, builder.lexical_read_scope,
                                                 int(original_source.row[-1]))
        if (original is None or original[2] != original_source or original[1] != parent
                or original[4][:5] != projection[:5] or original[4][6] != projection[6]
                or (pool_source is not None and pool_source != original_source)):
            return None
        role_fact = _fact(book, role_cell)
        required = {original[0], parent, original_source, *original[5], *expected_base}
        if (role_cell.row[1] != original_source or role_fact.control_cell != original[0]
                or role_fact.parent_cell != parent or role_fact.projection != original[4]
                or {p for p, _ in book.edges_into(role_cell)} != required):
            return None
        pool_source = original_source
        proof.update((role_cell, original[0], original_source, *original[5]))
        proof.update((current, bundle))
    definitions = {}
    for block in builder.blocks.values():
        for op in block.instrs:
            if op.res is not None:
                definitions.setdefault(int(op.res.id), []).append(op)
    def mint_operands(value, transform, ids):
        cell = book.latest_ref(SSA_VALUE, (scope, int(value)))
        pending, seen, births = [cell], set(), []
        while pending:
            current = pending.pop()
            if current is None or current in seen:
                continue
            seen.add(current)
            mint = book.mint_of(current)
            if mint is not None:
                births.append(mint)
            else:
                pending.extend(p for p, _ in book.edges_into(current)
                               if p.page == SSA_VALUE and p.row == cell.row)
        if len(births) != 1 or births[0][0] != transform or len(births[0][1]) != 1:
            return False
        operand, = births[0][1]
        captured = {p for p, _ in book.edges_into(operand)} if operand.page == CELL_SET else {operand}
        if operand.page == CELL_SET and set(_fact(book, operand)) != {p.key for p in captured}:
            return False
        if (len(ids) != len(set(ids)) or len(captured) != len(ids)
                or {(p.page, p.row) for p in captured} != {(SSA_VALUE, (scope, int(i))) for i in ids}):
            return False
        for p in captured:
            latest = book.latest_ref(SSA_VALUE, p.row)
            stack, history = [latest], set()
            while stack:
                q = stack.pop()
                if q is None or q in history:
                    continue
                history.add(q)
                stack.extend(r for r, _ in book.edges_into(q) if r.page == SSA_VALUE and r.row == p.row)
            if p not in history:
                return False
            proof.update(history)
        proof.update((*seen, operand, *captured))
        return True

    def operation(value, name, args=None):
        candidates = definitions.get(int(value), ())
        if len(candidates) != 1 or candidates[0].op != name:
            return None
        op = candidates[0]
        if args is not None and tuple(a.id for a in op.args) != tuple(args):
            return None
        transform = {'GetElementPtr': ADDRESS, 'Mul': CONTROL_EXPRESSION, 'Cast': CONTROL_CAST}.get(name)
        if transform is not None and not mint_operands(value, transform, tuple(a.id for a in op.args)):
            return None
        proof.add(book.latest_ref(SSA_VALUE, (scope, int(value))))
        return op
    handle = selection[1].id
    load = operation(handle, 'Load')
    if load is None or len(load.args) != 1:
        return None
    address = operation(load.args[0].id, 'GetElementPtr')
    if address is None or len(address.args) != 2 or address.args[0].id != handle_arena:
        return None
    if not mint_operands(handle, CONTROL_EXPRESSION, (handle_arena, address.args[1].id)):
        return None
    logical = book.latest_ref(CONTROL_VALUE_BINDING, (scope, int(value_id)))
    column_cell = book.latest_ref(SSA_VALUE, (scope, view.column_value_ids[0]))
    handle_cell = book.latest_ref(SSA_VALUE, (scope, handle))
    if (logical is None or _fact(book, logical).kind is not BindingKind.SEQUENCE_RESULT
            or _fact(book, logical).ssa_value != column_cell
            or not {canonical, handle_cell, column_cell} <= {p for p, _ in book.edges_into(logical)}):
        return None
    proof.add(logical)
    row_source = builder._canonical_cell(int(projection[5]))
    row_binding = book.latest_ref(CONTROL_VALUE_BINDING, (scope, int(projection[5])))
    if row_binding is None or row_source is None:
        return None
    row_fact = _fact(book, row_binding)
    index_cell = book.latest_ref(SSA_VALUE, (scope, address.args[1].id))
    if (row_fact.kind is not BindingKind.LOOP_CONTROL or row_fact.ssa_value != index_cell
            or (parent, row_source, induction, 'induction') not in
            _fact(book, control).projected_iterable_bindings
            or not {parent, row_source, index_cell} <=
            {p for p, _ in book.edges_into(row_binding)}):
        return None
    proof.update((row_source, row_binding, index_cell))
    # The same selected handle addresses the independent glyph row and length.
    column = operation(view.column_value_ids[0], 'GetElementPtr')
    if column is None or len(column.args) != 2 or column.args[0].id != pool.column_value_ids[0]:
        return None
    if operation(column.args[1].id, 'Mul', (handle, pool.row_stride_value_id)) is None:
        return None
    if operation(view.length_address_id, 'GetElementPtr', (pool.length_value_id, handle)) is None:
        return None
    if operation(view.capacity_value_id, 'Cast', (pool.row_stride_value_id,)) is None:
        return None
    if None in proof:
        return None
    return tuple(sorted(proof, key=lambda cell: repr(cell.key)))
