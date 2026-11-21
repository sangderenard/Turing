"""Conservative source lifetime for build-then-read local child views.

No allocation, escape, or mutation permission is granted here. A readonly
slice is safe only while its backing pool cannot be relocated or reused.
"""
from .concordance_declarations import (
    CANONICAL_VALUE, CONTROL_BLOCK, CONTROL_BLOCK_PLACEMENT, CONTROL_PROGRAM,
    PROCESS_GRAPH_OPERANDS, RETURN_SITE_SLOT, ROOT, SHELL, Arm, ControlBlockKind,
    SOURCE_CLOSURE_ENVIRONMENT, SOURCE_CLOSURE_FREE_NAMES, SOURCE_CLOSURE_CAPTURE,
    NAME_BINDING, INGESTION_VALUE, SOURCE_ACTIVITY_SCOPE, SOURCE_ACTIVITY_NODE,
    SOURCE_ACTIVITY_EDGE,
)
from .source_sequence_lifetime import _same_origin, _ast_kind


def owned_child_read_phase_sources(book, scope, sequence_id, consumer_loop_cell, target_id):
    """Return exact closed read-phase evidence, or refuse unsupported control."""
    try:
        return _read_phase(book, scope, int(sequence_id), consumer_loop_cell, int(target_id))
    except (ValueError, KeyError, AttributeError, TypeError):
        return None


def _read_phase(book, scope, sid, loop_source, target_id):
    proof = set()

    def fact(cell):
        if cell is None or book.latest_ref(cell.page, cell.row) != cell:
            raise ValueError('missing or superseded read-phase evidence')
        proof.add(cell)
        return book.page(cell.page).cell(cell.row, cell.column)

    resident = book.latest_ref(CANONICAL_VALUE, (scope, sid))
    target = book.latest_ref(CANONICAL_VALUE, (scope, target_id))
    fact(resident); fact(target); fact(loop_source)
    if loop_source.page != CANONICAL_VALUE or loop_source.row[0] != scope:
        return None
    # The builder's entry block takes its program identity from this same
    # declared shell row, not a conditional subprogram in the lexical scope.
    programs = [book.latest_ref(CONTROL_PROGRAM, (scope, SHELL))]
    program = fact(programs[0])
    activity_cell = book.latest_ref(SOURCE_ACTIVITY_SCOPE,(scope,))
    activity_scope = fact(activity_cell)
    pending, seen, entries = [activity_cell],set(),set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if fact(current) != activity_scope:
            return None
        for parent,_ in book.edges_into(current):
            if parent.page == SOURCE_ACTIVITY_SCOPE:
                pending.append(parent)
            elif parent.page == SOURCE_ACTIVITY_NODE and parent.row[0] == activity_scope:
                entries.add(parent)
    if len(entries) != 1:
        return None
    entry, = entries
    if fact(entry)[0] not in {'FunctionDef','AsyncFunctionDef'} or entry.row[2] != 'enter':
        return None
    root_definition = entry.row[1]
    # Definitions execute defaults, decorators and class bodies. Empty free
    # names do not prove that those operations retain no pool or slice. The
    # initial read-phase contract therefore excludes nested definitions by
    # their existing source-activity identity, not spelling or line order.
    def activity_point(point):
        kind,owners = fact(point)
        occurrence = point.row[1]
        if fact(occurrence).kind != kind or occurrence not in {p for p,_ in book.edges_into(point)}:
            raise ValueError('activity kind lacks its exact source declaration')
        if occurrence != root_definition and kind in {'FunctionDef','AsyncFunctionDef','Lambda','ClassDef'}:
            raise ValueError('nested definitions are outside the initial read-phase contract')
        return kind,owners

    activity_point(entry)
    for row in book.page(SOURCE_ACTIVITY_NODE).rows():
        if row[0] != activity_scope:
            continue
        point = book.latest_ref(SOURCE_ACTIVITY_NODE,row)
        activity_point(point)
    for row in book.page(SOURCE_ACTIVITY_EDGE).rows():
        if row[0] == activity_scope:
            edge=book.latest_ref(SOURCE_ACTIVITY_EDGE,row)
            if fact(edge) is not True:
                return None
            activity_point(row[1]); activity_point(row[2])
    for row in book.page(SOURCE_CLOSURE_FREE_NAMES).rows():
        if row[0] == activity_scope:
            fact(book.latest_ref(SOURCE_CLOSURE_FREE_NAMES,row))
            return None
    controls = [book.latest_ref(CONTROL_BLOCK, row)
                for row in book.page(CONTROL_BLOCK).rows() if row[0] == scope]
    loops = [cell for cell in controls if cell.row[1] == ControlBlockKind.LOOP
             and cell.row[2] == loop_source]
    if len(loops) != 1:
        return None
    consumer = loops[0]
    loop_fact = fact(consumer)
    # The source target relation must be independently posted, not inferred
    # from a caller-supplied integer or a matching variable spelling.
    bindings = getattr(program, 'iterable_bindings', ())
    if (resident, target, loop_fact.extra[0]) not in bindings:
        return None
    if not {resident,target} <= {p for p,_ in book.edges_into(programs[0])}:
        return None
    consumer_placement = fact(book.latest_ref(CONTROL_BLOCK_PLACEMENT, (scope, consumer)))
    if (consumer_placement.parent != ROOT or consumer_placement.arm != Arm.ROOT_SEQUENCE
            or consumer not in program.root_blocks):
        return None

    def root_position(cell):
        seen = set()
        while cell not in seen:
            seen.add(cell)
            fact(cell)
            placed = fact(book.latest_ref(CONTROL_BLOCK_PLACEMENT, (scope, cell)))
            if placed.parent == ROOT:
                if placed.arm != Arm.ROOT_SEQUENCE or cell not in program.root_blocks:
                    raise ValueError('unlisted root block')
                return placed.ordinal
            if placed.arm not in (Arm.BODY, Arm.ORELSE):
                raise ValueError('unsupported control corridor')
            cell = placed.parent
        raise ValueError('cyclic placement')

    roots = [(root_position(cell), cell) for cell in program.root_blocks]
    if tuple(position for position, _ in roots) != tuple(range(len(roots))):
        return None

    def family(seed):
        members = {seed}
        changed = True
        while changed:
            changed = False
            for row in book.page(CANONICAL_VALUE).rows():
                if row[0] != scope:
                    continue
                cell = book.latest_ref(CANONICAL_VALUE, row)
                if cell not in members and any(_same_origin(book, cell, old) for old in members):
                    members.add(cell); changed = True
            for left, right in program.value_aliases:
                if left in members or right in members:
                    before = len(members)
                    members.update((left, right)); changed |= len(members) != before
        for cell in members:
            fact(cell)
        return members

    pools, views = family(resident), family(target)
    if pools.intersection(views):
        return None
    from types import SimpleNamespace
    from .identity_concordance import CorrelationTable
    # This incumbent audit is explicitly book-only, including before SSA.
    # It closes the declared free-name inventory, so deleting environment or
    # capture rows cannot turn an escaping closure into an absent use.
    if CorrelationTable._source_closure_findings(SimpleNamespace(metadata={'identity_book':book})):
        return None
    for page in (SOURCE_CLOSURE_FREE_NAMES,SOURCE_CLOSURE_CAPTURE):
        for row in book.page(page).rows():
            fact(book.latest_ref(page,row))
    for row in book.page(SOURCE_CLOSURE_ENVIRONMENT).rows():
        environment = book.latest_ref(SOURCE_CLOSURE_ENVIRONMENT,row)
        parent, definition, address = fact(environment)
        if parent.page != NAME_BINDING:
            return None
        binding = fact(parent)
        source = book.latest_ref(CANONICAL_VALUE,(parent.row[0],binding.value_id))
        origins = {p for p,_ in book.edges_into(parent)
                   if p.page in (CANONICAL_VALUE,INGESTION_VALUE)}
        if source is not None:
            origins.add(source)
        if not origins:
            return None
        if any(_same_origin(book,origin,member) for origin in origins for member in pools | views):
            return None
    for row in book.page(RETURN_SITE_SLOT).rows():
        returned_cell = book.latest_ref(RETURN_SITE_SLOT, row)
        returned = fact(returned_cell)
        if getattr(returned, 'page', None) == CANONICAL_VALUE and any(
                _same_origin(book, returned, member) for member in pools | views):
            return None
    mutation_effects, length_effects = set(), set()
    for cell in controls:
        current = fact(cell)
        # A current block without a placement is not current executable
        # control and cannot establish permission for any operation.
        placement = book.latest_ref(CONTROL_BLOCK_PLACEMENT, (scope, cell))
        for mutation in current.sequence_mutations:
            destination, operation, arguments, effect, policy, kind, publication = mutation
            if destination not in pools and destination not in views:
                continue
            if placement is None or destination in views:
                return None
            if not {destination, effect, *arguments} <= {p for p, _ in book.edges_into(cell)}:
                return None
            if root_position(cell) >= consumer_placement.ordinal:
                return None
            mutation_effects.add(effect)
        if (current.kind == ControlBlockKind.SEQUENCE_QUERY and current.extra == ('length',)
                and placement is not None):
            length_effects.add(cell.row[2])

    # Close direct source uses. In particular alias values cannot be passed
    # to an unknown helper, placed in a container, or returned. Recognized
    # scalar computations do not propagate the borrowed storage identity.
    operand_rows = set(book.page(PROCESS_GRAPH_OPERANDS).rows())
    for row in book.page(CANONICAL_VALUE).rows():
        if row[0] == scope:
            source = book.latest_ref(CANONICAL_VALUE,row)
            if source not in pools | views and _ast_kind(book,source) in {
                    'value:Call','value:Attribute','value:For','value:Compare',
                    'value:Return','value:Subscript','value:Assign','value:AnnAssign','value:AugAssign',
                    'value:List','value:Tuple','value:Dict','value:Set','value:Lambda',
                    'value:BinOp','value:UnaryOp','value:BoolOp','value:IfExp','value:Slice',
                    'value:JoinedStr','value:FormattedValue','value:NamedExpr',
                    'value:ListComp','value:SetComp','value:DictComp','value:GeneratorExp'}:
                if row not in operand_rows or book.latest_ref(PROCESS_GRAPH_OPERANDS,row) is None:
                    return None
    for row in operand_rows:
        if row[0] != scope:
            continue
        operands_cell = book.latest_ref(PROCESS_GRAPH_OPERANDS, row)
        operands = fact(operands_cell)
        node = book.latest_ref(CANONICAL_VALUE, row)
        for operand, role in operands:
            if operand not in pools and operand not in views:
                continue
            fact(node)
            if node in pools or node in views:
                continue
            if operand in pools:
                if node in mutation_effects:
                    continue
                if node == loop_source and role == 'iterable':
                    continue
                if node in length_effects:
                    continue
                return None
            if node in length_effects or _ast_kind(book, node) == 'value:Compare':
                continue
            return None
    return tuple(sorted(proof, key=lambda cell: repr(cell.key)))
