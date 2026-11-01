"""Authoritative process-graph edge readers and their single-writer publication."""
from .concordance_declarations import PROCESS_GRAPH_OPERANDS, OPERAND_POSITION
from .identity_concordance import Derived, Mode, current_identity_book


def publish_operands(graph, node, transition_cells=()):
    from ..common.tensors.topological_reducer import node_identity_cell, _operand_position_scope
    scope = _operand_position_scope(graph)
    if scope is None:
        return
    book = current_identity_book()
    consumer = node_identity_cell(graph, int(node))
    from ..common.tensors.topological_reducer import _operand_positions
    target = getattr(graph, 'G', graph)
    # returned_growth: len(rows)'s callee and arg:0 changed list order,
    # but the incumbent operand-position writer correctly recorded no move.
    # Serialize that same role/ordinal map. Distinct roles have no relative
    # operand position; repeated roles retain their existing ordinals.
    positions = _operand_positions(target.nodes[node].get('parents', ()))
    edges = tuple((node_identity_cell(graph, int(parent)), str(role))
                  for role, ordinal, parent in sorted(positions, key=lambda item: (str(item[0]), item[1])))
    row = (scope, int(node))
    previous = book.latest_ref(PROCESS_GRAPH_OPERANDS, row)
    if previous is not None and book.page(PROCESS_GRAPH_OPERANDS).cell(previous.row, previous.column) == edges:
        return previous
    return book.post(PROCESS_GRAPH_OPERANDS, row, edges, stage=OPERAND_POSITION,
                     provenance=Derived(tuple(dict.fromkeys((consumer, *(cell for cell, _ in edges),
                                                            *transition_cells)))), mode=Mode.REVISE)


def operands(graph, node):
    from ..common.tensors.topological_reducer import _operand_position_scope
    book = current_identity_book()
    row = (_operand_position_scope(graph), int(node))
    cell = book.latest_ref(PROCESS_GRAPH_OPERANDS, row)
    if cell is None:
        raise ValueError(f'process node {node} has no concorded operand graph')
    return book.page(PROCESS_GRAPH_OPERANDS).cell(cell.row, cell.column), cell


def local_value(graph, identity):
    """Resolve an operand cell to its unique resident node without spelling."""
    from ..common.tensors.topological_reducer import node_identity_cell
    from .identity_concordance import _canonical_identity_base
    book = current_identity_book()
    target = getattr(graph, 'G', graph)
    resident = {int(node): node_identity_cell(graph, int(node)) for node in target}
    exact = [node for node, cell in resident.items() if cell == identity]
    if len(exact) == 1:
        return exact[0]
    subject = _canonical_identity_base(book, identity)
    values = [node for node, cell in resident.items()
              if _canonical_identity_base(book, cell) == subject]
    if len(values) != 1:
        raise ValueError('operand identity has no unique resident process node')
    return values[0]


def operation(graph, node):
    """Read an ingested process operation's semantic contract, not attributes."""
    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import SOURCE_PROCESS_OPERATION, CANONICAL_VALUE, INGESTION_VALUE
    return operation_from_identity(node_identity_cell(graph, int(node)))


def operation_from_identity(identity, *, book=None):
    """The same semantic reader for exact cross-function capture operands."""
    from .concordance_declarations import SOURCE_PROCESS_OPERATION, CANONICAL_VALUE, INGESTION_VALUE
    book = current_identity_book() if book is None else book
    frontier = [identity]
    visited, contracts = set(), set()
    while frontier:
        source = frontier.pop()
        if source in visited:
            continue
        visited.add(source)
        cell = book.latest_ref(SOURCE_PROCESS_OPERATION, (source,))
        if cell is not None:
            contracts.add(cell)
            continue
        if source.page == CANONICAL_VALUE:
            frontier.extend(parent for parent, _stage in book.edges_into(source)
                            if parent.page in {CANONICAL_VALUE, INGESTION_VALUE})
    if not contracts:
        return None, None
    if len(contracts) != 1:
        raise ValueError('process operation has ambiguous concorded provenance')
    cell, = contracts
    return book.page(SOURCE_PROCESS_OPERATION).cell(cell.row, cell.column), cell


def local_operands(graph, node):
    edges, cell = operands(graph, node)
    return tuple((local_value(graph, identity), role) for identity, role in edges), cell


def captured_input_contract(graph, node):
    """Find a capture's type proof through its exact formal binding."""
    from .concordance_declarations import SOURCE_CLOSURE_CAPTURE, NAME_BINDING, THREAD_CAPTURE_CONTRACT, CANONICAL_VALUE
    from .identity_concordance import _canonical_identity_base
    from ..common.tensors.topological_reducer import node_identity_cell
    book = current_identity_book()
    identity = _canonical_identity_base(book, node_identity_cell(graph, int(node)))
    contracts = []
    for row in book.page(SOURCE_CLOSURE_CAPTURE).rows():
        capture = book.latest_ref(SOURCE_CLOSURE_CAPTURE, row)
        binding = book.page(SOURCE_CLOSURE_CAPTURE).cell(capture.row, capture.column)[2]
        value = book.page(NAME_BINDING).cell(binding.row, binding.column).value_id
        formal = book.latest_ref(CANONICAL_VALUE, (binding.row[0], int(value)))
        if formal is not None and _canonical_identity_base(book, formal) == identity:
            contract = book.latest_ref(THREAD_CAPTURE_CONTRACT, (capture,))
            if contract is not None:
                contracts.append(contract)
    if len(set(contracts)) > 1:
        raise ValueError('captured input has ambiguous type contracts')
    if not contracts:
        return None, None
    cell = contracts[0]
    return book.page(THREAD_CAPTURE_CONTRACT).cell(cell.row, cell.column), cell


def indexed_write_bases(graph):
    """Read storage writes from typed nodes and exact base operand edges."""
    from ..common.tensors.topological_reducer import node_identity_cell
    book = current_identity_book()
    target = getattr(graph, 'G', graph)
    kinds = {}
    causes = []
    for node in target:
        identity = node_identity_cell(graph, int(node))
        from .concorded_literals import origin_cell
        origin = origin_cell(book, identity)
        fact = book.page(origin.page).cell(origin.row, origin.column)
        kinds[int(node)] = str(fact.type).casefold()
    written = set()
    for node, kind in kinds.items():
        if kind not in {'indexedstore', 'index_set', 'setitem'}:
            continue
        pending = [node]
        seen = set()
        while pending:
            consumer = pending.pop()
            if consumer in seen:
                continue
            seen.add(consumer)
            edges, cell = local_operands(graph, consumer)
            causes.extend((node_identity_cell(graph, consumer), cell))
            for parent, role in edges:
                if role != 'base':
                    continue
                written.add(parent)
                if kinds[parent] in {'indexedstore', 'index_set', 'setitem', 'indexed'}:
                    pending.append(parent)
    return frozenset(written), tuple(dict.fromkeys(causes))


def thread_activation(graph, node):
    from .concordance_declarations import SOURCE_THREAD_ACTIVATION
    from .identity_concordance import _canonical_identity_base
    from ..common.tensors.topological_reducer import node_identity_cell
    book = current_identity_book()
    source = _canonical_identity_base(book, node_identity_cell(graph, int(node)))
    return book.latest_ref(SOURCE_THREAD_ACTIVATION, (source,))


def captured_columns(graph, node):
    from .concordance_declarations import THREAD_CAPTURE_COLUMNS
    contract, cell = captured_input_contract(graph, int(node))
    if cell is None:
        return None, None
    book = current_identity_book()
    columns = book.latest_ref(THREAD_CAPTURE_COLUMNS, (cell,))
    if columns is None:
        return (), cell
    return book.page(THREAD_CAPTURE_COLUMNS).cell(columns.row, columns.column), columns


def resource_scope(graph, node, *, book=None):
    """The exact ingested resource contract carried through identity copies."""
    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import SOURCE_RESOURCE_SCOPE, CANONICAL_VALUE, INGESTION_VALUE
    book = current_identity_book() if book is None else book
    pending, seen, found = [node_identity_cell(graph, int(node), book=book)], set(), set()
    while pending:
        identity = pending.pop()
        if identity in seen:
            continue
        seen.add(identity)
        cell = book.latest_ref(SOURCE_RESOURCE_SCOPE, (identity,))
        if cell is not None:
            found.add(cell)
        elif identity.page == CANONICAL_VALUE:
            pending.extend(source for source, _stage in book.edges_into(identity)
                           if source.page in {CANONICAL_VALUE, INGESTION_VALUE})
    if len(found) > 1:
        raise ValueError('resource scope has ambiguous ingested identity')
    if not found:
        return None, None
    cell, = found
    return book.page(SOURCE_RESOURCE_SCOPE).cell(cell.row, cell.column), cell
