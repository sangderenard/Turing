"""Read loop exit state by exact ingested site identity, never coordinates."""
from .concordance_declarations import SOURCE_LOOP_SITE_EDGE, SOURCE_LOOP_SITE_VIEW, CANONICAL_VALUE, INGESTION_VALUE, CONTROL_PROGRAM_BUILD
from .identity_concordance import current_identity_book, Derived, Mode


def site_values(graph, site_id, loop_id=None, *, book=None):
    from ..common.tensors.topological_reducer import node_identity_cell, _operand_position_scope
    book = current_identity_book() if book is None else book
    def origins(identity):
        pending, seen, result = [identity], set(), set()
        while pending:
            cell = pending.pop()
            if cell in seen:
                continue
            seen.add(cell)
            if cell.page == INGESTION_VALUE:
                result.add(cell)
            elif cell.page == CANONICAL_VALUE:
                pending.extend(parent for parent, _stage in book.edges_into(cell)
                               if parent.page in {CANONICAL_VALUE, INGESTION_VALUE})
        return result
    sources = origins(node_identity_cell(graph, int(site_id), book=book))
    owners = None if loop_id is None else origins(node_identity_cell(graph, int(loop_id), book=book))
    matches = tuple(book.latest_ref(SOURCE_LOOP_SITE_EDGE, row)
                    for row in book.page(SOURCE_LOOP_SITE_EDGE).rows()
                    if row[1] in sources and (owners is None or row[0] in owners))
    if len(matches) != 1:
        raise ValueError(f'loop control site has no unique concorded edge: {sources}')
    cell = matches[0]
    indices, identity_cells = {}, {}
    scope = graph.G.graph.get('lexical_read_scope') or _operand_position_scope(graph)
    for row in book.page(CANONICAL_VALUE).scope_rows(scope):
        identity = book.latest_ref(CANONICAL_VALUE, row)
        identity_cells[int(row[1])] = identity
        for origin in origins(identity):
            indices.setdefault(origin, set()).add(int(row[1]))
    for node in graph.G:
        identity = node_identity_cell(graph, int(node), book=book)
        identity_cells[int(node)] = identity
        for origin in origins(identity):
            indices.setdefault(origin, set()).add(int(node))
    def local(identity):
        candidates = indices.get(identity, ())
        live = set(candidates) & set(graph.G)
        selected = live or set(candidates)
        if len(selected) > 1:
            raise ValueError('loop site value has multiple local identities')
        return next(iter(selected), None)
    result = {}
    for initial, updated, value in book.page(SOURCE_LOOP_SITE_EDGE).cell(cell.row, cell.column):
        initial_id, updated_id, value_id = local(initial), local(updated), local(value)
        if None not in (initial_id, updated_id, value_id):
            result[(initial_id, updated_id)] = value_id
    row = (scope, int(site_id))
    proposal = (cell, tuple((initial, updated, value) for (initial, updated), value in result.items()))
    previous = book.latest_ref(SOURCE_LOOP_SITE_VIEW, row)
    if previous is None or book.page(SOURCE_LOOP_SITE_VIEW).cell(previous.row, previous.column) != proposal:
        previous = book.post(SOURCE_LOOP_SITE_VIEW, row, proposal, stage=CONTROL_PROGRAM_BUILD,
            provenance=Derived(tuple(dict.fromkeys((cell, *(identity_cells[value] for triple in proposal[1]
                                                            for value in triple))))),
            mode=Mode.CONCORD if previous is None else Mode.REVISE)
    _edge, projected = book.page(SOURCE_LOOP_SITE_VIEW).cell(previous.row, previous.column)
    return {(initial, updated): value for initial, updated, value in projected}, previous
