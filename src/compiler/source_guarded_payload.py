"""Read exact source cells for a mapping payload on one present control arm."""

from .concordance_declarations import (
    CANONICAL_VALUE, INGESTION_VALUE, NAME_BINDING, PROCESS_GRAPH_OPERANDS,
    REDUCTION, SOURCE_ACTIVITY_BINDING, SOURCE_ACTIVITY_EDGE,
    SOURCE_ACTIVITY_NODE, SOURCE_GUARDED_MAPPING_PAYLOAD,
    SOURCE_LITERAL_VALUE, SOURCE_MAPPING_GET_READ, SOURCE_OPERATOR_KIND,
    SOURCE_RESOURCE_ADMISSION_DEMAND, SOURCE_SPAN, SOURCE_TERMINAL_RETURN,
)
from .identity_concordance import Derived, Mode


def _fact(book, cell):
    return book.page(cell.page).cell(cell.row, cell.column)


def _sources(book, cell):
    return {source for source, _stage in book.edges_into(cell)}


def _single(items):
    items = tuple(items)
    return items[0] if len(items) == 1 else None


def _ingestion(book, canonical):
    return _single(cell for cell in _sources(book, canonical)
                   if cell.page == INGESTION_VALUE)


def _span(book, ingestion):
    return _single(cell for cell in _sources(book, ingestion)
                   if cell.page == SOURCE_SPAN)


def _edge(book, scope, before, after, relation):
    return book.latest_ref(SOURCE_ACTIVITY_EDGE,
                           (scope, before, after, relation))


def _canonical_rows(book, lexical_scope):
    return tuple((int(row[1]), book.latest_ref(CANONICAL_VALUE, row))
                 for row in book.page(CANONICAL_VALUE).scope_rows(lexical_scope))


def _assignment(book, scope, get_binding, name_binding):
    """The authored target binding is the get result's unique next write."""
    get_exit = _fact(book, get_binding)[1]
    outgoing = tuple(row for row in book.page(SOURCE_ACTIVITY_EDGE).scope_rows(scope)
                     if row[1] == get_exit)
    if len(outgoing) != 1 or outgoing[0][3] != 'flow':
        return None
    target_enter = outgoing[0][2]
    if _fact(book, target_enter)[0] != 'Name' or target_enter.row[2] != 'enter':
        return None
    target_span = target_enter.row[1]
    binding_sources = _sources(book, name_binding)
    source_binding = _single(source for source in binding_sources
                             if source.page == NAME_BINDING)
    if (source_binding is None or source_binding.row[1:] != name_binding.row[1:]
            or target_span not in _sources(book, source_binding)):
        return None
    target_exit = book.latest_ref(SOURCE_ACTIVITY_NODE,
                                  (scope, target_span, 'exit'))
    assign_end = _single(row[2] for row in
                         book.page(SOURCE_ACTIVITY_EDGE).scope_rows(scope)
                         if row[1] == target_exit and row[3] == 'flow'
                         and _fact(book, row[2])[0] == 'Assign'
                         and row[2].row[2] == 'exit')
    if assign_end is None:
        return None
    return (source_binding, target_span,
            book.latest_ref(SOURCE_ACTIVITY_EDGE, outgoing[0]),
            _edge(book, scope, target_enter, target_exit, 'flow'),
            _edge(book, scope, target_exit, assign_end, 'flow'),
            assign_end)


def candidate_facts(book, scope, get_id, read):
    """Yield fully proved views; any absent/ambiguous cell refuses publication."""
    read_fact = _fact(book, read)
    get_value = read_fact[7]
    if get_value.page != CANONICAL_VALUE or get_value.row[1] != get_id:
        return ()
    lexical_scope = get_value.row[0]
    canonicals = _canonical_rows(book, lexical_scope)
    bindings = tuple(book.latest_ref(NAME_BINDING, row) for row in
                     book.page(NAME_BINDING).scope_rows(lexical_scope)
                     if (binding := book.page(NAME_BINDING).latest(row))
                     and binding.authored and int(binding.value_id) == get_id)
    name_binding = _single(bindings)
    if name_binding is None or get_value not in _sources(book, name_binding):
        return ()
    assignment = _assignment(book, scope, read_fact[13], name_binding)
    if assignment is None or any(item is None for item in assignment):
        return ()
    results = []
    for compare_id, compare_value in canonicals:
        ingestion = _ingestion(book, compare_value)
        if ingestion is None:
            continue
        kind = book.latest_ref(SOURCE_OPERATOR_KIND, (ingestion,))
        if kind is None or _fact(book, kind) != 'compare:Is':
            continue
        operands = book.latest_ref(PROCESS_GRAPH_OPERANDS,
                                   (lexical_scope, compare_id))
        if operands is None:
            continue
        edges = _fact(book, operands)
        if len(edges) != 2 or edges[0] != (get_value, 'lhs') or edges[1][1] != 'rhs':
            continue
        none_value = edges[1][0]
        literal = book.latest_ref(SOURCE_LITERAL_VALUE,
                                  (scope, none_value.row[1]))
        if literal is None or _fact(book, literal) is not None:
            continue
        compare_span = _span(book, ingestion)
        if compare_span is None or _fact(book, compare_span).kind != 'Compare':
            continue
        for if_id, if_value in canonicals:
            if_operands = book.latest_ref(PROCESS_GRAPH_OPERANDS,
                                          (lexical_scope, if_id))
            if if_operands is None or _fact(book, if_operands) != ((compare_value, 'test'),):
                continue
            if_ingestion = _ingestion(book, if_value)
            if_span = None if if_ingestion is None else _span(book, if_ingestion)
            if if_span is None or _fact(book, if_span).kind != 'If':
                continue
            if_binding = book.latest_ref(SOURCE_ACTIVITY_BINDING,
                                         (scope, if_ingestion))
            if if_binding is None:
                continue
            if_exit = _fact(book, if_binding)[1]
            select = book.latest_ref(SOURCE_ACTIVITY_NODE,
                                     (scope, if_span, 'select'))
            present_edge = _edge(book, scope, select, if_exit, 'orelse')
            if present_edge is None:
                continue
            incoming_if = tuple(row for row in
                                book.page(SOURCE_ACTIVITY_EDGE).scope_rows(scope)
                                if row[2] == if_exit)
            if incoming_if != (present_edge.row,):
                continue
            terminal_rows = tuple(row for row in
                                  book.page(SOURCE_TERMINAL_RETURN).scope_rows(scope)
                                  if (if_span, 'body') in _fact(book, book.latest_ref(
                                      SOURCE_TERMINAL_RETURN, row))[2])
            terminal_row = _single(terminal_rows)
            if terminal_row is None:
                continue
            terminal = book.latest_ref(SOURCE_TERMINAL_RETURN, terminal_row)
            return_enter, return_exit, _owners = _fact(book, terminal)
            absent_edge = _edge(book, scope, select, return_enter, 'body')
            if absent_edge is None or any(
                row[1] == return_exit for row in
                book.page(SOURCE_ACTIVITY_EDGE).scope_rows(scope)
            ):
                continue
            for demand_row in book.page(SOURCE_RESOURCE_ADMISSION_DEMAND).scope_rows(scope):
                demand = book.latest_ref(SOURCE_RESOURCE_ADMISSION_DEMAND,
                                         demand_row)
                _with_source, with_binding, calls = _fact(book, demand)
                if len(calls) != 1:
                    continue
                call_ingestion = calls[0][0]
                borrowed = _single((call_id, canonical)
                                   for call_id, canonical in canonicals
                                   if call_ingestion in _sources(book, canonical))
                if borrowed is None:
                    continue
                borrowed_id, borrowed_value = borrowed
                borrowed_operands = book.latest_ref(PROCESS_GRAPH_OPERANDS,
                    (lexical_scope, borrowed_id))
                if (borrowed_operands is None
                        or (get_value, 'operand') not in _fact(book, borrowed_operands)):
                    continue
                with_enter = _fact(book, with_binding)[0]
                continuation = _edge(book, scope, if_exit, with_enter, 'flow')
                if continuation is None:
                    continue
                incoming_with = tuple(row for row in
                                      book.page(SOURCE_ACTIVITY_EDGE).scope_rows(scope)
                                      if row[2] == with_enter)
                if incoming_with != (continuation.row,):
                    continue
                # Program: the get result is assigned once, tested by `is
                # None`, and only the nonreturning orelse reaches borrowed.
                # The stored payload is valid for this call arm only.
                facts = (read, read_fact[5], name_binding, *assignment,
                         kind, compare_value, operands, literal, compare_span,
                         if_value, if_operands, if_binding, select,
                         terminal, absent_edge, present_edge, continuation,
                         demand, with_binding, borrowed_value,
                         borrowed_operands)
                results.append(((scope, get_id, borrowed_id), facts))
    return tuple(results)


def post_guarded_mapping_payloads(book):
    for row in tuple(book.page(SOURCE_MAPPING_GET_READ).rows()):
        read = book.latest_ref(SOURCE_MAPPING_GET_READ, row)
        candidates = candidate_facts(book, row[0], int(row[1]), read)
        if len(candidates) != 1:
            continue
        target, facts = candidates[0]
        book.post(SOURCE_GUARDED_MAPPING_PAYLOAD, target, facts,
                  stage=REDUCTION, provenance=Derived(tuple(dict.fromkeys(facts))),
                  mode=Mode.CONCORD)
