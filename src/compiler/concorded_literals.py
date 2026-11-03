"""Read and evaluate admitted scalar literal graph nodes through concordance."""
import operator
from .concordance_declarations import (
    CANONICAL_VALUE, INGESTION_VALUE, SOURCE_LITERAL_VALUE, PROVEN_LITERAL,
    SOURCE_NUMERIC_SPECIALIZATION_PAGE, SYNCHRONIZATION_LITERAL, SOURCE_OPERATOR_KIND, CONTROL_PROGRAM_BUILD,
)
from .identity_concordance import Derived, Mode, current_identity_book
from ..transmogrifier.ssa_registry import Handler

MISSING = object()
_INTEGER_OPERATORS = {Handler.Or: operator.or_, Handler.And: operator.and_,
                      Handler.Xor: operator.xor, Handler.Neg: operator.neg,
                      Handler.Not: operator.invert, Handler.Add: operator.add,
                      Handler.Sub: operator.sub, Handler.Mul: operator.mul}


def origin_cell(book, cell):
    seen = set()
    while cell.page == CANONICAL_VALUE:
        if cell in seen:
            raise ValueError('cyclic canonical identity')
        seen.add(cell)
        parents = tuple(source for source, _stage in book.edges_into(cell)
                        if source.page in {CANONICAL_VALUE, INGESTION_VALUE})
        if len(parents) != 1:
            raise ValueError('canonical node has no unique ingested identity')
        cell = parents[0]
    return cell


def literal(graph, node, seen=()):
    from ..common.tensors.topological_reducer import node_identity_cell, _source_numeric_scope
    if node in seen:
        return MISSING, ()
    book = current_identity_book()
    node_cell = node_identity_cell(graph, int(node))
    scope = _source_numeric_scope(graph)
    local_scope = graph.G.graph.get('lexical_read_scope')
    scopes = list(dict.fromkeys(
        candidate for candidate in (local_scope, scope)
        if candidate is not None
    ))
    if isinstance(scope, tuple) and len(scope) == 3:
        specialization = book.latest_ref(SOURCE_NUMERIC_SPECIALIZATION_PAGE, (scope[0], scope[2]))
        if specialization is not None and book.page(specialization.page).cell(specialization.row, specialization.column)[0] == scope:
            scopes.append(scope[0])
    for candidate in scopes:
        cell = book.latest_ref(SOURCE_LITERAL_VALUE, (candidate, int(node)))
        if cell is not None:
            return book.page(cell.page).cell(cell.row, cell.column), (cell,)
    # An unrolled literal is born in the clone's lexical scope. A later
    # callsite fork copies its canonical identity, not its authored numeric
    # scope. Follow only that exact copy relation to the literal's own row;
    # walking arbitrary computation causes would mistake an operand's
    # literal for its consumer's result.
    pending, visited = [node_cell], set()
    while pending:
        identity = pending.pop()
        if identity in visited:
            continue
        visited.add(identity)
        if identity.page != CANONICAL_VALUE:
            continue
        cell = book.latest_ref(SOURCE_LITERAL_VALUE, identity.row)
        if cell is not None:
            return book.page(cell.page).cell(cell.row, cell.column), (cell,)
        pending.extend(source for source, stage in book.edges_into(identity)
                       if source.page == CANONICAL_VALUE
                       and stage.name == 'read_scope_fork')
    cell = book.latest_ref(PROVEN_LITERAL, (str(graph.G.graph.get('function_name')), int(node)))
    if cell is not None:
        return book.page(cell.page).cell(cell.row, cell.column), (cell,)
    source = origin_cell(book, node_cell)
    if source.page != INGESTION_VALUE:
        return MISSING, ()
    kind_cell = book.latest_ref(SOURCE_OPERATOR_KIND, (source,))
    if kind_cell is None:
        return MISSING, ()
    handler = Handler(book.page(SOURCE_OPERATOR_KIND).cell(kind_cell.row, kind_cell.column))
    operation = _INTEGER_OPERATORS.get(handler)
    if operation is None:
        return MISSING, ()
    from .process_graph_facts import operands as graph_operands
    edge_facts, edge_cell = graph_operands(graph, node)
    from .process_graph_facts import local_value
    parents = {role: local_value(graph, cell) for cell, role in edge_facts}
    if 'lhs' in parents and 'rhs' in parents:
        operands = (parents['lhs'], parents['rhs'])
    elif 'operand' in parents:
        operands = (parents['operand'],)
    else:
        return MISSING, ()
    from .concordance_declarations import LEXICAL_READ_BINDING
    from ..common.tensors.topological_reducer import _operand_position_scope
    edge_scope = _operand_position_scope(graph)
    # A literal producer used through a mutable lexical binding is not a
    # constant-expression proof. Do not freeze a loop-carried initial value.
    for role in parents:
        if book.latest_ref(LEXICAL_READ_BINDING, (edge_scope, int(node), role, 0)) is not None:
            return MISSING, ()
    values, causes = [], [node_cell, source, kind_cell, edge_cell]
    for parent in operands:
        value, cells = literal(graph, parent, (*seen, node))
        if type(value) not in {int, bool} and not isinstance(value, int):
            return MISSING, ()
        values.append(int(value)); causes.extend(cells)
    value = operation(*values)
    cell = book.post(SYNCHRONIZATION_LITERAL, (node_cell,),
                     (handler.value, tuple(node_identity_cell(graph, n) for n in operands), value),
                     stage=CONTROL_PROGRAM_BUILD, provenance=Derived(tuple(dict.fromkeys(causes))), mode=Mode.CONCORD)
    return value, (cell,)
