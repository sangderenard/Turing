"""Record an authored method receiver whose source class is unproved."""

import ast

from .concordance_declarations import (
    CALL_BINDING, CLASS_METHOD_DECLARATION, PROCESS_GRAPH_OPERANDS, REDUCTION,
    SOURCE_METHOD_RECEIVER_CLASS_DEMAND, SOURCE_VALUE_CLASS,
)
from .control_activity import binding
from .identity_concordance import Derived, Mode, current_identity_book


def candidate_method_receiver_demand(graph, call_id, receiver_id, *, book):
    """Read the exact Input/Call operand and control identity only."""
    from ..common.tensors.topological_reducer import node_identity_cell

    call_id, receiver_id = int(call_id), int(receiver_id)
    nodes = graph.G.nodes
    if call_id not in graph.G or receiver_id not in graph.G:
        return None
    call = nodes[call_id]
    receiver = nodes[receiver_id]
    expression = call.get('expr_obj')
    if (receiver.get('type') not in {'Input', 'input'}
            or not isinstance(expression, ast.Call)
            or not isinstance(expression.func, ast.Attribute)):
        return None
    scope = tuple(graph.G.graph.get('authored_function_identity') or ())
    if len(scope) != 2 or book.latest_ref(SOURCE_VALUE_CLASS,
                                          (scope, receiver_id)) is not None:
        return None
    methods = tuple(
        book.latest_ref(CLASS_METHOD_DECLARATION, row)
        for row in book.page(CLASS_METHOD_DECLARATION).rows()
        if row[0] == scope[0] and row[2] == expression.func.attr
    )
    if len(methods) != 1:
        return None
    call_cell = node_identity_cell(graph, call_id, book=book)
    receiver_cell = node_identity_cell(graph, receiver_id, book=book)
    operands = book.latest_ref(PROCESS_GRAPH_OPERANDS, (
        graph.G.graph.get('operand_position_scope'), call_id,
    ))
    if (call_cell is None or receiver_cell is None or operands is None
            or (receiver_cell, 'operand') not in book.page(
                PROCESS_GRAPH_OPERANDS).cell(operands.row, operands.column)):
        return None
    if book.latest_ref(CALL_BINDING,
                       (graph.G.graph.get('lexical_read_scope'), call_id)) is not None:
        return None
    try:
        points, activity = binding(graph, call_id, book=book, scope=scope)
    except (KeyError, ValueError):
        return None
    if points is None or activity is None:
        return None
    row = (scope, call_id, receiver_id)
    causes = (call_cell, receiver_cell, operands, methods[0], activity,
              points[0], points[1])
    return row, ('missing_source_value_class', *causes), causes


def post_method_receiver_demand(graph, call_id, receiver_id, *, book=None):
    """Publish an unresolved demand, leaving method selection untouched."""
    book = current_identity_book() if book is None else book
    candidate = candidate_method_receiver_demand(
        graph, call_id, receiver_id, book=book)
    if candidate is None:
        return None
    row, fact, causes = candidate
    previous = book.latest_ref(SOURCE_METHOD_RECEIVER_CLASS_DEMAND, row)
    incumbent = (None if previous is None else book.page(
        SOURCE_METHOD_RECEIVER_CLASS_DEMAND).cell(previous.row,
                                                   previous.column))
    if incumbent == fact:
        return previous
    return book.post(SOURCE_METHOD_RECEIVER_CLASS_DEMAND, row, fact,
                     stage=REDUCTION,
                     provenance=Derived((*((previous,) if previous else ()),
                                         *causes)),
                     mode=Mode.REVISE if previous else Mode.CONCORD)
