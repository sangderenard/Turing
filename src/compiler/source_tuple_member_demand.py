"""Name a missing callee-local tuple member after an exact source call."""

import ast

from .concordance_declarations import (
    CANONICAL_VALUE, PROCESS_GRAPH_OPERANDS, REDUCTION,
    SOURCE_AGGREGATE_LAYOUT, SOURCE_CALL_ACTUAL_FORMAL,
    SOURCE_CALL_TUPLE_MEMBER_PROJECTION_DEMAND, SOURCE_FUNCTION_REACHABILITY,
    SOURCE_LITERAL_VALUE, SOURCE_VALUE_CLASS,
)
from .control_activity import binding
from .identity_concordance import Derived, Mode, current_identity_book


def candidate_tuple_member_demands(function_table, *, book):
    """Read exact call, tuple and Indexed cells; claim no member transfer."""
    from ..common.tensors.topological_reducer import node_identity_cell

    entries = tuple(entry for entry in function_table
                    if getattr(entry, 'graph', None) is not None)
    by_scope = {
        tuple(entry.graph.G.graph.get('authored_function_identity') or ()): entry.graph
        for entry in entries
    }
    for binding_row in book.page(SOURCE_CALL_ACTUAL_FORMAL).rows():
        caller_scope, call_id, callee_scope, formal_id = binding_row
        caller, callee = by_scope.get(caller_scope), by_scope.get(callee_scope)
        if caller is None or callee is None or formal_id not in callee.G:
            continue
        call_binding = book.latest_ref(SOURCE_CALL_ACTUAL_FORMAL, binding_row)
        actual, role, _name, caller_operands, _formal_binding = (
            book.page(SOURCE_CALL_ACTUAL_FORMAL).cell(
                call_binding.row, call_binding.column)
        )
        reach = book.latest_ref(SOURCE_FUNCTION_REACHABILITY,
                                (caller_scope, call_id, callee_scope))
        formal_class = book.latest_ref(SOURCE_VALUE_CLASS,
                                       (callee_scope, formal_id))
        if (reach is None or formal_class is None or actual.page != CANONICAL_VALUE
                or role != 'arg:0' or book.page(SOURCE_VALUE_CLASS).cell(
                    formal_class.row, formal_class.column)[0] != 'builtins.tuple'):
            continue
        layout = book.latest_ref(SOURCE_AGGREGATE_LAYOUT, (
            caller.G.graph.get('source_aggregate_layout_scope'),
            actual.row[-1],
        ))
        layout_fact = (None if layout is None else
                       book.page(SOURCE_AGGREGATE_LAYOUT).cell(
                           layout.row, layout.column))
        if layout_fact is None or layout_fact.aggregate_kind != 'tuple':
            continue
        caller_operand_fact = book.page(PROCESS_GRAPH_OPERANDS).cell(
            caller_operands.row, caller_operands.column)
        if (actual, role) not in caller_operand_fact:
            continue
        try:
            caller_points, caller_activity = binding(
                caller, call_id, book=book, scope=caller_scope)
        except (KeyError, ValueError):
            continue
        if caller_points is None or caller_activity is None:
            continue
        formal_cell = node_identity_cell(callee, formal_id, book=book)
        for indexed_id, data in callee.G.nodes(data=True):
            if data.get('type') != 'Indexed' or not isinstance(
                    data.get('expr_obj'), ast.Subscript):
                continue
            indexed_id = int(indexed_id)
            indexed = node_identity_cell(callee, indexed_id, book=book)
            operands = book.latest_ref(PROCESS_GRAPH_OPERANDS, (
                callee.G.graph.get('operand_position_scope'), indexed_id,
            ))
            if operands is None or indexed is None:
                continue
            operand_fact = book.page(PROCESS_GRAPH_OPERANDS).cell(
                operands.row, operands.column)
            indices = tuple(cell for cell, role in operand_fact
                            if role == 'index')
            if (len(indices) != 1 or set(operand_fact) != {
                    (formal_cell, 'base'), (indices[0], 'index')}):
                continue
            index_cell = indices[0]
            index_literal = book.latest_ref(SOURCE_LITERAL_VALUE,
                                            (callee_scope, index_cell.row[-1]))
            if index_literal is None:
                continue
            index = book.page(SOURCE_LITERAL_VALUE).cell(
                index_literal.row, index_literal.column)
            if (type(index) is not int or index < 0
                    or index >= len(layout_fact.members)):
                continue
            if book.latest_ref(SOURCE_AGGREGATE_LAYOUT, (
                    callee.G.graph.get('source_aggregate_layout_scope'), formal_id
            )) is not None:
                continue
            member = layout_fact.members[index].value_cell
            if (member.page != CANONICAL_VALUE
                    or member.row[0] != actual.row[0]):
                continue
            try:
                indexed_points, indexed_activity = binding(
                    callee, indexed_id, book=book, scope=callee_scope)
            except (KeyError, ValueError):
                continue
            if indexed_points is None or indexed_activity is None:
                continue
            row = (caller_scope, call_id, callee_scope,
                   formal_id, indexed_id, index)
            causes = (reach, call_binding, actual, formal_cell, formal_class,
                      layout, member, caller_operands, indexed, operands,
                      index_cell, index_literal, caller_activity,
                      caller_points[0], indexed_activity, indexed_points[0])
            yield row, ('callee_member_identity_unpublished', *causes), causes


def post_tuple_member_demands(function_table, *, book=None):
    """Publish the missing member identity as a named unresolved demand."""
    book = current_identity_book() if book is None else book
    for row, fact, causes in candidate_tuple_member_demands(
            function_table, book=book):
        book.post(SOURCE_CALL_TUPLE_MEMBER_PROJECTION_DEMAND, row, fact,
                  stage=REDUCTION, provenance=Derived(causes),
                  mode=Mode.CONCORD)
