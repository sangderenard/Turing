"""Name an unproved constructor receiver/effect bridge at its exact Call."""

import ast

from .concordance_declarations import (
    CLASS_DECLARATION, CLASS_METHOD_DECLARATION, FUNCTION_ADDRESS,
    PROCESS_GRAPH_OPERANDS, REDUCTION, SOURCE_ACTIVITY_CALL,
    SOURCE_CALL_OBJECT_ALIAS, SOURCE_CONSTRUCTOR_FIELD_EFFECT_DEMAND,
    SOURCE_FUNCTION_EXIT_FIELD_STATE, SOURCE_FUNCTION_REACHABILITY,
    SOURCE_VALUE_CLASS,
)
from .control_activity import binding
from .identity_concordance import Derived, Mode, current_identity_book
from ..transmogrifier.function_table import function_address_identity


def candidate_constructor_field_demands(function_table, *, book):
    """Yield exact missing constructor bridges, without a field transfer."""
    from ..common.tensors.topological_reducer import node_identity_cell

    entries = tuple(entry for entry in function_table
                    if getattr(entry, 'graph', None) is not None)
    by_scope = {
        tuple(entry.graph.G.graph.get('authored_function_identity') or ()): entry
        for entry in entries
    }
    for caller_entry in entries:
        caller = caller_entry.graph
        scope = tuple(caller.G.graph.get('authored_function_identity') or ())
        if len(scope) != 2:
            continue
        for call_id, data in caller.G.nodes(data=True):
            call_id = int(call_id)
            if not isinstance(data.get('expr_obj'), ast.Call):
                continue
            class_cell = book.latest_ref(SOURCE_VALUE_CLASS, (scope, call_id))
            if class_cell is None:
                continue
            class_fact = book.page(SOURCE_VALUE_CLASS).cell(
                class_cell.row, class_cell.column)
            if (not isinstance(class_fact, tuple) or len(class_fact) != 3
                    or class_fact[2] != 'source class construction'):
                continue
            class_name = str(class_fact[0])
            class_decl = book.latest_ref(CLASS_DECLARATION,
                                         (scope[0], class_name))
            method = book.latest_ref(CLASS_METHOD_DECLARATION,
                                     (scope[0], class_name, '__init__'))
            if class_decl is None or method is None:
                continue
            method_fact = book.page(CLASS_METHOD_DECLARATION).cell(
                method.row, method.column)
            init_scope = (scope[0], method_fact.graph_identity)
            init_entry = by_scope.get(init_scope)
            if (init_entry is None or not method_fact.parameters
                    or method_fact.parameters[0] != 'self'):
                continue
            address = book.latest_ref(
                FUNCTION_ADDRESS, (function_address_identity(init_entry),))
            if (address is None or book.page(FUNCTION_ADDRESS).cell(
                    address.row, address.column,
            ) != int(init_entry.reference.address)):
                continue
            call_cell = node_identity_cell(caller, call_id, book=book)
            operand = book.latest_ref(PROCESS_GRAPH_OPERANDS, (
                caller.G.graph.get('operand_position_scope'), call_id,
            ))
            if operand is None:
                continue
            operand_fact = book.page(PROCESS_GRAPH_OPERANDS).cell(
                operand.row, operand.column)
            if len(operand_fact) != 1 or operand_fact[0][1] != 'callee':
                continue
            callee_cell = operand_fact[0][0]
            if (call_cell is None or callee_cell.row[0] != call_cell.row[0]
                    or not isinstance(data.get('expr_obj').func, ast.Name)
                    or data['expr_obj'].func.id != class_name):
                continue
            try:
                points, call_binding = binding(
                    caller, call_id, book=book, scope=scope)
            except (KeyError, ValueError):
                continue
            if points is None or call_binding is None:
                continue
            activity_call = book.latest_ref(
                SOURCE_ACTIVITY_CALL, (scope, points[0].row[1]))
            if (activity_call is None or book.page(SOURCE_ACTIVITY_CALL).cell(
                    activity_call.row, activity_call.column) != (0, ())):
                continue
            if book.latest_ref(SOURCE_FUNCTION_REACHABILITY,
                               (scope, call_id, init_scope)) is not None:
                continue
            for exit_row in book.page(SOURCE_FUNCTION_EXIT_FIELD_STATE).rows():
                if exit_row[0] != init_scope or exit_row[1] != 0:
                    continue
                field_exit = book.latest_ref(SOURCE_FUNCTION_EXIT_FIELD_STATE,
                                             exit_row)
                if book.latest_ref(SOURCE_CALL_OBJECT_ALIAS,
                                   (scope, call_id, init_scope, 0)) is not None:
                    continue
                row = (scope, call_id, init_scope, exit_row[2])
                causes = (call_cell, class_cell, class_decl, method, address,
                          field_exit, operand, callee_cell, call_binding,
                          points[0], points[1], activity_call)
                yield row, ('constructor_receiver_effect_unproved', *causes), causes


def post_constructor_field_demands(function_table, *, book=None):
    """Publish only the missing bridge; no constructor effect is assumed."""
    book = current_identity_book() if book is None else book
    for row, fact, causes in candidate_constructor_field_demands(
            function_table, book=book):
        book.post(SOURCE_CONSTRUCTOR_FIELD_EFFECT_DEMAND, row, fact,
                  stage=REDUCTION, provenance=Derived(causes),
                  mode=Mode.CONCORD)
