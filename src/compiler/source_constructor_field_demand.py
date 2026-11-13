"""Name an unproved constructor receiver/effect bridge at its exact Call."""

import ast

from .concordance_declarations import (
    CLASS_DECLARATION, CLASS_METHOD_DECLARATION, FUNCTION_ADDRESS,
    PROCESS_GRAPH_OPERANDS, REDUCTION, SOURCE_ACTIVITY_CALL,
    SOURCE_CALL_OBJECT_ALIAS, SOURCE_CONSTRUCTOR_FIELD_EFFECT_DEMAND,
    SOURCE_FUNCTION_EXIT_FIELD_STATE,
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
            # The bridge is the alias at ``__init__``'s exact ``self`` Input
            # (candidate_constructor_receiver_aliases), not formal 0.  A class
            # Call now has SOURCE_FUNCTION_REACHABILITY to its ``__init__``
            # (the reducer's call loop, constructor_initializer) whether or
            # not that alias is proven, so reachability no longer answers it.
            self_id, _self_cell, _self_binding = _initializer_self_binding(
                init_entry, book=book)
            for exit_row in book.page(SOURCE_FUNCTION_EXIT_FIELD_STATE).rows():
                if exit_row[0] != init_scope or exit_row[1] != 0:
                    continue
                field_exit = book.latest_ref(SOURCE_FUNCTION_EXIT_FIELD_STATE,
                                             exit_row)
                if self_id is not None and book.latest_ref(
                        SOURCE_CALL_OBJECT_ALIAS,
                        (scope, call_id, init_scope, int(self_id))) is not None:
                    continue
                row = (scope, call_id, init_scope, exit_row[2])
                causes = (call_cell, class_cell, class_decl, method, address,
                          field_exit, operand, callee_cell, call_binding,
                          points[0], points[1], activity_call)
                yield row, ('constructor_receiver_effect_unproved', *causes), causes


def _initializer_self_binding(init_entry, *, book):
    """``__init__``'s authored ``self`` Input and its exact NAME_BINDING cell."""
    from ..common.tensors.topological_reducer import node_identity_cell
    from .concordance_declarations import NAME_BINDING

    graph = init_entry.graph
    formals = tuple(
        int(node_id) for node_id, data in graph.G.nodes(data=True)
        if data.get('type') == 'Input'
        and (data.get('attributes') or {}).get('binding_name') == 'self'
    )
    if len(formals) != 1:
        return None, None, None
    formal_cell = node_identity_cell(graph, formals[0], book=book)
    if formal_cell is None:
        return None, None, None
    names = book.page(NAME_BINDING)
    bindings = tuple(
        book.latest_ref(NAME_BINDING, row)
        for row in names.scope_rows(formal_cell.row[0])
        if row[1] == 'self'
        and (fact := names.latest(row)) is not None
        and fact.authored and int(fact.value_id) == formals[0]
    )
    if len(bindings) != 1:
        return None, None, None
    return formals[0], formal_cell, bindings[0]


def constructor_initializer(caller, scope, call_id, by_scope, *, book):
    """The ``__init__`` a source class Call runs, and the cells that prove it.

    A Call whose SOURCE_VALUE_CLASS is a ``source class construction`` of
    class C, whose only callee operand is C's Name, where C declares
    ``__init__`` with a ``self`` receiver and no ``__new__``, runs that
    ``__init__`` (FunctionTable entry by its authored identity, at its
    published FUNCTION_ADDRESS).  Returns ``(entry, init_scope, (call_cell,
    class_cell, class_declaration, method_declaration, address, operand,
    callee_cell))`` or None.
    """
    from ..common.tensors.topological_reducer import node_identity_cell

    data = caller.G.nodes[int(call_id)]
    expression = data.get('expr_obj')
    if not isinstance(expression, ast.Call) or len(scope) != 2:
        return None
    class_cell = book.latest_ref(SOURCE_VALUE_CLASS, (scope, int(call_id)))
    if class_cell is None:
        return None
    class_fact = book.page(SOURCE_VALUE_CLASS).cell(
        class_cell.row, class_cell.column)
    if (not isinstance(class_fact, tuple) or len(class_fact) != 3
            or class_fact[2] != 'source class construction'):
        return None
    class_name = str(class_fact[0])
    class_decl = book.latest_ref(CLASS_DECLARATION, (scope[0], class_name))
    method = book.latest_ref(CLASS_METHOD_DECLARATION,
                             (scope[0], class_name, '__init__'))
    if (class_decl is None or method is None
            or book.latest_ref(CLASS_METHOD_DECLARATION, (
                scope[0], class_name, '__new__')) is not None):
        return None
    method_fact = book.page(CLASS_METHOD_DECLARATION).cell(
        method.row, method.column)
    init_scope = (scope[0], method_fact.graph_identity)
    init_entry = by_scope.get(init_scope)
    if (init_entry is None or not method_fact.parameters
            or method_fact.parameters[0] != 'self'):
        return None
    address = book.latest_ref(
        FUNCTION_ADDRESS, (function_address_identity(init_entry),))
    if (address is None or book.page(FUNCTION_ADDRESS).cell(
            address.row, address.column,
    ) != int(init_entry.reference.address)):
        return None
    call_cell = node_identity_cell(caller, int(call_id), book=book)
    operand = book.latest_ref(PROCESS_GRAPH_OPERANDS, (
        caller.G.graph.get('operand_position_scope'), int(call_id),
    ))
    if call_cell is None or operand is None:
        return None
    callees = tuple(
        cell for cell, role in book.page(PROCESS_GRAPH_OPERANDS).cell(
            operand.row, operand.column)
        if role == 'callee'
    )
    if (len(callees) != 1 or callees[0].row[0] != call_cell.row[0]
            or not isinstance(expression.func, ast.Name)
            or expression.func.id != class_name
            or call_cell.row[-1] != int(call_id)):
        return None
    return init_entry, init_scope, (
        call_cell, class_cell, class_decl, method, address, operand, callees[0])


def candidate_constructor_receiver_aliases(function_table, *, book):
    """Yield the proved constructed-object alias of each source class Call.

    Program: ``wrapper = CompositeGeometry("tetrahedron", "float64", "cpu")``
    then ``wrapper.configure_geometry(...)`` -> ``self.define_offsets(...)``,
    which reads ``dtype=self.precision``.  ``precision`` is written once, in
    ``__init__`` on its ``self``; the reducer's call aliases bind an operand
    to a formal, and the class Call has no operand for ``self``, so no record
    joined ``__init__``'s ``self`` to the constructed value and the
    single-write transfer stopped at ``__init__`` (define_offsets region
    value 80 had no settled dtype).  Rule: a Call whose value class is a
    ``source class construction`` of class C, whose only callee operand is
    C's Name, where C declares ``__init__`` (and no ``__new__``) with a
    ``self`` receiver whose memory contract is the caller's record alias,
    posts SOURCE_CALL_OBJECT_ALIAS (caller, Call, __init__, self) with the
    Call's own value as the actual: the object ``__init__`` receives as
    ``self`` IS the value the Call produces.  This is the bridge
    SOURCE_CONSTRUCTOR_FIELD_EFFECT_DEMAND names.
    """
    from .concordance_declarations import SOURCE_PARAMETER_MEMORY_CONTRACT
    from ..transmogrifier.function_table import (
        ParameterAccess, ParameterContract, ParameterScope, ParameterStorage,
        ParameterTransfer,
    )

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
            constructed = constructor_initializer(
                caller, scope, call_id, by_scope, book=book)
            if constructed is None:
                continue
            init_entry, init_scope, cells = constructed
            call_cell, class_cell, class_decl, method, address, operand, callee = cells
            self_id, self_cell, self_binding = _initializer_self_binding(
                init_entry, book=book)
            if self_binding is None:
                continue
            memory = book.latest_ref(SOURCE_PARAMETER_MEMORY_CONTRACT,
                                     (init_scope, 'self'))
            memory_fact = (None if memory is None else
                           book.page(SOURCE_PARAMETER_MEMORY_CONTRACT).cell(
                               memory.row, memory.column))
            contract = (memory_fact[0] if isinstance(memory_fact, tuple)
                        and len(memory_fact) == 1 else None)
            memory_sources = (set() if memory is None else
                              {source for source, _stage in book.edges_into(memory)})
            if not (isinstance(contract, ParameterContract)
                    and contract.name == 'self'
                    and contract.transfer is ParameterTransfer.ALIAS
                    and contract.access is ParameterAccess.INOUT
                    and contract.storage is ParameterStorage.RECORD
                    and contract.scope is ParameterScope.CALLER
                    and {method, class_decl, self_binding} <= memory_sources):
                continue
            row = (scope, call_id, init_scope, int(self_id))
            fact = (call_cell, self_binding, memory, class_cell, class_decl)
            causes = (call_cell, self_cell, self_binding, memory, class_cell,
                      class_decl, method, address, operand, callee)
            yield row, fact, causes


def post_constructor_receiver_aliases(function_table, *, book=None):
    """Publish each proved constructed-object alias."""
    from .concordance_declarations import SOURCE_CALL_OBJECT_ALIAS

    book = current_identity_book() if book is None else book
    for row, fact, causes in tuple(candidate_constructor_receiver_aliases(
            function_table, book=book)):
        book.post(SOURCE_CALL_OBJECT_ALIAS, row, fact,
                  stage=REDUCTION, provenance=Derived(causes),
                  mode=Mode.CONCORD)


def post_constructor_field_demands(function_table, *, book=None):
    """Publish only the missing bridge; no constructor effect is assumed."""
    book = current_identity_book() if book is None else book
    for row, fact, causes in candidate_constructor_field_demands(
            function_table, book=book):
        book.post(SOURCE_CONSTRUCTOR_FIELD_EFFECT_DEMAND, row, fact,
                  stage=REDUCTION, provenance=Derived(causes),
                  mode=Mode.CONCORD)
