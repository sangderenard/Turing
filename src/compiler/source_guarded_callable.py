"""Resolve one guarded mapping payload to its exact authored method address."""

import ast

from .concordance_declarations import (
    CALLABLE_IDENTITY, CLASS_DECLARATION, CLASS_METHOD_DECLARATION,
    FUNCTION_ADDRESS, PROCESS_GRAPH_OPERANDS, REDUCTION,
    SOURCE_GUARDED_MAPPING_PAYLOAD, SOURCE_MAPPING_GET_READ,
    SOURCE_SPAN,
)
from .identity_concordance import Derived, Mode
from .source_evidence_resolver import ClassEvidence, require_class
from ..transmogrifier.function_table import function_address_identity


def _sources(book, cell):
    return {source for source, _stage in book.edges_into(cell)}


def guarded_callable_facts(source_graph, book, row, guarded):
    """Return exact causes, or None when any source identity is ambiguous."""
    from ..common.tensors.topological_reducer import node_identity_cell

    scope, get_id, call_id = row
    graphs = tuple(
        entry.graph for entry in source_graph.function_table
        if getattr(entry, 'graph', None) is not None
        and tuple(entry.graph.G.graph.get('authored_function_identity') or ()) == scope
    )
    if len(graphs) != 1:
        return None
    graph = graphs[0]
    if call_id not in graph.G:
        return None
    expression = graph.G.nodes[call_id].get('expr_obj')
    if not (isinstance(expression, ast.Call)
            and isinstance(expression.func, ast.Attribute)):
        return None
    accessor_ids = tuple(
        int(node) for node, data in graph.G.nodes(data=True)
        if data.get('expr_obj') is expression.func
    )
    if len(accessor_ids) != 1:
        return None
    call_cell = node_identity_cell(graph, call_id, book=book)
    accessor_cell = node_identity_cell(graph, accessor_ids[0], book=book)
    guard_fact = book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).cell(
        guarded.row, guarded.column,
    )
    if (guard_fact[-2] != call_cell
            or guard_fact[-1].page != PROCESS_GRAPH_OPERANDS):
        return None
    borrowed_operands = guard_fact[-1]
    operand_fact = book.page(PROCESS_GRAPH_OPERANDS).cell(
        borrowed_operands.row, borrowed_operands.column,
    )
    read = guard_fact[0]
    if read.page != SOURCE_MAPPING_GET_READ or read.row[:2] != (scope, get_id):
        return None
    read_fact = book.page(SOURCE_MAPPING_GET_READ).cell(read.row, read.column)
    # The guarded call must consume this exact optional get result.
    if ((read_fact[7], 'operand') not in operand_fact
            or borrowed_operands.row[-1] != call_id):
        return None
    stored = guard_fact[1]
    class_evidence = require_class(
        scope, get_id, control_scope=guarded, book=book,
    )
    if (not isinstance(class_evidence, ClassEvidence)
            or class_evidence.subject_cell != read_fact[7]
            or class_evidence.control_scope != guarded
            or class_evidence.class_cell is None):
        return None
    class_cell = class_evidence.class_cell
    class_name = class_evidence.class_identity
    store_scope = class_cell.row[0]
    class_cell_decl = book.latest_ref(
        CLASS_DECLARATION, (store_scope[0], class_name),
    )
    method_cell = book.latest_ref(
        CLASS_METHOD_DECLARATION,
        (store_scope[0], class_name, expression.func.attr),
    )
    if class_cell_decl is None or method_cell is None:
        return None
    method_fact = book.page(CLASS_METHOD_DECLARATION).cell(
        method_cell.row, method_cell.column,
    )
    candidates = tuple(
        entry for entry in source_graph.function_table
        if entry.qualified_name == str(method_fact.graph_identity)
        and (entry.metadata or {}).get('method_owner') == class_name
    )
    if len(candidates) != 1:
        return None
    entry = candidates[0]
    source_node = (entry.metadata or {}).get('source_node')
    if source_node not in source_graph.G:
        return None
    definition = source_graph.G.nodes[source_node].get('expr_obj')
    if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    owners = tuple(
        (owner, index)
        for _node, data in source_graph.G.nodes(data=True)
        if isinstance((owner := data.get('expr_obj')), ast.ClassDef)
        and owner.name == class_name
        for index, member in enumerate(owner.body)
        if member is definition
    )
    if len(owners) != 1 or definition.name != expression.func.attr:
        return None
    _owner, member_index = owners[0]
    method_spans = tuple(
        cell for cell in _sources(book, method_cell)
        if cell.page == SOURCE_SPAN
        and cell.row == (store_scope[0], class_name, (('body', member_index),))
    )
    if len(method_spans) != 1:
        return None
    method_span = method_spans[0]
    address_cell = book.latest_ref(
        FUNCTION_ADDRESS, (function_address_identity(entry),),
    )
    if (address_cell is None or book.page(FUNCTION_ADDRESS).cell(
            address_cell.row, address_cell.column) != int(entry.reference.address)):
        return None
    return (
        int(entry.reference.address), call_cell, accessor_cell,
        guarded, read, stored, class_cell, class_cell_decl,
        method_cell, method_span, address_cell, borrowed_operands,
    )


def post_guarded_callables(source_graph, book):
    for row in tuple(book.page(SOURCE_GUARDED_MAPPING_PAYLOAD).rows()):
        guarded = book.latest_ref(SOURCE_GUARDED_MAPPING_PAYLOAD, row)
        facts = guarded_callable_facts(source_graph, book, row, guarded)
        if facts is None:
            continue
        (address, call, accessor, guard, read, stored, class_cell,
         class_decl, method, method_span, address_cell, operands) = facts
        # Both the class method declaration and the incumbent address now
        # cite the same authored FunctionDef span cell, established by the
        # exact class-body AST member at FunctionTable declaration.
        book.post(
            FUNCTION_ADDRESS, address_cell.row, address,
            stage=REDUCTION, provenance=Derived((method_span,)),
            mode=Mode.CONCORD,
        )
        book.post(
            CALLABLE_IDENTITY, (row[0], int(row[2])), address,
            stage=REDUCTION,
            provenance=Derived((address_cell, call, accessor, guard, read,
                                stored, class_cell, class_decl, method,
                                method_span, operands)),
            mode=Mode.CONCORD,
        )
