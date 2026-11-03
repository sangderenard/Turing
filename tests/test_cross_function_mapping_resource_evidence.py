"""The keyed IdentityBook writer must reach a guarded resource call by identity."""

import ast
from collections import deque
from types import SimpleNamespace

import pytest

from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
from src.compiler.concordance_declarations import (
    CALLABLE_IDENTITY,
    CLASS_FIELD_DECLARATION,
    INGESTION_VALUE,
    NATIVE_EXCEPTION_CONTROL_GAP,
    REDUCER_FIELD_STATE,
    SOURCE_ACTIVITY_BINDING,
    SOURCE_ACTIVITY_EDGE,
    SOURCE_ACTIVITY_NODE,
    SOURCE_AGGREGATE_LAYOUT,
    SOURCE_CALL_ACTUAL_FORMAL,
    SOURCE_CALL_FIELD_EFFECT_COMPLETION,
    SOURCE_CALL_OBJECT_ALIAS,
    SOURCE_FUNCTION_REACHABILITY,
    SOURCE_FUNCTION_EXIT_FIELD_STATE,
    SOURCE_RESOURCE_ADMISSION_DEMAND,
    SOURCE_VALUE_CLASS,
    FieldStateKind,
)
from src.compiler.exception_graph import MissingResourceControlTopology
from src.compiler.source_evidence_resolver import CallableEvidence, require_callable
from src.compiler.identity_concordance import CorrelationTable, begin_identity_book, end_identity_book
from src.transmogrifier.graph.graph_express2 import ProcessGraph


SOURCE = '''
class IdentityBook:
    def borrowed(self, name):
        return name

class Module:
    def __init__(self):
        self.metadata = {}

def publish(module: Module):
    module.metadata["identity_book"] = IdentityBook()

def consume(module: Module, name: str):
    book = module.metadata.get("identity_book")
    if book is None:
        return 0
    with book.borrowed(name) as page:
        return 1

def root(name: str):
    module = Module()
    publish(module)
    return consume(module, name)
'''


def _source_node(wrapper, predicate):
    matches = [
        int(node_id)
        for node_id, data in wrapper.G.nodes(data=True)
        if predicate(data.get("expr_obj"))
    ]
    assert len(matches) == 1, matches
    return matches[0]


def _proof_closure(book, cell):
    pending, seen = [cell], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(parent for parent, _stage in book.edges_into(current))
    return seen


def test_root_module_actual_binds_both_cross_function_formals():
    # The two root calls retain the same Module actual. This receipt proves
    # each call-frame binding without claiming the mutable metadata field
    # has been transferred or that borrowed() has a guarded callable.
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(SOURCE))
        with pytest.raises(MissingResourceControlTopology):
            reduce_abstract_tensor_topology(graph)
        root = graph.function_table.entry("root").graph
        published = graph.function_table.entry("publish").graph
        consumed = graph.function_table.entry("consume").graph
        root_scope = root.G.graph["authored_function_identity"]
        receipts = []
        for name, callee in (("publish", published), ("consume", consumed)):
            call = _source_node(root, lambda node: isinstance(node, ast.Call)
                                and isinstance(node.func, ast.Name)
                                and node.func.id == name)
            formal = next(
                int(node_id) for node_id, data in callee.G.nodes(data=True)
                if data.get("type") == "Input"
                and (data.get("attributes") or {}).get("binding_name") == "module"
            )
            callee_scope = callee.G.graph["authored_function_identity"]
            receipt = book.latest_ref(SOURCE_CALL_ACTUAL_FORMAL, (
                root_scope, call, callee_scope, formal,
            ))
            assert receipt is not None
            actual, role, parameter, _operands, _formal_binding = book.page(
                SOURCE_CALL_ACTUAL_FORMAL
            ).cell(receipt.row, receipt.column)
            assert role == "arg:0" and parameter == "module"
            alias = book.latest_ref(SOURCE_CALL_OBJECT_ALIAS, (
                root_scope, call, callee_scope, formal,
            ))
            assert alias is not None
            alias_actual, alias_binding, _contract, _class, _declaration = (
                book.page(SOURCE_CALL_OBJECT_ALIAS).cell(alias.row, alias.column)
            )
            assert alias_actual == actual
            assert alias_binding == _formal_binding
            receipts.append((receipt, actual))
        assert receipts[0][1] == receipts[1][1]
        publish_completion = book.latest_ref(
            SOURCE_CALL_FIELD_EFFECT_COMPLETION,
            (root_scope, _source_node(root, lambda node: isinstance(node, ast.Call)
                                      and isinstance(node.func, ast.Name)
                                      and node.func.id == "publish"),
             published.G.graph["authored_function_identity"],
             next(int(node_id) for node_id, data in published.G.nodes(data=True)
                  if data.get("type") == "Input"
                  and (data.get("attributes") or {}).get("binding_name") == "module"),
             "metadata"),
        )
        assert publish_completion is not None
        completed_actual, _formal, exit_state, callee_exit, _binding, call_exit = (
            book.page(SOURCE_CALL_FIELD_EFFECT_COMPLETION).cell(
                publish_completion.row, publish_completion.column,
            )
        )
        assert completed_actual == receipts[0][1]
        assert exit_state.page == SOURCE_FUNCTION_EXIT_FIELD_STATE
        assert callee_exit.page == SOURCE_ACTIVITY_NODE
        assert call_exit.page == SOURCE_ACTIVITY_NODE
        assert not CorrelationTable._source_call_field_effect_completion_findings(
            SimpleNamespace(metadata={"identity_book": book}),
        )
        assert not CorrelationTable._source_call_actual_formal_findings(
            SimpleNamespace(metadata={"identity_book": book}),
        )
        assert not CorrelationTable._source_call_object_alias_findings(
            SimpleNamespace(metadata={"identity_book": book}),
        )
        source = next(row[0] for row in book.page(NATIVE_EXCEPTION_CONTROL_GAP).rows())
        requirements = book.page(NATIVE_EXCEPTION_CONTROL_GAP).latest((source,))[1]
        assert "missing_callable_identity" not in requirements
        assert requirements == ("resource_enter", "resource_exit", "cleanup")
    finally:
        end_identity_book(token)


def test_keyed_write_reaches_guarded_borrowed_call_before_resource_refusal():
    # Program: root passes one Module to publish and consume. publish stores
    # an IdentityBook in metadata["identity_book"]; consume reads that key,
    # excludes None, and calls borrowed(name). Wrong value today: consume's
    # get result has no reaching class proof, so borrowed has no callable cell.
    # Rule: the guarded callable proof must cite the exact keyed writer,
    # object/call edges, field state, and control owner before resource cleanup
    # can be considered for admission.
    book, token = begin_identity_book()
    try:
        graph = ProcessGraph(materialize_memory=False)
        graph.build_from_ast(ast.parse(SOURCE))
        with pytest.raises(MissingResourceControlTopology):
            reduce_abstract_tensor_topology(graph)

        root = graph.function_table.entry("root").graph
        publish = graph.function_table.entry("publish").graph
        consume = graph.function_table.entry("consume").graph
        root_scope = root.G.graph["authored_function_identity"]
        publish_scope = publish.G.graph["authored_function_identity"]
        consume_scope = consume.G.graph["authored_function_identity"]

        publish_call = _source_node(
            root, lambda node: isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "publish",
        )
        consume_call = _source_node(
            root, lambda node: isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "consume",
        )
        # Both exact calls carry the same constructed Module value as arg:0.
        publish_actual = dict((role, parent) for parent, role in root.G.nodes[
            publish_call
        ]["parents"])["arg:0"]
        consume_actual = dict((role, parent) for parent, role in root.G.nodes[
            consume_call
        ]["parents"])["arg:0"]
        assert publish_actual == consume_actual
        publish_reach = book.latest_ref(SOURCE_FUNCTION_REACHABILITY, (
            root_scope, publish_call, publish_scope,
        ))
        consume_reach = book.latest_ref(SOURCE_FUNCTION_REACHABILITY, (
            root_scope, consume_call, consume_scope,
        ))
        assert publish_reach is not None and consume_reach is not None

        store = _source_node(
            publish, lambda node: isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "identity_book",
        )
        store_parents = dict((role, parent) for parent, role in publish.G.nodes[
            store
        ]["parents"])
        written_class = book.latest_ref(SOURCE_VALUE_CLASS, (
            publish_scope, store_parents["value"],
        ))
        assert written_class is not None
        assert book.page(SOURCE_VALUE_CLASS).cell(
            written_class.row, written_class.column,
        )[0] == "IdentityBook"
        layout = book.latest_ref(SOURCE_AGGREGATE_LAYOUT, (
            publish.G.graph["lexical_read_scope"], store,
        ))
        assert layout is not None
        members = book.page(SOURCE_AGGREGATE_LAYOUT).cell(
            layout.row, layout.column,
        ).members
        assert members[-1].value_id == store_parents["value"]

        field = book.latest_ref(CLASS_FIELD_DECLARATION, (
            publish_scope[0], "Module", "metadata",
        ))
        assert field is not None
        field_page = book.page(REDUCER_FIELD_STATE)
        published_states = tuple(
            (book.latest_ref(REDUCER_FIELD_STATE, row), field_page.latest(row))
            for row in field_page.rows()
            if row[0] == publish.G.graph["lexical_read_scope"]
            and row[-1] == "metadata"
        )
        consumed_states = tuple(
            (book.latest_ref(REDUCER_FIELD_STATE, row), field_page.latest(row))
            for row in field_page.rows()
            if row[0] == consume.G.graph["lexical_read_scope"]
            and row[-1] == "metadata"
        )
        written_state = next(
            (cell for cell, state in published_states
             if state.kind is FieldStateKind.ELEMENT_WRITTEN), None,
        )
        observed_state = next(
            (cell for cell, state in consumed_states
             if state.kind is FieldStateKind.OBSERVED), None,
        )
        assert written_state is not None and observed_state is not None

        guarded_if = _source_node(consume, lambda node: isinstance(node, ast.If))
        resource = _source_node(consume, lambda node: isinstance(node, ast.With))
        borrowed = _source_node(
            consume, lambda node: isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "borrowed",
        )
        demand = book.latest_ref(SOURCE_RESOURCE_ADMISSION_DEMAND, (
            consume_scope, id(consume.G.nodes[resource]["expr_obj"]),
        ))
        assert demand is not None
        source, activity, calls = book.page(SOURCE_RESOURCE_ADMISSION_DEMAND).cell(
            demand.row, demand.column,
        )
        assert source.page == INGESTION_VALUE
        assert source.row[1] == id(consume.G.nodes[resource]["expr_obj"])
        if_source = book.latest_ref(INGESTION_VALUE, (
            source.row[0], id(consume.G.nodes[guarded_if]["expr_obj"]),
        ))
        if_binding = book.latest_ref(SOURCE_ACTIVITY_BINDING, (
            consume_scope, if_source,
        ))
        assert if_binding is not None
        assert activity.page == SOURCE_ACTIVITY_BINDING
        assert calls[0][0].page == INGESTION_VALUE
        assert calls[0][0].row[1] == id(consume.G.nodes[borrowed]["expr_obj"])

        def root_call_activity(node_id):
            call_source = book.latest_ref(INGESTION_VALUE, (
                source.row[0], id(root.G.nodes[node_id]["expr_obj"]),
            ))
            binding = book.latest_ref(SOURCE_ACTIVITY_BINDING, (
                root_scope, call_source,
            ))
            assert binding is not None
            return book.page(SOURCE_ACTIVITY_BINDING).cell(
                binding.row, binding.column,
            )

        publish_activity = root_call_activity(publish_call)
        consume_activity = root_call_activity(consume_call)
        activity_edges = {}
        for row in book.page(SOURCE_ACTIVITY_EDGE).rows():
            if row[0] == root_scope:
                activity_edges.setdefault(row[1], []).append((
                    row[2], book.latest_ref(SOURCE_ACTIVITY_EDGE, row),
                ))
        frontier = deque(((publish_activity[1], ()),))
        seen = set()
        ordered_path = None
        while frontier:
            current, path = frontier.popleft()
            if current == consume_activity[0]:
                ordered_path = path
                break
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(
                (next_cell, (*path, edge_cell))
                for next_cell, edge_cell in activity_edges.get(current, ())
            )
        assert ordered_path
        gap = book.latest_ref(NATIVE_EXCEPTION_CONTROL_GAP, (source,))
        assert gap is not None
        kind, requirements = book.page(NATIVE_EXCEPTION_CONTROL_GAP).cell(
            gap.row, gap.column,
        )
        assert kind == "With" and "cleanup" in requirements

        # An optional get never has a global exact IdentityBook class. Its
        # present-arm call must instead carry the writer and guard's proof.
        callable_cell = book.latest_ref(CALLABLE_IDENTITY, (
            consume_scope, borrowed,
        ))
        assert callable_cell is not None
        proof = _proof_closure(book, callable_cell)
        assert written_class in proof
        assert layout in proof
        assert field in proof
        assert written_state in proof and observed_state in proof
        assert publish_reach in proof and consume_reach in proof
        assert if_binding in proof
        assert set(ordered_path) <= proof
        assert "missing_callable_identity" not in requirements
        resolved = require_callable(consume, consume_scope, borrowed, book=book)
        assert isinstance(resolved, CallableEvidence)
        assert resolved.control_scope == activity
        assert resolved.ordered_inputs == calls[0][2]
    finally:
        end_identity_book(token)
