"""Captured mappings keep the row contract of their exact parent binding."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from src.common.tensors.topological_reducer import reduce_abstract_tensor_topology
from src.compiler.concordance_declarations import (
    ANNOTATION_DECLARATION, NAME_BINDING, SEQUENCE_ROW_DTYPE,
    SOURCE_CLOSURE_CAPTURE, SOURCE_CLOSURE_ENVIRONMENT,
)
from src.compiler.identity_concordance import (
    CorrelationTable, begin_identity_book, end_identity_book,
)
from src.compiler.glsl_deployment_strategy import _planned_scalar_dtypes_from
from src.transmogrifier.graph.graph_express2 import ProcessGraph


@pytest.fixture
def book():
    book, token = begin_identity_book()
    try:
        yield book
    finally:
        end_identity_book(token)


def _captured_mapping(source, function, book, binding="ids"):
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(graph)
    child = graph.function_table.entry(function).graph
    capture = next(
        book.page(SOURCE_CLOSURE_CAPTURE).latest(row)
        for row in book.page(SOURCE_CLOSURE_CAPTURE).rows()
        if row[0] == child.G.graph["lexical_read_scope"] and row[1] == binding
    )
    environment, parent_binding, input_binding = capture
    assert book.page(SOURCE_CLOSURE_ENVIRONMENT).cell(
        environment.row, environment.column,
    )[0] == parent_binding
    parent = next(
        entry.graph for entry in graph.function_table
        if entry.graph.G.graph.get("lexical_read_scope") == parent_binding.row[0]
    )
    parent_id = book.page(NAME_BINDING).cell(
        parent_binding.row, parent_binding.column,
    ).value_id
    child_id = book.page(NAME_BINDING).cell(
        input_binding.row, input_binding.column,
    ).value_id
    assert not CorrelationTable._source_closure_findings(
        SimpleNamespace(metadata={"identity_book": book}),
    )
    return parent.G.nodes[parent_id]["attributes"], child.G.nodes[child_id]["attributes"]


def test_doc_id_capture_keeps_declared_string_key_and_integer_value(book):
    # Exact body and declaration from demo_pluck_gl.py, without runtime samples.
    source = '''def main():
    _doc_misc_ids: dict[str, int] = {}
    def _doc_id(key: str) -> int:
        if key not in _doc_misc_ids:
            _doc_misc_ids[key] = 100000 + len(_doc_misc_ids)
        return _doc_misc_ids[key]
'''
    parent, child = _captured_mapping(source, "_doc_id", book, "_doc_misc_ids")
    assert parent["mapping_key_dtype"] == "int64"
    assert parent["mapping_value_dtype"] == "int64"
    assert child["binding_kind"] == "closure"
    assert child["mapping_key_dtype"] == parent["mapping_key_dtype"]
    assert child["mapping_value_dtype"] == parent["mapping_value_dtype"]
    assert child["mapping_key_class_ref"] == "builtins.str"
    assert child["mapping_value_class_ref"] == "builtins.int"
    cell = child["mapping_value_annotation_cell"]
    assert cell.page == SEQUENCE_ROW_DTYPE
    assert book.page(SEQUENCE_ROW_DTYPE).cell(cell.row, cell.column)[0] == (
        "int64", "int64",
    )
    assert {SOURCE_CLOSURE_CAPTURE, ANNOTATION_DECLARATION, NAME_BINDING} <= {
        source.page for source, _stage in book.edges_into(cell)
    }


def test_unlinked_transitive_closure_does_not_invent_a_capture(book):
    source = '''def outer():
    ids: dict[int, float] = {}
    def middle():
        count = len(ids)
        def inner(key: int) -> float:
            return ids[key]
'''
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(graph)
    inner = graph.function_table.entry("inner").graph
    child = next(
        data["attributes"] for _node, data in inner.G.nodes(data=True)
        if (data.get("attributes") or {}).get("binding_name") == "ids"
    )
    # This pre-existing path has no capture edge to transport. A same-spelled
    # outer mapping is not authority to manufacture that missing identity.
    assert not any(
        row[0] == inner.G.graph["lexical_read_scope"] and row[1] == "ids"
        for row in book.page(SOURCE_CLOSURE_CAPTURE).rows()
    )
    assert "mapping_key_dtype" not in child
    assert "mapping_value_dtype" not in child


def test_inner_parameter_does_not_borrow_outer_mapping_contract(book):
    source = '''def outer():
    ids: dict[str, int] = {}
    def middle(ids):
        def inner(key: str):
            return ids[key]
'''
    parent, child = _captured_mapping(source, "inner", book)
    assert "mapping_key_dtype" not in parent
    assert "mapping_value_dtype" not in parent
    assert "mapping_key_dtype" not in child
    assert "mapping_value_dtype" not in child


def test_inner_untyped_local_does_not_borrow_outer_mapping_contract(book):
    source = '''def outer():
    ids: dict[str, int] = {}
    def middle():
        ids = {}
        def inner(key: int) -> float:
            return ids[key]
'''
    parent, child = _captured_mapping(source, "inner", book)
    assert "mapping_key_dtype" not in parent
    assert "mapping_value_dtype" not in parent
    assert "mapping_key_dtype" not in child
    assert "mapping_value_dtype" not in child


def test_untyped_empty_mapping_does_not_acquire_column_types(book):
    source = '''def outer():
    ids = {}
    def inner(key):
        return ids[key]
'''
    parent, child = _captured_mapping(source, "inner", book)
    assert "mapping_key_dtype" not in parent
    assert "mapping_value_dtype" not in parent
    assert "mapping_key_dtype" not in child
    assert "mapping_value_dtype" not in child


def test_alias_capture_reads_the_same_declared_storage(book):
    source = '''def outer():
    storage: dict[int, float] = {}
    ids = storage
    def inner(key: int) -> float:
        return ids[key]
'''
    parent, child = _captured_mapping(source, "inner", book)
    assert parent["mapping_value_dtype"] == "float64"
    assert child["mapping_key_dtype"] == "int64"
    assert child["mapping_value_dtype"] == "float64"


def _planner_capture(annotation, book, function="inner"):
    source = f'''def outer():
    ids: {annotation} = {{}}
    def {function}(key):
        return ids[key]
'''
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(graph)
    child = graph.function_table.entry(function).graph
    value_id = next(
        int(node) for node, data in child.G.nodes(data=True)
        if (data.get("attributes") or {}).get("binding_name") == "ids"
    )
    return child, value_id


@pytest.mark.parametrize("annotation, expected", [
    ("dict[str, int]", "int64"),
    ("dict[int, float]", "float64"),
    ("dict[float, int]", "int64"),
])
def test_planner_reads_declared_closure_value_rows(annotation, expected, book):
    child, value_id = _planner_capture(annotation, book)
    assert _planned_scalar_dtypes_from(child).get(value_id) == expected


def test_planner_does_not_type_closure_without_capture_proof(book):
    child, value_id = _planner_capture("dict[str, int]", book)
    child.G.nodes[value_id]["attributes"].pop("mapping_value_annotation_cell")
    assert value_id not in _planned_scalar_dtypes_from(child)


def test_planner_rejects_class_that_disagrees_with_capture_declaration(book):
    child, value_id = _planner_capture("dict[str, int]", book)
    child.G.nodes[value_id]["attributes"]["mapping_value_class_ref"] = "builtins.float"
    assert value_id not in _planned_scalar_dtypes_from(child)


def test_planner_rejects_proof_for_another_capture(book):
    child, value_id = _planner_capture("dict[str, int]", book)
    other, other_id = _planner_capture("dict[str, int]", book, "other")
    child.G.nodes[value_id]["attributes"]["mapping_value_annotation_cell"] = (
        other.G.nodes[other_id]["attributes"]["mapping_value_annotation_cell"]
    )
    assert value_id not in _planned_scalar_dtypes_from(child)


def test_planner_does_not_borrow_outer_contract_for_shadowed_parameter(book):
    source = '''def outer():
    ids: dict[str, int] = {}
    def middle(ids):
        def inner(key):
            return ids[key]
'''
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(graph)
    child = graph.function_table.entry("inner").graph
    value_id = next(
        int(node) for node, data in child.G.nodes(data=True)
        if (data.get("attributes") or {}).get("binding_name") == "ids"
    )
    assert value_id not in _planned_scalar_dtypes_from(child)
