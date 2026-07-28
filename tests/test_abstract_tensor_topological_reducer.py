from __future__ import annotations

import ast
import contextlib
import io

from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def test_only_name_assign_and_call_receive_existing_process_graph_aliases():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(
                """
def kernel(x):
    y = helper(x)
"""
            )
        )

    original_types = {
        node_id: data["type"] for node_id, data in graph.G.nodes(data=True)
    }
    reduced = reduce_abstract_tensor_topology(graph)

    assert reduced is graph
    aliases = {"Name": "Load", "Assign": "Store", "Call": "Call"}
    for node_id, original_type in original_types.items():
        data = graph.G.nodes[node_id]
        if original_type in aliases:
            assert data["type"] == aliases[original_type]
            assert data["op"] == aliases[original_type]
            assert data["attributes"]["source_type"] == original_type
        else:
            assert data["type"] == original_type


def test_expr_wrapper_is_removed_without_removing_its_interior():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse("helper(3)\n"))

    expr_ids = [
        node_id
        for node_id, data in graph.G.nodes(data=True)
        if data["type"] == "Expr"
    ]
    interior_ids = [
        node_id
        for node_id, data in graph.G.nodes(data=True)
        if data["type"] == "Call"
    ]
    assert expr_ids
    assert interior_ids
    reduce_abstract_tensor_topology(graph)

    assert not any(node_id in graph.G for node_id in expr_ids)
    assert all(node_id in graph.G for node_id in interior_ids)


def test_ellipsis_index_is_lowered_as_one_parent_aware_index_operation():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def kernel(tensor, zigzag):
    return tensor[..., zigzag]
"""
    )
    subscript = next(
        node for node in ast.walk(module) if isinstance(node, ast.Subscript)
    )
    index_tuple = subscript.slice
    assert isinstance(index_tuple, ast.Tuple)

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    indexed = graph.G.nodes[id(subscript)]
    assert indexed["type"] == "Indexed"
    assert indexed["parents"] == [
        (id(subscript.value), "base"),
        (id(index_tuple.elts[0]), "index"),
        (id(index_tuple.elts[1]), "index"),
    ]
    assert graph.G.nodes[id(index_tuple.elts[0])]["expr_obj"].value is Ellipsis
    assert id(index_tuple) not in graph.G


def test_imports_are_retained_as_logged_contextual_requirements(caplog):
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def kernel(x):
    from package.math import helper as operation
    return operation(x)
"""
    )
    imported = next(
        node for node in ast.walk(module) if isinstance(node, ast.ImportFrom)
    )
    imported_name = imported.names[0]

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    with caplog.at_level(
        "INFO",
        logger="src.common.tensors.topological_reducer",
    ):
        reduce_abstract_tensor_topology(graph)

    requirement = {
        "kind": "import_from",
        "module": "package.math",
        "level": 0,
        "names": (("helper", "operation"),),
    }
    assert graph.G.graph["contextual_requirements"] == (requirement,)
    assert (
        graph.G.nodes[id(imported)]["attributes"][
            "contextual_requirement"
        ]
        == requirement
    )
    assert (
        graph.G.nodes[id(imported_name)]["attributes"][
            "contextual_requirement"
        ]
        == requirement
    )
    assert "retaining ProcessGraph import" in caplog.text


def test_calls_reference_separate_local_function_subgraphs():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def helper(x):
    return x + 1

def kernel(x):
    return helper(x)
"""
    )
    helper = module.body[0]
    call = next(
        node for node in ast.walk(module.body[1])
        if isinstance(node, ast.Call)
    )

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    reference = graph.function_table.reference("helper")
    assert reference is not None
    assert (
        graph.G.nodes[id(call)]["attributes"]["callee_ref"]
        == reference.address
    )
    entry = graph.function_table.entry(reference)
    assert entry.graph is not graph
    assert id(helper) not in entry.graph.G
    assert any(
        data.get("type") == "Input" and data.get("label") == "x"
        for _node_id, data in entry.graph.G.nodes(data=True)
    )
    assert any(
        data.get("type") == "Add"
        for _node_id, data in entry.graph.G.nodes(data=True)
    )
    assert entry.graph.function_table is graph.function_table


def test_imported_callee_is_an_external_function_reference():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def kernel(x):
    from package.math import helper as operation
    return operation(x)
"""
    )
    call = next(node for node in ast.walk(module) if isinstance(node, ast.Call))

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    reference = graph.function_table.reference("operation")
    assert reference is not None
    assert (
        graph.G.nodes[id(call)]["attributes"]["callee_ref"]
        == reference.address
    )
    assert graph.function_table.entry(reference).qualified_name == (
        "package.math.helper"
    )


def test_recursive_call_is_retained_as_a_function_table_backedge():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def recur(x):
    return recur(x)
"""
    )
    call = next(node for node in ast.walk(module) if isinstance(node, ast.Call))

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    reference = graph.function_table.reference("recur")
    assert reference is not None
    assert graph.function_table.entry(reference).recursive
    assert graph.G.nodes[id(call)]["attributes"]["recursive_backedge"] is True
