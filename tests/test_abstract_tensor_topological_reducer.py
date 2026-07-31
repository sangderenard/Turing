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


def test_nested_matrix_multiplication_is_numerical_topology_not_a_call():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def dct(blocks, transform):
    return transform @ blocks @ transform
"""
    )
    matrix_products = [
        node for node in ast.walk(module) if isinstance(node, ast.BinOp)
    ]
    assert len(matrix_products) == 2

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    for expression in matrix_products:
        data = graph.G.nodes[id(expression)]
        assert data["type"] == "matmul"
        assert data["op"] == "matmul"
        assert data["parents"] == [
            (id(expression.left), "lhs"),
            (id(expression.right), "rhs"),
        ]


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


def test_imported_calls_use_distinct_external_python_table():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(
                """
from operator import add

def combine(left, right):
    return add(left, right)
"""
            )
        )
    reduce_abstract_tensor_topology(graph)

    assert graph.function_table.reference("combine") is not None
    assert graph.function_table.reference("add") is None
    external_reference = graph.external_function_table.reference("add")
    assert external_reference is not None
    call = next(
        data
        for _node_id, data in graph.G.nodes(data=True)
        if (data.get("attributes") or {}).get("external_callee_ref")
        == external_reference.address
    )
    assert "callee_ref" not in call["attributes"]

    graph.external_function_table.resolve_imports()
    assert graph.external_function_table.invoke(
        external_reference,
        4,
        7,
    ) == 11


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

    assert graph.function_table.reference("operation") is None
    reference = graph.external_function_table.reference("operation")
    assert reference is not None
    assert (
        graph.G.nodes[id(call)]["attributes"]["external_callee_ref"]
        == reference.address
    )
    assert graph.external_function_table.entry(reference).qualified_name == (
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


def test_lexical_occurrences_are_unique_then_reduce_to_monotonic_fanout():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def kernel(x):
    first = x + 1
    second = first * first
    return second
"""
    )
    first_loads = [
        node
        for node in ast.walk(module)
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == "first"
        )
    ]
    assert len(first_loads) == 2
    assert id(first_loads[0]) != id(first_loads[1])

    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    assert all(id(node) in graph.G for node in first_loads)

    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("kernel").graph

    assert list(function_graph.G) == list(range(len(function_graph.G)))
    assert all(
        data["value_id"] == node_id
        for node_id, data in function_graph.G.nodes(data=True)
    )
    assert not any(
        data.get("type") in {"Load", "Store"}
        for _node_id, data in function_graph.G.nodes(data=True)
    )
    inputs = [
        node_id
        for node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Input"
    ]
    assert len(inputs) == 1
    add = next(
        node_id
        for node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Add"
    )
    multiply = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Mul"
    )
    assert multiply["parents"] == [(add, "lhs"), (add, "rhs")]


def test_static_python_reference_chain_collapses_without_becoming_input():
    class StaticTensorAPI:
        @staticmethod
        def stack(values, dim=0):
            raise AssertionError("static collapse must not execute the method")

    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"StaticTensorAPI": StaticTensorAPI}
    module = ast.parse(
        """
def kernel(values):
    return StaticTensorAPI.stack(values, dim=0)
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("kernel").graph

    assert not any(
        data.get("label") == "StaticTensorAPI"
        for _node_id, data in function_graph.G.nodes(data=True)
    )
    operation = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "stack"
    )
    assert operation["attributes"]["static_python_reference"] == (
        "StaticTensorAPI.stack"
    )
    assert [
        data["label"]
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Input"
    ] == ["values"]


def test_static_python_reference_alias_uses_integer_wrapper_node():
    class StaticTensorAPI:
        @staticmethod
        def stack(values, dim=0):
            raise AssertionError("static collapse must not execute the method")

    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"StaticTensorAPI": StaticTensorAPI}
    module = ast.parse(
        """
def kernel(values):
    api = StaticTensorAPI
    operation = api.stack
    return operation(values, dim=0)
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("kernel").graph

    assert all(isinstance(node_id, int) for node_id in function_graph.G)
    reference_id, reference = next(
        (node_id, data)
        for node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "StaticReference"
    )
    assert reference["attributes"]["static_python_reference"] == (
        "StaticTensorAPI.stack"
    )
    operation = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Call"
    )
    assert (reference_id, "callee") in operation["parents"]
    assert not any(
        data.get("label") in {"api", "operation"}
        for _node_id, data in function_graph.G.nodes(data=True)
    )


def test_lambda_receives_an_anonymous_function_subgraph():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def kernel(value):
    transform = lambda item: item + 1
    return transform(value)
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    lambda_entry = next(
        entry
        for entry in graph.function_table
        if entry.metadata.get("source_type") == "Lambda"
    )
    assert lambda_entry.name.startswith("<lambda:")
    assert lambda_entry.graph is not None
    assert lambda_entry.graph.G.graph["function_parameters"] == ("item",)
    call = next(
        data
        for _node_id, data in graph.function_table.entry("kernel").graph.G.nodes(
            data=True
        )
        if isinstance(data.get("expr_obj"), ast.Call)
    )
    assert call["attributes"]["callee_ref"] == lambda_entry.reference.address


def test_augmented_assignment_lowers_to_value_operation():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(
                """
def accumulate(value, increment):
    value += increment
    return value
"""
            )
        )
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("accumulate").graph

    assert not any(
        isinstance(data.get("expr_obj"), ast.AugAssign)
        and data.get("type") == "AugAssign"
        for _node_id, data in function_graph.G.nodes(data=True)
    )
    add = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Add"
    )
    assert len(add["parents"]) == 2


def test_named_expression_binds_and_forwards_its_value():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(
                """
def select(value):
    return (saved := value) + saved
"""
            )
        )
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("select").graph

    assert not any(
        isinstance(data.get("expr_obj"), ast.NamedExpr)
        for _node_id, data in function_graph.G.nodes(data=True)
    )
    add = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Add"
    )
    assert len(add["parents"]) == 2
    assert add["parents"][0][0] == add["parents"][1][0]


def test_unary_plus_forwards_the_existing_value():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse("def positive(value):\n    return +value\n"))
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("positive").graph

    assert not any(
        data.get("type") == "Cast"
        for _node_id, data in function_graph.G.nodes(data=True)
    )
    assert [
        data["label"]
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Input"
    ] == ["value"]


def test_referenced_builtin_is_not_mislabeled_as_runtime_input():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse("def freeze(values):\n    return tuple(values)\n")
        )
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("freeze").graph

    assert [
        data["label"]
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Input"
    ] == ["values"]
    call = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
    )
    assert call["attributes"]["static_python_reference"] == "tuple"


def test_branch_assignments_merge_before_later_use():
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(
                """
def choose(condition, left, right):
    if condition:
        selected = left
    else:
        selected = right
    return selected
"""
            )
        )
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("choose").graph

    phi = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Phi"
    )
    assert {role for _parent, role in phi["parents"]} == {
        "test",
        "body",
        "orelse",
    }
    assert not any(
        data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_name") == "selected"
        for _node_id, data in function_graph.G.nodes(data=True)
    )


def test_referenced_module_literal_becomes_constant_not_input():
    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"TABLE": ((1, 2), (3, 4))}
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse("def lookup():\n    return TABLE\n")
        )
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("lookup").graph

    assert not any(
        data.get("type") == "Input"
        for _node_id, data in function_graph.G.nodes(data=True)
    )
    constant = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Constant"
    )
    assert constant["constant"] == ((1, 2), (3, 4))


def test_referenced_module_literal_remains_a_call_argument():
    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"TABLE": (1, 2, 3)}
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(
                """
def consume(value):
    return value

def entry():
    return consume(TABLE)
"""
            )
        )
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("entry").graph
    call = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
    )
    assert any(
        function_graph.G.nodes[parent].get("constant") == (1, 2, 3)
        for parent, role in call["parents"]
        if role == "arg:0"
    )


def test_reducer_preserves_every_python_constant_literal():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def constants():
    return (8, 0.5, None, ..., b"jpeg")
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("constants").graph

    literals = [
        data["constant"]
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "Constant"
    ]
    assert literals == [8, 0.5, None, Ellipsis, b"jpeg"]


def test_floor_division_remains_distinct_from_true_division():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def divide(x):
    return (x / 2, x // 2)
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("divide").graph

    operations = [
        data["type"]
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") in {"Div", "FloorDiv"}
    ]
    assert operations == ["Div", "FloorDiv"]


def test_attribute_assignment_lowers_to_setattr_with_object_and_value():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
class Accumulator:
    def append(self, value):
        self._pending = value
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("Accumulator.append").graph

    set_attr = next(
        data
        for _node_id, data in function_graph.G.nodes(data=True)
        if data.get("type") == "SetAttr"
    )
    assert set_attr["attributes"]["attribute"] == "_pending"
    assert {role for _parent, role in set_attr["parents"]} == {
        "object",
        "value",
    }
    assert not any(
        data.get("type") == "Attribute"
        and isinstance(data.get("expr_obj"), ast.Attribute)
        and isinstance(data["expr_obj"].ctx, ast.Store)
        for _node_id, data in function_graph.G.nodes(data=True)
    )


def test_try_value_survives_when_every_exception_handler_terminates():
    graph = ProcessGraph(materialize_memory=False)
    module = ast.parse(
        """
def convert(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError("invalid") from error
    return numeric
"""
    )
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    function_graph = graph.function_table.entry("convert").graph

    assert not any(
        data.get("type") == "Input"
        and (data.get("attributes") or {}).get("binding_name") == "numeric"
        for _node_id, data in function_graph.G.nodes(data=True)
    )
