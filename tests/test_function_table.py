from __future__ import annotations

import ast
import contextlib
import io

from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
from src.transmogrifier.function_table import (
    FunctionResolutionState,
    FunctionTable,
)
from src.transmogrifier.graph.graph_deep_compiler import GraphDeepCompiler
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.transmogrifier.ssa import IRModule


def test_function_table_exposes_recursive_resolution_as_one_backedge():
    table = FunctionTable()
    reference = table.declare("recur", qualified_name="module.recur")

    assert table.begin_resolution(reference)
    assert not table.begin_resolution(reference)
    entry = table.entry(reference)
    assert entry.state == FunctionResolutionState.RESOLVING
    assert entry.recursive

    graph = object()
    table.resolve_graph(reference, graph)
    assert entry.state == FunctionResolutionState.RESOLVED
    assert entry.graph is graph


def test_ssa_module_owns_the_same_neutral_function_table_type():
    module = IRModule(functions={})
    assert isinstance(module.function_table, FunctionTable)


def test_deep_compiler_calls_an_installed_function_table_target():
    table = FunctionTable()
    reference = table.declare("twice")
    table.install_implementation(reference, "python", lambda value: value * 2)
    graph = ProcessGraph(materialize_memory=False, function_table=table)

    graph.G.add_node(
        1, type="Input", label="callee", parents=[], children=[]
    )
    graph.G.add_node(
        2, type="Input", label="x", parents=[], children=[]
    )
    graph.G.add_node(
        3,
        type="Call",
        label="twice(x)",
        parents=[(1, "func"), (2, "args")],
        children=[(4, "value")],
        attributes={"callee_ref": reference.address},
    )
    graph.G.add_node(
        4,
        type="Store",
        label="result",
        parents=[(3, "value")],
        children=[],
    )
    graph.G.add_edges_from(((1, 3), (2, 3), (3, 4)))
    graph.levels = {1: 0, 2: 0, 3: 1, 4: 2}

    compiled = GraphDeepCompiler(
        graph,
        {},
        {
            "Store": {
                "min_inputs": 1,
                "max_inputs": 1,
                "min_outputs": 1,
                "max_outputs": 1,
            }
        },
    ).build_function()
    assert compiled(floatcallee=None, floatx=7) == (14,)


def test_process_graph_functions_assemble_static_definitions_with_kwargs():
    module = ast.parse(
        """
def affine(x, scale, offset):
    return x * scale + offset

def render_value(x):
    return affine(offset=4, x=x, scale=3)
"""
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    affine_ref = graph.function_table.reference("affine")
    render_ref = graph.function_table.reference("render_value")
    assert affine_ref is not None
    assert render_ref is not None

    render_graph = graph.function_table.entry(render_ref).graph
    call = next(
        data
        for _node_id, data in render_graph.G.nodes(data=True)
        if data.get("type") == "Call"
    )
    assert call["attributes"]["callee_ref"] == affine_ref.address
    assert {role for _parent, role in call["parents"]} == {
        "kw:offset",
        "kw:x",
        "kw:scale",
    }
    assert not any(
        data.get("type") == "keyword"
        for _node_id, data in render_graph.G.nodes(data=True)
    )

    definitions = GraphDeepCompiler.assemble_function_table(
        graph.function_table,
        {},
        {},
    )
    assert set(definitions) == {affine_ref, render_ref}
    assert all(
        entry.graph is not None
        and graph.function_table.implementation(
            entry.reference,
            "python",
        )
        is not None
        for entry in graph.function_table
    )
    render = graph.function_table.implementation(render_ref, "python")
    assert render is not None
    assert render(5) == 19
