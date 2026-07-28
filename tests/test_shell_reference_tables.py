import ast
import contextlib
import io

from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
from src.compiler.glsl_deployment_strategy import (
    strategize_glsl_deployment,
)
from src.compiler.shell_reference_tables import (
    build_shell_reference_tables,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _reduced_function_graph():
    module = ast.parse(
        """
def append_frame(state, frame):
    return frame

def record(writer, frame):
    marker = b"movi"
    writer.append_frame(frame)
    return marker
"""
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    return graph, graph.function_table.entry("record").graph


def test_shell_reference_tables_use_dense_monotonic_local_ids():
    graph, record_graph = _reduced_function_graph()
    tables = build_shell_reference_tables(record_graph)

    assert [entry.index for entry in tables.functions] == list(
        range(len(tables.functions))
    )
    assert [entry.index for entry in tables.constants] == list(
        range(len(tables.constants))
    )
    assert [entry.index for entry in tables.memory] == list(
        range(len(tables.memory))
    )

    append_reference = graph.function_table.reference("append_frame")
    assert append_reference is not None
    append_slot = next(
        entry
        for entry in tables.functions
        if entry.namespace == "graph"
        and entry.source_address == append_reference.address
    )
    append_nodes = {
        node_id
        for node_id, data in record_graph.G.nodes(data=True)
        if data.get("type") == "append_frame"
    }
    assert append_nodes
    assert any(
        correlation.table == "functions"
        and correlation.index == append_slot.index
        and correlation.graph_node_id in append_nodes
        for correlation in tables.correlations
    )

    marker = next(
        entry for entry in tables.constants if entry.value == b"movi"
    )
    assert any(
        correlation.table == "constants"
        and correlation.index == marker.index
        for correlation in tables.correlations
    )

    writer = next(
        entry for entry in tables.memory if entry.name == "writer"
    )
    assert "input" in writer.roles
    assert any(
        correlation.table == "memory"
        and correlation.index == writer.index
        and correlation.graph_node_id == writer.graph_node_id
        for correlation in tables.correlations
    )


def test_glsl_shell_packages_independent_reference_lists():
    module = ast.parse(
        """
def affine(value):
    return value * 3 + 4
"""
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)
    shell_type = strategize_glsl_deployment(graph)
    first = shell_type()
    second = shell_type()
    try:
        assert first.function_references == second.function_references
        assert first.constant_references == second.constant_references
        assert first.memory_references == second.memory_references
        assert first.reference_correlations == second.reference_correlations

        assert first.function_references is not second.function_references
        assert first.constant_references is not second.constant_references
        assert first.memory_references is not second.memory_references
        assert first.reference_correlations is not second.reference_correlations
    finally:
        first.release()
        second.release()
