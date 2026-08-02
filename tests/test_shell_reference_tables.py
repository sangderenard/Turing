import ast
import contextlib
import io

import pytest

from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
from src.compiler.glsl_deployment_strategy import (
    strategize_glsl_deployment,
)
from src.compiler.shell_reference_tables import (
    build_class_navigation_table,
    build_map_dependency_regions,
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


def test_class_map_and_runtime_dependency_regions_retain_different_method_compartments():
    module = ast.parse(
        '''
class Archive:
    capacity: int = 16

    def __init__(self):
        self.value = 0

    def read(self, index):
        return self.value + index

    def compact(self):
        self.value = 0
'''
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    map_ir = graph.G.graph["map_ir"]
    (object_map,) = map_ir["objects"]
    assert object_map["class_name"] == "Archive"
    assert [item["identity"] for item in object_map["attributes"]] == [
        "Archive.capacity", "Archive.value",
    ]
    assert [item["identity"] for item in map_ir["graphs"]] == [
        "Archive.__init__", "Archive.read", "Archive.compact",
    ]

    regions = build_map_dependency_regions(graph, "Archive.read")
    references = {
        entry.qualified_name: int(entry.reference.address)
        for entry in graph.function_table
    }
    assert regions.runtime == (references["Archive.read"],)
    assert set(regions.mapped) == {
        references["Archive.__init__"],
        references["Archive.read"],
        references["Archive.compact"],
    }
    assert set(regions.retained) == set(regions.mapped)
    assert set(regions.map_only) == {
        references["Archive.__init__"],
        references["Archive.compact"],
    }
    assert regions.bindings == (
        ("Archive.__init__", references["Archive.__init__"]),
        ("Archive.read", references["Archive.read"]),
        ("Archive.compact", references["Archive.compact"]),
    )

    # Each method is already an independent graph compartment suitable for
    # receiving its own method shell; retaining it does not add it to runtime.
    method_graphs = {
        name: graph.function_table.entry(name).graph
        for name in (
            "Archive.__init__", "Archive.read", "Archive.compact",
        )
    }
    assert len({id(item) for item in method_graphs.values()}) == 3
    assert all(
        item.G.graph["method_owner"] == "Archive"
        for item in method_graphs.values()
    )


def test_class_navigation_lut_resolves_instantiation_and_dot_through_permissions():
    module = ast.parse(
        '''
class Vault:
    capacity: int = 8

    def __init__(self):
        self.value = 0

    def read(self):
        return self.value

    def erase(self):
        self.value = 0
'''
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    object_map = graph.G.graph["map_ir"]["objects"][0]
    object_map["permissions"] = ("vault:enter",)
    object_map["attributes"][0]["permissions"] = ("vault:inspect",)
    method_permissions = {
        "__init__": ("vault:create",),
        "read": ("vault:read",),
        "erase": ("vault:erase",),
    }
    for method in object_map["methods"]:
        method["permissions"] = method_permissions[method["name"]]
    reduce_abstract_tensor_topology(graph)

    table = build_class_navigation_table(graph)
    references = {
        entry.qualified_name: int(entry.reference.address)
        for entry in graph.function_table
    }

    def evaluator(grants):
        granted = frozenset(grants)
        return lambda _identity, required: set(required) <= granted

    constructors = table.instantiate(
        "Vault", evaluator({"vault:enter", "vault:create"})
    )
    assert constructors == (references["Vault.__init__"],)

    read = table.resolve_dot(
        "Vault", "read", evaluator({"vault:enter", "vault:read"})
    )
    assert read.kind == "method"
    assert read.function_reference == references["Vault.read"]

    capacity = table.resolve_dot(
        "Vault", "capacity", evaluator({"vault:enter", "vault:inspect"})
    )
    assert capacity.kind == "attribute"
    assert capacity.storage == "class"
    assert capacity.function_reference is None

    with pytest.raises(PermissionError, match="Vault.erase"):
        table.resolve_dot(
            "Vault", "erase", evaluator({"vault:enter", "vault:read"})
        )
