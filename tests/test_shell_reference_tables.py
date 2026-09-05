import ast
import contextlib
import io

import pytest

from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
from src.compiler.glsl_deployment_strategy import (
    _is_dispatch_metadata_node,
    _walk_planned_shells,
    strategize_shell_deployment,
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


def test_shell_reference_tables_cache_recursion_regions():
    _graph, record_graph = _reduced_function_graph()
    record_graph.G.graph["recursion_table"] = {
        3: {
            "lower_as": "while",
            "members": (4, 5),
            "incoming": ((1, 4, "value"),),
            "outgoing": ((5, 8, "result"),),
            "feedback": ((5, 4, "carried"),),
        }
    }

    tables = build_shell_reference_tables(record_graph)

    entry, = tables.recursion
    assert entry.index == 0
    assert entry.region_id == 3
    assert entry.lower_as == "while"
    assert entry.feedback == ((5, 4, "carried"),)
    assert any(
        correlation.table == "recursion"
        and correlation.source_reference == 3
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
    shell_type = strategize_shell_deployment(graph)
    first = shell_type()
    second = shell_type()
    try:
        assert first.function_references == second.function_references
        assert first.constant_references == second.constant_references
        assert first.memory_references == second.memory_references
        assert first.recursion_references == second.recursion_references
        assert first.reference_correlations == second.reference_correlations

        assert first.function_references is not second.function_references
        assert first.constant_references is not second.constant_references
        assert first.memory_references is not second.memory_references
        assert first.recursion_references is not second.recursion_references
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


def test_deployment_shells_follow_proven_runtime_closure_not_map_only_catalogue():
    module = ast.parse(
        '''
class Archive:
    def read(self, index):
        return helper(index)

    def compact(self):
        return dead_helper()

def helper(index):
    return index + 1

def dead_helper():
    return 0
'''
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    regions = build_map_dependency_regions(graph, "Archive.read")
    graph.G.graph["compile_targets"] = ("Archive.read",)
    map_ir = dict(graph.G.graph.get("map_ir") or {})
    map_ir["dependency_regions"] = {
        "runtime": regions.runtime,
        "mapped": regions.mapped,
        "retained": regions.retained,
        "map_only": regions.map_only,
        "bindings": regions.bindings,
    }
    graph.G.graph["map_ir"] = map_ir

    deployment_type = strategize_shell_deployment(
        graph, runtime_closure_only=True
    )
    planned = set(deployment_type.function_shell_types)
    references = {
        entry.qualified_name: int(entry.reference.address)
        for entry in graph.function_table
    }

    assert planned == set(regions.runtime)
    assert references["Archive.read"] in planned
    assert references["helper"] in planned
    assert references["Archive.compact"] not in planned
    assert references["dead_helper"] not in planned
    assert deployment_type.activation_root_references == (
        references["Archive.read"],
    )
    assert set(deployment_type.catalogue_only_function_references) == (
        set(references.values()) - planned
    )
    # Liveness reduction does not erase the catalogue or its map-only class
    # records; it changes only which definitions become executable shells.
    assert graph.function_table.entry("Archive.compact").graph is not None
    assert references["Archive.compact"] in regions.map_only

    deployment = deployment_type()
    try:
        assert deployment.callsite_function_shells == {}
        read_shell = deployment.function_shells[references["Archive.read"]]
        activation_shells = tuple(_walk_planned_shells(
            read_shell, include_function_registry=False
        ))
        activated_names = {
            shell.process_graph.G.graph.get("function_name")
            for shell in activation_shells
        }
        assert "read" in activated_names
        assert "helper" in activated_names
        assert "Archive.compact" not in activated_names
        assert "dead_helper" not in activated_names
        # One selected root plus its one callsite activation. The helper's
        # catalogue definition remains separate, so the administrative walk
        # contains root deployment + two definitions + one activation. This
        # bound catches a return to expanding a suffix tree from every
        # catalogue definition.
        assert len(activation_shells) == 2
        assert len(tuple(_walk_planned_shells(deployment))) == 4
        execution_shells = tuple(_walk_planned_shells(
            deployment, include_function_registry=False
        ))
        assert len(execution_shells) == 3
        assert {
            shell.process_graph.G.graph.get("function_name")
            for shell in execution_shells
        } >= {"read", "helper"}
        for shell in activation_shells:
            expected_callsites = {
                int(node_id)
                for node_id, data in shell.process_graph.G.nodes(data=True)
                if any(
                    reference is not None and int(reference) in planned
                    for reference in (
                        (data.get("attributes") or {}).get("callee_ref"),
                        (data.get("attributes") or {}).get("method_ref"),
                    )
                )
            }
            assert set(shell.callsite_function_shells) == expected_callsites
        deployment.compile_process_graph()
        assert deployment.whole_program_compiled
        assert all(shell.whole_program_compiled for shell in activation_shells)
    finally:
        deployment.release()

    complete_catalogue = strategize_shell_deployment(
        graph, runtime_closure_only=False
    )
    assert references["Archive.compact"] in (
        complete_catalogue.function_shell_types
    )
    assert references["dead_helper"] in complete_catalogue.function_shell_types


def test_method_reference_requires_receiver_class_identity():
    module = ast.parse(
        '''
class Queue:
    def get(self, key):
        return key

def external_get(mapping):
    return mapping.get("value")

class Box:
    def read(self):
        return 1

def internal_read():
    box = Box()
    return box.read()
'''
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    deployment_type = strategize_shell_deployment(graph)
    external = deployment_type.function_shell_types[
        graph.function_table.entry("external_get").reference.address
    ].process_graph
    external_call = next(
        data
        for _, data in external.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and isinstance(data["expr_obj"].func, ast.Attribute)
        and data["expr_obj"].func.attr == "get"
    )
    assert (external_call.get("attributes") or {}).get("method_ref") is None

    internal = deployment_type.function_shell_types[
        graph.function_table.entry("internal_read").reference.address
    ].process_graph
    internal_call = next(
        data
        for _, data in internal.G.nodes(data=True)
        if isinstance(data.get("expr_obj"), ast.Call)
        and isinstance(data["expr_obj"].func, ast.Attribute)
        and data["expr_obj"].func.attr == "read"
    )
    assert (internal_call.get("attributes") or {})["method_ref"] == (
        graph.function_table.entry("Box.read").reference.address
    )


def test_tensor_method_candidate_requires_tensor_receiver_value():
    module = ast.parse(
        '''
def split_text(value):
    return value.split(".")

def negate_tensor(value):
    return value.neg()
'''
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    class TensorValue:
        shape = (2,)
        dtype = "float64"

    graph.function_table.entry(
        "negate_tensor"
    ).graph.G.graph["planner_specializations"] = {"value": TensorValue()}
    deployment_type = strategize_shell_deployment(graph)

    split_graph = deployment_type.function_shell_types[
        graph.function_table.entry("split_text").reference.address
    ].process_graph
    split_call = next(
        data for _, data in split_graph.G.nodes(data=True)
        if (data.get("attributes") or {}).get("tensor_candidate") == "split"
    )
    assert (split_call.get("attributes") or {}).get("tensor") is None
    split_node = next(
        node_id for node_id, data in split_graph.G.nodes(data=True)
        if data is split_call
    )
    assert _is_dispatch_metadata_node(split_graph, split_node)

    tensor_graph = deployment_type.function_shell_types[
        graph.function_table.entry("negate_tensor").reference.address
    ].process_graph
    neg_call = next(
        data for _, data in tensor_graph.G.nodes(data=True)
        if (data.get("attributes") or {}).get("tensor_candidate") == "neg"
    )
    assert (neg_call.get("attributes") or {})["tensor"] == "neg"
    neg_node = next(
        node_id for node_id, data in tensor_graph.G.nodes(data=True)
        if data is neg_call
    )
    assert not _is_dispatch_metadata_node(tensor_graph, neg_node)


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


def test_instance_field_shadows_same_named_non_data_method():
    module = ast.parse(
        '''
class Adapter:
    def __init__(self, nodes):
        self.nodes = nodes

    def nodes(self):
        return ()
'''
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    table = build_class_navigation_table(graph)
    allow = lambda _identity, _permissions: True

    instance_member = table.resolve_dot("Adapter", "nodes", allow)
    class_member = table.resolve_dot(
        "Adapter", "nodes", allow, receiver_kind="class"
    )

    assert instance_member.kind == "attribute"
    assert instance_member.storage == "instance"
    assert class_member.kind == "method"
    assert class_member.function_reference is not None


def test_same_named_classes_use_discovered_module_qualified_identity():
    first = ast.parse("class Edge:\n    left = 1\n").body[0]
    second = ast.parse("class Edge:\n    right = 2\n").body[0]
    first._python_source_identity = ("package.solver", "Edge")
    second._python_source_identity = ("package.renderer", "Edge")
    module = ast.Module(body=[first, second], type_ignores=[])
    ast.fix_missing_locations(module)

    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    assert {
        item["class_identity"]
        for item in graph.G.graph["map_ir"]["objects"]
    } == {"package.solver.Edge", "package.renderer.Edge"}
    table = build_class_navigation_table(graph)
    assert {record.identity for record in table.classes} == {
        "package.solver.Edge",
        "package.renderer.Edge",
    }
