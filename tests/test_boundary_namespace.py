import ast
import json

import pytest

from src.transmogrifier.graph.boundary_namespace import (
    BoundaryNamespace,
    BoundaryNamespaceError,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot


def _record(directory, name, payload):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.node.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_sparse_language_then_oop_scope_spoofs_one_exact_call(tmp_path):
    # The LeakyLayer directory is deliberately absent. Sparse traversal keeps
    # looking for the method level below the last directory that did exist.
    _record(tmp_path / "python" / "forward", "opaque_external", {
        "version": 1,
        "id": "python.leaky.forward.external",
        "action": "spoof",
        "node_type": "Call",
        "match": {"func.id": "external_runtime"},
        "graph_match": {"class_definitions": ["LeakyLayer"]},
        "result": {
            "type": "spoofed_external",
            "attributes": {"boundary": "opaque"},
            "attributes_from_node": {"callee": "func.id"},
            "attributes_from_graph": {"known_classes": "class_definitions"},
        },
    })
    graph = ProcessGraph(
        materialize_memory=False,
        boundary_namespace=tmp_path,
    )
    graph.build_from_ast(ast.parse("""
class LeakyLayer:
    def forward(self, value):
        return external_runtime(value)
"""))

    spoof = next(
        data for _node, data in graph.G.nodes(data=True)
        if data.get("type") == "spoofed_external"
    )
    assert spoof["attributes"] == {
        "boundary": "opaque",
        "callee": "external_runtime",
        "known_classes": ("LeakyLayer",),
    }
    assert spoof["boundary_rule"] == "python.leaky.forward.external"
    assert spoof["boundary_action"] == "spoof"
    assert graph.G.graph["boundary_namespace_receipts"][0]["scope"] == (
        "LeakyLayer", "forward"
    )


def test_precise_exclusion_removes_only_named_inherited_rule(tmp_path):
    _record(tmp_path / "python", "default_call", {
        "version": 1,
        "id": "python.default.call",
        "action": "spoof",
        "node_type": "Call",
        "match": {"func.id": "external_runtime"},
        "result": {"type": "default_spoof"},
    })
    _record(tmp_path / "python" / "ExactClass", "exclude_default", {
        "version": 1,
        "id": "python.exclusions.exact_class",
        "action": "exclude",
        "target": "python.default.call",
    })
    namespace = BoundaryNamespace(tmp_path)
    excluded = ast.parse("external_runtime(x)").body[0].value
    excluded._turing_source_scope = ("ExactClass",)
    ordinary = ast.parse("external_runtime(x)").body[0].value
    ordinary._turing_source_scope = ("OtherClass",)
    graph = ProcessGraph(materialize_memory=False)

    excluded_resolution = namespace.resolve(excluded, graph)
    ordinary_resolution = namespace.resolve(ordinary, graph)

    assert excluded_resolution.special_case is None
    assert excluded_resolution.excluded_rule_ids == ("python.default.call",)
    assert ordinary_resolution.special_case.type == "default_spoof"


def test_scope_schema_override_uses_existing_process_graph_walker(tmp_path):
    _record(tmp_path / "javascript" / "Widget", "widget_pair", {
        "version": 1,
        "id": "javascript.Widget.PairNode",
        "action": "schema",
        "node_type": "PairNode",
        "role_schema": {"up": {"left": 1, "right": 1}, "down": {}},
    })

    PairNode = type("PairNode", (), {})
    LeafNode = type("LeafNode", (), {})
    pair = PairNode()
    pair.left = LeafNode()
    pair.right = LeafNode()
    pair._turing_source_scope = ("Widget",)
    graph = ProcessGraph(
        materialize_memory=False,
        boundary_namespace=tmp_path,
        source_language="javascript",
    )
    graph.build_graph(pair)

    pair_data = graph.G.nodes[id(pair)]
    assert pair_data["boundary_rule"] == "javascript.Widget.PairNode"
    assert {role for _parent, role in pair_data["parents"]} == {"left", "right"}


def test_unknown_manifest_keys_fail_loud_instead_of_becoming_policy(tmp_path):
    _record(tmp_path / "python", "bad", {
        "version": 1,
        "id": "python.bad",
        "action": "spoof",
        "node_type": "Call",
        "execute_python": "please_do_not",
        "result": {"type": "bad"},
    })
    namespace = BoundaryNamespace(tmp_path)

    with pytest.raises(BoundaryNamespaceError, match="unknown boundary keys"):
        namespace.rules_for_scope(())


def test_aot_entrypoint_uses_the_same_boundary_namespace(tmp_path):
    _record(tmp_path / "python" / "kernel", "helper_boundary", {
        "version": 1,
        "id": "python.kernel.helper_boundary",
        "action": "spoof",
        "node_type": "Call",
        "match": {"func.id": "helper"},
        "result": {
            "type": "opaque_helper_boundary",
            "attributes": {"reason": "bounded test"},
        },
    })
    compilation = compile_ast_aot(
        """
def helper(value):
    return value * value

def kernel(value):
    return helper(value)
""",
        "kernel",
        {"value": 2.0},
        precompile_only=True,
        runtime_closure_only=True,
        boundary_namespace=tmp_path,
    )
    graph = compilation.deployment.process_graph

    assert any(
        data.get("type") == "opaque_helper_boundary"
        and data.get("boundary_rule") == "python.kernel.helper_boundary"
        for _node, data in graph.G.nodes(data=True)
    )
