"""Exact callable identity governs Token Orchard source-discovery edges."""

import ast
import sys
from pathlib import Path

import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.transmogrifier.graph.graph_express2 import (
    ProcessGraph,
    _expand_unresolved_ast_parents,
    _resolve_ast_parent_reference,
)


def test_builtin_bytes_does_not_link_to_string_namespace_method():
    core_root = Path(__file__).resolve().parents[2] / "string_core"
    if not (core_root / "token_orchard/api.py").is_file():
        pytest.skip("Token Orchard source is not checked out")
    sys.path.insert(0, str(core_root))
    from token_orchard.api import String

    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts/program_extraction.yaml"
    ).with_sources((
        ("token_orchard.api", core_root / "token_orchard/api.py"),
        ("token_orchard.__init__", core_root / "token_orchard/__init__.py"),
    ))
    tree, links, unresolved, _ = _expand_unresolved_ast_parents(
        ast.parse("def root(payload):\n    return String.bytes(payload)\n"),
        {"String": String}, include=contract, pursuit_roots=("root",),
    )
    byte_values = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_byte_values"
    )
    builtin_call = next(
        node for node in ast.walk(byte_values)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "bytes"
    )
    assert _resolve_ast_parent_reference(
        builtin_call.func, byte_values._python_bindings,
    ) is bytes
    assert not any(call is builtin_call for _definition, call in links)
    assert any(
        row["name"] == "bytes" and row["target_module"] == "builtins"
        and row["target_qualname"] == "bytes"
        for row in unresolved
    )

    graph = ProcessGraph(materialize_memory=False)
    graph.python_bindings = {"String": String}
    graph.build_from_ast(
        "def root(payload):\n    return String.bytes(payload)\n",
        resolve_unresolved_parents=True, pursuit_roots=("root",),
        parent_include=contract,
    )
    graph_calls = (
        data for _node_id, data in graph.G.nodes(data=True)
        if data.get("expr_obj") is not None
        and isinstance(data["expr_obj"], ast.Call)
        and isinstance(data["expr_obj"].func, ast.Name)
        and data["expr_obj"].func.id == "bytes"
    )
    call_record, = graph_calls
    assert call_record["attributes"]["extraction_identity"] == "builtins.bytes"
    assert "resolved_ast_parent" not in call_record["attributes"]
