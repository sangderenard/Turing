"""Verify a foreign language (JavaScript, via acorn) can reach ProcessGraph
through role_schemas the same way SymPy already does -- see
GRAPH_DESCRIPTION_LAYER_SURVEY.md and oop_language_translations.py's own
docstring for what this does and does not cover.
"""

import pytest

from src.compiler.dream_document import _run_javascript_ast_parser, DreamDocumentError
from src.transmogrifier.graph.graph_express2 import ProcessGraph
from src.transmogrifier.graph.oop_language_translations import (
    estree_dict_to_node,
    install_js_role_schemas,
)


def _js_process_graph(source: str) -> ProcessGraph:
    try:
        tree = _run_javascript_ast_parser(source)
    except DreamDocumentError as error:
        pytest.skip(f"no node executable available to parse JS: {error}")
    root = estree_dict_to_node(tree)
    graph = ProcessGraph(materialize_memory=False)
    install_js_role_schemas(graph)
    graph.build_graph(root)
    return graph


def _types(graph: ProcessGraph) -> list[str]:
    return [str(data.get("type")) for _, data in graph.G.nodes(data=True)]


def test_estree_dict_to_node_wraps_type_field_as_python_class_name():
    node = estree_dict_to_node({"type": "Identifier", "name": "x"})
    assert type(node).__name__ == "Identifier"
    assert node.name == "x"


def test_estree_dict_to_node_recurses_through_lists_and_nested_dicts():
    node = estree_dict_to_node({
        "type": "Program",
        "body": [{"type": "Identifier", "name": "a"}, {"type": "Identifier", "name": "b"}],
    })
    assert type(node).__name__ == "Program"
    assert [child.name for child in node.body] == ["a", "b"]


def test_simple_variable_declaration_reaches_process_graph_with_real_types():
    graph = _js_process_graph("const result = a + b;")
    types = _types(graph)
    assert "Program" in types
    assert "VariableDeclaration" in types
    assert "VariableDeclarator" in types
    assert "BinaryExpression" in types
    assert types.count("Identifier") == 3  # result, a, b


def test_arrow_function_and_call_expression_reach_process_graph():
    graph = _js_process_graph("const double = (x) => x * 2; double(21);")
    types = _types(graph)
    assert "ArrowFunctionExpression" in types
    assert "CallExpression" in types
    assert "BinaryExpression" in types
