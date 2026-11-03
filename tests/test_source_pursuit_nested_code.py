import ast

from src.transmogrifier.graph.graph_express2 import (
    _ast_definition_bindings,
    _expand_unresolved_ast_parents,
)


def _nested_code_helper(value):
    return value * 2


def _comprehension_owner(values):
    return [_nested_code_helper(value) for value in values]


def test_comprehension_global_helper_is_an_exact_source_dependency():
    bindings = _ast_definition_bindings(_comprehension_owner)

    assert bindings["_nested_code_helper"] is _nested_code_helper

    tree, links, unresolved, _root_bindings = (
        _expand_unresolved_ast_parents(
            ast.parse(
                "def entry(values):\n"
                "    return _comprehension_owner(values)\n"
            ),
            {"_comprehension_owner": _comprehension_owner},
            include=lambda _value: True,
            pursuit_roots=("entry",),
            activation_roots=("entry",),
        )
    )

    definitions = {
        statement.name
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef)
    }
    assert "_nested_code_helper" in definitions
    assert any(
        definition.name == "_nested_code_helper"
        and isinstance(call.func, ast.Name)
        and call.func.id == "_nested_code_helper"
        for definition, call in links
    )
    assert not any(
        item["name"] == "_nested_code_helper" for item in unresolved
    )
