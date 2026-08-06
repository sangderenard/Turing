"""Translation tables letting ``ProcessGraph`` understand OOP languages
other than Python, following the working SymPy precedent (see
``GRAPH_DESCRIPTION_LAYER_SURVEY.md``).

``ProcessGraph.build_graph`` dispatches on ``type(node).__name__`` and reads
child roles via ``getattr(node, role_name)`` against ``role_schemas``
(``operator_defs.py``). SymPy proves this needs no Python-AST-specific
machinery: real ``sympy`` objects are handed to the walker directly and it
works because their class names already match registered schema keys and
their attributes already match the declared shape.

A foreign language's parser rarely hands back objects shaped that way,
though -- acorn (this repo's JS parser, ``vendor/js_ast_parse.js``) returns
plain nested JSON dicts (``{"type": "BinaryExpression", "left": ..., ...}``),
which do not expose their keys as attributes and whose Python type is
always ``dict``, never ``"BinaryExpression"``. ``estree_dict_to_node``
bridges exactly that gap: it walks a JSON-shaped dict tree and produces
lightweight objects whose class name is the ESTree ``type`` field and whose
attributes are the dict's other keys, recursively. Once wrapped, the walker
needs no further changes -- it is the same generic path SymPy already uses.

Status, honestly, per language (update this as coverage changes):

* **JavaScript (ESTree, via acorn)** -- ``estree_dict_to_node`` plus
  ``JS_ROLE_SCHEMAS`` below are real and verified (see
  ``tests/test_oop_language_translations.py``) for a genuinely simple
  program: variable declarations, binary expressions, arrow functions,
  return statements, call expressions. NOT verified for classes,
  destructuring, template literals, async/await, or anything from the
  "torture test" (``examples/torture_five_languages.dream``) beyond what
  the test file actually exercises. Extend ``JS_ROLE_SCHEMAS`` and the
  special-case switch (``node_special_cases.py``) as real gaps are found,
  the same way the existing Python/SymPy entries grew.
* **C++ (narrow shell, via ``cpp_shell_desugar`` + ``pycparser``)** --
  real and verified for the same narrow scope
  ``cpp_shell_desugar.py``/``CPP_LIKE_SHELL_FOR_C_INTENT.md`` describe:
  classes with fields/methods/one constructor, single inheritance, no
  templates/virtual/operator-overloading/multiple-inheritance. Unlike
  JavaScript, ``pycparser``'s ``c_ast`` nodes are already real Python
  objects with real attributes (no dict-wrapping step needed) --
  ``C_ROLE_SCHEMAS`` below registers them directly.
* **Java** -- not started, no parser identified anywhere in this repo.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..operator_defs import role_schemas


_NODE_CLASS_CACHE: dict[str, type] = {}


def _node_class(type_name: str) -> type:
    cached = _NODE_CLASS_CACHE.get(type_name)
    if cached is not None:
        return cached
    # A real class per ESTree ``type`` string, not one generic wrapper
    # class -- ``type(node).__name__`` is exactly how the walker
    # dispatches, the same way it dispatches on ``ast.BinOp``/``ast.Call``
    # or ``sympy.Sum``/``sympy.Indexed``.
    created = type(type_name, (), {})
    _NODE_CLASS_CACHE[type_name] = created
    return created


def estree_dict_to_node(value: Any) -> Any:
    """Recursively convert an acorn ESTree JSON tree into attribute objects.

    A dict without a ``"type"`` key (acorn emits a few of these, e.g. some
    ``loc``/``range`` metadata when locations are enabled) is left as a
    plain dict -- nothing in ``JS_ROLE_SCHEMAS`` ever names it as a child
    role, so the walker never looks at it.
    """

    if isinstance(value, Mapping):
        node_type = value.get("type")
        if node_type is None:
            return value
        instance = _node_class(str(node_type))()
        for key, item in value.items():
            if key == "type":
                continue
            setattr(instance, key, estree_dict_to_node(item))
        return instance
    if isinstance(value, list):
        return [estree_dict_to_node(item) for item in value]
    return value


# Deliberately minimal: exactly what the verified test program below needs,
# not a speculative full ESTree grammar. Grow this the way the Python/SymPy
# tables grew -- one real construct at a time, verified, not guessed ahead.
JS_ROLE_SCHEMAS: dict[str, dict[str, dict[str, Any]]] = {
    "Program": {"up": {"body": "many"}, "down": {}},
    "VariableDeclaration": {"up": {"declarations": "many"}, "down": {}},
    "VariableDeclarator": {"up": {"id": 1, "init": 1}, "down": {}},
    "Identifier": {"up": {}, "down": {}},
    "Literal": {"up": {}, "down": {}},
    "BinaryExpression": {"up": {"left": 1, "operator": 1, "right": 1}, "down": {}},
    "ArrowFunctionExpression": {
        "up": {"params": "many", "body": 1}, "down": {},
    },
    "BlockStatement": {"up": {"body": "many"}, "down": {}},
    "ReturnStatement": {"up": {"argument": 1}, "down": {}},
    "CallExpression": {"up": {"callee": 1, "arguments": "many"}, "down": {}},
    "ExpressionStatement": {"up": {"expression": 1}, "down": {}},
}


def install_js_role_schemas(graph: Any) -> None:
    """Set ``graph.role_schemas`` to a copy carrying ``JS_ROLE_SCHEMAS``.

    Call once, on the specific ``ProcessGraph`` instance about to receive
    ``estree_dict_to_node`` output, *before* ``graph.build_graph(...)``.

    Deliberately does **not** mutate the shared module-level
    ``operator_defs.role_schemas`` dict in place (an earlier version of
    this function did exactly that, via ``role_schemas.update(...)``, and
    it was a real, live bug: several JS/C node-type names collide with
    Python AST names under a *different* shape -- ``Return`` is
    ``{'value': 1}`` in Python's own entry vs. ``{'expr': 1}`` for C's
    ``pycparser`` node, and ``UnaryOp`` differs the same way. Mutating the
    global silently corrupted every Python ``ProcessGraph`` built for the
    rest of the process after this function ran once -- caught only
    because a later, unrelated test file happened to run in the same
    pytest session and started failing on ordinary Python ``return``
    statements. Each graph now gets its own copy instead.
    """

    graph.role_schemas = {**role_schemas, **JS_ROLE_SCHEMAS}


# ``pycparser``'s ``c_ast`` nodes are already real Python objects (unlike
# acorn's JSON dicts) -- no wrapping step, just registration. Field names
# and shapes taken directly from ``pycparser.c_ast``'s ``__slots__``, not
# guessed: a scalar attribute (``op``, a plain string; ``name`` on ``Decl``,
# also a plain string) is left out of ``up`` the same way ``ast.Name.id``
# is left out of Python's own entries -- it is data on the node, not a
# child to descend into. ``StructRef`` is a case worth noting explicitly:
# despite the name, ``StructRef.name`` is the *base expression* node (e.g.
# the ``self`` in ``self->value``), not a string -- confirmed against a
# real parse (``ast.show()``) before writing this, not assumed from the
# attribute name.
C_ROLE_SCHEMAS: dict[str, dict[str, dict[str, Any]]] = {
    "FileAST": {"up": {"ext": "many"}, "down": {}},
    "Decl": {"up": {"type": 1, "init": 1}, "down": {}},
    "FuncDef": {"up": {"decl": 1, "body": 1}, "down": {}},
    "FuncDecl": {"up": {"args": 1, "type": 1}, "down": {}},
    "Struct": {"up": {"decls": "many"}, "down": {}},
    "TypeDecl": {"up": {"type": 1}, "down": {}},
    "IdentifierType": {"up": {}, "down": {}},
    "ParamList": {"up": {"params": "many"}, "down": {}},
    "PtrDecl": {"up": {"type": 1}, "down": {}},
    "Compound": {"up": {"block_items": "many"}, "down": {}},
    "Assignment": {"up": {"lvalue": 1, "rvalue": 1}, "down": {}},
    "BinaryOp": {"up": {"left": 1, "right": 1}, "down": {}},
    "ID": {"up": {}, "down": {}},
    "Constant": {"up": {}, "down": {}},
    "StructRef": {"up": {"name": 1, "field": 1}, "down": {}},
    "Return": {"up": {"expr": 1}, "down": {}},
    "Typedef": {"up": {"type": 1}, "down": {}},
    "InitList": {"up": {"exprs": "many"}, "down": {}},
    "UnaryOp": {"up": {"expr": 1}, "down": {}},
    "FuncCall": {"up": {"name": 1, "args": 1}, "down": {}},
    "ExprList": {"up": {"exprs": "many"}, "down": {}},
}


def install_c_role_schemas(graph: Any) -> None:
    """Set ``graph.role_schemas`` to a copy carrying ``C_ROLE_SCHEMAS``.

    Call once, on the specific ``ProcessGraph`` instance about to receive a
    ``pycparser`` ``FileAST`` (from ``desugar_cpp_shell`` output), *before*
    ``graph.build_graph(...)``. See ``install_js_role_schemas`` for why this
    sets a per-instance copy rather than mutating the shared
    ``operator_defs.role_schemas`` dict -- the same class-of-bug applies
    here (``Return``, ``UnaryOp`` collide with Python's entries under a
    different shape).
    """

    graph.role_schemas = {**role_schemas, **C_ROLE_SCHEMAS}


__all__ = [
    "estree_dict_to_node",
    "install_js_role_schemas",
    "JS_ROLE_SCHEMAS",
    "install_c_role_schemas",
    "C_ROLE_SCHEMAS",
]
