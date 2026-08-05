"""JavaScript expression <-> ProcessGraph operator-table compatibility.

Mirrors symbolic_process_graph.py's sympy<->ProcessGraph bridge, but is
driven by javascript_source_tables.py's lexical operator spellings instead
of a Python-object translation table. Those spellings select the same
Handler vocabulary GLSL and sympy already lower into (see
javascript_source_tables.py's module docstring), so a JS expression and an
equivalent Python/sympy expression land on identical canonical node types.

Nodes are built with the exact schema symbolic_process_graph.py's
``make_node`` uses (``type``/``op``/``label``/``attributes``/``constant``/
``tensor``/``bit_quanta``/``parents``/``children``, edges running operand
-> consumer) because that schema is what ``process_graph_to_ssa_instrs``
actually requires -- not an approximation of it. A hand-rolled node shape
that merely carries the right Handler *name* but not this schema fails in
the real AOT/SSA pipeline with ``KeyError: 'children'`` before it ever
reaches scheduling; this module is written and tested against the real
pipeline to rule that out (see test_javascript_process_graph.py's
``test_from_javascript_lowers_through_the_real_aot_ssa_pipeline``).

`ingest_javascript_expression` is the from-javascript direction: one ESTree
expression (as produced by vendor/js_ast_parse.js) becomes canonical
Handler-tagged ProcessGraph nodes. `javascript_source_from_graph` is the
to-javascript direction: it renders such nodes back into JS source text.
Operators with no exact Handler equivalent are recorded as shortfalls
rather than silently mis-lowered, matching glsl_source_ingestion.py's
GLSLSourceShortfall discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from ..transmogrifier.ssa_registry import Handler
from .javascript_source_tables import (
    JAVASCRIPT_BINARY_TO_SSA,
    JAVASCRIPT_DIRECT_CALLS,
    JAVASCRIPT_UNARY_TO_SSA,
    JAVASCRIPT_UNSUPPORTED_BINARY,
    JAVASCRIPT_UNSUPPORTED_UNARY,
    SSA_TO_JAVASCRIPT_BINARY,
    SSA_TO_JAVASCRIPT_UNARY,
)

_REVERSE_DIRECT_CALLS: Mapping[str, str] = {
    name: spelling for spelling, name in JAVASCRIPT_DIRECT_CALLS.items()
}


@dataclass(frozen=True, order=True)
class JavaScriptExpressionShortfall:
    code: str
    message: str

    def format(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass
class JavaScriptExpressionLowering:
    graph: Any
    root: int
    shortfalls: tuple[JavaScriptExpressionShortfall, ...]

    @property
    def complete(self) -> bool:
        return not self.shortfalls


def _callee_spelling(callee: dict | None) -> str:
    if not callee:
        return ""
    if callee.get("type") == "Identifier":
        return callee.get("name", "")
    if callee.get("type") == "MemberExpression" and not callee.get("computed"):
        object_spelling = _callee_spelling(callee.get("object"))
        property_name = (callee.get("property") or {}).get("name", "")
        return f"{object_spelling}.{property_name}" if object_spelling else property_name
    return ""


def ingest_javascript_expression(
    graph: Any, node: dict, *, owner: str,
) -> JavaScriptExpressionLowering:
    """Translate one ESTree expression into canonical Handler-tagged nodes.

    Node shape matches symbolic_process_graph.py's ``make_node`` exactly --
    the real schema the AOT/SSA pipeline consumes -- not a parallel one.
    """

    shortfalls: list[JavaScriptExpressionShortfall] = []
    identifiers: dict[str, int] = {}
    next_id = [
        max((item for item in graph.G if isinstance(item, int)), default=-1) + 1
    ]

    def make_node(
        node_type: str,
        operation: str | None,
        label: str,
        parents: Sequence[tuple[int, str]],
        *,
        attributes: Mapping[str, Any] | None = None,
        constant: Any = None,
    ) -> int:
        node_id = next_id[0]
        next_id[0] += 1
        attributes = dict(attributes or {})
        attributes.setdefault("source_language", "javascript")
        parents = list(parents)
        graph.G.add_node(
            node_id,
            type=node_type,
            op=operation,
            label=label,
            attributes=attributes,
            constant=constant,
            tensor={},
            bit_quanta={},
            parents=parents,
            children=[],
        )
        for parent_id, role in parents:
            graph.G.add_edge(parent_id, node_id)
            graph.G.nodes[parent_id].setdefault("children", []).append(
                (node_id, role)
            )
        return node_id

    def unsupported(spelling: str, code: str, message: str) -> int:
        shortfalls.append(JavaScriptExpressionShortfall(code, message))
        return make_node(
            "Unsupported", None, spelling, (),
            attributes={"spelling": spelling},
        )

    def lower(expr: dict) -> int:
        kind = expr.get("type")
        if kind in ("BinaryExpression", "LogicalExpression"):
            operator = expr.get("operator")
            handler = JAVASCRIPT_BINARY_TO_SSA.get(operator)
            if handler is None:
                reason = (
                    "has no exact Handler equivalent"
                    if operator in JAVASCRIPT_UNSUPPORTED_BINARY
                    else "is not a recognized JS binary operator"
                )
                return unsupported(
                    operator, "unsupported-binary-operator",
                    f"{operator!r} {reason}",
                )
            left = lower(expr["left"])
            right = lower(expr["right"])
            return make_node(
                handler.name, handler.name, handler.name,
                [(left, "left"), (right, "right")],
            )
        if kind == "UnaryExpression":
            operator = expr.get("operator")
            handler = JAVASCRIPT_UNARY_TO_SSA.get(operator)
            if handler is None:
                reason = (
                    "has no exact Handler equivalent"
                    if operator in JAVASCRIPT_UNSUPPORTED_UNARY
                    else "is not a recognized JS unary operator"
                )
                return unsupported(
                    operator, "unsupported-unary-operator",
                    f"{operator!r} {reason}",
                )
            operand = lower(expr["argument"])
            return make_node(
                handler.name, handler.name, handler.name,
                [(operand, "operand")],
            )
        if kind == "CallExpression":
            callee_spelling = _callee_spelling(expr.get("callee"))
            canonical = JAVASCRIPT_DIRECT_CALLS.get(callee_spelling)
            arguments = [lower(argument) for argument in expr.get("arguments", ())]
            parents = [
                (argument, f"arg{index}")
                for index, argument in enumerate(arguments)
            ]
            if canonical is None:
                shortfalls.append(JavaScriptExpressionShortfall(
                    "unresolved-call",
                    f"{callee_spelling!r} is not in JAVASCRIPT_DIRECT_CALLS",
                ))
                return make_node(
                    "Call", Handler.Call.name, callee_spelling, parents,
                    attributes={"callee": callee_spelling},
                )
            return make_node(
                "Call", Handler.Call.name, canonical, parents,
                attributes={"callee": canonical},
            )
        if kind == "Identifier":
            name = expr["name"]
            if name in identifiers:
                return identifiers[name]
            node_id = make_node(
                "Input", Handler.Load.name, name, (),
                attributes={"binding_name": name},
            )
            identifiers[name] = node_id
            return node_id
        if kind == "Literal":
            value = expr.get("value")
            return make_node(
                "Constant", Handler.Const.name, repr(value), (),
                attributes={"value": value}, constant=value,
            )
        return unsupported(
            kind, "unsupported-node-type",
            f"{kind!r} has no ProcessGraph translation",
        )

    root = lower(node)
    graph.roots = [root]
    return JavaScriptExpressionLowering(graph, root, tuple(shortfalls))


def _javascript_literal_spelling(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def javascript_source_from_graph(graph: Any, node_id: int) -> str:
    """Render one canonical Handler-tagged ProcessGraph node back to JS.

    The inverse of ingest_javascript_expression's binary/unary/call/const/
    load cases, reading the same ``parents`` list ``make_node`` populates,
    using SSA_TO_JAVASCRIPT_BINARY/UNARY -- the same tables whose reverse
    direction validate_invertible_tables() already proves bijective.
    """

    data = graph.G.nodes[node_id]
    node_type = data.get("type")
    if node_type == "Constant":
        return _javascript_literal_spelling(data["attributes"].get("value"))
    if node_type == "Input":
        return str(data["attributes"].get("binding_name", node_id))
    if node_type == "Call":
        callee = data["attributes"].get("callee", "")
        spelling = _REVERSE_DIRECT_CALLS.get(callee, callee)
        by_role = {role: parent for parent, role in data["parents"]}
        arguments = [
            javascript_source_from_graph(graph, by_role[role])
            for role in sorted(by_role, key=lambda item: int(item[3:]))
        ]
        return f"{spelling}({', '.join(arguments)})"
    try:
        handler = Handler[node_type]
    except KeyError as error:
        raise ValueError(f"cannot render node type {node_type!r} to javascript") from error
    by_role = {role: parent for parent, role in data["parents"]}
    if handler in SSA_TO_JAVASCRIPT_UNARY:
        rendered = javascript_source_from_graph(graph, by_role["operand"])
        return f"{SSA_TO_JAVASCRIPT_UNARY[handler]}({rendered})"
    if handler in SSA_TO_JAVASCRIPT_BINARY:
        left = javascript_source_from_graph(graph, by_role["left"])
        right = javascript_source_from_graph(graph, by_role["right"])
        return f"({left} {SSA_TO_JAVASCRIPT_BINARY[handler]} {right})"
    raise ValueError(f"cannot render Handler {handler} to javascript")


__all__ = [
    "JavaScriptExpressionLowering",
    "JavaScriptExpressionShortfall",
    "ingest_javascript_expression",
    "javascript_source_from_graph",
]
