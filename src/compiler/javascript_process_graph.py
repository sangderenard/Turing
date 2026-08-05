"""JavaScript expression <-> ProcessGraph operator-table compatibility.

Mirrors symbolic_process_graph.py's sympy<->ProcessGraph bridge, but is
driven by javascript_source_tables.py's lexical operator spellings instead
of a Python-object translation table. Those spellings select the same
Handler vocabulary GLSL and sympy already lower into (see
javascript_source_tables.py's module docstring), so a JS expression and an
equivalent Python/sympy expression land on identical canonical node types.

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
from typing import Any, Mapping

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
    root: str
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
    """Translate one ESTree expression into canonical Handler-tagged nodes."""

    shortfalls: list[JavaScriptExpressionShortfall] = []
    counter = [0]

    def next_id() -> str:
        counter[0] += 1
        return f"{owner}:expr:{counter[0]}"

    def unsupported(node_id: str, spelling: str, code: str, message: str) -> None:
        shortfalls.append(JavaScriptExpressionShortfall(code, message))
        graph.G.add_node(node_id, type="Unsupported", spelling=spelling)

    def lower(expr: dict) -> str:
        kind = expr.get("type")
        if kind in ("BinaryExpression", "LogicalExpression"):
            operator = expr.get("operator")
            handler = JAVASCRIPT_BINARY_TO_SSA.get(operator)
            node_id = next_id()
            if handler is None:
                reason = (
                    "has no exact Handler equivalent"
                    if operator in JAVASCRIPT_UNSUPPORTED_BINARY
                    else "is not a recognized JS binary operator"
                )
                unsupported(
                    node_id, operator, "unsupported-binary-operator",
                    f"{operator!r} {reason}",
                )
                return node_id
            left = lower(expr["left"])
            right = lower(expr["right"])
            graph.G.add_node(node_id, type=handler.name, operation=handler.name)
            graph.G.add_edge(node_id, left, role="left")
            graph.G.add_edge(node_id, right, role="right")
            return node_id
        if kind == "UnaryExpression":
            operator = expr.get("operator")
            handler = JAVASCRIPT_UNARY_TO_SSA.get(operator)
            node_id = next_id()
            if handler is None:
                reason = (
                    "has no exact Handler equivalent"
                    if operator in JAVASCRIPT_UNSUPPORTED_UNARY
                    else "is not a recognized JS unary operator"
                )
                unsupported(
                    node_id, operator, "unsupported-unary-operator",
                    f"{operator!r} {reason}",
                )
                return node_id
            operand = lower(expr["argument"])
            graph.G.add_node(node_id, type=handler.name, operation=handler.name)
            graph.G.add_edge(node_id, operand, role="operand")
            return node_id
        if kind == "CallExpression":
            callee_spelling = _callee_spelling(expr.get("callee"))
            canonical = JAVASCRIPT_DIRECT_CALLS.get(callee_spelling)
            node_id = next_id()
            arguments = [lower(argument) for argument in expr.get("arguments", ())]
            if canonical is None:
                shortfalls.append(JavaScriptExpressionShortfall(
                    "unresolved-call",
                    f"{callee_spelling!r} is not in JAVASCRIPT_DIRECT_CALLS",
                ))
                graph.G.add_node(
                    node_id, type="Call", operation=Handler.Call.name,
                    callee=callee_spelling,
                )
            else:
                graph.G.add_node(
                    node_id, type="Call", operation=Handler.Call.name,
                    callee=canonical,
                )
            for argument in arguments:
                graph.G.add_edge(node_id, argument, role="argument")
            return node_id
        if kind == "Identifier":
            node_id = expr["name"]
            if node_id not in graph.G:
                graph.G.add_node(
                    node_id, type="Load", operation=Handler.Load.name,
                    binding_name=node_id,
                )
            return node_id
        if kind == "Literal":
            node_id = next_id()
            graph.G.add_node(
                node_id, type="Const", operation=Handler.Const.name,
                value=expr.get("value"),
            )
            return node_id
        node_id = next_id()
        unsupported(
            node_id, kind, "unsupported-node-type",
            f"{kind!r} has no ProcessGraph translation",
        )
        return node_id

    root = lower(node)
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


def javascript_source_from_graph(graph: Any, node_id: str) -> str:
    """Render one canonical Handler-tagged ProcessGraph node back to JS.

    The inverse of ingest_javascript_expression's binary/unary/call/const/
    load cases, using SSA_TO_JAVASCRIPT_BINARY/UNARY -- the same tables
    whose reverse direction validate_invertible_tables() already proves
    bijective.
    """

    data = graph.G.nodes[node_id]
    node_type = data.get("type")
    if node_type == "Const":
        return _javascript_literal_spelling(data.get("value"))
    if node_type == "Load":
        return str(data.get("binding_name", node_id))
    if node_type == "Call":
        callee = data.get("callee", "")
        spelling = _REVERSE_DIRECT_CALLS.get(callee, callee)
        arguments = [
            javascript_source_from_graph(graph, target)
            for _, target, edge in graph.G.out_edges(node_id, data=True)
            if edge.get("role") == "argument"
        ]
        return f"{spelling}({', '.join(arguments)})"
    try:
        handler = Handler[node_type]
    except KeyError as error:
        raise ValueError(f"cannot render node type {node_type!r} to javascript") from error
    if handler in SSA_TO_JAVASCRIPT_UNARY:
        operand = next(
            target for _, target, edge in graph.G.out_edges(node_id, data=True)
            if edge.get("role") == "operand"
        )
        rendered = javascript_source_from_graph(graph, operand)
        return f"{SSA_TO_JAVASCRIPT_UNARY[handler]}({rendered})"
    if handler in SSA_TO_JAVASCRIPT_BINARY:
        edges = {
            edge.get("role"): target
            for _, target, edge in graph.G.out_edges(node_id, data=True)
        }
        left = javascript_source_from_graph(graph, edges["left"])
        right = javascript_source_from_graph(graph, edges["right"])
        return f"({left} {SSA_TO_JAVASCRIPT_BINARY[handler]} {right})"
    raise ValueError(f"cannot render Handler {handler} to javascript")


__all__ = [
    "JavaScriptExpressionLowering",
    "JavaScriptExpressionShortfall",
    "ingest_javascript_expression",
    "javascript_source_from_graph",
]
