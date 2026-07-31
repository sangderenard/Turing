"""Canonical SymPy projection for semantic ProcessGraphs.

AST and SymPy are source languages.  Neither source object's field layout is
the ProcessGraph schema: both are normalized to explicit value nodes and
canonical operations before another source language is rendered.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import sympy


_SYMPY_TO_CANONICAL = {
    sympy.Add: "Add",
    sympy.Mul: "Mul",
    sympy.Pow: "Pow",
    sympy.Equality: "Equality",
    sympy.Unequality: "Unequality",
    sympy.StrictLessThan: "StrictLessThan",
    sympy.LessThan: "LessThanOrEqual",
    sympy.StrictGreaterThan: "StrictGreaterThan",
    sympy.GreaterThan: "GreaterThanOrEqual",
}


@dataclass(frozen=True)
class SymbolicReductionReport:
    source_nodes: int
    rebuilt_nodes: int
    original: tuple[sympy.Basic, ...]
    reduced: tuple[sympy.Basic, ...]

_CANONICAL_FUNCTIONS = {
    "Sin": sympy.sin,
    "sin": sympy.sin,
    "Cos": sympy.cos,
    "cos": sympy.cos,
    "Tan": sympy.tan,
    "tan": sympy.tan,
    "Exp": sympy.exp,
    "exp": sympy.exp,
    "Log": sympy.log,
    "log": sympy.log,
    "Abs": sympy.Abs,
    "abs": sympy.Abs,
    "Sqrt": sympy.sqrt,
    "sqrt": sympy.sqrt,
}

_BINARY = {
    "Add": lambda a, b: a + b,
    "add": lambda a, b: a + b,
    "Sub": lambda a, b: a - b,
    "sub": lambda a, b: a - b,
    "Mul": lambda a, b: a * b,
    "mul": lambda a, b: a * b,
    "Div": lambda a, b: a / b,
    "div": lambda a, b: a / b,
    "truediv": lambda a, b: a / b,
    "FloorDiv": lambda a, b: sympy.floor(a / b),
    "floordiv": lambda a, b: sympy.floor(a / b),
    "Mod": sympy.Mod,
    "mod": sympy.Mod,
    "Pow": lambda a, b: a**b,
    "pow": lambda a, b: a**b,
    "Equality": sympy.Eq,
    "equal": sympy.Eq,
    "Unequality": sympy.Ne,
    "not_equal": sympy.Ne,
    "StrictLessThan": sympy.Lt,
    "less": sympy.Lt,
    "LessThanOrEqual": sympy.Le,
    "less_equal": sympy.Le,
    "StrictGreaterThan": sympy.Gt,
    "greater": sympy.Gt,
    "GreaterThanOrEqual": sympy.Ge,
    "greater_equal": sympy.Ge,
}


def ingest_sympy_expression(graph: Any, expression: sympy.Basic) -> int:
    """Populate ``graph`` from a SymPy expression using canonical value IDs."""

    graph.domain_shape = (1,)
    graph.roots = []
    memo: dict[sympy.Basic, int] = {}
    next_id = max(
        (int(node_id) for node_id in graph.G if isinstance(node_id, int)),
        default=-1,
    ) + 1

    def add_node(value: sympy.Basic) -> int:
        nonlocal next_id
        if value in memo:
            return memo[value]
        parent_ids = (
            ()
            if isinstance(value, (sympy.Symbol, sympy.IndexedBase))
            else tuple(add_node(arg) for arg in value.args)
        )
        node_id = next_id
        next_id += 1
        memo[value] = node_id
        if isinstance(value, (sympy.Symbol, sympy.IndexedBase)):
            node_type = "Input"
            operation = "input"
            attributes = {
                "binding_name": str(value),
                "binding_kind": "symbol",
            }
        elif value is sympy.true or value is sympy.false:
            node_type = "Constant"
            operation = "const"
            attributes = {"value": bool(value)}
        elif value.is_Number:
            node_type = "Constant"
            operation = "const"
            attributes = {"value": value}
        else:
            node_type = _SYMPY_TO_CANONICAL.get(
                type(value), type(value).__name__
            )
            operation = node_type
            attributes = {"source_type": type(value).__name__}
        if isinstance(value, sympy.Indexed):
            roles = ("base", *("index" for _ in value.indices))
        else:
            roles = tuple(f"arg:{index}" for index in range(len(parent_ids)))
        parents = list(zip(parent_ids, roles))
        graph.G.add_node(
            node_id,
            type=node_type,
            op=operation,
            label=str(value),
            expr_obj=value,
            attributes=attributes,
            constant=attributes.get("value"),
            tensor={},
            parents=parents,
            children=[],
        )
        graph.node_map[node_id] = value
        for index, parent_id in enumerate(parent_ids):
            graph.G.add_edge(parent_id, node_id)
            graph.G.nodes[parent_id].setdefault("children", []).append(
                (node_id, f"arg:{index}")
            )
        return node_id

    root = add_node(expression)
    graph.roots.append(root)
    graph.G.graph["function_outputs"] = ("result",)
    graph.G.graph["symbolic_source"] = "sympy"
    return root


def process_graph_to_sympy_expressions(
    graph: Any,
    output_ids: Iterable[int] | None = None,
) -> tuple[sympy.Basic, ...]:
    """Render canonical ProcessGraph outputs as SymPy expressions."""

    if output_ids is None:
        deployment_outputs = tuple(
            int(value)
            for value in graph.G.graph.get("deployment_outputs", ())
            if value in graph.G
        )
        identity_table = graph.G.graph.get("identity_table") or {}
        output_names = graph.G.graph.get("function_outputs") or ()
        selected = [
            int(identity_table[name][-1])
            for name in output_names
            if name in identity_table and identity_table[name]
        ]
        output_ids = deployment_outputs or selected or tuple(graph.roots)

    cache: dict[int, sympy.Basic] = {}
    identity_names = {
        int(value_id): str(name)
        for name, value_ids in (
            graph.G.graph.get("identity_table") or {}
        ).items()
        for value_id in value_ids
    }
    indexed_bases = {
        int(parent_id)
        for _node_id, data in graph.G.nodes(data=True)
        if str(data.get("op") or data.get("type")) == "Indexed"
        for parent_id, role in data.get("parents") or ()
        if str(role) == "base"
    }

    def emit(node_id: int) -> sympy.Basic:
        if node_id in cache:
            return cache[node_id]
        data = graph.G.nodes[node_id]
        operation = str(data.get("op") or data.get("type"))
        attributes = data.get("attributes") or {}
        parents_by_role: dict[str, list[tuple[int, sympy.Basic]]] = defaultdict(list)
        for parent_id, role in data.get("parents") or ():
            parents_by_role[str(role)].append(
                (int(parent_id), emit(int(parent_id)))
            )

        if operation in {"input", "Input", "Symbol"}:
            name = str(
                attributes.get("binding_name")
                or identity_names.get(node_id)
                or data.get("label")
                or f"value_{node_id}"
            )
            result = (
                sympy.IndexedBase(name)
                if node_id in indexed_bases
                else sympy.Symbol(name)
            )
        elif operation in {
            "const", "Constant", "Integer", "Float", "Rational"
        }:
            value = attributes.get("value", data.get("constant"))
            result = sympy.sympify(value)
        else:
            ordered = []
            for role in ("lhs", "rhs", "operand"):
                ordered.extend(value for _node, value in parents_by_role.pop(role, ()))
            ordered.extend(
                value
                for role in sorted(parents_by_role)
                for _node, value in parents_by_role[role]
            )
            if operation in _BINARY and len(ordered) >= 2:
                result = ordered[0]
                for value in ordered[1:]:
                    result = _BINARY[operation](result, value)
            elif operation in _CANONICAL_FUNCTIONS:
                result = _CANONICAL_FUNCTIONS[operation](*ordered)
            elif operation == "Indexed":
                base = parents_by_role.get("base", ())
                indices = parents_by_role.get("index", ())
                if len(base) != 1 or not indices:
                    raise ValueError(
                        f"Indexed node {node_id} lacks base/index roles"
                    )
                result = sympy.Indexed(
                    base[0][1],
                    *(value for _node, value in indices),
                )
            elif operation in {"Neg", "neg"} and len(ordered) == 1:
                result = -ordered[0]
            else:
                result = sympy.Function(operation)(*ordered)
        cache[node_id] = result
        return result

    return tuple(emit(int(node_id)) for node_id in output_ids)


def symbolically_reduce_process_graph(graph: Any):
    """Round-trip one planner-filtered graph through SymPy simplification."""

    from ..transmogrifier.graph.graph_express2 import ProcessGraph

    original = process_graph_to_sympy_expressions(graph)
    reduced = tuple(sympy.simplify(expression) for expression in original)
    if len(reduced) != 1:
        raise NotImplementedError(
            "symbolic ProcessGraph reconstruction currently requires one "
            f"deployment output, got {len(reduced)}"
        )
    rebuilt = ProcessGraph(materialize_memory=False)
    ingest_sympy_expression(rebuilt, reduced[0])

    source_by_name = {}
    identity_table = graph.G.graph.get("identity_table") or {}
    for name, value_ids in identity_table.items():
        for value_id in value_ids:
            if value_id in graph.G:
                source_by_name[str(name)] = graph.G.nodes[value_id]
    for _node_id, data in rebuilt.G.nodes(data=True):
        if data.get("type") != "Input":
            continue
        name = str((data.get("attributes") or {}).get("binding_name"))
        source = source_by_name.get(name)
        if source is not None:
            data["tensor"] = dict(source.get("tensor") or {})

    output_id = rebuilt.roots[0]
    store_id = max(int(node_id) for node_id in rebuilt.G) + 1
    rebuilt.G.add_node(
        store_id,
        type="Store",
        op="store",
        label="symbolic_result",
        expr_obj=None,
        attributes={"symbolically_reduced": True},
        constant=None,
        tensor=dict(rebuilt.G.nodes[output_id].get("tensor") or {}),
        parents=[(output_id, "value")],
        children=[],
    )
    rebuilt.G.add_edge(output_id, store_id)
    rebuilt.G.nodes[output_id]["children"].append((store_id, "value"))
    rebuilt.roots = [store_id]
    rebuilt.G.graph.update(
        source_kind="sympy_reduction",
        deployment_inputs=tuple(
            node_id
            for node_id, data in rebuilt.G.nodes(data=True)
            if data.get("type") == "Input"
        ),
        deployment_outputs=(output_id,),
        symbolic_original=tuple(map(str, original)),
        symbolic_reduced=tuple(map(str, reduced)),
    )
    return rebuilt, SymbolicReductionReport(
        source_nodes=graph.G.number_of_nodes(),
        rebuilt_nodes=rebuilt.G.number_of_nodes(),
        original=original,
        reduced=reduced,
    )


def process_graph_to_sympy_package(graph: Any):
    """Compatibility package matching the historical ``to_sympy`` API."""

    from ..transmogrifier.graph.graph_express2 import ExpressionTensor

    expressions = process_graph_to_sympy_expressions(graph)
    registry = list(expressions)
    try:
        import torch
    except Exception:
        torch = None
    if torch is None:
        import numpy as np

        data = np.arange(len(expressions), dtype=int).reshape(1, 1, -1)
    else:
        data = torch.arange(
            len(expressions), dtype=torch.long
        ).reshape(1, 1, -1)
    return registry, ExpressionTensor(
        data,
        contexts=[0],
        sequence_length=1,
        domain_shape=(len(expressions),),
        function_index=None,
    )


__all__ = [
    "ingest_sympy_expression",
    "process_graph_to_sympy_expressions",
    "process_graph_to_sympy_package",
    "symbolically_reduce_process_graph",
    "SymbolicReductionReport",
]
