"""Trace shape-fact mutations across ProcessGraph structural folding.

This is a read-only compiler diagnostic.  It wraps the structural fold in one
Python process and prints only nodes matching requested function/name/id
filters; it never rewrites a graph beyond the compiler pass being observed.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _shape_map(values):
    return {
        str(name): tuple((descriptor or {}).get("shape") or ())
        for name, descriptor in (values or {}).items()
    }


def _matching_nodes(graph, node_ids: set[int], expressions: tuple[str, ...]):
    for node_id, data in graph.G.nodes(data=True):
        expression = data.get("expr_obj")
        try:
            source = ast.unparse(expression) if expression is not None else ""
        except (TypeError, ValueError):
            source = ""
        if int(node_id) not in node_ids and not any(
            token in source for token in expressions
        ):
            continue
        yield int(node_id), data, source


def _dump(label, graph, node_ids: set[int], expressions: tuple[str, ...]):
    metadata = graph.G.graph
    print(
        "TRACE",
        label,
        metadata.get("function_name"),
        "planner=", _shape_map(metadata.get("planner_tensor_descriptors")),
        "callsite=", metadata.get("callsite_tensor_descriptor_names"),
        "abi=", _shape_map(metadata.get("parameter_value_abi")),
        flush=True,
    )
    for node_id, data, source in _matching_nodes(graph, node_ids, expressions):
        print(
            "NODE",
            node_id,
            data.get("type"),
            data.get("op"),
            "tensor=", data.get("tensor"),
            "parents=", data.get("parents"),
            "attributes=", data.get("attributes"),
            "source=", source,
            flush=True,
        )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", default="balloon_tire_vector_step")
    parser.add_argument("--node", type=int, action="append", default=[])
    parser.add_argument(
        "--expression",
        action="append",
        default=["selection_minimum", "selection_distance.min"],
    )
    arguments = parser.parse_args(argv)

    import src.compiler.glsl_deployment_strategy as strategy

    original = strategy._fold_callsite_structural_values
    calls = 0

    def traced(graph):
        nonlocal calls
        function_name = str(graph.G.graph.get("function_name") or "")
        if arguments.function not in function_name:
            return original(graph)
        calls += 1
        node_ids = set(arguments.node)
        expressions = tuple(arguments.expression)
        _dump(f"{calls}:before", graph, node_ids, expressions)
        result = original(graph)
        _dump(f"{calls}:after", graph, node_ids, expressions)
        return result

    strategy._fold_callsite_structural_values = traced
    from src.compiler.vehicle_python_compilation import (
        lower_balloon_tire_managed_python_ssa,
    )

    try:
        lower_balloon_tire_managed_python_ssa()
    except Exception as error:
        print("FAIL", type(error).__name__, str(error), flush=True)
        return 1
    print("COMPILED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
