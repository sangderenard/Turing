"""Dump the authored graph around a comprehension: nodes, ops, parents.

    python tools/dump_comprehension_graph.py <case>

Cases come from tools/repro_keyed_get.py.
"""

from __future__ import annotations

import ast
import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.tensors.topological_reducer import (  # noqa: E402
    reduce_abstract_tensor_topology,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph  # noqa: E402

from repro_keyed_get import CASES  # noqa: E402


def main() -> int:
    case = sys.argv[1]
    reduce_first = "--raw" not in sys.argv
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(
            ast.parse(CASES[case]), resolve_unresolved_parents=True,
        )
        if reduce_first:
            reduce_abstract_tensor_topology(graph)
    root = graph.function_table.entry("root").graph
    for node_id, data in sorted(root.G.nodes(data=True)):
        expression = data.get("expr_obj")
        attributes = data.get("attributes") or {}
        interesting = {
            key: attributes[key]
            for key in (
                "tensor", "tensor_candidate", "producer_kind",
                "static_python_reference", "loop_result_id",
                "comprehension_result", "source_conditional_id",
            )
            if key in attributes
        }
        print(
            f"{node_id:>6}  type={data.get('type')!r:<24} "
            f"op={data.get('op')!r:<14} "
            f"ast={type(expression).__name__ if expression is not None else '-':<14} "
            f"label={str(data.get('label'))[:28]!r:<30} "
            f"parents={[(p, r) for p, r in (data.get('parents') or ())]}"
            + (f" {interesting}" if interesting else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
