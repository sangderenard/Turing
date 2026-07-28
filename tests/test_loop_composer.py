from __future__ import annotations

import ast
import contextlib
import io

from src.common.tensors.accelerator_backends.glsl_backend import (
    emit_native_for_loop,
)
from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
from src.compiler.loop_composer import (
    LoopBackendCapabilities,
    LoopComposer,
    LoopStrategy,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def _function_graph(source: str, name: str):
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(graph)
    return graph.function_table.entry(name).graph


def _glsl_composer(*, unroll_limit: int = 8) -> LoopComposer:
    return LoopComposer(
        LoopBackendCapabilities(
            backend="glsl",
            native_for=True,
            native_while=True,
            dynamic_bounds=True,
            unroll_limit=unroll_limit,
        )
    )


def test_loop_composer_unrolls_small_static_range():
    graph = _function_graph(
        "def kernel(x):\n"
        "    for index in range(4):\n"
        "        x = x + index\n"
        "    return x\n",
        "kernel",
    )

    plan, = _glsl_composer().compose(graph)

    assert plan.strategy is LoopStrategy.UNROLL
    assert plan.loop.target == "index"
    assert plan.loop.trip_count == 4
    assert plan.loop.body_nodes


def test_loop_composer_keeps_larger_range_in_glsl_source():
    graph = _function_graph(
        "def kernel(x):\n"
        "    for index in range(64):\n"
        "        x = x + index\n"
        "    return x\n",
        "kernel",
    )

    plan, = _glsl_composer().compose(graph)

    assert plan.strategy is LoopStrategy.NATIVE_SOURCE
    assert plan.loop.trip_count == 64


def test_glsl_native_loop_wraps_an_already_lowered_region():
    source = emit_native_for_loop(
        ("float next = state + delta;", "state = next;"),
        induction="iteration",
        start=0,
        stop=64,
        step=1,
    )

    assert source == (
        "    for (int iteration = int(0); iteration < int(64); "
        "iteration += int(1)) {",
        "        float next = state + delta;",
        "        state = next;",
        "    }",
    )
