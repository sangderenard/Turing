"""One-function symbolic ProcessGraph spring image program."""

from __future__ import annotations

import math

import sympy

from ..compiler.symbolic_process_graph import symbolically_reduce_process_graph
from ..transmogrifier.graph.graph_express2 import ProcessGraph
from .opengl_render.fluxspring_shader import load_fluxspring_graph_shaders
from .precompiled_graph import run_precompiled_graph


def run_symbolic_spring_image(expression_text: str) -> None:
    """Solve, graph, spring-simulate, shade, and present one expression.

    This function is deliberately the whole program and the AST compiler's
    sole application entrypoint. Backend lowering must begin from its source;
    callers must not manufacture a numeric backend program from a selected
    subgraph.
    """

    expression = sympy.sympify(expression_text, evaluate=False)
    source_graph = ProcessGraph(materialize_memory=False)
    source_graph.build_from_expression(expression)
    process_graph, _reduction = symbolically_reduce_process_graph(
        source_graph,
        aggressive=True,
    )
    shader_sources = load_fluxspring_graph_shaders()
    run_precompiled_graph(
        process_graph.graph_accessor(),
        duration=math.inf,
        shader_sources=shader_sources,
    )


__all__ = ["run_symbolic_spring_image"]
