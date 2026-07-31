from __future__ import annotations

import ast
import contextlib
import io

import numpy as np
import sympy

from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
from src.compiler.symbolic_process_graph import (
    process_graph_to_sympy_expressions,
    symbolically_reduce_process_graph,
)
from src.transmogrifier.graph.graph_express2 import ProcessGraph


def test_sympy_ingestion_round_trips_semantically():
    x, y = sympy.symbols("x y")
    expression = (x + 2 * y) ** 2 + sympy.sin(x)
    graph = ProcessGraph(materialize_memory=False)

    graph.build_from_expression(expression)
    rebuilt, = process_graph_to_sympy_expressions(graph)

    assert graph.G.number_of_nodes() > 1
    assert sympy.simplify(rebuilt - expression) == 0


def test_reduced_ast_function_projects_through_canonical_schema():
    source = (
        "def f(x, y):\n"
        "    z = x + y * 2\n"
        "    return z\n"
    )
    module = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        module.build_from_ast(ast.parse(source))
    reduce_abstract_tensor_topology(module)
    graph = module.function_table.entry("f").graph

    rebuilt, = process_graph_to_sympy_expressions(graph)
    x, y = sympy.symbols("x y")

    assert sympy.simplify(rebuilt - (x + y * 2)) == 0


def test_to_sympy_compatibility_package_uses_current_expression_tensor_api():
    x = sympy.Symbol("x")
    graph = ProcessGraph(materialize_memory=False)
    graph.build_from_expression(x + 1)

    registry, tensor = graph.to_sympy()

    assert registry == [x + 1]
    assert tuple(tensor.domain_shape) == (1,)


def test_first_filtered_mandelbrot_avi_region_round_trips_into_compiler():
    from src.common.tensors.accelerator_backends.demo_mandelbrot_fusion import (
        build_parametric_mandelbrot_glsl_deployment,
    )
    from src.transmogrifier.graph.graph_deep_compiler import (
        GraphDeepCompiler,
    )
    from src.transmogrifier.operator_defs import (
        abstract_tensor_funcs,
        abstract_tensor_sigs,
    )
    from src.common.tensors.abstraction import AbstractTensor

    deployment, _module = build_parametric_mandelbrot_glsl_deployment(8)
    filtered = deployment.dispatch_subgraphs[0]
    rebuilt, report = symbolically_reduce_process_graph(filtered)
    rebuilt.compute_levels("asap")
    compiler = GraphDeepCompiler(
        rebuilt,
        dict(abstract_tensor_funcs),
        abstract_tensor_sigs,
    )

    compiled = compiler.build_function()
    with AbstractTensor.use_backend("numpy"):
        frames = AbstractTensor.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        )
        result = compiled(
            floatframes=frames,
            floatframe_index=1,
        )[0]

    assert callable(compiled)
    assert filtered.G.graph["deployment_nodes"] == (28,)
    assert report.original == report.reduced
    assert any(
        data.get("type") == "Indexed"
        for _node_id, data in rebuilt.G.nodes(data=True)
    )
    assert "op_" in compiler._code
    np.testing.assert_array_equal(
        result.numpy(),
        np.arange(12, 24, dtype=np.float32).reshape(3, 4),
    )
