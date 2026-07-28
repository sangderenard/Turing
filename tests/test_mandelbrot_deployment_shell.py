from __future__ import annotations

import ast
import contextlib
import io

import pytest

from src.common.tensors.accelerator_backends.demo_mandelbrot_fusion import (
    build_parametric_mandelbrot_glsl_deployment,
    compile_parametric_mandelbrot_glsl,
)


@pytest.mark.parametrize(
    "builder",
    (
        build_parametric_mandelbrot_glsl_deployment,
        compile_parametric_mandelbrot_glsl,
    ),
)
def test_mandelbrot_builders_return_uninstalled_deployment_shell(builder):
    deployment, graph = builder(4)

    entry = graph.function_table.entry("mandelbrot_recording_program")
    assert deployment.process_graph is not entry.graph
    assert set(entry.graph.G) <= set(deployment.process_graph.G)
    assert all(
        deployment.process_graph.G.nodes[node_id]["type"] == "Store"
        for node_id in set(deployment.process_graph.G) - set(entry.graph.G)
    )
    assert deployment.module_shell.process_graph is graph
    assert deployment.source_node_count == deployment.process_graph.G.number_of_nodes()
    assert deployment.dispatch_count == len(deployment.dispatch_subgraphs)
    assert deployment.dispatch_count > 1
    assert len(graph.function_table) > 50
    assert tuple(
        entry.qualified_name for entry in graph.external_function_table
    ) == ("..compression.containers.avi.MJPEGAVIWriter",)
    for helper_name in (
        "parametric_mandelbrot_escape",
        "mandelbrot_jpeg_planes",
        "block_view_2d",
        "dct_2d_blocks",
        "collect_component_block_coefficient_events",
        "encode_baseline_color_component_scan",
        "finalize_entropy_scan",
        "_color_header",
        "prepare_jpeg_encoding_resources",
        "_avi_headers",
    ):
        assert graph.function_table.reference(helper_name) is not None
    assert deployment.entry_reference == entry.reference
    assert (
        deployment.module_shell.function_shells[entry.reference.address]
        is deployment
    )
    shared_error_buffer = deployment.module_shell.error_buffer
    pending_shells = [deployment.module_shell]
    visited_shells = set()
    while pending_shells:
        shell = pending_shells.pop()
        if id(shell) in visited_shells:
            continue
        visited_shells.add(id(shell))
        assert shell.error_buffer is shared_error_buffer
        pending_shells.extend(shell.function_shells.values())
    try:
        raise ValueError("shared shell error")
    except ValueError as error:
        deployment._profiler.record_exception(
            error,
            path=deployment.profile_path,
            phase="test",
        )
    errors = deployment.module_shell.exception_report()
    assert len(errors) == 1
    assert errors[0]["exception_type"] == "ValueError"
    assert "shared shell error" in errors[0]["traceback"]
    shared_error_buffer.clear()
    assert all(entry.graph is not None for entry in graph.function_table)
    assert not deployment.ready
    with pytest.raises(RuntimeError, match="planned but not installed"):
        deployment.require_ready()

    function_graph = entry.graph
    static_contexts = function_graph.G.graph["static_contexts"]
    assert len(static_contexts) == 1
    assert static_contexts[0]["reference"] == "autograd.no_grad"
    assert static_contexts[0]["effect"] == "disable_backward_recording"
    assert not any(
        data.get("type") in {"no_grad", "With", "withitem"}
        for _node_id, data in function_graph.G.nodes(data=True)
    )
    semantic_only = (
        ast.Attribute,
        ast.BoolOp,
        ast.If,
        ast.Slice,
        ast.Starred,
        ast.With,
        ast.withitem,
    )
    assert all(
        not isinstance(data.get("expr_obj"), semantic_only)
        for subgraph in deployment.dispatch_subgraphs
        for _node_id, data in subgraph.G.nodes(data=True)
        if data.get("type") != "Input"
    )
    for compiler in deployment.deep_compilers:
        with contextlib.redirect_stdout(io.StringIO()):
            compiler.build_function(device="glsl")
