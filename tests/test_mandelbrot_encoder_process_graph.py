import ast

import networkx as nx
import numpy as np
import pytest

from src.common.tensors.accelerator_backends.mandelbrot_encoder_program import (
    build_mandelbrot_encoder_process_graph,
    build_mandelbrot_recording_process_graph,
    mandelbrot_encoder_source_files,
    mandelbrot_jpeg_master,
)
from src.common.tensors.accelerator_backends.demo_mandelbrot_fusion import (
    normalized_plane,
)
from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.compression.jpeg.frame import (
    prepare_jpeg_encoding_resources,
)
from src.common.tensors.numpy_backend import NumPyTensorOperations
from src.transmogrifier.operator_defs import role_schemas


ENCODER_CHAIN = (
    "parametric_mandelbrot_escape",
    "mandelbrot_jpeg_planes",
    "encode_ycbcr_jfif",
    "iter_ycbcr_jfif_chunks",
    "jpeg_ycbcr_coefficients",
    "collect_component_block_coefficient_events",
    "encode_baseline_color_component_scan",
    "compact_codewords",
    "_stuff_entropy_octets",
    "tensor_octets_to_bytes",
)


@pytest.fixture(scope="module")
def encoder_graph():
    return build_mandelbrot_encoder_process_graph()


@pytest.fixture(scope="module")
def recording_graph():
    return build_mandelbrot_recording_process_graph()


def test_every_ast_object_in_the_original_source_bundle_has_a_role_schema():
    encountered = {
        type(node).__name__
        for path in mandelbrot_encoder_source_files()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
    }
    assert encountered <= set(role_schemas)


def test_whole_mandelbrot_encoder_ingests_as_one_semantic_process_graph(
    encoder_graph,
):
    graph = encoder_graph
    assert nx.is_directed_acyclic_graph(graph.G)
    assert len(graph.roots) == 1
    assert graph.G.nodes[graph.roots[0]]["label"] == "mandelbrot_jpeg_master"
    assert not any(
        data["op"] == "opaque_python"
        for _, data in graph.G.nodes(data=True)
    )

    reachable = nx.descendants(graph.G, graph.roots[0]) | set(graph.roots)
    for function_name in ENCODER_CHAIN:
        definitions = {
            node_id
            for node_id, data in graph.G.nodes(data=True)
            if data["op"] == "function_def"
            and data["label"] == function_name
        }
        assert definitions
        assert definitions & reachable

    operations = {
        data["op"] for _, data in graph.G.nodes(data=True)
    }
    assert {
        "matmul",
        "scatter",
        "cumsum",
        "to_dtype",
        "index",
        "slice_spec",
        "for",
        "with",
        "yield",
    } <= operations


def test_ast_math_is_routed_to_existing_reduction_systems(encoder_graph):
    graph = encoder_graph
    add = next(
        data for _, data in graph.G.nodes(data=True) if data["op"] == "add"
    )
    bitand = next(
        data for _, data in graph.G.nodes(data=True) if data["op"] == "bitand"
    )
    assert add["attributes"]["canonical_operation"] == "add"
    assert add["attributes"]["bitops_capable"] is True
    assert add["attributes"]["bitops_candidate"] is False
    assert bitand["attributes"]["bitops_capable"] is True


def test_compiler_profile_filters_python_noise_and_types_tensor_nodes():
    complete = build_mandelbrot_encoder_process_graph(profile="complete")
    compiler_graph = build_mandelbrot_encoder_process_graph()

    assert compiler_graph.G.graph["semantic_profile"] == "tensor_control"
    assert compiler_graph.G.graph["complete_node_count"] == len(complete.G)
    assert len(compiler_graph.G) < len(complete.G) // 4
    assert not any(
        data["op"] == "import"
        for _, data in compiler_graph.G.nodes(data=True)
    )

    matmul = next(
        data
        for _, data in compiler_graph.G.nodes(data=True)
        if data["op"] == "matmul"
    )
    assert matmul["attributes"]["semantic_kind"] == "tensor_operation"
    assert matmul["attributes"]["execution_domain"] == "abstract_tensor"

    kinds = {
        data["attributes"]["semantic_kind"]
        for _, data in compiler_graph.G.nodes(data=True)
    }
    assert {"tensor_operation", "control", "host_boundary"} <= kinds


def test_master_function_executes_the_original_numpy_tensor_encoder():
    unit_x, unit_y = normalized_plane(8, 8)
    scalar = lambda value: NumPyTensorOperations.tensor(
        np.asarray([value], dtype=np.float32)
    )
    with AbstractTensor.use_backend("numpy"):
        x = NumPyTensorOperations.tensor(unit_x)
        y = NumPyTensorOperations.tensor(unit_y)
        resources = prepare_jpeg_encoding_resources(x)
        counts, encoded = mandelbrot_jpeg_master(
            x,
            y,
            scalar(-0.72),
            scalar(0.1),
            scalar(2.4),
            scalar(0.0),
            scalar(-0.72),
            scalar(0.24),
            scalar(0.0),
            scalar(0.52),
            width=8,
            height=8,
            iterations=4,
            resources=resources,
        )
    assert counts.shape == (64,)
    assert encoded.startswith(b"\xFF\xD8")
    assert encoded.endswith(b"\xFF\xD9")


def test_recording_program_is_one_forward_reachable_start_to_finish_graph(
    recording_graph,
):
    graph = recording_graph
    root = graph.roots[0]
    reachable = nx.descendants(graph.G, root) | {root}

    assert graph.G.graph["semantic_profile"] == "program"
    assert graph.G.graph["entrypoint_expanded"] is True
    assert graph.G.graph["program_entrypoint"] == "animate_glsl"
    assert reachable == set(graph.G)
    assert nx.is_directed_acyclic_graph(graph.G)

    required_functions = {
        "animate_glsl",
        "mandelbrot_display_master",
        "parametric_mandelbrot_escape",
        "mandelbrot_jpeg_planes",
        "tensor_ycbcr_jpeg_bytes",
        "encode_ycbcr_jfif",
        "iter_ycbcr_jfif_chunks",
        "jpeg_ycbcr_coefficients",
        "collect_component_block_coefficient_events",
        "encode_baseline_color_component_scan",
        "compact_codewords",
        "tensor_octets_to_bytes",
        "append_frame",
        "append_audio_tensor",
        "append_audio",
        "_finish_segment",
        "_patch_superindex",
        "_patch_u32",
        "close",
    }
    present = {
        data["label"]
        for _, data in graph.G.nodes(data=True)
        if data["op"] == "function_def"
    }
    assert required_functions <= present

    operations = {data["op"] for _, data in graph.G.nodes(data=True)}
    assert {
        "while",
        "for",
        "matmul",
        "scatter",
        "cumsum",
        "yield",
        "with",
        "try",
        "call",
    } <= operations


def test_recording_program_links_compiled_source_and_writer_methods(
    recording_graph,
):
    graph = recording_graph
    compiled_links = [
        (node_id, (data.get("attributes") or {}).get("compiled_entrypoint"))
        for node_id, data in graph.G.nodes(data=True)
        if (data.get("attributes") or {}).get("compiled_entrypoint") is not None
    ]
    assert len(compiled_links) == 1
    compiler_call, source_root = compiled_links[0]
    assert graph.G.has_edge(compiler_call, source_root)
    assert graph.G.edges[compiler_call, source_root]["role"] == "compiles"
    assert graph.G.nodes[source_root]["label"] == "mandelbrot_display_master"

    for spelling, target_name in (
        ("recorder.append_frame", "append_frame"),
        ("recorder.append_audio_tensor", "append_audio_tensor"),
        ("recorder.close", "close"),
    ):
        call = next(
            data
            for _, data in graph.G.nodes(data=True)
            if data["op"] == "call"
            and (data.get("attributes") or {}).get("function") == spelling
        )
        target = call["attributes"]["callee"]
        assert call["attributes"]["resolved"] is True
        assert graph.G.nodes[target]["label"] == target_name

    retained_definitions = {
        data["label"]
        for _, data in graph.G.nodes(data=True)
        if data["op"] in {"class_def", "function_def"}
    }
    unresolved_direct_project_calls = [
        (data.get("attributes") or {}).get("function")
        for _, data in graph.G.nodes(data=True)
        if data["op"] == "call"
        and not (data.get("attributes") or {}).get("resolved")
        and "." not in str((data.get("attributes") or {}).get("function"))
        and (data.get("attributes") or {}).get("function")
        in retained_definitions
    ]
    assert unresolved_direct_project_calls == []

    root = graph.roots[0]
    environment_targets = [
        target
        for source, target, edge in graph.G.edges(data=True)
        if source == root and edge.get("role") == "environment"
    ]
    assert environment_targets
    assert {
        (graph.G.nodes[target].get("attributes") or {}).get("scope")
        for target in environment_targets
    } == {"<module>"}
