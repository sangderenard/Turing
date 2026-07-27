import ast

import networkx as nx
import numpy as np
import pytest

from src.common.tensors.accelerator_backends.mandelbrot_encoder_program import (
    build_mandelbrot_encoder_process_graph,
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
    assert add["attributes"]["bitops_candidate"] is True
    assert bitand["attributes"]["bitops_candidate"] is True


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
