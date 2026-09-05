"""The OOP interchange layer's two crossings, proven on real library source.

1. A class captured at ingestion becomes a transportable SCHEMA V1 document
   (the C++ ``GenericObject`` shell consumes the same document -- see
   nodus/tests/generic_object_test.cpp document mode).
2. A non-fundamental composite method shallow-interprets into nodus GraphIR
   whose operations are exclusively canonical vocabulary (the
   ``tensor_graph_execute`` harness executes the same script -- see
   nodus/tests/tensor_graph_execute_test.cpp).

Both inputs are ordinary abstract_nn sources, untouched.
"""

import ast
from pathlib import Path

import pytest

TENSOR_ROOT = Path(__file__).resolve().parents[1] / "src" / "common" / "tensors"


@pytest.fixture(scope="module")
def process_graph_module():
    from src.transmogrifier.graph import graph_express2
    return graph_express2


def test_linear_capture_derives_schema_v1(process_graph_module):
    from src.compiler.oop_schema import (
        class_schemas_from_process_graph,
        parse_schema_text,
        serialize_schema_text,
    )

    source = (TENSOR_ROOT / "abstract_nn" / "core.py").read_text(encoding="utf-8")
    graph = process_graph_module.ProcessGraph(0, False, materialize_memory=False)
    graph.build_from_ast(ast.parse(source), filename="core.py",
                         resolve_unresolved_parents=False, semantic=False)

    schemas = {s.identity: s for s in class_schemas_from_process_graph(graph)}
    assert "Linear" in schemas
    linear = schemas["Linear"]
    field_names = {f.name for f in linear.fields}
    assert {"W", "b"} <= field_names
    method_names = {m.name for m in linear.methods}
    assert {"__init__", "forward"} <= method_names

    # Arity crosses: forward(self, x) is one destination-visible argument.
    forward = linear.method_named("forward")
    assert forward is not None
    assert [p.name for p in forward.parameters] == ["x"]

    text = serialize_schema_text(linear)
    roundtrip = parse_schema_text(text)
    assert roundtrip.identity == "Linear"
    assert len(roundtrip.fields) == len(linear.fields)
    assert len(roundtrip.methods) == len(linear.methods)
    assert len(roundtrip.method_named("forward").parameters) == 1
    # The wire form must stay whitespace-clean per record.
    assert all(len(line.split()) >= 2 for line in text.strip().splitlines())


def test_relu6_forward_shallow_interprets_to_canonical_ops(process_graph_module):
    from src.compiler.shallow_interpretation import method_to_graph_ir
    from src.common.tensors.operator_catalog import (
        CANONICAL_ABSTRACT_TENSOR_OPERATORS,
    )

    source = (TENSOR_ROOT / "abstract_nn" / "activations.py").read_text(encoding="utf-8")
    result = method_to_graph_ir(source, "ReLU6.forward", filename="activations.py")

    # The calling convention survives: x is the only input (self sliced away
    # with the rest of the non-dataflow machinery).
    assert result.inputs == ("x",)

    # Every operation in the slice is canonical vocabulary -- the boundary
    # held, and nothing opaque leaked into the numeric path.
    assert result.operations
    for operation in result.operations:
        assert operation in CANONICAL_ABSTRACT_TENSOR_OPERATORS, operation
    assert "opaque_python" not in result.operations
    assert "call" not in result.operations

    # The wire text is real GraphIR the nodus side parses.
    assert "tensor_node(" in result.graph_ir
    assert "connect(" in result.graph_ir


def test_gelu_forward_pursues_helpers_and_class_constants(process_graph_module):
    """A composite whose body calls a module helper and reads class constants
    still decomposes to canonical leaves: `_tanh_stable` inlines to `tanh`,
    `self._K`/`self._C` become constants, nothing opaque survives."""

    from src.compiler.shallow_interpretation import method_to_graph_ir

    source = (TENSOR_ROOT / "abstract_nn" / "activations.py").read_text(encoding="utf-8")
    result = method_to_graph_ir(source, "GELU.forward", filename="activations.py")

    assert result.inputs == ("x",)
    assert set(result.operations) == {"mul", "add", "tanh"}
    assert "call" not in result.operations
    assert "opaque_python" not in result.operations


def test_mse_loss_forward_reaches_reduction_leaves(process_graph_module):
    """A loss decomposes to sub/mul/mean with both tensors as inputs; the
    hook machinery carries no dataflow and is sliced away."""

    from src.compiler.shallow_interpretation import method_to_graph_ir

    source = (TENSOR_ROOT / "abstract_nn" / "losses.py").read_text(encoding="utf-8")
    result = method_to_graph_ir(source, "MSELoss.forward", filename="losses.py")

    assert result.inputs == ("pred", "target")
    assert set(result.operations) == {"sub", "mul", "mean"}
