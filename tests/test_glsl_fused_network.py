from __future__ import annotations

import ast
import contextlib
import io

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.glsl_fused_network import (
    GLSLFusedProgramNetwork,
)
from src.common.tensors.accelerator_backends.glsl_backend import (
    GLContextUnavailable,
    compile_captured_fused_program,
    execute_captured_fused_program,
    require_gl_context,
)
from src.common.tensors.accelerator_backends.c_primitive_program import (
    CapturedFusedProgram,
    compile_recorded_fused_tape,
)
from src.common.tensors.accelerator_backends.glsl_tensor_backend import (
    GLSLTensorOperations,
)
from src.common.tensors.autograd import autograd
from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
from src.common.tensors.topological_reducer import (
    reduce_abstract_tensor_topology,
)
from src.compiler.glsl_deployment_strategy import strategize_glsl_deployment
from src.transmogrifier.graph.graph_express2 import ProcessGraph


@pytest.fixture(scope="session")
def gl():
    try:
        return require_gl_context()
    except GLContextUnavailable as exc:
        pytest.skip(f"no OpenGL 4.3+ compute context: {exc}")


def _program(feeds, steps, outputs, ids):
    return FusedProgram(
        version=1,
        feeds=set(feeds),
        steps=[
            OpStep(index, op, inputs, attrs, result_id)
            for index, (op, result_id, inputs, attrs) in enumerate(steps)
        ],
        outputs=outputs,
        meta={
            value_id: Meta(shape=(32,), dtype="float32", device="glsl")
            for value_id in ids
        },
    )


def _capture_glsl(operation):
    with autograd.forward_capture() as tape:
        result = operation()
    captured = compile_recorded_fused_tape(tape)
    compile_captured_fused_program(captured)
    replayed = execute_captured_fused_program(captured, {})["result_0"]
    return result, replayed, captured


@pytest.mark.parametrize(
    ("name", "operation"),
    [
        (
            "permute",
            lambda: GLSLTensorOperations.tensor(
                np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            ).permute(2, 0, 1),
        ),
        (
            "repeat",
            lambda: GLSLTensorOperations.tensor(
                np.arange(6, dtype=np.float32).reshape(2, 3)
            ).repeat(2, 1),
        ),
        (
            "matmul",
            lambda: GLSLTensorOperations.tensor(
                np.arange(6, dtype=np.float32).reshape(2, 3)
            )
            @ GLSLTensorOperations.tensor(
                np.arange(12, dtype=np.float32).reshape(3, 4)
            ),
        ),
        (
            "sum",
            lambda: GLSLTensorOperations.tensor(
                np.arange(12, dtype=np.float32).reshape(3, 4)
            ).sum(dim=1, keepdim=True),
        ),
        (
            "cumsum",
            lambda: GLSLTensorOperations.tensor(
                np.arange(12, dtype=np.float32).reshape(3, 4)
            ).cumsum(1),
        ),
        (
            "gather",
            lambda: GLSLTensorOperations.tensor(
                np.arange(12, dtype=np.float32).reshape(3, 4)
            ).gather(
                GLSLTensorOperations.tensor(
                    np.asarray([3, 1], dtype=np.int32)
                ),
                1,
            ),
        ),
        (
            "stack",
            lambda: GLSLTensorOperations.stack(
                [
                    GLSLTensorOperations.tensor(
                        np.arange(6, dtype=np.float32).reshape(2, 3)
                    ),
                    GLSLTensorOperations.tensor(
                        np.arange(6, 12, dtype=np.float32).reshape(2, 3)
                    ),
                ],
                dim=1,
            ),
        ),
        (
            "cat",
            lambda: GLSLTensorOperations.cat(
                [
                    GLSLTensorOperations.tensor(
                        np.arange(6, dtype=np.float32).reshape(2, 3)
                    ),
                    GLSLTensorOperations.tensor(
                        np.arange(6, 12, dtype=np.float32).reshape(2, 3)
                    ),
                ],
                dim=0,
            ),
        ),
    ],
)
def test_captured_native_glsl_kernels_replay_as_one_shader(gl, name, operation):
    expected, actual, captured = _capture_glsl(operation)

    assert captured.program.extras["kernel_kind"] in {
        "cat",
        "cumsum",
        "index_select",
        "matmul",
        "permute",
        "reduce",
        "repeat",
        "stack",
    }
    np.testing.assert_allclose(actual.numpy(), expected.numpy())


@pytest.mark.parametrize("operation", ["sum", "mean", "min", "max", "any", "all"])
def test_every_native_reduction_captures_and_replays(gl, operation):
    source = GLSLTensorOperations.tensor(
        np.arange(12, dtype=np.float32).reshape(3, 4)
    )
    expected, actual, captured = _capture_glsl(
        lambda: getattr(source, operation)(dim=1)
    )

    assert captured.program.extras["kernel_kind"] == "reduce"
    assert captured.program.steps[0].attrs["axis"] == 1
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (
            lambda: GLSLTensorOperations.zeros(
                (2, 3), cls=GLSLTensorOperations
            ),
            np.zeros((2, 3), dtype=np.float32),
        ),
        (
            lambda: GLSLTensorOperations.ones(
                (2, 3), cls=GLSLTensorOperations
            ),
            np.ones((2, 3), dtype=np.float32),
        ),
        (
            lambda: GLSLTensorOperations.full(
                (2, 3), 7, dtype=np.int32, cls=GLSLTensorOperations
            ),
            np.full((2, 3), 7, dtype=np.int32),
        ),
        (
            lambda: GLSLTensorOperations.arange(
                2, 11, 3, cls=GLSLTensorOperations
            ),
            np.arange(2, 11, 3, dtype=np.int32),
        ),
    ],
)
def test_captured_glsl_creation_is_device_native(gl, operation, expected):
    _, actual, captured = _capture_glsl(operation)

    assert captured.program.extras["kernel_kind"] in {"arange", "fill"}
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_vertical_fused_programs_route_through_resident_spsc_lanes(gl):
    first = _program(
        [0],
        [("mul", 1, [0], {"right_scalar": 2.0})],
        {"vertical": 1},
        [0, 1],
    )
    second = _program(
        [1],
        [("add", 2, [1], {"right_scalar": 3.0})],
        {"result": 2},
        [1, 2],
    )
    network = GLSLFusedProgramNetwork((first, second), fifo_slots=2)
    try:
        values = np.arange(32, dtype=np.float32)
        first_result = network.execute({0: values})["result"]
        np.testing.assert_allclose(first_result.numpy(), values * 2.0 + 3.0)

        second_result = network.execute({0: values + 1.0})["result"]
        np.testing.assert_allclose(
            second_result.numpy(),
            (values + 1.0) * 2.0 + 3.0,
        )
        assert all(lane.unread == 0 for lane in network.arena.lanes.values())
    finally:
        network.release()


def test_fanout_gets_one_spsc_lane_per_consumer(gl):
    producer = _program(
        [0],
        [("mul", 1, [0], {"right_scalar": 2.0})],
        {"shared": 1},
        [0, 1],
    )
    left = _program(
        [1],
        [("add", 2, [1], {"right_scalar": 1.0})],
        {"left": 2},
        [1, 2],
    )
    right = _program(
        [1],
        [("sub", 3, [1], {"right_scalar": 1.0})],
        {"right": 3},
        [1, 3],
    )
    network = GLSLFusedProgramNetwork((producer, left, right))
    try:
        shared_routes = [
            route for route in network.routes if route.value_id == 1
        ]
        assert len(shared_routes) == 2
        assert {route.consumer for route in shared_routes} == {1, 2}

        values = np.arange(32, dtype=np.float32)
        outputs = network.execute({0: values})
        np.testing.assert_allclose(outputs["left"].numpy(), values * 2.0 + 1.0)
        np.testing.assert_allclose(outputs["right"].numpy(), values * 2.0 - 1.0)
    finally:
        network.release()


def test_glsl_deployment_accepts_ephemeral_vertical_programs(gl):
    graph = ProcessGraph(materialize_memory=False)
    graph.G.add_node(
        0,
        label="input",
        type="input",
        op="input",
        parents=[],
        children=[],
    )
    deployment = strategize_glsl_deployment(graph)()
    program = _program(
        [0],
        [("mul", 1, [0], {"right_scalar": 4.0})],
        {"result": 1},
        [0, 1],
    )
    deployment.install_fused_programs((program,))
    try:
        values = np.arange(32, dtype=np.float32)
        result = deployment.execute({0: values})["result"]
        np.testing.assert_allclose(result.numpy(), values * 4.0)
    finally:
        deployment.release()


def test_glsl_deployment_shell_owns_named_and_fifo_execution(gl):
    graph = ProcessGraph(materialize_memory=False)
    graph.G.add_node(
        0,
        label="samples",
        type="input",
        op="input",
        parents=[],
        children=[],
    )
    deployment = strategize_glsl_deployment(graph)(
        input_slots=2,
        output_slots=2,
    )
    program = _program(
        [0],
        [("mul", 1, [0], {"right_scalar": 2.0})],
        {"network_result": 1},
        [0, 1],
    )

    assert not deployment.ready
    assert deployment.programs == ()
    with pytest.raises(RuntimeError, match="planned but not installed"):
        deployment.require_ready()

    captured = CapturedFusedProgram(program, {})
    deployment.install_fused_programs(
        (captured,),
        input_bindings={"samples": 0},
        output_bindings={"result": "network_result"},
    )
    try:
        assert deployment.ready
        assert deployment.programs == (program,)

        values = np.arange(32, dtype=np.float32)
        direct = deployment({"samples": values})
        np.testing.assert_allclose(direct["result"].numpy(), values * 2.0)

        deployment.submit({"samples": values + 1.0})
        assert deployment.run_pending()
        available, queued = deployment.receive()
        assert available
        np.testing.assert_allclose(
            queued["result"].numpy(),
            (values + 1.0) * 2.0,
        )
        assert not deployment.run_pending()
        assert deployment.receive() == (False, None)
    finally:
        deployment.release()

    assert not deployment.ready


def test_glsl_deployment_installs_compiled_tapes(gl):
    graph = ProcessGraph(materialize_memory=False)
    graph.G.add_node(
        0,
        label="samples",
        type="input",
        op="input",
        parents=[],
        children=[],
    )
    deployment = strategize_glsl_deployment(graph)()
    program = _program(
        [0],
        [("add", 1, [0], {"right_scalar": 5.0})],
        {"result": 1},
        [0, 1],
    )
    deployment.compiled_tapes = (CapturedFusedProgram(program, {}),)
    deployment.install_compiled_tapes(
        input_bindings={"samples": 0},
    )
    try:
        values = np.arange(32, dtype=np.float32)
        result = deployment.execute_named({"samples": values})["result"]
        np.testing.assert_allclose(result.numpy(), values + 5.0)
    finally:
        deployment.release()


def test_glsl_deployment_captures_compiles_and_routes_scheduled_regions(gl):
    graph = ProcessGraph(materialize_memory=False)
    graph.G.add_node(
        0,
        label="samples",
        type="Input",
        op="input",
        parents=[],
        children=[(2, "lhs")],
    )
    graph.G.add_node(
        1,
        label="2.0",
        type="Constant",
        op="const",
        constant=2.0,
        parents=[],
        children=[(2, "rhs")],
    )
    graph.G.add_node(
        2,
        label="mul",
        type="Mul",
        op="mul",
        parents=[(0, "lhs"), (1, "rhs")],
        children=[(4, "lhs")],
    )
    graph.G.add_node(
        3,
        label="3.0",
        type="Constant",
        op="const",
        constant=3.0,
        parents=[],
        children=[(4, "rhs")],
    )
    graph.G.add_node(
        4,
        label="add",
        type="Add",
        op="add",
        parents=[(2, "lhs"), (3, "rhs")],
        children=[(5, "value")],
    )
    graph.G.add_node(
        5,
        label="result",
        type="Store",
        op="store",
        parents=[(4, "value")],
        children=[],
    )
    graph.G.add_edges_from(
        ((0, 2), (1, 2), (2, 4), (3, 4), (4, 5))
    )
    graph.compute_levels(method="asap", order="dependency")

    deployment = strategize_glsl_deployment(
        graph,
        max_nodes_per_dispatch=1,
    )()
    values = np.arange(32, dtype=np.float32)
    deployment.capture_scheduled_forward_tapes({0: values})
    sources = deployment.compile_forward_tapes()
    deployment.install_compiled_tapes(
        input_bindings={"samples": 0},
        output_bindings={"result": "value_4"},
    )
    try:
        assert len(sources) == 2
        assert len(deployment.programs) == 2
        result = deployment.execute_named({"samples": values})["result"]
        np.testing.assert_allclose(result.numpy(), values * 2.0 + 3.0)
        assert all(
            lane.unread == 0
            for lane in deployment.fused_network.arena.lanes.values()
        )
    finally:
        deployment.release()


def test_glsl_deployment_coordinates_structural_result_around_numeric_region(gl):
    graph = ProcessGraph(materialize_memory=False)
    graph.G.add_node(
        0,
        label="samples",
        type="Input",
        op="input",
        parents=[],
        children=[(2, "lhs")],
        attributes={"binding_name": "samples"},
    )
    graph.G.add_node(
        1,
        label="2.0",
        type="Constant",
        op="const",
        constant=2.0,
        parents=[],
        children=[(2, "rhs")],
        attributes={"value": 2.0},
    )
    graph.G.add_node(
        2,
        label="mul",
        type="Mul",
        op="mul",
        parents=[(0, "lhs"), (1, "rhs")],
        children=[(3, "elts")],
        attributes={},
    )
    tuple_expression = ast.parse("(value,)").body[0].value
    graph.G.add_node(
        3,
        label="tuple_result",
        type="Tuple",
        op="tuple",
        expr_obj=tuple_expression,
        parents=[(2, "elts")],
        children=[],
        attributes={},
    )
    graph.G.add_edges_from(((0, 2), (1, 2), (2, 3)))
    graph.roots = [3]
    graph.compute_levels(method="asap", order="dependency")

    deployment = strategize_glsl_deployment(graph)()
    values = np.arange(16, dtype=np.float32)
    result = deployment.coordinate_first_invocation({"samples": values})
    assert isinstance(result, tuple)
    np.testing.assert_allclose(result[0].numpy(), values * 2.0)
    assert len(deployment.forward_tapes) == 1

    deployment.compile_process_graph()
    try:
        compiled = deployment.execute_named({"samples": values})["result_0"]
        np.testing.assert_allclose(compiled.numpy(), values * 2.0)
    finally:
        deployment.release()


def test_glsl_planner_constructs_and_executes_function_table_shells(gl):
    module = ast.parse(
        """
def affine(x, scale, offset):
    return x * scale + offset

def render_value(x):
    return affine(offset=4, x=x, scale=3)
"""
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    expected = {
        entry.reference.address: entry.graph
        for entry in graph.function_table
        if entry.graph is not None
    }
    deployment_type = strategize_glsl_deployment(graph)

    assert set(deployment_type.function_shell_types) == set(expected)
    deployment = deployment_type(profiling=True)
    try:
        assert set(deployment.function_shells) == set(expected)
        for reference, shell in deployment.function_shells.items():
            assert set(expected[reference].G) <= set(shell.process_graph.G)
            assert all(
                parent in shell.process_graph.G
                for _node_id, data in shell.process_graph.G.nodes(data=True)
                for parent, _role in data.get("parents", ())
            )
            assert shell.function_shells is deployment.function_shells

        deployment.compile_process_graph()
        render_reference = graph.function_table.reference("render_value")
        assert render_reference is not None
        render_shell = deployment.function_shells[
            render_reference.address
        ]
        samples = np.arange(8, dtype=np.float32)
        result = render_shell.execute_named({"x": samples})["result_0"]
        np.testing.assert_allclose(result.numpy(), samples * 3.0 + 4.0)
        report = deployment.profile_report()
        assert report["total_ms"] > 0
        assert any(
            row["section"] == "dispatch"
            and row["dispatches"] > 0
            for row in report["rows"]
        )
        shell_paths = {
            row["path"]
            for row in report["rows"]
            if row["section"] == "shell"
        }
        assert any("render_value" in path for path in shell_paths)
        assert any("affine" in path for path in shell_paths)
        summary = deployment.profile_summary(window=8)
        assert summary["frames"] == 1
        assert summary["total_p95_ms"] >= summary["total_mean_ms"]
    finally:
        deployment.release()
