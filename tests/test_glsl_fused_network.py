from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.glsl_fused_network import (
    GLSLFusedProgramNetwork,
)
from src.common.tensors.accelerator_backends.glsl_backend import (
    GLContextUnavailable,
    require_gl_context,
)
from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep
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
