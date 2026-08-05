from __future__ import annotations

import ast
import contextlib
import io
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import sympy

from src.common.tensors.accelerator_backends.glsl_fused_network import (
    GLSLFusedProgramNetwork,
)
from src.common.tensors.accelerator_backends.glsl_backend import (
    GLContextUnavailable,
    InstalledGLSLControlShell,
    build_control_shader_artifact,
    compile_captured_fused_program,
    execute_captured_fused_program,
    require_gl_context,
)
from src.compiler.control_source import (
    ControlProgram,
    LoopBlock,
    SequenceBlock,
    StatementBlock,
    StreamPublishBlock,
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
from src.compiler.glsl_deployment_strategy import (
    _diagnostic_value_summary,
    _observe_process_graph_node,
    _planned_capture_context,
    _capture_feed_aliases,
    _tensorize_graph_input,
    _unique_runtime_feed_aliases,
    strategize_shell_deployment,
)
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


def test_dtype_descriptor_remains_structural_graph_metadata():
    dtype = np.dtype("float32")

    assert _tensorize_graph_input(dtype, device=None) is dtype


def test_module_exporting_shape_function_remains_structural():
    assert callable(sympy.shape)
    assert _tensorize_graph_input(sympy, device=None) is sympy
    assert _diagnostic_value_summary(sympy).startswith("module:")


def test_shape_only_nested_domain_object_is_not_tensorized():
    class DomainValue:
        shape = (1,)
        tolist = list

    owner = SimpleNamespace(field=DomainValue())

    assert _tensorize_graph_input(owner, device=None) is owner


def test_installed_control_shell_executes_loop_in_one_dispatch(gl):
    program = _program(
        [10],
        [("add", 20, [10], {"right_scalar": 1.0})],
        {"result": 20},
        [10, 20],
    )
    control = ControlProgram(
        LoopBlock(
            "iteration",
            "0",
            "16",
            "1",
            StatementBlock(("__scheduled_region_3__",)),
            carried_aliases=((20, 10),),
        ),
        region_indices=(3,),
        value_aliases=((20, 10),),
    )
    artifact = build_control_shader_artifact(
        control,
        {3: CapturedFusedProgram(program, {})},
        instrumentation=True,
    )
    installed = InstalledGLSLControlShell(artifact)
    try:
        from src.common.tensors.accelerator_backends.glsl_backend import (
            dispatch_stats,
        )

        before = dispatch_stats()["calls"]
        result = installed.execute({
            10: np.arange(32, dtype=np.float32),
        })["result"]
        after = dispatch_stats()["calls"]

        np.testing.assert_allclose(
            result.numpy(),
            np.arange(32, dtype=np.float32) + 16.0,
        )
        assert after - before == 1
        codes = tuple(record[0] for record in installed.last_debug_records)
        assert codes[0] == 1
        assert 2 in codes
        assert 3 in codes
        assert 4 in codes
        assert codes[-1] == 5
        assert installed.last_debug_header[1] == 0
        assert installed.last_gpu_ms >= 0.0
    finally:
        installed.release()


def test_control_shader_commits_carried_values_only_in_their_own_loop():
    first = _program(
        [10],
        [("add", 20, [10], {"right_scalar": 1.0})],
        {"first": 20},
        [10, 20],
    )
    second = _program(
        [30],
        [("add", 40, [30], {"right_scalar": 2.0})],
        {"second": 40},
        [30, 40],
    )
    control = ControlProgram(
        SequenceBlock((
            LoopBlock(
                "first_iteration",
                "0",
                "2",
                "1",
                StatementBlock(("__scheduled_region_1__",)),
                carried_aliases=((20, 10),),
            ),
            LoopBlock(
                "second_iteration",
                "0",
                "3",
                "1",
                StatementBlock(("__scheduled_region_2__",)),
                carried_aliases=((40, 30),),
            ),
        )),
        region_indices=(1, 2),
        value_aliases=((20, 10), (40, 30)),
    )

    source = build_control_shader_artifact(
        control,
        {
            1: CapturedFusedProgram(first, {}),
            2: CapturedFusedProgram(second, {}),
        },
    ).source

    assert source.count(
        "arena[u_slot[0] + control_gid] = "
        "arena[u_slot[1] + control_gid];"
    ) == 1
    assert source.count(
        "arena[u_slot[2] + control_gid] = "
        "arena[u_slot[3] + control_gid];"
    ) == 1


def test_installed_control_shell_publishes_resident_stream_ranges(gl):
    program = _program(
        [10],
        [("add", 20, [10], {"right_scalar": 1.0})],
        {"result": 20},
        [10, 20],
    )
    control = ControlProgram(
        SequenceBlock((
            StatementBlock(("__scheduled_region_3__",)),
            StreamPublishBlock(stream_id=7, value_id=20, final=True),
        )),
        region_indices=(3,),
    )
    artifact = replace(
        build_control_shader_artifact(
            control,
            {3: CapturedFusedProgram(program, {})},
            instrumentation=True,
            device_resident=True,
        ),
        stream_word_capacity=64,
        stream_descriptor_capacity=4,
    )
    installed = InstalledGLSLControlShell(artifact)
    try:
        source = artifact.source
        assert "buffer TuringStreamState" in source
        assert "buffer TuringStreamWords" in source
        assert "turing_stream_publish(7u" in source

        samples = np.arange(32, dtype=np.float32)
        installed.execute({10: samples})
        item, = installed.drain_stream()

        assert item["stream_id"] == 7
        assert item["final"]
        np.testing.assert_array_equal(
            item["words"],
            (samples + 1.0).view(np.uint32),
        )
        assert installed.drain_stream() == ()
    finally:
        installed.release()


def test_installed_control_shell_drains_multiple_publications_in_one_batch(gl):
    program = _program(
        [10],
        [("add", 20, [10], {"right_scalar": 1.0})],
        {"result": 20},
        [10, 20],
    )
    control = ControlProgram(
        LoopBlock(
            "iteration",
            "0",
            "4",
            "1",
            SequenceBlock((
                StatementBlock(("__scheduled_region_3__",)),
                StreamPublishBlock(stream_id=7, value_id=20),
            )),
            carried_aliases=((20, 10),),
        ),
        region_indices=(3,),
        value_aliases=((20, 10),),
    )
    artifact = replace(
        build_control_shader_artifact(
            control,
            {3: CapturedFusedProgram(program, {})},
            device_resident=True,
        ),
        stream_word_capacity=128,
        stream_descriptor_capacity=4,
    )
    installed = InstalledGLSLControlShell(artifact)
    try:
        samples = np.arange(32, dtype=np.float32)
        installed.execute({10: samples})
        items = installed.drain_stream()

        assert len(items) == 4
        for iteration, item in enumerate(items, start=1):
            np.testing.assert_array_equal(
                item["words"],
                (samples + float(iteration)).view(np.uint32),
            )
        assert installed.drain_stream() == ()
    finally:
        installed.release()


def test_resident_stream_resumes_without_replaying_completed_iterations(gl):
    program = _program(
        [10],
        [("add", 20, [10], {"right_scalar": 1.0})],
        {"result": 20},
        [10, 20],
    )
    control = ControlProgram(
        LoopBlock(
            "iteration",
            "0",
            "4",
            "1",
            SequenceBlock((
                StatementBlock(("__scheduled_region_3__",)),
                StreamPublishBlock(stream_id=7, value_id=20),
            )),
            carried_aliases=((20, 10),),
        ),
        region_indices=(3,),
        value_aliases=((20, 10),),
    )
    artifact = replace(
        build_control_shader_artifact(
            control,
            {3: CapturedFusedProgram(program, {})},
            device_resident=True,
        ),
        stream_word_capacity=64,
        stream_descriptor_capacity=1,
    )
    assert artifact.stream_continuation_count == 1
    assert "turing_resume_marker_0" in artifact.source
    installed = InstalledGLSLControlShell(artifact)
    try:
        samples = np.arange(32, dtype=np.float32)
        installed.execute({10: samples})
        published = []
        while True:
            published.extend(installed.drain_stream())
            if installed.last_stream_status != 1:
                break
            installed.resume()

        assert len(published) == 4
        for iteration, item in enumerate(published, start=1):
            np.testing.assert_array_equal(
                item["words"],
                (samples + float(iteration)).view(np.uint32),
            )
    finally:
        installed.release()


def test_closure_aggregate_fields_use_one_native_loop_body():
    program = _program(
        [20, 21],
        [("add", 22, [20, 21], {})],
        {"result": 22},
        [20, 21, 22, 30, 31, 40, 41],
    )
    induction = "iteration_9"
    control = ControlProgram(
        LoopBlock(
            induction,
            "0",
            "__iterable_extent_99__",
            "1",
            StatementBlock(("__scheduled_region_3__",)),
        ),
        region_indices=(3,),
        closure_iterable_bindings=(
            (99, 20, induction, (30, 31)),
            (99, 21, induction, (40, 41)),
        ),
    )

    artifact = build_control_shader_artifact(
        control,
        {3: CapturedFusedProgram(program, {})},
        device_resident=True,
    )

    assert artifact.source.count(
        f"for (int {induction} = 0; {induction} < 2;"
    ) == 1
    assert artifact.source.count("float s2 = s0 + s1;") == 1
    assert artifact.source.count(f"switch (int({induction}))") == 2


def test_static_loop_target_is_present_in_artifact_slot_abi():
    program = _program(
        [10],
        [("add", 20, [10], {"right_scalar": 1.0})],
        {"result": 20},
        [10, 20],
    )
    induction = "iteration_9"
    control = ControlProgram(
        LoopBlock(
            induction,
            "0",
            "3",
            "1",
            StatementBlock(("__scheduled_region_3__",)),
        ),
        region_indices=(3,),
        static_iterable_bindings=((99, 30, induction, (1, 5, 9)),),
    )

    artifact = build_control_shader_artifact(
        control,
        {3: CapturedFusedProgram(program, {})},
        device_resident=True,
    )

    assert 30 in artifact.slot_value_ids
    assert artifact.value_meta[30] == Meta((), "int32", "glsl")


def test_collection_target_storage_follows_loop_and_source_extent():
    program = _program(
        [10],
        [("add", 20, [10], {"right_scalar": 1.0})],
        {"result": 20},
        [10, 20],
    )
    induction = "iteration_9"
    control = ControlProgram(
        LoopBlock(
            induction,
            "0",
            "4",
            "1",
            StatementBlock(("__scheduled_region_3__",)),
        ),
        region_indices=(3,),
        collection_bindings=((20, 30, induction, 0),),
    )

    artifact = build_control_shader_artifact(
        control,
        {3: CapturedFusedProgram(program, {})},
        device_resident=True,
    )

    assert 30 in artifact.slot_value_ids
    assert artifact.value_meta[30] == Meta((4, 32), "float32", "glsl")
    assert (
        "for (uint collection_gid_0 = gl_LocalInvocationID.x;"
        in artifact.source
    )
    assert "collection_gid_0 += gl_WorkGroupSize.x" in artifact.source


def test_artifact_cache_identity_ignores_independent_step_schedule_order():
    first = _program(
        [10, 11],
        [
            ("add", 20, [10], {"right_scalar": 1.0}),
            ("mul", 21, [11], {"right_scalar": 2.0}),
        ],
        {"result": 21},
        [10, 11, 20, 21],
    )
    second = _program(
        [10, 11],
        [
            ("mul", 21, [11], {"right_scalar": 2.0}),
            ("add", 20, [10], {"right_scalar": 1.0}),
        ],
        {"result": 21},
        [10, 11, 20, 21],
    )
    # Step IDs are schedule positions and may consequently differ.  Producer
    # IDs and dependencies are the semantic record.
    control = ControlProgram(
        StatementBlock(("__scheduled_region_3__",)),
        region_indices=(3,),
    )

    left = build_control_shader_artifact(
        control, {3: CapturedFusedProgram(first, {})}
    )
    right = build_control_shader_artifact(
        control, {3: CapturedFusedProgram(second, {})}
    )

    assert left.source != right.source
    assert left.phase_cache_identities == right.phase_cache_identities


def test_c_dispatch_shell_computes_retained_loop_commands(gl):
    program = _program(
        [10],
        [("add", 20, [10], {"right_scalar": 1.0})],
        {"frame": 20},
        [10, 20],
    )
    induction = "iteration_9"
    control = ControlProgram(
        LoopBlock(
            induction,
            "1",
            "7",
            "2",
            StatementBlock(("__scheduled_region_3__",)),
            dispatch_shell="c",
        ),
        region_indices=(3,),
        collection_bindings=((20, 30, induction, 1),),
    )
    artifact = build_control_shader_artifact(
        control,
        {3: CapturedFusedProgram(program, {})},
        device_resident=True,
        terminal_outputs={"frames": 30},
    )

    assert artifact.c_dispatch_loop_bounds == ("1", "7", "2")
    assert "for (int iteration_9" not in artifact.source
    assert "int iteration_9 = u_dispatch_iteration;" in artifact.source

    installed = InstalledGLSLControlShell(artifact)
    try:
        result = installed.execute({
            10: np.arange(32, dtype=np.float32),
        })
        np.testing.assert_allclose(
            result["frames"].numpy(),
            np.stack(
                [np.arange(32, dtype=np.float32) + 1.0] * 3
            ),
        )
        assert installed.last_dispatches == 3
    finally:
        installed.release()


def test_one_shader_dispatch_maps_frame_batch_to_workgroups(gl):
    program = _program(
        [10],
        [("add", 20, [10], {"right_scalar": 1.0})],
        {"frame": 20},
        [10, 20],
    )
    induction = "iteration_9"
    control = ControlProgram(
        LoopBlock(
            induction,
            "0",
            "3",
            "1",
            StatementBlock(("__scheduled_region_3__",)),
            parallel_iterations=True,
        ),
        region_indices=(3,),
        collection_bindings=((20, 30, induction, 0),),
    )
    artifact = build_control_shader_artifact(
        control,
        {3: CapturedFusedProgram(program, {})},
        device_resident=True,
        terminal_outputs={"frames": 30},
    )

    assert artifact.workgroup_loop_bounds == ("0", "3", "1")
    assert artifact.private_value_capacities[20] == 3 * 32
    assert "gl_WorkGroupID.x" in artifact.source

    installed = InstalledGLSLControlShell(artifact)
    try:
        result = installed.execute({
            10: np.arange(32, dtype=np.float32),
        })
        np.testing.assert_allclose(
            result["frames"].numpy(),
            np.stack(
                [np.arange(32, dtype=np.float32) + 1.0] * 3
            ),
        )
        assert installed.last_dispatches == 1
    finally:
        installed.release()


def test_logging_ssbo_records_are_visible_in_profile_lines():
    from src.compiler.glsl_deployment_strategy import DeploymentProfiler

    profiler = DeploymentProfiler(enabled=True)
    token = profiler.begin_shell("root/kernel")
    profiler.record_device_trace(
        path="root/kernel",
        records=((1, 7, 0, 0), (3, 11, 128, 4), (5, 19, 128, 0)),
        header=(3, 0, 1, 0),
    )
    profiler.end_shell("root/kernel", token)

    summary = profiler.summary()
    assert tuple(row["label"] for row in summary["device_rows"]) == (
        "closure-enter",
        "region-execute",
        "output-publish",
    )
    assert any(
        "logging-ssbo | region-execute[3] | "
        "events 1.0/invocation | payload 128.0,4.0"
        in line
        for line in (
            f"  {row['path']} | logging-ssbo | "
            f"{row['label']}[{row['code']}] | "
            f"events {row['events_mean']:.1f}/invocation | "
            f"payload {row['payload0_mean']:.1f},"
            f"{row['payload1_mean']:.1f}"
            for row in summary["device_rows"]
        )
    )


def _capture_glsl(operation):
    with autograd.forward_capture() as tape:
        result = operation()
    captured = compile_recorded_fused_tape(tape)
    compile_captured_fused_program(captured)
    replayed = execute_captured_fused_program(captured, {})["result_0"]
    return result, replayed, captured


def test_capture_feed_aliases_reuse_the_boundary_graph_identity():
    storage = object()
    boundary = SimpleNamespace(data=storage)
    alias = SimpleNamespace(data=storage)
    captured = CapturedFusedProgram(
        _program(
            feeds={101, 102},
            steps=[("add", 103, (101, 102), {})],
            outputs={"result_0": 103},
            ids={101, 102, 103},
        ),
        {101: boundary, 102: alias},
    )

    assert _capture_feed_aliases(captured, {101: 19}) == {
        101: 19,
        102: 19,
    }


def test_planned_capture_observer_does_not_merge_shared_storage_endpoints():
    storage = object()
    left = SimpleNamespace(data=storage)
    right = SimpleNamespace(data=storage)
    left_node = SimpleNamespace(ctx={}, parents=[(101, 0)])
    right_node = SimpleNamespace(ctx={}, parents=[(101, 0)])
    tape = SimpleNamespace(_nodes={
        id(left): left_node,
        id(right): right_node,
    })
    facts = {
        "tape": tape,
        "node_capture_ids": {},
        "step_input_ids": {},
    }

    token = _planned_capture_context.set(facts)
    try:
        assert _observe_process_graph_node(
            17, (3,), (SimpleNamespace(),), left
        ) is left
        assert _observe_process_graph_node(
            23, (9,), (SimpleNamespace(),), right
        ) is right
    finally:
        _planned_capture_context.reset(token)

    assert facts["node_capture_ids"] == {
        17: [id(left)],
        23: [id(right)],
    }
    assert left_node.ctx["process_graph_node_id"] == 17
    assert right_node.ctx["process_graph_node_id"] == 23
    assert facts["step_input_ids"] == {
        id(left): ((101, 3),),
        id(right): ((101, 9),),
    }


def test_planned_capture_observer_does_not_rename_passthrough_result():
    result = SimpleNamespace()
    primitive = SimpleNamespace(ctx={}, parents=[(102, 1), (101, 0)])
    tape = SimpleNamespace(_nodes={id(result): primitive})
    facts = {
        "tape": tape,
        "node_capture_ids": {},
        "step_input_ids": {},
    }

    token = _planned_capture_context.set(facts)
    try:
        left = SimpleNamespace()
        right = SimpleNamespace()
        primitive.parents = [(id(right), 1), (id(left), 0)]
        _observe_process_graph_node(17, (3, 4), (left, right), result)
        _observe_process_graph_node(
            999999, (17,), (result,), result
        )
    finally:
        _planned_capture_context.reset(token)

    assert primitive.ctx["process_graph_node_id"] == 17
    assert facts["node_capture_ids"] == {17: [id(result)]}
    assert facts["step_input_ids"] == {
        id(result): ((id(left), 3), (id(right), 4)),
    }


@pytest.mark.parametrize(
    ("collection_owners", "dim", "expected_materialization"),
    (
        (frozenset(), 0, False),
        (frozenset({5}), 0, True),
        (frozenset({5}), 1, False),
    ),
)
def test_planned_collection_materialization_requires_owned_compatible_storage(
    collection_owners,
    dim,
    expected_materialization,
):
    result = SimpleNamespace()
    primitive = SimpleNamespace(
        op="stack",
        ctx={"params": {"dim": dim}},
        parents=[],
    )
    facts = {
        "tape": SimpleNamespace(_nodes={id(result): primitive}),
        "node_capture_ids": {},
        "step_input_ids": {},
        "collection_materializations": {},
        "value_aliases": {},
        "collection_owner_ids": collection_owners,
    }

    token = _planned_capture_context.set(facts)
    try:
        _observe_process_graph_node(
            17,
            (5,),
            ((SimpleNamespace(),),),
            result,
        )
    finally:
        _planned_capture_context.reset(token)

    if expected_materialization:
        assert facts["collection_materializations"] == {id(result): 5}
    else:
        assert facts["collection_materializations"] == {}
    assert facts["value_aliases"] == {}


def test_runtime_feed_alias_requires_one_shape_and_dtype_match(gl):
    vector = GLSLTensorOperations.tensor(
        np.arange(12, dtype=np.int32)
    )
    scalar_a = GLSLTensorOperations.tensor(
        np.array(1, dtype=np.int32)
    )
    scalar_b = GLSLTensorOperations.tensor(
        np.array(2, dtype=np.int32)
    )
    program = _program(
        feeds={101, 102},
        steps=[("add", 103, (101, 102), {})],
        outputs={"result_0": 103},
        ids={101, 102, 103},
    )
    program.meta[101] = Meta(
        shape=(12,), dtype="int32", device="glsl"
    )
    program.meta[102] = Meta(
        shape=(), dtype="int32", device="glsl"
    )

    assert _unique_runtime_feed_aliases(
        program,
        (101, 102),
        {19: vector, 84: (scalar_a, scalar_b)},
    ) == {101: vector}


def test_recorded_tensor_constructor_is_a_stage_not_an_external_feed(gl):
    source = GLSLTensorOperations.tensor(
        np.arange(8, dtype=np.float32)
    )
    with autograd.forward_capture() as tape:
        scalar_tensor = source.ensure_tensor(4.0)
        result = source + scalar_tensor

    captured = compile_recorded_fused_tape(
        tape,
        outputs={"result_0": result},
    )

    assert id(source) in captured.program.feeds
    assert id(scalar_tensor) not in captured.program.feeds
    assert not any(
        (stage.extras or {}).get("kernel_kind") == "constant"
        for stage in captured.stages
    )
    assert any(
        step.attrs.get("right_scalar") == 4.0
        for step in captured.program.steps
    )
    replayed = execute_captured_fused_program(captured, {})["result_0"]
    np.testing.assert_allclose(
        replayed.numpy(),
        np.arange(8, dtype=np.float32) + 4.0,
    )


def test_native_index_stage_reuses_the_tensor_producers_identity(gl):
    source = GLSLTensorOperations.tensor(
        np.arange(8, dtype=np.float32)
    )
    with autograd.forward_capture() as tape:
        indices = (source * 0).to_dtype("int64")
        result = source[indices]

    captured = compile_recorded_fused_tape(
        tape,
        outputs={"result_0": result},
    )

    assert id(indices.data) not in captured.program.feeds
    replayed = execute_captured_fused_program(captured, {})["result_0"]
    np.testing.assert_allclose(
        replayed.numpy(),
        np.zeros(8, dtype=np.float32),
    )


def test_explicit_long_conversion_lowers_after_layout_stage(gl):
    source = GLSLTensorOperations.tensor(
        np.arange(8, dtype=np.float32)
    )
    with autograd.forward_capture() as tape:
        result = source.reshape(2, 4).long()

    captured = compile_recorded_fused_tape(
        tape,
        outputs={"result": result},
    )
    replayed = execute_captured_fused_program(captured, {})["result"]

    assert any(
        step.op_name == "fptosi"
        and stage.meta[step.result_id].dtype == str(result.dtype)
        for stage in captured.execution_programs
        for step in stage.steps
    )
    np.testing.assert_array_equal(
        replayed.numpy(),
        np.arange(8, dtype=np.int32).reshape(2, 4),
    )


def test_stage_partition_preserves_fanout_consumed_inside_and_after_stage(gl):
    source = GLSLTensorOperations.tensor(
        np.arange(8, dtype=np.float32)
    )
    with autograd.forward_capture() as tape:
        shared = source * 2.0
        first = shared + 1.0
        reshaped = first.reshape(2, 4)
        later = shared + 3.0

    captured = compile_recorded_fused_tape(
        tape,
        outputs={"reshaped": reshaped, "later": later},
    )

    # ``shared`` has a consumer in its first elementwise stage and another
    # consumer after a layout stage.  It is therefore a resident stage
    # live-out, never a fabricated external feed.
    assert id(shared) not in captured.program.feeds
    assert any(
        id(shared) in stage.outputs.values()
        for stage in captured.stages
    )
    replayed = execute_captured_fused_program(captured, {})
    np.testing.assert_allclose(replayed["reshaped"].numpy(), reshaped.numpy())
    np.testing.assert_allclose(replayed["later"].numpy(), later.numpy())


def test_mixed_slice_elementwise_reduction_tape_compiles_without_fallback(gl):
    source = GLSLTensorOperations.tensor(
        np.arange(30, dtype=np.float32).reshape(5, 6)
    )
    with autograd.forward_capture() as tape:
        result = ((source[:, 1:5] * 2.0) + 1.0).sum(
            dim=1,
            keepdim=True,
        )

    captured = compile_recorded_fused_tape(tape)
    source_text = compile_captured_fused_program(captured)
    replayed = execute_captured_fused_program(captured, {})["result_0"]

    assert captured.stages
    assert "captured stage" in source_text
    np.testing.assert_allclose(
        replayed.numpy(),
        result.numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_requested_outputs_trim_abandoned_tape_branches(gl):
    source = GLSLTensorOperations.tensor(
        np.arange(24, dtype=np.float32).reshape(4, 6)
    )
    with autograd.forward_capture() as tape:
        kept = source * 3.0 + 2.0
        _abandoned = source[:, 1:5].sum(dim=1)

    captured = compile_recorded_fused_tape(
        tape,
        outputs={"kept": kept},
    )
    source_text = compile_captured_fused_program(captured)
    replayed = execute_captured_fused_program(captured, {})["kept"]

    stage_ops = {
        step.op_name
        for stage in captured.execution_programs
        for step in stage.steps
    }
    assert "slice" not in stage_ops
    assert "sum" not in stage_ops
    assert "tensor_from_list" not in stage_ops
    assert "sum" not in source_text
    np.testing.assert_allclose(replayed.numpy(), kept.numpy())


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
    deployment = strategize_shell_deployment(graph)(
        legacy_fused_network=True,
    )
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


def test_composed_control_is_default_and_legacy_network_is_explicit(gl):
    graph = ProcessGraph(materialize_memory=False)
    graph.G.add_node(
        0,
        label="input",
        type="input",
        op="input",
        parents=[],
        children=[],
    )
    deployment = strategize_shell_deployment(graph)()
    program = _program(
        [0],
        [("mul", 1, [0], {"right_scalar": 4.0})],
        {"result": 1},
        [0, 1],
    )

    assert deployment.control_runtime == "composed_control"
    assert not deployment.legacy_fused_network
    with pytest.raises(RuntimeError, match="is retired"):
        deployment.install_fused_programs((program,))


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
    deployment = strategize_shell_deployment(graph)(
        input_slots=2,
        output_slots=2,
        legacy_fused_network=True,
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
    deployment = strategize_shell_deployment(graph)(
        legacy_fused_network=True,
    )
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


def test_glsl_deployment_rejects_per_region_tape_capture(gl):
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

    deployment = strategize_shell_deployment(
        graph,
        max_nodes_per_dispatch=1,
    )(legacy_fused_network=True)
    values = np.arange(32, dtype=np.float32)
    with pytest.raises(RuntimeError, match="per-region tape capture"):
        deployment.capture_forward_tapes(({"samples": values},))
    with pytest.raises(
        RuntimeError,
        match="per-region scheduled tape capture",
    ):
        deployment.capture_scheduled_forward_tapes({0: values})


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

    deployment = strategize_shell_deployment(graph)()
    values = np.arange(16, dtype=np.float32)
    deployment.compile_process_graph()
    try:
        deployment.capture_fused_programs({"samples": values})
        assert deployment._discovery_tape_creations == 1
        assert deployment._discovery_tape is None
        assert deployment.installed_control_shell is not None
        compiled_outputs = deployment.execute_named({"samples": values})
        assert len(compiled_outputs) == 1
        compiled = next(iter(compiled_outputs.values()))
        np.testing.assert_allclose(compiled.numpy(), values * 2.0)
    finally:
        deployment.release()


def test_precompile_only_observes_numerics_without_glsl_compilation(
    monkeypatch,
):
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
        children=[],
        attributes={},
    )
    graph.G.add_edges_from(((0, 2), (1, 2)))
    graph.roots = [2]
    graph.compute_levels(method="asap", order="dependency")

    from src.common.tensors.accelerator_backends import glsl_backend

    def reject_glsl_compile(*_args, **_kwargs):
        raise AssertionError("precompile-only discovery compiled GLSL")

    monkeypatch.setattr(glsl_backend, "_compile", reject_glsl_compile)
    deployment = strategize_shell_deployment(graph)()
    try:
        deployment.compile_process_graph()
        deployment.capture_fused_programs(
            {"samples": np.arange(16, dtype=np.float32)},
            precompile_only=True,
        )
        assert deployment.compiled_shell_program is not None
        assert deployment.compiled_shell_program.program.steps
        assert deployment.composed_shell_blockers == ("precompile-only",)
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
    deployment_type = strategize_shell_deployment(graph)

    assert set(deployment_type.function_shell_types) == set(expected)
    deployment = deployment_type(
        profiling=True,
    )
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
        render_shell.capture_fused_programs({"x": samples})
        result = render_shell.execute_named({"x": samples})["result_0"]
        np.testing.assert_allclose(result.numpy(), samples * 3.0 + 4.0)
        report = deployment.profile_report()
        assert report["total_ms"] > 0
        assert any(
            row["section"] == "compiled-glsl"
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
        program_table = "\n".join(render_shell.program_table_lines())
        assert "compiled program shell hierarchy" in program_table
        assert "compiled program region compartments" in program_table
        assert "callsite-" in program_table
        assert "affine" in program_table
        summary = deployment.profile_summary(window=8)
        assert summary["frames"] == 1
        assert summary["total_p95_ms"] >= summary["total_mean_ms"]
    finally:
        deployment.release()


def test_nested_shell_tree_uses_one_discovery_tape_and_one_lowering(gl):
    module = ast.parse(
        """
def affine(x, scale):
    return x * scale

def render(x):
    return affine(x, 3.0) + 4.0
"""
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    deployment = strategize_shell_deployment(graph)()
    try:
        deployment.compile_process_graph()
        render_ref = graph.function_table.reference("render")
        assert render_ref is not None
        render = deployment.function_shells[render_ref.address]
        assert render.callsite_function_shells
        affine = next(iter(render.callsite_function_shells.values()))
        samples = np.arange(8, dtype=np.float32)

        render.capture_fused_programs({"x": samples})

        assert render._discovery_tape_creations == 1
        assert render._discovery_tape_lowerings == 1
        assert affine._discovery_tape_creations == 0
        assert affine._discovery_tape_lowerings == 0
        assert render._discovery_tape is None
        assert affine._discovery_tape is None
        assert render._discovery_session is None
        assert not render.forward_region_planned_capture_ids
        assert not render.forward_region_planned_input_ids
        assert not affine.forward_region_planned_capture_ids
        assert not affine.forward_region_planned_input_ids
        assert render.compiled_shell_program is not None
        assert affine.compiled_shell_program is not None

        result = render.execute_named({"x": samples})["result_0"]
        np.testing.assert_allclose(result.numpy(), samples * 3.0 + 4.0)
        assert render._discovery_tape_creations == 1
        assert affine._discovery_tape_creations == 0
    finally:
        deployment.release()


def test_nested_shell_structured_argument_preserves_leaf_identities(gl):
    module = ast.parse(
        """
def make_planes(x):
    return (x + 1.0, x + 2.0, x + 3.0)

def consume_planes(planes):
    return planes[0] + planes[1] + planes[2]

def render(x):
    return consume_planes(make_planes(x))
"""
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    deployment = strategize_shell_deployment(graph)()
    try:
        deployment.compile_process_graph()
        render_ref = graph.function_table.reference("render")
        assert render_ref is not None
        render = deployment.function_shells[render_ref.address]
        samples = np.arange(8, dtype=np.float32)

        render.capture_fused_programs({"x": samples})

        result = render.execute_named({"x": samples})["result_0"]
        np.testing.assert_allclose(result.numpy(), samples * 3.0 + 6.0)
        produced_ids = [
            int(value_id)
            for row in render.installed_control_shell.artifact.snippet_diagnostics
            for _name, value_id in row["outputs"]
        ]
        assert len(produced_ids) == len(set(produced_ids)), (
            "a structured call boundary copied a producer into its consumer "
            "scope, giving one forward SSA value multiple writers"
        )
        assert render._discovery_tape_creations == 1
        assert render._discovery_tape_lowerings == 1
        assert all(
            child._discovery_tape_creations == 0
            and child._discovery_tape_lowerings == 0
            for child in render.callsite_function_shells.values()
        )
    finally:
        deployment.release()


def test_glsl_shell_replays_loop_carried_regions_and_streams_verbose_trace(gl):
    module = ast.parse(
        """
def recurrent(x, iterations):
    total = x * 0.0
    state = x
    for _ in range(iterations):
        total = total + state
        state = state * 2.0
    return total, state
"""
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    deployment_type = strategize_shell_deployment(graph)
    deployment = deployment_type(
        profiling=True,
        verbose_profile=True,
    )
    try:
        reference = graph.function_table.reference("recurrent")
        assert reference is not None
        shell = deployment.function_shells[reference.address]
        assert (
            "planner loop-to-shader reduction analysis"
            in "\n".join(shell.program_table_lines())
        )
        deployment.compile_process_graph()
        samples = np.arange(1, 9, dtype=np.float32)
        feeds = {"x": samples, "iterations": 4}
        with contextlib.redirect_stdout(io.StringIO()):
            shell.capture_fused_programs(feeds)
        assert shell.installed_control_shell is not None
        assert shell._discovery_tape_creations == 1
        assert shell._discovery_tape_lowerings == 1
        assert shell.compiled_shell_program is not None
        assert shell._discovery_tape_complete
        assert shell._discovery_tape is None
        assert not shell.forward_tapes
        with pytest.raises(RuntimeError, match="one discovery tape"):
            shell.capture_fused_programs(feeds)
        shell._profiler.trace_history.clear()
        from src.common.tensors.accelerator_backends.glsl_backend import (
            dispatch_stats,
        )
        dispatches_before = dispatch_stats()["calls"]
        with contextlib.redirect_stdout(io.StringIO()) as streamed:
            result = shell.execute_named(feeds)
        dispatches_after = dispatch_stats()["calls"]

        np.testing.assert_allclose(
            result["total"].numpy(),
            samples * 15.0,
        )
        np.testing.assert_allclose(
            result["state"].numpy(),
            samples * 16.0,
        )
        iterations = [
            record
            for record in shell.trace_report()
            if record["section"] == "loop-iteration"
        ]
        assert iterations == []
        assert "loop-iteration" not in streamed.getvalue()
        assert dispatches_after - dispatches_before == 1
        assert not shell.exception_report()
    finally:
        deployment.release()


def test_retained_collection_and_stack_keep_distinct_producer_ids(gl):
    module = ast.parse(
        """
def collect(x, iterations):
    values = []
    for index in range(iterations):
        values.append(x + index)
    return AbstractTensor.stack(values, dim=0)
"""
    )
    graph = ProcessGraph(materialize_memory=False)
    with contextlib.redirect_stdout(io.StringIO()):
        graph.build_from_ast(module)
    reduce_abstract_tensor_topology(graph)

    deployment_type = strategize_shell_deployment(graph)
    reference = graph.function_table.reference("collect")
    assert reference is not None
    shell_type = deployment_type.function_shell_types[reference.address]
    collection_id = next(
        collection
        for reduction in shell_type.loop_shader_reductions
        if reduction.control_program is not None
        for _source, collection, _induction, _start
        in reduction.control_program.collection_bindings
    )
    stack_id = next(
        node_id
        for node_id, data in shell_type.process_graph.G.nodes(data=True)
        if str(data.get("op") or data.get("type")).lower() == "stack"
    )

    assert collection_id != stack_id
    assert any(
        int(parent) == int(collection_id)
        for parent, _role in (
            shell_type.process_graph.G.nodes[stack_id].get("parents") or ()
        )
    )
