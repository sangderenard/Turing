from src.compiler.control_source import (
    ControlProgram,
    ControlTarget,
    LoopBlock,
    ParallelDeployment,
    RegionCode,
    SequenceBlock,
    StateMachineTick,
    StatementBlock,
    render_control_program,
    render_c_shell,
    project_control_regions,
    render_python_shell,
    compile_python_shell,
)
from src.common.tensors.accelerator_backends.glsl_backend import (
    compose_control_shader,
)
from src.common.tensors.accelerator_backends.c_primitive_program import (
    CapturedFusedProgram,
)
from src.common.tensors.fused_ir import FusedProgram, Meta, OpStep


def test_same_planned_loop_renders_as_python_c_and_glsl():
    root = LoopBlock(
        induction="iteration",
        start="0",
        stop="count",
        step="1",
        body=StatementBlock(("state = state + delta;",)),
    )

    program = ControlProgram(root)
    python = render_control_program(program, ControlTarget.PYTHON)
    c = render_control_program(program, ControlTarget.C)
    glsl = render_control_program(program, ControlTarget.GLSL)

    assert python.splitlines()[0] == "for iteration in range(0, count, 1):"
    assert c.splitlines()[0] == (
        "for (int iteration = 0; iteration < count; iteration += 1) {"
    )
    assert glsl == c


def test_state_machine_tick_is_one_transition_without_polling():
    tick = StateMachineTick(
        state="state",
        cases=(
            ("READY", StatementBlock(("state = RUNNING;",))),
            ("RUNNING", StatementBlock(("work();",))),
        ),
    )

    rendered = render_control_program(
        ControlProgram(tick),
        ControlTarget.GLSL,
    )

    assert rendered.startswith("switch (state) {")
    assert "while" not in rendered
    assert "case READY:" in rendered


def test_parallel_lanes_share_one_compiled_source_without_host_execution():
    parallel = ParallelDeployment((
        SequenceBlock((StatementBlock(("left();",)),)),
        SequenceBlock((StatementBlock(("right();",)),)),
    ))

    rendered = render_control_program(
        ControlProgram(parallel),
        ControlTarget.GLSL,
    )

    assert rendered.splitlines() == ["left();", "right();"]


def test_glsl_control_composition_absorbs_region_body_into_loop():
    program = FusedProgram(
        version=1,
        feeds={10},
        steps=[OpStep(
            0,
            "add",
            [10],
            {"right_scalar": 1.0},
            result_id=20,
        )],
        outputs={"result": 20},
        meta={
            10: Meta(shape=(8,), dtype="float32"),
            20: Meta(shape=(8,), dtype="float32"),
        },
    )
    control = ControlProgram(
        LoopBlock(
            "iteration",
            "0",
            "16",
            "1",
            StatementBlock(("__scheduled_region_3__",)),
        ),
        region_indices=(3,),
    )

    source = compose_control_shader(
        control,
        {3: CapturedFusedProgram(program, {})},
    )

    assert source.count("#version 430") == 1
    assert source.count("void main()") == 1
    assert "__scheduled_region_3__" not in source
    assert "for (int iteration = 0; iteration < 16;" in source
    assert "float s1 = s0 + float(1.0);" in source


def test_language_is_selected_after_logical_region_composition():
    logical = ControlProgram(
        LoopBlock(
            "iteration",
            "0",
            "count",
            "1",
            StatementBlock(("__scheduled_region_7__",)),
        ),
        region_indices=(7,),
    )
    c_source = render_c_shell(
        logical,
        (RegionCode(
            7,
            ControlTarget.C,
            StatementBlock(("state += delta;",)),
        ),),
        function_name="run_shell",
        parameters=("float *state", "float delta", "int count"),
    )
    python_source = render_python_shell(
        logical,
        (RegionCode(
            7,
            ControlTarget.PYTHON,
            StatementBlock(("state += delta",)),
        ),),
        function_name="run_shell",
        parameters=("state", "delta", "count"),
    )

    assert "void run_shell(float *state, float delta, int count)" in c_source
    assert "for (int iteration = 0;" in c_source
    assert python_source.startswith("def run_shell(state, delta, count):")
    assert "for iteration in range(0, count, 1):" in python_source


def test_project_control_regions_collapses_resolved_loop_body():
    logical = ControlProgram(
        LoopBlock(
            "i", "0", "4", "1",
            StatementBlock(("__scheduled_region_3__",)),
        ),
        region_indices=(3,),
    )

    projected = project_control_regions(logical, ())

    assert projected.region_indices == ()
    assert isinstance(projected.root, SequenceBlock)
    assert projected.root.blocks == ()


def test_project_control_regions_keeps_structural_binding_for_live_loop():
    logical = ControlProgram(
        LoopBlock(
            "iteration_9",
            "0",
            "__iterable_extent_40__",
            "1",
            StatementBlock(("__scheduled_region_3__",)),
        ),
        region_indices=(3,),
        iterable_bindings=((40, 41, "iteration_9"),),
    )

    projected = project_control_regions(
        logical,
        (3,),
        retained_value_ids=(100, 101),
    )

    assert projected.iterable_bindings == ((40, 41, "iteration_9"),)


def test_control_shader_preserves_repeated_structural_operand_slots():
    program = FusedProgram(
        version=1,
        feeds={20, 22},
        steps=[OpStep(
            0,
            "stack",
            [20, 22, 22],
            {"dim": 0},
            result_id=30,
        )],
        outputs={"result": 30},
        meta={
            20: Meta(shape=(4,), dtype="float32"),
            22: Meta(shape=(4,), dtype="float32"),
            30: Meta(shape=(3, 4), dtype="float32"),
        },
        extras={"kernel_kind": "stack"},
    )
    control = ControlProgram(
        StatementBlock(("__scheduled_region_0__",)),
        region_indices=(0,),
    )

    source = compose_control_shader(
        control,
        {0: CapturedFusedProgram(program, {})},
    )

    assert "u_slot[0]" in source
    assert "u_slot[1]" in source
    assert "u_slot[2]" in source
    assert "u_slot[3]" in source


def test_control_shader_routes_structured_profile_events_through_ssbo():
    program = FusedProgram(
        version=1,
        feeds={10},
        steps=[OpStep(
            0, "add", [10], {"right_scalar": 1.0}, result_id=20
        )],
        outputs={"result": 20},
        meta={
            10: Meta(shape=(8,), dtype="float32"),
            20: Meta(shape=(8,), dtype="float32"),
        },
    )
    control = ControlProgram(
        LoopBlock(
            "iteration",
            "0",
            "4",
            "1",
            StatementBlock(("__scheduled_region_0__",)),
        ),
        region_indices=(0,),
    )

    source = compose_control_shader(
        control,
        {0: CapturedFusedProgram(program, {})},
        instrumentation=True,
    )

    assert "buffer TuringDebugLog" in source
    assert "atomicAdd(debug_words[1], 1u)" in source
    assert "turing_debug_event(1u" in source
    assert "turing_debug_event(2u" in source
    assert "turing_debug_event(3u" in source
    assert "turing_debug_event(5u" in source


def test_glsl_shell_rejects_non_glsl_interior_at_late_selection():
    from src.compiler.control_source import compose_region_code

    logical = ControlProgram(
        StatementBlock(("__scheduled_region_2__",)),
        region_indices=(2,),
    )

    try:
        compose_region_code(
            logical,
            ControlTarget.GLSL,
            (RegionCode(
                2,
                ControlTarget.C,
                StatementBlock(("work();",)),
            ),),
        )
    except ValueError as error:
        assert "glsl shell requires glsl interiors" in str(error)
    else:
        raise AssertionError("mixed-language GLSL shell was accepted")


def test_c_shell_launches_glsl_interior_without_ingesting_shader_source():
    logical = ControlProgram(
        StatementBlock(("__scheduled_region_2__",)),
        region_indices=(2,),
    )

    source = render_c_shell(
        logical,
        (RegionCode(
            2,
            ControlTarget.GLSL,
            StatementBlock(("void main() { shader_work(); }",)),
            launch_body=StatementBlock((
                "launch_glsl_region(2, inputs, outputs);",
            )),
        ),),
        function_name="run_shell",
        parameters=("void *inputs", "void *outputs"),
    )

    assert "launch_glsl_region(2, inputs, outputs);" in source
    assert "shader_work" not in source


def test_c_shell_requires_explicit_launcher_for_glsl_interior():
    logical = ControlProgram(
        StatementBlock(("__scheduled_region_2__",)),
        region_indices=(2,),
    )

    try:
        render_c_shell(
            logical,
            (RegionCode(
                2,
                ControlTarget.GLSL,
                StatementBlock(("void main() {}",)),
            ),),
            function_name="run_shell",
        )
    except ValueError as error:
        assert "explicit C launch block" in str(error)
    else:
        raise AssertionError("foreign interior had no C launch ABI")


def test_python_shell_does_not_inherit_c_foreign_launch_permission():
    logical = ControlProgram(
        StatementBlock(("__scheduled_region_2__",)),
        region_indices=(2,),
    )

    try:
        render_python_shell(
            logical,
            (RegionCode(
                2,
                ControlTarget.GLSL,
                StatementBlock(("void main() {}",)),
                launch_body=StatementBlock(("launch_glsl_region(2)",)),
            ),),
            function_name="run_shell",
        )
    except ValueError as error:
        assert "python shell requires python interiors" in str(error)
    else:
        raise AssertionError("Python inherited the C shell's GLSL permission")


def test_python_shell_finalizes_to_callable_only_after_selection():
    logical = ControlProgram(
        LoopBlock(
            "iteration",
            "0",
            "count",
            "1",
            StatementBlock(("__scheduled_region_1__",)),
        ),
        region_indices=(1,),
    )
    function = compile_python_shell(
        logical,
        (RegionCode(
            1,
            ControlTarget.PYTHON,
            StatementBlock(("state.append(iteration)",)),
        ),),
        function_name="collect",
        parameters=("state", "count"),
    )
    state = []

    function(state, 4)

    assert state == [0, 1, 2, 3]
    assert function.__compiled_shell_source__.startswith("def collect")
