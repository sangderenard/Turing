from __future__ import annotations

import contextlib
import io

from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from src.common.dt_system.state_table import StateTable
from src.common.dt_system.dt_controller import STController, Targets
from src.common.dt_system.time_runtime import (
    ManagedTimeRuntime,
    TimeWindowRequest,
)
from src.computational_world import (
    BoundSpringParameters,
    ComputationalWorld,
    ComputationalWorldState,
    install_bound_spring,
)
from src.computational_world.spring import advance_bound_spring
from src.compiler.control_source import (
    CallBlock,
    ParallelDeployment,
    SequenceBlock,
    StateMachineTick,
    WhileBlock,
)
from src.compiler.precompile_to_ssa import lower_precompile_and_control_to_ssa
from src.compiler.wasm_class_coordinator import (
    build_class_inventory,
    emit_wasm_control_coordinator,
)
from src.compiler.wasm_class_modules import emit_control_region_modules


def test_canonical_bound_spring_step_reaches_compiled_region_ir():
    state = ComputationalWorldState.empty()
    parameters = BoundSpringParameters(
        c_repulse=0.0,
        growth_rate=0.0,
        relax_rate=0.0,
        cycle_period=0.1,
        boundary_radius=100.0,
        max_force=16.0,
        max_velocity=1.0,
        max_displacement=1.0,
    )
    install_bound_spring(
        state,
        ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0, 1),),
        parameters=parameters,
    )
    state.spring_position = type(state.spring_position).tensor(
        [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype="float32"
    )
    source = """
def page(state, dt, parameters):
    ok, metrics = advance_bound_spring(state, dt, parameters)
    return state.spring_position, state.spring_velocity
"""

    with contextlib.redirect_stdout(io.StringIO()):
        compilation = compile_ast_aot(
            source,
            "page",
            {"state": state, "dt": 0.01, "parameters": parameters},
            precompile_only=True,
            python_bindings={"advance_bound_spring": advance_bound_spring},
        )

    operations = {
        step.op_name
        for program in compilation.region_programs.values()
        for step in program.steps
    }
    assert compilation.function_outputs == ("result_0", "result_1")
    assert compilation.region_programs
    assert compilation.shell_control_program.region_indices
    assert "sext" in operations
    assert "maximum" in operations
    assert "zeros_like" not in operations
    assert "clamp" not in operations


def test_canonical_computational_world_step_reaches_compiled_region_ir():
    state = ComputationalWorldState.empty()
    parameters = BoundSpringParameters(
        c_repulse=0.0,
        growth_rate=0.0,
        relax_rate=0.0,
        max_force=16.0,
        max_velocity=1.0,
        max_displacement=1.0,
    )
    install_bound_spring(
        state,
        ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0, 1),),
        parameters=parameters,
    )
    world = ComputationalWorld(state, spring_parameters=parameters)
    source = """
def page(world, state, dt, state_table):
    ok, metrics, returned = advance_world(
        world, state, dt, state_table=state_table
    )
    return state.spring_position, state.spring_velocity
"""

    with contextlib.redirect_stdout(io.StringIO()):
        compilation = compile_ast_aot(
            source,
            "page",
            {
                "world": world,
                "state": state,
                "dt": 0.01,
                "state_table": StateTable(),
            },
            precompile_only=True,
            python_bindings={
                "advance_world": ComputationalWorld.advance_world
            },
        )

    assert compilation.function_outputs == ("result_0", "metrics", "state")
    assert compilation.region_programs


def test_managed_time_runtime_captures_real_world_spring_chain():
    state = ComputationalWorldState.empty()
    parameters = BoundSpringParameters(
        c_repulse=0.0,
        growth_rate=0.0,
        relax_rate=0.0,
        cycle_period=0.1,
        boundary_radius=100.0,
        max_force=16.0,
        max_velocity=1.0,
        max_displacement=1.0,
    )
    install_bound_spring(
        state,
        ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0, 1),),
        parameters=parameters,
    )
    world = ComputationalWorld(state, spring_parameters=parameters)
    state_table = StateTable()

    def world_advance(managed_state, dt):
        ok, metrics, returned = world.advance_world(
            managed_state, dt, state_table=state_table
        )
        assert returned is managed_state
        return ok, metrics

    runtime = ManagedTimeRuntime(
        state,
        world_advance,
        dx=1.0,
        targets=Targets(
            cfl=1.0,
            div_max=1.0,
            mass_max=1.0,
            error_limits={
                "world_sparse_shape": 0.0,
                "spring_causal_dt_excess": 0.0,
            },
        ),
        controller=STController(dt_min=1.0e-9),
    )
    request = TimeWindowRequest(
        request_id=0,
        generation=0,
        t_start=0.0,
        t_end=0.01,
        dt_initial=0.01,
    )
    source = """
def page(runtime, request):
    report = advance(runtime, request)
    return (
        runtime.state.managed_time,
        runtime.state.spring_position,
        runtime.state.spring_velocity,
    )
"""

    with contextlib.redirect_stdout(io.StringIO()):
        compilation = compile_ast_aot(
            source,
            "page",
            {"runtime": runtime, "request": request},
            precompile_only=True,
            python_bindings={"advance": ManagedTimeRuntime.advance},
        )

    program = getattr(
        compilation.compiled_shell_program,
        "program",
        compilation.compiled_shell_program,
    )
    assert len(program.steps) > 0
    assert compilation.control_shortfalls == ()
    def contains_while(block):
        if isinstance(block, WhileBlock):
            return True
        if isinstance(block, SequenceBlock):
            return any(contains_while(child) for child in block.blocks)
        if isinstance(block, StateMachineTick):
            return any(contains_while(body) for _case, body in block.cases) or (
                block.default is not None and contains_while(block.default)
            )
        if isinstance(block, ParallelDeployment):
            return any(contains_while(lane) for lane in block.lanes)
        if isinstance(block, CallBlock):
            return contains_while(block.callee)
        return False

    assert contains_while(compilation.shell_control_program.root)
    lowered = lower_precompile_and_control_to_ssa(
        compilation.compiled_shell_program,
        compilation.shell_control_program,
        region_programs=dict(compilation.region_programs),
        identity_table=compilation.identity_table,
        function_outputs=compilation.function_outputs,
        function_parameters=compilation.function_parameters,
    )
    assert not tuple(
        shortfall for shortfall in lowered.shortfalls
        if shortfall.domain == "control"
    )
    assert any(
        cycle.function == "planned_control"
        for cycle in lowered.cycles
    )
    modules, manifest = emit_control_region_modules(
        compilation.shell_control_program,
        compilation.region_programs,
        owner_name="managed_spring_page",
        module_dir="wasm",
        dtype="float64",
    )
    assert modules
    assert all(module.complete for module in modules.values())
    inventory = build_class_inventory(manifest)
    field_slots = {
        field.key: int(field.index) for field in inventory.fields
    }
    coordinator = emit_wasm_control_coordinator(
        inventory,
        compilation.shell_control_program,
        region_methods={
            int(entry["region_index"]): int(method.index)
            for entry, method in zip(manifest["modules"], inventory.methods)
        },
        value_slots={
            int(value_id): field_slots[key]
            for value_id, key in manifest.get("value_bindings", {}).items()
            if key in field_slots
        },
        region_signatures={
            int(region): (
                tuple(sorted(map(int, region_program.feeds))),
                tuple(map(int, region_program.outputs.values())),
            )
            for region, region_program in compilation.region_programs.items()
        },
        name="managed_spring_control",
    )
    assert coordinator.binary.startswith(b"\x00asm")
    assert state.managed_time.tolist() == [0.01]
    assert state.spring_position.tolist()[0][0] > -1.0
    assert state.spring_position.tolist()[1][0] < 1.0
    assert state.spring_velocity.tolist()[0][0] > 0.0
    assert state.spring_velocity.tolist()[1][0] < 0.0
