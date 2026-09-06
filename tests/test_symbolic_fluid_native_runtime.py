from src.common.dt_system.dt_controller import STController, Targets, run_superstep
from src.compiler.symbolic_fluid_dt import SymbolicFluidGridState
from src.compiler.symbolic_fluid_native_runtime import (
    compile_native_symbolic_fluid_step,
    load_symbolic_fluid_managed_functions,
)


def test_native_sympy_fluid_step_rejects_rolls_back_and_lands_on_frame(tmp_path):
    native = compile_native_symbolic_fluid_step(tmp_path)
    advance = load_symbolic_fluid_managed_functions(tmp_path)[
        "symbolic_fluid_advance"
    ]
    state = SymbolicFluidGridState.initial(4, 4)
    initial_mass = float(state.height.sum())
    attempts = []

    advanced, dt_next, metrics = run_superstep(
        state,
        0.2,
        0.2,
        state.dx,
        Targets(
            cfl=0.45,
            div_max=1.0,
            mass_max=1.0e-8,
            error_limits={"height_positivity": 0.0, "tracer_bounds": 0.0},
        ),
        STController(dt_min=1.0e-8, dt_max=0.2),
        advance,
        attempt_log=attempts,
    )

    assert native.artifact.complete
    assert float(advanced) == 0.2
    assert [(round(row["dt"], 12), row["accepted"]) for row in attempts] == [
        (0.2, False),
        (0.1, True),
        (0.1, True),
    ]
    assert 0.0 < float(dt_next) <= 0.2
    assert abs(float(state.height.sum()) - initial_mass) <= 1.0e-12
    assert metrics.mass_err <= 1.0e-15
    assert state.last_height_violation == 0.0
    assert state.last_tracer_violation == 0.0
