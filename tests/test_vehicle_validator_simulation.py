"""Compare the compiled entry's authored source with the live validator tick."""

import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.common.tensors import AbstractTensor
from src.common.dt_system.dt_scaler import coerce_metrics
from src.compiler.vehicle_validator_simulation import (
    STATE_FIELDS, eager_functions, load_validator_state, simulation_inputs,
)


@pytest.fixture(scope="module")
def initialized_validator():
    import tools.run_vehicle_native_assembly as runner

    path = Path(runner.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_run_dually_python_profile")
    prefix = []
    for statement in entry.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "state" for target in statement.targets
        ):
            break
        prefix.append(statement)
    entry.body = prefix + [ast.parse(
        "return material, vehicle_in, contact_in, fixture_in, vehicle_out, profile").body[0]]
    ast.fix_missing_locations(entry)
    namespace = dict(vars(runner))
    exec(compile(ast.Module(body=[entry], type_ignores=[]), str(path), "exec"), namespace)
    artifact = os.environ.get("TURING_VALIDATOR_NATIVE_ARTIFACT")
    lanes = (json.loads((Path(artifact) / "manifest.json").read_text())["batch_size"]
             if artifact else 1)
    args = SimpleNamespace(python_material=True, rate_hz=1024, lanes=lanes,
                           dt_rollback_threshold_multiplier=2.0, tire_fidelity_mode="fine")
    return namespace[entry.name](args, Path("unused"))


@pytest.mark.parametrize("tire_active", [False, True])
def test_simulation_advance_matches_actual_validator_tick_and_feedback(initialized_validator, tire_active):
    live, vehicle, contact, fixture, output, profile = initialized_validator
    if tire_active:
        for wheel in range(4):
            live.set_tire_assembly(wheel, 1.0)
    inputs = simulation_inputs(live.lanes)
    state = load_validator_state(inputs, live, vehicle, contact, fixture, output)
    advance, _window = eager_functions(inputs)
    dt = min(2.0 ** -20, live.tire_critical_dt_s * 0.1)
    before = live._data(live.feeds["tire_state"])[0].copy()
    with AbstractTensor.use_backend("numpy"):
        physical, metrics = advance(state, dt)
    metrics = coerce_metrics(metrics)
    vehicle[live.vi["dt"]] = fixture[live.fi["dt"]] = dt
    live.feeds["microstep_count"] = 1
    live.tick(vehicle, contact, fixture, output, publish_visual=False)
    for out_index, name in enumerate(live.output_names):
        if name.endswith("_next") and name[:-5] in live.vi:
            vehicle[live.vi[name[:-5]]] = output[out_index]
    for field, expected in (("vehicle_in", vehicle), ("contact_in", contact),
                            ("fixture_in", fixture), ("vehicle_out", output)):
        np.testing.assert_allclose(live._data(getattr(state, field)), tuple(expected),
                                   rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=field)
    for name in inputs.prepared.feeds:
        if hasattr(live.feeds[name], "data"):
            np.testing.assert_allclose(live._data(getattr(state, name)), live._data(live.feeds[name]),
                                       rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=name)
    after = live._data(live.feeds["tire_state"])[0]
    displacement = np.linalg.norm(after[..., :3] - before[..., :3], axis=-1).max()
    assert float(metrics.error_channels["maximum_substep_displacement_m"]) == pytest.approx(displacement)
    assert bool(physical)
    if tire_active:
        assert np.any(after[..., :3] != 0.0)
        assert np.max(live._data(live.feeds["tire_output"])[..., 6]) > 0.0
    assert np.asarray(live._data(state.rig_reactions)).shape[-1] == 6
    assert np.asarray(live._data(state.fixture_output)).shape[-1] == 5


def test_simulation_snapshot_restores_all_persistent_publications():
    from src.compiler.vehicle_validator_simulation import ValidatorSimulationState

    state = ValidatorSimulationState(**{name: np.array([i], dtype=float)
                                        for i, name in enumerate(STATE_FIELDS)})
    saved = state.copy_shallow()
    for name in STATE_FIELDS:
        getattr(state, name)[:] = -100.0
    state.restore(saved)
    for i, name in enumerate(STATE_FIELDS):
        np.testing.assert_array_equal(getattr(state, name), [float(i)])


@pytest.mark.skipif(not os.environ.get("TURING_VALIDATOR_NATIVE_ARTIFACT"),
                    reason="requires an explicitly selected compiled simulation artifact")
def test_native_windows_preserve_physical_and_controller_state(initialized_validator):
    from src.compiler.vehicle_validator_simulation import NativeValidatorSimulation

    live, vehicle, contact, fixture, output, _profile = initialized_validator
    for wheel in range(4):
        live.set_tire_assembly(wheel, 1.0)
    # Start the comparison from real, initialized tires, after their birth
    # publication has been handled by the existing validator.
    vehicle[live.vi["dt"]] = fixture[live.fi["dt"]] = 2.0 ** -20
    live.feeds["microstep_count"] = 1
    live.tick(vehicle, contact, fixture, output, publish_visual=False)
    for out_index, name in enumerate(live.output_names):
        if name.endswith("_next") and name[:-5] in live.vi:
            vehicle[live.vi[name[:-5]]] = output[out_index]
    native = NativeValidatorSimulation(os.environ["TURING_VALIDATOR_NATIVE_ARTIFACT"], live)
    inputs = simulation_inputs(live.lanes)
    eager_state = load_validator_state(inputs, live, vehicle, contact, fixture, output)
    _advance, eager_window = eager_functions(inputs)
    native_controller = copy.deepcopy(inputs.controller)
    next_dt = 2.0 ** -20
    for _ in range(2):
        with AbstractTensor.use_backend("numpy"):
            expected = eager_window(eager_state, inputs.targets, inputs.controller,
                                    2.0 ** -19, next_dt)
        actual = native.step(2.0 ** -19, next_dt, native_controller, inputs.targets,
                             vehicle, contact, fixture, output)
        assert actual.advanced == pytest.approx(expected[-3], rel=1e-8, abs=1e-10)
        assert actual.dt_next == pytest.approx(expected[-2], rel=1e-8, abs=1e-10)
        for field in STATE_FIELDS:
            np.testing.assert_allclose(live._data(getattr(native.inputs.material, field)),
                                       live._data(getattr(eager_state, field)),
                                       rtol=1e-8, atol=1e-10, equal_nan=False, err_msg=field)
        for name, expected_value in vars(inputs.controller).items():
            actual_value = getattr(native_controller, name)
            if isinstance(expected_value, (int, float, AbstractTensor)):
                np.testing.assert_allclose(live._data(actual_value), live._data(expected_value),
                                           rtol=1e-8, atol=1e-10, err_msg=name)
            else:
                assert actual_value == expected_value, name
        next_dt = actual.dt_next
