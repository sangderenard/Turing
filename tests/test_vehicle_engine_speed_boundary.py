"""Exact stopped-crank boundary from the authored engine integration chain."""
import ast
import inspect
import math

import pytest
import sympy

from src.compiler import abstract_ui_vehicles as vehicle


def authored_speed_step(speed, net_torque, dt=1/360, redline=4400*math.pi/30, alpha=1):
    # Execute only the original integration assignments, without copying their
    # equations into the test or invoking an unrelated full compiler build.
    tree = ast.parse(inspect.getsource(vehicle._symbolic_vehicle_equations_authored))
    targets = {'effective_engine_inertia', 'engine_angular_acceleration',
               'engine_angular_speed_raw', 'engine_angular_speed_next'}
    statements = [node for node in tree.body[0].body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id in targets for t in node.targets)]
    assert len(statements) == len(targets)
    state = dict(vehicle.load_default_car_configuration().parameter_defaults())
    state.update(engine_angular_speed=speed, engine_redline_angular_speed=redline,
                 assembly_alpha_drivetrain=alpha)
    scope = {'s': state, 'dt': dt, 'engine_acceleration_torque': net_torque,
             'sympy': sympy, '_hard_clamp': vehicle._hard_clamp,
             '_c2_clamp': vehicle._c2_clamp}
    exec(compile(ast.Module(body=statements, type_ignores=[]), '<authored-engine-step>', 'exec'), scope)
    return float(scope['engine_angular_speed_raw']), float(scope['engine_angular_speed_next']), float(scope['effective_engine_inertia'])


@pytest.mark.parametrize('raw', [-1000, -.01, 0, .01, 50, 650*math.pi/30, 450, 1000])
def test_authored_speed_boundary_is_exact_and_cannot_add_energy(raw):
    before, after, inertia = authored_speed_step(raw, 0)
    ceiling = 4400*math.pi/30*1.03
    assert before == raw
    assert after == pytest.approx(min(ceiling, max(0, raw)), abs=1e-12)
    assert 0 <= after <= ceiling
    assert .5*inertia*after**2 <= .5*inertia*raw**2 + 1e-12
    if raw == 0:
        assert after == 0


def test_authored_powered_crank_retains_interior_torque_integration():
    idle = 650*math.pi/30
    raw, after, inertia = authored_speed_step(idle, 100)
    assert raw == pytest.approx(idle + (1/360)*100/inertia)
    assert after == raw
    assert after > idle
    raw, after, _ = authored_speed_step(idle, -100)
    assert 0 < after < idle
    assert after == raw
    assert authored_speed_step(0, 100)[1] > 0  # Actual applied starter/drive torque can start a crank.
    assert authored_speed_step(0, 0)[1] == 0
    assert authored_speed_step(idle, 100, alpha=0)[1] == 0
