"""Exercise the authored advance prefix without initializing the tire solver."""

import ast
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np


def test_dually_advance_carries_vehicle_outputs_into_the_next_tick():
    source = Path(__file__).resolve().parents[1] / "tools/run_vehicle_native_assembly.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "_run_dually_python_profile")
    advance = next(n for n in entry.body if isinstance(n, ast.FunctionDef) and n.name == "advance")
    end = next(i for i, n in enumerate(advance.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "tire_state" for t in n.targets))
    prefix = ast.FunctionDef(name="advance", args=advance.args, body=advance.body[:end],
                             decorator_list=[])
    feedback = next(n for n in entry.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "state_feedback" for t in n.targets))
    state_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_DuallyDTState")
    code = compile(ast.fix_missing_locations(ast.Module(body=[feedback, state_class, prefix], type_ignores=[])),
                   str(source), "exec")
    vehicle_in, vehicle_out = [0.0, 0.1, 7.0], [0.0, 999.0]
    seen = []

    def tick(inputs, contact, fixture, outputs, **kwargs):
        seen.append(inputs[0])
        outputs[0] = inputs[0] + 2 * inputs[1]

    namespace = dict(
        material=SimpleNamespace(feeds={"tire_state": SimpleNamespace(data=np.zeros((1, 1, 1, 6)))},
                                 _data=lambda v: v.data, tick=tick, tire_critical_dt_s=1,
                                 _visual_lock=threading.Lock(), _visual_snapshot=None,
                                 _pending_visual_snapshot=None),
        vehicle_in=vehicle_in, vehicle_out=vehicle_out, contact_in=[], fixture_in=[0.1],
        vi={"position_x": 0, "dt": 1, "torque_command": 2}, fi={"dt": 0},
        accepted_clock=[0], tire_dt_fraction=1, tire_microstep_count=lambda *a: 1,
        time=SimpleNamespace(perf_counter=lambda: 0), _tick_profiler=None, tick_seconds=[0],
        output_names=("position_x_next", "torque_diagnostic"), np=np,
    )
    exec(code, namespace)
    state = namespace["_DuallyDTState"](namespace["material"], vehicle_in,
                                       namespace["contact_in"], namespace["fixture_in"], vehicle_out)
    saved = state.copy_shallow()
    namespace["advance"](None, 0.1)
    namespace["advance"](None, 0.1)
    np.testing.assert_allclose(seen, [0, 0.2])
    np.testing.assert_allclose(vehicle_in, [0.4, 0.1, 7])
    state.restore(saved)
    np.testing.assert_allclose(vehicle_in, [0, 0.1, 7])
    np.testing.assert_allclose(vehicle_out, [0, 999])
    namespace["advance"](None, 0.1)
    assert seen[-1] == 0
    np.testing.assert_allclose(vehicle_in, [0.2, 0.1, 7])
