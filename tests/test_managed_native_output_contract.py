import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.compiler import vehicle_python_compilation as vehicle
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue


@pytest.mark.parametrize('missing_input', [False, True])
def test_managed_wrapper_allocates_returns_but_never_missing_inputs(tmp_path, monkeypatch, missing_input):
    argument = SSAValue(0, 'float64')
    result = SSAValue(1, 'float64')
    root = Function('root', [argument], {'entry': BasicBlock('entry', [
        Instr('Ret', [result], None),
    ])}, metadata={'named_outputs': (('advanced', 1),)})
    lowered = vehicle.VehiclePythonSSALowering(IRModule({'root': root}), 'root', {}, ())
    captured = {}

    def compile_standalone(directory, feeds, **kwargs):
        captured.update(feeds)
        return SimpleNamespace(directory=tmp_path)

    artifact = SimpleNamespace(name='test', buffer_order=(0, 1),
                               buffer_dtypes=('float64', 'float64'),
                               buffer_shapes=((), (3, 5)), compile_standalone=compile_standalone)
    monkeypatch.setattr(vehicle, 'balloon_tire_managed_python_compilation_inputs',
                        lambda *a, **kw: SimpleNamespace(feeds={}))
    monkeypatch.setattr(vehicle, 'emit_balloon_tire_managed_python_c', lambda **kw: (lowered, artifact))
    monkeypatch.setattr(vehicle, '_managed_native_feeds_by_id',
                        lambda *a: {} if missing_input else {0: np.asarray(2.0)})
    if missing_input:
        with pytest.raises(RuntimeError, match='unnamed public buffers: \\(0,\\)'):
            vehicle.compile_balloon_tire_managed_python_native(tmp_path)
        assert not captured
    else:
        vehicle.compile_balloon_tire_managed_python_native(tmp_path)
        assert captured[0] == 2.0
        assert captured[1].shape == () and captured[1].item() == 0.0
        manifest = json.loads((tmp_path / 'balloon_tire_managed.manifest.json').read_text())
        output = manifest['buffers'][1]
        assert (output['name'], output['role'], output['return_index']) == ('advanced', 'output', 0)
