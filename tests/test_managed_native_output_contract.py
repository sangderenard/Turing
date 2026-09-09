import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.compiler import vehicle_python_compilation as vehicle
from src.transmogrifier.ssa import BasicBlock, Function, Instr, IRModule, SSAValue


@pytest.mark.parametrize('field', ['dt_min', 'dt_max', 'dt_limit', 'other'])
def test_managed_feed_rejects_absence_without_optional_abi(field):
    argument = SSAValue(0, 'float64', accounting={
        'program_abi_parameter': 'controller', 'program_abi_field': field,
    })
    root = Function('root', [argument], {})
    lowered = vehicle.VehiclePythonSSALowering(IRModule({'root': root}), 'root', {}, ())
    controller = SimpleNamespace(**{field: None})
    with pytest.raises(ValueError, match='optional presence'):
        vehicle._managed_native_feeds_by_id(lowered, {'controller': controller})
    setattr(controller, field, 0.125)
    assert vehicle._managed_native_feeds_by_id(lowered, {'controller': controller}) == {0: 0.125}


@pytest.mark.parametrize("value", [None, 0.0, 0.125])
def test_managed_feed_packs_optional_presence_and_payload(value):
    payload = SSAValue(10, "float64", accounting={
        "program_abi_parameter": "controller",
        "program_abi_field": "dt_min",
        "program_abi_optional_payload": True,
    })
    presence = SSAValue(11, "bool", accounting={
        "program_abi_parameter": "controller",
        "program_abi_field": "dt_min",
        "program_abi_optional_presence": True,
        "program_abi_optional_present_when": True,
    })
    root = Function("root", [payload, presence], {})
    lowered = vehicle.VehiclePythonSSALowering(
        IRModule({"root": root}), "root", {}, (),
    )

    packed = vehicle._managed_native_feeds_by_id(
        lowered, {"controller": SimpleNamespace(dt_min=value)},
    )

    assert packed[11] is (value is not None)
    assert packed[10] == (0.0 if value is None else value)


def test_managed_feed_packs_linked_storage_for_absent_optional_field():
    payload = SSAValue(10, "float64", accounting={
        "program_abi_parameter": "controller",
        "program_abi_field": "dt_min",
        "program_abi_optional_payload": True,
    })
    presence = SSAValue(11, "bool", accounting={
        "program_abi_parameter": "controller",
        "program_abi_field": "dt_min",
        "program_abi_optional_presence": True,
        "program_abi_optional_present_when": True,
    })
    linked_storage = SSAValue(12, "float64", shape=(1,), accounting={
        "program_abi_parameter": "controller",
        "program_abi_field": "dt_min",
        "linked_parameter_provenance": "exact_receiver_field",
        "linked_call_frame_storage": "step",
    })
    root = Function("root", [payload, presence, linked_storage], {})
    lowered = vehicle.VehiclePythonSSALowering(
        IRModule({"root": root}), "root", {}, (),
    )

    packed = vehicle._managed_native_feeds_by_id(
        lowered, {"controller": SimpleNamespace(dt_min=None)},
    )

    assert packed == {10: 0.0, 11: False, 12: 0.0}


def test_managed_feed_initializes_private_linked_result_storage():
    workspace = SSAValue(20, "float64", shape=(2,), accounting={
        "program_abi_parameter": "value",
        "program_abi_field": "max_vel",
        "linked_call_frame_storage": "run_superstep",
        "callsite_id": 43,
    })
    root = Function("root", [workspace], {})
    lowered = vehicle.VehiclePythonSSALowering(
        IRModule({"root": root}), "root", {}, (),
    )

    packed = vehicle._managed_native_feeds_by_id(lowered, {})

    assert packed[20].dtype == np.dtype("float64")
    np.testing.assert_array_equal(packed[20], np.zeros((2,)))


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
