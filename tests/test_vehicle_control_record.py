"""The car game's controls are one declared record.

``VEHICLE_CONTROL_RECORD`` is the control ABI the worker hands the compiled
vehicle: ``control_abi`` is its names, every WGSL read of the resident
``controls`` buffer is an index derived from it, the worker builds the
buffer by name in that order, and the page's ``control_mapping`` is a view
of its semantic actions.  Nothing keeps a positional array by hand any more.
"""

import inspect
import re

from src.compiler import abstract_ui_vehicles as vehicles
from src.compiler import state_loop_deployment


def test_the_record_is_the_control_abi_and_names_its_shell_controls():
    names = [name for name, _control in vehicles.VEHICLE_CONTROL_RECORD]
    assert len(names) == 36 and len(set(names)) == 36
    assert names[:3] == ["throttle", "steering", "brake"]
    assert vehicles.VEHICLE_CONTROL_INDEX["tire_pressure_pa"] == 29
    assert vehicles.VEHICLE_CONTROL_INDEX["body_shell_active"] == 24
    assert vehicles.VEHICLE_CONTROL_INDEX["front_left_knuckle_angle"] == 25
    rows = vehicles.vehicle_control_record_data()
    by_name = {row["name"]: row for row in rows}
    assert by_name["throttle"]["control"] == "+move-forward -move-backward"
    assert by_name["steering"]["control"] == "+strafe-right -strafe-left"
    assert by_name["brake"]["control"] == "jump"
    assert by_name["forward_gear_ratio"]["source"] == "worker-state"
    assert [row["index"] for row in rows] == list(range(36))


def test_no_wgsl_reads_the_controls_buffer_by_a_hand_written_index():
    source = inspect.getsource(vehicles)
    literal = re.findall(r"controls\[\d+u", source)
    assert literal == [], literal
    # Every read is a record lookup or a placeholder the record fills.
    assert "controls[{VEHICLE_CONTROL_INDEX['throttle']}u]" in source
    assert "@@CONTROL_BODY_SHELL_ACTIVE@@" in source and "@@CONTROL_KNUCKLE_ANGLE_BASE@@" in source


def test_the_worker_fills_the_buffer_by_name_in_published_order():
    source = inspect.getsource(state_loop_deployment)
    start = source.index("function residentVehicleControlValues(body,gpu)")
    body = source[start:source.index("\n}\n", start)]
    assert "gpu.adapters.control_abi" in body
    assert "Float32Array.from(abi,name=>" in body
    assert "vehicle control record has no value for" in body
    for name, _control in vehicles.VEHICLE_CONTROL_RECORD:
        assert re.search(rf"\b{name}:", body), name
    assert source.count("residentVehicleControlValues(body,gpu)") == 3


def test_control_mapping_is_a_view_of_the_record():
    source = inspect.getsource(vehicles)
    assert '{**{control: name for name, control in VEHICLE_CONTROL_RECORD' in source
    assert '"control_record": vehicle_control_record_data()' in source


def test_axis_controls_follow_the_one_grammar_and_the_page_samples_them():
    from src.compiler import abstract_ui_div_map
    from src.compiler.abstract_ui_viewports import control_terms
    import pytest

    assert control_terms("jump") == ((1, "jump"),)
    assert control_terms("+move-forward -move-backward") == ((1, "move-forward"), (-1, "move-backward"))
    assert control_terms("+strafe-right -strafe-left") == ((1, "strafe-right"), (-1, "strafe-left"))
    with pytest.raises(ValueError):
        control_terms("+move-forward backward")       # a bare term inside an axis
    for name, control in vehicles.VEHICLE_CONTROL_RECORD:
        if control is not None:
            control_terms(control)                      # every declared hookup parses
    page = inspect.getsource(abstract_ui_div_map)
    assert "function vehicleControlValue(control, gamepad)" in page
    assert "for(const row of vehicleRuntime.active.control_record||[])" in page
    assert "controls[row.name]=mobileClamp(vehicleControlValue(row.control,gamepad))" in page
    assert "updateActiveVehicle(dt,throttle,Number(controls.steering||0),Number(controls.brake||0))" in page
    assert "const throttle=mobileClamp(forward*(running?1.18:1))" not in page
