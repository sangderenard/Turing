import ctypes
import subprocess
import sys

import pytest

from src.compiler.vehicle_native_assembly import (
    assembled_point_mass_properties,
    combine_c_function_artifacts,
    compile_brace_on_balance_c,
    compile_leveling_controller_c,
    compile_leveling_sensor_bank_c,
    compile_wheel_mesh_balance_c,
    load_vehicle_qualification_spec,
    native_vehicle_assembly_stages,
    negotiate_wheel_fixture,
    qualification_stage_policy,
)
from src.compiler.abstract_ui_dually_axle import roadside_dually_axle_assembly


def test_assembly_is_dependency_ordered_and_each_stage_is_gated():
    stages = native_vehicle_assembly_stages()
    assert [stage.identity for stage in stages] == [
        "clamp-pan", "engine-pan", "engine", "transmission",
        "transfer-and-differentials", "brace-on-balance", "pillar-hubs",
        "mount-tire-casings", "inflate-tires-on-pillars", "wheel-mesh-balance",
        "set-suspension-rest-pose", "front-linkages",
        "rear-linkages", "suspension-load-transfer", "armature-range-readiness",
        "rolling-start", "equipment",
        "accessory-installation", "post-accessory-ballast-balance",
        "leveling-controller-program-capture", "differential-wrench-proof",
        "destructive-drivetrain-pull",
        "release",
    ]
    assert all("energy_residual" in stage.solver_metrics for stage in stages)
    assert stages[0].corner_alphas == (0.0, 0.0, 0.0, 0.0)
    assert stages[-1].corner_alphas == (1.0, 1.0, 1.0, 1.0)
    assert stages[5].operation.startswith("solve and install density-sized")
    transfer = next(stage for stage in stages if stage.identity == "suspension-load-transfer")
    assert transfer.maximum_settle_seconds == 24.0
    rolling_start = next(stage for stage in stages if stage.identity == "rolling-start")
    assert rolling_start.maximum_settle_seconds == 16.0
    assert "engine_catch" in rolling_start.solver_metrics
    pillar = next(stage for stage in stages if stage.identity == "pillar-hubs")
    mount = next(stage for stage in stages if stage.identity == "mount-tire-casings")
    inflate = next(stage for stage in stages if stage.identity == "inflate-tires-on-pillars")
    assert "lower the rollers clear" in pillar.operation
    assert "rollers held down and clear" in mount.operation
    assert "roller-to-hub bead-capture distance" in inflate.operation
    assert "complete_bead_capture" in inflate.solver_metrics


def test_engine_dependent_validator_stages_are_optional():
    without_engine = native_vehicle_assembly_stages(enabled_systems=frozenset())
    identities = {stage.identity for stage in without_engine}
    assert "engine-pan" not in identities
    assert "engine" not in identities
    assert "rolling-start" not in identities
    assert "destructive-drivetrain-pull" not in identities
    assert "inflate-tires-on-pillars" in identities
    assert "differential-wrench-proof" in identities


def test_dually_fixture_is_negotiated_from_wheels_hubs_and_solid_axle():
    model = roadside_dually_axle_assembly(
        "validator:test", center_x=0.0, center_z=0.0).model
    plan = negotiate_wheel_fixture(model)
    assert len(plan.wheel_identities) == 4
    assert len(plan.structural_support_identities) == 4
    assert all(sum(row) == pytest.approx(1.0)
               for row in plan.wheel_to_structural_support)
    assert plan.wheel_to_structural_support[0] == pytest.approx(
        plan.wheel_to_structural_support[1])
    assert plan.wheel_to_structural_support[2] == pytest.approx(
        plan.wheel_to_structural_support[3])
    assert len(plan.pillars) == 4
    assert all(pillar.gravity_parallel_axis == (0.0, 1.0, 0.0)
               for pillar in plan.pillars)
    assert len(plan.tire_mounting_rollers) == 4
    assert {roller.kind for roller in plan.tire_mounting_rollers} == {
        "per-wheel-pair"}
    assert len(plan.articulated_dyno_rollers) == 2
    assert {roller.kind for roller in plan.articulated_dyno_rollers} == {
        "long-pair"}
    assert len(plan.rigid_axle_dyno_rollers) == 1
    assert plan.rigid_axle_dyno_rollers[0].kind == "axle-spanning-pair"
    assert plan.rigid_axle_dyno_rollers[0].wheel_identities == plan.wheel_identities


def test_fixture_negotiation_accepts_zero_independent_and_tracked_wheels():
    empty = negotiate_wheel_fixture({"wheels": []})
    assert empty.pillars == ()
    assert empty.tire_mounting_rollers == ()
    tracked = negotiate_wheel_fixture({
        "wheels": [
            {"identity": "idler", "hub_identity": "idler-hub"},
            {"identity": "spare", "hub_identity": "spare-hub"},
        ],
        "running_gear": {"track_installed": True},
    })
    assert len(tracked.pillars) == 2
    assert tracked.post_track_surface == "ground-projection-no-roller"
    assert all(roller.kind == "per-wheel-pair"
               for roller in tracked.articulated_dyno_rollers)


def test_producer_neutral_qualification_spec_makes_load_transfer_observable():
    spec = load_vehicle_qualification_spec()
    policy = qualification_stage_policy(spec, "suspension-load-transfer")
    observation_seconds = policy["window_samples"] / spec["observation"]["sample_hz"]
    additional_windows_seconds = ((policy["required_stable_windows"] - 1)
                                  * policy["evaluation_stride_samples"]
                                  / spec["observation"]["sample_hz"])
    assert policy["minimum_seconds"] / 2 >= observation_seconds + additional_windows_seconds
    contact = spec["contact_tolerances"]
    assert contact["minimum_clamped_corner_load_fraction"] < 0.01
    assert contact["minimum_released_total_weight_fraction"] >= 0.80
    assert spec["leveling_tolerances"]["minimum_supported_corner_fraction"] == 1.0
    assert spec["leveling_tolerances"]["maximum_corner_pose_error_m"] == 0.0005


def test_wheel_mesh_balance_is_runtime_compiled_and_cancels_radial_first_moment(tmp_path):
    artifact = compile_wheel_mesh_balance_c()
    assert artifact.complete
    source = tmp_path / "wheel_balance.c"
    library = tmp_path / "wheel_balance.dll"
    source.write_text(artifact.source, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "ziglang", "cc", "-shared", "-O2",
         str(source), "-o", str(library)], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    function = ctypes.CDLL(str(library)).vehicle_wheel_mesh_balance
    function.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    values = dict(mesh_mass=18.0, mesh_first_moment_x=.08, mesh_first_moment_y=-.03,
                  mesh_polar_inertia=1.7, ballast_radius=.21, ballast_density=11340.0,
                  ballast_axial_width=.025, ballast_radial_depth=.018,
                  maximum_ballast_thickness=.100)
    inputs = (ctypes.c_double * len(artifact.input_names))(
        *(values[name] for name in artifact.input_names))
    outputs = (ctypes.c_double * len(artifact.output_names))()
    function(inputs, outputs)
    result = dict(zip(artifact.output_names, outputs))
    assert result["corrected_first_moment_x"] == pytest.approx(0.0, abs=1e-12)
    assert result["corrected_first_moment_y"] == pytest.approx(0.0, abs=1e-12)
    assert result["ballast_mass"] > 0.0
    assert result["fit_margin"] > 0.0


def test_brace_on_balance_is_compiler_emitted_and_cancels_both_moments(tmp_path):
    artifact = compile_brace_on_balance_c()
    assert artifact.complete
    source = tmp_path / "balance.c"
    library = tmp_path / "balance.dll"
    source.write_text(artifact.source, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "ziglang", "cc", "-shared", "-O2",
         str(source), "-o", str(library)], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    dll = ctypes.CDLL(str(library))
    function = dll.vehicle_brace_on_balance
    function.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    values = dict(moment_x=120.0, moment_z=-32.0, half_length=.62,
                  half_width=.3276, density=11340.0)
    inputs = (ctypes.c_double * len(artifact.input_names))(
        *(values[name] for name in artifact.input_names))
    outputs = (ctypes.c_double * len(artifact.output_names))()
    function(inputs, outputs)
    result = dict(zip(artifact.output_names, outputs))
    assert result["corrected_moment_x"] == pytest.approx(0.0, abs=1e-10)
    assert result["corrected_moment_z"] == pytest.approx(0.0, abs=1e-10)
    assert all(result[name] >= 0 for name in result if name.startswith("mass_"))


def test_leveling_controller_is_runtime_compiled_and_load_aware():
    artifact = compile_leveling_controller_c()
    assert artifact.complete
    assert set(artifact.output_names) >= {
        *(f"command_{corner}" for corner in (
            "front_left", "front_right", "rear_left", "rear_right")),
    }
    source = artifact.source
    assert "opposing_force_front_left" in artifact.input_names
    assert "measured_pose_error_front_left" in artifact.input_names
    assert "previous_correction_front_left" in artifact.input_names
    assert "correction_front_left_next" in artifact.output_names
    assert "trim_front_left_next" in artifact.output_names
    assert "hydraulic_force_capacity" in artifact.output_names
    assert "cross_weight_error" in artifact.output_names
    assert "vehicle_leveling_controller" in source


def test_leveling_observations_are_compiled_massless_bounded_signal_state(tmp_path):
    artifact = compile_leveling_sensor_bank_c()
    assert artifact.complete
    assert "truth_force_front_left" in artifact.input_names
    assert "previous_force_front_left" in artifact.input_names
    assert "observed_force_front_left" in artifact.output_names
    assert "maximum_normalized_residual" in artifact.output_names
    source = tmp_path / "leveling_sensors.c"
    library = tmp_path / "leveling_sensors.dll"
    source.write_text(artifact.source, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "ziglang", "cc", "-shared", "-O2",
         str(source), "-o", str(library)], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    function = ctypes.CDLL(str(library)).vehicle_leveling_sensor_bank
    function.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    values = {name: 0.0 for name in artifact.input_names}
    values.update({
        "dt": 1 / 1024, "force_bandwidth_hz": 90.0,
        "pose_bandwidth_hz": 120.0, "pressure_bandwidth_hz": 40.0,
        "motion_bandwidth_hz": 80.0, "force_range_n": 50000.0,
        "pose_range_m": .75, "pressure_range_pa": 1_500_000.0,
        "motion_range_m_s": 20.0, "truth_force_front_left": 1000.0,
    })
    inputs = (ctypes.c_double * len(artifact.input_names))(
        *(values[name] for name in artifact.input_names))
    outputs = (ctypes.c_double * len(artifact.output_names))()
    function(inputs, outputs)
    result = dict(zip(artifact.output_names, outputs))
    assert 0.0 < result["observed_force_front_left"] < 1000.0
    assert result["residual_force_front_left"] == pytest.approx(
        1000.0 - result["observed_force_front_left"])
    assert result["maximum_normalized_residual"] > 0.0


def test_leveling_controller_and_observer_share_one_c_prelude():
    source = combine_c_function_artifacts(
        compile_leveling_controller_c(), compile_leveling_sensor_bank_c())
    assert source.count("static long long turing_imod") == 1
    assert source.count("TURING_EXPORT void vehicle_leveling_controller") == 1
    assert source.count("TURING_EXPORT void vehicle_leveling_sensor_bank") == 1


def test_leveling_controller_bounds_grounded_motion_and_freezes_force_hunt_in_fall(tmp_path):
    artifact = compile_leveling_controller_c()
    source = tmp_path / "leveling.c"
    library = tmp_path / "leveling.dll"
    source.write_text(artifact.source, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "ziglang", "cc", "-shared", "-O2",
         str(source), "-o", str(library)], capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    function = ctypes.CDLL(str(library)).vehicle_leveling_controller
    function.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    values = {name: 0.0 for name in artifact.input_names}
    values.update({
        "corner_stiffness": 26000.0, "maximum_offset": .62, "dt": 1 / 1024,
        "pose_feedback_gain": 2.5, "trim_feedback_gain": 9.0,
        "calibrated_heave_gain": 1.0, "calibrated_roll_gain": 1.0,
        "calibrated_pitch_gain": 1.0, "calibrated_cross_weight_gain": .72,
        "hydraulic_pressure": 12e6, "piston_area": .0018, "maximum_flow": .00042,
        "hydraulic_efficiency": .88, "pressure_force_reserve_fraction": .18,
        "coarse_rate": .055, "trim_rate": .22, "trim_stroke": .014,
        "trim_entry_error": .020, "support_fraction": 1.0,
        "minimum_grounded_support_fraction": .65, "fall_velocity_threshold": .35,
        "fall_velocity_blend": 1.0, "fall_policy_selector": 1.0,
        "landing_ready_corner_offset": -.045, "unloaded_placement_rate": .08,
        "round_robin_corner": 0.0, "measured_pose_error_front_left": .10,
    })

    def run(inputs):
        packed = (ctypes.c_double * len(artifact.input_names))(
            *(inputs[name] for name in artifact.input_names))
        outputs = (ctypes.c_double * len(artifact.output_names))()
        function(packed, outputs)
        return dict(zip(artifact.output_names, outputs))

    grounded = run(values)
    assert grounded["support_authority"] == pytest.approx(1.0)
    assert 0 < grounded["correction_front_left_next"] <= .055 / 1024 + 1e-12
    assert grounded["hydraulic_force_capacity"] == pytest.approx(12e6 * .0018 * .88)

    falling_values = dict(values)
    falling_values.update({
        "support_fraction": 0.0, "chassis_vertical_velocity": -2.0,
        "fall_policy_selector": 0.0, "previous_correction_front_left": .05,
    })
    falling = run(falling_values)
    assert falling["falling_weight"] > 0.99
    assert falling["correction_front_left_next"] == pytest.approx(.05)

    landing_ready_values = dict(falling_values)
    landing_ready_values["fall_policy_selector"] = 1.0
    landing_ready = run(landing_ready_values)
    assert landing_ready["command_front_left"] < falling["command_front_left"]

    terrain_values = dict(falling_values)
    terrain_values.update({
        "fall_policy_selector": 2.0, "predicted_landing_offset_front_left": .20,
    })
    terrain = run(terrain_values)
    assert terrain["command_front_left"] > falling["command_front_left"]

    trim_values = dict(values)
    trim_values["measured_pose_error_front_left"] = .01
    trimmed = run(trim_values)
    assert trimmed["trim_front_left_next"] > 0.0
    assert trimmed["trim_front_right_next"] == pytest.approx(0.0, abs=1e-12)


def test_installed_component_mass_reduction_changes_live_com_and_inertia():
    first = {"identity": "pan", "mass_kg": 10.0, "local_position": [0.0, 0.0, 0.0]}
    engine = {"identity": "engine", "mass_kg": 20.0, "local_position": [1.0, 0.0, 0.0]}
    pan = assembled_point_mass_properties((first,))
    combined = assembled_point_mass_properties((first, engine))
    assert pan["mass_kg"] == 10.0
    assert combined["mass_kg"] == 30.0
    assert combined["center_of_mass"][0] == pytest.approx(2 / 3)
    assert combined["inertia_kg_m2"]["yaw"] > pan["inertia_kg_m2"]["yaw"]
