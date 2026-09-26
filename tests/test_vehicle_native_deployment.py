from __future__ import annotations

import math
import pytest
import sympy

from src.compiler.vehicle_native_deployment import (
    HOLDER_MODES,
    _mechanical_edge_endpoints,
    split_symbolic_constants_to_double_double,
    derive_vehicle_rig_rate_hz,
    compile_vehicle_roller_fixture_c,
    emit_double_double_c,
    render_native_vehicle_tick_shell,
    render_native_scientific_viewer_shell,
    scientific_visualization_abi,
)
from src.compiler.vehicle_balloon_tire_native import NativeBalloonTireAssembly


def test_default_rig_rate_is_derived_from_physical_bandwidth():
    config = load_default_car_configuration()
    assert derive_vehicle_rig_rate_hz(config) == 1024
from src.compiler.ssa_c_backend import CFunctionArtifact
from src.compiler.ssa_c_backend import emit_ssa_function_to_c
from src.compiler.ssa_llvm_backend import emit_ssa_function_to_llvm
from src.compiler.abstract_ui_vehicles import (
    compile_torus_plane_contact_arc_ssa,
    load_default_car_configuration,
    symbolic_torus_plane_patch_boundary_integral,
)
from src.compiler.symbolic_equation_compiler import (
    SymbolicPublication,
    compile_sympy_equations,
)


def test_dually_python_validator_keeps_the_canonical_preallocated_batch_axis():
    import inspect

    from src.compiler.vehicle_native_graph_program import BATCH_CAPACITY
    from src.compiler.vehicle_python_compilation import (
        dually_vehicle_python_compilation_inputs,
    )

    parameter = inspect.signature(
        dually_vehicle_python_compilation_inputs
    ).parameters["batch_size"]
    assert parameter.default == BATCH_CAPACITY


def test_native_deployment_accepts_legacy_and_general_mechanical_edge_schemas():
    assert _mechanical_edge_endpoints({"a": "frame", "b": "hub"}) == (
        "frame", "hub")
    assert _mechanical_edge_endpoints({"nodes": ["casing", "bearing"]}) == (
        "casing", "bearing")


def _fixture_input(artifact, **updates):
    values = {name: 0.0 for name in artifact.input_names}
    values.update({
        "dt": 1.0 / 300.0,
        "gravity": -9.81,
        "floor_y": 0.0,
        "carriage_mass": 12.0,
        "neutral_buoyancy": 1.0,
        "passive_damping": 8.0,
        "lock_stiffness": 24000.0,
        "lock_damping": 1200.0,
        "maximum_actuator_force": 18000.0,
    })
    for corner in ("front_left", "front_right", "rear_left", "rear_right"):
        values[f"hub_y_{corner}"] = 1.0
        values[f"carriage_y_{corner}"] = 0.25
        values[f"command_y_{corner}"] = 0.25
    values.update(updates)
    return values


def test_cage_drive_fixture_is_unilateral_and_cannot_pull_departing_hub(tmp_path):
    artifact = compile_vehicle_roller_fixture_c().compile(tmp_path)
    output = dict(zip(artifact.output_names, artifact.run(_fixture_input(
        artifact,
        mode=HOLDER_MODES["cage-drive"],
        hub_velocity_y_front_left=2.0,
        carriage_velocity_y_front_left=0.0,
    ))))
    assert output["fixture_hub_force_front_left"] == pytest.approx(0.0)
    assert output["fixture_compensation_force_front_left"] == pytest.approx(12.0 * 9.81)


def test_suspension_test_fixture_lock_is_bidirectional(tmp_path):
    artifact = compile_vehicle_roller_fixture_c().compile(tmp_path)
    upward = dict(zip(artifact.output_names, artifact.run(_fixture_input(
        artifact,
        mode=HOLDER_MODES["suspension-test"],
        command_y_front_left=0.35,
    ))))
    downward = dict(zip(artifact.output_names, artifact.run(_fixture_input(
        artifact,
        mode=HOLDER_MODES["suspension-test"],
        command_y_front_left=0.15,
    ))))
    assert upward["fixture_actuator_force_front_left"] > 0.0
    assert downward["fixture_actuator_force_front_left"] < 0.0
    assert upward["fixture_hub_force_front_left"] == pytest.approx(0.0)
    assert downward["fixture_hub_force_front_left"] == pytest.approx(0.0)


def test_double_double_c_abi_preserves_both_lanes_across_ticks(tmp_path):
    x, y = sympy.symbols("x y", real=True)
    compilation = compile_sympy_equations(
        (sympy.Eq(sympy.Symbol("next_state"), x * y + x / y, evaluate=False),),
        name="persistent_lane_probe",
        publications=(SymbolicPublication("next_state", "probe.next_state"),),
    )
    artifact = emit_double_double_c(
        compilation, entry_name="persistent_lane_probe_dd",
    ).compile(tmp_path)
    assert artifact.input_names == ("x", "y", "x__limb1", "y__limb1")
    assert artifact.output_names == ("next_state__hi", "next_state__limb1")
    high, low = artifact.run((1.0, 10.0, 2.0 ** -54, 0.0))
    assert low != 0.0
    assert high + low == pytest.approx((1.0 + 2.0 ** -54) * 10.1)


def test_double_double_c_abi_can_publish_a_passthrough_argument(tmp_path):
    x = sympy.Symbol("x", real=True)
    compilation = compile_sympy_equations(
        (sympy.Eq(sympy.Symbol("same"), x, evaluate=False),
         sympy.Eq(sympy.Symbol("worked"), x * 2, evaluate=False)),
        name="persistent_lane_passthrough",
        publications=(SymbolicPublication("same", "probe.same"),
                      SymbolicPublication("worked", "probe.worked")),
    )
    artifact = emit_double_double_c(
        compilation, entry_name="persistent_lane_passthrough_dd",
    ).compile(tmp_path)
    result = artifact.run((3.5, 2.0 ** -52))
    assert result[:2] == pytest.approx((3.5, 2.0 ** -52))


def test_native_tick_shell_calls_balloon_appendage_and_never_legacy_contact():
    corners = ("front_left", "front_right", "rear_left", "rear_right")
    vehicle_inputs = (
        *(f"contact_normal_force_{corner}" for corner in corners),
        *(f"longitudinal_force_{corner}" for corner in corners),
        *(f"tire_reaction_torque_{corner}" for corner in corners),
        *(f"compression_{corner}" for corner in corners),
        *(f"compression_velocity_{corner}" for corner in corners),
        *(f"wheel_omega_{corner}" for corner in corners),
        *(f"wheel_angle_{corner}" for corner in corners),
        *(f"material_plastic_set_{corner}" for corner in corners),
        *(f"material_survival_{corner}" for corner in corners),
        "dt",
        "position_x", "position_y", "position_z", "velocity_y",
        "velocity_x", "velocity_z", "roll", "pitch", "yaw",
        "roll_velocity", "pitch_velocity", "yaw_velocity",
        "assembly_alpha_drivetrain",
        *(f"assembly_alpha_{corner}" for corner in corners),
        *(f"contact_wrench_force_{axis}" for axis in "xyz"),
        *(f"contact_wrench_torque_{axis}" for axis in "xyz"),
    )
    contact_inputs = (
        *(f"normal_{axis}" for axis in "xyz"),
        *(f"forward_{axis}" for axis in "xyz"),
        *(f"attachment_{axis}" for axis in "xyz"),
        "support", "tire_radial_compression", "tire_radial_velocity",
        "tire_major_radius", "tire_section_radius",
    )
    fixture_inputs = (
        *(f"roller_reaction_{corner}" for corner in corners),
        *(f"hub_y_{corner}" for corner in corners),
        *(f"hub_velocity_y_{corner}" for corner in corners),
        *(f"carriage_y_{corner}" for corner in corners),
        *(f"carriage_velocity_y_{corner}" for corner in corners),
            "surface_mode", "terrain_phase_x", "terrain_phase_z",
            "terrain_period_x", "terrain_period_z", "terrain_velocity_x",
            "terrain_velocity_z", "lock_stiffness", "lock_damping",
            "maximum_actuator_force",
    )
    fixture_outputs = tuple(f"fixture_hub_force_{corner}" for corner in corners)
    source = render_native_vehicle_tick_shell(
        CFunctionArtifact("abstract_ui_vehicle_step", "", vehicle_inputs, (), ()),
        CFunctionArtifact("abstract_ui_wheel_contact", "", contact_inputs,
                          ("chassis_force_x", "chassis_force_y", "chassis_force_z"), ()),
        CFunctionArtifact("vehicle_roller_fixture_step", "", fixture_inputs, fixture_outputs, ()),
        NativeBalloonTireAssembly(
            "balloon_tire_appendage_step", "",
                 ("dt", "gravity_y", "vertex_mass_kg", *(f"{corner}.hub_position_{axis}" for corner in corners for axis in "xyz"),
                 *(f"{corner}.hub_velocity_{axis}" for corner in corners for axis in "xyz"),
                 *(f"{corner}.hub_basis_{local}_{world}" for corner in corners for local in "xyz" for world in "xyz"),
                 *(f"{corner}.hub_angular_velocity_{axis}" for corner in corners for axis in "xyz"),
                 *(f"{corner}.hub_angle_rad" for corner in corners),
             *(f"{corner}.hub_angular_velocity_z" for corner in corners),
             *(f"{corner}.surface_kind" for corner in corners),
             *(f"{corner}.cylinder_radius_m" for corner in corners),
             *(f"{corner}.plane_count" for corner in corners),
             *(f"{corner}.plane_{plane}_{quantity}_{axis}" for corner in corners for plane in range(2) for quantity in ("point", "normal", "velocity") for axis in "xyz")),
            tuple(f"{corner}.{name}" for corner in corners for name in (
                "rim_force_x_n", "rim_force_y_n", "rim_force_z_n",
                "rim_moment_x_nm", "rim_moment_y_nm", "rim_moment_z_nm",
                "gas_pressure_pa", "volume_ratio", "gas_temperature_k",
                    "contact_count", "minimum_skin_y_m", "strain_energy_j",
                    "dissipation_power_w", "bending_energy_j")),
            3072, 128, 256,
        ),
    )
    assert "abstract_ui_wheel_contact(" not in source
    assert "torus_plane_contact_arc(" not in source
    assert source.count("balloon_tire_appendage_step(tire_in,tire_state,tire_out);") == 1
    assert "outer_dt/48.0" in source
    assert "for(int micro=0;micro<48;++micro)" in source
    assert "wrench_sum[6*w+a]/48.0" in source
    assert "contact_peak[w]=fmax" in source
    assert "tire_plane_previous[w][p][a]+alpha*" in source
    assert "vehicle_periodic_terrain_plane" in source
    assert "abstract_ui_vehicle_step(vehicle_in, vehicle_out);" in source
    assert "vehicle_member_material_step(mi_,mo_);" in source
    assert "vehicle_native_material_state" in source
    assert "vehicle_native_material_diagnostics" in source
    assert "vehicle_native_material_step(vehicle_in);" in source
    assert "vehicle_roller_fixture_step(fixture_in, fixture_out);" in source
    assert "torque[2]+=ax*residual_y-ay*wheel_fx;" in source
    assert "torque[2]+=ax*residual_y-ay*wheel_fx+tire_out[oo+5]" not in source
    assert "present*tire_out[oo+3]" in source
    assert "present*tire_out[oo+4]" in source
    assert "vehicle_native_apply_rig_points(vehicle_in,force,torque);" in source
    assert "vehicle_native_rig_point_configure" in source
    assert "vehicle_native_rig_point_reactions" in source
    assert "vehicle_native_pillar_reactions" in source
    assert "const double hub_x=articulated_hub[0]" in source
    assert "pillar_force_y[corner]=pillar_alpha" in source
    assert "pillar_alpha)*articulated_hub" not in source
    assert "vehicle_native_set_roller_anchor" in source
    assert "vehicle_native_roller_anchor[corner]" in source
    assert "vehicle_native_graph_tick_batch" in source
    assert "VehicleNativeBatchLane vehicle_native_batch_lane[VEHICLE_NATIVE_BATCH_CAPACITY]" in source
    assert "vehicle_native_batch_save(&scalar_lane)" in source
    assert "vehicle_native_batch_load(&scalar_lane)" in source
    assert "vehicle_native_graph_batch_tire_state" in source
    assert "vehicle_native_restore_tire_state" in source
    assert "suspension_rig" not in source
    assert "laplacian" not in source.casefold()


def test_symbolic_equation_constants_split_to_two_binary64_lanes_without_freezing_parameters():
    high, low = split_symbolic_constants_to_double_double({"pi_gain": sympy.pi})["pi_gain"]
    assert high == math.pi
    assert low != 0.0
    assert abs(low) < math.ulp(high)


def test_scientific_renderer_is_a_read_only_compiler_abi_consumer():
    visualization = scientific_visualization_abi()
    assert visualization["physics_effect"] == "none"
    attributes = {item["name"]: item for item in visualization["attributes"]}
    assert attributes["axial_strain"]["units"] == "m/m"
    assert attributes["yield_strain"]["units"] == "m/m"
    assert visualization["joint_kinds"] == {
        "0": "member", "1": "bushing-dot", "2": "bearing-dot",
    }


def test_torus_plane_contact_arc_integral_reduces_to_c_and_llvm():
    compilation = compile_torus_plane_contact_arc_ssa()
    c_artifact = emit_ssa_function_to_c(
        compilation.module, compilation.function.name,
    )
    llvm_artifact = emit_ssa_function_to_llvm(
        compilation.module, compilation.function.name,
    )
    assert c_artifact.complete, c_artifact.shortfalls
    assert llvm_artifact.complete, llvm_artifact.shortfalls
    assert symbolic_torus_plane_patch_boundary_integral().has(sympy.Integral)


def test_native_scientific_viewer_is_one_compiler_emitted_runnable_shell():
    vehicle = CFunctionArtifact(
        "abstract_ui_vehicle_step", "",
        ("dt", "position_y", "compression_front_left", "compression_front_right",
         "compression_rear_left", "compression_rear_right",
         "spring_stiffness", "pneumatic_compression_damping",
         "pneumatic_rebound_damping", "angular_damping"),
        ("position_x_next", "position_y_next", "position_z_next",
         "roll_next", "pitch_next", "yaw_next",
         "velocity_x_next", "velocity_y_next", "velocity_z_next",
         "engine_rpm", "engine_torque", "driveline_torque",
         "wheel_omega_front_left_next", "wheel_omega_front_right_next",
         "wheel_omega_rear_left_next", "wheel_omega_rear_right_next",
         "compression_front_left_next", "compression_front_right_next",
         "compression_rear_left_next", "compression_rear_right_next",
         "spring_force_front_left", "spring_force_front_right",
         "spring_force_rear_left", "spring_force_rear_right"), (),
    )
    contact = CFunctionArtifact(
        "abstract_ui_wheel_contact", "",
        ("attachment_x", "attachment_y", "attachment_z", "tire_pressure",
         "radial_carcass_loss", "sidewall_shear_damping",
         "sidewall_shear_stiffness_longitudinal",
         "sidewall_shear_stiffness_lateral", "support",
         "tire_radial_compression"), (), (),
    )
    fixture = CFunctionArtifact(
        "vehicle_roller_fixture_step", "",
        ("dt", "carriage_y_front_left", "carriage_y_front_right",
         "carriage_y_rear_left", "carriage_y_rear_right", "surface_mode",
         "terrain_phase_x", "terrain_phase_z", "terrain_period_x",
         "terrain_period_z", "mode", "command_y_front_left",
         "command_y_front_right", "command_y_rear_left",
         "command_y_rear_right"), (), (),
    )
    source = render_native_scientific_viewer_shell(vehicle, contact, fixture)
    assert "vehicle_native_graph_tick_batch(1,vehicle_in,contact_in,fixture_in,vehicle_out)" in source
    assert "vehicle_native_graph_tick(vehicle_in,contact_in,fixture_in,vehicle_out)" not in source
    assert "glDrawArrays(GL_LINES" in source
    assert "glDrawArrays(GL_POINTS" in source
    assert "LAPLACIAN E" in source
    assert "TOP 6 ENERGY OR STRAIN SUSPECTS" in source
    assert "vehicle_native_tire_diagnostics" in source
    assert "vehicle_native_energy_diagnostics" in source
    assert "vehicle_native_graph_batch_tire_state(1,tire_state)" in source
    assert "tire_edges[TIRE_EDGE_COUNT]" in source
    assert "memcpy(tire_state" in source
    assert "ENSEMBLE_BATCH_COUNT 8" in source
    assert "vehicle_native_graph_tick_batch(ENSEMBLE_BATCH_COUNT" in source
    assert "vehicle_native_graph_batch_tire_state(ENSEMBLE_BATCH_COUNT" in source
    assert "glBlendFunc(GL_ONE,GL_ONE)" in source
    assert 'uAlpha"),1.0f/ENSEMBLE_BATCH_COUNT' in source
    assert "ENSEMBLE %s  BATCH %d  E TOGGLE" in source
    assert 'HUD("SIM %s  TARGET DT 1/%d"' in source
    assert 'HUD("SUBSTEP %s A %.0f OK %.0f REJ %.0f NF %.0f"' in source
    assert 'HUD("SDT MIN %.3G MAX %.3G NEXT %.3G"' in source
    assert 'HUD("WINDOW %.6G/%.6G DISP %.3G/%.3G X%.3G"' in source
    assert 'HUD("REJECT DT %.3G %.3G %.3G %.3G"' in source
    # Physics runs on its own thread; the SDL/render thread never calls the
    # compiled kernels while live. Snapshots publish whole, generation-
    # tagged, under the snapshot lock; UI edits queue through commands.
    assert "physics_thread_main" in source
    assert "CreateThread(NULL,0,physics_thread_main" in source
    assert "if(simulate)for(i=0;i<1;i++)" in source
    assert "AcquireSRWLockShared(&snapshot_lock)" in source
    assert "staging.generation++" in source
    assert "commands.simulate=!commands.simulate" in source
    assert "SDL_WINDOW_INPUT_FOCUS" in source
    assert "window_focused=(sdl_window_flags(win)&SDL_WINDOW_INPUT_FOCUS)!=0" in source
    assert 'HUD("DISPATCH POOL %d' in source
    assert "turing_pool_workers" in source
    assert "double duty[4],td[76],te[4]" in source
    assert "td[14*i+9]" in source
    assert "td[13*i+9]" not in source
    assert 'uPointSize"),1.65f' in source
    assert 'uPrimitiveKind"),2' in source
    assert "@" not in source
