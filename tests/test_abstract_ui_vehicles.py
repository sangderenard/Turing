"""Vehicle slots, general contact surfaces, and baked WebGPU contact physics."""

import json
import base64
import math
import shutil
import subprocess

import pytest
import sympy

import src.compiler.abstract_ui_vehicles as vehicle_module
from src.compiler.abstract_ui_div_map import project_class_to_div_map
from src.compiler.mechanical_creature import MechanicalCreature
from src.compiler.abstract_ui_surfaces import (
    linear_gradient_solid,
    sampled_mud_oval_height_field,
    sampled_offroad_playground_height_field,
    support_surface_model,
)
from src.compiler.abstract_ui_vehicles import (
    contact_lane_count,
    CONTACT_PATCH_OUTPUTS,
    CONTACT_PATCH_TENSOR_REDUCER_OUTPUTS,
    VEHICLE_STATE_OUTPUTS,
    WHEEL_NAMES,
    _passive_radial_ringdown,
    compile_symbolic_vehicle_physics,
    compile_symbolic_vehicle_physics_webgpu,
    compile_symbolic_vehicle_physics_wasm,
    fit_vehicle_chassis_to_power_unit,
    fit_vehicle_chassis_to_body_packaging,
    fit_vehicle_wheelbase_under_body_mass,
    compile_wheel_contact_abstract_tensor,
    compile_wheel_contact_ssa,
    compile_wheel_contact_wasm,
    compile_wheel_contact_webgpu,
    extra_precision_closure,
    load_default_car_configuration,
    solve_vehicle_body_packaging,
    symbolic_vehicle_equations,
    symbolic_torus_plane_contact_arc_equations,
    symbolic_wheel_contact_equations,
    vehicle_configuration_from_mapping,
    vehicle_webgpu_program_model,
)


def test_body_packaging_expands_short_cab_and_preserves_wheel_placement():
    base = load_default_car_configuration()
    source = json.loads(base.canonical_json)
    source["body_packaging"].update({
        "preset": "bus-chassis", "requested_cab_length_m": .4,
        "bed_length_m": 1.8, "seat_rows": 6,
    })
    custom = vehicle_configuration_from_mapping(source)
    solved = solve_vehicle_body_packaging(custom)
    assert solved["cab_was_expanded"] is True
    assert solved["effective_cab_length_m"] >= (
        6 * source["body_packaging"]["minimum_row_pitch_m"]
        + source["body_packaging"]["front_seating_clearance_m"]
        + source["body_packaging"]["rear_seating_clearance_m"]
    )
    fitted, fit = fit_vehicle_chassis_to_body_packaging(custom)
    assert fit["frame_fit_satisfied"] is True
    assert fit["wheel_placement_mutated"] is False
    assert fitted.source["wheels"] == custom.source["wheels"]
    graph = vehicle_module._vehicle_mechanical_graph(fitted)
    assert graph["body_packaging"]["effective_cab_length_m"] == pytest.approx(
        fit["effective_cab_length_m"])


def test_wheelbase_solver_places_axles_under_rebuilt_mass_without_moving_mass():
    base = load_default_car_configuration()
    fitted_body, _ = fit_vehicle_chassis_to_body_packaging(base)
    fitted, result = fit_vehicle_wheelbase_under_body_mass(fitted_body)
    assert result["policy"] == "center-under-mass"
    assert result["body_mass_was_moved_to_fit_axles"] is False
    assert result["achieved_front_axle_load_fraction"] == pytest.approx(
        result["target_front_axle_load_fraction"], abs=2e-6)
    assert result["front_axle_x_m"] > result["center_of_mass_x_m"]
    assert result["rear_axle_x_m"] < result["center_of_mass_x_m"]
    assert fitted.source["mass"] == fitted_body.source["mass"]


def test_composite_helix_spring_and_complementary_bump_stop_share_one_force_edge():
    config = load_default_car_configuration()
    graph = vehicle_module._vehicle_mechanical_graph(config)
    coilover = next(edge for edge in graph["edges"]
                    if edge["identity"] == "suspension.front_left.coilover")
    assert coilover["constitutive_models"]["2"]["kind"] == "composite-parametric-helix"
    assert coilover["bump_stop"]["role"] == (
        "complementary-terminal-branch-no-parallel-double-count")
    assert coilover["bump_stop"]["start_compression_m"] < config.source["suspension"]["travel"]
    equations, symbols = symbolic_vehicle_equations()
    spring = next(eq.rhs for eq in equations if str(eq.lhs) == "spring_force_front_left")
    for name in ("spring_primary_wire_diameter_m", "spring_secondary_active_turns",
                 "spring_progressive_cubic_n_per_m3", "bump_stop_damping"):
        assert spring.has(symbols[name])


def test_frame_occupant_cell_and_drivetrain_have_no_adjustment_separation_joint():
    graph = vehicle_module._vehicle_mechanical_graph(load_default_car_configuration())
    assert not any(node["identity"].startswith("body_sled") for node in graph["nodes"])
    assert not any(edge["identity"].startswith("body_sled") for edge in graph["edges"])
    edges = {edge["identity"]: edge for edge in graph["edges"]}
    for longitudinal in ("front", "rear"):
        for lateral in ("left", "right"):
            mount = edges[f"cage.frame_mount.{longitudinal}_{lateral}"]
            assert mount["b"] == f"frame.{longitudinal}_{lateral}"
    assert edges["body_pin.frame_mount.hood_left"]["b"] == "frame.front_left"
    assert edges["body_pin.frame_mount.cab_left"]["b"] == "cage.front_left.upper"


class VehicleProbe:
    speed: float


def test_engine_pan_fit_rebuilds_shared_chassis_graph_and_mass():
    base = load_default_car_configuration()
    fitted, fit = fit_vehicle_chassis_to_power_unit(
        base,
        engine_envelope_m=(2.2, 1.25, 1.1),
        oil_pan_envelope_m=(1.5, .35, .8),
        engine_mass_kg=744.0,
    )
    assert fit["authority"] == "shared-vehicle-configuration-and-mechanical-graph-rebuild"
    assert fit["chassis_half_length_m"] > float(base.source["chassis"]["half_length"])
    assert fit["wheelbase_m"] == 2 * float(base.source["wheels"]["wheelbase_half_length"])
    assert fit["track_m"] == 2 * float(base.source["wheels"]["track_half_width"])
    assert fit["wheel_placement_changed_by_pan_fit"] is False
    assert fit["wheel_placement"]["mutates_wheel_placement"] is False
    assert fit["wheel_placement"]["satisfied"] is False
    assert fit["wheel_placement"]["sweep"]["minimum_clearance_m"] < 0
    assert fit["total_mass_kg"] > float(base.source["mass"])
    graph = vehicle_module._vehicle_mechanical_graph(fitted)
    axle = next(node for node in graph["nodes"] if node["identity"] == "suspension.front_left.hub")
    assert axle["reference_position"][0] == pytest.approx(
        fitted.source["wheels"]["wheelbase_half_length"]
    )
    frame = next(node for node in graph["nodes"] if node["identity"] == "frame.front_left")
    assert frame["reference_position"][0] == pytest.approx(fitted.source["chassis"]["half_length"])
    post_mount = next(edge for edge in graph["edges"]
                      if edge["identity"] == "suspension.front_left.upper_pickup_mount_forward")
    assert post_mount["b"] == "suspension_mount_post.front_left.upper"


def test_vehicle_engine_profile_switch_includes_aircraft_and_heavy_diesel_without_runtime_compile(monkeypatch):
    monkeypatch.setattr(vehicle_module, "vehicle_webgpu_program_model",
                        lambda *_args, **_kwargs: {"identity": "stub-no-backend-compile"})
    vehicle = vehicle_module.vehicle_slot_model("engine-profile-probe", "driver")["vehicles"][0]
    profiles = {preset["identity"]: preset for preset in vehicle["power_unit_presets"]}
    merlin = profiles["packard-merlin-v1650"]
    diesel = profiles["cat-c18-industrial-diesel"]
    assert len(profiles) == 13
    assert merlin["architecture"]["cylinders"] == 12
    assert merlin["package"]["chassis_fit"]["authority"] == (
        "shared-vehicle-configuration-and-mechanical-graph-rebuild"
    )
    assert merlin["package"]["chassis_fit"]["wheelbase_m"] == pytest.approx(1.24)
    assert merlin["package"]["chassis_fit"]["wheel_placement_changed_by_pan_fit"] is False
    assert merlin["configuration"]["engine_mass_kg"] == 744
    assert merlin["fuel_compatibility"]["pump-gasoline-93"] == pytest.approx(.48)
    assert merlin["preferred_ignition_profile"] == "aircraft-dual-magneto"
    assert diesel["configuration"]["displacement_liters"] == pytest.approx(18.1)
    assert diesel["configuration"]["engine_mass_kg"] == 1673
    assert diesel["fuel_compatibility"]["ultra-low-sulfur-diesel"] == 1
    assert diesel["fuel_compatibility"]["pump-gasoline-93"] < .05
    assert diesel["ignition_compatibility"]["gasoline-distributor"] < .05
    switch = vehicle["engine_kernel_switch"]
    assert switch["runtime_compilation"] is False
    assert switch["cases"][-1]["compiled_selectors"]["symbolic-fidelity"] == 25
    clutches = {part["identity"]: part for part in vehicle["clutch_presets"]}
    assert clutches["old-soft-organic"]["maximum_torque_nm"] == 235
    assert clutches["aircraft-heavy-multiplate"]["maximum_torque_nm"] == 3800
    assert clutches["industrial-twin-disc"]["maximum_torque_nm"] == 4800


def test_json_car_configuration_is_strict_parametric_and_content_addressed():
    config = load_default_car_configuration()
    assert config.source["tires"]["pressure_pa"] == 135000.0
    assert config.source["solid_contact"]["kinetic_friction"] == pytest.approx(.54)
    assert sum(config.source["mass_distribution"].values()) == pytest.approx(1.0)
    assert config.parameter_defaults()["inverse_mass"] == pytest.approx(0.001907882279068623)
    assert config.parameter_defaults()["unsprung_mass_rear_left"] == pytest.approx(108.145)
    assert config.source["powertrain"]["displacement_liters"] == 4.227
    assert config.source["powertrain"]["brake_mean_effective_pressure_pa"] == 835000.0
    assert config.parameter_defaults()["engine_rotating_inertia"] == pytest.approx(.68)
    mass = config.mass_properties()
    assert mass["total_mass_kg"] == pytest.approx(956.7213534634711)
    assert mass["allocated_component_mass_kg"] == pytest.approx(956.7193781825611)
    assert mass["residual_frame_cage_driver_mass_kg"] == pytest.approx(.0019752809100737068)
    assert sum(item["mass_kg"] for item in mass["components"]) == pytest.approx(956.7213534634711)
    assert mass["center_of_mass"] == pytest.approx(
        [-.05461548423774686, .07250059381356276, -.0018452603713808562], abs=1e-8)
    assert mass["derived_axle_fractions"]["front"] == pytest.approx(.45595525464697834)
    assert len([item for item in mass["components"]
                if item["identity"].startswith("suspension_mount_post_")]) == 4
    placement = vehicle_module.solve_vehicle_wheel_placement_mounts(config)
    assert placement["satisfied"] is True
    assert placement["criteria"]["axle_group_fore_aft_m"]["unconstrained"] is True
    assert placement["selected_architecture"] == "double-wishbone-coilover"
    assert placement["planned_architectures"] == [
        "solid-axle-leaf-spring", "solid-axle-coilover"]
    assert mass["live_mass_components"] == [
        "fuel_live", "ballast_front_left", "ballast_front_right",
        "ballast_rear_left", "ballast_rear_right",
    ]
    for axis in ("roll", "pitch", "yaw"):
        assert config.parameter_defaults()[f"inverse_inertia_{axis}"] == pytest.approx(
            1 / mass["inertia_kg_m2"][axis])
    assert config.source["powertrain"]["transmission_mass_kg"] == 42
    assert config.source["powertrain"]["transfer_case_mass_kg"] == 24
    assert config.source["drivetrain"]["wheel_mass_kg"] == 68
    assert config.source["drivetrain"]["tire_mass_kg"] == 14
    assert config.wheel_rotational_inertia() == pytest.approx(21.137624512)
    assert [config.parameter_defaults()[f"crank_axis_{axis}"] for axis in "xyz"] == pytest.approx([1, 0, 0])
    assert config.source["suspension"]["pneumatic_compression_damping"] == 3200.0
    assert config.source["suspension"]["pneumatic_rebound_damping"] == 4100.0
    assert config.source["suspension"]["pneumatic_efficiency"] == .96
    assert config.source["suspension"]["maximum_compression_speed"] == 1.25
    assert config.source["suspension"]["active_damping_minimum_scale"] == .88
    assert config.source["suspension"]["active_damping_maximum_scale"] == 1.18
    assert config.source["suspension"]["bump_stop_stiffness_n_per_m"] == 180000
    assert config.source["suspension"]["bump_stop_progressive_stiffness_n_per_m2"] == 1600000
    assert config.source["tires"]["lateral_deformation_mode_frequency_hz"] == 7.0
    assert config.source["tires"]["sidewall_deformation_damping_ratio"] > 1
    assert config.source["controls"]["angular_damping"] == 4.2
    assert config.parameter_defaults()["air_density"] == pytest.approx(1.225)
    assert config.parameter_defaults()["drag_longitudinal_coefficient"] == pytest.approx(.72)
    assert config.parameter_defaults()["drag_lateral_vector_z"] == pytest.approx(1)
    assert config.parameter_defaults()["drag_vertical_reference_area"] == pytest.approx(4.2)
    assert config.source["traction_control"]["target_friction_utilization"] == .92
    assert config.source["traction_control"]["slip_sensor_damping_ratio"] > 1
    assert config.source["traction_control"]["utilization_sensor_damping_ratio"] > 1
    assert config.source["presentation"]["world_tile_size"] == .35
    assert config.source["transmission"]["mode_default"] == "automatic"
    assert config.source["transmission"]["starting_gear"] == 1
    assert config.source["transmission"]["forward_ratios"] == [3.52, 2.27, 1.46, 1.0]
    assert config.source["transmission"]["low_range_ratio"] == 2.62
    assert config.source["transmission"]["ultra_low_range_ratio"] == 5.24
    assert config.source["drivetrain"]["differential_lock_maximum_torque_nm"] == 680.0
    assert config.source["wheels"]["track_half_width"] > (
        config.source["chassis"]["half_width"] + config.source["tires"]["width"] / 2)
    assert config.source["suspension"]["travel"] < 2 * config.source["tires"]["radius"]
    assert config.source["presentation"]["world_tile_strength"] == .78
    assert config.source["presentation"]["chase_camera_distance"] == 2.6
    assert "maximum_speed" not in config.source["controls"]
    assert len(config.digest) == 64

    invalid = json.loads(config.canonical_json)
    invalid["surprise_browser_policy"] = True
    with pytest.raises(ValueError, match="unknown"):
        vehicle_configuration_from_mapping(invalid)


def test_generic_corner_wrenches_carry_bumpers_and_density_sized_ballast():
    config = load_default_car_configuration()
    graph = vehicle_module._vehicle_mechanical_graph(config)
    nodes = {node["identity"]: node for node in graph["nodes"]}
    edges = {edge["identity"]: edge for edge in graph["edges"]}
    for corner in WHEEL_NAMES:
        boss = nodes[f"attachment.{corner}"]
        mount = edges[f"attachment.mount.{corner}"]
        assert boss["kind"] == "generic-six-axis-wrench-attachment"
        assert mount["a"] == f"frame.{corner}"
        assert mount["b"] == f"attachment.{corner}"
        assert mount["constraint"] == "breakable-six-axis-braze-on"
        assert mount["yield_force_n"] < mount["fracture_force_n"]
        assert edges[f"ballast.hanger.{corner}"]["a"] == f"attachment.{corner}"
    for end in ("front", "rear"):
        assert edges[f"bumper.cross_tube.{end}"]["radius"] > .04
        for side in ("left", "right"):
            shock = edges[f"bumper.shock.{end}_{side}"]
            assert shock["constraint"] == "preloaded-bumper-shock-absorber"
            assert shock["preload_force_n"] > 0
            assert shock["compression_damping_n_s_per_m"] > 10000

    loaded = json.loads(config.canonical_json)
    loaded["ballast"]["requested_mass_kg"]["front_left"] = 100.0
    loaded_config = vehicle_configuration_from_mapping(loaded)
    block = next(row for row in loaded_config.chassis_attachment_layout()["ballast"]
                 if row["identity"] == "front_left")
    assert block["volume_m3"] == pytest.approx(100 / 11340)
    assert loaded_config.mass_properties()["total_mass_kg"] == pytest.approx(1056.7213534634711)
    assert any(node["identity"] == "ballast.front_left.weight"
               for node in vehicle_module._vehicle_mechanical_graph(loaded_config)["nodes"])

    impossible = json.loads(config.canonical_json)
    impossible["ballast"]["requested_mass_kg"]["rear_right"] = 100000.0
    with pytest.raises(ValueError, match="fits at most"):
        vehicle_configuration_from_mapping(impossible)


def test_runtime_drag_vector_set_projects_quadratic_force_in_body_directions():
    equations, symbols = symbolic_vehicle_equations()
    velocity_x = next(equation.rhs for equation in equations
                      if str(equation.lhs) == "velocity_x_next")
    values = {symbol: 0.0 for symbol in velocity_x.free_symbols}
    values.update({
        symbols["velocity_x"]: 10.0, symbols["dt"]: .01,
        symbols["inverse_mass"]: .001, symbols["yaw_cos"]: 1.0,
        symbols["yaw_sin"]: 0.0, symbols["air_density"]: 1.225,
        symbols["drag_longitudinal_vector_x"]: 1.0,
        symbols["drag_longitudinal_coefficient"]: .72,
        symbols["drag_longitudinal_reference_area"]: 1.85,
    })
    produced = float(velocity_x.subs(values))
    expected_drag = -.5 * 1.225 * .72 * 1.85 * (10.0 ** 2 + 1e-8) ** .5 * 10.0
    assert produced == pytest.approx(10.0 + .01 * .001 * expected_drag)

    invalid = json.loads(load_default_car_configuration().canonical_json)
    invalid["aerodynamics"]["drag_vectors"]["lateral"]["vector"] = [0.0, 0.0, 2.0]
    with pytest.raises(ValueError, match="unit length"):
        vehicle_configuration_from_mapping(invalid)


def test_silver_upright_mass_is_connected_to_the_real_unsprung_corner_graph():
    config = load_default_car_configuration()
    graph = vehicle_module._vehicle_mechanical_graph(config)
    nodes = {node["identity"]: node for node in graph["nodes"]}
    edges = {edge["identity"]: edge for edge in graph["edges"]}
    for corner in WHEEL_NAMES:
        prefix = f"suspension.{corner}"
        upright = edges[f"{prefix}.upright"]
        assert (upright["a"], upright["b"]) == (
            f"{prefix}.upper_ball_joint", f"{prefix}.lower_ball_joint")
        assert upright["damage"]["model"] == "elastic-plastic-member-with-shear-fracture"
        carrier = edges[f"{prefix}.hub_carrier"]
        bearing = edges[f"{prefix}.wheel_bearing"]
        assert (carrier["a"], carrier["b"], carrier["constraint"]) == (
            f"{prefix}.upper_ball_joint", f"{prefix}.knuckle", "rigid-offset")
        assert (bearing["a"], bearing["b"], bearing["constraint"]) == (
            f"{prefix}.knuckle", f"{prefix}.hub", "rotational-bearing")
        assert bearing["structural_constraint"] == "five-axis-support-one-axis-free-rotation"
        assert bearing["force_path"] == "upright-through-bearing-to-wheel"
        assert "fixed_to" not in bearing
        assert edges[f"{prefix}.upper_arm_forward"]["a"] == f"{prefix}.upper_pickup_forward"
        assert edges[f"{prefix}.upper_arm_forward"]["b"] == f"{prefix}.upper_ball_joint"
        post = f"suspension_mount_post.{corner}.upper"
        assert edges[f"{prefix}.upper_pickup_mount_forward"]["b"] == post
        assert edges[f"suspension_mount_post.{corner}.upper_half"]["a"] == f"frame.{corner}"
        assert edges[f"suspension_mount_post.{corner}.upper_half"]["b"] == post
        assert nodes[f"{prefix}.knuckle"]["mass_frame"] == "corner-unsprung"
        assert nodes[f"{prefix}.knuckle"]["moves_with"] == f"{prefix}.knuckle"
        assert nodes[f"{prefix}.knuckle"]["generalized_coordinate"] == f"compression_{corner}"
        assert nodes[f"{prefix}.hub"]["reference_position"][1] == pytest.approx(
            -float(config.source["chassis"]["clearance"])
        )
        assert f"{prefix}.wheel_rim" not in nodes
        assert f"{prefix}.tire_carcass" not in nodes
        assert nodes[f"{prefix}.pneumatic_wheel_valve"]["balance_mass_delta_kg"] > 0.0
        assert nodes[f"{prefix}.pneumatic_rotary_union"]["assembly_custody"] == (
            "pillar-then-knuckle-stationary-side")
        assert nodes[f"{prefix}.pneumatic_bearing_rotor"]["moves_with"] == f"{prefix}.hub"
        assert edges[f"pneumatics.bearing_rotary_seal.{corner}"]["stationary_owner"] == f"{prefix}.knuckle"
        assert edges[f"pneumatics.outer_service_valve.{corner}"]["user_accessible"] is True
        assert edges[f"pneumatics.tube_stem_install.{corner}"]["synchronization"] == (
            "tube-stem-becomes-outer-service-valve-on-install")
        assert nodes[f"{prefix}.lower_ball_joint"]["mass_frame"] == "corner-unsprung"
        assert nodes[f"{prefix}.coilover_chassis"]["mass_frame"] == "chassis-sprung"


def test_wheel_pneumatic_ports_transfer_custody_and_enter_balance_mass():
    config = load_default_car_configuration()
    graph = vehicle_module._vehicle_mechanical_graph(config)
    nodes = {node["identity"]: node for node in graph["nodes"]}
    edges = {edge["identity"]: edge for edge in graph["edges"]}
    for corner in WHEEL_NAMES:
        prefix = f"suspension.{corner}"
        valve = nodes[f"{prefix}.pneumatic_wheel_valve"]
        assert valve["removed_hub_material_kg"] == pytest.approx(.025)
        assert valve["balance_mass_delta_kg"] == pytest.approx(.055)
        assert nodes[f"{prefix}.pneumatic_rotary_union"]["assembly_custody"] == (
            "pillar-then-knuckle-stationary-side")
        assert nodes[f"{prefix}.pneumatic_bearing_rotor"]["moves_with"] == f"{prefix}.hub"
        assert edges[f"pneumatics.bearing_rotary_seal.{corner}"]["stationary_owner"] == f"{prefix}.knuckle"
        assert edges[f"pneumatics.outer_service_valve.{corner}"]["user_accessible"] is True
        assert edges[f"pneumatics.tube_stem_install.{corner}"]["synchronization"] == (
            "tube-stem-becomes-outer-service-valve-on-install")
        for seat in ("inner", "outer"):
            assert edges[f"pneumatics.hub_passage.{corner}.{seat}"]["terminal"] == (
                f"{prefix}.tire_skin.closed_volume")
    tire_network = graph["service_port_api"]["networks"]["tire_pressure"]
    assert tire_network["tractor_default"] == (
        "bearing-feed-and-user-accessible-outer-rim-valve-both-live")


def test_gradient_solid_has_no_ramp_specific_physics_category():
    contract = support_surface_model("world")
    assert contract["consumers"] == [
        "platformer-body", "projectile-body", "vehicle-contact-patch", "rigid-body",
    ]
    assert "contact-manifold" in contract["selection"]["rule"]
    assert contract["world_bottom"]["thickness"] == 8.0
    assert contract["world_bottom"]["sampled_surface_guard_depth"] == .75
    assert contract["world_bottom"]["role"].startswith("emergency-containment")
    slope = linear_gradient_solid(
        "world/slope", "world", center_x=2, center_z=3, half_width=1,
        half_run=2, low_height=.1, high_height=1.1,
    )
    assert slope["physics"]["collider"] == "solid-contact-surface"
    assert slope["physics"]["contact_mode"] == "normal-constraint-with-coulomb-friction"
    assert slope["surface"]["gradient"] == pytest.approx([.25, 0])
    assert slope["kind"] == "static-gradient-solid"
    assert slope["geometry_mode"] == "height-field-prism"


def test_sampled_mud_oval_has_crawler_relief_and_a_smooth_cut_course():
    terrain = sampled_mud_oval_height_field("terrain", "world", center_x=0, center_z=0)
    surface = terrain["surface"]
    columns, rows = surface["resolution"]
    assert (columns, rows) == (49, 33)
    assert len(surface["heights"]) == columns * rows
    assert min(surface["heights"]) <= .13
    assert max(surface["heights"]) - min(surface["heights"]) >= .75
    assert surface["features"] == {
        "landscape": "procedural-crawler-hills", "course": "smooth-cut-oval",
        "authority": "outer-courtyard-depth-map",
    }
    assert terrain["geometry_mode"] == "sampled-height-field-prism"


def test_offroad_playground_has_distinct_obstacles_on_a_bounded_grid():
    terrain = sampled_offroad_playground_height_field(
        "terrain", "world", center_x=0, center_z=0, half_x=28, half_z=28,
    )
    surface = terrain["surface"]
    assert surface["resolution"] == [81, 81]
    assert surface["tracking_scope"] == "bounded-play-area-only"
    assert surface["features"]["zones"] == [
        "hill-climb", "rock-crawl", "whoops", "dry-creek-bed",
        "sampled-opposing-ramps",
    ]
    assert max(surface["heights"]) - min(surface["heights"]) >= 2.5
    assert len(set(surface["heights"])) > 500


def test_contact_patch_equations_publish_pneumatic_load_and_coulomb_force_pair():
    equations, _ = symbolic_wheel_contact_equations()
    ordered = sorted(set().union(*(equation.rhs.free_symbols for equation in equations)), key=str)
    evaluate = sympy.lambdify(ordered, [equation.rhs for equation in equations], "math")
    values = {str(symbol): 0.0 for symbol in ordered}
    values.update({
        "dt": 1 / 120, "support": 1, "hub_height": .25, "geometric_compression": .21,
        "previous_compression": .21, "surface_height": 0,
        "normal_y": 1, "forward_x": 1, "right_z": 1,
        "slip_longitudinal": .01, "attachment_x": .54, "attachment_y": -.22,
        "corner_weight": 620 * 9.81 * .27, "suspension_rest_length": .24,
        "chassis_clearance": .22, "suspension_travel": .34,
            "spring_stiffness": 7200, "pneumatic_compression_damping": 1350,
            "bump_stop_stiffness": 180000,
            "bump_stop_progressive_stiffness": 1600000,
            "bump_stop_damping": 5200,
            "linkage_motion_ratio": 1.0,
        "pneumatic_rebound_damping": 1850, "pneumatic_efficiency": .96,
        "maximum_compression_speed": 1.8,
        "active_damping_minimum_scale": .88, "active_damping_maximum_scale": 1.18,
        "active_damping_body_velocity_gain_s_per_m": .22,
        "active_damping_rebound_release_gain_s_per_m": .08,
        "wheelbase_half_length": .62, "track_half_width": .56,
        "corner_front_sign": 1, "corner_side_sign": -1,
        "tire_pressure": 185000, "minimum_contact_area": .006,
        "maximum_contact_area": .045, "mu_static": 1.18, "mu_kinetic": .92,
        "load_sensitivity": .075, "slip_transition_speed": .38,
        "tire_major_radius": .205, "tire_section_radius": .115,
        "tire_effective_tread_width": .2496,
        "tire_reference_volume": 2 * math.pi ** 2 * .205 * .115 ** 2,
        "tire_gas_polytropic_exponent": 1.38, "radial_carcass_loss": 1320,
        "tire_radial_effective_mass": 96,
        "sidewall_deformation_longitudinal": .01,
        "sidewall_deformation_velocity_longitudinal": 0,
        "sidewall_deformation_lateral": 0,
        "sidewall_deformation_velocity_lateral": 0,
        "sidewall_shear_stiffness_longitudinal": 420000,
        "sidewall_shear_stiffness_lateral": 330000,
        "sidewall_shear_damping": 420,
        "tire_radial_compression": .04, "tire_radial_velocity": 0,
        "suspension_alignment": 1,
    })
    result = dict(zip(CONTACT_PATCH_OUTPUTS,
                      evaluate(*(values[str(symbol)] for symbol in ordered))))
    assert .006 <= result["contact_area"] <= .045
    assert result["chassis_force_y"] > 0
    assert result["chassis_force_x"] < 0
    # A non-suspension contact lane exposes the pneumatic toroid's capacity.
    # More flattening displaces more enclosed air and produces a larger load;
    # there is no radial k*x tire spring or overlap-rejection impulse.
    values.update({"suspension_alignment": 0, "tire_radial_compression": .02})
    shallow = dict(zip(CONTACT_PATCH_OUTPUTS,
                       evaluate(*(values[str(symbol)] for symbol in ordered))))
    values["tire_radial_compression"] = .08
    deep = dict(zip(CONTACT_PATCH_OUTPUTS,
                    evaluate(*(values[str(symbol)] for symbol in ordered))))
    assert deep["chassis_force_y"] > shallow["chassis_force_y"]
    # The terrain kernel owns only the pneumatic reaction at the tire node.
    # Suspension over-travel is intentionally not pre-collapsed into that
    # answer: its progressive bump-stop reaction acts across the unsprung mass
    # in the vehicle graph integrator.
    values.update({"suspension_alignment": 1, "tire_pressure": 500000,
                   "tire_radial_compression": .08, "geometric_compression": .34,
                   "previous_compression": .34})
    at_bind = dict(zip(CONTACT_PATCH_OUTPUTS,
                       evaluate(*(values[str(symbol)] for symbol in ordered))))
    values["geometric_compression"] = .44
    beyond_bind = dict(zip(CONTACT_PATCH_OUTPUTS,
                           evaluate(*(values[str(symbol)] for symbol in ordered))))
    assert beyond_bind["chassis_force_y"] == pytest.approx(at_bind["chassis_force_y"])
    assert result["chassis_torque_x"] == pytest.approx(
        values["attachment_y"] * result["chassis_force_z"]
        - values["attachment_z"] * result["chassis_force_y"])
    assert result["chassis_torque_y"] == pytest.approx(
        values["attachment_z"] * result["chassis_force_x"]
        - values["attachment_x"] * result["chassis_force_z"])
    assert result["chassis_torque_z"] == pytest.approx(
        values["attachment_x"] * result["chassis_force_y"]
        - values["attachment_y"] * result["chassis_force_x"])
    values["slip_longitudinal"] = 2.0
    sliding = dict(zip(CONTACT_PATCH_OUTPUTS,
                       evaluate(*(values[str(symbol)] for symbol in ordered))))
    assert abs(sliding["chassis_force_x"]) <= (
        values["mu_static"] * 1.18 * sliding["chassis_force_y"]
    )
    # A ravine/drop impact can traverse the whole suspension range in one
    # fixed step. The same compiled law must stay finite and Coulomb-bounded.
    for compression in (-.08, 0, .12, .34, .7):
        for previous in (0, .17, .34):
            for vertical_speed in (-18, 0, 12):
                values.update({
                    "geometric_compression": compression,
                    "previous_compression": previous,
                    "chassis_velocity_y": vertical_speed,
                    "hub_velocity_y": vertical_speed,
                    "roll_velocity": 4.5,
                    "pitch_velocity": -5.5,
                    "slip_longitudinal": 7.0,
                    "slip_lateral": -5.0,
                })
                impact = dict(zip(CONTACT_PATCH_OUTPUTS,
                                  evaluate(*(values[str(symbol)] for symbol in ordered))))
                assert all(math.isfinite(float(value)) for value in impact.values())
                normal_load = impact["chassis_force_y"]
                tangent_load = math.hypot(impact["chassis_force_x"],
                                          impact["chassis_force_z"])
                assert normal_load >= 0
                assert tangent_load <= values["mu_static"] * 1.18 * normal_load + 1e-6


def test_radial_ringdown_conserves_or_dissipates_the_local_mode_energy():
    x, v, k, c, mass, dt = sympy.symbols("x v k c mass dt", positive=True)
    next_x, next_v, impulse, loss = _passive_radial_ringdown(x, v, k, c, mass, dt)
    evaluate = sympy.lambdify(
        (x, v, k, c, mass, dt), (next_x, next_v, impulse, loss), "math")
    initial_x, initial_v = .052, 1.7
    stiffness, effective_mass, step = 185000.0, 96.0, 1 / 360
    initial_energy = .5 * stiffness * initial_x ** 2 + .5 * effective_mass * initial_v ** 2
    for damping in (0.0, 1900.0):
        final_x, final_v, outward_impulse, dissipated = evaluate(
            initial_x, initial_v, stiffness, damping, effective_mass, step)
        final_energy = .5 * stiffness * final_x ** 2 + .5 * effective_mass * final_v ** 2
        assert outward_impulse > 0
        assert final_energy + dissipated == pytest.approx(initial_energy, rel=2e-12, abs=2e-12)
        if damping == 0:
            assert dissipated == 0
        else:
            assert dissipated > 0
            assert final_energy < initial_energy


def test_torus_plane_closed_form_matches_direct_active_arc_quadrature():
    equations, symbols = symbolic_torus_plane_contact_arc_equations()
    ordered = tuple(symbols.values())
    evaluate = sympy.lambdify(ordered, [equation.rhs for equation in equations], "math")
    distance, major, projection, section = .31, .465, .92, .085
    result = dict(zip(
        vehicle_module.TORUS_PLANE_CONTACT_ARC_OUTPUTS,
        evaluate(distance, major, projection, section),
    ))
    projected = major * projection
    boundary = max(-1.0, min(1.0, (section - distance) / projected))
    start, finish = math.acos(boundary), 2 * math.pi - math.acos(boundary)
    intervals = 4096
    step = (finish - start) / intervals
    samples = [section - distance - projected * math.cos(start + index * step)
               for index in range(intervals + 1)]
    quadrature = step / 3 * (
        samples[0] + samples[-1]
        + 4 * sum(samples[1:-1:2]) + 2 * sum(samples[2:-1:2]))
    assert result["contact_arc_angle"] == pytest.approx(finish - start, rel=1e-12)
    assert result["integrated_penetration"] == pytest.approx(quadrature, rel=1e-11)
    assert result["mean_penetration"] == pytest.approx(
        quadrature / (finish - start), rel=1e-11)


def test_lateral_slip_is_absorbed_by_a_damped_sidewall_state_before_contact_force():
    equations, _ = symbolic_vehicle_equations()
    by_name = {str(equation.lhs): equation.rhs for equation in equations}
    deformation = by_name["tire_deformation_lateral_front_left_next"]
    velocity = by_name["tire_deformation_velocity_lateral_front_left_next"]
    inputs = sorted(deformation.free_symbols | velocity.free_symbols, key=str)
    evaluate = sympy.lambdify(inputs, [deformation, velocity], "math")
    values = {str(symbol): 0.0 for symbol in inputs}
    values.update({
        "dt": 1 / 120,
        "slip_lateral_front_left": 1.0,
        "tire_lateral_deformation_frequency_hz": 5.0,
        "tire_sidewall_deformation_damping_ratio": 1.05,
        "tire_maximum_sidewall_deformation": .055,
    })
    next_deformation, next_velocity = evaluate(
        *(values[str(symbol)] for symbol in inputs))
    quasi_static_deformation = 1 / (2 * math.pi * values["tire_lateral_deformation_frequency_hz"])
    assert 0 < next_deformation < quasi_static_deformation
    assert 0 < next_velocity < values["slip_lateral_front_left"]

    values.update({"slip_lateral_front_left": 0,
                   "tire_deformation_velocity_lateral_front_left": .5})
    _, damped_velocity = evaluate(*(values[str(symbol)] for symbol in inputs))
    assert 0 < damped_velocity < .5


def test_contact_patch_compiles_through_repository_ssa_to_vectorized_webgpu():
    compiled = compile_wheel_contact_ssa()
    assert compiled.function.metadata["symbolic_dtype"] == "float32"
    assert compiled.process_graph.G.graph["sympy_translation_fallbacks"] == ()
    emitted = compile_wheel_contact_webgpu()
    assert emitted.complete
    assert emitted.launch_plan.workgroup_size == (32, 1, 1)
    assert emitted.launch_plan.groups == (2, 1, 1)
    assert "@compute @workgroup_size(32, 1, 1)" in emitted.source
    assert len(CONTACT_PATCH_OUTPUTS) == 7
    vehicle_equations, _ = symbolic_vehicle_equations()
    arguments = {
        str(symbol)
        for equation in vehicle_equations
        for symbol in equation.rhs.free_symbols
    }
    assert {f"contact_wrench_force_{axis}" for axis in "xyz"} <= arguments
    assert {f"contact_wrench_torque_{axis}" for axis in "xyz"} <= arguments


def test_scalar_contact_wasm_fallback_puts_pneumatic_toroid_load_into_the_chassis():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute WebAssembly")
    artifact = compile_wheel_contact_wasm()
    values = {name: 0.0 for name in artifact.input_names}
    values.update({
        "dt": 1 / 120, "support": 1, "geometric_compression": .2,
        "suspension_alignment": 1.0,
        "previous_compression": .2, "normal_y": 1, "forward_x": 1, "right_z": 1,
        "corner_weight": 620 * 9.81 / 4, "suspension_travel": .34,
        "spring_stiffness": 7200, "linkage_motion_ratio": 1,
        "bump_stop_stiffness": 180000,
        "bump_stop_progressive_stiffness": 1600000,
        "bump_stop_damping": 5200,
        "pneumatic_compression_damping": 3200, "pneumatic_rebound_damping": 4100,
        "pneumatic_efficiency": .96, "maximum_compression_speed": 1.25,
        "active_damping_minimum_scale": .88, "active_damping_maximum_scale": 1.18,
        "active_damping_body_velocity_gain_s_per_m": .22,
        "active_damping_rebound_release_gain_s_per_m": .08,
        "tire_pressure": 155000, "minimum_contact_area": .008,
        "maximum_contact_area": .06, "mu_static": 1.18, "mu_kinetic": .92,
        "load_sensitivity": .075, "slip_transition_speed": .38,
        "tire_major_radius": .205, "tire_section_radius": .115,
        "tire_effective_tread_width": .2496,
        "tire_reference_volume": 2 * math.pi ** 2 * .205 * .115 ** 2,
        "tire_gas_polytropic_exponent": 1.38, "radial_carcass_loss": 1320,
        "tire_radial_effective_mass": 96,
        "tire_radial_compression": .05, "tire_radial_velocity": 0,
        "sidewall_deformation_longitudinal": .008,
        "sidewall_deformation_velocity_longitudinal": 0,
        "sidewall_deformation_lateral": 0,
        "sidewall_deformation_velocity_lateral": 0,
        "sidewall_shear_stiffness_longitudinal": 420000,
        "sidewall_shear_stiffness_lateral": 330000,
        "sidewall_shear_damping": 420,
    })
    script = r"""
const payload=JSON.parse(require("fs").readFileSync(0,"utf8")),bytes=Buffer.from(payload.bytes,"base64");
(async()=>{const {instance}=await WebAssembly.instantiate(bytes,{}),view=new DataView(instance.exports.memory.buffer);
payload.inputs.forEach((value,index)=>view.setFloat64(payload.inputOffsets[index],value,true));
instance.exports.abstract_ui_wheel_contact(0);console.log(JSON.stringify(payload.outputOffsets.map(
  offset=>view.getFloat64(offset,true))));})().catch(error=>{console.error(error);process.exit(1)});
"""
    completed = subprocess.run([node, "-e", script], input=json.dumps({
        "bytes": base64.b64encode(artifact.binary).decode("ascii"),
        "inputs": [values[name] for name in artifact.input_names],
        "inputOffsets": list(artifact.input_offsets), "outputOffsets": list(artifact.output_offsets),
    }), capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    result = dict(zip(artifact.output_names, json.loads(completed.stdout)))
    assert result["chassis_force_y"] > 1000
    assert result["contact_area"] > 0


def test_sympy_contact_precompile_is_one_opt_in_packed_tensor_dispatch():
    compiled = compile_wheel_contact_abstract_tensor(packed_outputs=True)
    regions = [name for name in compiled.module.functions
               if name.startswith("numerical_region_")]
    artifact = compiled.artifacts[0]
    metadata = artifact.api.to_mapping()["metadata"]

    assert regions == ["numerical_region_0"]
    assert compiled.packed_outputs is True
    assert artifact.complete
    # The launch covers every contact lane (40 today: four tyres plus the
    # cage nodes and member midpoints); the workgroup width is the planner's
    # choice for that count, not a fixed number.
    assert artifact.launch_plan.count == contact_lane_count()
    assert artifact.launch_plan.workgroup_size[0] * artifact.launch_plan.groups[0] >= contact_lane_count()
    assert metadata["packed_outputs"] is True
    assert compiled.output_names == CONTACT_PATCH_OUTPUTS
    assert metadata["output_span"][-1] < metadata["output_span"][0]
    assert f"outputs[0u + linear_index] = v_{metadata['output_span'][0]};" in artifact.source
    assert len(metadata["io_layout"]["outputs"]) == 1
    assert len(metadata["output_span"]) == len(CONTACT_PATCH_OUTPUTS)
    # The stage is the compiler's own materialization of the contact SSA: the
    # square root is the SSA Pow with a constant exponent, never a helper
    # such as ``tensor_sqrt`` and never a scalar ``math`` spelling.
    assert "tensor_sqrt" not in compiled.source and "math." not in compiled.source
    assert ".sqrt()" in compiled.source or " ** " in compiled.source


def test_packed_contact_wrench_rows_translate_past_tensor_contact_area():
    assert CONTACT_PATCH_OUTPUTS[:6] == (
        "chassis_force_x", "chassis_force_y", "chassis_force_z",
        "chassis_torque_x", "chassis_torque_y", "chassis_torque_z",
    )
    wheel_wrenches = sympy.Matrix(6, 4, tuple(range(1, 25)))
    reduced = wheel_wrenches * sympy.ones(4, 1)
    assert list(reduced) == [sum(range(start, start + 4))
                             for start in range(1, 25, 4)]
    assert CONTACT_PATCH_TENSOR_REDUCER_OUTPUTS[0] == "contact_area"


def test_vehicle_tensor_precision_closure_is_explicit_and_collapses_at_boundary():
    from src.common.tensors.abstraction import AbstractTensor

    ordinary = extra_precision_closure(lambda value: value + 1, limbs=1)
    widened = extra_precision_closure(
        lambda value: (value * value + 1).sqrt(), limbs=2,
    )
    values = AbstractTensor.get_tensor([0.0, 3.0])

    assert ordinary(values).tolist() == [1.0, 4.0]
    assert widened(values).tolist() == pytest.approx([1.0, 10.0 ** .5])


def test_living_map_has_a_vehicle_slot_not_a_car_specific_control_mode():
    projection = project_class_to_div_map(VehicleProbe, seed="vehicle-probe")
    slot = projection.model["vehicle_slot"]
    vehicle = slot["vehicles"][0]
    creature = projection.model["mechanical_creature"]
    assert creature["subject_class"].endswith(".MechanicalCreature")
    assert creature["active_instance"] == vehicle["identity"]
    assert vehicle["mechanical_creature"] == creature["identity"]
    assert set(creature["meta_objects"]) == {
        "structure", "actuators", "stabilizers", "contact_surfaces", "parametric_engine",
    }
    assert creature["buffer_policy"] == "common-resident-locations-no-inter-stage-host-transfer"
    assert set(MechanicalCreature.__annotations__) == {
        "structure", "actuators", "stabilizers", "contact_surfaces", "parametric_engine",
    }
    assert vehicle["wheel_part"] == "tall-thin-tractor-steel-disc"
    assert [part["identity"] for part in vehicle["wheel_parts"]] == [
        "balloon-black-current", "legacy-small-brown", "tall-thin-tractor-steel-disc",
    ]
    assert all(part["realization"] == "parametric-pneumatic-wheel-renderer"
               for part in vehicle["wheel_parts"])
    brown = next(part for part in vehicle["wheel_parts"]
                 if part["identity"] == "legacy-small-brown")
    assert brown["compound"] == "tacky-race-rubber"
    assert brown["dry_grip_scale"] > 1
    tractor = next(part for part in vehicle["wheel_parts"]
                   if part["identity"] == "tall-thin-tractor-steel-disc")
    assert tractor["rim_profile"] == "solid-steel-plate"
    assert tractor["radius_scale"] == 1
    assert tractor["rim_scale"] == 1
    assert tractor["wheel_mass_kg"] > 60
    assert vehicle["body_shell"] == "clear-polycarbonate-rc"
    assert {shell["identity"] for shell in vehicle["body_shells"]} == {
        "clear-polycarbonate-rc", "fiberglass-monster-pickup", "bare-frame", "six-body-pin-carrier",
    }
    assert all(shell["physics"] is True for shell in vehicle["body_shells"] if shell["identity"] != "bare-frame")
    assert vehicle["fuel_profile"] == "pump-gasoline-93"
    assert {item["identity"] for item in vehicle["fuel_profiles"]} == {
        "pump-gasoline-93", "nitromethane-race", "aviation-gasoline-100-130",
        "ultra-low-sulfur-diesel",
    }
    assert {item["identity"] for item in vehicle["driving_modes"]} == {"trail", "road", "sport"}
    assert vehicle["driving_mode"] == "road"
    assert "tilt-wheelie-suppression" in vehicle["vehicle_computer"]["dispatches"]
    assert vehicle["wiring_harness"]["protection"] == "fusebox-and-relay-dispatch"
    assert any(preset["identity"] == "monster-540-blown-methanol"
               for preset in vehicle["power_unit_presets"])
    assert any(preset["identity"] == "monster-632-twin-turbo"
               for preset in vehicle["power_unit_presets"])
    merlin = next(preset for preset in vehicle["power_unit_presets"]
                  if preset["identity"] == "packard-merlin-v1650")
    assert merlin["architecture"]["layout"] == "sixty-degree-v12"
    assert merlin["configuration"]["displacement_liters"] == pytest.approx(27.04)
    assert merlin["preferred_fuel_profile"] == "aviation-gasoline-100-130"
    assert merlin["fuel_compatibility"]["pump-gasoline-93"] < .5
    heavy_diesel = next(preset for preset in vehicle["power_unit_presets"]
                        if preset["identity"] == "cat-c18-industrial-diesel")
    assert heavy_diesel["configuration"]["engine_mass_kg"] == 1673
    assert heavy_diesel["configuration"]["torque_peak_rpm"] == 1400
    assert heavy_diesel["fuel_compatibility"]["pump-gasoline-93"] < .05
    assert vehicle["power_unit_preset"] == "amc-258-jeep-i6"
    assert vehicle["engine_kernel_switch"]["runtime_compilation"] is False
    assert vehicle["engine_kernel_switch"]["default_equation_mode"] == "linear-playable"
    assert vehicle["engine_kernel_switch"]["equation_modes"] == ["linear-playable", "symbolic-fidelity"]
    assert len(vehicle["engine_kernel_switch"]["cases"]) == len(vehicle["power_unit_presets"])
    assert {case["selector"] for case in vehicle["engine_kernel_switch"]["cases"]} == set(
        range(len(vehicle["power_unit_presets"])))
    jeep_i6 = next(preset for preset in vehicle["power_unit_presets"]
                   if preset["identity"] == "amc-258-jeep-i6")
    assert jeep_i6["architecture"]["firing_order"] == [1, 5, 3, 6, 2, 4]
    assert any(preset["identity"] == "honda-style-commuter-i4-1500"
               for preset in vehicle["power_unit_presets"])
    assert vehicle["transmission_preset"] == "cj-wide-four-speed"
    assert vehicle["clutch_preset"] == "old-soft-organic"
    assert next(item for item in vehicle["clutch_presets"] if item["default"])["stiffness_nm_per_rad_s"] == 4.2
    assert next(item for item in vehicle["clutch_presets"]
                if item["identity"] == "industrial-twin-disc")["maximum_torque_nm"] == 4800
    assert all(preset["architecture"]["mount_vibration_reference"]
               for preset in vehicle["power_unit_presets"])
    assert all(preset["energy_system"]["mass_authority"].startswith("storage-shell")
               for preset in vehicle["power_unit_presets"])
    monster = next(preset for preset in vehicle["power_unit_presets"]
                   if preset["identity"] == "monster-632-twin-turbo")
    assert monster["architecture"]["cylinders"] == 8
    assert monster["architecture"]["banks"] == 2
    assert monster["energy_system"]["conversion"].startswith("intake-air")
    servo = next(preset for preset in vehicle["power_unit_presets"]
                 if preset["identity"] == "servo-direct-drive-400")
    assert servo["kind"] == "servo-electric"
    assert {"propulsion", "steering-assist", "chassis-articulation",
            "auxiliary-actuation"} <= set(servo["mechanical_roles"])
    assert vehicle["steering_control"]["front_axle_enabled"] is True
    assert vehicle["steering_control"]["rear_axle_enabled"] is True
    assert vehicle["steering_control"]["rear_phase"] == -1.0
    assert vehicle["steering_control"]["velocity_rate_control_enabled"] is True
    assert vehicle["steering_control"]["parking_steering_rate_per_s"] > vehicle["steering_control"]["highway_steering_rate_per_s"]
    assert "velocity-sensitive-steering-rate" in vehicle["vehicle_computer"]["dispatches"]
    assert vehicle["pleasure_course"]["kind"] == "blue-meandering-wave-road"
    assert vehicle["pleasure_course"]["maximum_height"] == pytest.approx(2.25)
    assert vehicle["loop_test"]["kind"] == "blue-stepped-nonadhesive-loop"
    assert len(vehicle["loop_test"]["segments"]) == 32
    assert vehicle["loop_test"]["adhesion"] is False
    graph = vehicle["physics"]["mechanical_graph"]
    assert any(node["identity"] == "electrical.ecu" for node in graph["nodes"])
    assert any(node["identity"] == "lighting.tail.left.center" for node in graph["nodes"])
    assert any(edge["identity"] == "electrical.wire.alternator_charge" for edge in graph["edges"])
    assert any(edge["identity"] == "hydraulics.line.front_left" for edge in graph["edges"])
    assert any(edge["identity"] == "pneumatics.tire_line.rear_right" for edge in graph["edges"])
    assert any(node["identity"] == "electrical.steering_servo" for node in graph["nodes"])
    assert any(edge["identity"] == "electrical.wire.steering_servo_feed" for edge in graph["edges"])
    assert any(edge["identity"] == "steering.assist.motor_to_column" for edge in graph["edges"])
    nodes = {node["identity"] for node in graph["nodes"]}
    assert {"steering.proportioner", "steering.rear_pinion", "steering.rear_rack.center"} <= nodes
    edges = {edge["identity"]: edge for edge in graph["edges"]}
    assert edges["drivetrain.engine_to_alternator_cvt"]["drive"] == "no-belt"
    assert edges["drivetrain.alternator_cvt_to_bank"]["ratio_coordinate"] == (
        "alternator_cvt_ratio_state")
    wire = edges["electrical.wire.alternator_charge"]
    hydraulic = edges["hydraulics.line.front_left"]
    pneumatic = edges["pneumatics.tire_line.rear_right"]
    assert wire["routing"] == "relaxed-multi-segment-harness"
    assert wire["slack_ratio"] > pneumatic["slack_ratio"] > hydraulic["slack_ratio"]
    assert wire["relaxation_rate_hz"] < pneumatic["relaxation_rate_hz"] < hydraulic["relaxation_rate_hz"]
    assert hydraulic["pressure_rating_pa"] > pneumatic["pressure_rating_pa"]
    assert edges["actuator.throttle.cable_0"]["subsystem_class"] == "routed-control-actuator"
    assert edges["actuator.throttle.cable_0"]["tension_only"] is True
    assert edges["actuator.throttle.lever"]["travel_table"][-1] == [1.0, .038]
    assert graph["execution_bands"]["medium_critical_30_to_60_hz"]
    assert edges["steering.proportioner.rear"]["subsystem_class"] == "rotary-transmission"
    assert edges["suspension.rear_left.tie_rod"]["subsystem_class"] == "articulation-linkage"
    for corner in WHEEL_NAMES:
        assert edges[f"drivetrain.{corner}_halfshaft"]["b"] == (
            f"suspension.{corner}.halfshaft_joint")
        assert edges[f"suspension.{corner}.outer_halfshaft_joint"]["a"] == (
            f"suspension.{corner}.halfshaft_joint")
        assert edges[f"suspension.{corner}.outer_halfshaft_joint"]["b"] == f"suspension.{corner}.hub"
        assert edges[f"suspension.{corner}.steering_arm"]["a"] == f"suspension.{corner}.knuckle"
    assert edges["drivetrain.front_differential_brake"]["modulation"].startswith("abs-authority")
    assert edges["drivetrain.rear_differential_brake"]["command"] == "rear_differential_brake"
    assert edges["drivetrain.front_differential_brake"]["rotor_polar_inertia_kg_m2"] > 0
    assert edges["drivetrain.front_differential_brake"]["momentum_integration_status"].startswith(
        "integrated-live")
    coilover = edges["suspension.front_left.coilover"]
    assert coilover["visualization"]["kind"] == "helical-coilover"
    assert coilover["static_preload_compression_m"] > 0
    terrain = next(box for box in projection.model["document_geometry"]["boxes"]
                   if box["identity"] == vehicle["offroad_terrain"])
    courtyard = next(box for box in projection.model["document_geometry"]["boxes"]
                     if box["kind"] == "courtyard")
    boxes = projection.model["document_geometry"]["boxes"]
    assert terrain["parent_identity"].endswith("/representation:global")
    assert terrain["half_extent"] == [28.0, 28.0]
    assert terrain["surface"]["resolution"] == [81, 81]
    assert max(terrain["surface"]["cell_size"]) <= .71
    assert sum(box["kind"] == "courtyard" for box in boxes) == 1
    assert not any(box["surface"]["kind"] == "sampled-height-field"
                   and box["parent_identity"] == courtyard["identity"]
                   for box in boxes if box.get("surface"))
    assert vehicle["driving_area"]["geometry"] == "world-floor-no-boundary-box"
    spawn_x, _, spawn_z = vehicle["pose"]["position"]
    terrain_x, terrain_z = terrain["center"]
    assert abs(spawn_x - terrain_x) <= terrain["half_extent"][0]
    assert spawn_z > terrain_z + terrain["half_extent"][1]
    practice_ramps = [box for box in boxes if "practice-ramp" in box["identity"]]
    assert len(practice_ramps) == 2
    course = vehicle["elevated_skill_course"]
    course_boxes = [box for box in boxes if box["identity"] in {
        *course["approaches"], *course["slabs"],
    }]
    assert course["kind"] == "thin-clear-ramp-road"
    assert course["depth_map"] == "local-ramp-tops-only"
    assert course["maximum_grade_percent"] < 20
    assert len(course["approaches"]) == 2
    assert course["slabs"] == []
    assert len(course["disabled_opaque_slabs"]) == 6
    approaches = [box for box in course_boxes if box["identity"] in course["approaches"]]
    slabs = [box for box in course_boxes if box["identity"] in course["slabs"]]
    assert all(box["surface"]["kind"] == "sampled-height-field" for box in approaches)
    assert all(box["surface"]["features"]["support"] == "full-depth-slab" for box in approaches)
    assert slabs == []
    assert all(box["physics"]["collider"] == "solid-contact-surface" for box in course_boxes)
    inventory_item = next(item for item in projection.model["inventory"]["items"] if item["slot"] == 9)
    depth_item = next(item for item in projection.model["inventory"]["items"] if item["slot"] == 10)
    assert inventory_item["properties"]["operation"] == "mount-vehicle-slot"
    assert depth_item["name"] == "Depth map"
    assert vehicle["physics"]["parallel_spring_lanes"] == list(WHEEL_NAMES)
    fallback_capabilities = {plugin["capability"] for plugin in projection.model["world"]["plugins"]}
    assert {"vehicle-physics", "vehicle-contact-fallback"} <= fallback_capabilities
    assert vehicle["physics"]["wasm_fallback"]["selection"] == (
        "worker-webgpu-unavailable-or-runtime-fault"
    )
    assert vehicle["physics"]["contact_fallback"]["plugin"]
    webgpu_program = next(
        program for program in slot["programs"]
        if program["identity"] == vehicle["physics"]["webgpu_program"]
    )
    wrench = webgpu_program["wrench_reduction"]
    integration = webgpu_program["vehicle_integration"]
    tensor_contact = webgpu_program["tensor_contact_precompile"]
    assert tensor_contact["packed_outputs"] is True
    assert len(tensor_contact["kernel"]["io"]["outputs"]) == 1
    assert len(tensor_contact["kernel"]["output_span"]) == len(CONTACT_PATCH_OUTPUTS)
    assert tensor_contact["storage_rows"] == {
        name: index for index, name in enumerate(CONTACT_PATCH_OUTPUTS)
    }
    assert tensor_contact["reducer_output_order"] == list(CONTACT_PATCH_TENSOR_REDUCER_OUTPUTS)
    assert tensor_contact["precompile_output_translation"] == "reducer-publications-to-authored-contact-abi"
    assert tensor_contact["wrench_row_start"] == 0
    assert tensor_contact["wrench_row_count"] == 6
    published_contact_values = len(CONTACT_PATCH_OUTPUTS) * len(WHEEL_NAMES)
    assert wrench["input_shapes"] == [[6, published_contact_values], [published_contact_values, 1]]
    assert wrench["output_shape"] == [6, 1]
    assert ".matmul(" in wrench["abstract_tensor_source"]
    assert wrench["kernel"]["backend_variant"] == "webgpu_tiled_gemm"
    assert wrench["kernel"]["problem_shape"] == {
        "m": 6, "n": 1, "k": published_contact_values,
    }
    assert wrench["kernel"]["scalars"] == {"alpha": 1.0, "beta": 0.0}
    assert integration["outputs"] == list(VEHICLE_STATE_OUTPUTS)
    assert integration["state_residency"] == "gpu-persistent-with-passive-presentation-snapshots"
    assert integration["host_transfers_between_stages"] == 0
    assert integration["shared_output_buffer"].startswith("one-allocation")
    assert integration["default_specialization"]["pipeline_swap_moves_state"] is False
    assert integration["default_specialization"]["folded_inputs"] == []
    assert integration["default_specialization"]["fixed_inputs"] == []
    assert integration["engine_profile_variants"] == []
    assert integration["engine_profile_dispatch"] == (
        "durable-profile-parameters-feed-single-live-parametric-kernel")
    assert vehicle["power_unit_presets"]
    assert [stage["identity"] for stage in integration["stages"]] == [
        "chassis-transition", "tire-suspension-control", "powertrain-reactions",
    ]
    assert all(len(stage["kernel"]["io"]["feeds"]) == 1 for stage in integration["stages"])
    assert all(len(stage["kernel"]["io"]["outputs"]) == 1 for stage in integration["stages"])
    assert all("parametric_kernel" not in stage for stage in integration["stages"])
    assert set(integration["output_slots"]) == set(VEHICLE_STATE_OUTPUTS)
    assert all(stage["output_offset_floats"] % 64 == 0 for stage in integration["stages"])
    assert "var<workgroup> tile_A" in wrench["kernel"]["source"]
    assert projection.model["motion_cues"]["world_tiling"]["physics_effect"] == "none"
    assert projection.model["motion_cues"]["vehicle_camera"]["writes"] == ["presentation-camera"]
    assert projection.model["motion_cues"]["camera_depth"]["format"] == "DEPTH_COMPONENT24"
    assert projection.model["motion_cues"]["camera_depth"]["resolution"] == "half-viewport"
    script = projection.javascript
    assert "architectureSpecs=[" in script
    assert '"routed-tension-cable"' in script
    assert 'body.appliedThrottle=Number(body.effectiveThrottle' in script
    assert 'body.appliedThrottle=secondOrderChannel(body,"appliedThrottle"' not in script
    assert "function updateVehicleChaseCamera" in script
    assert "function initializeVehicleFirstExperience" in script
    assert "function armBrowserFullscreenOnFirstGesture" in script
    assert "initializeVehicleFirstExperience();" in script
    assert 'setShaderOnlyMode(true)' in script
    assert 'worker.postMessage({type:"vehicle-disable-gpu"' in script
    assert '"body-shell-glass"' in script
    assert "gl.uniform1i(renderPass,2)" in script
    assert "function renderCameraDepthPass" in script
    assert 'box.geometry_mode==="vehicle-wheel"' in script
    assert "wheelAngles[index]+omega" in script
    assert "vWorldPosition.xz/tileSize" in script
    assert terrain["surface"]["kind"] == "sampled-height-field"
    assert "gradient_test_solid" not in vehicle
    assert "function sampleDeclaredSurface" in script
    assert "function applyDepthMapTool" in script
    assert 'recover.textContent="RIGHT CAR"' in script
    assert 'respawn.textContent="RESPAWN"' in script
    assert "function respawnActiveVehicleAtAuthoredPose" in script
    assert "respawnActiveVehicleAtOrigin" not in script
    assert 'vehicleSpawnState(vehicle,false,{reuseParked:false})' in script
    assert 'viewportControls.yaw=spawn.yaw' in script
    assert 'model.vehicle_slot?.initial_state?.placement==="authored-world-pose"' in script
    assert 'const placeAtActor=model.vehicle_slot?.initial_state?.placement==="at-player-spawn"' in script
    assert 'setActiveVehicle(item.properties.vehicle,{placeAtActor:true,inventoryItem:item})' not in script
    assert 'position:[...state.position],yaw:state.yaw,roll:state.roll,pitch:state.pitch' in script
    assert "respawned at world origin" not in script
    assert 'box.geometry_mode==="sampled-height-field-prism"' in script
    assert "...vehicleRuntime.rollCageBoxes" in script
    assert "...vehicleRuntime.powertrainBoxes" in script
    assert 'kind:"vehicle-frame-member"' in script
    assert 'kind:"vehicle-mechanical-edge"' in script
    assert 'geometry_mode:"vehicle-link"' in script
    assert "solveVehicleMechanicalGraph" in script
    assert "controlVehicleTransmission" in script
    assert '[["HIGH","high"],["L1","l1"],["L2 CRAWL","l2"]]' in script
    assert 'clutchSelect.dataset.clutchPreset="true"' in script
    assert '"CENTER LOCK","centerDiffLock"' in script
    assert 'dataset.frontDriveShare="true"' in script
    assert 'powerUnitSelect.dataset.powerUnitPreset="true"' in script
    assert 'wheelPartSelect.dataset.wheelPart="true"' in script
    assert 'vehicleRuntime.wheelPart!=="balloon-black-current"' in script
    assert "function selectVehicleWheelPart" in script
    assert 'button.textContent=label' in script
    assert "rebuildPortableSceneMesh({dynamicOnly:true})" in script
    assert "removeVehicleInventoryItem" in script
    assert 'event.code==="KeyV"' in script
    assert "viewportControls.yaw=state.yaw" not in script
    assert 'reportRuntimeFault("vehicle-step"' in script
    assert 'type: "player-jump"' in script
    assert "supportSuppressedUntil" in script
    assert 'SMOOTH LAUNCH' in script
    assert 'FL HOLD' in script and 'RR HOLD' in script
    assert 'reportRuntimeFault(vehicleRuntime.active?"mounted-frame":"world-frame"' in script
    assert 'disabledPresentationStages.add("wheel-shader")' in script
    assert "function synchronizeVehicleLookYaw" in script
    assert 'primitive:"balloon-tire-sidewall-and-tread"' in script
    assert 'primitive:"heavy-six-spoke-wheel-hub-and-brake"' in script
    assert "state.frontKnuckleSteerAngle||0" in script
    assert "-Number(steering||0)*maximum*frontGain" not in script
    assert 'type:"vehicle-fuel-ignition"' in script
    assert 'type:"vehicle-driver-assistance"' in script
    assert "function controlVehicleDriverAssistance" in script
    assert 'modeSelect.dataset.drivingMode="true"' in script
    assert 'governor.dataset.governorRpm="true"' in script
    assert 'cruise.dataset.cruiseToggle="true"' in script
    assert '["TILT","tiltEnabled"]' in script
    assert 'vehicleRuntime.electrical.brakeLightsOn' in script
    assert "uniform vec3 uTailLightLeft" in script
    assert "uniform float uBrakeLightActive" in script
    assert "vec3(1.0,.055,.018)*tailLight" in script
    assert "vehicleRuntime.electrical.tailLightsOn?1:0" in script
    assert "vehicleRuntime.electrical.brakeLightsOn?1:0" in script
    assert "function applyVehicleHydraulicPose" in script
    assert "function programVehicleHydraulicPose" in script
    assert "function controlVehicleTirePressure" in script
    assert 'pressure.dataset.tirePressure="true"' in script
    assert "deformedRadius=(angle,value)" in script
    assert "gl_FrontFacing?vNormal:-vNormal" in script
    assert "blueCourse*.48" in script
    assert "annularDiscX(center" in script
    assert '"steering-servo","vehicle-steering-servo"' in script
    assert 'box.mechanical_edge?.routing==="relaxed-multi-segment-harness"' in script
    assert "state.routeLocalPoints?.length>=2" in script
    assert '"flexible-hydraulic-hose","flexible-air-line"' in script
    assert "1-Math.exp(-rate*elapsed)" in script
    assert "function updateParkedVehicle(dt)" in script
    assert 'type:"remove",identity:state.identity' not in script[script.index("function clearActiveVehicle"):script.index("function recoverActiveVehicle")]
    assert "state.wheelSteerAngles?.[name]" in script
    assert "state.steeringWrench?.frontRackTravel" in script
    assert 'collider:"subdivided-shell-samples"' in script
    assert "Math.cos(spin),ss=Math.sin(spin)" in script
    assert "design_supported_mass_kg" in script
    assert 'details.textContent="STATS"' in script
    assert 'TC 0%' in script and 'ABS 0%' in script
    assert 'tractionIntervention=(1-tractionScale)*100' in script
    assert 'brakeIntervention=(1-brakeScale)*100' in script
    assert 'indicator.setAttribute("aria-valuenow",intervention.toFixed(1))' in script
    assert "const VEHICLE_HUD_VERTEX_SHADER" in script
    assert "function drawVehicleShaderHud" in script
    assert "drawVehicleShaderHud(gl,width,height)" in script
    assert "function drawVehicleCanvasHud" in script
    assert 'if(!liveDom)return' in script
    assert 'if(!shaderViewer.shaderOnly&&vehicleRuntime.contactMonitor?.classList.contains("expanded"))updateVehicleContactMonitor();' in script
    assert 'if(vehicleRuntime.contactMonitor)vehicleRuntime.contactMonitor.hidden=false;' in script
    assert 'if(vehicleRuntime.contactMonitor)vehicleRuntime.contactMonitor.hidden=true;' in script
    assert 'kind:"vehicle-driver-seat"' not in script
    assert 'shaderViewer.geometry.push(...vehicleRuntime.frameBoxes' in script
    assert 'const mountedVehicleGeometry=vehicleRuntime.active?[' in script
    assert 'const geometry = [...baseGeometry,...mountedVehicleGeometry]' in script
    assert '["transmission-shaft","vehicle-transmission"' in script
    assert '["front-differential","vehicle-differential"' in script
    assert 'steer:index<2?' in script
    assert "basePitch+viewportControls.pitch" in script
    assert "const config=vehicle.configuration,chassis=config.chassis" in script
    assert slot["terrain_sampling"]["service"] == projection.model["contact_surfaces"]["identity"]
    assert projection.model["contact_surfaces"]["consumers"][0] == "platformer-body"
    assert vehicle["configuration"]["tires"]["pressure_pa"] == 135000.0
    assert vehicle["configuration"]["presentation"]["wheel_palette_role"] == "drivetrain-black"
    worker = projection.model["loop_deployment"]["workers"][0]["source"]
    assert "dispatchVehicleContacts" in worker
    assert 'mode:"packed-contact-gemm"' in worker
    assert "contactPass.dispatchWorkgroups" in worker
    assert "wrenchMatrix" not in worker
    assert "reductionPass.dispatchWorkgroups" in worker
    assert "result.chassis_wrench" in worker
    assert "contacts.chassis_wrench" in worker
    assert "dispatchVehicleContactWasm" in worker
    assert "lockstep-worker-contact-wasm+vehicle-wasm" in worker
    assert "function vehicleDynoRecords" in worker
    assert 'm.type==="vehicle-dyno"' in worker
    assert 'type:"vehicle-dyno-result"' in worker
    assert 'dynoButton.textContent="DYNO"' in script
    assert "function vehicleTerrainRestPose" in script
    assert "wrenchMatrix" not in script
    fallback = vehicle["physics"]["contact_fallback"]
    fallback_plugin = next(plugin for plugin in projection.model["world"]["plugins"]
                           if plugin["identity"] == fallback["plugin"])
    assert fallback_plugin["capability"] == "vehicle-contact-fallback"
    assert fallback_plugin["abi"]["authority"] == "same-symbolic-law-as-primary-webgpu-contact-kernel"
    assert "if(!Number.isFinite(state.shiftAge))" in worker
    assert "ultra_low_range_ratio" in worker
    assert 'body.transmission?.frontDiffMode==="limited-slip"?.32' in worker
    assert 'body.transmission?.rearDiffMode==="limited-slip"?.32' in worker
    assert 'body.transmission?.centerDiffMode==="limited-slip"?.32' in worker
    assert "traction_control_authority" in worker
    assert "abs_authority" in worker
    assert '"chassis_torque_x","chassis_torque_y","chassis_torque_z"' in worker
    assert "attachment[1]*force[2]-attachment[2]*force[1]" not in worker
    assert "geometric_compression:geometricCompression" in worker
    assert "c2Unit((alignment-.18)/.34)" in worker
    assert "alignment>.18" not in worker
    assert "target_compression_${name}" in worker
    assert "distanceAlongSuspension>0" in worker
    assert "resolveVehicleSolidContact" in worker
    assert "vehicleCageContactWrench" in worker
    assert "resolveVehicleCagePenetration" in worker
    assert 'node.kind==="roll-cage-node"' in worker
    assert "cage_contact_stiffness" in worker
    assert "resolveWorldBottom" in worker
    assert "swept=previousY>=boundary&&nextY<bottom" in worker
    assert "contactSurfaceAt" in worker
    assert "function sampleSurface(s,x,z)" in worker
    assert 's?.kind==="sampled-height-field"' in worker
    assert "function radialTireContact" in worker
    assert "radialAngles=[-.95,-.48,0,.48,.95]" in worker
    assert "lateralFractions=[-.38,0,.38]" in worker
    assert "rollingRaw=[axle[1]*radialOut[2]" in worker
    assert "cage_static_friction" in worker and "cage_kinetic_friction" in worker
    assert "function installVehiclePresentationMesh" in script
    assert "buildExtrudedBoxMesh(dynamicGeometry" in script
    assert "drawSceneMeshes(gl)" in script
    assert "targetSpeed" not in worker
    assert "wheelOmegas" in worker
    assert "total_torque_y" in worker
    assert "rotateBodyVector" in worker
    assert "centerOfMass=c.mechanical_graph?.load_audit?.center_of_mass" in worker
    assert "steer=i<2?steeringAngle:0" in worker
    assert "steeringAngle=-(body.appliedSteering||0)" in worker
    assert "steeringAngle=-(body.controls?.steering||0)" not in worker
    assert "previous_slip_longitudinal_" in worker
    assert "friction_utilization_" in worker
    assert "body.tractionScales" in worker
    assert "frontDifferentialTorque:out.front_differential_torque" in worker
    assert 'm.type==="vehicle-recover"' in worker
    assert 'm.type==="vehicle-respawn"' in worker
    assert 'm.type==="vehicle-chassis-profile"' in worker
    assert 'm.type==="vehicle-chassis-geometry"' in worker
    assert 'm.type==="vehicle-chassis-leveling"' in worker
    assert "function updateVehicleChassisLeveling(body,dt)" in worker
    assert "body.levelingOffsets" in worker
    assert "function selectVehicleChassisProfile" in script
    assert "function selectVehicleChassisGeometry" in script
    assert 'input.dataset.chassisGeometry=key' in script
    assert "function controlVehicleChassisLeveling" in script
    assert 'action:"chassis-leveling"' in script
    assert "body.damperScales" in worker
    assert "resolveVehicleSuspensionTravelStop(body)" in worker
    assert "support.height+s.rest_length+actuator-s.travel-attachment[1]" in worker
    channel = projection.model["loop_deployment"]["channels"][0]
    assert channel["record_layout"][8] == "vehicle.roll"
    structure = vehicle["physics"]["chassis_structure"]
    assert len(structure["nodes"]) == 4
    assert len(structure["members"]) == 6
    assert structure["pose_reduction"].startswith("sum-node-forces")
    assert structure["material"]["solver_interpretation"].startswith("finite-axial-stiffness")
    graph = vehicle["physics"]["mechanical_graph"]
    assert graph["state_law"].startswith("node-force-and-node-moment")
    assert all(node["wrench"].keys() == {"force", "moment"} for node in graph["nodes"])
    profiles = vehicle["chassis_profiles"]
    assert len(profiles) == 4
    assert vehicle["chassis_profile"] == "dom-44x3"
    assert {profile["material"] for profile in profiles} == {
        "AISI 4130 chromoly", "1020 DOM steel", "A36 mild steel", "6061-T6 aluminum",
    }
    assert all(profile["outer_diameter_m"] > 2 * profile["wall_thickness_m"] > 0
               for profile in profiles)
    assert all(profile["member_mass_kg"] > 0 and profile["axial_yield_force_n"] > 0
               for profile in profiles)
    chassis_members = [edge for edge in graph["edges"] if edge.get("chassis_profile_member")]
    assert chassis_members
    assert all(edge["identity"].startswith(("frame.", "cage.")) for edge in chassis_members)
    assert all(edge["damage"]["axial_stiffness_n_per_m"] > 0 for edge in chassis_members)
    assert all(edge["mass_kg"] > 0 and edge["mass_in_total"] for edge in chassis_members)
    assert all(edge["mass_accounting"] == "allocated-within-frame-cage-driver-residual"
               for edge in chassis_members)
    assert all(edge["broken_mass_policy"].startswith("mass-remains-split")
               for edge in chassis_members)
    assert sum(edge["mass_kg"] for edge in chassis_members) == pytest.approx(
        next(profile for profile in profiles if profile["identity"] == "dom-44x3")["member_mass_kg"])
    structural_nodes = [node for node in graph["nodes"]
                        if node["identity"].startswith(("frame.", "cage."))]
    assert all(node.get("structural_deformable") for node in structural_nodes)
    assert all(edge.get("joint_bushings") for edge in chassis_members)
    assert all(edge["joint_bushings"][end]["linear_damping_n_s_per_m"] > 0
               for edge in chassis_members for end in ("a", "b"))
    assert all(edge["joint_bushings"][end]["parameter_pack"] == "performance-polyurethane-calculated-static-v2"
               for edge in chassis_members for end in ("a", "b"))
    assert all(edge["joint_bushings"][end]["frame_mount"]
               for edge in chassis_members for end in ("a", "b"))
    assert all(edge["joint_bushings"][end]["preload_force_n"] > 0
               and edge["joint_bushings"][end]["yield_force_n"] >
                   edge["joint_bushings"][end]["preload_force_n"]
               and edge["joint_bushings"][end]["fracture_force_n"] >
                   edge["joint_bushings"][end]["yield_force_n"]
               for edge in chassis_members for end in ("a", "b"))
    geometry = vehicle["chassis_geometry_parameters"]
    assert geometry["chassis_length_m"] == pytest.approx(1.44)
    assert geometry["wheelbase_m"] == pytest.approx(1.24)
    assert all(node.get("longitudinal_parameterization") for node in graph["nodes"])
    assert "!node.structural_deformable" in worker
    assert "jointBushingDissipationPowerW" in worker
    assert "linearDamping*linearSpeed*linearSpeed" in worker
    assert 'm.type==="vehicle-wheel-alignment"' in worker
    assert "function updateVehicleWheelAlignment(body,dt)" in worker
    assert "trustedContinuous" in worker
    assert "body.alignmentStrainReliefState" in worker
    assert "body.alignmentActuatorReliefHeatJ" in worker
    assert "body.alignmentActuatorPumpPowerW" in worker
    assert "sacrificial_break_bushing" in worker
    assert 'edge.constraint==="replaceable-sacrificial-knuckle-bushing"' in worker
    assert "Number(body.linkLengthModifiers[edge.identity]||0)+" in worker
    assert "function controlVehicleWheelAlignment" in script
    assert '"FULL AUTO","full-time-auto"' in script
    assert "function selectVehicleEngineEquationMode" in script
    assert '"linear-playable","LINEAR · PLAYABLE"' in script
    assert "engineProfileVariants" in worker
    assert "compiled-engine-${variant.identity}-${stage.identity}" in worker
    assert "powerUnitMassDeltaKg" in worker
    assert "fuelCompatibility" in worker
    assert "ignitionCompatibility" in worker
    leveling = vehicle["chassis_leveling"]
    assert leveling["force_law"].startswith("pose-changes-only-through-existing-spring")
    assert leveling["maximum_actuator_rate_m_s"] < .1
    alignment = vehicle["wheel_alignment"]
    assert alignment["actuator_object"] == "force-limited-alignment-strain-relief-v1"
    assert alignment["break_bushing_object"] == "replaceable-knuckle-break-bushing-v1"
    actuator_family = graph["actuator_family"]
    relief_definition = actuator_family[alignment["actuator_object"]]
    fuse_definition = actuator_family[alignment["break_bushing_object"]]
    assert relief_definition["holding_force_n"] < relief_definition["relief_force_n"]
    assert relief_definition["maximum_relief_stroke_m"] > 0
    assert fuse_definition["yield_force_n"] < fuse_definition["fracture_force_n"]
    relief_edges = [edge for edge in graph["edges"]
                    if edge.get("alignment_strain_relief_actuator")]
    fuse_edges = [edge for edge in graph["edges"] if edge.get("sacrificial_break_bushing")]
    assert len(relief_edges) == 12
    assert len(fuse_edges) == 12
    assert all(edge["linear_actuator"]["authority"].startswith("alignment-and-lvl")
               for edge in relief_edges)
    assert all(edge["constraint"] == "replaceable-sacrificial-knuckle-bushing"
               for edge in fuse_edges)
    for corner in WHEEL_NAMES:
        upper = next(edge for edge in fuse_edges
                     if edge["identity"] == f"suspension.{corner}.upper_break_bushing")
        lower = next(edge for edge in fuse_edges
                     if edge["identity"] == f"suspension.{corner}.lower_break_bushing")
        tie = next(edge for edge in fuse_edges
                   if edge["identity"] == f"suspension.{corner}.tie_rod_break_bushing")
        assert upper["a"] == f"suspension.{corner}.upper_ball_joint"
        assert upper["b"] == f"suspension.{corner}.upper_knuckle_socket"
        assert lower["a"] == f"suspension.{corner}.lower_ball_joint"
        assert lower["b"] == f"suspension.{corner}.lower_knuckle_socket"
        assert tie["a"] == f"suspension.{corner}.tie_rod_outer"
        assert tie["b"] == f"suspension.{corner}.steering_arm"
        protected_yield = min(
            float(edge["damage"]["axial_yield_force_n"])
            for edge in graph["edges"] if edge["identity"] in {
                f"suspension.{corner}.upper_arm_forward",
                f"suspension.{corner}.upper_arm_rear",
                f"suspension.{corner}.tie_rod",
            })
        assert fuse_definition["fracture_force_n"] < protected_yield
    mass_nodes = {node["identity"]: node["mass_kg"] for node in graph["nodes"]
                  if node.get("mass_in_total")}
    assert mass_nodes["powertrain.engine"] == 220
    assert mass_nodes["powertrain.transmission"] == 42
    assert mass_nodes["powertrain.transfer_case"] == 24
    assert mass_nodes["powertrain.front_differential"] == 18
    assert mass_nodes["powertrain.rear_differential"] == 20
    assert all(mass_nodes[f"suspension.{corner}.hub"] == pytest.approx(67.975)
               for corner in WHEEL_NAMES)
    for corner in WHEEL_NAMES:
        skin_total = sum(mass for identity, mass in mass_nodes.items()
                         if identity.startswith(f"suspension.{corner}.tire_skin.vertex_"))
        assert skin_total == pytest.approx(14.0)
    assert all(mass_nodes[f"suspension.{corner}.knuckle"] == 9.5 for corner in WHEEL_NAMES)
    assert all(mass_nodes[f"suspension.{corner}.brake_caliper"] == 4.2 for corner in WHEEL_NAMES)
    assert all(mass_nodes[f"suspension.{corner}.brake_rotor"] == 7.4 for corner in WHEEL_NAMES)
    assert all(mass_nodes[f"suspension.{corner}.lower_ball_joint"] == 3.6 for corner in WHEEL_NAMES)
    assert all(mass_nodes[f"suspension.{corner}.coilover_chassis"] == 4.4 for corner in WHEEL_NAMES)
    for corner in WHEEL_NAMES:
        assert mass_nodes[f"suspension.{corner}.pneumatic_rotary_union"] == pytest.approx(.55)
        assert mass_nodes[f"suspension.{corner}.pneumatic_service_loop"] == pytest.approx(.18)
        assert mass_nodes[f"suspension.{corner}.brake_service_port"] == pytest.approx(.16)
        assert mass_nodes[f"suspension.{corner}.alignment_service_port"] == pytest.approx(.20)
    assert '"front-half-shaft","vehicle-half-shaft"' not in script
    assert graph["drivetrain_wrench_api"]["ports"] == [
        "powertrain.pre_clutch_flywheel_wrench",
        "powertrain.front_differential_brake_wrench",
        "powertrain.rear_differential_brake_wrench",
    ]
    for axle in ("front", "rear"):
        shaft = next(edge for edge in graph["edges"] if edge["identity"] ==
                     f"drivetrain.{axle}_differential_brake_shaft_extension")
        assert shaft["constraint"] == "torque-shaft-wrench-extension"
    load_audit = graph["load_audit"]
    assert load_audit["spring_load_sum_n"] == pytest.approx(
        load_default_car_configuration().sprung_mass() * 9.81)
    assert load_audit["corners"]["rear_left"]["unsprung_mass_kg"] == pytest.approx(108.145)
    assert abs(load_audit["configured_vs_derived_front_fraction_error"]) < .03
    assert load_audit["corners"]["rear_left"]["design_supported_mass_kg"] > (
        load_audit["corners"]["front_left"]["design_supported_mass_kg"])
    constraints = {edge["constraint"] for edge in graph["edges"]}
    assert {"rigid-distance", "spring-damper", "constant-velocity-torque-shaft",
            "six-axis-compliant-mount"} <= constraints
    assert sum(edge["identity"].startswith("body_shell.mount.") for edge in graph["edges"]) == 4
    assert sum(node["kind"].startswith("body-shell-contact-") for node in graph["nodes"]) == 12
    turret = next(shell for shell in vehicle["body_shells"] if shell["identity"] == "six-body-pin-carrier")
    assert len(turret["turrets"]) == 6
    body_pin_locks = [node for node in graph["nodes"]
                      if node["kind"] == "rc-body-pin-lock-and-gimbal-interface"]
    assert len(body_pin_locks) == 6
    assert all(node.get("fixed_to") != "chassis" for node in body_pin_locks)
    assert all(node["generalized_coordinate"].startswith("body_pin_compression_")
               for node in body_pin_locks)
    assert turret["armor"]["mass_kg"] > 100
    assert turret["assembly_mass_kg"] > turret["armor"]["mass_kg"]
    ammo = turret["ammunition"]
    assert ammo["initial_count"] * ammo["round_mass_kg"] <= ammo["capacity_mass_kg"]
    assert ammo["initial_count"] * ammo["round_volume_m3"] <= ammo["capacity_volume_m3"]
    assert turret["fire_control"]["primary_fire_takeover_default"] is True
    assert turret["outriggers"]["anchor"].startswith("persistent-six-axis")
    assert turret["outriggers"]["maximum_extension_m"] > 1.5
    assert turret["outriggers"]["inboard_reserve_m"] > .5
    assert sum(edge["identity"].startswith("turret.recoil.") for edge in graph["edges"]) == 6
    assert sum(edge["identity"].startswith("body_pin.retainer_spring.")
               for edge in graph["edges"]) == 6
    assert all(edge["constraint"] == "actuated-damped-clutch-gimbal-base"
               and edge["angular_damping_nm_s_per_rad"] > 0
               for edge in graph["edges"] if edge["identity"].startswith("turret.mount."))
    assert {item["identity"] for item in turret["turrets"]} == {
        "hood_left", "hood_right", "cab_left", "cab_right", "bed_left", "bed_right"}
    assert turret["body_pins"]["retains_body_without_weapon_payload"] is True
    assert sum(edge["identity"].startswith("outrigger.actuator.") for edge in graph["edges"]) == 4
    assert all(edge["force_authority"] == "canonical-vehicle-total-wrench-input"
               for edge in graph["edges"] if edge["identity"].startswith("outrigger.actuator."))
    assert turret["outriggers"]["hydraulic_accumulator_capacity_j"] > 8000
    assert turret["outriggers"]["hand_pump_displacement_m3_per_click"] > 0
    assert vehicle["body_assembly_interface"]["events"][-1] == "point-impulse-applied"
    winch = vehicle["accessory_presets"][0]
    assert winch["identity"] == "generic-loadout-winch"
    assert winch["cable_stepper"]["status"].endswith("loadout-device-work")
    assert winch["hook_wrench"]["reaction"].startswith("equal-and-opposite")
    assert vehicle["chassis_leveling"]["maximum_corner_offset_m"] > .5
    actuated_links = [edge for edge in graph["edges"] if edge.get("linear_actuator")]
    assert len(actuated_links) == 20
    assert all(edge["linear_actuator"]["maximum_extension_m"] > .4 for edge in actuated_links)
    for corner in WHEEL_NAMES:
        corner_edges = [edge for edge in graph["edges"]
                        if edge["identity"].startswith(f"suspension.{corner}.")]
        assert sum("_arm_" in edge["identity"] for edge in corner_edges) == 4
        assert any(edge["identity"].endswith(".upright") for edge in corner_edges)
        assert any(edge["identity"].endswith(".coilover") for edge in corner_edges)
        assert sum("_pickup_mount_" in edge["identity"] for edge in corner_edges) == 4
        assert any(edge["identity"].endswith(".coilover_tower") for edge in corner_edges)
        assert any(edge["identity"].endswith(".hub_to_wheel") for edge in corner_edges)
        assert any(edge["identity"].endswith(".wheel_to_tire") for edge in corner_edges)
        assert any(edge["identity"].endswith(".wheel_bearing") for edge in corner_edges)
        assert any(edge["identity"].endswith(".rotor_mount") for edge in corner_edges)
        assert any(edge["identity"].endswith(".caliper_mount") for edge in corner_edges)
        assert any(edge["identity"].endswith(".service_brake") for edge in corner_edges)
        assert any(edge["identity"] == f"pneumatics.tire_service_loop.{corner}"
                   for edge in graph["edges"])
        assert any(edge["identity"] == f"brakes.service_hose.{corner}"
                   for edge in graph["edges"])
        assert any(edge["identity"] == f"alignment.service_loop.{corner}"
                   for edge in graph["edges"])
    assert graph["service_port_api"]["device_policy"].endswith("declared-by-loadout")
    assert len(graph["service_port_api"]["networks"]["parking_brake"]["terminals"]) == 2
    assert any(edge["identity"] == "steering.rack_and_pinion" for edge in graph["edges"])
    assert sum(edge["identity"].startswith("steering.wheel.rim_")
               for edge in graph["edges"]) == 8
    assert vehicle["physics"]["contact_patch_shape"]["shape"] == "wide-short-balloon-tire-footprint"
    assert vehicle["physics"]["cage_contact"]["samples"] == [
        "roll-cage-nodes", "cage-member-midpoints",
    ]
    initial = slot["initial_state"]
    assert initial == {
        "mounted_vehicle": vehicle["identity"],
        "placement": "authored-world-pose",
        "presentation": "full-viewport-driving",
        "browser_fullscreen": "user-invoked-only",
        "dismount_enabled": True,
    }
    torque_graph = vehicle["physics"]["torque_graph"]
    assert [node["identity"] for node in torque_graph["nodes"][:7]] == [
        "engine", "clutch", "transmission", "transfer_case", "final_drive",
        "front_differential", "rear_differential",
    ]
    transmission = torque_graph["nodes"][2]
    assert transmission["starting_gear"] == 1
    assert transmission["crawler_gear"] == 1
    assert transmission["forward_ratios"][0] > transmission["forward_ratios"][1]
    transfer_case = torque_graph["nodes"][3]
    assert transfer_case["kind"] == "three-range-transfer-case"
    assert transfer_case["ultra_low_range_ratio"] == 5.24
    assert any(edge.get("loss_model") == "smooth-drag-plus-parametric-efficiency"
               for edge in torque_graph["edges"])
    assert vehicle["physics"]["transmission_policy"]["authority"] == "lockstep-worker-state"
    assert any(edge["channel"] == "powertrain_reaction_torque_xyz"
               for edge in torque_graph["edges"])
    assert "if(!instance||tickInFlight)return" in worker
    assert "main contact bridge scheduler attempted a concurrent GPU dispatch" in projection.script
    assert "await step(body,dt)" in worker
    assert "if(vehicleRuntime.active)return;" in projection.script
    assert "vehicle-contact-monitor" in projection.script
    assert "updateVehicleContactMonitor" in projection.script
    assert 'geometry_mode === "ramp"' not in projection.script
    assert "function ramp(" not in projection.script
    assert "worldBottom:model.contact_surfaces?.world_bottom" in projection.script


def test_chassis_wasm_contract_is_force_driven_and_persists_full_pose_and_wheel_spin():
    compiled = compile_symbolic_vehicle_physics()
    arguments = set(compiled.function.metadata["argument_names"])
    publications = {item.output for item in compiled.publications}
    assert {"total_force_x", "total_force_y", "total_force_z",
            "total_torque_x", "total_torque_y", "total_torque_z"} <= arguments
    assert {"roll_next", "pitch_next", "yaw_next",
            "roll_velocity_next", "pitch_velocity_next", "yaw_velocity_next"} <= publications
    assert {f"wheel_omega_{wheel}_next" for wheel in WHEEL_NAMES} <= publications
    assert {f"traction_scale_{wheel}" for wheel in WHEEL_NAMES} <= publications
    assert {f"brake_scale_{wheel}" for wheel in WHEEL_NAMES} <= publications
    assert {f"damper_scale_{wheel}" for wheel in WHEEL_NAMES} <= publications
    assert {"front_differential_lock", "rear_differential_lock", "center_differential_lock",
            "traction_control_authority", "abs_authority", "differential_lock_stiffness",
            "differential_lock_maximum_torque"} <= arguments
    assert {"active_damping_minimum_scale", "active_damping_maximum_scale",
            "active_damping_body_velocity_gain_s_per_m"} <= arguments
    assert {"engine_torque", "clutch_torque", "transmission_output_torque",
            "driveline_torque", "front_differential_torque",
            "rear_differential_torque", "engine_angular_acceleration",
            "powertrain_reaction_torque_x",
            "wheel_gyroscopic_reaction_torque_x",
            "wheel_gyroscopic_reaction_torque_y"} <= publications
    assert "maximum_speed" not in arguments


def test_engine_off_gross_weight_suspension_quiescence_for_twenty_seconds():
    """Headless 20 s corner soak: 120 Hz outer clock, three fixed substeps."""
    config = load_default_car_configuration().source
    suspension = config["suspension"]
    gross_mass = float(config["mass"]) + 334.0 + 51.0
    corner_fraction = float(config["mass_distribution"]["rear_left"])
    corner_mass = gross_mass * corner_fraction
    stiffness = float(suspension["stiffness"])
    damping = float(suspension["pneumatic_rebound_damping"])
    equilibrium = corner_mass * abs(float(config["world"]["gravity"])) / stiffness
    assert 0 < equilibrium < float(suspension["travel"])

    displacement = 0.02
    velocity = 0.0
    dt = 1.0 / (120.0 * 3.0)
    previous_velocity = velocity
    zero_crossings = 0
    contact_dropouts = 0
    initial_energy = 0.5 * stiffness * displacement * displacement
    maximum_energy = initial_energy
    for _ in range(int(20.0 / dt)):
        acceleration = (-stiffness * displacement - damping * velocity) / corner_mass
        velocity += acceleration * dt
        displacement += velocity * dt
        compression = equilibrium + displacement
        contact_dropouts += compression <= 0
        energy = 0.5 * corner_mass * velocity * velocity + 0.5 * stiffness * displacement * displacement
        maximum_energy = max(maximum_energy, energy)
        if previous_velocity * velocity < 0 and (abs(displacement) > 1e-5 or abs(velocity) > 1e-4):
            zero_crossings += 1
        previous_velocity = velocity

    assert contact_dropouts == 0
    assert maximum_energy <= initial_energy * 1.000001
    assert abs(displacement) < 1e-8
    assert abs(velocity) < 1e-8
    assert zero_crossings <= 4


def test_complete_vehicle_transition_compiles_to_one_packed_webgpu_kernel():
    artifact = compile_symbolic_vehicle_physics_webgpu()
    assert artifact.complete
    assert artifact.shortfalls == ()
    assert "@compute" in artifact.source
    assert "var<storage, read> feeds" in artifact.source
    assert "var<storage, read_write> outputs" in artifact.source


def test_gpu_terrain_contact_is_spatially_and_temporally_top_skin_crossing_aware():
    program = vehicle_webgpu_program_model(load_default_car_configuration())
    source = program["terrain_contact_geometry"]["kernel"]["source"]
    assert "spatial_crossing" in source
    assert "temporal_crossing" in source
    assert "fn terrain_segment_crossing" in source
    assert "for (var subdivision = 1u; subdivision <= 8u" in source
    assert "for (var iteration = 0u; iteration < 8u" in source
    assert "integral_position += evaluation_position * quadrature_weight" in source
    assert "tire_radial_compression = integral_penetration / 15.0f" in source
    assert "radial_probes[lane * 15u" in source
    assert "wall_colliders: array<f32>" in source
    assert "let wall_count = u32(terrain_parameters[1u])" in source
    assert "let field_count = u32(terrain_parameters[0u])" in source
    assert "var inside_field_domain = false" in source
    assert "if (!inside_field_domain && 0.0f <= query_y" in source
    assert "return TerrainSample(0.0f, vec3<f32>(0.0f, 1.0f, 0.0f), 1u)" in source
    assert "tire_half_space_overlap_volume" not in source
    assert "tire_overlap_volume" not in source
    assert "lane >= " in source and "controls[24u]" in source
    assert "0.0f, 0.06f" not in source
    assert "if (any(world_hub < wall_minimum - broadphase_reach)" in source
    assert "let slice_hub = world_hub + axle" in source
    assert "tire_radial_compression" in source
    assert program["terrain_contact_geometry"]["terrain_parameter_abi"][:2] == ["field_count", "wall_count"]
    assembly = program["graph_adapters"]["assembly"]["source"]
    assert "normal_load * normal_load" not in assembly
    assert "normal_load_0 * normal_load_0" in assembly
    assert program["terrain_contact_geometry"]["terrain_upload_policy"] == (
        "initialization-or-authored-height-edit-only"
    )
    assert program["graph_adapters"]["host_awaits_between_nodes"] == 0


def test_abs_and_traction_sensors_are_compiled_persistent_second_order_states():
    equations, symbols = symbolic_vehicle_equations()
    publications = {str(equation.lhs) for equation in equations}
    assert "slip_sensor_frequency_hz" in symbols
    assert "utilization_sensor_frequency_hz" in symbols
    outputs = {str(equation.lhs): equation.rhs for equation in equations}
    assert symbols["traction_control_authority"] in outputs["traction_scale_front_left"].free_symbols
    assert symbols["abs_authority"] in outputs["brake_scale_front_left"].free_symbols
    for wheel in WHEEL_NAMES:
        assert f"slip_sensor_velocity_{wheel}" in symbols
        assert f"measured_friction_utilization_{wheel}" in symbols
        assert f"slip_sensor_velocity_{wheel}_next" in publications
        assert f"friction_utilization_{wheel}_next" in publications
        assert f"friction_utilization_sensor_velocity_{wheel}_next" in publications


def test_symbolic_diff_lock_exchanges_equalizing_torque_across_each_axle():
    equations, symbols = symbolic_vehicle_equations()
    outputs = {str(equation.lhs): equation.rhs for equation in equations}
    config = load_default_car_configuration()
    values = {name: 0.0 for name in symbols}
    values.update(config.parameter_defaults())
    values.update({"dt": 1 / 120, "yaw_cos": 1.0, "transfer_case_ratio": 1.0,
                   "wheel_omega_front_left": 12.0, "wheel_omega_front_right": 0.0})

    def wheel_pair(locked: float) -> tuple[float, float]:
        substitutions = {symbols[name]: value for name, value in values.items() if name in symbols}
        substitutions[symbols["front_differential_lock"]] = locked
        return tuple(float(outputs[f"wheel_omega_{name}_next"].evalf(subs=substitutions))
                     for name in ("front_left", "front_right"))

    open_left, open_right = wheel_pair(0.0)
    limited_left, limited_right = wheel_pair(.32)
    locked_left, locked_right = wheel_pair(1.0)
    assert locked_left < limited_left < open_left
    assert open_right < limited_right < locked_right
    assert locked_left < open_left
    assert locked_right > open_right


def test_symbolic_center_limited_slip_transfers_equal_torque_between_axles():
    equations, symbols = symbolic_vehicle_equations()
    outputs = {str(equation.lhs): equation.rhs for equation in equations}
    config = load_default_car_configuration()
    values = {name: 0.0 for name in symbols}
    values.update(config.parameter_defaults())
    values.update({"dt": 1 / 120, "yaw_cos": 1.0, "transfer_case_ratio": 1.0,
                   "wheel_omega_front_left": 0.0, "wheel_omega_front_right": 0.0,
                   "wheel_omega_rear_left": 12.0, "wheel_omega_rear_right": 12.0})

    def wheel_speeds(coupling: float) -> tuple[float, float]:
        substitutions = {symbols[name]: value for name, value in values.items() if name in symbols}
        substitutions[symbols["center_differential_lock"]] = coupling
        front = sum(float(outputs[f"wheel_omega_{name}_next"].evalf(subs=substitutions))
                    for name in ("front_left", "front_right")) / 2
        rear = sum(float(outputs[f"wheel_omega_{name}_next"].evalf(subs=substitutions))
                   for name in ("rear_left", "rear_right")) / 2
        return front, rear

    open_front, open_rear = wheel_speeds(0.0)
    limited_front, limited_rear = wheel_speeds(.32)
    locked_front, locked_rear = wheel_speeds(1.0)
    assert open_front < limited_front < locked_front
    assert open_rear > limited_rear > locked_rear


def test_symbolic_drivetrain_conserves_wheel_and_chassis_reaction_torque():
    equations, _ = symbolic_vehicle_equations()
    outputs = {str(equation.lhs): equation.rhs for equation in equations}
    residual = (
        outputs["drivetrain_chassis_reaction_torque"]
        + outputs["front_differential_torque"]
        + outputs["rear_differential_torque"]
        - outputs["tire_contact_reaction_torque"]
        - outputs["service_brake_reaction_torque"]
        - outputs["rolling_resistance_reaction_torque"]
    )
    assert sympy.simplify(residual) == 0


def test_slipping_clutch_does_not_reflect_engine_inertia_into_wheel_acceleration():
    equations, symbols = symbolic_vehicle_equations()
    outputs = {str(equation.lhs): equation.rhs for equation in equations}
    wheel_step = outputs["wheel_omega_front_left_next"]
    assert sympy.simplify(sympy.diff(wheel_step, symbols["engine_rotating_inertia"])) == 0
    assert sympy.simplify(sympy.diff(wheel_step, symbols["maximum_wheel_speed"])) == 0


def test_external_hub_wrench_drives_inertia_and_neutral_really_opens_driveline():
    equations, symbols = symbolic_vehicle_equations()
    outputs = {str(equation.lhs): equation.rhs for equation in equations}
    wheel_step = outputs["wheel_omega_front_left_next"]
    assert sympy.simplify(sympy.diff(
        wheel_step, symbols["external_hub_torque_front_left"])) != 0
    assert sympy.simplify(outputs["clutch_torque"].subs(symbols["drive_direction"], 0)) == 0
    assert sympy.simplify(outputs["engine_torque"].subs({
        symbols["engine_angular_speed"]: 0,
        symbols["accessory_load_torque"]: 0,
    })) == 0


def test_locking_hubs_isolate_differential_wrench_but_not_hub_side_rig_wrench():
    equations, symbols = symbolic_vehicle_equations()
    outputs = {str(equation.lhs): equation.rhs for equation in equations}
    wheel_step = outputs["wheel_omega_front_left_next"]
    open_hub = {symbols["hub_locker_engagement_front_left"]: 0}
    shaft_step = outputs["differential_wrench_shaft_omega_front_next"]
    assert shaft_step.has(symbols["external_differential_wrench_torque_front"])
    open_wheel_step = wheel_step.subs(open_hub)
    assert not open_wheel_step.has(symbols["differential_wrench_shaft_omega_front"])
    assert open_wheel_step.has(symbols["external_hub_torque_front_left"])
    assert wheel_step.subs(symbols["hub_locker_engagement_front_left"], 1).has(
        symbols["differential_wrench_shaft_omega_front"])


def test_alternator_cvt_and_ev_regen_are_live_drivetrain_energy_paths():
    equations, symbols = symbolic_vehicle_equations()
    outputs = {str(equation.lhs): equation.rhs for equation in equations}
    assert outputs["alternator_reaction_torque_nm"].has(
        symbols["alternator_electrical_demand_w"])
    assert outputs["engine_angular_acceleration"].has(
        symbols["alternator_rotor_inertia_each"])
    assert outputs["engine_angular_acceleration"].has(
        symbols["alternator_cvt_ratio_state"])
    assert outputs["engine_angular_acceleration"].has(
        symbols["external_engine_flywheel_inertia"])
    assert outputs["traction_battery_charge_fraction_next"].has(
        symbols["power_unit_electric_mode"])
    assert outputs["traction_battery_charge_fraction_next"].has(
        symbols["traction_battery_target_charge_fraction"])
    assert outputs["clutch_torque"].has(symbols["clutch_wear"], symbols["clutch_glaze"])
    assert outputs["alternator_generated_power_w"].has(
        symbols["alternator_cvt_wear"], symbols["alternator_cvt_glaze"])
    assert outputs["accessory_motor_engine_reaction_torque_nm"].has(
        symbols["accessory_motor_command"], symbols["alternator_cvt_ratio_state"])
    assert outputs["accessory_motor_bus_power_w"].has(
        symbols["accessory_battery_cube_internal_resistance_ohm"])
    assert outputs["accessory_battery_cube_charge_fraction_next"].has(
        symbols["accessory_motor_command"],
        symbols["accessory_battery_cube_charge_fraction"])
    assert outputs["engine_angular_acceleration"].has(
        symbols["accessory_motor_command"])
    assert outputs["compressor_engine_reaction_torque_nm"].has(
        symbols["high_pressure_compressor_command"],
        symbols["air_mix_reserve_gas_mass_kg"])
    assert outputs["air_mix_reserve_pressure_pa"].has(
        symbols["air_mix_reserve_volume_m3"],
        symbols["air_mix_reserve_temperature_k"])
    assert outputs["engine_angular_acceleration"].has(
        symbols["high_pressure_compressor_command"])
    for wheel in WHEEL_NAMES:
        assert outputs[f"wheel_omega_{wheel}_next"].has(
            symbols[f"hub_locker_wear_{wheel}"], symbols[f"hub_locker_glaze_{wheel}"])
    for axle in ("front", "rear", "center"):
        assert f"differential_locker_wear_{axle}_next" in outputs
        assert f"differential_locker_glaze_{axle}_next" in outputs
    assert outputs["direct_drive_bypass_engagement_next"].has(
        symbols["direct_drive_bypass_command"])
    assert outputs["transmission_output_torque"].has(
        symbols["direct_drive_bypass_engagement"])
    assert outputs["direct_drive_bypass_tooth_health_next"].has(
        symbols["direct_drive_bypass_tooth_health"])
    assert sympy.simplify(outputs["optional_fluid_coupling_torque_nm"].subs(
        symbols["optional_fluid_coupling_engagement"], 0)) == 0


def test_differential_brake_rotor_inertia_lives_once_on_its_shaft_coordinate():
    equations, symbols = symbolic_vehicle_equations()
    outputs = {str(equation.lhs): equation.rhs for equation in equations}
    rotor_inertia = symbols["differential_brake_rotor_inertia"]
    assert outputs["differential_wrench_shaft_omega_front_next"].has(rotor_inertia)
    assert outputs["differential_wrench_shaft_omega_rear_next"].has(rotor_inertia)
    assert not outputs["wheel_omega_front_left_next"].has(rotor_inertia)
    assert not outputs["wheel_gyroscopic_reaction_torque_x"].has(rotor_inertia)


def test_emitted_wasm_turns_pedal_into_wheel_spin_not_commanded_chassis_speed():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute WebAssembly")
    artifact = compile_symbolic_vehicle_physics_wasm()
    config = load_default_car_configuration()
    values = {name: 0.0 for name in artifact.input_names}
    values.update(config.parameter_defaults())
    values.update({"dt": 1 / 120, "throttle": 1.0, "yaw_cos": 1.0,
                   "transfer_case_ratio": 1.0,
                   "roll_velocity": .5, "pitch_velocity": 1.0,
                   "drive_fraction_front_left": .21, "drive_fraction_front_right": .21,
                   "drive_fraction_rear_left": .29, "drive_fraction_rear_right": .29,
                   "friction_utilization_front_left": 1.3,
                   "slip_longitudinal_front_left": 1.5,
                   "previous_slip_longitudinal_front_left": .1})
    script = r"""
    const payload=JSON.parse(require("fs").readFileSync(0,"utf8")),bytes=Buffer.from(payload.bytes,"base64");
    (async()=>{const {instance}=await WebAssembly.instantiate(bytes,{}),memory=new Float64Array(instance.exports.memory.buffer);
    memory.set(payload.values,Number(process.argv[1])/8);instance.exports.abstract_ui_vehicle_step(0);
    const start=Number(process.argv[2])/8,count=Number(process.argv[3]);
console.log(JSON.stringify(Array.from(memory.slice(start,start+count))));})().catch(error=>{console.error(error);process.exit(1)});
"""
    completed = subprocess.run([
        node, "-e", script, str(artifact.input_offsets[0]), str(artifact.output_offsets[0]),
        str(len(artifact.output_names)),
    ], input=json.dumps({"bytes": base64.b64encode(artifact.binary).decode("ascii"),
                         "values": [values[name] for name in artifact.input_names]}),
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    result = dict(zip(artifact.output_names, json.loads(completed.stdout)))
    assert result["velocity_x_next"] == pytest.approx(0.0)
    assert result["wheel_omega_front_left_next"] > 0
    assert result["wheel_omega_rear_left_next"] > result["wheel_omega_front_left_next"]
    assert 0.08 <= result["traction_scale_front_left"] < 1.0
    assert .08 <= result["brake_scale_front_left"] < 1.0
    assert result["brake_scale_front_left"] <= result["traction_scale_front_left"]
    assert .9 < result["traction_scale_front_right"] <= 1.0
    assert 1.0 < result["damper_scale_front_left"] <= 1.18
    assert result["damper_scale_front_left"] > result["damper_scale_front_right"]
    assert result["engine_torque"] > 0
    assert result["clutch_torque"] > result["engine_torque"]
    assert result["transmission_output_torque"] > result["clutch_torque"]
    assert result["driveline_torque"] > result["transmission_output_torque"]
    assert result["front_differential_torque"] == pytest.approx(result["driveline_torque"] * .42)
    assert result["rear_differential_torque"] == pytest.approx(result["driveline_torque"] * .58)
    assert result["engine_angular_acceleration"] < 0
    assert result["powertrain_reaction_torque_x"] == pytest.approx(
        -result["engine_acceleration_torque"])
    assert result["powertrain_reaction_torque_y"] == pytest.approx(0)
    assert result["powertrain_reaction_torque_z"] == pytest.approx(0)
