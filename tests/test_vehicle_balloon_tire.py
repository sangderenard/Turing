from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import sympy

from src.compiler.abstract_ui_vehicles import load_default_car_configuration
from src.compiler.vehicle_balloon_tire import (
    balloon_tire_graph_abi,
    build_balloon_tire_topology,
    compile_balloon_bead_constraint_c,
    compile_balloon_contact_geometry_c,
    compile_balloon_cylinder_contact_geometry_c,
    compile_balloon_contact_impulse_c,
    compile_balloon_gas_c,
    compile_balloon_membrane_face_c,
    symbolic_balloon_membrane_face_equations,
)
from src.compiler.vehicle_balloon_tire_native import compile_native_balloon_tire_assembly
from src.compiler.vehicle_balloon_tire_program import balloon_tire_python_program


def _topology():
    return build_balloon_tire_topology(
        major_radius_m=0.46,
        section_radius_m=0.09,
        circumferential_segments=12,
        section_segments=8,
    )


def test_balloon_skin_is_a_closed_oriented_torus_not_a_runtime_analytic_shape():
    topology = _topology()
    counts: dict[tuple[int, int], int] = {}
    for face in topology.faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((a, b)))
            counts[edge] = counts.get(edge, 0) + 1
    assert set(counts.values()) == {2}
    assert len(topology.rest_positions) - len(topology.edges) + len(topology.faces) == 0
    assert topology.reference_volume_m3 > 0.0
    analytic_volume = 2 * math.pi**2 * 0.46 * 0.09**2
    # The compile-static polygonal skin converges from below to the analytic
    # rest generator; runtime volume is always evaluated from this skin.
    assert topology.reference_volume_m3 == pytest.approx(analytic_volume, rel=0.16)


def test_balloon_graph_abi_carries_every_vertex_mass_and_state():
    config = load_default_car_configuration().source
    abi = balloon_tire_graph_abi(config)
    assert abi["identity"] == "compiled-balloon-skin-v1"
    assert abi["collision_authority"] == "deformed-skin-vertex-triangle-ccd"
    assert len(abi["state"]) == 6 * abi["vertex_count"]
    assert abi["parameters"]["vertex_mass_kg"] * abi["vertex_count"] == pytest.approx(
        config["drivetrain"]["tire_mass_kg"]
    )
    assert abi["bead_vertex_count"] == 2 * config["tire_skin"]["circumferential_segments"]
    assert set(abi["topology"].face_zones) == {
        "tread", "sidewall", "bead", "rim-closure"}


def test_authored_membrane_uses_face_zoned_material_properties():
    program = balloon_tire_python_program()
    material = program.constants["face_material"]
    topology = balloon_tire_graph_abi(load_default_car_configuration().source)["topology"]
    assert material.shape == (program.face_count, 11)
    representatives = {
        zone: material[topology.face_zones.index(zone)]
        for zone in ("tread", "sidewall", "bead")
    }
    assert representatives["tread"][1] > representatives["sidewall"][1]
    assert representatives["bead"][1] > representatives["tread"][1]
    assert representatives["sidewall"][0] < representatives["tread"][0]


def test_cheap_retread_refines_mesh_and_activates_directional_laminate():
    program = balloon_tire_python_program(
        ("dually",), pneumatic_mode="tubeless",
        material_profile="cheap-commercial-retread")
    assert program.pneumatic_mode == "tubeless"
    assert program.material_profile == "cheap-commercial-retread"
    assert program.vertex_count == 32 * 25
    assert program.face_count == 2 * 32 * 24 + 2 * 32
    material = program.constants["face_material"]
    assert material.shape == (program.face_count, 11)
    assert np.max(np.abs(material[:, 5:])) > 0.0
    assert np.ptp(program.constants["face_material_basis_rad"]) > 0.0
    assert program.constants["material_coordinates_uv"].shape == (
        program.vertex_count, 2)
    assert program.constants["natural_position_uv"].shape == (
        program.vertex_count, 3)
    assert program.constants["face_material_uv"].shape == (
        program.face_count, 2)
    assert program.constants["face_natural_jacobian_uv"].shape == (
        program.face_count, 6)
    assert program.constants["face_natural_metric_uv"].shape == (
        program.face_count, 3)
    assert program.constants["face_directional_coefficients_uv"].shape == (
        program.face_count, 6)
    assert np.shares_memory(
        program.constants["face_directional_coefficients_uv"],
        program.constants["face_material"],
    )
    assert int(program.constants["rim_closure_face_mask"].sum()) == 2 * 32


def test_tubeless_rest_surface_has_cylindrical_tread_and_molded_sidewalls():
    topology = build_balloon_tire_topology(
        major_radius_m=0.3925,
        section_radius_m=0.1425,
        section_width_m=0.285,
        rim_radius_m=0.28575,
        circumferential_segments=32,
        section_segments=24,
        pneumatic_mode="tubeless",
        mold_profile="cheap-commercial-retread",
    )
    row_count = topology.section_segments + 1
    radial = [math.hypot(*topology.rest_positions[row][0:2])
              for row in range(row_count)]
    axial = [topology.rest_positions[row][2] for row in range(row_count)]
    # All authored tread rows are one cylindrical rolling circumference.
    tread_rows = range(7, 18)
    assert np.ptp([radial[row] for row in tread_rows]) < 1.0e-12
    assert radial[12] == pytest.approx(0.535)
    # Sidewalls are real center-surface panels between shoulder and bead; they
    # reach the tire's section width and terminate at the rim seat radius.
    assert max(abs(value) for value in axial) > abs(axial[0])
    assert radial[0] == pytest.approx(0.28575)
    assert radial[-1] == pytest.approx(0.28575)
    assert topology.rest_surface_kind.startswith("open-uv-casing")


def test_shell_state_is_one_invariant_center_surface_with_two_oriented_sides():
    abi = balloon_tire_graph_abi(load_default_car_configuration().source)
    authority = abi["shell_surface_authority"]
    assert authority["state_surface"] == "single-invariant-center-surface"
    assert authority["position_dofs"].startswith("one-position-per-vertex")
    assert authority["exterior_side"].startswith("positive-outward")
    assert authority["interior_side"].startswith("negative-outward")


def test_layer_contract_keeps_oriented_composite_steel_and_tube_heat_distinct():
    config = load_default_car_configuration().source
    tubeless = balloon_tire_graph_abi(config)
    bead_layers = tubeless["layer_stacks"]["bead"]
    assert {layer["material"] for layer in bead_layers} >= {
        "rubber", "composite-cord", "steel", "low-permeability-rubber"}
    cord_angles = [layer["orientation_rad"] for layer in bead_layers
                   if layer["material"] == "composite-cord"]
    assert cord_angles[0] == pytest.approx(-cord_angles[1])
    assert all(layer["specific_heat_j_per_kg_k"] > 0.0 for layer in bead_layers)
    assert all(layer["thermal_conductivity_w_per_m_k"] > 0.0 for layer in bead_layers)
    assert tubeless["rim_boundary"]["closes_pressure_volume"] is True

    tube_config = copy.deepcopy(config)
    tube_config["tire_skin"]["pneumatic_mode"] = "tube"
    tubed = balloon_tire_graph_abi(tube_config)
    assert tubed["rim_boundary"]["closes_pressure_volume"] is False
    assert tubed["rim_boundary"]["pressure_membrane"] == "tube"
    assert any(layer["material"] == "tube-rubber"
               for layer in tubed["layer_stacks"]["sidewall"])


def test_membrane_face_is_stress_free_at_rest_and_internal_forces_conserve():
    topology = _topology()
    face = topology.faces[0]
    rest_data = topology.face_rest_data[0]
    equations, symbols = symbolic_balloon_membrane_face_equations()
    output_names = [str(equation.lhs) for equation in equations]
    function = sympy.lambdify(
        tuple(symbols.values()), tuple(equation.rhs for equation in equations), "numpy"
    )
    values = {name: 0.0 for name in symbols}
    for local, vertex in enumerate(face):
        for axis, component in zip("xyz", topology.rest_positions[vertex]):
            values[f"x{local}_{axis}"] = component
            values[f"r{local}_{axis}"] = component
    values.update({
        "rest_inverse_00": rest_data[0],
        "rest_inverse_01": rest_data[1],
        "rest_inverse_10": rest_data[2],
        "rest_inverse_11": rest_data[3],
        "rest_area_m2": rest_data[4],
        "natural_metric_00": rest_data[5],
        "natural_metric_01": rest_data[6],
        "natural_metric_11": rest_data[7],
        "skin_thickness_m": 0.012,
        "lame_lambda_pa": 6.2e6,
        "lame_mu_pa": 4.1e6,
        "membrane_damping_lambda_pa_s": 5400.0,
        "membrane_damping_mu_pa_s": 3600.0,
        "gas_pressure_pa": 0.0,
        "reference_pressure_pa": 0.0,
    })
    result = dict(zip(output_names, function(*(values[name] for name in symbols))))
    assert result["strain_energy_j"] == pytest.approx(0.0, abs=1e-18)
    forces = np.array([
        [result[f"force_{vertex}_{axis}_n"] for axis in "xyz"]
        for vertex in range(3)
    ])
    assert forces == pytest.approx(np.zeros((3, 3)), abs=1e-8)

    # A finite deformation and velocity must retain zero internal resultant,
    # zero internal moment, and non-positive viscous power.
    values["x1_x"] += 0.017
    values["x2_y"] -= 0.011
    values["v0_x"], values["v1_y"], values["v2_z"] = 0.3, -0.2, 0.1
    result = dict(zip(output_names, function(*(values[name] for name in symbols))))
    forces = np.array([
        [result[f"force_{vertex}_{axis}_n"] for axis in "xyz"]
        for vertex in range(3)
    ])
    positions = np.array([
        [values[f"x{vertex}_{axis}"] for axis in "xyz"] for vertex in range(3)
    ])
    assert forces.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-7)
    assert np.cross(positions, forces).sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-7)
    assert result["dissipation_power_w"] <= 1e-9


def test_inflated_reference_state_has_no_construction_pressure_startup_impulse():
    topology = _topology()
    face = topology.faces[0]
    rest_data = topology.face_rest_data[0]
    equations, symbols = symbolic_balloon_membrane_face_equations()
    output_names = [str(equation.lhs) for equation in equations]
    function = sympy.lambdify(
        tuple(symbols.values()), tuple(equation.rhs for equation in equations), "numpy"
    )
    values = {name: 0.0 for name in symbols}
    for local, vertex in enumerate(face):
        for axis, component in zip("xyz", topology.rest_positions[vertex]):
            values[f"x{local}_{axis}"] = component
            values[f"r{local}_{axis}"] = component
    values.update({
        "rest_inverse_00": rest_data[0], "rest_inverse_01": rest_data[1],
        "rest_inverse_10": rest_data[2], "rest_inverse_11": rest_data[3],
        "rest_area_m2": rest_data[4], "skin_thickness_m": 0.012,
        "natural_metric_00": rest_data[5],
        "natural_metric_01": rest_data[6],
        "natural_metric_11": rest_data[7],
        "lame_lambda_pa": 6.2e6, "lame_mu_pa": 4.1e6,
        "membrane_damping_lambda_pa_s": 5400.0,
        "membrane_damping_mu_pa_s": 3600.0,
        "gas_pressure_pa": 135000.0, "reference_pressure_pa": 135000.0,
    })
    result = dict(zip(output_names, function(*(values[name] for name in symbols))))
    total = np.asarray([
        [result[f"force_{vertex}_{axis}_n"] for axis in "xyz"]
        for vertex in range(3)
    ])
    pressure = np.asarray([
        [result[f"pressure_force_{vertex}_{axis}_n"] for axis in "xyz"]
        for vertex in range(3)
    ])
    construction = np.asarray([
        [result[f"construction_force_{vertex}_{axis}_n"] for axis in "xyz"]
        for vertex in range(3)
    ])
    np.testing.assert_allclose(pressure, -construction, atol=1e-10)
    np.testing.assert_allclose(total, 0.0, atol=1e-8)


def test_gas_and_bead_equations_lower_to_native_c_without_a_second_law(tmp_path):
    gas = compile_balloon_gas_c().compile(tmp_path / "gas")
    gas_values = dict(zip(gas.output_names, gas.run({
        "reference_pressure_pa": 135000.0,
        "reference_volume_m3": 0.08,
        "current_volume_m3": 0.04,
        "gas_polytropic_exponent": 1.4,
        "minimum_volume_fraction": 0.2,
        "reference_temperature_k": 300.0,
    })))
    assert gas_values["gas_pressure_pa"] == pytest.approx(135000.0 * 2.0**1.4)
    assert gas_values["gas_temperature_k"] == pytest.approx(300.0 * 2.0**0.4)

    bead = compile_balloon_bead_constraint_c().compile(tmp_path / "bead")
    inputs = {name: 0.0 for name in bead.input_names}
    inputs.update({"vertex_x": 0.01, "bead_stiffness_n_per_m": 1000.0,
                   "bead_damping_n_s_per_m": 10.0})
    bead_values = dict(zip(bead.output_names, bead.run(inputs)))
    for axis in "xyz":
        assert bead_values[f"skin_force_{axis}_n"] == pytest.approx(
            -bead_values[f"rim_force_{axis}_n"]
        )

    # This is an emission gate, not a native build: the expensive derivative
    # kernel must still be a complete compiler product.
    assert compile_balloon_membrane_face_c().complete


def test_deformed_skin_contact_uses_actual_crossing_and_equal_opposite_impulse(tmp_path):
    native_source = compile_native_balloon_tire_assembly().source
    # Once a vertex crosses, finite-step penetration must never make it
    # ineligible for unilateral support.  Contact remains impulse-only: there
    # is no penetration spring or overlap rejection term.
    assert "TIRE_CONTACT_ACTIVE_THICKNESS_FRACTION" not in native_source
    assert "one finite-step overshoot" in native_source
    assert "if(xout[2]<=0.0&&" in native_source
    assert "previous[0]+dt*pv[0]" in native_source
    geometry = compile_balloon_contact_geometry_c().compile(tmp_path / "geometry")
    geometry_inputs = {name: 0.0 for name in geometry.input_names}
    geometry_inputs.update({
        "previous_x": 0.25, "previous_y": 0.10, "previous_z": 0.25,
        "current_x": 0.25, "current_y": -0.10, "current_z": 0.25,
        "triangle_a_x": 0.0, "triangle_a_y": 0.0, "triangle_a_z": 0.0,
        "triangle_b_x": 0.0, "triangle_b_y": 0.0, "triangle_b_z": 1.0,
        "triangle_c_x": 1.0, "triangle_c_y": 0.0, "triangle_c_z": 0.0,
        "skin_offset_m": 0.0,
    })
    hit = dict(zip(geometry.output_names, geometry.run(geometry_inputs)))
    assert hit["time_of_impact_fraction"] == pytest.approx(0.5)
    assert hit["previous_signed_distance_m"] > 0.0
    assert hit["current_signed_distance_m"] < 0.0
    assert hit["contact_y_m"] == pytest.approx(0.0)
    assert min(hit["barycentric_u"], hit["barycentric_v"], hit["barycentric_w"]) >= 0.0

    response = compile_balloon_contact_impulse_c().compile(tmp_path / "response")
    response_inputs = {name: 0.0 for name in response.input_names}
    response_inputs.update({
        "contact_active": 1.0,
        "normal_x": hit["normal_x"], "normal_y": hit["normal_y"],
        "normal_z": hit["normal_z"],
        "velocity_x": 0.4, "velocity_y": -2.0, "velocity_z": 0.0,
        "inverse_effective_mass_per_kg": 2.0,
        "restitution": 0.0,
        "friction_coefficient": 0.8,
    })
    impulse = dict(zip(response.output_names, response.run(response_inputs)))
    assert impulse["normal_impulse_ns"] > 0.0
    for axis in "xyz":
        assert impulse[f"skin_impulse_{axis}_ns"] == pytest.approx(
            -impulse[f"terrain_impulse_{axis}_ns"]
        )


def test_roller_contact_uses_exact_cylinder_crossing_not_a_hub_facing_plane(tmp_path):
    geometry = compile_balloon_cylinder_contact_geometry_c().compile(tmp_path / "cylinder")
    values = {name: 0.0 for name in geometry.input_names}
    values.update({
        "previous_x": 0.0, "previous_y": .25, "previous_z": .1,
        "current_x": 0.0, "current_y": .05, "current_z": .1,
        "cylinder_center_x": 0.0, "cylinder_center_y": 0.0,
        "cylinder_center_z": 0.0, "cylinder_radius_m": .13,
        "skin_offset_m": .01,
    })
    hit = dict(zip(geometry.output_names, geometry.run(values)))
    assert hit["previous_signed_distance_m"] == pytest.approx(.11)
    assert hit["current_signed_distance_m"] == pytest.approx(-.09)
    assert hit["time_of_impact_fraction"] == pytest.approx(.55)
    assert hit["normal_y"] == pytest.approx(1.0)
    assert hit["contact_y_m"] == pytest.approx(.13)
