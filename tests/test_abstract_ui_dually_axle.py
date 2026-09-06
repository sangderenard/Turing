"""Contracts for the discoverable commercial dually rear-end artifact."""

from src.compiler.abstract_ui_dually_axle import (
    DUALLY_VALIDATOR_STAGES,
    roadside_dually_axle_assembly,
    roadside_dually_axle_geometry_boxes,
)
from src.compiler.vehicle_native_graph_program import (
    vehicle_graph_constants_from_model,
)
from src.compiler.vehicle_native_assembly import (
    infer_structural_grasp_frame, negotiate_wheel_fixture,
)


def _assembly():
    return roadside_dually_axle_assembly(
        "world:probe", center_x=12.0, center_z=30.0,
    )


def test_dually_is_one_complete_cased_axle_with_four_tires_and_no_suspension():
    model = _assembly().model
    assert len(model["wheels"]) == 4
    assert {wheel["dual_layer"] for wheel in model["wheels"]} == {"inner", "outer"}
    assert {wheel["side"] for wheel in model["wheels"]} == {"left", "right"}
    assert all(len(group["wheels"]) == 2 for group in model["wheel_groups"].values())
    assert model["structure"]["suspension"] is None
    assert len(infer_structural_grasp_frame(model).support_corners) == 4
    assert model["axle_casing"]["kind"] == (
        "load-bearing-full-floating-banjo-axle-casing"
    )


def test_dually_casing_is_a_four_support_structural_graph_not_four_corners():
    model = _assembly().model
    fixture = negotiate_wheel_fixture(model)
    constants = vehicle_graph_constants_from_model(
        model, len(fixture.structural_support_identities))
    assert constants.node_structural_support_binding.shape == (
        len(model["mechanical_graph"]["nodes"]), 4)
    assert constants.structural_support_edge_mask.shape[0] == 4


def test_frame_grabbers_infer_u_bolt_regions_and_allow_one_arm_to_search():
    grasp = infer_structural_grasp_frame(_assembly().model)
    assert grasp.source == "inferred-strong-accessible-region-pair"
    assert len(grasp.arms) == 4
    assert grasp.regrasp_policy["searching_arms"] == 1
    assert grasp.regrasp_policy["minimum_engaged_arms"] == 3
    assert grasp.objective["primary"] == (
        "minimum-predicted-structural-strain-energy")
    assert all("u-bolt" in corner["mount_analogue"]
               for corner in grasp.support_corners)


def test_bearing_nodes_join_structural_and_drivetrain_edges():
    graph = _assembly().model["mechanical_graph"]
    nodes = {node["identity"]: node for node in graph["nodes"]}
    for side in ("left", "right"):
        suffix = f"/hubs/{side}-bearing-interface"
        identity = next(name for name in nodes if name.endswith(suffix))
        bearing = nodes[identity]
        assert set(bearing["interface_roles"]) == {
            "structural-reaction", "drivetrain-rotation"}
        incident = [edge for edge in graph["edges"]
                    if identity in edge["nodes"]]
        assert {edge.get("edge_class") for edge in incident} >= {
            "load-bearing-structure", "drivetrain"}
        races = {edge["edge_class"]: edge["bearing_race"]
                 for edge in incident if "bearing_race" in edge}
        assert races == {
            "load-bearing-structure": "stationary",
            "drivetrain": "rotating",
        }
        assert bearing["constraint"]["free_coordinate"] == (
            "rotation-about-axis")
        transformer = bearing["force_transfer_transformer"]
        assert transformer["schema"] == "bearing-wrench-transformer-v1"
        parameters = transformer["parameters"]
        assert 0.0 < parameters["bore_m"] < parameters["outer_diameter_m"]
        assert parameters["radial_damping_n_s_m"] > 0.0
        assert parameters["rolling_friction_coefficient"] > 0.0
        assert parameters["radial_static_limit_n"] > 0.0
        assert bearing["constraint"]["degrees_of_freedom"][
            "rotation-about-axis"]["kind"] == "free-periodic"
        assert "seizure" in transformer["failure_law"]


def test_axle_casing_is_a_serviceable_structural_and_lubricant_boundary():
    model = _assembly().model
    casing = model["axle_casing"]
    lubricant = casing["lubricant_boundary"]
    assert lubricant["removable_differential_cover"] is True
    assert lubricant["service_fill_l"] > 0.0
    assert len(lubricant["shaft_seals"]) == 2
    assert len(lubricant["hub_seals"]) == 2
    assert casing["mounting"]["reuse_contract"].startswith("casing-is-the-structural")
    tube = casing["mounting"]["structural_tube"]
    assert len(tube["weld_qualified_zones"]) == 4
    assert "hub-seal-land" in tube["keep_clear_zones"]
    future = model["structure"]["future_platform_boundary"]
    assert future["owner"] == "separate-n-axle-suspension-platform"
    assert "walking-beam-load-rocker" in future["supported_families"]


def test_open_pinion_is_a_generic_reaction_wrench_port_for_validator_machinery():
    model = _assembly().model
    port = model["differential"]["external_input"]
    assert port["schema"] == "rotating-six-axis-drivetrain-wrench-port-v1"
    assert port["kind"] == "generic-rotational-torque-input"
    assert "angular-velocity-is-observed-state-never-a-command" in port["law"]
    assert port["fixture_policy"].startswith("validator-dynamometer-is-temporary")
    analogues = model["validator_program"]["fixture_analogues"]
    assert analogues["engine-or-transmission-drive"] == port["identity"]
    assert analogues["differential-brake-wrench-port"] == port["identity"]


def test_axle_can_swap_between_terminal_cap_and_middle_axle_through_drive():
    interface = _assembly().model["axle_chain_interface"]
    assert interface["active_part"] == "sealed-terminal-cap"
    assert interface["parts"]["sealed-terminal-cap"]["axle_role"] == "terminal"
    cartridge = interface["parts"]["through-drive-cartridge"]
    assert cartridge["axle_role"] == "intermediate"
    assert cartridge["rotational_port"]["role"] == "output"
    assert interface["chain_contract"]["inter_axle_shaft_is_separate_part"] is True
    assert any("terminal-cap-or-through-drive-cartridge" in step
               for step in interface["conversion_sequence"])


def test_hydraulic_brakes_and_wheel_pneumatics_are_complete_on_both_dual_groups():
    model = _assembly().model
    stations = model["brakes"]["stations"]
    assert len(stations) == 2
    assert {station["side"] for station in stations} == {"left", "right"}
    assert all(station["kind"].startswith("hydraulic-commercial") for station in stations)
    assert all(station["maximum_brake_torque_nm"] > 0.0 for station in stations)
    assert model["pneumatics"]["outer_service_valves"] == 4
    assert model["pneumatics"]["bearing_fed_paths"] == 0
    assert model["pneumatics"]["hub_passages"] == 0
    assert all(wheel["ports"]["rim_seat_ports"] == []
               for wheel in model["wheels"])
    assert all(wheel["ports"]["bearing_rotary_union"] is None
               for wheel in model["wheels"])
    assert all(wheel["material_profile"] == "cheap-commercial-retread"
               for wheel in model["wheels"])


def test_wheel_hub_bearing_rim_bead_casing_and_boundary_are_distinct_objects():
    model = _assembly().model
    graph_nodes = {node["identity"]: node
                   for node in model["mechanical_graph"]["nodes"]}
    for wheel in model["wheels"]:
        parts = wheel["abstract_ui_assembly"]["parts"]
        identities = {
            parts[name]["identity"] for name in (
                "wheel_center", "rim", "bead_inboard", "bead_outboard",
                "sidewall", "tread", "pneumatic_boundary",
            )
        }
        assert len(identities) == 7
        assert identities <= graph_nodes.keys()
        assert parts["wheel_center"]["fastens_to"] == "axle-hub-not-bearing"
        assert parts["rim"]["joined_to"] == parts["wheel_center"]["identity"]
        assert parts["bead_inboard"]["seats_on"].endswith(":inboard-seat")
        assert parts["bead_inboard"]["beadlock"] is False
        assert parts["bead_inboard"]["fasteners"] == []
        assert parts["sidewall"]["kind"] == "oriented-composite-tire-sidewall"
        boundary = parts["pneumatic_boundary"]
        assert boundary["kind"] == "tubeless-casing-rim-bead-seal-boundary"
        assert parts["rim"]["identity"] in boundary["sealed_by"]
        assert any(layer.get("orientation_degrees") == 22.0
                   for layer in boundary["thermal_layers"])
        assert any(layer["material"] == "rubberized-steel-cord"
                   for layer in boundary["thermal_layers"])
    for side in ("left", "right"):
        hub = graph_nodes[next(identity for identity in graph_nodes
                               if identity.endswith(f"/hubs/{side}-dual"))]
        bearing = graph_nodes[next(identity for identity in graph_nodes
                                   if identity.endswith(
                                       f"/hubs/{side}-bearing-interface"))]
        assert hub["schema"] == "abstract-ui-rotating-hub-node-v1"
        assert bearing["schema"] == "mechanical-bearing-interface-node-v1"
        assert hub["identity"] != bearing["identity"]


def test_validator_runs_the_whole_axle_and_stops_after_differential_brake_proof():
    program = _assembly().model["validator_program"]
    assert program["tire_batch"] == 4
    assert tuple(program["stages"]) == DUALLY_VALIDATOR_STAGES
    assert program["stop_after"] == "differential-and-hydraulic-brake-torque-proof"
    assert "suspension" in program["forbidden_stages"]
    assert program["torque_proof_sequence"][0].endswith("open-pinion-flange")
    assert program["torque_proof_sequence"][-1].endswith("detach-dynamometer")


def test_world_realization_keeps_the_axle_as_one_discoverable_parent_object():
    assembly = _assembly()
    assert len(assembly.world_objects) == 1
    item = assembly.world_objects[0]
    assert item.parent == "world:probe"
    assert item.physics["suspension"] is False
    assert "discover" in item.capabilities
    boxes = roadside_dually_axle_geometry_boxes(assembly)
    assert len(boxes) == 6
    assert all(box["parent_identity"] == item.identity for box in boxes)
