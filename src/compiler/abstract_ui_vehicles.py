"""JSON-configured vehicle slots and compiled parallel suspension physics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import sympy

from .abstract_ui_world import WorldWasmPlugin
from .ssa_wasm_backend import SSAWasmArtifact, emit_ssa_function_to_wasm
from .ssa_c_backend import CFunctionArtifact, emit_ssa_function_to_c
from .ssa_webgpu_backend import WGSLModule, emit_module as emit_webgpu_module
from .symbolic_equation_compiler import (
    SymbolicEquationCompilation,
    SymbolicPublication,
    compile_sympy_equations,
    compile_symbolic_program,
    symbolic_equations_cached,
)
from .vehicle_balloon_tire import balloon_tire_graph_abi


ABSTRACT_UI_VEHICLE_VERSION = "abstract-ui-vehicle-slot-v0"
DEFAULT_CAR_CONFIG = Path(__file__).parents[2] / "configs" / "vehicles" / "fun_car.json"
WHEEL_NAMES = ("front_left", "front_right", "rear_left", "rear_right")
DRAG_VECTOR_NAMES = ("longitudinal", "lateral", "vertical")
TIRE_RADIAL_RINGDOWN_STAGES = 3

_vehicle_build_progress_sink: Callable[[str, int, int, str], None] | None = None


def set_vehicle_build_progress_sink(
    sink: Callable[[str, int, int, str], None] | None,
) -> None:
    """Install a build-only progress sink; vehicle runtime data never depends on it."""
    global _vehicle_build_progress_sink
    _vehicle_build_progress_sink = sink


def _vehicle_build_progress(stage: str, completed: int, total: int, detail: str) -> None:
    if _vehicle_build_progress_sink is not None:
        _vehicle_build_progress_sink(stage, completed, total, detail)


@lru_cache(maxsize=1)
def contact_patch_lanes() -> tuple[tuple[str, str], ...]:
    """Every patch the chassis can present to the terrain, on one axis.

    ``constraint_reduction.contact`` declares this as
    "tire-patch-and-cage-node/member-midpoint-terrain-wrenches-to-chassis".
    The tyres and the shell are therefore not two systems bolted together --
    they are one lane axis through one contact law, reduced by one GEMM.  A
    cage bar landing on rock is the same event as a tyre landing on it, and
    nothing downstream should be able to tell them apart.

    Bolting a second, hand-written contact onto the side of the solved graph is
    what puts energy into the vehicle: a penalty spring and a stop-the-slide
    friction term are bounded by nothing the solver knows about.  Sharing this
    axis is what makes the shell obey the same saturating Coulomb law that
    keeps the tyres honest.
    """

    graph = _vehicle_mechanical_graph(load_default_car_configuration())
    lanes: list[tuple[str, str]] = [("tire-patch", wheel) for wheel in WHEEL_NAMES]
    lanes.extend(
        ("cage-node", node["identity"])
        for node in graph["nodes"]
        if node["kind"] == "roll-cage-node"
    )
    lanes.extend(
        ("cage-member", edge["identity"])
        for edge in graph["edges"]
        if str(edge["identity"]).startswith("cage.")
    )
    lanes.extend(
        ("body-shell-node", node["identity"])
        for node in graph["nodes"]
        if str(node["kind"]).startswith("body-shell-contact-")
    )
    return tuple(lanes)


def contact_lane_count() -> int:
    return len(contact_patch_lanes())
VEHICLE_STATE_OUTPUTS = (
    "position_x_next", "position_y_next", "position_z_next",
    "velocity_x_next", "velocity_y_next", "velocity_z_next",
    "roll_next", "pitch_next", "yaw_next",
    "roll_velocity_next", "pitch_velocity_next", "yaw_velocity_next",
    *(f"wheel_omega_{wheel}_next" for wheel in WHEEL_NAMES),
    "differential_wrench_shaft_omega_front_next",
    "differential_wrench_shaft_omega_rear_next",
    *(f"wheel_angle_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"slip_longitudinal_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"slip_sensor_velocity_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"friction_utilization_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"friction_utilization_sensor_velocity_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"tire_deformation_longitudinal_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"tire_deformation_velocity_longitudinal_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"tire_deformation_lateral_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"tire_deformation_velocity_lateral_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"traction_scale_{wheel}" for wheel in WHEEL_NAMES),
    *(f"brake_scale_{wheel}" for wheel in WHEEL_NAMES),
    *(f"compression_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"compression_velocity_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"spring_force_{wheel}" for wheel in WHEEL_NAMES),
    *(f"damper_scale_{wheel}" for wheel in WHEEL_NAMES),
    "engine_angular_speed_next", "engine_rpm",
    "traction_battery_charge_fraction_next", "regenerative_charge_power_w",
    "clutch_temperature_k_next", "clutch_health_next", "clutch_slip_power_w",
    "clutch_wear_next", "clutch_glaze_next",
    *(f"hub_locker_wear_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"hub_locker_glaze_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"differential_locker_wear_{axle}_next" for axle in ("front", "rear", "center")),
    *(f"differential_locker_glaze_{axle}_next" for axle in ("front", "rear", "center")),
    "alternator_cvt_ratio_state_next", "alternator_generated_power_w",
    "alternator_reaction_torque_nm", "alternator_cvt_wear_next", "alternator_cvt_glaze_next",
    "accessory_battery_cube_charge_fraction_next", "accessory_motor_shaft_torque_nm",
    "accessory_motor_engine_reaction_torque_nm", "accessory_motor_bus_power_w",
    "air_mix_reserve_gas_mass_kg_next", "air_mix_reserve_temperature_k_next",
    "air_mix_reserve_pressure_pa", "high_pressure_compressor_mass_flow_kg_s",
    "compressor_shaft_reaction_torque_nm", "compressor_engine_reaction_torque_nm",
    "direct_drive_bypass_engagement_next", "direct_drive_bypass_tooth_health_next",
    "direct_drive_bypass_torque_nm",
    "optional_fluid_coupling_torque_nm",
    "engine_torque", "clutch_torque", "transmission_output_torque",
    "driveline_torque", "front_differential_torque", "rear_differential_torque",
    "front_differential_wrench_torque", "rear_differential_wrench_torque",
    "engine_acceleration_torque", "engine_angular_acceleration",
    "powertrain_reaction_torque_x", "powertrain_reaction_torque_y",
    "powertrain_reaction_torque_z", "engine_mount_torque_x",
    "engine_mount_torque_y", "engine_mount_torque_z",
    "wheel_gyroscopic_reaction_torque_x", "wheel_gyroscopic_reaction_torque_y",
    "wheel_gyroscopic_reaction_torque_z",
    "traction_control_dissipation_torque", "service_brake_reaction_torque",
    "rolling_resistance_reaction_torque", "tire_contact_reaction_torque",
    "drivetrain_chassis_reaction_torque",
)


def _number(mapping: Mapping[str, Any], name: str, *, positive: bool = False) -> float:
    value = mapping.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"vehicle field {name!r} must be numeric")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"vehicle field {name!r} must be positive")
    return result


def _hollow_tube_mass(*, length: float, outer_radius: float,
                      wall_thickness: float, density: float) -> float:
    inner_radius = outer_radius - wall_thickness
    if length <= 0 or density <= 0 or outer_radius <= 0 or inner_radius <= 0:
        raise ValueError("hollow vehicle tube dimensions and density must be positive")
    return math.pi * (outer_radius ** 2 - inner_radius ** 2) * length * density


@dataclass(frozen=True, slots=True)
class VehicleConfiguration:
    """Validated, JSON-originating car configuration embedded in the product."""

    source: Mapping[str, Any]
    canonical_json: str
    digest: str

    @property
    def identity(self) -> str:
        return str(self.source["identity"])

    @property
    def name(self) -> str:
        return str(self.source["name"])

    def chassis_attachment_layout(self) -> dict[str, Any]:
        """Derive bumper/hanger geometry, density mass, and fit constraints."""

        source = self.source
        chassis, wheels = source["chassis"], source["wheels"]
        attachments = source["wrench_attachments"]
        bumpers, ballast = source["bumpers"], source["ballast"]
        half_length = _number(chassis, "half_length", positive=True)
        half_width = _number(chassis, "half_width", positive=True)
        wheelbase = _number(wheels, "wheelbase_half_length", positive=True)
        frame_corner_x = _number(chassis, "half_length", positive=True)
        frame_y = _number(chassis, "height", positive=True) * .72
        components: list[tuple[str, float, list[float]]] = []
        if bool(attachments["enabled"]):
            boss_mass = _number(attachments, "boss_mass_kg_each", positive=True)
            for corner in WHEEL_NAMES:
                longitudinal, lateral = corner.split("_")
                components.append((
                    f"wrench_attachment_{corner}", boss_mass,
                    [frame_corner_x if longitudinal == "front" else -frame_corner_x,
                     frame_y, -half_width * .78 if lateral == "left" else half_width * .78],
                ))
        bumper_rows: list[dict[str, Any]] = []
        if bool(bumpers["enabled"]):
            density = _number(bumpers, "material_density_kg_m3", positive=True)
            cross_radius = _number(bumpers, "cross_tube_outer_radius_m", positive=True)
            cross_wall = _number(bumpers, "cross_tube_wall_thickness_m", positive=True)
            cross_length = 2 * half_width * _number(
                bumpers, "cross_tube_width_scale", positive=True)
            mount_radius = _number(bumpers, "mount_tube_outer_radius_m", positive=True)
            mount_wall = _number(bumpers, "mount_tube_wall_thickness_m", positive=True)
            mount_length = _number(bumpers, "mount_tube_length_m", positive=True)
            rest_extension = _number(bumpers, "rest_extension_m", positive=True)
            cross_mass = _hollow_tube_mass(
                length=cross_length, outer_radius=cross_radius,
                wall_thickness=cross_wall, density=density)
            mount_mass = _hollow_tube_mass(
                length=mount_length, outer_radius=mount_radius,
                wall_thickness=mount_wall, density=density)
            shock_mass = _number(bumpers, "shock_body_mass_kg_each", positive=True)
            for longitudinal, sign in (("front", 1.0), ("rear", -1.0)):
                position = [sign * (half_length + mount_length + rest_extension),
                            frame_y - .02, 0.0]
                assembly_mass = cross_mass + 2 * (mount_mass + shock_mass)
                components.append((f"bumper_{longitudinal}", assembly_mass, position))
                bumper_rows.append({
                    "identity": longitudinal,
                    "center": position,
                    "cross_tube_length_m": cross_length,
                    "assembly_mass_kg": assembly_mass,
                })

        ballast_density = _number(ballast, "material_density_kg_m3", positive=True)
        block_width = _number(ballast, "block_width_m", positive=True)
        block_depth = _number(ballast, "block_depth_m", positive=True)
        maximum_drop = _number(ballast, "maximum_drop_m", positive=True)
        margin = _number(ballast, "ground_clearance_margin_m", positive=True)
        hanger_density = _number(ballast, "hanger_material_density_kg_m3", positive=True)
        hanger_radius = _number(ballast, "hanger_tube_outer_radius_m", positive=True)
        hanger_wall = _number(ballast, "hanger_tube_wall_thickness_m", positive=True)
        hanger_mass = _hollow_tube_mass(
            length=maximum_drop, outer_radius=hanger_radius,
            wall_thickness=hanger_wall, density=hanger_density)
        maximum_block_height = maximum_drop - margin
        if maximum_block_height <= 0:
            raise ValueError("ballast ground-clearance margin consumes the hanger drop")
        ballast_rows: list[dict[str, Any]] = []
        requested = ballast["requested_mass_kg"]
        for corner in WHEEL_NAMES:
            mass = float(requested[corner])
            if mass < 0 or not math.isfinite(mass):
                raise ValueError(f"ballast mass for {corner} must be finite and nonnegative")
            longitudinal, lateral = corner.split("_")
            x = frame_corner_x if longitudinal == "front" else -frame_corner_x
            z = -half_width * .78 if lateral == "left" else half_width * .78
            components.append((f"ballast_hanger_{corner}", hanger_mass,
                               [x, frame_y - maximum_drop * .5, z]))
            height = 0.0 if mass == 0 else mass / (
                ballast_density * block_width * block_depth)
            if height > maximum_block_height + 1e-12:
                capacity = ballast_density * block_width * block_depth * maximum_block_height
                raise ValueError(
                    f"ballast {corner} requests {mass:.6g} kg but the density/clearance "
                    f"geometry fits at most {capacity:.6g} kg")
            position = [x, frame_y - height * .5, z]
            if mass > 0:
                components.append((f"ballast_{corner}", mass, position))
            ballast_rows.append({
                "identity": corner, "requested_mass_kg": mass,
                "density_kg_m3": ballast_density,
                "volume_m3": mass / ballast_density,
                "dimensions_m": [block_depth, height, block_width],
                "center": position, "maximum_height_m": maximum_block_height,
                "fits": True,
            })
        return {
            "components": components,
            "bumpers": bumper_rows,
            "ballast": ballast_rows,
            "additional_ballast_mass_kg": sum(row["requested_mass_kg"] for row in ballast_rows),
            "fit_policy": "reject-configuration-before-graph-construction-on-volume-clearance-overflow",
        }

    def mass_properties(self) -> dict[str, Any]:
        """Derive one conserved rigid-body mass model from the component layout."""

        source = self.source
        attachment_layout = self.chassis_attachment_layout()
        total = (_number(source, "mass", positive=True)
                 + float(attachment_layout["additional_ballast_mass_kg"]))
        chassis, wheels, tires = source["chassis"], source["wheels"], source["tires"]
        powertrain, drivetrain = source["powertrain"], source["drivetrain"]
        fuel, electrical, body_shell = source["fuel_system"], source["electrical"], source["body_shell"]
        service_lines = source["service_lines"]
        wheelbase = _number(wheels, "wheelbase_half_length", positive=True)
        axle_offset = _number(wheels, "axle_group_offset_x_m")
        track = _number(wheels, "track_half_width", positive=True)
        radius = _number(tires, "radius", positive=True)
        wheel_mass = _number(drivetrain, "wheel_mass_kg", positive=True)
        tire_mass = _number(drivetrain, "tire_mass_kg", positive=True)
        engine_position = [float(value) for value in powertrain["engine_position"]]
        transmission_position = [engine_position[0] + .29, engine_position[1] - .015, 0.0]
        transfer_case_position = [engine_position[0] + .39, engine_position[1] - .035, 0.0]
        component_specs = [
            ("engine", _number(powertrain, "engine_mass_kg", positive=True), engine_position),
            ("transmission", _number(powertrain, "transmission_mass_kg", positive=True),
             transmission_position),
            ("transfer_case", _number(powertrain, "transfer_case_mass_kg", positive=True),
             transfer_case_position),
            ("front_differential", _number(powertrain, "front_differential_mass_kg", positive=True),
             [axle_offset + wheelbase, .065, 0.0]),
            ("rear_differential", _number(powertrain, "rear_differential_mass_kg", positive=True),
             [axle_offset - wheelbase, .065, 0.0]),
            ("fuel_tank_shell", _number(fuel, "tank_shell_mass_kg", positive=True),
             [float(fuel["tank_position_x"]), float(fuel["tank_position_y"]), float(fuel["tank_position_z"])]),
            ("fuel_live", _number(fuel, "initial_fuel_mass_kg", positive=True),
             [float(fuel["tank_position_x"]), float(fuel["tank_position_y"]), float(fuel["tank_position_z"])]),
            ("starter_battery", _number(electrical, "battery_mass_kg", positive=True), [.34, .15, -.22]),
            ("starter_motor", _number(electrical, "starter_mass_kg", positive=True),
             [engine_position[0] + .08, engine_position[1], -.16]),
            ("alternator", _number(electrical, "alternator_mass_kg", positive=True),
             [engine_position[0] - .10, engine_position[1] + .08, .16]),
            ("alternator_cvt", _number(electrical, "alternator_cvt_mass_kg", positive=True),
             [engine_position[0] - .05, engine_position[1] + .08, .08]),
            ("wiring_harness", _number(electrical, "wiring_and_lamps_mass_kg", positive=True), [0.0, .20, 0.0]),
            ("vehicle_computer", _number(electrical, "ecu_mass_kg", positive=True), [.18, .26, -.18]),
            ("transmission_control_unit", _number(electrical, "tcu_mass_kg", positive=True),
             [.16, .24, .14]),
            ("fusebox_relays", _number(electrical, "fusebox_and_relays_mass_kg", positive=True), [.30, .22, .18]),
            ("lamp_assemblies", _number(electrical, "lamp_assemblies_mass_kg", positive=True), [0.0, .22, 0.0]),
            ("steering_servo", _number(electrical, "steering_servo_mass_kg", positive=True), [.08, .29, -.10]),
            ("hydraulic_pump", _number(electrical, "hydraulic_pump_mass_kg", positive=True), [-.10, .10, .24]),
            ("pneumatic_compressor", _number(electrical, "pneumatic_compressor_mass_kg", positive=True), [-.28, .12, -.24]),
            ("pneumatic_accumulator", _number(electrical, "pneumatic_accumulator_mass_kg", positive=True), [-.42, .13, 0.0]),
            ("pneumatic_tire_manifold", _number(service_lines, "pneumatic_manifold_mass_kg", positive=True), [-.38, .15, .08]),
            ("pneumatic_chassis_lines", _number(service_lines, "pneumatic_chassis_line_mass_kg", positive=True), [0.0, .12, 0.0]),
            ("brake_master_manifold", _number(service_lines, "brake_master_manifold_mass_kg", positive=True), [.30, .20, -.08]),
            ("brake_chassis_lines", _number(service_lines, "brake_chassis_line_mass_kg", positive=True), [0.0, .11, 0.0]),
            ("parking_brake_equalizer", _number(service_lines, "parking_brake_equalizer_mass_kg", positive=True), [-.18, .16, 0.0]),
            ("alignment_manifold", _number(service_lines, "alignment_manifold_mass_kg", positive=True), [-.12, .18, .12]),
            ("alignment_chassis_lines", _number(service_lines, "alignment_chassis_line_mass_kg", positive=True), [0.0, .14, 0.0]),
            ("cosmetic_body_shell", _number(body_shell, "shell_mass_kg", positive=True), [0.0, .38, 0.0]),
            ("body_shell_mounts", _number(body_shell, "mount_mass_kg", positive=True), [0.0, .22, 0.0]),
            *attachment_layout["components"],
        ]
        mount_solution = solve_vehicle_wheel_placement_mounts(source)
        for corner, post in mount_solution["standard_corner_posts"].items():
            center = [(post["lower"][axis] + post["upper"][axis]) * .5 for axis in range(3)]
            component_specs.append((
                f"suspension_mount_post_{corner}",
                float(mount_solution["post_mass_kg_each"]), center,
            ))
        suspension = source["suspension"]
        knuckle_mass = _number(suspension, "knuckle_upright_mass_kg", positive=True)
        caliper_mass = _number(suspension, "brake_caliper_mass_kg", positive=True)
        rotor_mass = _number(suspension, "brake_rotor_mass_kg", positive=True)
        coilover_mass = _number(suspension, "coilover_mass_kg", positive=True)
        coilover_unsprung_fraction = _number(suspension, "coilover_unsprung_fraction", positive=True)
        alignment_actuator_mass = _number(
            suspension, "alignment_strain_relief_actuator_mass_kg_each", positive=True)
        knuckle_break_bushing_mass = _number(
            suspension, "knuckle_break_bushing_mass_kg_each", positive=True)
        service_unsprung_per_corner = (
            _number(service_lines, "pneumatic_service_loop_mass_kg_each", positive=True)
            + _number(service_lines, "pneumatic_rotary_union_mass_kg_each", positive=True)
            + _number(service_lines, "pneumatic_wheel_valve_mass_kg_each", positive=True)
            + _number(service_lines, "brake_service_hose_mass_kg_each", positive=True)
            + _number(service_lines, "alignment_service_loop_mass_kg_each", positive=True)
            + _number(service_lines, "parking_brake_cable_mass_kg", positive=True) / 4
        )
        removed_hub_material = _number(
            service_lines, "pneumatic_outer_valve_removed_hub_material_kg_each", positive=True)
        unsprung_per_corner = (wheel_mass - removed_hub_material + tire_mass
                                + knuckle_mass + caliper_mass + rotor_mass
                                + coilover_mass * coilover_unsprung_fraction
                                + 3 * alignment_actuator_mass
                                + 3 * knuckle_break_bushing_mass
                                + service_unsprung_per_corner)
        for corner in WHEEL_NAMES:
            longitudinal, lateral = corner.split("_")
            # The generalized compression coordinate owns suspension travel.
            # The wheel/hub reference must be the same chassis-local attachment
            # used by the compiled contact kernel; adding rest length, static
            # compression, and tire radius here displaced the rendered/massed
            # rim from its bead-bound physical tire by a constant offset.
            hub_y = -float(chassis["clearance"])
            position = [axle_offset + (wheelbase if longitudinal == "front" else -wheelbase),
                        hub_y, -track if lateral == "left" else track]
            component_specs.append((f"wheel_{corner}", wheel_mass - removed_hub_material, position))
            component_specs.append((f"tire_{corner}", tire_mass, list(position)))
            knuckle_position = [position[0], position[1],
                                 (-1 if lateral == "left" else 1) * (
                                     track - float(wheels["hub_face_offset"]))]
            coilover_top = [position[0] - .025, float(chassis["height"]) * .72 + .13,
                            (-1 if lateral == "left" else 1) * float(chassis["half_width"]) * .30]
            component_specs.extend((
                (f"knuckle_upright_{corner}", knuckle_mass, knuckle_position),
                (f"brake_caliper_{corner}", caliper_mass,
                 [knuckle_position[0] - .025, knuckle_position[1] + .025, knuckle_position[2]]),
                (f"brake_rotor_{corner}", rotor_mass, list(position)),
                (f"coilover_unsprung_{corner}", coilover_mass * coilover_unsprung_fraction,
                 [knuckle_position[0], knuckle_position[1] - .085, knuckle_position[2]]),
                (f"coilover_sprung_{corner}", coilover_mass * (1 - coilover_unsprung_fraction),
                 coilover_top),
                (f"pneumatic_service_loop_{corner}",
                 _number(service_lines, "pneumatic_service_loop_mass_kg_each", positive=True),
                 list(knuckle_position)),
                (f"pneumatic_rotary_union_{corner}",
                 _number(service_lines, "pneumatic_rotary_union_mass_kg_each", positive=True),
                 list(knuckle_position)),
                (f"pneumatic_wheel_valve_{corner}",
                 _number(service_lines, "pneumatic_wheel_valve_mass_kg_each", positive=True),
                 list(position)),
                (f"brake_service_hose_{corner}",
                 _number(service_lines, "brake_service_hose_mass_kg_each", positive=True),
                 list(knuckle_position)),
                (f"alignment_service_loop_{corner}",
                 _number(service_lines, "alignment_service_loop_mass_kg_each", positive=True),
                 list(knuckle_position)),
                (f"alignment_actuator_{corner}_upper_forward", alignment_actuator_mass,
                 list(knuckle_position)),
                (f"alignment_actuator_{corner}_upper_rear", alignment_actuator_mass,
                 list(knuckle_position)),
                (f"alignment_actuator_{corner}_tie_rod", alignment_actuator_mass,
                 list(knuckle_position)),
                (f"knuckle_break_bushing_{corner}_upper", knuckle_break_bushing_mass,
                 list(knuckle_position)),
                (f"knuckle_break_bushing_{corner}_lower", knuckle_break_bushing_mass,
                 list(knuckle_position)),
                (f"knuckle_break_bushing_{corner}_tie_rod", knuckle_break_bushing_mass,
                 list(knuckle_position)),
            ))
            if longitudinal == "rear":
                component_specs.append((
                    f"parking_brake_cable_{corner}",
                    _number(service_lines, "parking_brake_cable_mass_kg", positive=True) / 2,
                    list(knuckle_position),
                ))
        allocated = sum(item[1] for item in component_specs)
        residual = total - allocated
        if residual <= 0:
            raise ValueError("vehicle component masses must leave positive frame/cage/driver mass")
        base_position = [0.0, float(chassis["height"]) * .5, 0.0]
        components = [("frame_cage_driver", residual, base_position), *component_specs]
        center = [sum(mass * position[axis] for _, mass, position in components) / total
                  for axis in range(3)]
        length = 2 * float(chassis["half_length"])
        width = 2 * float(chassis["half_width"])
        height = float(chassis["height"])
        inertias = [
            residual * (height ** 2 + width ** 2) / 12,
            residual * (length ** 2 + height ** 2) / 12,
            residual * (length ** 2 + width ** 2) / 12,
        ]
        for _, mass, position in components:
            offset = [position[axis] - center[axis] for axis in range(3)]
            inertias[0] += mass * (offset[1] ** 2 + offset[2] ** 2)
            inertias[1] += mass * (offset[0] ** 2 + offset[1] ** 2)
            inertias[2] += mass * (offset[0] ** 2 + offset[2] ** 2)
        front_fraction = max(0.0, min(
            1.0, (center[0] - axle_offset + wheelbase) / (2 * wheelbase)))
        return {
            "total_mass_kg": total,
            "allocated_component_mass_kg": allocated,
            "residual_frame_cage_driver_misc_mass_kg": residual,
            "residual_frame_cage_driver_mass_kg": residual,
            "live_mass_components": [
                "fuel_live", *(f"ballast_{corner}" for corner in WHEEL_NAMES),
            ],
            "chassis_attachment_layout": attachment_layout,
            "center_of_mass": center,
            "inertia_kg_m2": {axis: inertias[index] for index, axis in enumerate(("roll", "pitch", "yaw"))},
            "derived_axle_fractions": {"front": front_fraction, "rear": 1 - front_fraction},
            "components": [{"identity": identity, "mass_kg": mass, "local_position": position}
                           for identity, mass, position in components],
        }

    def wheel_rotational_inertia(self) -> float:
        """Wheel, hub, and tire polar inertia derived from declared masses."""

        wheels, tires, drivetrain, suspension = (self.source["wheels"], self.source["tires"],
                                     self.source["drivetrain"], self.source["suspension"])
        rim_radius = _number(wheels, "rim_radius", positive=True)
        tire_radius = _number(tires, "radius", positive=True)
        wheel_mass = _number(drivetrain, "wheel_mass_kg", positive=True)
        tire_mass = _number(drivetrain, "tire_mass_kg", positive=True)
        scale = _number(drivetrain, "rotational_inertia_scale", positive=True)
        rotor_mass = _number(suspension, "brake_rotor_mass_kg", positive=True)
        rotor_radius = _number(suspension, "brake_rotor_effective_radius_m", positive=True)
        # Rim is approximately a hoop; balloon tire is a thick annulus.
        return (scale * (wheel_mass * rim_radius ** 2
                         + .5 * tire_mass * (rim_radius ** 2 + tire_radius ** 2))
                + .5 * rotor_mass * rotor_radius ** 2)

    def unsprung_mass_per_corner(self) -> float:
        suspension = self.source["suspension"]
        drivetrain = self.source["drivetrain"]
        service_lines = self.source["service_lines"]
        return (
            _number(drivetrain, "wheel_mass_kg", positive=True)
            + _number(drivetrain, "tire_mass_kg", positive=True)
            + _number(suspension, "knuckle_upright_mass_kg", positive=True)
            + _number(suspension, "brake_caliper_mass_kg", positive=True)
            + _number(suspension, "brake_rotor_mass_kg", positive=True)
            + _number(suspension, "coilover_mass_kg", positive=True)
            * _number(suspension, "coilover_unsprung_fraction", positive=True)
            + 3 * _number(suspension, "alignment_strain_relief_actuator_mass_kg_each", positive=True)
            + 3 * _number(suspension, "knuckle_break_bushing_mass_kg_each", positive=True)
            + _number(service_lines, "pneumatic_service_loop_mass_kg_each", positive=True)
            + _number(service_lines, "pneumatic_rotary_union_mass_kg_each", positive=True)
            + _number(service_lines, "pneumatic_wheel_valve_mass_kg_each", positive=True)
            + _number(service_lines, "brake_service_hose_mass_kg_each", positive=True)
            + _number(service_lines, "alignment_service_loop_mass_kg_each", positive=True)
            + _number(service_lines, "parking_brake_cable_mass_kg", positive=True) / 4
        )

    def sprung_mass(self) -> float:
        mass = _number(self.source, "mass", positive=True) - len(WHEEL_NAMES) * self.unsprung_mass_per_corner()
        if mass <= 0:
            raise ValueError("vehicle unsprung assemblies consume all chassis sprung mass")
        return mass

    def parameter_defaults(self) -> dict[str, float]:
        suspension = self.source["suspension"]
        controls = self.source["controls"]
        aerodynamics = self.source["aerodynamics"]
        drivetrain = self.source["drivetrain"]
        tires = self.source["tires"]
        powertrain = self.source["powertrain"]
        electrical = self.source["electrical"]
        chassis = self.source["chassis"]
        mass_properties = self.mass_properties()
        unsprung_mass = self.unsprung_mass_per_corner()
        sprung_mass = self.sprung_mass()
        world = self.source["world"]
        roll, yaw, pitch = (math.radians(float(value))
                            for value in powertrain["engine_orientation_degrees"])
        cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll),
                                  math.cos(pitch), math.sin(pitch),
                                  math.cos(yaw), math.sin(yaw))
        # The reference crank follows the longitudinal engine-to-gearbox edge.
        # Apply the same local roll -> pitch -> yaw convention as the chassis.
        rolled = (1.0, 0.0, 0.0)
        pitched = (rolled[0] * cp - rolled[1] * sp,
                   rolled[0] * sp + rolled[1] * cp, rolled[2])
        crank_axis = (pitched[0] * cy - pitched[2] * sy, pitched[1],
                      pitched[0] * sy + pitched[2] * cy)
        engine_position = tuple(float(value) for value in powertrain["engine_position"])
        transmission_position = (engine_position[0] + .29, engine_position[1] - .015, 0.0)
        transfer_case_position = (engine_position[0] + .39, engine_position[1] - .035, 0.0)
        drag_defaults: dict[str, float] = {
            "air_density": _number(aerodynamics, "air_density_kg_m3", positive=True),
        }
        for identity in DRAG_VECTOR_NAMES:
            drag = aerodynamics["drag_vectors"][identity]
            for index, axis in enumerate("xyz"):
                drag_defaults[f"drag_{identity}_vector_{axis}"] = float(drag["vector"][index])
            drag_defaults[f"drag_{identity}_coefficient"] = _number(
                drag, "coefficient", positive=True)
            drag_defaults[f"drag_{identity}_reference_area"] = _number(
                drag, "reference_area_m2", positive=True)
        return {
            # The chassis integrator owns only the sprung graph. Each corner's
            # wheel/upright/brake/lower-coilover assembly is integrated through
            # its suspension coordinate below; including it here as well would
            # silently bind the same mass to two generalized coordinates.
            "inverse_mass": 1.0 / sprung_mass,
            "gravity": _number(world, "gravity"),
            "suspension_rest_length": _number(suspension, "rest_length", positive=True),
            "spring_model_selector": _number(suspension, "spring_model_selector"),
            **{name: _number(suspension, name, positive=True) for name in (
                "spring_progressive_quadratic_n_per_m2", "spring_progressive_cubic_n_per_m3",
                "spring_primary_wire_diameter_m", "spring_primary_mean_coil_diameter_m",
                "spring_primary_active_turns", "spring_primary_shear_modulus_pa",
                "spring_secondary_wire_diameter_m", "spring_secondary_mean_coil_diameter_m",
                "spring_secondary_active_turns", "spring_secondary_shear_modulus_pa",
                "spring_secondary_engagement_compression_m",
                "spring_composite_coupling_efficiency", "bump_stop_start_fraction_of_travel",
            )},
            "suspension_travel": _number(suspension, "travel", positive=True),
            "chassis_clearance": _number(self.source["chassis"], "clearance", positive=True),
            "spring_stiffness": _number(suspension, "stiffness", positive=True),
            "bump_stop_stiffness": _number(
                suspension, "bump_stop_stiffness_n_per_m", positive=True),
            "bump_stop_progressive_stiffness": _number(
                suspension, "bump_stop_progressive_stiffness_n_per_m2", positive=True),
            "bump_stop_damping": _number(
                suspension, "bump_stop_damping_n_s_per_m", positive=True),
            "pneumatic_compression_damping": _number(
                suspension, "pneumatic_compression_damping", positive=True),
            "pneumatic_rebound_damping": _number(
                suspension, "pneumatic_rebound_damping", positive=True),
            "pneumatic_efficiency": _number(suspension, "pneumatic_efficiency", positive=True),
            "maximum_compression_speed": _number(
                suspension, "maximum_compression_speed", positive=True),
            **{name: _number(suspension, name, positive=True) for name in (
                "active_damping_minimum_scale", "active_damping_maximum_scale",
                "active_damping_body_velocity_gain_s_per_m",
                "active_damping_rebound_release_gain_s_per_m",
            )},
            "wheelbase_half_length": _number(self.source["wheels"], "wheelbase_half_length", positive=True),
            "axle_group_offset_x": _number(self.source["wheels"], "axle_group_offset_x_m"),
            "track_half_width": _number(self.source["wheels"], "track_half_width", positive=True),
            **{f"linkage_motion_ratio_{wheel}": 1.0 for wheel in WHEEL_NAMES},
            **{f"material_plastic_set_{wheel}": 0.0 for wheel in WHEEL_NAMES},
            **{f"material_survival_{wheel}": 1.0 for wheel in WHEEL_NAMES},
            "assembly_alpha_drivetrain": 1.0,
            **{f"assembly_alpha_{wheel}": 1.0 for wheel in WHEEL_NAMES},
            **{f"external_hub_torque_{wheel}": 0.0 for wheel in WHEEL_NAMES},
            **{f"hub_locker_engagement_{wheel}": 1.0 for wheel in WHEEL_NAMES},
            "external_differential_wrench_torque_front": 0.0,
            "external_differential_wrench_torque_rear": 0.0,
            "external_differential_inertia_front": 0.0,
            "external_differential_inertia_rear": 0.0,
            "differential_wrench_shaft_omega_front": 0.0,
            "differential_wrench_shaft_omega_rear": 0.0,
            "power_unit_electric_mode": 0.0,
            "traction_battery_charge_fraction": 0.72,
            "traction_battery_target_charge_fraction": 0.80,
            "traction_battery_capacity_j": 216_000_000.0,
            "regenerative_charge_efficiency": 0.84,
            "clutch_temperature_k": 300.0,
            "clutch_health": 1.0,
            "clutch_thermal_capacity_j_per_k": 22_000.0,
            "clutch_cooling_w_per_k": 45.0,
            "clutch_failure_temperature_k": 650.0,
            "clutch_wear_energy_j": 24_000_000.0,
            "clutch_wear": 0.0,
            "clutch_glaze": 0.0,
            **{f"hub_locker_wear_{wheel}": 0.0 for wheel in WHEEL_NAMES},
            **{f"hub_locker_glaze_{wheel}": 0.0 for wheel in WHEEL_NAMES},
            **{f"differential_locker_wear_{axle}": 0.0
               for axle in ("front", "rear", "center")},
            **{f"differential_locker_glaze_{axle}": 0.0
               for axle in ("front", "rear", "center")},
            "alternator_cvt_wear": 0.0,
            "alternator_cvt_glaze": 0.0,
            "direct_drive_bypass_command": 0.0,
            "direct_drive_bypass_engagement": 0.0,
            "direct_drive_bypass_tooth_health": 1.0,
            "direct_drive_bypass_shift_rate_per_s": 8.0,
            "direct_drive_bypass_maximum_sync_slip_rad_s": 2.0,
            "external_engine_flywheel_inertia": 0.0,
            "optional_fluid_coupling_engagement": 0.0,
            "optional_fluid_coupling_coefficient_nm_per_rad_s2": 0.02,
            "optional_fluid_coupling_maximum_torque_nm": 420.0,
            "alternator_count": _number(electrical, "alternator_count", positive=True),
            "alternator_max_power_w": _number(electrical, "alternator_max_power_w", positive=True),
            "alternator_rotor_inertia_each": _number(
                electrical, "alternator_rotor_inertia_kg_m2_each", positive=True),
            "alternator_efficiency": _number(electrical, "alternator_efficiency", positive=True),
            "alternator_cvt_ratio": _number(electrical, "alternator_cvt_ratio", positive=True),
            "alternator_cvt_ratio_state": _number(
                electrical, "alternator_cvt_ratio", positive=True),
            "alternator_cvt_efficiency": _number(
                electrical, "alternator_cvt_efficiency", positive=True),
            "alternator_cvt_ratio_response_hz": _number(
                electrical, "alternator_cvt_ratio_response_hz", positive=True),
            "alternator_electrical_demand_w": (
                _number(electrical, "base_load_w", positive=True)
                + _number(electrical, "ecu_load_w", positive=True)),
            "accessory_motor_command": _number(electrical, "accessory_motor_command"),
            "accessory_motor_peak_power_w": _number(
                electrical, "accessory_motor_peak_power_w", positive=True),
            "accessory_motor_peak_torque_nm": _number(
                electrical, "accessory_motor_peak_torque_nm", positive=True),
            "accessory_motor_drive_efficiency": _number(
                electrical, "accessory_motor_drive_efficiency", positive=True),
            "accessory_motor_regeneration_efficiency": _number(
                electrical, "accessory_motor_regeneration_efficiency", positive=True),
            "accessory_battery_cube_capacity_j": _number(
                electrical, "accessory_battery_cube_capacity_j", positive=True),
            "accessory_battery_cube_charge_fraction": _number(
                electrical, "accessory_battery_cube_initial_charge_fraction", positive=True),
            "accessory_battery_cube_nominal_voltage_v": _number(
                electrical, "accessory_battery_cube_nominal_voltage_v", positive=True),
            "accessory_battery_cube_maximum_discharge_current_a": _number(
                electrical, "accessory_battery_cube_maximum_discharge_current_a", positive=True),
            "accessory_battery_cube_maximum_charge_current_a": _number(
                electrical, "accessory_battery_cube_maximum_charge_current_a", positive=True),
            "accessory_battery_cube_internal_resistance_ohm": _number(
                electrical, "accessory_battery_cube_internal_resistance_ohm", positive=True),
            "high_pressure_compressor_command": _number(
                electrical, "high_pressure_compressor_command"),
            **{name: _number(electrical, name, positive=True) for name in (
                "high_pressure_compressor_displacement_m3_per_rev",
                "high_pressure_compressor_volumetric_efficiency",
                "high_pressure_compressor_isentropic_efficiency",
                "high_pressure_compressor_maximum_pressure_pa",
                "air_mix_reserve_volume_m3", "air_mix_reserve_initial_temperature_k",
                "air_mix_reserve_inlet_pressure_pa", "air_mix_reserve_inlet_temperature_k",
                "air_mix_reserve_specific_gas_constant_j_per_kg_k",
                "air_mix_reserve_specific_heat_ratio", "air_mix_reserve_cooling_w_per_k",
            )},
            "air_mix_reserve_gas_mass_kg": _number(
                electrical, "air_mix_reserve_initial_gas_mass_kg", positive=True),
            "air_mix_reserve_temperature_k": _number(
                electrical, "air_mix_reserve_initial_temperature_k", positive=True),
            "air_mix_reserve_gas_demand_kg_s": _number(
                electrical, "air_mix_reserve_gas_demand_kg_s"),
            **{f"unsprung_mass_{wheel}": unsprung_mass for wheel in WHEEL_NAMES},
            "tire_radial_effective_mass": unsprung_mass * _number(
                tires, "radial_contact_effective_mass_fraction_of_unsprung", positive=True),
            **{f"inverse_inertia_{axis}": 1.0 / mass_properties["inertia_kg_m2"][axis]
               for axis in ("roll", "pitch", "yaw")},
            **{f"center_of_mass_{axis}": mass_properties["center_of_mass"][index]
               for index, axis in enumerate("xyz")},
            "engine_displacement_m3": _number(powertrain, "displacement_liters", positive=True) / 1000,
            "engine_enabled": 1.0,
            "fuel_torque_scale": 1.0,
            "ignition_torque_scale": 1.0,
            "accessory_load_torque": 0.0,
            "governor_angular_speed": _number(powertrain, "redline_rpm", positive=True) * 2 * math.pi / 60,
            "brake_mean_effective_pressure": _number(
                powertrain, "brake_mean_effective_pressure_pa", positive=True),
            "engine_braking_mean_effective_pressure": _number(
                powertrain, "engine_braking_mean_effective_pressure_pa", positive=True),
            **{f"engine_{name}_angular_speed": _number(powertrain, f"{name}_rpm", positive=True)
               * 2 * math.pi / 60 for name in ("idle", "torque_peak", "power_peak", "redline")},
            "engine_angular_speed": _number(powertrain, "idle_rpm", positive=True) * 2 * math.pi / 60,
            "drive_direction": 1.0,
            # Runtime selection may replace this with L1/L2, but an omitted
            # control must mean the mechanically connected high range rather
            # than silently opening the transfer case.
            "transfer_case_ratio": 1.0,
            "drive_fraction_front_left": _number(
                drivetrain, "front_drive_fraction", positive=True) / 2,
            "drive_fraction_front_right": _number(
                drivetrain, "front_drive_fraction", positive=True) / 2,
            "drive_fraction_rear_left": _number(
                drivetrain, "rear_drive_fraction", positive=True) / 2,
            "drive_fraction_rear_right": _number(
                drivetrain, "rear_drive_fraction", positive=True) / 2,
            "clutch_stiffness": _number(powertrain, "clutch_stiffness_nm_per_rad_s", positive=True),
            "clutch_maximum_torque": _number(powertrain, "clutch_maximum_torque_nm", positive=True),
            **{name: _number(powertrain, name, positive=True) for name in (
                "combustion_efficiency", "clutch_efficiency", "forward_gear_ratio",
                "reverse_gear_ratio", "final_drive_ratio", "driveline_efficiency",
            )},
            "engine_mass": _number(powertrain, "engine_mass_kg", positive=True),
            "transmission_mass": _number(powertrain, "transmission_mass_kg", positive=True),
            "transfer_case_mass": _number(powertrain, "transfer_case_mass_kg", positive=True),
            "engine_rotating_inertia": _number(
                powertrain, "engine_rotating_inertia_kg_m2", positive=True),
            **{f"engine_position_{axis}": engine_position[index]
               for index, axis in enumerate("xyz")},
            **{f"transmission_position_{axis}": transmission_position[index]
               for index, axis in enumerate("xyz")},
            **{f"transfer_case_position_{axis}": transfer_case_position[index]
               for index, axis in enumerate("xyz")},
            **{f"crank_axis_{axis}": crank_axis[index]
               for index, axis in enumerate("xyz")},
            "brake_torque": _number(drivetrain, "brake_torque_nm", positive=True),
            "differential_brake_torque": _number(
                drivetrain, "differential_brake_torque_nm", positive=True),
            "differential_brake_rotor_inertia": _number(
                drivetrain, "differential_brake_rotor_inertia_kg_m2", positive=True),
            "wheel_inertia": self.wheel_rotational_inertia(),
            "differential_lock_stiffness": _number(
                drivetrain, "differential_lock_stiffness_nm_per_rad_s", positive=True),
            "differential_lock_maximum_torque": _number(
                drivetrain, "differential_lock_maximum_torque_nm", positive=True),
            "transfer_case_efficiency": _number(
                drivetrain, "transfer_case_efficiency", positive=True),
            "transfer_case_drag_torque": _number(
                drivetrain, "transfer_case_drag_torque_nm", positive=True),
            "rolling_resistance_torque": _number(
                drivetrain, "rolling_resistance_torque_nm", positive=True),
            "maximum_wheel_speed": _number(
                drivetrain, "maximum_wheel_speed_rad_s", positive=True),
            "wheel_radius": _number(self.source["tires"], "radius", positive=True),
            "angular_damping": _number(controls, "angular_damping", positive=True),
            **drag_defaults,
            **{name: _number(self.source["traction_control"], name, positive=True) for name in (
                "target_friction_utilization", "throttle_intervention_gain",
                "brake_intervention_gain", "slip_growth_gain",
                "slip_growth_reference_m_s2", "minimum_torque_fraction",
                "slip_sensor_frequency_hz", "slip_sensor_damping_ratio",
                "utilization_sensor_frequency_hz", "utilization_sensor_damping_ratio",
            )},
            "tire_longitudinal_deformation_frequency_hz": _number(
                tires, "longitudinal_deformation_mode_frequency_hz", positive=True),
            "tire_lateral_deformation_frequency_hz": _number(
                tires, "lateral_deformation_mode_frequency_hz", positive=True),
            "tire_sidewall_deformation_damping_ratio": _number(
                tires, "sidewall_deformation_damping_ratio", positive=True),
            "tire_maximum_sidewall_deformation": _number(
                tires, "maximum_sidewall_deformation_m", positive=True),
        }

    def to_data(self) -> dict[str, Any]:
        result = json.loads(self.canonical_json)
        result["source"] = {
            "format": "json", "sha256": self.digest,
            "path_hint": "configs/vehicles/fun_car.json",
        }
        return result


def vehicle_configuration_from_mapping(value: Mapping[str, Any]) -> VehicleConfiguration:
    required = {
        "schema", "identity", "name", "kind", "mass", "mass_distribution", "chassis", "body_packaging", "wheels",
        "wheel_placement_demands", "tires", "tire_skin",
        "suspension", "solid_contact", "drivetrain", "transmission", "powertrain", "fuel_system",
        "electrical", "service_lines", "body_shell", "wrench_attachments", "bumpers", "ballast",
        "controls", "aerodynamics", "traction_control",
        "world", "presentation",
    }
    missing = required - set(value)
    unknown = set(value) - required
    if missing or unknown:
        raise ValueError(f"vehicle configuration fields missing={sorted(missing)} unknown={sorted(unknown)}")
    if value["schema"] != "abstract-ui-vehicle-config-v1" or value["kind"] != "car":
        raise ValueError("the initial vehicle configuration must be a v1 car")
    for name in ("identity", "name"):
        if not isinstance(value[name], str) or not value[name]:
            raise ValueError(f"vehicle field {name!r} must be a non-empty string")
    _number(value, "mass", positive=True)
    expected = {
        "chassis": {"half_width", "half_length", "height", "clearance", "camera_height"},
        "body_packaging": {"preset", "requested_cab_length_m", "bed_length_m",
                           "front_clip_length_m", "seat_rows", "minimum_row_pitch_m",
                           "front_seating_clearance_m", "rear_seating_clearance_m",
                           "minimum_cab_length_m", "frame_end_margin_m", "wheelbase_policy",
                           "wheelbase_to_body_length_ratio", "minimum_intertire_clearance_m",
                           "target_front_axle_load_fraction", "wheelbase_solver_iterations"},
        "wheels": {"track_half_width", "wheelbase_half_length", "axle_group_offset_x_m",
                   "rim_radius", "hub_face_offset"},
        "tires": {"radius", "pressure_pa", "width", "minimum_contact_area", "maximum_contact_area",
                  "static_friction", "kinetic_friction", "load_sensitivity",
                  "toroid_section_radius_m", "effective_tread_width_fraction", "gas_polytropic_exponent",
                  "reference_temperature_k", "gas_molar_mass_kg_per_mol",
                  "gas_specific_heat_ratio", "membrane_gas_permeability_mol_m_per_m2_s_pa",
                  "gas_permeability_activation_energy_j_per_mol",
                  "radial_carcass_loss_n_s_per_m", "radial_contact_effective_mass_fraction_of_unsprung",
                  "sidewall_shear_stiffness_longitudinal_n_per_m",
                  "sidewall_shear_stiffness_lateral_n_per_m", "sidewall_shear_damping_n_s_per_m",
                  "longitudinal_deformation_mode_frequency_hz", "lateral_deformation_mode_frequency_hz",
                  "sidewall_deformation_damping_ratio", "maximum_sidewall_deformation_m",
                  "slip_transition_speed"},
        "tire_skin": {"model", "circumferential_segments", "section_segments",
                      "skin_thickness_m", "lame_lambda_pa", "lame_mu_pa",
                      "membrane_damping_lambda_pa_s", "membrane_damping_mu_pa_s",
                      "bending_stiffness_nm",
                      "bead_stiffness_n_per_m", "bead_damping_n_s_per_m",
                      "minimum_volume_fraction", "contact_skin_offset_m",
                      "contact_restitution",
                      "tread_thickness_scale", "tread_stiffness_scale", "tread_damping_scale",
                      "sidewall_thickness_scale", "sidewall_stiffness_scale", "sidewall_damping_scale",
                      "bead_thickness_scale", "bead_stiffness_scale", "bead_damping_scale",
                      "pneumatic_mode", "ambient_temperature_k",
                      "rubber_density_kg_m3", "rubber_specific_heat_j_per_kg_k",
                      "rubber_thermal_conductivity_w_per_m_k", "rubber_thermal_expansion_per_k",
                      "composite_density_kg_m3", "composite_specific_heat_j_per_kg_k",
                      "composite_thermal_conductivity_w_per_m_k", "composite_bias_angle_deg",
                      "steel_density_kg_m3", "steel_specific_heat_j_per_kg_k",
                      "steel_thermal_conductivity_w_per_m_k", "tube_thickness_m",
                      "tube_density_kg_m3", "tube_specific_heat_j_per_kg_k",
                      "tube_thermal_conductivity_w_per_m_k"},
        "mass_distribution": set(WHEEL_NAMES),
        "suspension": {"rest_length", "assembly_hub_height_m", "travel", "stiffness",
                       "spring_model_selector", "spring_progressive_quadratic_n_per_m2",
                       "spring_progressive_cubic_n_per_m3", "spring_primary_wire_diameter_m",
                       "spring_primary_mean_coil_diameter_m", "spring_primary_active_turns",
                       "spring_primary_shear_modulus_pa", "spring_secondary_wire_diameter_m",
                       "spring_secondary_mean_coil_diameter_m", "spring_secondary_active_turns",
                       "spring_secondary_shear_modulus_pa",
                       "spring_secondary_engagement_compression_m",
                       "spring_composite_coupling_efficiency",
                       "bump_stop_start_fraction_of_travel",
                       "bump_stop_stiffness_n_per_m",
                       "bump_stop_progressive_stiffness_n_per_m2",
                       "bump_stop_damping_n_s_per_m", "pneumatic_compression_damping",
                       "pneumatic_rebound_damping", "pneumatic_efficiency",
                       "maximum_compression_speed",
                       "leveling_actuator_piston_area_m2",
                       "leveling_manifold_pressure_pa",
                       "leveling_maximum_flow_m3_s",
                       "leveling_hydraulic_efficiency",
                       "leveling_pressure_force_reserve_fraction",
                       "leveling_coarse_rate_m_s", "leveling_trim_rate_m_s",
                       "leveling_trim_stroke_m", "leveling_trim_entry_error_m",
                       "leveling_sensor_force_bandwidth_hz",
                       "leveling_sensor_position_bandwidth_hz",
                       "leveling_sensor_pressure_bandwidth_hz",
                       "leveling_sensor_motion_bandwidth_hz",
                       "leveling_sensor_force_range_n", "leveling_sensor_position_range_m",
                       "leveling_sensor_pressure_range_pa", "leveling_sensor_motion_range_m_s",
                       "active_damping_minimum_scale",
                       "active_damping_maximum_scale", "active_damping_body_velocity_gain_s_per_m",
                       "active_damping_rebound_release_gain_s_per_m",
                       "knuckle_upright_mass_kg", "brake_caliper_mass_kg", "brake_rotor_mass_kg",
                       "brake_rotor_effective_radius_m", "coilover_mass_kg", "coilover_unsprung_fraction",
                       "alignment_strain_relief_actuator_mass_kg_each",
                       "alignment_strain_relief_stiffness_n_per_m",
                       "alignment_strain_relief_damping_n_s_per_m",
                       "alignment_strain_relief_holding_force_n",
                       "alignment_strain_relief_relief_force_n",
                       "alignment_strain_relief_maximum_stroke_m",
                       "alignment_strain_relief_maximum_rate_m_per_s",
                       "alignment_strain_relief_recenter_force_n",
                       "alignment_strain_relief_recenter_rate_m_per_s",
                       "knuckle_break_bushing_mass_kg_each",
                       "knuckle_break_bushing_yield_force_n",
                       "knuckle_break_bushing_fracture_force_n",
                       "knuckle_break_bushing_yield_displacement_m",
                       "knuckle_break_bushing_fracture_displacement_m",
                       "knuckle_break_bushing_yield_moment_nm",
                       "knuckle_break_bushing_fracture_moment_nm"},
        "solid_contact": {"static_friction", "kinetic_friction", "restitution",
                          "penetration_bias", "maximum_correction_speed", "cage_contact_radius",
                          "cage_contact_stiffness", "cage_contact_damping", "cage_contact_maximum_force",
                          "cage_static_friction", "cage_kinetic_friction"},
        "drivetrain": {"brake_torque_nm", "differential_brake_torque_nm",
                       "differential_brake_rotor_inertia_kg_m2",
                       "front_drive_fraction", "rear_drive_fraction",
                       "rolling_resistance_torque_nm", "maximum_wheel_speed_rad_s",
                       "wheel_mass_kg", "tire_mass_kg", "rotational_inertia_scale",
                       "differential_lock_stiffness_nm_per_rad_s",
                       "differential_lock_maximum_torque_nm", "transfer_case_efficiency",
                       "transfer_case_drag_torque_nm"},
        "transmission": {"mode_default", "starting_gear", "crawler_gear", "forward_ratios", "reverse_ratio",
                         "low_range_ratio", "ultra_low_range_ratio",
                         "minimum_shift_interval_s", "upshift_wheel_speed_rad_s",
                         "downshift_wheel_speed_rad_s", "upshift_torque_reserve",
                         "downshift_torque_reserve", "crawler_entry_throttle",
                         "crawler_entry_speed_m_s", "ratio_response_frequency_hz",
                         "ratio_response_damping_ratio", "downshift_demand_frequency_hz",
                         "downshift_demand_damping_ratio", "downshift_commit_level"},
        "powertrain": {"displacement_liters", "brake_mean_effective_pressure_pa",
                       "engine_braking_mean_effective_pressure_pa", "idle_rpm", "torque_peak_rpm",
                       "power_peak_rpm", "redline_rpm", "clutch_stiffness_nm_per_rad_s",
                       "clutch_maximum_torque_nm",
                       "combustion_efficiency", "clutch_efficiency", "forward_gear_ratio",
                       "reverse_gear_ratio", "final_drive_ratio", "driveline_efficiency",
                       "engine_mass_kg", "transmission_mass_kg", "front_differential_mass_kg",
                       "rear_differential_mass_kg", "transfer_case_mass_kg",
                       "engine_rotating_inertia_kg_m2", "engine_position",
                       "engine_orientation_degrees"},
        "fuel_system": {"capacity_kg", "initial_fuel_mass_kg", "tank_shell_mass_kg",
                        "fuel_energy_density_j_per_kg", "tank_position_x", "tank_position_y",
                        "tank_position_z"},
        "electrical": {"battery_mass_kg", "battery_capacity_wh", "initial_state_of_charge",
                       "nominal_voltage", "starter_mass_kg", "starter_power_w", "starter_torque_nm",
                       "starter_cranking_speed_rad_s", "alternator_mass_kg", "alternator_count",
                       "alternator_max_power_w", "alternator_rotor_inertia_kg_m2_each",
                       "alternator_efficiency", "alternator_cvt_mass_kg", "alternator_cvt_ratio",
                       "alternator_cvt_efficiency", "alternator_cvt_ratio_response_hz",
                       "accessory_motor_command", "accessory_motor_peak_power_w",
                       "accessory_motor_peak_torque_nm", "accessory_motor_drive_efficiency",
                       "accessory_motor_regeneration_efficiency",
                       "accessory_battery_cube_capacity_j",
                       "accessory_battery_cube_initial_charge_fraction",
                       "accessory_battery_cube_nominal_voltage_v",
                       "accessory_battery_cube_maximum_discharge_current_a",
                       "accessory_battery_cube_maximum_charge_current_a",
                       "accessory_battery_cube_internal_resistance_ohm",
                       "high_pressure_compressor_command",
                       "high_pressure_compressor_displacement_m3_per_rev",
                       "high_pressure_compressor_volumetric_efficiency",
                       "high_pressure_compressor_isentropic_efficiency",
                       "high_pressure_compressor_maximum_pressure_pa",
                       "air_mix_reserve_volume_m3", "air_mix_reserve_initial_gas_mass_kg",
                       "air_mix_reserve_initial_temperature_k",
                       "air_mix_reserve_inlet_pressure_pa",
                       "air_mix_reserve_inlet_temperature_k",
                       "air_mix_reserve_specific_gas_constant_j_per_kg_k",
                       "air_mix_reserve_specific_heat_ratio",
                       "air_mix_reserve_cooling_w_per_k",
                       "air_mix_reserve_gas_demand_kg_s",
                       "base_load_w", "ecu_load_w", "headlight_load_w", "tail_light_load_w",
                       "brake_light_load_w", "horn_load_w", "wiring_and_lamps_mass_kg", "ecu_mass_kg",
                       "tcu_mass_kg",
                       "fusebox_and_relays_mass_kg", "lamp_assemblies_mass_kg", "hydraulic_pump_mass_kg",
                       "steering_servo_mass_kg", "steering_servo_peak_power_w", "steering_servo_idle_power_w",
                       "hydraulic_pump_power_w", "pneumatic_compressor_mass_kg",
                       "pneumatic_accumulator_mass_kg", "pneumatic_compressor_power_w",
                       "pneumatic_pressure_rate_pa_s", "minimum_tire_pressure_pa", "maximum_tire_pressure_pa"},
        "service_lines": {"pneumatic_manifold_mass_kg", "pneumatic_chassis_line_mass_kg",
                          "pneumatic_service_loop_mass_kg_each", "pneumatic_rotary_union_mass_kg_each",
                          "pneumatic_wheel_valve_mass_kg_each",
                          "pneumatic_outer_valve_removed_hub_material_kg_each",
                          "pneumatic_hard_line_radius_m",
                          "pneumatic_hose_radius_m", "brake_master_manifold_mass_kg",
                          "brake_chassis_line_mass_kg", "brake_service_hose_mass_kg_each",
                          "brake_hard_line_radius_m", "brake_hose_radius_m",
                          "parking_brake_equalizer_mass_kg", "parking_brake_cable_mass_kg",
                          "parking_brake_cable_radius_m", "alignment_manifold_mass_kg",
                          "alignment_chassis_line_mass_kg", "alignment_service_loop_mass_kg_each",
                          "alignment_hard_line_radius_m", "alignment_hose_radius_m"},
        "body_shell": {"shell_mass_kg", "mount_mass_kg", "contact_radius_m",
                       "contact_stiffness_n_per_m", "contact_damping_n_s_per_m",
                       "contact_maximum_force_n", "mount_yield_force_n", "mount_fracture_force_n"},
        "wrench_attachments": {"enabled", "boss_mass_kg_each", "bolt_circle_radius_m",
                               "maximum_force_n", "maximum_moment_nm", "yield_force_n",
                               "fracture_force_n", "yield_moment_nm", "fracture_moment_nm"},
        "bumpers": {"enabled", "material_density_kg_m3", "cross_tube_outer_radius_m",
                    "cross_tube_wall_thickness_m", "cross_tube_width_scale",
                    "mount_tube_outer_radius_m", "mount_tube_wall_thickness_m",
                    "mount_tube_length_m", "shock_body_mass_kg_each", "rest_extension_m",
                    "maximum_compression_m", "preload_force_n", "compression_stiffness_n_per_m",
                    "compression_damping_n_s_per_m", "rebound_damping_n_s_per_m",
                    "maximum_force_n"},
        "ballast": {"enabled", "material", "material_density_kg_m3", "block_width_m",
                    "block_depth_m", "maximum_drop_m", "ground_clearance_margin_m",
                    "hanger_tube_outer_radius_m", "hanger_tube_wall_thickness_m",
                    "hanger_material_density_kg_m3", "requested_mass_kg"},
        "controls": {"maximum_steering_angle_degrees", "angular_damping",
                     "throttle_rise_rate_per_s", "throttle_fall_rate_per_s",
                     "input_response_frequency_hz", "input_response_damping_ratio"},
        "aerodynamics": {"air_density_kg_m3", "drag_vectors"},
        "traction_control": {"target_friction_utilization", "throttle_intervention_gain",
                             "brake_intervention_gain", "slip_growth_gain",
                             "slip_growth_reference_m_s2", "minimum_torque_fraction",
                             "response_frequency_hz", "response_damping_ratio",
                             "slip_sensor_frequency_hz", "slip_sensor_damping_ratio",
                             "utilization_sensor_frequency_hz", "utilization_sensor_damping_ratio"},
        "world": {"gravity", "fixed_step_hz", "maximum_substeps"},
        "presentation": {
            "palette_role", "terrain_palette_role", "wheel_palette_role",
            "wheel_tread_palette_role", "world_tile_size", "world_tile_major_every",
            "world_tile_strength", "chase_camera_distance", "chase_camera_height",
            "chase_camera_look_ahead", "chase_camera_position_response",
            "chase_camera_facing_response", "chase_camera_speed_pullback",
            "chase_camera_ground_clearance",
        },
    }
    for section, names in expected.items():
        record = value[section]
        if not isinstance(record, Mapping) or set(record) != names:
            raise ValueError(f"vehicle section {section!r} must contain exactly {sorted(names)}")
    placement_demands = value["wheel_placement_demands"]
    placement_names = {
        "lateral_wheel_distance_m", "longitudinal_wheel_distance_m", "hub_height_m",
        "axle_group_fore_aft_m", "actuation_frame_clearance_m", "mount_synthesis",
    }
    if not isinstance(placement_demands, Mapping) or set(placement_demands) != placement_names:
        raise ValueError(
            "vehicle wheel_placement_demands must contain exactly "
            f"{sorted(placement_names)}")
    for criterion in placement_names - {"mount_synthesis"}:
        bounds = placement_demands[criterion]
        if not isinstance(bounds, Mapping) or set(bounds) != {"minimum", "maximum"}:
            raise ValueError(f"wheel-placement demand {criterion!r} needs minimum/maximum")
        resolved = []
        for bound_name in ("minimum", "maximum"):
            bound = bounds[bound_name]
            if bound is None:
                resolved.append(None)
                continue
            if (not isinstance(bound, (int, float)) or isinstance(bound, bool)
                    or not math.isfinite(float(bound))):
                raise ValueError(
                    f"wheel-placement demand {criterion}.{bound_name} must be finite or null")
            resolved.append(float(bound))
        if resolved[0] is not None and resolved[1] is not None and resolved[0] > resolved[1]:
            raise ValueError(f"wheel-placement demand {criterion} minimum exceeds maximum")
    mount_synthesis = placement_demands["mount_synthesis"]
    if not isinstance(mount_synthesis, Mapping) or set(mount_synthesis) != {
            "enabled", "actuation_samples", "steering_samples", "maximum_pickup_shift_m",
            "post_height_m", "post_outer_radius_m", "post_wall_thickness_m",
            "post_material_density_kg_m3", "selected_architecture",
            "compatible_architectures", "planned_architectures"}:
        raise ValueError("wheel-placement mount_synthesis has an invalid record")
    if not isinstance(mount_synthesis["enabled"], bool):
        raise ValueError("wheel-placement mount_synthesis.enabled must be boolean")
    for name in ("actuation_samples", "steering_samples"):
        count = mount_synthesis[name]
        if not isinstance(count, int) or isinstance(count, bool) or count < 3:
            raise ValueError(f"wheel-placement mount_synthesis.{name} must be an integer >= 3")
    _number(mount_synthesis, "maximum_pickup_shift_m", positive=True)
    for name in ("post_height_m", "post_outer_radius_m", "post_wall_thickness_m",
                 "post_material_density_kg_m3"):
        _number(mount_synthesis, name, positive=True)
    if float(mount_synthesis["post_wall_thickness_m"]) >= float(
            mount_synthesis["post_outer_radius_m"]):
        raise ValueError("wheel-placement post wall thickness must be smaller than radius")
    architectures = mount_synthesis["compatible_architectures"]
    planned_architectures = mount_synthesis["planned_architectures"]
    selected_architecture = mount_synthesis["selected_architecture"]
    if (not isinstance(architectures, list) or not architectures
            or any(not isinstance(item, str) or not item for item in architectures)
            or len(set(architectures)) != len(architectures)):
        raise ValueError("wheel-placement compatible_architectures must be unique names")
    if selected_architecture not in architectures:
        raise ValueError("wheel-placement selected architecture must be compatible with its post")
    if (not isinstance(planned_architectures, list)
            or any(not isinstance(item, str) or not item for item in planned_architectures)
            or set(planned_architectures) & set(architectures)):
        raise ValueError("planned suspension architectures must be named and not claim compatibility")
    for section in ("wrench_attachments", "bumpers", "ballast"):
        if not isinstance(value[section]["enabled"], bool):
            raise ValueError(f"vehicle {section}.enabled must be boolean")
    tire_skin = value["tire_skin"]
    if tire_skin["model"] != "compiled-balloon-skin-v1":
        raise ValueError("vehicle tire_skin.model must select compiled-balloon-skin-v1")
    for name in ("circumferential_segments", "section_segments"):
        count = tire_skin[name]
        if not isinstance(count, int) or isinstance(count, bool) or count < 6:
            raise ValueError(f"vehicle tire_skin.{name} must be an integer >= 6")
    if tire_skin["section_segments"] % 2:
        raise ValueError("vehicle tire_skin.section_segments must be even")
    if tire_skin["pneumatic_mode"] not in {"tubeless", "tube"}:
        raise ValueError("vehicle tire_skin.pneumatic_mode must be tubeless or tube")
    for name in expected["tire_skin"] - {"model", "pneumatic_mode", "circumferential_segments", "section_segments"}:
        _number(tire_skin, name, positive=True)
    if float(tire_skin["minimum_volume_fraction"]) >= 1.0:
        raise ValueError("vehicle tire_skin.minimum_volume_fraction must be below one")
    if not isinstance(value["ballast"]["material"], str) or not value["ballast"]["material"]:
        raise ValueError("vehicle ballast.material must be a non-empty string")
    requested_ballast = value["ballast"]["requested_mass_kg"]
    if not isinstance(requested_ballast, Mapping) or set(requested_ballast) != set(WHEEL_NAMES):
        raise ValueError("vehicle ballast.requested_mass_kg must name all four chassis corners")
    for corner in WHEEL_NAMES:
        mass = requested_ballast[corner]
        if (not isinstance(mass, (int, float)) or isinstance(mass, bool)
                or not math.isfinite(float(mass)) or float(mass) < 0):
            raise ValueError(f"vehicle ballast mass for {corner} must be finite and nonnegative")
    for name in expected["bumpers"] - {"enabled"}:
        _number(value["bumpers"], name, positive=True)
    for name in expected["wrench_attachments"] - {"enabled"}:
        _number(value["wrench_attachments"], name, positive=True)
    for name in expected["ballast"] - {"enabled", "material", "requested_mass_kg"}:
        _number(value["ballast"], name, positive=True)
    for section, outer_name, wall_name in (
        ("bumpers", "cross_tube_outer_radius_m", "cross_tube_wall_thickness_m"),
        ("bumpers", "mount_tube_outer_radius_m", "mount_tube_wall_thickness_m"),
        ("ballast", "hanger_tube_outer_radius_m", "hanger_tube_wall_thickness_m"),
    ):
        if float(value[section][wall_name]) >= float(value[section][outer_name]):
            raise ValueError(f"vehicle {section}.{wall_name} must be smaller than {outer_name}")
    packaging = value["body_packaging"]
    if not isinstance(packaging["preset"], str) or not packaging["preset"]:
        raise ValueError("vehicle body_packaging.preset must be a non-empty name")
    if packaging["wheelbase_policy"] not in {"manual", "center-under-mass"}:
        raise ValueError("vehicle body_packaging.wheelbase_policy must be manual or center-under-mass")
    for integer_name in ("seat_rows", "wheelbase_solver_iterations"):
        if (not isinstance(packaging[integer_name], int)
                or isinstance(packaging[integer_name], bool) or packaging[integer_name] < 1):
            raise ValueError(f"vehicle body_packaging.{integer_name} must be a positive integer")
    for name in expected["body_packaging"] - {
            "preset", "wheelbase_policy", "seat_rows", "wheelbase_solver_iterations"}:
        _number(packaging, name, positive=name != "frame_end_margin_m")
    if float(packaging["frame_end_margin_m"]) < 0:
        raise ValueError("vehicle body_packaging.frame_end_margin_m cannot be negative")
    if not 0 < float(packaging["target_front_axle_load_fraction"]) < 1:
        raise ValueError("vehicle target front axle load fraction must be between zero and one")
    for section in ("chassis", "wheels", "tires", "suspension", "solid_contact", "drivetrain",
                    "electrical", "service_lines", "body_shell", "controls", "traction_control",
                    "mass_distribution"):
        for name in expected[section]:
            _number(value[section], name,
                    positive=name not in {"axle_group_offset_x_m", "accessory_motor_command",
                                          "high_pressure_compressor_command",
                                          "air_mix_reserve_gas_demand_kg_s",
                                          "spring_model_selector"})
    spring_selector = value["suspension"]["spring_model_selector"]
    if (not isinstance(spring_selector, int) or isinstance(spring_selector, bool)
            or spring_selector not in {0, 1, 2}):
        raise ValueError("vehicle suspension.spring_model_selector must be 0, 1, or 2")
    for name in ("spring_composite_coupling_efficiency", "bump_stop_start_fraction_of_travel",
                 "leveling_hydraulic_efficiency", "leveling_pressure_force_reserve_fraction"):
        if float(value["suspension"][name]) > 1:
            raise ValueError(f"vehicle suspension.{name} cannot exceed one")
    aerodynamics = value["aerodynamics"]
    _number(aerodynamics, "air_density_kg_m3", positive=True)
    drag_vectors = aerodynamics["drag_vectors"]
    if not isinstance(drag_vectors, Mapping) or set(drag_vectors) != set(DRAG_VECTOR_NAMES):
        raise ValueError("vehicle aerodynamics.drag_vectors must name the three compiled axes")
    for identity in DRAG_VECTOR_NAMES:
        row = drag_vectors[identity]
        if not isinstance(row, Mapping) or set(row) != {
                "vector", "coefficient", "reference_area_m2"}:
            raise ValueError(f"vehicle drag vector {identity!r} has an invalid record")
        _number(row, "coefficient", positive=True)
        _number(row, "reference_area_m2", positive=True)
        vector = row["vector"]
        if not isinstance(vector, list) or len(vector) != 3:
            raise ValueError(f"vehicle drag vector {identity!r} must have three components")
        components = [float(component) for component in vector]
        if any(not math.isfinite(component) for component in components):
            raise ValueError(f"vehicle drag vector {identity!r} must be finite")
        if not math.isclose(sum(component * component for component in components),
                            1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"vehicle drag vector {identity!r} must be unit length")
    for name in expected["powertrain"] - {"engine_position", "engine_orientation_degrees"}:
        _number(value["powertrain"], name, positive=True)
    for name in ("capacity_kg", "initial_fuel_mass_kg", "tank_shell_mass_kg",
                 "fuel_energy_density_j_per_kg"):
        _number(value["fuel_system"], name, positive=True)
    for name in ("tank_position_x", "tank_position_y", "tank_position_z"):
        _number(value["fuel_system"], name)
    if float(value["fuel_system"]["initial_fuel_mass_kg"]) > float(value["fuel_system"]["capacity_kg"]):
        raise ValueError("initial fuel mass cannot exceed tank capacity")
    if float(value["electrical"]["initial_state_of_charge"]) > 1:
        raise ValueError("initial battery state of charge cannot exceed one")
    if float(value["electrical"]["accessory_battery_cube_initial_charge_fraction"]) > 1:
        raise ValueError("accessory battery cube state of charge cannot exceed one")
    transmission = value["transmission"]
    if transmission["mode_default"] != "automatic":
        raise ValueError("the initial transmission mode must be automatic")
    ratios = transmission["forward_ratios"]
    upshift = transmission["upshift_wheel_speed_rad_s"]
    downshift = transmission["downshift_wheel_speed_rad_s"]
    if (not isinstance(ratios, list) or len(ratios) < 2 or
            any(not isinstance(item, (int, float)) or isinstance(item, bool) or item <= 0 for item in ratios)):
        raise ValueError("transmission forward_ratios must contain at least two positive ratios")
    if any(ratios[index] <= ratios[index + 1] for index in range(len(ratios) - 1)):
        raise ValueError("transmission ratios must descend from crawler to top gear")
    if (not isinstance(upshift, list) or not isinstance(downshift, list)
            or len(upshift) != len(ratios) or len(downshift) != len(ratios)):
        raise ValueError("transmission speed schedules must match the forward ratio count")
    for schedule_name, schedule in (("upshift", upshift), ("downshift", downshift)):
        if any(not isinstance(item, (int, float)) or isinstance(item, bool) or item < 0 for item in schedule):
            raise ValueError(f"transmission {schedule_name} schedule must be non-negative")
    for name in ("starting_gear", "crawler_gear"):
        gear = transmission[name]
        if not isinstance(gear, int) or isinstance(gear, bool) or not 1 <= gear <= len(ratios):
            raise ValueError(f"transmission {name} must select a forward ratio")
    for name in ("reverse_ratio", "low_range_ratio", "ultra_low_range_ratio", "minimum_shift_interval_s", "upshift_torque_reserve",
                 "downshift_torque_reserve", "crawler_entry_throttle", "crawler_entry_speed_m_s",
                 "ratio_response_frequency_hz", "ratio_response_damping_ratio",
                 "downshift_demand_frequency_hz", "downshift_demand_damping_ratio",
                 "downshift_commit_level"):
        _number(transmission, name, positive=True)
    for name in ("engine_position", "engine_orientation_degrees"):
        vector = value["powertrain"][name]
        if (not isinstance(vector, list) or len(vector) != 3 or
                any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in vector)):
            raise ValueError(f"vehicle powertrain field {name!r} must be a numeric xyz triple")
    for name in ("world_tile_size", "world_tile_major_every", "world_tile_strength",
                 "chase_camera_distance", "chase_camera_height", "chase_camera_look_ahead",
                 "chase_camera_position_response", "chase_camera_facing_response",
                 "chase_camera_speed_pullback", "chase_camera_ground_clearance"):
        _number(value["presentation"], name, positive=True)
    for name in ("palette_role", "terrain_palette_role", "wheel_palette_role",
                 "wheel_tread_palette_role"):
        if not isinstance(value["presentation"][name], str) or not value["presentation"][name]:
            raise ValueError(f"vehicle presentation field {name!r} must be a non-empty string")
    if abs(sum(float(value["mass_distribution"][wheel]) for wheel in WHEEL_NAMES) - 1.0) > 1e-6:
        raise ValueError("vehicle mass-distribution fractions must sum to one")
    if abs(float(value["drivetrain"]["front_drive_fraction"])
           + float(value["drivetrain"]["rear_drive_fraction"]) - 1.0) > 1e-6:
        raise ValueError("front and rear drive fractions must sum to one")
    allocated_mass = (float(value["powertrain"]["engine_mass_kg"])
                      + float(value["powertrain"]["transmission_mass_kg"])
                      + float(value["powertrain"]["transfer_case_mass_kg"])
                      + float(value["powertrain"]["front_differential_mass_kg"])
                      + float(value["powertrain"]["rear_differential_mass_kg"])
                      + float(value["fuel_system"]["initial_fuel_mass_kg"])
                      + float(value["fuel_system"]["tank_shell_mass_kg"])
                      + float(value["electrical"]["battery_mass_kg"])
                      + float(value["electrical"]["starter_mass_kg"])
                      + float(value["electrical"]["alternator_mass_kg"])
                      + float(value["electrical"]["alternator_cvt_mass_kg"])
                      + float(value["electrical"]["tcu_mass_kg"])
                      + float(value["electrical"]["wiring_and_lamps_mass_kg"])
                      + float(value["electrical"]["steering_servo_mass_kg"])
                      + sum(float(value["service_lines"][name]) * multiplier for name, multiplier in (
                          ("pneumatic_manifold_mass_kg", 1),
                          ("pneumatic_chassis_line_mass_kg", 1),
                          ("pneumatic_service_loop_mass_kg_each", 4),
                          ("pneumatic_rotary_union_mass_kg_each", 4),
                          ("pneumatic_wheel_valve_mass_kg_each", 4),
                          ("brake_master_manifold_mass_kg", 1),
                          ("brake_chassis_line_mass_kg", 1),
                          ("brake_service_hose_mass_kg_each", 4),
                          ("parking_brake_equalizer_mass_kg", 1),
                          ("parking_brake_cable_mass_kg", 1),
                          ("alignment_manifold_mass_kg", 1),
                          ("alignment_chassis_line_mass_kg", 1),
                          ("alignment_service_loop_mass_kg_each", 4),
                      ))
                      + float(value["body_shell"]["shell_mass_kg"])
                      + float(value["body_shell"]["mount_mass_kg"])
                      + 12 * float(value["suspension"]["alignment_strain_relief_actuator_mass_kg_each"])
                      + 12 * float(value["suspension"]["knuckle_break_bushing_mass_kg_each"])
                      + 4 * (float(value["drivetrain"]["wheel_mass_kg"])
                             + float(value["drivetrain"]["tire_mass_kg"])
                             + float(value["suspension"]["knuckle_upright_mass_kg"])
                             + float(value["suspension"]["brake_caliper_mass_kg"])
                             + float(value["suspension"]["brake_rotor_mass_kg"])
                             + float(value["suspension"]["coilover_mass_kg"])))
    if allocated_mass >= float(value["mass"]):
        raise ValueError("vehicle component masses must leave positive frame/cage/driver mass")
    if float(value["suspension"]["pneumatic_efficiency"]) > 1.0:
        raise ValueError("pneumatic efficiency cannot exceed one")
    if float(value["suspension"]["coilover_unsprung_fraction"]) > 1.0:
        raise ValueError("coilover unsprung fraction cannot exceed one")
    if not (float(value["suspension"]["alignment_strain_relief_holding_force_n"])
            < float(value["suspension"]["alignment_strain_relief_relief_force_n"])
            < float(value["suspension"]["knuckle_break_bushing_yield_force_n"])
            < float(value["suspension"]["knuckle_break_bushing_fracture_force_n"])):
        raise ValueError("alignment relief and break-bushing force thresholds must be strictly ordered")
    if not (float(value["suspension"]["knuckle_break_bushing_yield_displacement_m"])
            < float(value["suspension"]["knuckle_break_bushing_fracture_displacement_m"])):
        raise ValueError("knuckle break-bushing displacement thresholds must be strictly ordered")
    if not (float(value["suspension"]["active_damping_minimum_scale"]) <= 1.0 <=
            float(value["suspension"]["active_damping_maximum_scale"])):
        raise ValueError("active damping scale range must contain the passive 1.0 setting")
    if float(value["traction_control"]["minimum_torque_fraction"]) > 1.0:
        raise ValueError("minimum torque fraction cannot exceed one")
    for name in ("combustion_efficiency", "clutch_efficiency", "driveline_efficiency"):
        if float(value["powertrain"][name]) > 1.0:
            raise ValueError(f"powertrain efficiency {name!r} cannot exceed one")
    if float(value["drivetrain"]["transfer_case_efficiency"]) > 1.0:
        raise ValueError("transfer-case efficiency cannot exceed one")
    _number(value["world"], "gravity")
    for name in ("fixed_step_hz", "maximum_substeps"):
        _number(value["world"], name, positive=True)
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    configuration = VehicleConfiguration(
        value, canonical, hashlib.sha256(canonical.encode()).hexdigest())
    configuration.chassis_attachment_layout()
    return configuration


@lru_cache(maxsize=1)
def load_default_car_configuration() -> VehicleConfiguration:
    return vehicle_configuration_from_mapping(json.loads(DEFAULT_CAR_CONFIG.read_text(encoding="utf-8")))


def _distance_point_to_segment(point: tuple[float, float, float],
                               a: tuple[float, float, float],
                               b: tuple[float, float, float]) -> float:
    delta = tuple(b[index] - a[index] for index in range(3))
    length_squared = sum(component * component for component in delta)
    if length_squared <= 1e-18:
        return math.sqrt(sum((point[index] - a[index]) ** 2 for index in range(3)))
    fraction = max(0.0, min(1.0, sum(
        (point[index] - a[index]) * delta[index] for index in range(3)
    ) / length_squared))
    return math.sqrt(sum(
        (point[index] - (a[index] + fraction * delta[index])) ** 2
        for index in range(3)
    ))


def solve_vehicle_wheel_placement_mounts(
    configuration: VehicleConfiguration | Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate wheel demands and synthesize the invariant corner-post brackets.

    This is deliberately an evaluator, not a hidden geometry mutator.  It owns
    the frame-to-suspension interface and reports whether the selected wheel
    coordinates clear the *structural frame members* through sampled suspension
    travel and steering.  A caller may explicitly apply a later proposal, but
    pan fitting cannot silently change wheel placement.
    """

    source = configuration.source if isinstance(configuration, VehicleConfiguration) else configuration
    chassis, wheels = source["chassis"], source["wheels"]
    suspension, tires = source["suspension"], source["tires"]
    demands = source["wheel_placement_demands"]
    mount = demands["mount_synthesis"]
    half_length = float(chassis["half_length"])
    half_width = float(chassis["half_width"])
    wheelbase = float(wheels["wheelbase_half_length"])
    axle_offset = float(wheels["axle_group_offset_x_m"])
    track = float(wheels["track_half_width"])
    frame_y = float(chassis["height"]) * .72
    post_height = float(mount["post_height_m"])
    post_z_scale = .78
    selected = {
        "lateral_wheel_distance_m": 2 * track,
        "longitudinal_wheel_distance_m": 2 * wheelbase,
        "hub_height_m": float(suspension["assembly_hub_height_m"]),
        "axle_group_fore_aft_m": axle_offset,
    }

    def bound_result(name: str, value: float) -> dict[str, Any]:
        bounds = demands[name]
        minimum = None if bounds["minimum"] is None else float(bounds["minimum"])
        maximum = None if bounds["maximum"] is None else float(bounds["maximum"])
        return {
            "value": value, "minimum": minimum, "maximum": maximum,
            "satisfied": ((minimum is None or value >= minimum)
                          and (maximum is None or value <= maximum)),
            "unconstrained": minimum is None and maximum is None,
        }

    criteria = {name: bound_result(name, value) for name, value in selected.items()}
    post_positions: dict[str, dict[str, list[float]]] = {}
    pickup_positions: dict[str, dict[str, list[float]]] = {}
    for corner in WHEEL_NAMES:
        longitudinal, lateral = corner.split("_")
        front = 1.0 if longitudinal == "front" else -1.0
        side = -1.0 if lateral == "left" else 1.0
        post_x = front * half_length
        post_z = side * half_width * post_z_scale
        post_positions[corner] = {
            "lower": [post_x, frame_y - post_height * .5, post_z],
            "upper": [post_x, frame_y + post_height * .5, post_z],
        }
        # Brackets stay on the standard post. Their fore/aft separation makes
        # each paired revolute axis physical while the arms may reach a hub
        # whose wheelbase is independent of frame length.
        pickup_positions[corner] = {
            "upper_pickup_forward": [post_x + .09, frame_y + post_height * .30, post_z],
            "upper_pickup_rear": [post_x - .09, frame_y + post_height * .30, post_z],
            "lower_pickup_forward": [post_x + .11, frame_y - post_height * .34, post_z],
            "lower_pickup_rear": [post_x - .11, frame_y - post_height * .34, post_z],
            "coilover_chassis": [post_x, frame_y + post_height * .44, post_z],
        }

    frame_corners = {
        "front_left": (half_length, frame_y, -half_width * post_z_scale),
        "front_right": (half_length, frame_y, half_width * post_z_scale),
        "rear_left": (-half_length, frame_y, -half_width * post_z_scale),
        "rear_right": (-half_length, frame_y, half_width * post_z_scale),
    }
    frame_pairs = (
        ("front_left", "front_right"), ("front_right", "rear_right"),
        ("rear_right", "rear_left"), ("rear_left", "front_left"),
        ("front_left", "rear_right"), ("front_right", "rear_left"),
    )
    frame_segments = [(frame_corners[a], frame_corners[b]) for a, b in frame_pairs]
    outer_radius = float(tires["radius"])
    section_radius = float(tires["toroid_section_radius_m"])
    major_radius = max(1e-6, outer_radius - section_radius)
    travel = float(suspension["travel"])
    nominal_hub_y = -float(chassis["clearance"])
    steering_limit = math.radians(float(source["controls"]["maximum_steering_angle_degrees"]))
    actuation_samples = int(mount["actuation_samples"])
    steering_samples = int(mount["steering_samples"])
    minimum_clearance = math.inf
    limiting_sample: dict[str, Any] | None = None
    # The torus sampling is only a packaging sweep. Runtime collision remains
    # the compiled deformable membrane/triangle CCD, never this approximation.
    for corner in WHEEL_NAMES:
        longitudinal, lateral = corner.split("_")
        front = 1.0 if longitudinal == "front" else -1.0
        hub_x = axle_offset + front * wheelbase
        hub_z = (-1.0 if lateral == "left" else 1.0) * track
        for travel_index in range(actuation_samples):
            compression = -travel * .5 + travel * travel_index / (actuation_samples - 1)
            hub_y = nominal_hub_y + compression
            for steering_index in range(steering_samples):
                steer = (-steering_limit + 2 * steering_limit * steering_index
                         / (steering_samples - 1)) if longitudinal == "front" else 0.0
                cosine, sine = math.cos(steer), math.sin(steer)
                for ring_index in range(24):
                    ring_angle = 2 * math.pi * ring_index / 24
                    for section_index in range(8):
                        section_angle = 2 * math.pi * section_index / 8
                        radial = major_radius + section_radius * math.cos(section_angle)
                        local_x = radial * math.cos(ring_angle)
                        local_y = radial * math.sin(ring_angle)
                        local_z = section_radius * math.sin(section_angle)
                        point = (hub_x + cosine * local_x + sine * local_z,
                                 hub_y + local_y,
                                 hub_z - sine * local_x + cosine * local_z)
                        clearance = min(_distance_point_to_segment(point, a, b)
                                        for a, b in frame_segments) - .018
                        if clearance < minimum_clearance:
                            minimum_clearance = clearance
                            limiting_sample = {
                                "corner": corner, "compression_m": compression,
                                "steering_angle_rad": steer, "skin_point_m": list(point),
                            }
    clearance_result = bound_result("actuation_frame_clearance_m", minimum_clearance)
    criteria["actuation_frame_clearance_m"] = clearance_result
    maximum_pickup_shift = float(mount["maximum_pickup_shift_m"])
    maximum_used_shift = .11
    post_wall = float(mount["post_wall_thickness_m"])
    post_outer = float(mount["post_outer_radius_m"])
    post_mass_each = _hollow_tube_mass(
        length=post_height, outer_radius=post_outer, wall_thickness=post_wall,
        density=float(mount["post_material_density_kg_m3"]),
    )
    return {
        "schema": "springtail-wheel-placement-and-mount-solution-v1",
        "authority": "standard-corner-post-to-explicit-suspension-graph",
        "mutates_wheel_placement": False,
        "selected": selected,
        "criteria": criteria,
        "satisfied": all(item["satisfied"] for item in criteria.values()),
        "standard_corner_posts": post_positions,
        "synthesized_pickups": pickup_positions,
        "post_mass_kg_each": post_mass_each,
        "post_total_mass_kg": 4 * post_mass_each,
        "selected_architecture": mount["selected_architecture"],
        "compatible_architectures": list(mount["compatible_architectures"]),
        "planned_architectures": list(mount["planned_architectures"]),
        "selector_admission": "implemented-complete-force-graph-only",
        "architecture_interfaces": {
            "double-wishbone-coilover": {
                "differential_mount": "chassis-sprung",
                "torque_route": "differential-articulated-halfshafts-cv-joints-locking-hubs",
            },
            "solid-axle-leaf-spring": {
                "differential_mount": "moving-unsprung-axle-housing",
                "torque_route": "transfer-case-telescoping-u-joint-propshaft-differential-enclosed-axle-shafts",
                "required_graph": ["axle-housing", "differential-carrier", "two-leaf-packs",
                                   "four-eyes", "two-shackles", "u-bolts", "dampers"],
            },
            "solid-axle-coilover": {
                "differential_mount": "moving-unsprung-axle-housing",
                "torque_route": "transfer-case-telescoping-u-joint-propshaft-differential-enclosed-axle-shafts",
                "required_graph": ["axle-housing", "differential-carrier", "locating-links",
                                   "panhard-or-watts-link", "coilovers"],
            },
        },
        "maximum_pickup_shift_m": maximum_pickup_shift,
        "maximum_used_pickup_shift_m": maximum_used_shift,
        "pickup_shift_satisfied": maximum_used_shift <= maximum_pickup_shift,
        "sweep": {
            "runtime_collision_authority": "compiled-balloon-membrane-triangle-ccd",
            "packaging_precheck": "sampled-torus-against-structural-frame-member-capsules",
            "actuation_samples": actuation_samples,
            "steering_samples": steering_samples,
            "minimum_clearance_m": minimum_clearance,
            "limiting_sample": limiting_sample,
        },
    }


def solve_vehicle_body_packaging(
    configuration: VehicleConfiguration | Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve arbitrary cab/bed input into a seatable longitudinal envelope."""

    source = configuration.source if isinstance(configuration, VehicleConfiguration) else configuration
    package = source["body_packaging"]
    minimum_from_seats = (
        int(package["seat_rows"]) * float(package["minimum_row_pitch_m"])
        + float(package["front_seating_clearance_m"])
        + float(package["rear_seating_clearance_m"])
    )
    effective_cab = max(
        float(package["requested_cab_length_m"]),
        float(package["minimum_cab_length_m"]),
        minimum_from_seats,
    )
    front_clip = float(package["front_clip_length_m"])
    bed = float(package["bed_length_m"])
    total = front_clip + effective_cab + bed
    front_end = total / 2.0
    cab_front = front_end - front_clip
    cab_rear = cab_front - effective_cab
    rear_end = cab_rear - bed
    required_half_length = total / 2.0 + float(package["frame_end_margin_m"])
    return {
        "schema": "springtail-body-packaging-solution-v1",
        "preset": package["preset"],
        "requested_cab_length_m": float(package["requested_cab_length_m"]),
        "minimum_cab_length_m": max(
            float(package["minimum_cab_length_m"]), minimum_from_seats),
        "effective_cab_length_m": effective_cab,
        "cab_was_expanded": effective_cab > float(package["requested_cab_length_m"]),
        "bed_length_m": bed,
        "front_clip_length_m": front_clip,
        "total_body_length_m": total,
        "required_chassis_half_length_m": required_half_length,
        "current_chassis_half_length_m": float(source["chassis"]["half_length"]),
        "frame_fit_satisfied": float(source["chassis"]["half_length"]) >= required_half_length,
        "longitudinal_stations_m": {
            "front_end": front_end,
            "hood_center": front_end - front_clip / 2.0,
            "cab_front": cab_front,
            "cab_center": (cab_front + cab_rear) / 2.0,
            "cab_rear": cab_rear,
            "bed_center": (cab_rear + rear_end) / 2.0,
            "rear_end": rear_end,
        },
        "wheel_placement_mutated": False,
    }


def fit_vehicle_chassis_to_body_packaging(
    configuration: VehicleConfiguration | Mapping[str, Any],
) -> tuple[VehicleConfiguration, dict[str, Any]]:
    """Expand only the structural frame needed by the selected cab/bed package."""

    source = copy.deepcopy(
        configuration.source if isinstance(configuration, VehicleConfiguration) else dict(configuration)
    )
    original = vehicle_configuration_from_mapping(source)
    solution = solve_vehicle_body_packaging(original)
    old_half_length = float(source["chassis"]["half_length"])
    new_half_length = max(old_half_length, float(solution["required_chassis_half_length_m"]))
    if new_half_length > old_half_length:
        old_attachment_mass = original.chassis_attachment_layout()["additional_ballast_mass_kg"]
        source["chassis"]["half_length"] = new_half_length
        added_rail_mass = 3.4 * 4.0 * (new_half_length - old_half_length)
        expanded_attachment_mass = vehicle_configuration_from_mapping(
            source).chassis_attachment_layout()["additional_ballast_mass_kg"]
        source["mass"] = float(source["mass"]) + added_rail_mass + (
            expanded_attachment_mass - old_attachment_mass)
    fitted = vehicle_configuration_from_mapping(source)
    result = solve_vehicle_body_packaging(fitted)
    result.update({
        "authority": "shared-vehicle-configuration-and-mechanical-graph-rebuild",
        "added_chassis_length_m": 2.0 * (new_half_length - old_half_length),
        "wheelbase_m": 2.0 * float(source["wheels"]["wheelbase_half_length"]),
        "track_m": 2.0 * float(source["wheels"]["track_half_width"]),
        "wheel_placement_mutated": False,
    })
    return fitted, result


def fit_vehicle_wheelbase_under_body_mass(
    configuration: VehicleConfiguration | Mapping[str, Any],
) -> tuple[VehicleConfiguration, dict[str, Any]]:
    """Place two axles beneath the rebuilt body/load COM at a requested load split."""

    source = copy.deepcopy(
        configuration.source if isinstance(configuration, VehicleConfiguration) else dict(configuration)
    )
    package = source["body_packaging"]
    initial_wheels = dict(source["wheels"])
    if package["wheelbase_policy"] == "manual":
        config = vehicle_configuration_from_mapping(source)
        return config, {
            "schema": "springtail-wheelbase-under-mass-solution-v1",
            "policy": "manual", "changed": False,
            "wheelbase_m": 2.0 * float(source["wheels"]["wheelbase_half_length"]),
            "axle_group_offset_x_m": float(source["wheels"]["axle_group_offset_x_m"]),
        }
    body = solve_vehicle_body_packaging(source)
    desired_wheelbase = max(
        float(body["total_body_length_m"])
        * float(package["wheelbase_to_body_length_ratio"]),
        2.0 * float(source["tires"]["radius"])
        + float(package["minimum_intertire_clearance_m"]),
    )
    front_fraction = float(package["target_front_axle_load_fraction"])
    history: list[dict[str, float]] = []
    for iteration in range(int(package["wheelbase_solver_iterations"])):
        candidate = vehicle_configuration_from_mapping(source)
        com_x = float(candidate.mass_properties()["center_of_mass"][0])
        front_x = com_x + (1.0 - front_fraction) * desired_wheelbase
        rear_x = com_x - front_fraction * desired_wheelbase
        half_wheelbase = (front_x - rear_x) / 2.0
        axle_offset = (front_x + rear_x) / 2.0
        previous_half = float(source["wheels"]["wheelbase_half_length"])
        previous_offset = float(source["wheels"]["axle_group_offset_x_m"])
        source["wheels"]["wheelbase_half_length"] = half_wheelbase
        source["wheels"]["axle_group_offset_x_m"] = axle_offset
        history.append({
            "iteration": float(iteration), "center_of_mass_x_m": com_x,
            "front_axle_x_m": front_x, "rear_axle_x_m": rear_x,
            "wheelbase_m": desired_wheelbase,
        })
        if (abs(previous_half - half_wheelbase) < 1e-7
                and abs(previous_offset - axle_offset) < 1e-7):
            break
    fitted = vehicle_configuration_from_mapping(source)
    final_com_x = float(fitted.mass_properties()["center_of_mass"][0])
    front_x = float(source["wheels"]["axle_group_offset_x_m"]) + float(
        source["wheels"]["wheelbase_half_length"])
    rear_x = float(source["wheels"]["axle_group_offset_x_m"]) - float(
        source["wheels"]["wheelbase_half_length"])
    achieved_front_fraction = (final_com_x - rear_x) / max(front_x - rear_x, 1e-12)
    return fitted, {
        "schema": "springtail-wheelbase-under-mass-solution-v1",
        "policy": "center-under-mass", "changed": source["wheels"] != initial_wheels,
        "body_length_m": float(body["total_body_length_m"]),
        "wheelbase_m": front_x - rear_x,
        "front_axle_x_m": front_x, "rear_axle_x_m": rear_x,
        "axle_group_offset_x_m": (front_x + rear_x) / 2.0,
        "center_of_mass_x_m": final_com_x,
        "target_front_axle_load_fraction": front_fraction,
        "achieved_front_axle_load_fraction": achieved_front_fraction,
        "iterations": history,
        "body_mass_was_moved_to_fit_axles": False,
        "wheel_placement_authority": "axles-solved-under-rebuilt-total-mass-center",
    }


def fit_vehicle_chassis_to_power_unit(
    configuration: VehicleConfiguration | Mapping[str, Any],
    *,
    engine_envelope_m: tuple[float, float, float],
    oil_pan_envelope_m: tuple[float, float, float],
    engine_mass_kg: float,
) -> tuple[VehicleConfiguration, dict[str, Any]]:
    """Rebuild shared vehicle geometry/mass around an installed power unit.

    The oil pan is the first hard package constraint; the overall engine then
    expands the same bay.  This function belongs to the shared vehicle model
    so the game, native rig, and design tools cannot acquire different fits.
    """

    source = copy.deepcopy(
        configuration.source if isinstance(configuration, VehicleConfiguration)
        else dict(configuration)
    )
    original_configuration = vehicle_configuration_from_mapping(source)
    original_attachment_mass = original_configuration.chassis_attachment_layout()[
        "additional_ballast_mass_kg"]
    original_residual_mass = original_configuration.mass_properties()[
        "residual_frame_cage_driver_misc_mass_kg"]
    engine_length, engine_height, engine_width = map(float, engine_envelope_m)
    pan_length, pan_depth, pan_width = map(float, oil_pan_envelope_m)
    if min(engine_length, engine_height, engine_width, pan_length, pan_depth, pan_width,
           float(engine_mass_kg)) <= 0:
        raise ValueError("power-unit package dimensions and mass must be positive")
    chassis, wheels, powertrain = source["chassis"], source["wheels"], source["powertrain"]
    original_half_length = float(chassis["half_length"])
    original_half_width = float(chassis["half_width"])
    mount_margin, pan_margin = .11, .13
    bay_length = max(engine_length + 2 * mount_margin, pan_length + 2 * pan_margin)
    bay_width = max(engine_width + 2 * mount_margin, pan_width + 2 * pan_margin)
    required_half_length = bay_length / 2 + .70
    required_half_width = bay_width / 2 + .09
    chassis["half_length"] = max(original_half_length, required_half_length)
    chassis["half_width"] = max(original_half_width, required_half_width)
    added_rail_length = 4 * max(0.0, float(chassis["half_length"]) - original_half_length)
    added_crossmember_length = 4 * max(0.0, float(chassis["half_width"]) - original_half_width)
    added_frame_mass = 3.4 * (added_rail_length + added_crossmember_length)
    expanded_attachment_mass = vehicle_configuration_from_mapping(
        source).chassis_attachment_layout()["additional_ballast_mass_kg"]
    added_attachment_mass = expanded_attachment_mass - original_attachment_mass
    source["mass"] = (
        float(source["mass"]) - float(powertrain["engine_mass_kg"])
        + float(engine_mass_kg) + added_frame_mass + added_attachment_mass
    )
    powertrain["engine_mass_kg"] = float(engine_mass_kg)
    powertrain["engine_position"] = [
        min(.18, bay_length * .08),
        max(float(powertrain["engine_position"][1]), pan_depth + .035),
        0.0,
    ]
    # Tube/member and bumper geometry are authoritative, so estimate formulas
    # must never leave a negative residual when the bay expands. Measure the
    # rebuilt component allocation with temporary headroom, then preserve the
    # original frame/cage/driver residual exactly.
    mass_probe = copy.deepcopy(source)
    mass_probe["mass"] = float(source["mass"]) + 10_000.0
    rebuilt_allocated_mass = vehicle_configuration_from_mapping(
        mass_probe).mass_properties()["allocated_component_mass_kg"]
    geometry_mass_correction = max(
        0.0, rebuilt_allocated_mass + original_residual_mass - float(source["mass"]))
    source["mass"] = float(source["mass"]) + geometry_mass_correction
    fitted = vehicle_configuration_from_mapping(source)
    wheel_placement = solve_vehicle_wheel_placement_mounts(fitted)
    return fitted, {
        "schema": "springtail-power-unit-chassis-fit-v1",
        "engine_envelope_m": [engine_length, engine_height, engine_width],
        "oil_pan_envelope_m": [pan_length, pan_depth, pan_width],
        "minimum_expanded_bay_m": [bay_length, bay_width],
        "chassis_half_length_m": float(chassis["half_length"]),
        "chassis_half_width_m": float(chassis["half_width"]),
        "wheelbase_m": 2 * float(wheels["wheelbase_half_length"]),
        "track_m": 2 * float(wheels["track_half_width"]),
        "wheel_placement_changed_by_pan_fit": False,
        "wheel_placement": wheel_placement,
        "added_frame_mass_kg": added_frame_mass,
        "added_attachment_mass_kg": added_attachment_mass,
        "geometry_mass_correction_kg": geometry_mass_correction,
        "total_mass_kg": float(source["mass"]),
        "authority": "shared-vehicle-configuration-and-mechanical-graph-rebuild",
    }


def _symbols() -> dict[str, sympy.Symbol]:
    names = [
        "position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z",
        "roll", "pitch", "yaw", "roll_velocity", "pitch_velocity", "yaw_velocity",
        "dt", "throttle", "brake", "drive_direction", "yaw_cos", "yaw_sin", "engine_enabled",
        *(f"external_hub_torque_{wheel}" for wheel in WHEEL_NAMES),
        *(f"hub_locker_engagement_{wheel}" for wheel in WHEEL_NAMES),
        "external_differential_wrench_torque_front", "external_differential_wrench_torque_rear",
        "external_differential_inertia_front", "external_differential_inertia_rear",
        "differential_wrench_shaft_omega_front", "differential_wrench_shaft_omega_rear",
        "power_unit_electric_mode", "traction_battery_charge_fraction",
        "traction_battery_target_charge_fraction", "traction_battery_capacity_j",
        "regenerative_charge_efficiency",
        "clutch_temperature_k", "clutch_health", "clutch_thermal_capacity_j_per_k",
        "clutch_cooling_w_per_k", "clutch_failure_temperature_k", "clutch_wear_energy_j",
        "clutch_wear", "clutch_glaze",
        *(f"hub_locker_wear_{wheel}" for wheel in WHEEL_NAMES),
        *(f"hub_locker_glaze_{wheel}" for wheel in WHEEL_NAMES),
        *(f"differential_locker_wear_{axle}" for axle in ("front", "rear", "center")),
        *(f"differential_locker_glaze_{axle}" for axle in ("front", "rear", "center")),
        "alternator_cvt_wear", "alternator_cvt_glaze",
        "direct_drive_bypass_command", "direct_drive_bypass_engagement",
        "direct_drive_bypass_tooth_health", "direct_drive_bypass_shift_rate_per_s",
        "direct_drive_bypass_maximum_sync_slip_rad_s",
        "external_engine_flywheel_inertia",
        "optional_fluid_coupling_engagement",
        "optional_fluid_coupling_engagement",
        "optional_fluid_coupling_coefficient_nm_per_rad_s2",
        "optional_fluid_coupling_maximum_torque_nm",
        "alternator_count", "alternator_max_power_w", "alternator_rotor_inertia_each",
        "alternator_efficiency", "alternator_cvt_ratio", "alternator_cvt_ratio_state",
        "alternator_cvt_efficiency", "alternator_cvt_ratio_response_hz",
        "alternator_electrical_demand_w",
        "accessory_motor_command", "accessory_motor_peak_power_w",
        "accessory_motor_peak_torque_nm", "accessory_motor_drive_efficiency",
        "accessory_motor_regeneration_efficiency",
        "accessory_battery_cube_capacity_j", "accessory_battery_cube_charge_fraction",
        "accessory_battery_cube_nominal_voltage_v",
        "accessory_battery_cube_maximum_discharge_current_a",
        "accessory_battery_cube_maximum_charge_current_a",
        "accessory_battery_cube_internal_resistance_ohm",
        "high_pressure_compressor_command",
        "high_pressure_compressor_displacement_m3_per_rev",
        "high_pressure_compressor_volumetric_efficiency",
        "high_pressure_compressor_isentropic_efficiency",
        "high_pressure_compressor_maximum_pressure_pa",
        "air_mix_reserve_volume_m3", "air_mix_reserve_gas_mass_kg",
        "air_mix_reserve_temperature_k", "air_mix_reserve_initial_temperature_k",
        "air_mix_reserve_inlet_pressure_pa", "air_mix_reserve_inlet_temperature_k",
        "air_mix_reserve_specific_gas_constant_j_per_kg_k",
        "air_mix_reserve_specific_heat_ratio", "air_mix_reserve_cooling_w_per_k",
        "air_mix_reserve_gas_demand_kg_s",
        "fuel_torque_scale", "ignition_torque_scale", "accessory_load_torque", "governor_angular_speed",
        "inverse_mass", "gravity", "suspension_travel",
        "spring_stiffness", "bump_stop_stiffness", "bump_stop_progressive_stiffness",
        "spring_model_selector", "spring_progressive_quadratic_n_per_m2",
        "spring_progressive_cubic_n_per_m3", "spring_primary_wire_diameter_m",
        "spring_primary_mean_coil_diameter_m", "spring_primary_active_turns",
        "spring_primary_shear_modulus_pa", "spring_secondary_wire_diameter_m",
        "spring_secondary_mean_coil_diameter_m", "spring_secondary_active_turns",
        "spring_secondary_shear_modulus_pa", "spring_secondary_engagement_compression_m",
        "spring_composite_coupling_efficiency", "bump_stop_start_fraction_of_travel",
        "bump_stop_damping", "pneumatic_compression_damping", "pneumatic_rebound_damping",
        "pneumatic_efficiency", "maximum_compression_speed", "active_damping_minimum_scale",
        "active_damping_maximum_scale", "active_damping_body_velocity_gain_s_per_m",
        "active_damping_rebound_release_gain_s_per_m", "wheelbase_half_length",
        "axle_group_offset_x", "track_half_width",
        "inverse_inertia_roll", "inverse_inertia_pitch",
        "inverse_inertia_yaw", "angular_damping", "air_density",
        "engine_displacement_m3", "brake_mean_effective_pressure",
        "engine_braking_mean_effective_pressure", "engine_angular_speed",
        "engine_idle_angular_speed", "engine_torque_peak_angular_speed",
        "engine_power_peak_angular_speed", "engine_redline_angular_speed",
        "clutch_stiffness", "clutch_maximum_torque", "combustion_efficiency",
        "clutch_efficiency", "forward_gear_ratio", "reverse_gear_ratio", "transfer_case_ratio",
        "final_drive_ratio", "driveline_efficiency", "engine_mass", "transmission_mass",
        "transfer_case_mass", "transfer_case_efficiency", "transfer_case_drag_torque",
        "engine_rotating_inertia", "engine_position_x", "engine_position_y",
        "engine_position_z", "transmission_position_x", "transmission_position_y",
        "transmission_position_z", "transfer_case_position_x", "transfer_case_position_y",
        "transfer_case_position_z", "crank_axis_x", "crank_axis_y", "crank_axis_z",
        "brake_torque", "wheel_inertia", "differential_brake_rotor_inertia",
        "wheel_radius",
        "front_differential_lock", "rear_differential_lock", "center_differential_lock",
        "front_differential_brake", "rear_differential_brake", "differential_brake_torque",
        "differential_lock_stiffness", "differential_lock_maximum_torque",
        "rolling_resistance_torque", "maximum_wheel_speed",
        "target_friction_utilization", "throttle_intervention_gain", "brake_intervention_gain",
        "slip_growth_gain", "slip_growth_reference_m_s2", "minimum_torque_fraction",
        "traction_control_enabled", "abs_enabled", "traction_control_authority", "abs_authority",
        "slip_sensor_frequency_hz", "slip_sensor_damping_ratio",
        "utilization_sensor_frequency_hz", "utilization_sensor_damping_ratio",
        "tire_longitudinal_deformation_frequency_hz", "tire_lateral_deformation_frequency_hz",
        "tire_sidewall_deformation_damping_ratio", "tire_maximum_sidewall_deformation",
        "total_force_x", "total_force_y", "total_force_z",
        "total_torque_x", "total_torque_y", "total_torque_z",
        "assembly_alpha_drivetrain",
        *(f"assembly_alpha_{wheel}" for wheel in WHEEL_NAMES),
        "contact_wrench_force_x", "contact_wrench_force_y", "contact_wrench_force_z",
        "contact_wrench_torque_x", "contact_wrench_torque_y", "contact_wrench_torque_z",
    ]
    for identity in DRAG_VECTOR_NAMES:
        names.extend((
            *(f"drag_{identity}_vector_{axis}" for axis in "xyz"),
            f"drag_{identity}_coefficient", f"drag_{identity}_reference_area",
        ))
    for wheel in WHEEL_NAMES:
        names.extend((f"compression_{wheel}", f"compression_velocity_{wheel}",
                      f"material_plastic_set_{wheel}", f"material_survival_{wheel}",
                      f"contact_normal_force_{wheel}", f"unsprung_mass_{wheel}",
                      f"wheel_support_{wheel}",
                      f"target_compression_{wheel}",
                      f"wheel_omega_{wheel}", f"wheel_angle_{wheel}",
                      f"longitudinal_force_{wheel}", f"tire_reaction_torque_{wheel}",
                      f"slip_longitudinal_{wheel}", f"slip_lateral_{wheel}", f"previous_slip_longitudinal_{wheel}",
                      f"slip_sensor_velocity_{wheel}",
                      f"measured_friction_utilization_{wheel}",
                      f"friction_utilization_{wheel}",
                      f"friction_utilization_sensor_velocity_{wheel}",
                      f"tire_deformation_longitudinal_{wheel}",
                      f"tire_deformation_velocity_longitudinal_{wheel}",
                      f"tire_deformation_lateral_{wheel}",
                      f"tire_deformation_velocity_lateral_{wheel}",
                      f"drive_fraction_{wheel}", f"linkage_motion_ratio_{wheel}",
                      f"brake_lock_{wheel}"))
    return {name: sympy.Symbol(name, real=True) for name in names}


def _c2_unit(value: sympy.Basic) -> sympy.Basic:
    """Compact C-infinity transition retained as one Tanh graph primitive."""
    return (1 + sympy.tanh(8 * (value - sympy.Rational(1, 2)))) / 2


def _c2_positive(value: sympy.Basic, width: sympy.Basic | float) -> sympy.Basic:
    return (value + sympy.sqrt(value ** 2 + sympy.sympify(width) ** 2)) / 2


def _c2_clamp(value: sympy.Basic, lower: sympy.Basic | float,
              upper: sympy.Basic | float, width: sympy.Basic | float) -> sympy.Basic:
    return (lower + _c2_positive(value - lower, width)
            - _c2_positive(value - upper, width))


def _smooth_abs(value: sympy.Basic, epsilon: str = "1e-4") -> sympy.Basic:
    return sympy.sqrt(value ** 2 + sympy.Float(epsilon) ** 2)


def _passive_radial_ringdown(
    compression: sympy.Basic,
    closing_velocity: sympy.Basic,
    stiffness: sympy.Basic,
    damping: sympy.Basic,
    effective_mass: sympy.Basic,
    dt: sympy.Basic,
) -> tuple[sympy.Basic, sympy.Basic, sympy.Basic, sympy.Basic]:
    """Integrate the local tire mode with three passive midpoint solves.

    Positive velocity closes the tire against the support plane. Implicit
    midpoint exactly conserves quadratic radial-mode energy when damping is
    zero and removes ``h*c*v_mid**2`` when damping is positive. The fixed
    series is part of the equation graph shared by C, Wasm, and WebGPU.
    """

    x = compression
    v = closing_velocity
    dissipated = sympy.Integer(0)
    h = dt / TIRE_RADIAL_RINGDOWN_STAGES
    half_h = h / 2
    for _stage in range(TIRE_RADIAL_RINGDOWN_STAGES):
        denominator = effective_mass + half_h * damping + half_h ** 2 * stiffness
        x_next = (
            (effective_mass + half_h * damping - half_h ** 2 * stiffness) * x
            + 2 * half_h * effective_mass * v
        ) / denominator
        v_next = (x_next - x) / half_h - v
        midpoint_velocity = (v + v_next) / 2
        dissipated += h * damping * midpoint_velocity ** 2
        x, v = x_next, v_next
    outward_impulse = effective_mass * (closing_velocity - v)
    return x, v, outward_impulse, dissipated


def _hard_positive(value: sympy.Basic) -> sympy.Basic:
    """Exact positive part, spelled as the single relational select ``Max``.

    ``(value + Abs(value)) / 2`` is the same function in real arithmetic but
    relies on ``value`` and ``Abs(value)`` cancelling exactly; SymPy's
    automatic evaluation distributes the halving and re-spells the operands,
    so a compiled schedule keeps a few-ULP residue (measured on the member
    material's identical gate: a phantom 2**-14 failure fraction).  ``Max``
    is supported by every backend and by the AbstractTensor printer.
    """
    return sympy.Max(value, 0)


def _hard_clamp(value: sympy.Basic, lower: sympy.Basic | float,
                upper: sympy.Basic | float) -> sympy.Basic:
    """Branch-free clamp kept compact for the AbstractTensor contact kernel."""
    lower = sympy.sympify(lower)
    upper = sympy.sympify(upper)
    return lower + _hard_positive(value - lower) - _hard_positive(value - upper)


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def _symbolic_vehicle_equations_loaded() -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    return symbolic_equations_cached(_symbolic_vehicle_equations_authored)


def symbolic_vehicle_equations() -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """The authored vehicle transition, built once per revision of this file.

    ``_symbolic_vehicle_equations_authored`` is the numerical authority.  Its
    sympy construction alone takes minutes of automatic evaluation, which
    every process used to pay before any cache could even be consulted; the
    persistent symbolic cache keys on this file's digest instead, so the
    expressions are built once and any edit here rebuilds them.
    """

    equations, symbols = _symbolic_vehicle_equations_loaded()
    return tuple(equations), dict(symbols)


def _symbolic_vehicle_equations_authored() -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """One transition with four independent spring lanes and one body reduction."""

    s = _symbols()
    dt = s["dt"]
    compressions: dict[str, sympy.Basic] = {}
    compression_velocities: dict[str, sympy.Basic] = {}
    forces: dict[str, sympy.Basic] = {}
    damping_scales: dict[str, sympy.Basic] = {}
    for wheel_index, wheel in enumerate(WHEEL_NAMES):
        raw_compression = (s[f"compression_{wheel}"]
                           - s[f"material_plastic_set_{wheel}"])
        compression = _hard_clamp(raw_compression, 0, s["suspension_travel"])
        bump_stop_start = (s["suspension_travel"]
                           * s["bump_stop_start_fraction_of_travel"])
        bump_stop_residual = _hard_positive(raw_compression - bump_stop_start)
        rate = _c2_clamp(s[f"compression_velocity_{wheel}"],
                         -s["maximum_compression_speed"], s["maximum_compression_speed"],
                         sympy.Float("0.08"))
        front_sign = 1 if wheel_index < 2 else -1
        side_sign = -1 if wheel_index % 2 == 0 else 1
        corner_body_velocity = (s["velocity_y"]
                                + s["pitch_velocity"] * (
                                    s["axle_group_offset_x"]
                                    + front_sign * s["wheelbase_half_length"])
                                - side_sign * s["roll_velocity"] * s["track_half_width"])
        raw_damping_scale = (1
                             + s["active_damping_body_velocity_gain_s_per_m"]
                             * _smooth_abs(corner_body_velocity)
                             - s["active_damping_rebound_release_gain_s_per_m"]
                             * _c2_positive(-rate, sympy.Float("0.08")))
        damping_scales[wheel] = _c2_clamp(raw_damping_scale, s["active_damping_minimum_scale"],
                                          s["active_damping_maximum_scale"], sympy.Float("0.025"))
        pneumatic = s["pneumatic_efficiency"] * (
            damping_scales[wheel] * s["pneumatic_compression_damping"]
            * _c2_positive(rate, sympy.Float("0.08"))
            - damping_scales[wheel] * s["pneumatic_rebound_damping"]
            * _c2_positive(-rate, sympy.Float("0.08"))
        )
        motion_ratio = s[f"linkage_motion_ratio_{wheel}"]
        linear_spring = s["spring_stiffness"] * compression
        progressive_spring = (
            linear_spring
            + s["spring_progressive_quadratic_n_per_m2"] * compression ** 2
            + s["spring_progressive_cubic_n_per_m3"] * compression ** 3
        )
        primary_helix_stiffness = (
            s["spring_primary_shear_modulus_pa"] * s["spring_primary_wire_diameter_m"] ** 4
            / (8 * s["spring_primary_mean_coil_diameter_m"] ** 3
               * s["spring_primary_active_turns"])
        )
        secondary_helix_stiffness = (
            s["spring_secondary_shear_modulus_pa"] * s["spring_secondary_wire_diameter_m"] ** 4
            / (8 * s["spring_secondary_mean_coil_diameter_m"] ** 3
               * s["spring_secondary_active_turns"])
        )
        composite_spring = (
            primary_helix_stiffness * compression
            + s["spring_composite_coupling_efficiency"] * secondary_helix_stiffness
            * _hard_positive(compression - s["spring_secondary_engagement_compression_m"])
        )
        selector = s["spring_model_selector"]
        selector_weights = tuple(
            1 - _hard_clamp(sympy.Abs(selector - index), 0, 1)
            for index in range(3)
        )
        spring_force = (
            selector_weights[0] * linear_spring
            + selector_weights[1] * progressive_spring
            + selector_weights[2] * composite_spring
        )
        bump_activation = bump_stop_residual / (bump_stop_residual + sympy.Float("1e-6"))
        bump_stop = (s["bump_stop_stiffness"] * bump_stop_residual
                     + s["bump_stop_progressive_stiffness"] * bump_stop_residual ** 2
                     + bump_activation * s["bump_stop_damping"]
                     * _c2_positive(rate, sympy.Float("0.08")))
        forces[wheel] = (s[f"assembly_alpha_{wheel}"]
                         * _hard_clamp(s[f"material_survival_{wheel}"], 0, 1)
                         * _c2_positive(
                             (spring_force * motion_ratio
                              + pneumatic + bump_stop) * motion_ratio,
                             sympy.Float("60")))

    # Wheel-normal terrain reactions stop at the unsprung corner nodes.  The
    # chassis receives their equal-and-opposite reaction only through the four
    # spring/damper graph edges. Tangential tyre and cage/shell contact wrenches
    # remain in contact_wrench_* and enter through their declared constraints.
    spring_total = sum(forces.values())
    net_force = (
        s["total_force_x"] + s["contact_wrench_force_x"],
        s["total_force_y"] + s["contact_wrench_force_y"] + spring_total,
        s["total_force_z"] + s["contact_wrench_force_z"],
    )
    net_torque = tuple(s[f"total_torque_{axis}"] + s[f"contact_wrench_torque_{axis}"]
                       for axis in "xyz")
    # A fixed-size runtime vector set lets each body provide its own drag
    # coefficients and projected areas without changing the compiled equation.
    # Directions are authored in chassis coordinates and rotate with yaw.
    drag_force = [sympy.Integer(0), sympy.Integer(0), sympy.Integer(0)]
    for identity in DRAG_VECTOR_NAMES:
        local_x = s[f"drag_{identity}_vector_x"]
        local_y = s[f"drag_{identity}_vector_y"]
        local_z = s[f"drag_{identity}_vector_z"]
        drag_direction = (
            s["yaw_cos"] * local_x - s["yaw_sin"] * local_z,
            local_y,
            s["yaw_sin"] * local_x + s["yaw_cos"] * local_z,
        )
        projected_speed = sum(
            s[f"velocity_{axis}"] * drag_direction[index]
            for index, axis in enumerate("xyz")
        )
        drag_magnitude = (-sympy.Rational(1, 2) * s["air_density"]
                          * s[f"drag_{identity}_coefficient"]
                          * s[f"drag_{identity}_reference_area"]
                          * _smooth_abs(projected_speed) * projected_speed)
        for index in range(3):
            drag_force[index] += drag_magnitude * drag_direction[index]
    velocity_x_next = s["velocity_x"] + dt * s["inverse_mass"] * (
        net_force[0] + drag_force[0])
    velocity_y_next = s["velocity_y"] + dt * (s["gravity"] + s["inverse_mass"] * (
        net_force[1] + drag_force[1]))
    velocity_z_next = s["velocity_z"] + dt * s["inverse_mass"] * (
        net_force[2] + drag_force[2])
    chassis_vertical_acceleration_without_gravity = s["inverse_mass"] * (
        net_force[1] + drag_force[1])
    for wheel in WHEEL_NAMES:
        unsprung_acceleration_relative_to_chassis = (
            (s[f"contact_normal_force_{wheel}"] - forces[wheel])
            / s[f"unsprung_mass_{wheel}"]
            - chassis_vertical_acceleration_without_gravity
        )
        candidate_velocity = s[f"assembly_alpha_{wheel}"] * _c2_clamp(
            s[f"compression_velocity_{wheel}"] + dt * unsprung_acceleration_relative_to_chassis,
            -s["maximum_compression_speed"], s["maximum_compression_speed"], sympy.Float("0.02"),
        )
        old_compression = _hard_clamp(s[f"compression_{wheel}"], 0, s["suspension_travel"])
        next_compression = _hard_clamp(
            old_compression + dt * candidate_velocity, 0, s["suspension_travel"])
        compressions[wheel] = next_compression
        # Deriving the stored velocity from the accepted displacement makes
        # both travel stops passive: residual velocity cannot accumulate behind
        # a clamp and explode when the corner leaves the stop.
        compression_velocities[wheel] = (next_compression - old_compression) / dt
    throttle_magnitude = _c2_clamp(_smooth_abs(s["throttle"]), 0, 1, sympy.Float("0.025"))
    indicated_torque = (s["brake_mean_effective_pressure"] * s["engine_displacement_m3"]
                        / (4 * sympy.pi))
    torque_peak_width = (s["engine_power_peak_angular_speed"]
                         - s["engine_torque_peak_angular_speed"])
    torque_curve = (sympy.Float("0.58") + sympy.Float("0.42") / (1 + (
        (s["engine_angular_speed"] - s["engine_torque_peak_angular_speed"])
        / torque_peak_width) ** 2))
    redline_rolloff = (1 - sympy.tanh(
        sympy.Float("8.0") * (
            s["engine_angular_speed"] / s["engine_redline_angular_speed"]
            - sympy.Float("0.94")
        )
    )) / 2
    governor_rolloff = (1 - sympy.tanh(sympy.Float("7.0") * (
        s["engine_angular_speed"] / s["governor_angular_speed"] - sympy.Float("0.965")))) / 2
    crank_threshold = s["engine_idle_angular_speed"] * sympy.Float("0.22")
    combustion_rotation_gate = (s["engine_angular_speed"] ** 2 / (
        s["engine_angular_speed"] ** 2 + crank_threshold ** 2))
    combustion_torque = (s["engine_enabled"] * combustion_rotation_gate
                         * s["fuel_torque_scale"] * s["ignition_torque_scale"]
                         * throttle_magnitude * indicated_torque
                         * s["combustion_efficiency"] * torque_curve * redline_rolloff * governor_rolloff)
    idle_error = _c2_clamp((s["engine_idle_angular_speed"] - s["engine_angular_speed"])
                           / (s["engine_idle_angular_speed"] * sympy.Float("0.18")),
                           0, 1, sympy.Float("0.035"))
    idle_governor_torque = (s["engine_enabled"] * combustion_rotation_gate * indicated_torque
                            * sympy.Float("0.38") * idle_error)
    engine_braking_torque = (s["engine_braking_mean_effective_pressure"]
                             * s["engine_displacement_m3"] / (4 * sympy.pi)
                             * (1 - throttle_magnitude) * s["engine_angular_speed"]
                             / sympy.sqrt(s["engine_angular_speed"] ** 2
                                          + s["engine_idle_angular_speed"] ** 2))
    direction = _c2_clamp(s["drive_direction"], -1, 1, sympy.Float("0.025"))
    # Direction is also the transmission coupling coordinate: +1 forward,
    # -1 reverse and exactly zero neutral.  Multiplying by direction prevents
    # the former forward/reverse blend from leaving a phantom ratio at zero.
    signed_gear = direction * (
        (1 + direction) / 2 * s["forward_gear_ratio"]
        + (1 - direction) / 2 * s["reverse_gear_ratio"])
    mean_wheel_speed = sum(_smooth_abs(s[f"wheel_omega_{wheel}"]) for wheel in WHEEL_NAMES) / 4
    mean_differential_speed = (
        _smooth_abs(s["differential_wrench_shaft_omega_front"])
        + _smooth_abs(s["differential_wrench_shaft_omega_rear"])) / 2
    coupled_crank_speed = (mean_differential_speed * _smooth_abs(signed_gear)
                           * s["transfer_case_ratio"] * s["final_drive_ratio"])
    # An automatic clutch is genuinely open at idle/near-rest, engages with
    # pedal demand, and only reconnects on overrun once road speed is high
    # enough for meaningful engine braking.  The previous nonzero lower clamp
    # manufactured reflected torque while stationary and acted like a missing
    # neutral gear.
    pedal_clutch = _c2_unit((throttle_magnitude - sympy.Float("0.035"))
                            / sympy.Float("0.20"))
    overrun_clutch = _c2_unit((mean_wheel_speed - sympy.Float("2.0"))
                              / sympy.Float("3.0"))
    clutch_engagement = _c2_clamp(
        1 - (1 - pedal_clutch) * (1 - overrun_clutch), 0, 1,
        sympy.Float("0.001"),
    ) * direction ** 2
    alternator_cvt_rate = 2 * sympy.pi * s["alternator_cvt_ratio_response_hz"]
    alternator_cvt_ratio_state_next = _hard_clamp(
        s["alternator_cvt_ratio_state"] + dt * alternator_cvt_rate
        * (s["alternator_cvt_ratio"] - s["alternator_cvt_ratio_state"]),
        sympy.Float("0.25"), sympy.Float("8.0"))
    alternator_speed_gate = (s["engine_angular_speed"] ** 2 / (
        s["engine_angular_speed"] ** 2 + sympy.Float("1600.0")))
    alternator_cvt_friction_state = ((1 - s["alternator_cvt_wear"])
        * (1 - sympy.Float("0.35") * s["alternator_cvt_glaze"]))
    alternator_generated_power = alternator_cvt_friction_state * alternator_speed_gate * _c2_clamp(
        s["alternator_electrical_demand_w"], 0, s["alternator_max_power_w"],
        sympy.Float("0.5"))
    alternator_reaction_torque = (
        alternator_generated_power
        / (_smooth_abs(s["engine_angular_speed"], epsilon="1.0")
           * s["alternator_efficiency"] * s["alternator_cvt_efficiency"]))
    alternator_cvt_loss_power = _c2_positive(
        alternator_reaction_torque * _smooth_abs(s["engine_angular_speed"], epsilon="1.0")
        - alternator_generated_power, sympy.Float("0.05"))
    alternator_cvt_wear_next = _hard_clamp(
        s["alternator_cvt_wear"] + dt * alternator_cvt_loss_power / sympy.Float("18000000.0"),
        0, 1)
    alternator_cvt_glaze_next = _hard_clamp(
        s["alternator_cvt_glaze"] + dt * _c2_positive(
            alternator_cvt_loss_power - sympy.Float("1200.0"), sympy.Float("0.05"))
        / sympy.Float("9000000.0"), 0, 1)
    accessory_shaft_speed = (_smooth_abs(s["engine_angular_speed"], epsilon="1.0")
                             * s["alternator_cvt_ratio_state"])
    motor_positive_request = (_c2_positive(s["accessory_motor_command"], sympy.Float("0.001"))
                              * s["accessory_motor_peak_torque_nm"])
    motor_negative_request = (_c2_positive(-s["accessory_motor_command"], sympy.Float("0.001"))
                              * s["accessory_motor_peak_torque_nm"])
    discharge_power_limit = (s["accessory_battery_cube_nominal_voltage_v"]
                             * s["accessory_battery_cube_maximum_discharge_current_a"])
    charge_power_limit = (s["accessory_battery_cube_nominal_voltage_v"]
                          * s["accessory_battery_cube_maximum_charge_current_a"])
    motor_positive_limit = sympy.Min(
        s["accessory_motor_peak_torque_nm"],
        s["accessory_motor_peak_power_w"] / accessory_shaft_speed,
        discharge_power_limit * s["accessory_motor_drive_efficiency"]
        / accessory_shaft_speed)
    motor_negative_limit = sympy.Min(
        s["accessory_motor_peak_torque_nm"],
        s["accessory_motor_peak_power_w"] / accessory_shaft_speed,
        charge_power_limit
        / (accessory_shaft_speed * s["accessory_motor_regeneration_efficiency"]))
    accessory_motor_shaft_torque = (
        _hard_clamp(motor_positive_request, 0, motor_positive_limit)
        * _c2_unit(s["accessory_battery_cube_charge_fraction"] / sympy.Float("0.02"))
        - _hard_clamp(motor_negative_request, 0, motor_negative_limit)
        * _c2_unit((1 - s["accessory_battery_cube_charge_fraction"])
                   / sympy.Float("0.02")))
    accessory_motor_engine_reaction_torque = (
        accessory_motor_shaft_torque * s["alternator_cvt_ratio_state"]
        * s["alternator_cvt_efficiency"])
    motor_positive_shaft_power = (_c2_positive(accessory_motor_shaft_torque,
                                                sympy.Float("0.001"))
                                  * accessory_shaft_speed)
    motor_negative_shaft_power = (_c2_positive(-accessory_motor_shaft_torque,
                                                sympy.Float("0.001"))
                                  * accessory_shaft_speed)
    accessory_motor_raw_bus_power = (
        motor_positive_shaft_power / s["accessory_motor_drive_efficiency"]
        - motor_negative_shaft_power * s["accessory_motor_regeneration_efficiency"])
    accessory_motor_bus_current = (
        accessory_motor_raw_bus_power / s["accessory_battery_cube_nominal_voltage_v"])
    accessory_motor_bus_power = (
        accessory_motor_raw_bus_power
        + accessory_motor_bus_current ** 2
        * s["accessory_battery_cube_internal_resistance_ohm"])
    accessory_battery_cube_charge_fraction_next = _hard_clamp(
        s["accessory_battery_cube_charge_fraction"]
        - dt * accessory_motor_bus_power / s["accessory_battery_cube_capacity_j"], 0, 1)
    gas_r = s["air_mix_reserve_specific_gas_constant_j_per_kg_k"]
    gas_gamma = s["air_mix_reserve_specific_heat_ratio"]
    gas_cv = gas_r / (gas_gamma - 1)
    gas_cp = gas_gamma * gas_cv
    reserve_pressure = (s["air_mix_reserve_gas_mass_kg"] * gas_r
                        * s["air_mix_reserve_temperature_k"]
                        / s["air_mix_reserve_volume_m3"])
    compressor_pressure_gate = _c2_unit(
        (s["high_pressure_compressor_maximum_pressure_pa"] - reserve_pressure)
        / sympy.Float("250000.0"))
    compressor_command = _hard_clamp(s["high_pressure_compressor_command"], 0, 1)
    inlet_density = (s["air_mix_reserve_inlet_pressure_pa"]
                     / (gas_r * s["air_mix_reserve_inlet_temperature_k"]))
    high_pressure_compressor_mass_flow = (
        compressor_command * compressor_pressure_gate
        * accessory_shaft_speed / (2 * sympy.pi)
        * s["high_pressure_compressor_displacement_m3_per_rev"]
        * s["high_pressure_compressor_volumetric_efficiency"] * inlet_density)
    compressor_pressure_ratio = (
        1 + _c2_positive(reserve_pressure - s["air_mix_reserve_inlet_pressure_pa"],
                         sympy.Float("100.0"))
        / s["air_mix_reserve_inlet_pressure_pa"])
    compressor_discharge_temperature = (
        s["air_mix_reserve_inlet_temperature_k"]
        * compressor_pressure_ratio ** ((gas_gamma - 1) / gas_gamma))
    compressor_power = (
        high_pressure_compressor_mass_flow * gas_cp
        * (compressor_discharge_temperature - s["air_mix_reserve_inlet_temperature_k"])
        / s["high_pressure_compressor_isentropic_efficiency"])
    compressor_shaft_reaction_torque = compressor_power / accessory_shaft_speed
    compressor_engine_reaction_torque = (
        compressor_shaft_reaction_torque * s["alternator_cvt_ratio_state"]
        / s["alternator_cvt_efficiency"])
    reserve_outflow = _hard_clamp(
        s["air_mix_reserve_gas_demand_kg_s"], 0,
        s["air_mix_reserve_gas_mass_kg"] / dt)
    air_mix_reserve_gas_mass_kg_next = _hard_clamp(
        s["air_mix_reserve_gas_mass_kg"]
        + dt * (high_pressure_compressor_mass_flow - reserve_outflow),
        sympy.Float("0.000001"), sympy.Float("1000000.0"))
    reserve_internal_energy_next = (
        s["air_mix_reserve_gas_mass_kg"] * gas_cv
        * s["air_mix_reserve_temperature_k"]
        + dt * (high_pressure_compressor_mass_flow * gas_cp
                * compressor_discharge_temperature
                - reserve_outflow * gas_cp * s["air_mix_reserve_temperature_k"]
                - s["air_mix_reserve_cooling_w_per_k"]
                * (s["air_mix_reserve_temperature_k"]
                   - s["air_mix_reserve_initial_temperature_k"])))
    air_mix_reserve_temperature_k_next = _hard_clamp(
        reserve_internal_energy_next / (air_mix_reserve_gas_mass_kg_next * gas_cv),
        sympy.Float("120.0"), sympy.Float("1200.0"))
    air_mix_reserve_pressure_pa = (
        air_mix_reserve_gas_mass_kg_next * gas_r * air_mix_reserve_temperature_k_next
        / s["air_mix_reserve_volume_m3"])
    engine_torque = s["assembly_alpha_drivetrain"] * (
        combustion_torque + idle_governor_torque - engine_braking_torque
        - s["accessory_load_torque"] - alternator_reaction_torque
        - compressor_engine_reaction_torque + accessory_motor_engine_reaction_torque)
    clutch_friction_state = (s["clutch_health"] * (1 - s["clutch_wear"])
                             * (1 - sympy.Float("0.45") * s["clutch_glaze"]))
    relative_drive_speed = s["engine_angular_speed"] - coupled_crank_speed
    direct_sync_gate = _c2_unit((s["direct_drive_bypass_maximum_sync_slip_rad_s"]
                                 - _smooth_abs(relative_drive_speed, epsilon="0.01"))
                                / sympy.Float("0.5"))
    direct_target = s["direct_drive_bypass_command"] * direct_sync_gate
    direct_drive_bypass_engagement_next = _hard_clamp(
        s["direct_drive_bypass_engagement"] + _hard_clamp(
            direct_target - s["direct_drive_bypass_engagement"],
            -dt * s["direct_drive_bypass_shift_rate_per_s"],
            dt * s["direct_drive_bypass_shift_rate_per_s"]), 0, 1)
    # A TCU bypass request first opens the friction path. The dog remains
    # gated by relative speed, so an unsynchronised command leaves both sides
    # isolated instead of forcing two torque paths to fight during the shift.
    raw_clutch_torque = ((1 - s["direct_drive_bypass_command"])
                         * clutch_engagement * clutch_friction_state
                         * s["clutch_maximum_torque"]
                         * sympy.tanh(s["clutch_stiffness"]
                                      * relative_drive_speed
                                      / s["clutch_maximum_torque"]))
    # The clutch torque is a bounded reaction to relative crank/input-shaft
    # speed.  It is allowed to exceed instantaneous combustion torque: that is
    # how a real engaged clutch decelerates and, under enough load, stalls the
    # engine.  Capping it below engine torque manufactured permanent slip and
    # made stalling impossible.
    clutch_torque = raw_clutch_torque
    direct_drive_bypass_torque = (
        s["direct_drive_bypass_engagement"] * s["direct_drive_bypass_tooth_health"]
        * s["clutch_maximum_torque"] * sympy.Float("1.8")
        * sympy.tanh(s["clutch_stiffness"] * relative_drive_speed
                     / (s["clutch_maximum_torque"] * sympy.Float("1.8"))))
    optional_fluid_coupling_torque = (
        s["optional_fluid_coupling_engagement"]
        * s["optional_fluid_coupling_maximum_torque_nm"]
        * sympy.tanh(s["optional_fluid_coupling_coefficient_nm_per_rad_s2"]
                     * relative_drive_speed * _smooth_abs(relative_drive_speed, epsilon="0.01")
                     / s["optional_fluid_coupling_maximum_torque_nm"]))
    transmitted_input_torque = (clutch_torque + direct_drive_bypass_torque
                                + optional_fluid_coupling_torque)
    direct_tooth_impact_power = _smooth_abs(
        direct_drive_bypass_torque * relative_drive_speed, epsilon="0.01")
    direct_drive_bypass_tooth_health_next = _hard_clamp(
        s["direct_drive_bypass_tooth_health"] - dt * _c2_positive(
            direct_tooth_impact_power - sympy.Float("60000.0"), sympy.Float("0.05"))
        / sympy.Float("45000000.0"), 0, 1)
    clutch_slip_power = _smooth_abs(
        clutch_torque * (s["engine_angular_speed"] - coupled_crank_speed),
        epsilon="0.01")
    clutch_cooling_power = s["clutch_cooling_w_per_k"] * _c2_positive(
        s["clutch_temperature_k"] - sympy.Float("300.0"), sympy.Float("0.05"))
    clutch_temperature_k_next = _hard_clamp(
        s["clutch_temperature_k"] + dt * (
            clutch_slip_power - clutch_cooling_power) / s["clutch_thermal_capacity_j_per_k"],
        sympy.Float("250.0"), sympy.Float("1400.0"))
    thermal_failure_rate = _c2_positive(
        s["clutch_temperature_k"] - s["clutch_failure_temperature_k"],
        sympy.Float("0.05")) / sympy.Float("50.0")
    clutch_wear_next = _hard_clamp(
        s["clutch_wear"] + dt * clutch_slip_power / s["clutch_wear_energy_j"], 0, 1)
    clutch_glaze_next = _hard_clamp(
        s["clutch_glaze"] + dt * _c2_positive(
            s["clutch_temperature_k"] - sympy.Float("470.0"), sympy.Float("0.05"))
        / sympy.Float("9000.0"), 0, 1)
    clutch_health_next = _hard_clamp(
        s["clutch_health"] - dt * thermal_failure_rate, 0, 1)
    transmission_output_torque = transmitted_input_torque * signed_gear * s["clutch_efficiency"]
    transfer_case_input_torque = transmission_output_torque * s["transfer_case_ratio"]
    transfer_case_direction = (transfer_case_input_torque
        / _smooth_abs(transfer_case_input_torque, epsilon="15.0"))
    transfer_case_output_torque = transfer_case_direction * _c2_positive(
        _smooth_abs(transfer_case_input_torque) - s["transfer_case_drag_torque"],
        sympy.Float("0.5"))
    driveline_torque = (transfer_case_output_torque * s["transfer_case_efficiency"]
                        * s["final_drive_ratio"] * s["driveline_efficiency"])
    transmitted_crank_load = transmitted_input_torque
    engine_acceleration_torque = engine_torque - transmitted_crank_load
    effective_engine_inertia = (s["engine_rotating_inertia"]
        + s["external_engine_flywheel_inertia"]
        + s["alternator_count"] * s["alternator_rotor_inertia_each"]
          * s["alternator_cvt_ratio_state"] ** 2)
    engine_angular_acceleration = engine_acceleration_torque / effective_engine_inertia
    engine_angular_speed_raw = (s["engine_angular_speed"] + dt * engine_angular_acceleration)
    engine_angular_speed_next = s["assembly_alpha_drivetrain"] * _c2_clamp(
        engine_angular_speed_raw, 0, s["engine_redline_angular_speed"] * sympy.Float("1.03"),
        sympy.Float("0.5"),
    )
    charge_request = _c2_unit((s["traction_battery_target_charge_fraction"]
                               - s["traction_battery_charge_fraction"])
                              / sympy.Float("0.01"))
    regenerative_charge_power = (
        s["power_unit_electric_mode"] * charge_request
        * s["regenerative_charge_efficiency"]
        * combustion_rotation_gate
        * _c2_positive(-transmitted_input_torque * s["engine_angular_speed"],
                       sympy.Float("0.5")))
    traction_battery_charge_fraction_next = _hard_clamp(
        s["traction_battery_charge_fraction"]
        + dt * regenerative_charge_power / s["traction_battery_capacity_j"],
        0, s["traction_battery_target_charge_fraction"])

    acceleration = (
        s["inverse_mass"] * (net_force[0] + drag_force[0]),
        s["gravity"] + s["inverse_mass"] * (net_force[1] + drag_force[1]),
        s["inverse_mass"] * (net_force[2] + drag_force[2]),
    )
    acceleration_chassis = (
        acceleration[0] * s["yaw_cos"] + acceleration[2] * s["yaw_sin"],
        acceleration[1] - s["gravity"],
        -acceleration[0] * s["yaw_sin"] + acceleration[2] * s["yaw_cos"],
    )
    engine_inertial_force_local = tuple(
        -s["assembly_alpha_drivetrain"] * s["engine_mass"] * value
        for value in acceleration_chassis)
    transmission_inertial_force_local = tuple(
        -s["assembly_alpha_drivetrain"] * s["transmission_mass"] * value
        for value in acceleration_chassis)
    transfer_case_inertial_force_local = tuple(
        -s["assembly_alpha_drivetrain"] * s["transfer_case_mass"] * value
        for value in acceleration_chassis)
    engine_position_local = tuple(s[f"engine_position_{axis}"] for axis in "xyz")
    transmission_position_local = tuple(s[f"transmission_position_{axis}"] for axis in "xyz")
    transfer_case_position_local = tuple(s[f"transfer_case_position_{axis}"] for axis in "xyz")
    crank_axis_local = tuple(s[f"crank_axis_{axis}"] for axis in "xyz")
    def cross(position: tuple[sympy.Basic, ...], force: tuple[sympy.Basic, ...]) -> tuple[sympy.Basic, ...]:
        return (position[1] * force[2] - position[2] * force[1],
                position[2] * force[0] - position[0] * force[2],
                position[0] * force[1] - position[1] * force[0])
    engine_mount_torque = cross(engine_position_local, engine_inertial_force_local)
    transmission_mount_torque = cross(transmission_position_local, transmission_inertial_force_local)
    transfer_case_mount_torque = cross(transfer_case_position_local, transfer_case_inertial_force_local)
    mount_torque = tuple(engine_mount_torque[index] + transmission_mount_torque[index]
                         + transfer_case_mount_torque[index]
                         for index in range(3))
    # Combustion torque passed through the clutch is internal to the complete
    # driveline graph.  Only net crank acceleration reacts at the engine mount;
    # applying full indicated torque here double-counts it against tire force.
    powertrain_reaction = tuple(mount_torque[index] - engine_acceleration_torque * crank_axis_local[index]
                                for index in range(3))
    # World torque is projected onto the yawed chassis axes. Roll is local forward,
    # pitch local right, and yaw world-up; this keeps the ABI useful before a full
    # quaternion lowering is needed.
    # The differential brake rotor owns an independently integrated shaft
    # coordinate below.  The wheel coordinate therefore contains only the
    # wheel/tire inertia: reflecting the rotor here as well would create a
    # second copy of its angular mass and corrupt both acceleration and gyro
    # accounting whenever the locking hub couples the two coordinates.
    effective_wheel_inertia = s["wheel_inertia"]
    wheel_angular_momentum = effective_wheel_inertia * sum(
        s[f"assembly_alpha_{wheel}"] * s[f"wheel_omega_{wheel}"]
        for wheel in WHEEL_NAMES
    )
    # Signed Ω×H reaction of the spinning wheel/tire assemblies. H is along
    # chassis-right; reverse wheel rotation therefore reverses the couple.
    wheel_gyroscopic_reaction = (
        -s["yaw_velocity"] * wheel_angular_momentum,
        s["roll_velocity"] * wheel_angular_momentum,
        sympy.Integer(0),
    )
    spring_roll_torque = sum(
        (-1 if index % 2 else 1) * s["track_half_width"] * forces[wheel]
        for index, wheel in enumerate(WHEEL_NAMES)
    )
    spring_pitch_torque = sum(
        (s["axle_group_offset_x"]
         + (1 if index < 2 else -1) * s["wheelbase_half_length"]) * forces[wheel]
        for index, wheel in enumerate(WHEEL_NAMES)
    )
    roll_torque = (net_torque[0] * s["yaw_cos"] + net_torque[2] * s["yaw_sin"]
                   + spring_roll_torque
                   + powertrain_reaction[0] + wheel_gyroscopic_reaction[0])
    pitch_torque = (-net_torque[0] * s["yaw_sin"] + net_torque[2] * s["yaw_cos"]
                    + spring_pitch_torque + powertrain_reaction[2])
    yaw_torque = net_torque[1] + powertrain_reaction[1] + wheel_gyroscopic_reaction[1]
    roll_velocity_next = (s["roll_velocity"] + dt * roll_torque * s["inverse_inertia_roll"]) / (1 + dt * s["angular_damping"])
    pitch_velocity_next = (s["pitch_velocity"] + dt * pitch_torque * s["inverse_inertia_pitch"]) / (1 + dt * s["angular_damping"])
    yaw_velocity_next = (s["yaw_velocity"] + dt * yaw_torque * s["inverse_inertia_yaw"]) / (1 + dt * s["angular_damping"])

    wheel_omegas: dict[str, sympy.Basic] = {}
    traction_scales: dict[str, sympy.Basic] = {}
    brake_scales: dict[str, sympy.Basic] = {}
    delivered_axle_torques: dict[str, sympy.Basic] = {}
    traction_intervention_torques: dict[str, sympy.Basic] = {}
    service_brake_torques: dict[str, sympy.Basic] = {}
    rolling_resistance_torques: dict[str, sympy.Basic] = {}
    tire_reaction_torques: dict[str, sympy.Basic] = {}
    drivetrain_chassis_reactions: dict[str, sympy.Basic] = {}
    hub_locker_wears: dict[str, sympy.Basic] = {}
    hub_locker_glazes: dict[str, sympy.Basic] = {}
    drive_torque = s["assembly_alpha_drivetrain"] * driveline_torque
    front_axle_speed = s["differential_wrench_shaft_omega_front"]
    rear_axle_speed = s["differential_wrench_shaft_omega_rear"]
    # The transfer-case coupling is the same bounded speed-sensitive clutch law
    # as an axle differential.  A continuous command gives three useful device
    # states without changing kernels: 0=open, a fractional capacity=LSD, and
    # 1=fully locked.  Positive torque flows from a faster rear shaft to the
    # slower front shaft; the equal negative term keeps it internal to the
    # complete driveline graph.
    center_locker_friction_state = ((1 - s["differential_locker_wear_center"])
        * (1 - sympy.Float("0.35") * s["differential_locker_glaze_center"]))
    center_coupling_torque = (
        center_locker_friction_state * s["center_differential_lock"]
        * s["differential_lock_maximum_torque"]
        * sympy.tanh(
            s["differential_lock_stiffness"]
            * (rear_axle_speed - front_axle_speed)
            / s["differential_lock_maximum_torque"]
        )
    )
    differential_locker_slip_power: dict[str, sympy.Basic] = {
        "front": sympy.Float("0.0"), "rear": sympy.Float("0.0"),
        "center": _smooth_abs(center_coupling_torque * (
            rear_axle_speed - front_axle_speed), epsilon="0.01"),
    }
    opposite_wheel = {
        "front_left": "front_right", "front_right": "front_left",
        "rear_left": "rear_right", "rear_right": "rear_left",
    }
    slip_sensor_omega = 2 * sympy.pi * s["slip_sensor_frequency_hz"]
    utilization_sensor_omega = 2 * sympy.pi * s["utilization_sensor_frequency_hz"]
    filtered_slips: dict[str, sympy.Basic] = {}
    slip_sensor_velocities: dict[str, sympy.Basic] = {}
    filtered_utilizations: dict[str, sympy.Basic] = {}
    utilization_sensor_velocities: dict[str, sympy.Basic] = {}
    tire_deformations_longitudinal: dict[str, sympy.Basic] = {}
    tire_deformation_velocities_longitudinal: dict[str, sympy.Basic] = {}
    tire_deformations_lateral: dict[str, sympy.Basic] = {}
    tire_deformation_velocities_lateral: dict[str, sympy.Basic] = {}
    for wheel in WHEEL_NAMES:
        assembly_alpha = s[f"assembly_alpha_{wheel}"]
        omega = s[f"wheel_omega_{wheel}"]
        for axis, frequency_name, slip_name, deformation_store, velocity_store in (
            ("longitudinal", "tire_longitudinal_deformation_frequency_hz", f"slip_longitudinal_{wheel}",
             tire_deformations_longitudinal, tire_deformation_velocities_longitudinal),
            ("lateral", "tire_lateral_deformation_frequency_hz", f"slip_lateral_{wheel}",
             tire_deformations_lateral, tire_deformation_velocities_lateral),
        ):
            mode_omega = 2 * sympy.pi * s[frequency_name]
            deformation = s[f"tire_deformation_{axis}_{wheel}"]
            deformation_velocity = s[f"tire_deformation_velocity_{axis}_{wheel}"]
            # The tread belt is a real state rather than a force lookup. Hub
            # slip first moves/deforms the sidewall mode; only that filtered
            # displacement and velocity reach the contact law next tick.
            target_deformation = s[slip_name] / mode_omega
            deformation_acceleration = (
                mode_omega ** 2 * (target_deformation - deformation)
                - 2 * s["tire_sidewall_deformation_damping_ratio"] * mode_omega
                * deformation_velocity
            )
            velocity_limit = s["tire_maximum_sidewall_deformation"] * mode_omega * 4
            next_velocity = _c2_clamp(deformation_velocity + dt * deformation_acceleration,
                                      -velocity_limit, velocity_limit, sympy.Float("0.02"))
            next_deformation = _c2_clamp(deformation + dt * next_velocity,
                                         -s["tire_maximum_sidewall_deformation"],
                                         s["tire_maximum_sidewall_deformation"], sympy.Float("0.001"))
            deformation_store[wheel] = next_deformation
            velocity_store[wheel] = next_velocity
        slip_sensor_acceleration = (
            slip_sensor_omega ** 2
            * (s[f"slip_longitudinal_{wheel}"] - s[f"previous_slip_longitudinal_{wheel}"])
            - 2 * s["slip_sensor_damping_ratio"] * slip_sensor_omega
            * s[f"slip_sensor_velocity_{wheel}"]
        )
        slip_sensor_velocity = (s[f"slip_sensor_velocity_{wheel}"]
                                + dt * slip_sensor_acceleration)
        filtered_slip = (s[f"previous_slip_longitudinal_{wheel}"]
                         + dt * slip_sensor_velocity)
        utilization_sensor_acceleration = (
            utilization_sensor_omega ** 2
            * (s[f"measured_friction_utilization_{wheel}"]
               - s[f"friction_utilization_{wheel}"])
            - 2 * s["utilization_sensor_damping_ratio"] * utilization_sensor_omega
            * s[f"friction_utilization_sensor_velocity_{wheel}"]
        )
        utilization_sensor_velocity = (
            s[f"friction_utilization_sensor_velocity_{wheel}"]
            + dt * utilization_sensor_acceleration
        )
        filtered_utilization = _c2_positive(
            s[f"friction_utilization_{wheel}"] + dt * utilization_sensor_velocity,
            sympy.Float("0.001"),
        )
        filtered_slips[wheel] = filtered_slip
        slip_sensor_velocities[wheel] = slip_sensor_velocity
        filtered_utilizations[wheel] = filtered_utilization
        utilization_sensor_velocities[wheel] = utilization_sensor_velocity
        utilization_excess = _c2_positive(
            filtered_utilization - s["target_friction_utilization"],
            sympy.Float("0.08"))
        # Differentiate the filtered state analytically.  A raw one-tick
        # difference aliases radial-probe handoff and terrain tessellation into
        # false ABS/TCS intervention.
        slip_magnitude_growth = (filtered_slip / _smooth_abs(filtered_slip, epsilon="0.02")
                                 * slip_sensor_velocity)
        slip_growth = _c2_positive(
            slip_magnitude_growth / s["slip_growth_reference_m_s2"],
            sympy.Float("0.08"),
        )
        traction_target = 1 / (
            1 + s["traction_control_enabled"] * s["traction_control_authority"] * (
                s["throttle_intervention_gain"] * utilization_excess
                + s["slip_growth_gain"] * slip_growth
            )
        )
        brake_target = 1 / (
            1 + s["abs_enabled"] * s["abs_authority"] * (
                s["brake_intervention_gain"] * utilization_excess
                + s["slip_growth_gain"] * slip_growth
            )
        )
        traction_scales[wheel] = _c2_clamp(traction_target, s["minimum_torque_fraction"], 1,
                                            sympy.Float("0.035"))
        brake_scales[wheel] = _c2_clamp(brake_target, s["minimum_torque_fraction"], 1,
                                        sympy.Float("0.035"))
        smooth_direction = omega / _smooth_abs(omega, epsilon="0.6")
        axle = "front" if wheel.startswith("front") else "rear"
        # The rotor-end rig wrench acts on the differential side of the
        # wheel-end locking hub. A disengaged clutch ring therefore isolates
        # the braked wheel while hub-side torque still acts on wheel inertia.
        # Engagement is an interlocked physical clutch coordinate supplied in
        # [0, 1]. Do not smooth its zero: FREE must be an exact open torque
        # path, not a tiny numerical clutch that accumulates work over time.
        hub_locker = s[f"hub_locker_engagement_{wheel}"]
        # A bounded torsional clutch is the actual locking-hub connection.
        # Differential drive and external rotor-end wrench torque first change
        # the differential shaft momentum below; only this reaction crosses
        # the clutch ring into the wheel.
        hub_friction_state = ((1 - s[f"hub_locker_wear_{wheel}"])
            * (1 - sympy.Float("0.35") * s[f"hub_locker_glaze_{wheel}"]))
        raw_axle_torque = (hub_locker * hub_friction_state
            * s["differential_lock_maximum_torque"]
            * sympy.tanh(s["differential_lock_stiffness"]
                         * (s[f"differential_wrench_shaft_omega_{axle}"] - omega)
                         / s["differential_lock_maximum_torque"]))
        hub_slip_power = _smooth_abs(raw_axle_torque * (
            s[f"differential_wrench_shaft_omega_{axle}"] - omega), epsilon="0.01")
        hub_locker_wears[wheel] = _hard_clamp(
            s[f"hub_locker_wear_{wheel}"] + dt * hub_slip_power / sympy.Float("12000000.0"),
            0, 1)
        hub_locker_glazes[wheel] = _hard_clamp(
            s[f"hub_locker_glaze_{wheel}"] + dt * _c2_positive(
                hub_slip_power - sympy.Float("18000.0"), sympy.Float("0.05"))
            / sympy.Float("12000000.0"), 0, 1)
        traction_intervention = raw_axle_torque * (1 - traction_scales[wheel])
        axle_torque = raw_axle_torque - traction_intervention
        delivered_axle_torques[wheel] = axle_torque
        traction_intervention_torques[wheel] = traction_intervention
        axle_lock = (s["front_differential_lock"] if wheel.startswith("front")
                     else s["rear_differential_lock"])
        axle_locker_friction_state = ((1 - s[f"differential_locker_wear_{axle}"])
            * (1 - sympy.Float("0.35") * s[f"differential_locker_glaze_{axle}"]))
        opposite_hub_locker = s[f"hub_locker_engagement_{opposite_wheel[wheel]}"]
        lock_torque = (hub_locker * opposite_hub_locker * axle_locker_friction_state * axle_lock
                       * s["differential_lock_maximum_torque"]
                       * sympy.tanh(s["differential_lock_stiffness"]
                                    * (s[f"wheel_omega_{opposite_wheel[wheel]}"] - omega)
                                    / s["differential_lock_maximum_torque"]))
        differential_locker_slip_power[axle] += _smooth_abs(
            lock_torque * (s[f"wheel_omega_{opposite_wheel[wheel]}"] - omega),
            epsilon="0.01") / 2
        # The contact appendage returns its actual equal/opposite rim moment.
        # A rigid-radius force surrogate is not valid for a deforming skin and
        # would duplicate or omit stored sidewall work during transients.
        tire_reaction = assembly_alpha * s[f"tire_reaction_torque_{wheel}"]
        tire_reaction_torques[wheel] = tire_reaction
        brake_command = _c2_clamp(
            s["brake"] * brake_scales[wheel] + s[f"brake_lock_{wheel}"],
            0, 1, sympy.Float("0.01"),
        )
        service_brake_torque = smooth_direction * s["brake_torque"] * brake_command
        rolling_resistance_torque = smooth_direction * s["rolling_resistance_torque"]
        service_brake_torques[wheel] = service_brake_torque
        rolling_resistance_torques[wheel] = rolling_resistance_torque
        resisting = service_brake_torque + rolling_resistance_torque
        drivetrain_chassis_reactions[wheel] = (
            -raw_axle_torque - lock_torque + traction_intervention
            + tire_reaction + service_brake_torque + rolling_resistance_torque)
        # The engine is already a separately integrated rotating body and the
        # finite clutch torque is the coupling between it and the wheel graph.
        # Reflecting engine inertia through ratio**2 here would only be valid
        # after imposing a locked-speed constraint; doing it while this clutch
        # is slipping double-counts engine inertia and becomes a crawler-gear
        # wheel-speed governor.  Wheel acceleration therefore uses the actual
        # wheel/tire inertia acted on by the delivered axle torque.
        # Laboratory hubs, tow starts and future accessories enter through a
        # real external torque wrench.  This is not a commanded wheel speed:
        # wheel inertia, tire contact, clutch reaction and every driveline loss
        # still determine the resulting angular acceleration.
        external_hub_torque = assembly_alpha * s[f"external_hub_torque_{wheel}"]
        free_omega = omega + dt * (axle_torque + lock_torque + external_hub_torque
                                   - tire_reaction - resisting) / effective_wheel_inertia
        raw_omega = free_omega
        # Wheel speed has no independent governor. Engine speed, selected
        # ratios, clutch torque, contact force and losses determine it. The
        # configured display maximum remains useful for HUD normalization but
        # must not saturate authoritative angular momentum.
        wheel_omegas[wheel] = assembly_alpha * raw_omega

    # These publications are the torque that actually reaches each axle after
    # front/rear proportioning, broken-halfshaft routing and traction control.
    # The previous pre-intervention values could misleadingly show torque on
    # an axle whose wheel paths had already been reduced to zero.
    front_differential_torque = sum(delivered_axle_torques[wheel]
                                    for wheel in WHEEL_NAMES if wheel.startswith("front"))
    rear_differential_torque = sum(delivered_axle_torques[wheel]
                                   for wheel in WHEEL_NAMES if wheel.startswith("rear"))
    differential_shaft_omegas: dict[str, sympy.Basic] = {}
    differential_brake_reactions: dict[str, sympy.Basic] = {}
    for axle, center_torque in (("front", center_coupling_torque),
                                ("rear", -center_coupling_torque)):
        axle_drive_fraction = sum(s[f"drive_fraction_{wheel}"] for wheel in WHEEL_NAMES
                                  if wheel.startswith(axle))
        shaft_input_torque = (drive_torque * axle_drive_fraction + center_torque
                              + s[f"external_differential_wrench_torque_{axle}"])
        shaft_output_torque = (front_differential_torque if axle == "front"
                               else rear_differential_torque)
        shaft_omega = s[f"differential_wrench_shaft_omega_{axle}"]
        shaft_inertia = (s["differential_brake_rotor_inertia"]
                         + s[f"external_differential_inertia_{axle}"])
        free_shaft_omega = shaft_omega + dt * (
            shaft_input_torque - shaft_output_torque) / shaft_inertia
        brake_command = s[f"{axle}_differential_brake"]
        # The differential brake rotor lives on this shaft, so its implicit
        # passive decay acts here rather than being duplicated at both wheels.
        brake_decay = (1 + dt * brake_command * s["differential_brake_torque"]
                       / (shaft_inertia * sympy.Float("0.05")))
        next_shaft_omega = free_shaft_omega / brake_decay
        differential_shaft_omegas[axle] = next_shaft_omega
        differential_brake_reactions[axle] = (
            (free_shaft_omega - next_shaft_omega)
            * shaft_inertia / dt)
    differential_locker_wears = {
        axle: _hard_clamp(s[f"differential_locker_wear_{axle}"]
            + dt * differential_locker_slip_power[axle] / sympy.Float("18000000.0"), 0, 1)
        for axle in ("front", "rear", "center")
    }
    differential_locker_glazes = {
        axle: _hard_clamp(s[f"differential_locker_glaze_{axle}"]
            + dt * _c2_positive(differential_locker_slip_power[axle]
                                - sympy.Float("22000.0"), sympy.Float("0.05"))
              / sympy.Float("18000000.0"), 0, 1)
        for axle in ("front", "rear", "center")
    }
    traction_control_dissipation_torque = sum(traction_intervention_torques.values())
    service_brake_reaction_torque = (sum(service_brake_torques.values())
                                     + sum(differential_brake_reactions.values()))
    rolling_resistance_reaction_torque = sum(rolling_resistance_torques.values())
    tire_contact_reaction_torque = sum(tire_reaction_torques.values())
    drivetrain_chassis_reaction_torque = (sum(drivetrain_chassis_reactions.values())
                                           + sum(differential_brake_reactions.values()))
    pitch_velocity_next = (s["pitch_velocity"] + dt * (
        pitch_torque + drivetrain_chassis_reaction_torque) * s["inverse_inertia_pitch"]
    ) / (1 + dt * s["angular_damping"])

    expressions: dict[str, sympy.Basic] = {
        "position_x_next": s["position_x"] + dt * velocity_x_next,
        "position_y_next": s["position_y"] + dt * velocity_y_next,
        "position_z_next": s["position_z"] + dt * velocity_z_next,
        "velocity_x_next": velocity_x_next,
        "velocity_y_next": velocity_y_next,
        "velocity_z_next": velocity_z_next,
        "roll_next": s["roll"] + dt * roll_velocity_next,
        "pitch_next": s["pitch"] + dt * pitch_velocity_next,
        "yaw_next": s["yaw"] + dt * yaw_velocity_next,
        "roll_velocity_next": roll_velocity_next,
        "pitch_velocity_next": pitch_velocity_next,
        "yaw_velocity_next": yaw_velocity_next,
        **{f"wheel_omega_{wheel}_next": wheel_omegas[wheel] for wheel in WHEEL_NAMES},
        **{f"hub_locker_wear_{wheel}_next": hub_locker_wears[wheel]
           for wheel in WHEEL_NAMES},
        **{f"hub_locker_glaze_{wheel}_next": hub_locker_glazes[wheel]
           for wheel in WHEEL_NAMES},
        **{f"differential_wrench_shaft_omega_{axle}_next": differential_shaft_omegas[axle]
           for axle in ("front", "rear")},
        **{f"differential_locker_wear_{axle}_next": differential_locker_wears[axle]
           for axle in ("front", "rear", "center")},
        **{f"differential_locker_glaze_{axle}_next": differential_locker_glazes[axle]
           for axle in ("front", "rear", "center")},
        **{f"wheel_angle_{wheel}_next": s[f"wheel_angle_{wheel}"]
           + dt * wheel_omegas[wheel] for wheel in WHEEL_NAMES},
        **{f"slip_longitudinal_{wheel}_next": filtered_slips[wheel]
           for wheel in WHEEL_NAMES},
        **{f"slip_sensor_velocity_{wheel}_next": slip_sensor_velocities[wheel]
           for wheel in WHEEL_NAMES},
        **{f"friction_utilization_{wheel}_next": filtered_utilizations[wheel]
           for wheel in WHEEL_NAMES},
        **{f"friction_utilization_sensor_velocity_{wheel}_next":
           utilization_sensor_velocities[wheel] for wheel in WHEEL_NAMES},
        **{f"tire_deformation_longitudinal_{wheel}_next": tire_deformations_longitudinal[wheel]
           for wheel in WHEEL_NAMES},
        **{f"tire_deformation_velocity_longitudinal_{wheel}_next":
           tire_deformation_velocities_longitudinal[wheel] for wheel in WHEEL_NAMES},
        **{f"tire_deformation_lateral_{wheel}_next": tire_deformations_lateral[wheel]
           for wheel in WHEEL_NAMES},
        **{f"tire_deformation_velocity_lateral_{wheel}_next":
           tire_deformation_velocities_lateral[wheel] for wheel in WHEEL_NAMES},
        **{f"traction_scale_{wheel}": traction_scales[wheel] for wheel in WHEEL_NAMES},
        **{f"brake_scale_{wheel}": brake_scales[wheel] for wheel in WHEEL_NAMES},
        **{f"compression_{wheel}_next": compressions[wheel] for wheel in WHEEL_NAMES},
        **{f"compression_velocity_{wheel}_next": compression_velocities[wheel]
           for wheel in WHEEL_NAMES},
        **{f"spring_force_{wheel}": forces[wheel] for wheel in WHEEL_NAMES},
        **{f"damper_scale_{wheel}": damping_scales[wheel] for wheel in WHEEL_NAMES},
        "engine_angular_speed_next": engine_angular_speed_next,
        "engine_rpm": engine_angular_speed_next * 60 / (2 * sympy.pi),
        "traction_battery_charge_fraction_next": traction_battery_charge_fraction_next,
        "regenerative_charge_power_w": regenerative_charge_power,
        "clutch_temperature_k_next": clutch_temperature_k_next,
        "clutch_health_next": clutch_health_next,
        "clutch_slip_power_w": clutch_slip_power,
        "clutch_wear_next": clutch_wear_next,
        "clutch_glaze_next": clutch_glaze_next,
        "alternator_cvt_ratio_state_next": alternator_cvt_ratio_state_next,
        "alternator_generated_power_w": alternator_generated_power,
        "alternator_reaction_torque_nm": alternator_reaction_torque,
        "alternator_cvt_wear_next": alternator_cvt_wear_next,
        "alternator_cvt_glaze_next": alternator_cvt_glaze_next,
        "accessory_battery_cube_charge_fraction_next": (
            accessory_battery_cube_charge_fraction_next),
        "accessory_motor_shaft_torque_nm": accessory_motor_shaft_torque,
        "accessory_motor_engine_reaction_torque_nm": (
            accessory_motor_engine_reaction_torque),
        "accessory_motor_bus_power_w": accessory_motor_bus_power,
        "air_mix_reserve_gas_mass_kg_next": air_mix_reserve_gas_mass_kg_next,
        "air_mix_reserve_temperature_k_next": air_mix_reserve_temperature_k_next,
        "air_mix_reserve_pressure_pa": air_mix_reserve_pressure_pa,
        "high_pressure_compressor_mass_flow_kg_s": high_pressure_compressor_mass_flow,
        "compressor_shaft_reaction_torque_nm": compressor_shaft_reaction_torque,
        "compressor_engine_reaction_torque_nm": compressor_engine_reaction_torque,
        "direct_drive_bypass_engagement_next": direct_drive_bypass_engagement_next,
        "direct_drive_bypass_tooth_health_next": direct_drive_bypass_tooth_health_next,
        "direct_drive_bypass_torque_nm": direct_drive_bypass_torque,
        "optional_fluid_coupling_torque_nm": optional_fluid_coupling_torque,
        "engine_torque": engine_torque,
        "clutch_torque": clutch_torque,
        "transmission_output_torque": transmission_output_torque,
        "driveline_torque": driveline_torque,
        "front_differential_torque": front_differential_torque,
        "rear_differential_torque": rear_differential_torque,
        "front_differential_wrench_torque": s["external_differential_wrench_torque_front"],
        "rear_differential_wrench_torque": s["external_differential_wrench_torque_rear"],
        "engine_acceleration_torque": engine_acceleration_torque,
        "engine_angular_acceleration": engine_angular_acceleration,
        **{f"powertrain_reaction_torque_{axis}": powertrain_reaction[index]
           for index, axis in enumerate("xyz")},
        **{f"engine_mount_torque_{axis}": mount_torque[index]
           for index, axis in enumerate("xyz")},
        **{f"wheel_gyroscopic_reaction_torque_{axis}": wheel_gyroscopic_reaction[index]
           for index, axis in enumerate("xyz")},
        "traction_control_dissipation_torque": traction_control_dissipation_torque,
        "service_brake_reaction_torque": service_brake_reaction_torque,
        "rolling_resistance_reaction_torque": rolling_resistance_reaction_torque,
        "tire_contact_reaction_torque": tire_contact_reaction_torque,
        "drivetrain_chassis_reaction_torque": drivetrain_chassis_reaction_torque,
    }
    equations = tuple(sympy.Eq(sympy.Symbol(name, real=True), expression, evaluate=False)
                      for name, expression in expressions.items())
    return equations, s


@lru_cache(maxsize=1)
def compile_symbolic_vehicle_physics() -> SymbolicEquationCompilation:
    publications = tuple(SymbolicPublication(name, f"world.vehicle.{name}")
                         for name in VEHICLE_STATE_OUTPUTS)
    return compile_symbolic_program(
        _symbolic_vehicle_equations_authored, name="abstract_ui_vehicle_step",
        publications=publications,
    )


@lru_cache(maxsize=1)
def compile_symbolic_vehicle_physics_wasm() -> SSAWasmArtifact:
    compiled = compile_symbolic_vehicle_physics()
    # C2 contact/control laws deliberately use the sqrt family.  License the
    # repository's documented bounded identity set for this deployable game
    # artifact without changing the process-wide compiler contract.
    artifact = emit_ssa_function_to_wasm(
        compiled.module, compiled.function.name, work_contract="deploy",
    )
    if not artifact.complete:
        reasons = "; ".join(item.reason for item in artifact.shortfalls)
        raise RuntimeError(f"vehicle physics does not lower to WASM: {reasons}")
    return artifact


@lru_cache(maxsize=1)
def compile_symbolic_vehicle_physics_gpu_ssa() -> SymbolicEquationCompilation:
    publications = tuple(SymbolicPublication(name, f"world.vehicle.{name}")
                         for name in VEHICLE_STATE_OUTPUTS)
    return compile_symbolic_program(
        _symbolic_vehicle_equations_authored, name="abstract_ui_vehicle_step_gpu",
        publications=publications, dtype="float32",
    )


@lru_cache(maxsize=1)
def compile_symbolic_vehicle_physics_webgpu() -> WGSLModule:
    """Compile the complete authored vehicle transition as one GPU kernel."""

    compiled = compile_symbolic_vehicle_physics_gpu_ssa()
    returned = next(
        instruction.args
        for block in compiled.function.blocks.values()
        for instruction in block.instrs
        if instruction.op.lower() in {"ret", "return"}
    )
    artifact = emit_webgpu_module(
        compiled.module, name=compiled.function.name,
        outputs={compiled.function.name: returned}, count=1, packed_outputs=True,
    )
    if not artifact.complete:
        raise RuntimeError("vehicle physics does not lower to WebGPU: " + "; ".join(
            item.format() for item in artifact.shortfalls
        ))
    return artifact


@lru_cache(maxsize=1)
def compile_symbolic_vehicle_physics_c() -> CFunctionArtifact:
    """Emit the complete authored vehicle transition through repository C."""

    compiled = compile_symbolic_vehicle_physics()
    artifact = emit_ssa_function_to_c(
        compiled.module, compiled.function.name,
        entry_name="abstract_ui_vehicle_step",
    )
    if not artifact.complete:
        reasons = "; ".join(item.reason for item in artifact.shortfalls)
        raise RuntimeError(f"complete vehicle transition does not lower to C: {reasons}")
    return artifact


SUSPENSION_RIG_OUTPUTS = (
    *(f"compression_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"compression_velocity_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"spring_force_{wheel}" for wheel in WHEEL_NAMES),
    "velocity_y_next",
)


@lru_cache(maxsize=1)
def compile_symbolic_vehicle_suspension_rig_wasm() -> SSAWasmArtifact:
    """Compile the production corner reduction without the page or drivetrain.

    This is the fast stability/dyno loop.  It selects the exact suspension and
    chassis-vertical publications from ``symbolic_vehicle_equations`` rather
    than maintaining a friendly-looking duplicate oscillator.
    """

    equations, _ = symbolic_vehicle_equations()
    by_name = {str(equation.lhs): equation for equation in equations}
    selected = tuple(by_name[name] for name in SUSPENSION_RIG_OUTPUTS)
    publications = tuple(SymbolicPublication(name, f"world.vehicle.rig.{name}")
                         for name in SUSPENSION_RIG_OUTPUTS)
    compiled = compile_sympy_equations(
        selected, name="abstract_ui_vehicle_suspension_rig",
        publications=publications, dtype="float64",
    )
    artifact = emit_ssa_function_to_wasm(
        compiled.module, compiled.function.name, work_contract="deploy",
    )
    if not artifact.complete:
        reasons = "; ".join(item.reason for item in artifact.shortfalls)
        raise RuntimeError(f"vehicle suspension rig does not lower to WASM: {reasons}")
    return artifact


@lru_cache(maxsize=1)
def compile_symbolic_vehicle_suspension_rig_c() -> CFunctionArtifact:
    """Emit the production suspension reduction as direct native C.

    This selects the same authored equations as the browser fallback, but the
    artifact is repository-SSA -> C and therefore has no Wasm or web runtime.
    """

    equations, _ = symbolic_vehicle_equations()
    by_name = {str(equation.lhs): equation for equation in equations}
    selected = tuple(by_name[name] for name in SUSPENSION_RIG_OUTPUTS)
    publications = tuple(SymbolicPublication(name, f"world.vehicle.rig.{name}")
                         for name in SUSPENSION_RIG_OUTPUTS)
    compiled = compile_sympy_equations(
        selected, name="abstract_ui_vehicle_suspension_rig",
        publications=publications, dtype="float64",
    )
    artifact = emit_ssa_function_to_c(
        compiled.module, compiled.function.name,
        entry_name="abstract_ui_vehicle_suspension_rig",
    )
    if not artifact.complete:
        reasons = "; ".join(item.reason for item in artifact.shortfalls)
        raise RuntimeError(f"vehicle suspension rig does not lower to C: {reasons}")
    return artifact


@lru_cache(maxsize=1)
def compile_symbolic_vehicle_physics_webgpu_stages() -> tuple[dict[str, Any], ...]:
    """Lower one symbolic vehicle graph into independently compilable kernels.

    Every stage reads the same resident input buffer and writes a disjoint,
    256-byte-aligned view of one output GPUBuffer.  The split is therefore a
    compiler/scheduling boundary, not a host memory handoff.
    """

    compiled = compile_symbolic_vehicle_physics_gpu_ssa()
    returned = next(
        instruction.args
        for block in compiled.function.blocks.values()
        for instruction in block.instrs
        if instruction.op.lower() in {"ret", "return"}
    )
    values = dict(zip(VEHICLE_STATE_OUTPUTS, returned, strict=True))
    groups = {
        "chassis-transition": tuple(name for name in VEHICLE_STATE_OUTPUTS if
                                    name.startswith(("position_", "velocity_", "roll_", "pitch_", "yaw_"))
                                    or name in {"roll_next", "pitch_next", "yaw_next"}),
        "tire-suspension-control": tuple(name for name in VEHICLE_STATE_OUTPUTS if
                                         name.startswith(("wheel_omega_", "wheel_angle_", "slip_", "friction_", "traction_",
                                                          "brake_", "compression_", "spring_", "damper_", "tire_"))),
    }
    assigned = set().union(*groups.values())
    groups["powertrain-reactions"] = tuple(name for name in VEHICLE_STATE_OUTPUTS if name not in assigned)
    stages: list[dict[str, Any]] = []
    # Storage-buffer binding offsets must satisfy WebGPU's conservative
    # 256-byte alignment. Each stage gets a view into this one allocation.
    output_cursor = 0
    for identity, names in groups.items():
        output_cursor = ((output_cursor + 63) // 64) * 64
        artifact = emit_webgpu_module(
            compiled.module, name=compiled.function.name,
            outputs={compiled.function.name: tuple(values[name] for name in names)},
            count=1, packed_outputs=True,
        )
        if not artifact.complete:
            raise RuntimeError(f"vehicle physics stage {identity} does not lower to WebGPU: " + "; ".join(
                item.format() for item in artifact.shortfalls
            ))
        stages.append({
            "identity": identity, "outputs": names, "output_offset_floats": output_cursor,
            "kernel": artifact,
        })
        output_cursor += len(names)
    return tuple(stages)


@lru_cache(maxsize=1)
def compile_default_specialized_vehicle_physics_gpu_ssa() -> SymbolicEquationCompilation:
    """Compile the default vehicle with stable configuration constants folded."""

    equations, symbols = symbolic_vehicle_equations()
    defaults = load_default_car_configuration().parameter_defaults()
    always_live = {
        "engine_angular_speed", "drive_direction", "forward_gear_ratio", "reverse_gear_ratio",
        *(f"external_hub_torque_{wheel}" for wheel in WHEEL_NAMES),
        *(f"hub_locker_engagement_{wheel}" for wheel in WHEEL_NAMES),
        "external_differential_wrench_torque_front", "external_differential_wrench_torque_rear",
        "external_differential_inertia_front", "external_differential_inertia_rear",
        "differential_wrench_shaft_omega_front", "differential_wrench_shaft_omega_rear",
        "power_unit_electric_mode", "traction_battery_charge_fraction",
        "traction_battery_target_charge_fraction", "traction_battery_capacity_j",
        "regenerative_charge_efficiency",
        "clutch_temperature_k", "clutch_health",
        "clutch_wear", "clutch_glaze", "alternator_cvt_wear", "alternator_cvt_glaze",
        "direct_drive_bypass_command", "direct_drive_bypass_engagement",
        "direct_drive_bypass_tooth_health",
        "external_engine_flywheel_inertia",
        "optional_fluid_coupling_engagement",
        *(f"hub_locker_wear_{wheel}" for wheel in WHEEL_NAMES),
        *(f"hub_locker_glaze_{wheel}" for wheel in WHEEL_NAMES),
        *(f"differential_locker_wear_{axle}" for axle in ("front", "rear", "center")),
        *(f"differential_locker_glaze_{axle}" for axle in ("front", "rear", "center")),
        "alternator_cvt_ratio", "alternator_cvt_ratio_state", "alternator_electrical_demand_w",
        "accessory_motor_command", "accessory_battery_cube_charge_fraction",
        "high_pressure_compressor_command", "air_mix_reserve_gas_mass_kg",
        "air_mix_reserve_temperature_k", "air_mix_reserve_gas_demand_kg_s",
        *(f"linkage_motion_ratio_{wheel}" for wheel in WHEEL_NAMES),
    }
    substitutions = {
        symbols[name]: sympy.Float(str(value))
        for name, value in defaults.items()
        if name in symbols and name not in always_live
    }
    specialized = tuple(sympy.Eq(equation.lhs, equation.rhs.xreplace(substitutions), evaluate=False)
                        for equation in equations)
    publications = tuple(SymbolicPublication(name, f"world.vehicle.{name}")
                         for name in VEHICLE_STATE_OUTPUTS)
    return compile_sympy_equations(
        specialized, name="abstract_ui_vehicle_step_gpu_default_fixed",
        publications=publications, dtype="float32",
    )


@lru_cache(maxsize=1)
def compile_default_specialized_vehicle_physics_webgpu_stages() -> tuple[dict[str, Any], ...]:
    compiled = compile_default_specialized_vehicle_physics_gpu_ssa()
    returned = next(
        instruction.args
        for block in compiled.function.blocks.values()
        for instruction in block.instrs
        if instruction.op.lower() in {"ret", "return"}
    )
    values = dict(zip(VEHICLE_STATE_OUTPUTS, returned, strict=True))
    groups = {
        "chassis-transition": tuple(name for name in VEHICLE_STATE_OUTPUTS if
                                    name.startswith(("position_", "velocity_", "roll_", "pitch_", "yaw_"))
                                    or name in {"roll_next", "pitch_next", "yaw_next"}),
        "tire-suspension-control": tuple(name for name in VEHICLE_STATE_OUTPUTS if
                                         name.startswith(("wheel_omega_", "wheel_angle_", "slip_", "friction_", "traction_",
                                                          "brake_", "compression_", "spring_", "damper_", "tire_"))),
    }
    assigned = set().union(*groups.values())
    groups["powertrain-reactions"] = tuple(name for name in VEHICLE_STATE_OUTPUTS if name not in assigned)
    stages: list[dict[str, Any]] = []
    output_cursor = 0
    for identity, names in groups.items():
        output_cursor = ((output_cursor + 63) // 64) * 64
        artifact = emit_webgpu_module(
            compiled.module, name=compiled.function.name,
            outputs={compiled.function.name: tuple(values[name] for name in names)},
            count=1, packed_outputs=True,
        )
        if not artifact.complete:
            raise RuntimeError(f"default-fixed vehicle stage {identity} does not lower to WebGPU: " + "; ".join(
                item.format() for item in artifact.shortfalls
            ))
        stages.append({"identity": identity, "outputs": names,
                       "output_offset_floats": output_cursor, "kernel": artifact})
        output_cursor += len(names)
    return tuple(stages)


@lru_cache(maxsize=32)
def engine_specialized_symbolic_equations(
    profile_identity: str, parameter_items: tuple[tuple[str, float], ...],
) -> tuple[tuple[sympy.Equality, ...], dict[str, Any]]:
    """Substitute and attempt a bounded, equivalence-preserving SymPy reduction."""

    equations, symbols = symbolic_vehicle_equations()
    defaults = dict(parameter_items)
    always_live = {
        "engine_angular_speed", "drive_direction", "forward_gear_ratio", "reverse_gear_ratio",
        "clutch_stiffness", "clutch_maximum_torque", "clutch_efficiency",
        *(f"external_hub_torque_{wheel}" for wheel in WHEEL_NAMES),
        *(f"hub_locker_engagement_{wheel}" for wheel in WHEEL_NAMES),
        "external_differential_wrench_torque_front", "external_differential_wrench_torque_rear",
        "external_differential_inertia_front", "external_differential_inertia_rear",
        "differential_wrench_shaft_omega_front", "differential_wrench_shaft_omega_rear",
        "power_unit_electric_mode", "traction_battery_charge_fraction",
        "traction_battery_target_charge_fraction", "traction_battery_capacity_j",
        "regenerative_charge_efficiency",
        "clutch_temperature_k", "clutch_health",
        "clutch_wear", "clutch_glaze", "alternator_cvt_wear", "alternator_cvt_glaze",
        "direct_drive_bypass_command", "direct_drive_bypass_engagement",
        "direct_drive_bypass_tooth_health",
        "external_engine_flywheel_inertia",
        *(f"hub_locker_wear_{wheel}" for wheel in WHEEL_NAMES),
        *(f"hub_locker_glaze_{wheel}" for wheel in WHEEL_NAMES),
        *(f"differential_locker_wear_{axle}" for axle in ("front", "rear", "center")),
        *(f"differential_locker_glaze_{axle}" for axle in ("front", "rear", "center")),
        "alternator_cvt_ratio", "alternator_cvt_ratio_state", "alternator_electrical_demand_w",
        "accessory_motor_command", "accessory_battery_cube_charge_fraction",
        "high_pressure_compressor_command", "air_mix_reserve_gas_mass_kg",
        "air_mix_reserve_temperature_k", "air_mix_reserve_gas_demand_kg_s",
        *(f"linkage_motion_ratio_{wheel}" for wheel in WHEEL_NAMES),
    }
    substitutions = {symbols[name]: sympy.Float(str(value)) for name, value in defaults.items()
                     if name in symbols and name not in always_live}
    substituted = tuple(sympy.Eq(equation.lhs, equation.rhs.xreplace(substitutions), evaluate=False)
                        for equation in equations)
    # Expressions such as the pitch and whole-drivetrain reaction publications
    # intentionally inline tens of thousands of operations.  Asking generic
    # factor/CSE passes to rediscover their authored sharing is dramatically
    # slower than lowering them.  Keep the reduction attempt explicitly bounded.
    reduction_operation_budget = 120
    reduced: list[sympy.Equality] = []
    authored_ops = 0
    reduced_ops = 0
    accepted = 0
    skipped = 0
    cse_candidates: list[sympy.Basic] = []
    for equation in substituted:
        authored_count = int(sympy.count_ops(equation.rhs))
        if authored_count <= reduction_operation_budget:
            candidate = sympy.factor_terms(equation.rhs)
            candidate_count = int(sympy.count_ops(candidate))
            use_candidate = candidate_count <= authored_count
            cse_candidates.append(candidate if use_candidate else equation.rhs)
        elif identity.startswith(("body_shell.", "body_pin.", "turret.", "cage.")):
            item["longitudinal_parameterization"] = {
                "authority": "body-packaging-cab-bed-stations",
                "resolved_x_m": float(item["reference_position"][0]),
            }
        else:
            candidate = equation.rhs
            candidate_count = authored_count
            use_candidate = False
            skipped += 1
        reduced.append(sympy.Eq(equation.lhs, candidate if use_candidate else equation.rhs, evaluate=False))
        authored_ops += authored_count
        reduced_ops += candidate_count if use_candidate else authored_count
        accepted += int(use_candidate and candidate != equation.rhs)
    replacements, _cse_outputs = sympy.cse(cse_candidates, order="canonical")
    audit = {"attempted": True, "method": "bounded-factor-terms-plus-bounded-cross-output-cse-audit",
             "authored_operation_count": authored_ops, "accepted_operation_count": reduced_ops,
             "accepted_publication_rewrites": accepted, "cross_output_common_subexpressions": len(replacements),
             "operation_budget_per_publication": reduction_operation_budget,
             "budget_skipped_publications": skipped,
             "fallback": "retain-substituted-expression-when-budget-is-exceeded-or-operation-count-does-not-improve"}
    return tuple(reduced), audit


@lru_cache(maxsize=32)
def engine_playable_linear_equations(
    profile_identity: str, parameter_items: tuple[tuple[str, float], ...],
) -> tuple[tuple[sympy.Equality, ...], dict[str, Any]]:
    nonlinear, reduction_audit = engine_specialized_symbolic_equations(profile_identity, parameter_items)
    linear_system_operation_budget = 240
    eligible_prefixes = ("engine_", "clutch_", "transmission_", "driveline_", "powertrain_reaction_",
                         "engine_mount_", "rolling_resistance_reaction_")
    linear_system = tuple(equation for equation in nonlinear
                          if str(equation.lhs).startswith(eligible_prefixes)
                          and int(sympy.count_ops(equation.rhs)) <= linear_system_operation_budget)
    outputs = tuple(equation.lhs for equation in linear_system)
    exact_solved = False
    exact_reason = "no-operation-count-improvement"
    try:
        matrix, vector = sympy.linear_eq_to_matrix(
            [equation.lhs - equation.rhs for equation in linear_system], outputs)
        solution = tuple(next(iter(sympy.linsolve((matrix, vector), outputs))))
        original_ops = sum(int(sympy.count_ops(equation.rhs)) for equation in linear_system)
        solved_ops = sum(int(sympy.count_ops(expression)) for expression in solution)
        exact_solved = solved_ops < original_ops
        if exact_solved:
            solved_by_output = dict(zip(outputs, solution, strict=True))
            return tuple(sympy.Eq(equation.lhs, solved_by_output.get(equation.lhs, equation.rhs), evaluate=False)
                         for equation in nonlinear), {
                **reduction_audit, "linear_mode": "exact-linsolve", "exact_solver_succeeded": True,
                "linear_system_publications": len(linear_system),
                "accepted_operation_count": solved_ops,
            }
    except (ValueError, TypeError, NotImplementedError, StopIteration) as error:
        exact_reason = type(error).__name__
    symbol_by_name = {str(symbol): symbol for equation in nonlinear for symbol in equation.rhs.free_symbols}
    defaults = {**load_default_car_configuration().parameter_defaults(), **dict(parameter_items)}
    defaults.update({"dt": 1 / 120, "throttle": .35, "brake": 0.0, "drive_direction": 1.0,
                     "transfer_case_ratio": 1.0})
    selected_names = ("engine_angular_speed", "throttle", "brake", "forward_gear_ratio",
                      "transfer_case_ratio", "drive_direction",
                      *(f"wheel_omega_{wheel}" for wheel in WHEEL_NAMES))
    selected = tuple(symbol_by_name[name] for name in selected_names if name in symbol_by_name)
    operating = {symbol: sympy.Float(str(defaults.get(str(symbol), 0.0))) for symbol in selected}
    approximated: list[sympy.Equality] = []
    accepted = 0
    for equation in nonlinear:
        name = str(equation.lhs)
        eligible = (name.startswith(("engine_", "clutch_", "transmission_", "driveline_",
                                    "front_differential_", "rear_differential_", "wheel_omega_"))
                    or name.endswith("_reaction_torque"))
        operation_count = int(sympy.count_ops(equation.rhs))
        if not eligible or operation_count > linear_system_operation_budget:
            approximated.append(equation)
            continue
        base = equation.rhs.xreplace(operating)
        candidate = base + sum(
            sympy.diff(equation.rhs, symbol).xreplace(operating) * (symbol - operating[symbol])
            for symbol in selected if symbol in equation.rhs.free_symbols
        )
        candidate = sympy.factor_terms(candidate)
        if candidate.has(sympy.Derivative):
            approximated.append(equation)
        else:
            approximated.append(sympy.Eq(equation.lhs, candidate, evaluate=False));accepted += 1
    return tuple(approximated), {
        **reduction_audit, "linear_mode": "first-order-engine-driveline-operating-point",
        "exact_solver_succeeded": exact_solved, "exact_solver_fallback_reason": exact_reason,
        "linear_system_publications": len(linear_system),
        "linear_system_operation_budget": linear_system_operation_budget,
        "approximated_publications": accepted, "contact_and_chassis_fidelity_retained": True,
        "operating_point": {str(symbol): float(value) for symbol, value in operating.items()},
    }


@lru_cache(maxsize=32)
def compile_engine_specialized_vehicle_physics_gpu_ssa(
    profile_identity: str, parameter_items: tuple[tuple[str, float], ...], equation_mode: str = "symbolic-fidelity",
) -> SymbolicEquationCompilation:
    """Cook one engine profile into its own solved vehicle equation path."""

    specialized, _audit = (engine_playable_linear_equations(profile_identity, parameter_items)
                           if equation_mode == "linear-playable" else
                           engine_specialized_symbolic_equations(profile_identity, parameter_items))
    publications = tuple(SymbolicPublication(name, f"world.vehicle.{name}")
                         for name in VEHICLE_STATE_OUTPUTS)
    safe_identity = re.sub(r"[^A-Za-z0-9_]", "_", f"{profile_identity}_{equation_mode}")
    return compile_sympy_equations(
        specialized, name=f"abstract_ui_vehicle_step_gpu_engine_{safe_identity}",
        publications=publications, dtype="float32",
    )


@lru_cache(maxsize=32)
def compile_engine_specialized_vehicle_physics_webgpu_stages(
    profile_identity: str, parameter_items: tuple[tuple[str, float], ...], equation_mode: str = "symbolic-fidelity",
) -> tuple[dict[str, Any], ...]:
    compiled = compile_engine_specialized_vehicle_physics_gpu_ssa(profile_identity, parameter_items, equation_mode)
    returned = next(instruction.args for block in compiled.function.blocks.values()
                    for instruction in block.instrs if instruction.op.lower() in {"ret", "return"})
    values = dict(zip(VEHICLE_STATE_OUTPUTS, returned, strict=True))
    groups = {
        "chassis-transition": tuple(name for name in VEHICLE_STATE_OUTPUTS if
                                    name.startswith(("position_", "velocity_", "roll_", "pitch_", "yaw_"))
                                    or name in {"roll_next", "pitch_next", "yaw_next"}),
        "tire-suspension-control": tuple(name for name in VEHICLE_STATE_OUTPUTS if
                                         name.startswith(("wheel_omega_", "wheel_angle_", "slip_", "friction_", "traction_",
                                                          "brake_", "compression_", "spring_", "damper_", "tire_"))),
    }
    assigned = set().union(*groups.values())
    groups["powertrain-reactions"] = tuple(name for name in VEHICLE_STATE_OUTPUTS if name not in assigned)
    stages: list[dict[str, Any]] = []
    output_cursor = 0
    for identity, names in groups.items():
        output_cursor = ((output_cursor + 63) // 64) * 64
        artifact = emit_webgpu_module(compiled.module, name=compiled.function.name,
                                      outputs={compiled.function.name: tuple(values[name] for name in names)},
                                      count=1, packed_outputs=True)
        if not artifact.complete:
            raise RuntimeError(f"engine profile {profile_identity} stage {identity} does not lower to WebGPU: " +
                               "; ".join(item.format() for item in artifact.shortfalls))
        stages.append({"identity": identity, "outputs": names, "output_offset_floats": output_cursor,
                       "kernel": artifact})
        output_cursor += len(names)
    return tuple(stages)


@lru_cache(maxsize=1)
def symbolic_vehicle_physics_wasm_plugin() -> WorldWasmPlugin:
    equations, _ = symbolic_vehicle_equations()
    compiled = compile_symbolic_vehicle_physics()
    artifact = compile_symbolic_vehicle_physics_wasm()
    return WorldWasmPlugin(
        "abstract-ui/plugins/symbolic-vehicle-physics",
        "integrate-json-configured-car-with-four-parallel-springs",
        artifact.binary, artifact.name,
        ({"name": "io", "role": "arena-base", "dtype": "int32"},),
        "\n".join(str(equation) for equation in equations),
        source_language="sympy", capability="vehicle-physics",
        operation_count=sum(len(block.instrs) for block in compiled.function.blocks.values()),
        reserved_bytes=max((*artifact.input_offsets, *artifact.output_offsets)) + 8,
        abi={
            "kind": "ssa-scalar-arena-v0", "invocation": "arena-base-pointer", "dtype": "float64",
            "input_names": list(artifact.input_names), "output_names": list(artifact.output_names),
            "input_offsets": list(artifact.input_offsets), "output_offsets": list(artifact.output_offsets),
            "spring_lanes": list(WHEEL_NAMES), "reduction": "sum-spring-force-to-chassis",
        },
    )


CONTACT_PATCH_OUTPUTS = (
    "chassis_force_x", "chassis_force_y", "chassis_force_z",
    "chassis_torque_x", "chassis_torque_y", "chassis_torque_z",
    "contact_area",
)
# The AbstractTensor reducer publishes the shared contact-area dependency
# before the six dependent wrench rows.  WGSL storage therefore follows this
# order even though the authored SymPy equations keep the public force/torque/
# area ABI above.  Keep the translation explicit at the vehicle precompile
# boundary; do not teach the generic compiler vehicle semantics.
CONTACT_PATCH_TENSOR_REDUCER_OUTPUTS = ("contact_area", *CONTACT_PATCH_OUTPUTS[:6])


@dataclass(frozen=True, slots=True)
class WheelContactTensorPrecompile:
    """Python/AbstractTensor capture of the SymPy-authored four-wheel law."""

    source: str
    module: Any
    artifacts: tuple[WGSLModule, ...]
    function_name: str
    argument_names: tuple[str, ...]
    output_names: tuple[str, ...]
    packed_outputs: bool


@dataclass(frozen=True, slots=True)
class SympyTensorBackendPrecompile:
    """One SymPy matrix graph captured as AbstractTensor backend regions."""

    source: str
    symbolic_source: str
    artifacts: tuple[WGSLModule, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]


def extra_precision_closure(function, *, limbs: int = 2):
    """Wrap elementwise AbstractTensor math in the repository Precision type.

    Matrix assembly/GEMM remains outside this closure so its semantic identity
    stays available to backend substitution.  The closure is intended for the
    sensitive contact-law portion, whose sqrt/tanh/arithmetic surface is
    explicitly supported by ``Precision``.
    """

    from ..common.tensors.extended_precision import Precision

    width = int(limbs)
    if width < 2:
        return function

    def widened(*values):
        promoted = tuple(Precision.of(value, width) for value in values)
        result = function(*promoted)
        items = result if isinstance(result, tuple) else (result,)
        collapsed = tuple(
            item.collapse() if isinstance(item, Precision) else item
            for item in items
        )
        return collapsed if isinstance(result, tuple) else collapsed[0]

    return widened


def compile_sympy_matrix_to_abstract_tensor_backend(
    expression: sympy.MatrixExpr,
    *,
    name: str,
    input_shapes: Mapping[str, tuple[int, ...]],
) -> SympyTensorBackendPrecompile:
    """SymPy MatrixExpr -> AbstractTensor Python -> planned SSA -> WGSL.

    Fixed shapes belong to this prebake helper, not to the generic compiler.
    The AOT region keeps the ordinary ``AbstractTensor.matmul`` operation;
    tensor SSA preserves its GEMM identity and WebGPU selects the registered
    GPU intrinsic.
    """

    import contextlib
    import io
    import warnings

    import numpy as np

    from ..common.tensors.accelerator_backends.aot_compile import compile_ast_aot
    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from ..transmogrifier.ssa import IRModule
    from .precompile_to_ssa import lower_precompile_and_control_to_ssa
    from .tensor_ssa_lowering import lower_tensor_calls_to_repository_ssa

    if not isinstance(expression, sympy.MatMul) or len(expression.args) != 2:
        raise TypeError("the tensor backend helper currently requires one SymPy matrix product")
    argument_names = tuple(str(argument) for argument in expression.args)
    if set(argument_names) != set(input_shapes):
        raise ValueError("matrix input_shapes must exactly name the SymPy operands")
    left_shape, right_shape = (tuple(map(int, input_shapes[name])) for name in argument_names)
    if len(left_shape) != 2 or len(right_shape) != 2 or left_shape[1] != right_shape[0]:
        raise ValueError("SymPy/AbstractTensor matrix operands must have compatible rank-two shapes")
    output_shape = (left_shape[0], right_shape[1])
    source = (
        f"def {name}({', '.join(argument_names)}):\n"
        f"    return {argument_names[0]}.matmul({argument_names[1]})\n"
    )
    feeds = {
        operand: np.ones(input_shapes[operand], dtype=np.float32)
        for operand in argument_names
    }
    captured = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(captured):
        warnings.simplefilter("ignore")
        compilation = compile_ast_aot(
            source, name, feeds, backend="webgpu", precompile_only=True,
            mutable_parameters=argument_names,
        )
    lowered = lower_precompile_and_control_to_ssa(
        compilation.compiled_shell_program,
        compilation.shell_control_program,
        numerical_name=name,
        control_name=f"{name}_control",
        region_programs=compilation.region_programs,
    )
    if lowered.shortfalls:
        raise RuntimeError("SymPy AbstractTensor planning failed: " + "; ".join(
            item.format() for item in lowered.shortfalls
        ))
    artifacts = []
    for function_name, function in lowered.module.functions.items():
        matrix_call = next((
            instruction
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.attributes.get("tensor_operation") == "matmul"
        ), None)
        if matrix_call is None:
            continue
        for value, shape in zip(matrix_call.args[:2], (left_shape, right_shape)):
            value.shape = shape
            value.dtype = "float32"
        if matrix_call.res is None:
            raise RuntimeError("planned AbstractTensor matmul has no result")
        matrix_call.res.shape = output_shape
        matrix_call.res.dtype = "float32"
        region = IRModule({function_name: function})
        shortfalls = lower_tensor_calls_to_repository_ssa(
            region, c_backend_repository_ssa_reference()
        )
        if shortfalls:
            raise RuntimeError("AbstractTensor matrix SSA lowering failed: " + "; ".join(
                item.format() for item in shortfalls
            ))
        outputs = next(
            instruction.args
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.op.lower() in {"ret", "return"}
        )
        artifact = emit_webgpu_module(
            region, name=function_name, outputs={function_name: outputs}
        )
        if not artifact.complete:
            raise RuntimeError("AbstractTensor matrix WGSL emission failed: " + "; ".join(
                item.format() for item in artifact.shortfalls
            ))
        artifacts.append(artifact)
    if not artifacts:
        raise RuntimeError("AbstractTensor planner did not retain the SymPy matrix product region")
    return SympyTensorBackendPrecompile(
        source=source,
        symbolic_source=sympy.srepr(expression),
        artifacts=tuple(artifacts),
        input_shapes=(left_shape, right_shape),
        output_shape=output_shape,
    )


@lru_cache(maxsize=1)
def compile_vehicle_wrench_reduction_webgpu() -> SympyTensorBackendPrecompile:
    lanes = contact_lane_count()
    wheel_wrenches = sympy.MatrixSymbol("wheel_wrenches", 6, lanes)
    unit_column = sympy.MatrixSymbol("unit_column", lanes, 1)
    return compile_sympy_matrix_to_abstract_tensor_backend(
        wheel_wrenches * unit_column,
        name="reduce_vehicle_wheel_wrenches",
        input_shapes={"wheel_wrenches": (6, lanes), "unit_column": (lanes, 1)},
    )


TORUS_PLANE_CONTACT_ARC_OUTPUTS = (
    "contact_arc_angle",
    "contact_arc_length",
    "integrated_penetration",
    "mean_penetration",
    "peak_penetration",
    "centroid_radial_cosine",
)


@lru_cache(maxsize=1)
def symbolic_torus_plane_contact_arc_equations(
) -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """Exact active-arc moments for a torus cut by a support plane.

    This starts as a genuine integral over the analytically selected contact
    arc.  SymPy reduces the penetration and centroid moments to elementary
    ``acos``/``sqrt`` expressions before they enter repository SSA.  No radial
    terrain probes participate in this geometry.
    """

    theta = sympy.Symbol("contact_arc_theta", real=True)
    symbols = {
        name: sympy.Symbol(name, real=True)
        for name in (
            "center_plane_distance", "major_radius",
            "plane_radial_projection", "section_radius",
        )
    }
    s = symbols
    epsilon = sympy.Float("1e-12")
    projected_major_radius = sympy.Max(
        s["major_radius"] * s["plane_radial_projection"], epsilon,
    )
    boundary_cosine = sympy.Min(sympy.Max(
        (s["section_radius"] - s["center_plane_distance"])
        / projected_major_radius,
        -1,
    ), 1)
    arc_start = sympy.acos(boundary_cosine)
    arc_end = 2 * sympy.pi - arc_start
    integration_radius = sympy.Symbol(
        "projected_major_radius_positive", positive=True, real=True,
    )
    penetration = (
        s["section_radius"] - s["center_plane_distance"]
        - integration_radius * sympy.cos(theta)
    )
    arc_angle = sympy.simplify(arc_end - arc_start)
    integrated_penetration = sympy.simplify(sympy.Integral(
        penetration, (theta, arc_start, arc_end),
    ).doit(conds="none").subs(integration_radius, projected_major_radius))
    radial_first_moment = sympy.simplify(sympy.Integral(
        sympy.cos(theta) * penetration,
        (theta, arc_start, arc_end),
    ).doit(conds="none").subs(integration_radius, projected_major_radius))
    values = {
        "contact_arc_angle": arc_angle,
        "contact_arc_length": s["major_radius"] * arc_angle,
        "integrated_penetration": integrated_penetration,
        "mean_penetration": integrated_penetration / (arc_angle + epsilon),
        "peak_penetration": sympy.Max(
            0,
            s["section_radius"] - s["center_plane_distance"]
            + projected_major_radius,
        ),
        "centroid_radial_cosine": radial_first_moment
        / (integrated_penetration + epsilon),
    }
    return (
        tuple(
            sympy.Eq(sympy.Symbol(name, real=True), values[name], evaluate=False)
            for name in TORUS_PLANE_CONTACT_ARC_OUTPUTS
        ),
        symbols,
    )


def symbolic_torus_plane_patch_boundary_integral() -> sympy.Integral:
    """The unreduced plane-boundary width integral for the active torus arc."""

    theta, distance, major_radius, section_radius, projection = sympy.symbols(
        "contact_arc_theta center_plane_distance major_radius "
        "section_radius plane_radial_projection",
        real=True,
    )
    epsilon = sympy.Float("1e-12")
    projected = sympy.Max(major_radius * projection, epsilon)
    boundary_cosine = sympy.Min(sympy.Max(
        (section_radius - distance) / projected, -1,
    ), 1)
    start = sympy.acos(boundary_cosine)
    signed_tube_distance = distance + projected * sympy.cos(theta)
    chord = 2 * sympy.sqrt(sympy.Max(
        0, section_radius ** 2 - signed_tube_distance ** 2,
    ))
    return sympy.Integral(
        major_radius * chord,
        (theta, start, 2 * sympy.pi - start),
    )


@lru_cache(maxsize=1)
def compile_torus_plane_contact_arc_ssa() -> SymbolicEquationCompilation:
    equations, _symbols = symbolic_torus_plane_contact_arc_equations()
    return compile_sympy_equations(
        equations,
        name="torus_plane_contact_arc",
        publications=tuple(
            SymbolicPublication(name, f"world.vehicle.contact.torus.{name}")
            for name in TORUS_PLANE_CONTACT_ARC_OUTPUTS
        ),
        dtype="float64",
    )


@lru_cache(maxsize=1)
def compile_torus_plane_contact_arc_c() -> CFunctionArtifact:
    """Emit the same reduced torus contact-arc program as scalar native C."""

    compilation = compile_torus_plane_contact_arc_ssa()
    artifact = emit_ssa_function_to_c(
        compilation.module, compilation.function.name,
        entry_name="torus_plane_contact_arc",
    )
    if not artifact.complete:
        reasons = "; ".join(item.reason for item in artifact.shortfalls)
        raise RuntimeError(f"torus/plane contact arc does not lower to C: {reasons}")
    return artifact


@lru_cache(maxsize=1)
def compile_torus_plane_contact_arc_llvm():
    """Lower the SymPy-reduced contact-arc integral through SSA to LLVM."""

    from .ssa_llvm_backend import emit_ssa_function_to_llvm

    compilation = compile_torus_plane_contact_arc_ssa()
    return emit_ssa_function_to_llvm(
        compilation.module, compilation.function.name,
    )


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def _symbolic_wheel_contact_equations_loaded() -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    return symbolic_equations_cached(_symbolic_wheel_contact_equations_authored)


def symbolic_wheel_contact_equations() -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """The authored contact law, built once per revision of this file."""

    equations, symbols = _symbolic_wheel_contact_equations_loaded()
    return tuple(equations), dict(symbols)


def _symbolic_wheel_contact_equations_authored() -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """Per-wheel mesh-contact, pneumatic patch, and Coulomb friction law."""

    names = (
        "dt support hub_height hub_velocity_y previous_compression compression_velocity geometric_compression surface_height "
        "chassis_velocity_y roll_velocity pitch_velocity wheelbase_half_length axle_group_offset_x track_half_width "
        "corner_front_sign corner_side_sign active_damping_minimum_scale active_damping_maximum_scale "
        "active_damping_body_velocity_gain_s_per_m active_damping_rebound_release_gain_s_per_m "
        "normal_x normal_y normal_z forward_x forward_y forward_z right_x right_y right_z "
        "tire_radial_compression tire_radial_velocity suspension_alignment "
        "tire_major_radius tire_section_radius tire_effective_tread_width tire_reference_volume "
        "tire_gas_polytropic_exponent radial_carcass_loss tire_radial_effective_mass "
        "slip_longitudinal slip_lateral attachment_x attachment_y attachment_z "
        "sidewall_deformation_longitudinal sidewall_deformation_velocity_longitudinal "
        "sidewall_deformation_lateral sidewall_deformation_velocity_lateral "
        "sidewall_shear_stiffness_longitudinal sidewall_shear_stiffness_lateral sidewall_shear_damping "
        "corner_weight suspension_rest_length chassis_clearance suspension_travel "
        "spring_stiffness bump_stop_stiffness bump_stop_progressive_stiffness bump_stop_damping "
        "linkage_motion_ratio pneumatic_compression_damping pneumatic_rebound_damping pneumatic_efficiency "
        "maximum_compression_speed tire_pressure minimum_contact_area maximum_contact_area "
        "mu_static mu_kinetic load_sensitivity "
        "slip_transition_speed"
    )
    s = {name: sympy.Symbol(name, real=True) for name in names.split()}
    epsilon = sympy.Float("1e-5")
    compression = sympy.Min(sympy.Max(s["previous_compression"], 0),
                            s["suspension_travel"])
    compression_rate = sympy.Min(
        sympy.Max(s["compression_velocity"],
                  -s["maximum_compression_speed"]),
        s["maximum_compression_speed"])
    corner_body_velocity = (s["chassis_velocity_y"]
                            + s["corner_front_sign"] * s["pitch_velocity"]
                            * s["wheelbase_half_length"]
                            + s["pitch_velocity"] * s["axle_group_offset_x"]
                            - s["corner_side_sign"] * s["roll_velocity"]
                            * s["track_half_width"])
    raw_damping_scale = (1
                         + s["active_damping_body_velocity_gain_s_per_m"]
                         * _smooth_abs(corner_body_velocity)
                         - s["active_damping_rebound_release_gain_s_per_m"]
                         * _c2_positive(-compression_rate, sympy.Float("0.08")))
    damping_scale = sympy.Min(sympy.Max(raw_damping_scale,
                                       s["active_damping_minimum_scale"]),
                              s["active_damping_maximum_scale"])
    pneumatic = s["pneumatic_efficiency"] * (
        damping_scale * s["pneumatic_compression_damping"]
        * _c2_positive(compression_rate, sympy.Float("0.08"))
        - damping_scale * s["pneumatic_rebound_damping"]
        * _c2_positive(-compression_rate, sympy.Float("0.08"))
    )
    suspension_load = sympy.Max(0,
        (s["spring_stiffness"] * compression * s["linkage_motion_ratio"] + pneumatic
         + s["bump_stop_stiffness"] * sympy.Max(
             0, s["geometric_compression"] - s["suspension_travel"])
         + s["bump_stop_progressive_stiffness"] * sympy.Max(
             0, s["geometric_compression"] - s["suspension_travel"]) ** 2
         + s["bump_stop_damping"] * sympy.Max(0, -s["tire_radial_velocity"]))
        * s["linkage_motion_ratio"]) * s["support"]
    # Toroidal pneumatic capacity. The flattened chord and tread width define
    # the real footprint; displaced toroid volume raises gas pressure. There is
    # no radial k*x tire spring. Suspension demand and pneumatic capacity are
    # two sides of a series load path, so the transmitted load is their smooth
    # minimum rather than their sum.
    radial_deformation = sympy.Min(sympy.Max(s["tire_radial_compression"], 0),
                                   s["tire_section_radius"] * sympy.Float("1.65"))
    chord_squared = sympy.Max(
        0, 2 * s["tire_major_radius"] * radial_deformation - radial_deformation ** 2)
    geometric_area = 2 * sympy.sqrt(chord_squared) * s["tire_effective_tread_width"]
    contact_area = sympy.Min(sympy.Max(geometric_area, s["minimum_contact_area"]),
                             s["maximum_contact_area"]) * s["support"]
    displaced_volume = contact_area * radial_deformation * sympy.Float("0.55")
    volume_strain = sympy.Min(sympy.Max(
        displaced_volume / (s["tire_reference_volume"] + epsilon), 0),
        sympy.Float("0.65"))
    compressed_pressure = s["tire_pressure"] * (
        1 + s["tire_gas_polytropic_exponent"] * volume_strain
        + s["tire_gas_polytropic_exponent"]
        * (s["tire_gas_polytropic_exponent"] + 1) * volume_strain ** 2 / 2)
    elastic_pneumatic_force = compressed_pressure * contact_area
    # The contact response is the impulse of a real radial mode, not a
    # positional rejection or an instantaneous penalty force. Three implicit
    # midpoint stages conserve the undamped mode and make carcass loss strictly
    # dissipative. Max is the unilateral release: terrain never attracts tire.
    radial_stiffness = elastic_pneumatic_force / (
        radial_deformation + s["tire_section_radius"] * sympy.Float("1e-5") + epsilon)
    _ring_x, _ring_v, radial_impulse, _ring_loss = _passive_radial_ringdown(
        radial_deformation,
        -s["tire_radial_velocity"],
        radial_stiffness,
        s["radial_carcass_loss"],
        s["tire_radial_effective_mass"],
        s["dt"],
    )
    pneumatic_capacity = sympy.Max(
        0, radial_impulse / (s["dt"] + sympy.Float("1e-12"))) * s["support"]
    # This kernel publishes the terrain reaction at the tyre node. It must not
    # pre-collapse that force with suspension demand: the unsprung mass between
    # the pneumatic tyre and coilover is a real graph node and the difference
    # between those two forces is precisely what accelerates it.
    normal_load = pneumatic_capacity * s["support"]
    reference_area = s["corner_weight"] / s["tire_pressure"]
    patch_scale = sympy.Min(sympy.Max(contact_area / (reference_area + epsilon),
                                     sympy.Float("0.62")), sympy.Float("1.18"))
    overload = sympy.Max(0, normal_load / (s["corner_weight"] + epsilon) - 1)
    load_scale = sympy.Min(sympy.Max(1 - s["load_sensitivity"] * overload,
                                    sympy.Float("0.58")), 1)
    requested_long = -(s["sidewall_shear_stiffness_longitudinal"]
                       * s["sidewall_deformation_longitudinal"]
                       + s["sidewall_shear_damping"]
                       * s["sidewall_deformation_velocity_longitudinal"])
    requested_lateral = -(s["sidewall_shear_stiffness_lateral"]
                          * s["sidewall_deformation_lateral"]
                          + s["sidewall_shear_damping"]
                          * s["sidewall_deformation_velocity_lateral"])
    requested_magnitude = sympy.sqrt(requested_long ** 2 + requested_lateral ** 2 + epsilon ** 2)
    residual_longitudinal_slip = s["slip_longitudinal"] - s["sidewall_deformation_velocity_longitudinal"]
    residual_lateral_slip = s["slip_lateral"] - s["sidewall_deformation_velocity_lateral"]
    slip_speed = sympy.sqrt(residual_longitudinal_slip ** 2 + residual_lateral_slip ** 2
                            + epsilon ** 2)
    stribeck_ratio = slip_speed / (s["slip_transition_speed"] + epsilon)
    effective_mu = (s["mu_kinetic"]
                    + (s["mu_static"] - s["mu_kinetic"])
                    / (1 + stribeck_ratio ** 2))
    friction_limit = effective_mu * load_scale * patch_scale * normal_load
    # Combined-slip bristle demand under one Stribeck/Coulomb constitutive
    # law.  This is not a cross-fade between static and kinetic force answers:
    # slip determines one coefficient, and the bristle response saturates at
    # that physical contact-patch limit.
    friction_magnitude = friction_limit * sympy.tanh(
        requested_magnitude / (friction_limit + epsilon))
    long_force = requested_long / requested_magnitude * friction_magnitude
    lateral_force = requested_lateral / requested_magnitude * friction_magnitude
    force = {
        axis: normal_load * s[f"normal_{axis}"]
        + long_force * s[f"forward_{axis}"] + lateral_force * s[f"right_{axis}"]
        for axis in "xyz"
    }
    torque = {
        "x": s["attachment_y"] * force["z"] - s["attachment_z"] * force["y"],
        "y": s["attachment_z"] * force["x"] - s["attachment_x"] * force["z"],
        "z": s["attachment_x"] * force["y"] - s["attachment_y"] * force["x"],
    }
    expressions = {
        **{f"chassis_force_{axis}": force[axis] for axis in "xyz"},
        **{f"chassis_torque_{axis}": torque[axis] for axis in "xyz"},
        "contact_area": contact_area,
    }
    return tuple(sympy.Eq(sympy.Symbol(name, real=True), expression, evaluate=False)
                 for name, expression in expressions.items()), s


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def compile_wheel_contact_ssa() -> SymbolicEquationCompilation:
    publications = tuple(SymbolicPublication(name, f"world.vehicle.contact.{name}")
                         for name in CONTACT_PATCH_OUTPUTS)
    return compile_symbolic_program(
        _symbolic_wheel_contact_equations_authored, name="abstract_ui_wheel_contact",
        publications=publications, dtype="float32",
    )


@lru_cache(maxsize=1)
def compile_wheel_contact_wasm() -> SSAWasmArtifact:
    """Compile the scalar contact law for browsers without WebGPU."""

    compiled = compile_wheel_contact_ssa()
    artifact = emit_ssa_function_to_wasm(
        compiled.module, compiled.function.name, work_contract="deploy",
    )
    if not artifact.complete:
        reasons = "; ".join(item.reason for item in artifact.shortfalls)
        raise RuntimeError(f"wheel contact does not lower to WASM: {reasons}")
    return artifact


@lru_cache(maxsize=1)
def compile_wheel_contact_c() -> CFunctionArtifact:
    """Emit the authored pneumatic wheel contact law as direct native C."""

    compiled = compile_wheel_contact_ssa()
    artifact = emit_ssa_function_to_c(
        compiled.module, compiled.function.name,
        entry_name="abstract_ui_wheel_contact",
    )
    if not artifact.complete:
        reasons = "; ".join(item.reason for item in artifact.shortfalls)
        raise RuntimeError(f"wheel contact does not lower to C: {reasons}")
    return artifact


@lru_cache(maxsize=1)
def symbolic_wheel_contact_wasm_plugin() -> WorldWasmPlugin:
    """Publish the same tire law as a scalar, four-call fallback ABI."""

    equations, _ = symbolic_wheel_contact_equations()
    compiled = compile_wheel_contact_ssa()
    artifact = compile_wheel_contact_wasm()
    return WorldWasmPlugin(
        "abstract-ui/plugins/symbolic-wheel-contact",
        "evaluate-one-json-configured-wheel-contact",
        artifact.binary, artifact.name,
        ({"name": "io", "role": "arena-base", "dtype": "int32"},),
        "\n".join(str(equation) for equation in equations),
        source_language="sympy", capability="vehicle-contact-fallback",
        operation_count=sum(len(block.instrs) for block in compiled.function.blocks.values()),
        reserved_bytes=max((*artifact.input_offsets, *artifact.output_offsets)) + 8,
        abi={
            "kind": "ssa-scalar-arena-v0", "invocation": "arena-base-pointer",
            "dtype": "float64", "lane_count": len(WHEEL_NAMES),
            "input_names": list(artifact.input_names), "output_names": list(artifact.output_names),
            "input_offsets": list(artifact.input_offsets), "output_offsets": list(artifact.output_offsets),
            "authority": "same-symbolic-law-as-primary-webgpu-contact-kernel",
        },
    )


@lru_cache(maxsize=2)
def compile_wheel_contact_abstract_tensor(
    *, packed_outputs: bool = False,
) -> WheelContactTensorPrecompile:
    """Vectorize the SymPy law on a real four-wheel AbstractTensor axis.

    SymPy remains the mathematical authority.  Common-subexpression
    elimination is a precompile concern: it produces compact Python whose
    arguments are four-element tensors, and the ordinary AOT frontend then
    captures those tensor operations and lowers them to repository SSA.
    """

    import contextlib
    import copy
    import io
    import warnings

    import numpy as np

    from ..common.tensors.accelerator_backends.aot_compile import compile_ast_aot
    from ..common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from ..transmogrifier.ssa import IRModule
    from .precompile_to_ssa import lower_precompile_and_control_to_ssa
    from .tensor_ssa_lowering import lower_tensor_calls_to_repository_ssa

    # One program, three stages: the contact law's AbstractTensor stage is the
    # compiler's own materialization of the SSA that compile_wheel_contact_ssa
    # produced (CSE, identities, precision contracts), not a sympy printer.
    # That SSA is disk-cached per source revision, so this costs nothing.
    from .vehicle_python_compilation import symbolic_abstract_tensor_source

    compiled_law = compile_wheel_contact_ssa()
    argument_names = tuple(compiled_law.function.metadata["argument_names"])
    source = symbolic_abstract_tensor_source(
        compiled_law, "abstract_ui_wheel_contact_tensor",
    )
    feeds = {name: np.ones(4, dtype=np.float32) for name in argument_names}
    # Give discovery physically non-singular bounds without baking them into
    # the program; every argument remains mutable and runtime-parametric.
    for name, value in {
        "dt": 1 / 120, "support": 1, "suspension_travel": .34,
        "maximum_compression_speed": 1.25, "tire_pressure": 155000,
        "minimum_contact_area": .006, "maximum_contact_area": .045,
        "corner_weight": 1520, "slip_transition_speed": .38,
    }.items():
        if name in feeds:
            feeds[name].fill(value)
    captured = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(captured):
        warnings.simplefilter("ignore")
        compilation = compile_ast_aot(
            source,
            "abstract_ui_wheel_contact_tensor",
            feeds,
            backend="webgpu",
            precompile_only=True,
            mutable_parameters=argument_names,
        )
    lowered = lower_precompile_and_control_to_ssa(
        compilation.compiled_shell_program,
        compilation.shell_control_program,
        numerical_name="abstract_ui_wheel_contact_tensor",
        control_name="abstract_ui_wheel_contact_tensor_control",
        region_programs=compilation.region_programs,
    )
    if lowered.shortfalls:
        raise RuntimeError("wheel-contact AbstractTensor lowering failed: " + "; ".join(
            item.format() for item in lowered.shortfalls
        ))
    numerical_names = tuple(
        name for name in lowered.module.functions
        if name.startswith("numerical_region_")
    )
    if len(numerical_names) != 1:
        raise RuntimeError(
            "wheel-contact precompile must remain one vectorized numerical "
            f"region, found {len(numerical_names)}"
        )
    numerical_name = numerical_names[0]
    numerical = copy.deepcopy(lowered.module.functions[numerical_name])
    # Every mutable input is a contact-patch vector: four tyres followed by the
    # cage nodes and member midpoints.  AOT deliberately leaves shapes abstract;
    # this purpose-baked precompile owns the fixed lane count.
    lanes = contact_lane_count()
    for value in numerical.args:
        value.shape = (lanes,)
        value.dtype = "float32"
    for block in numerical.blocks.values():
        for instruction in block.instrs:
            if instruction.res is not None:
                instruction.res.shape = (lanes,)
                instruction.res.dtype = "float32"
    numerical_module = IRModule({numerical_name: numerical})
    tensor_shortfalls = lower_tensor_calls_to_repository_ssa(
        numerical_module, c_backend_repository_ssa_reference()
    )
    if tensor_shortfalls:
        raise RuntimeError("wheel-contact tensor SSA lowering failed: " + "; ".join(
            item.format() for item in tensor_shortfalls
        ))
    numerical_outputs = next(
        instruction.args
        for block in numerical.blocks.values()
        for instruction in block.instrs
        if instruction.op.lower() in {"ret", "return"}
    )
    if len(numerical_outputs) != len(CONTACT_PATCH_TENSOR_REDUCER_OUTPUTS):
        raise RuntimeError("wheel-contact tensor reducer output count changed")
    reducer_publications = dict(zip(
        CONTACT_PATCH_TENSOR_REDUCER_OUTPUTS, numerical_outputs, strict=True,
    ))
    numerical_outputs = tuple(reducer_publications[name] for name in CONTACT_PATCH_OUTPUTS)
    artifact = emit_webgpu_module(
        numerical_module,
        name=numerical_name,
        outputs={numerical_name: numerical_outputs},
        count=contact_lane_count(),
        preferred_local_size=contact_lane_count(),
        packed_outputs=packed_outputs,
    )
    if not artifact.complete:
        raise RuntimeError("wheel-contact tensor WGSL emission failed: " + "; ".join(
            item.format() for item in artifact.shortfalls
        ))
    return WheelContactTensorPrecompile(
        source=source,
        module=lowered.module,
        artifacts=(artifact,),
        function_name="abstract_ui_wheel_contact_tensor_control",
        argument_names=argument_names,
        output_names=CONTACT_PATCH_OUTPUTS,
        packed_outputs=bool(packed_outputs),
    )


@lru_cache(maxsize=1)
def compile_wheel_contact_webgpu() -> WGSLModule:
    # Source the storage shader from the vectorized SymPy -> AbstractTensor
    # compiler path; its lexical publication order is described alongside it.
    return compile_wheel_contact_abstract_tensor(
        packed_outputs=False,
    ).artifacts[0]


def _wgsl_number(value: float) -> str:
    text = repr(float(value))
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return f"{text}f"


def _vehicle_terrain_contact_webgpu(
    config: VehicleConfiguration,
    contact_inputs: tuple[str, ...],
    vehicle_inputs: tuple[str, ...],
) -> dict[str, Any]:
    """Bake graph-configured radial tire probes into a GPU terrain gather.

    This kernel is coordination WGSL around compiler-owned math kernels.  It
    performs the indexed height-field gather that scalar SymPy does not own,
    and publishes the exact packed feed ABI consumed by the compiled contact
    law.  Terrain storage is read-only and changes only when the authored
    height field changes.
    """

    source = config.source
    chassis, suspension = source["chassis"], source["suspension"]
    wheels, tires = source["wheels"], source["tires"]
    graph = _vehicle_mechanical_graph(config)
    nodes = {node["identity"]: node for node in graph["nodes"]}
    load_audit = graph["load_audit"]
    contact_index = {name: index for index, name in enumerate(contact_inputs)}
    vehicle_index = {name: index for index, name in enumerate(vehicle_inputs)}

    def vehicle(name: str) -> str:
        return f"vehicle_feed[{vehicle_index[name]}u]"

    def wheel_vehicle(field: str) -> str:
        expression = "0.0f"
        for lane, wheel in reversed(tuple(enumerate(WHEEL_NAMES))):
            expression = f"select({expression}, {vehicle(f'{field}_{wheel}')}, lane == {lane}u)"
        return expression

    lanes = contact_lane_count()

    def store(name: str, expression: str) -> str:
        return f"  contact_feed[{contact_index[name]}u * {lanes}u + lane] = {expression};"

    def cage_store(name: str, expression: str) -> str:
        return f"  contact_feed[{contact_index[name]}u * {lanes}u + lane] = {expression};"

    sprung_mass = config.sprung_mass()
    static_compressions = {
        wheel: min(
            float(suspension["travel"]),
            sprung_mass * abs(float(source["world"]["gravity"]))
            * float(source["mass_distribution"][wheel]) / float(suspension["stiffness"]),
        )
        for wheel in WHEEL_NAMES
    }
    hubs = [nodes[f"suspension.{wheel}.hub"]["reference_position"] for wheel in WHEEL_NAMES]
    hub_cases = []
    for lane, (wheel, hub) in enumerate(zip(WHEEL_NAMES, hubs)):
        hub_cases.append(
            f"  if (lane == {lane}u) {{ hub_local = vec3<f32>("
            f"{_wgsl_number(hub[0])}, {_wgsl_number(hub[1] - static_compressions[wheel])} + compression, "
            f"{_wgsl_number(hub[2])}); }}"
        )
    corner_weights = [
        float(source["mass"]) * abs(float(source["world"]["gravity"]))
        * float(source["mass_distribution"][wheel])
        for wheel in WHEEL_NAMES
    ]
    constants = {
        "active_damping_body_velocity_gain_s_per_m": suspension["active_damping_body_velocity_gain_s_per_m"],
        "active_damping_maximum_scale": suspension["active_damping_maximum_scale"],
        "active_damping_minimum_scale": suspension["active_damping_minimum_scale"],
        "active_damping_rebound_release_gain_s_per_m": suspension["active_damping_rebound_release_gain_s_per_m"],
        "dt": 1 / 120,
        "load_sensitivity": tires["load_sensitivity"],
        "radial_carcass_loss": tires["radial_carcass_loss_n_s_per_m"],
        "tire_radial_effective_mass": config.unsprung_mass_per_corner()
        * tires["radial_contact_effective_mass_fraction_of_unsprung"],
        "sidewall_shear_stiffness_longitudinal": tires["sidewall_shear_stiffness_longitudinal_n_per_m"],
        "sidewall_shear_stiffness_lateral": tires["sidewall_shear_stiffness_lateral_n_per_m"],
        "sidewall_shear_damping": tires["sidewall_shear_damping_n_s_per_m"],
        "maximum_compression_speed": suspension["maximum_compression_speed"],
        "bump_stop_stiffness": suspension["bump_stop_stiffness_n_per_m"],
        "bump_stop_progressive_stiffness": suspension["bump_stop_progressive_stiffness_n_per_m2"],
        "bump_stop_damping": suspension["bump_stop_damping_n_s_per_m"],
        "maximum_contact_area": tires["maximum_contact_area"],
        "minimum_contact_area": tires["minimum_contact_area"],
        "mu_kinetic": tires["kinetic_friction"],
        "mu_static": tires["static_friction"],
        "pneumatic_compression_damping": suspension["pneumatic_compression_damping"],
        "pneumatic_efficiency": suspension["pneumatic_efficiency"],
        "pneumatic_rebound_damping": suspension["pneumatic_rebound_damping"],
        "slip_transition_speed": tires["slip_transition_speed"],
        "spring_stiffness": suspension["stiffness"],
        "suspension_travel": suspension["travel"],
        "tire_pressure": tires["pressure_pa"],
        "tire_major_radius": tires["radius"] - tires["toroid_section_radius_m"],
        "tire_section_radius": tires["toroid_section_radius_m"],
        "tire_effective_tread_width": tires["width"] * tires["effective_tread_width_fraction"],
        "tire_reference_volume": 2 * math.pi ** 2 * (tires["radius"] - tires["toroid_section_radius_m"])
        * tires["toroid_section_radius_m"] ** 2,
        "tire_gas_polytropic_exponent": tires["gas_polytropic_exponent"],
        "track_half_width": wheels["track_half_width"],
        "wheelbase_half_length": wheels["wheelbase_half_length"],
    }
    expressions = {
        **{name: _wgsl_number(value) for name, value in constants.items()},
        "attachment_x": "attachment.x", "attachment_y": "attachment.y", "attachment_z": "attachment.z",
        "chassis_velocity_y": vehicle("velocity_y"),
        "compression_velocity": (
            f"select(select({vehicle('compression_velocity_rear_left')}, "
            f"{vehicle('compression_velocity_rear_right')}, lane == 3u), "
            f"select({vehicle('compression_velocity_front_left')}, "
            f"{vehicle('compression_velocity_front_right')}, lane == 1u), lane < 2u)"
        ),
        "corner_front_sign": "select(-1.0f, 1.0f, lane < 2u)",
        "corner_side_sign": "select(1.0f, -1.0f, (lane & 1u) == 0u)",
        "corner_weight": "CORNER_WEIGHTS[lane]",
        "forward_x": "rolling.x", "forward_y": "rolling.y", "forward_z": "rolling.z",
        "geometric_compression": "geometric_compression",
        "linkage_motion_ratio": "1.0f",
        "normal_x": "surface_normal.x", "normal_y": "surface_normal.y", "normal_z": "surface_normal.z",
        "tire_radial_compression": "tire_radial_compression",
        "tire_radial_velocity": "tire_radial_velocity",
        "suspension_alignment": "suspension_alignment",
        "pitch_velocity": vehicle("pitch_velocity"),
        "previous_compression": "compression",
        "right_x": "right.x", "right_y": "right.y", "right_z": "right.z",
        "roll_velocity": vehicle("roll_velocity"),
        "slip_lateral": "slip_lateral", "slip_longitudinal": "slip_longitudinal",
        "sidewall_deformation_longitudinal": wheel_vehicle("tire_deformation_longitudinal"),
        "sidewall_deformation_velocity_longitudinal": wheel_vehicle("tire_deformation_velocity_longitudinal"),
        "sidewall_deformation_lateral": wheel_vehicle("tire_deformation_lateral"),
        "sidewall_deformation_velocity_lateral": wheel_vehicle("tire_deformation_velocity_lateral"),
        "support": "support",
    }
    expressions["tire_pressure"] = "max(1000.0f, controls[29u])"
    missing = set(contact_inputs) - set(expressions)
    if missing:
        raise RuntimeError(f"terrain contact kernel does not populate contact inputs: {sorted(missing)}")
    stores = "\n".join(store(name, expressions[name]) for name in contact_inputs)
    template = r'''struct TerrainSample {
  height: f32,
  normal: vec3<f32>,
  valid: u32,
};
struct TerrainCrossing {
  fraction: f32,
  point: vec3<f32>,
  normal: vec3<f32>,
  valid: u32,
};
@group(0) @binding(0) var<storage, read> terrain_heights: array<f32>;
@group(0) @binding(1) var<storage, read> terrain_parameters: array<f32>;
@group(0) @binding(2) var<storage, read> vehicle_feed: array<f32>;
@group(0) @binding(3) var<storage, read_write> contact_feed: array<f32>;
@group(0) @binding(4) var<storage, read> controls: array<f32>;
// Axis-aligned solid wall prisms, packed as minimum.xyz, maximum.xyz.
// They are uploaded only when authored world geometry changes.
@group(0) @binding(5) var<storage, read> wall_colliders: array<f32>;
// Resident 4 x 15 torus-contact diagnostic lanes. Slots 0..2 per wheel carry
// arc angle, integrated penetration, and mean penetration; the remaining
// slots are reserved. The snapshot stage reads the buffer directly.
@group(0) @binding(6) var<storage, read_write> radial_probes: array<f32>;

const RADIAL_ANGLES: array<f32, 5> = array<f32, 5>(-0.95f, -0.48f, 0.0f, 0.48f, 0.95f);
const LATERAL_FRACTIONS: array<f32, 3> = array<f32, 3>(-0.38f, 0.0f, 0.38f);
const CORNER_WEIGHTS: array<f32, 4> = array<f32, 4>(@@CORNER_WEIGHTS@@);
const CAGE_PATCH_LOCAL: array<vec3<f32>, @@CAGE_PATCH_COUNT@@> =
  array<vec3<f32>, @@CAGE_PATCH_COUNT@@>(@@CAGE_PATCH_LOCAL@@);

fn c2_unit(value: f32) -> f32 {
  let t = clamp(value, 0.0f, 1.0f);
  return t * t * t * (10.0f + t * (-15.0f + 6.0f * t));
}
fn safe_normalize(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
  let magnitude = length(value);
  if (magnitude <= 1.0e-6f) { return fallback; }
  return value / magnitude;
}
fn rotate_body(v: vec3<f32>, roll: f32, pitch: f32, yaw: f32) -> vec3<f32> {
  let cr = cos(roll); let sr = sin(roll); let cp = cos(pitch); let sp = sin(pitch);
  let cy = cos(yaw); let sy = sin(yaw);
  let r = vec3<f32>(v.x, v.y * cr - v.z * sr, v.y * sr + v.z * cr);
  let p = vec3<f32>(r.x * cp - r.y * sp, r.x * sp + r.y * cp, r.z);
  return vec3<f32>(p.x * cy - p.z * sy, p.y, p.x * sy + p.z * cy);
}
fn terrain_sample(x: f32, z: f32, query_y: f32) -> TerrainSample {
  // Header: field_count, wall_count. Each field then owns twelve floats:
  // origin.xyz, cell.xz, columns, rows, domain x/x/z/z, height-buffer offset.
  // Selecting locally by domain and reachable height allows a raised ramp to
  // overlap the ground below it without becoming an infinite collision plane.
  let field_count = u32(terrain_parameters[0u]);
  var best = TerrainSample(0.0f, vec3<f32>(0.0f, 1.0f, 0.0f), 0u);
  var best_height = -1.0e20f;
  var inside_field_domain = false;
  for (var field_index = 0u; field_index < field_count; field_index += 1u) {
    let base = 2u + field_index * 12u;
    let origin_x = terrain_parameters[base]; let origin_y = terrain_parameters[base + 1u];
    let origin_z = terrain_parameters[base + 2u]; let cell_x = terrain_parameters[base + 3u];
    let cell_z = terrain_parameters[base + 4u]; let columns = u32(terrain_parameters[base + 5u]);
    let rows = u32(terrain_parameters[base + 6u]); let minimum_x = terrain_parameters[base + 7u];
    let maximum_x = terrain_parameters[base + 8u]; let minimum_z = terrain_parameters[base + 9u];
    let maximum_z = terrain_parameters[base + 10u]; let height_offset = u32(terrain_parameters[base + 11u]);
    if (columns < 2u || rows < 2u || x < minimum_x || x > maximum_x || z < minimum_z || z > maximum_z) {
      continue;
    }
    inside_field_domain = true;
    let u = clamp((x - origin_x) / cell_x, 0.0f, f32(columns - 1u));
    let v = clamp((z - origin_z) / cell_z, 0.0f, f32(rows - 1u));
    let column = min(columns - 2u, u32(floor(u))); let row = min(rows - 2u, u32(floor(v)));
    let tx = u - f32(column); let tz = v - f32(row); let sample_base = height_offset + row * columns + column;
    let h00 = terrain_heights[sample_base]; let h10 = terrain_heights[sample_base + 1u];
    let h01 = terrain_heights[sample_base + columns]; let h11 = terrain_heights[sample_base + columns + 1u];
    var height: f32; var gradient: vec2<f32>;
    if (tx >= tz) {
      height = origin_y + h00 + (h10 - h00) * tx + (h11 - h10) * tz;
      gradient = vec2<f32>((h10 - h00) / cell_x, (h11 - h10) / cell_z);
    } else {
      height = origin_y + h00 + (h11 - h01) * tx + (h01 - h00) * tz;
      gradient = vec2<f32>((h11 - h01) / cell_x, (h01 - h00) / cell_z);
    }
    if (height <= query_y + @@CONTACT_REACH@@ && height > best_height) {
      best_height = height;
      best = TerrainSample(height, safe_normalize(vec3<f32>(-gradient.x, 1.0f, -gradient.y),
        vec3<f32>(0.0f, 1.0f, 0.0f)), 1u);
    }
  }
  // Sampled fields replace the ordinary floor only inside their authored
  // domains. Outside them the same one-sided solid half-space used by the
  // platformer remains ground; it is not an AABB/prism with a collidable
  // underside. Omitting this fallback left GPU tyres with no ground at all in
  // the open staging yard.
  if (!inside_field_domain && 0.0f <= query_y + @@CONTACT_REACH@@) {
    return TerrainSample(0.0f, vec3<f32>(0.0f, 1.0f, 0.0f), 1u);
  }
  return best;
}

// Find the first top-surface crossing over eight trajectory cells, then
// bisect that cell against the actual sampled height field. This is a
// time-of-impact query; it never converts end-of-step overlap into force.
fn terrain_segment_crossing(start: vec3<f32>, finish: vec3<f32>) -> TerrainCrossing {
  var previous_fraction = 0.0f;
  var previous_sample = terrain_sample(start.x, start.z, start.y);
  var previous_clearance = start.y - previous_sample.height;
  for (var subdivision = 1u; subdivision <= 8u; subdivision += 1u) {
    let fraction = f32(subdivision) / 8.0f;
    let candidate_position = mix(start, finish, fraction);
    let candidate_sample = terrain_sample(candidate_position.x, candidate_position.z, candidate_position.y);
    let candidate_clearance = candidate_position.y - candidate_sample.height;
    if (previous_sample.valid != 0u && candidate_sample.valid != 0u
        && previous_clearance >= 0.0f && candidate_clearance <= 0.0f) {
      var lower = previous_fraction;
      var upper = fraction;
      var hit_sample = candidate_sample;
      for (var iteration = 0u; iteration < 8u; iteration += 1u) {
        let middle = (lower + upper) * 0.5f;
        let middle_position = mix(start, finish, middle);
        let middle_sample = terrain_sample(middle_position.x, middle_position.z, middle_position.y);
        if (middle_sample.valid != 0u && middle_position.y - middle_sample.height <= 0.0f) {
          upper = middle;
          hit_sample = middle_sample;
        } else {
          lower = middle;
        }
      }
      let trajectory_point = mix(start, finish, upper);
      return TerrainCrossing(upper,
        vec3<f32>(trajectory_point.x, hit_sample.height, trajectory_point.z),
        hit_sample.normal, 1u);
    }
    previous_fraction = fraction;
    previous_sample = candidate_sample;
    previous_clearance = candidate_clearance;
  }
  return TerrainCrossing(0.0f, start, vec3<f32>(0.0f, 1.0f, 0.0f), 0u);
}

// A CAGE PATCH IS A CONTACT ROW, NOT A SECOND FORCE LAW.
//
// The shell shares the tyres' lane axis, their contact law and their
// reduction, because that is what constraint_reduction.contact declares: the
// tyre patches and the cage node/member midpoints reduce to the chassis
// together.  Zero suspension_alignment closes the geometric branch of the law,
// so the patch is carried by the same radial contact the carcass uses and
// braked by the same tanh-saturated Coulomb friction -- which is exactly why
// it cannot manufacture energy the way a hand-written penalty contact does.
fn vehicle_cage_patch_contact(lane: u32, position: vec3<f32>, velocity: vec3<f32>,
    roll: f32, pitch: f32, yaw: f32) {
  let patch_local = CAGE_PATCH_LOCAL[lane - @@WHEEL_LANES@@u];
  let patch_offset = rotate_body(patch_local, roll, pitch, yaw);
  let patch_world = position + patch_offset;
  let sample = terrain_sample(patch_world.x, patch_world.z, patch_world.y);
  let is_shell = lane >= @@SHELL_LANE_START@@u;
  let patch_radius = select(@@CAGE_RADIUS@@, @@SHELL_RADIUS@@, is_shell);
  let squash = sample.height + patch_radius - patch_world.y;
  let shell_enabled = select(1.0f, controls[24u], is_shell);
  let touching = select(0.0f, shell_enabled, sample.valid != 0u && squash > 0.0f
    && squash <= @@CAGE_MAXIMUM_SQUASH@@);
  let forward_axis = rotate_body(vec3<f32>(1.0f, 0.0f, 0.0f), roll, pitch, yaw);
  let right_axis = rotate_body(vec3<f32>(0.0f, 0.0f, 1.0f), roll, pitch, yaw);
  let pitch_axis = vec3<f32>(-sin(yaw), 0.0f, cos(yaw));
  let angular = forward_axis * @@ROLL_VELOCITY@@ + vec3<f32>(0.0f, @@YAW_VELOCITY@@, 0.0f)
    + pitch_axis * @@PITCH_VELOCITY@@;
  let arm = patch_offset - rotate_body(@@CENTER_OF_MASS@@, roll, pitch, yaw);
  let patch_velocity = velocity + cross(angular, arm);
  let normal = sample.normal;
  let along = safe_normalize(forward_axis - normal * dot(forward_axis, normal), forward_axis);
  let across = safe_normalize(cross(along, normal), right_axis);
@@CAGE_STORES@@
}

@compute @workgroup_size(@@LANES@@, 1, 1)
fn vehicle_terrain_contact_geometry(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lane = gid.x; if (lane >= @@LANES@@u) { return; }
  let position = vec3<f32>(@@POSITION@@);
  let velocity = vec3<f32>(@@VELOCITY@@);
  let roll = @@ROLL@@; let pitch = @@PITCH@@; let yaw = @@YAW@@;
  if (lane >= @@WHEEL_LANES@@u) {
    vehicle_cage_patch_contact(lane, position, velocity, roll, pitch, yaw);
    return;
  }
  let compression = vehicle_feed[@@COMPRESSION_BASE@@u + lane];
  var hub_local = vec3<f32>(0.0f);
@@HUB_CASES@@
  let hub_offset = rotate_body(hub_local, roll, pitch, yaw);
  let world_hub = position + hub_offset;
  let forward_axis = rotate_body(vec3<f32>(1.0f, 0.0f, 0.0f), roll, pitch, yaw);
  let right_axis = rotate_body(vec3<f32>(0.0f, 0.0f, 1.0f), roll, pitch, yaw);
  let down = rotate_body(vec3<f32>(0.0f, -1.0f, 0.0f), roll, pitch, yaw);
  let pitch_axis = vec3<f32>(-sin(yaw), 0.0f, cos(yaw));
  let angular = forward_axis * @@ROLL_VELOCITY@@ + vec3<f32>(0.0f, @@YAW_VELOCITY@@, 0.0f)
    + pitch_axis * @@PITCH_VELOCITY@@;
  let hub_velocity = velocity + cross(angular, hub_offset);
  // Per-wheel knuckle coordinates are solved from column -> pinion -> rack ->
  // spherical tie-rod length constraints.  The contact frame consumes that
  // joint coordinate directly; it is not a cosmetic steering transform.
  let steer = controls[25u + lane];
  let rolling_axis = forward_axis * cos(steer) + right_axis * sin(steer);
  let axle = right_axis * cos(steer) - forward_axis * sin(steer);
  var best_score = -1.0e20f; var surface_point = world_hub + down * @@TIRE_RADIUS@@;
  var surface_normal = vec3<f32>(0.0f, 1.0f, 0.0f); var support = 0.0f;
  var support_position = position; var tire_radial_compression = 0.0f;
  // Exact torus-centerline contact arc against the local terrain plane.  The
  // major ring is continuous: the active interval comes from one concentric
  // ring inequality and its penetration moment is the closed form emitted by
  // symbolic_torus_plane_contact_arc_equations.  These slots remain only as
  // diagnostic transport; they are no longer terrain probes.
  for (var diagnostic_index = 0u; diagnostic_index < 15u; diagnostic_index += 1u) {
    radial_probes[lane * 15u + diagnostic_index] = 0.0f;
  }
  let plane_sample = terrain_sample(world_hub.x, world_hub.z, world_hub.y);
  if (plane_sample.valid != 0u) {
    var candidate_normal = safe_normalize(plane_sample.normal, vec3<f32>(0.0f, 1.0f, 0.0f));
    var plane_point = vec3<f32>(world_hub.x, plane_sample.height, world_hub.z);
    let axle_dot_normal = clamp(dot(axle, candidate_normal), -1.0f, 1.0f);
    let plane_radial_projection = sqrt(max(0.0f, 1.0f - axle_dot_normal * axle_dot_normal));
    let projected_major_radius = max(1.0e-8f, @@TIRE_MAJOR_RADIUS@@ * plane_radial_projection);
    let normal_distance = dot(world_hub - plane_point, candidate_normal);
    let boundary_cosine = clamp((@@TIRE_SECTION_RADIUS@@ - normal_distance)
      / projected_major_radius, -1.0f, 1.0f);
    let arc_start = acos(boundary_cosine);
    let contact_arc_angle = 2.0f * (3.141592653589793f - arc_start);
    let boundary_sine = sqrt(max(0.0f, 1.0f - boundary_cosine * boundary_cosine));
    let integrated_arc_penetration = max(0.0f,
      2.0f * (3.141592653589793f - arc_start)
        * (@@TIRE_SECTION_RADIUS@@ - normal_distance)
      + 2.0f * projected_major_radius * boundary_sine);
    var peak_arc_penetration = max(0.0f,
      @@TIRE_SECTION_RADIUS@@ - normal_distance + projected_major_radius);
    let radial_toward_plane = safe_normalize(
      -candidate_normal + axle * axle_dot_normal, down);
    let ring_support_point = world_hub
      + radial_toward_plane * @@TIRE_MAJOR_RADIUS@@
      - candidate_normal * @@TIRE_SECTION_RADIUS@@;
    let next_ring_support_point = ring_support_point + hub_velocity * @@FIXED_DT@@;
    let temporal_hit = terrain_segment_crossing(ring_support_point, next_ring_support_point);
    var evaluation_position = position;
    if (temporal_hit.valid != 0u) {
      candidate_normal = safe_normalize(temporal_hit.normal, candidate_normal);
      plane_point = temporal_hit.point;
      peak_arc_penetration = max(peak_arc_penetration,
        max(0.0f, -dot(next_ring_support_point - temporal_hit.point, candidate_normal)));
      evaluation_position = position + velocity * (@@FIXED_DT@@ * temporal_hit.fraction);
    }
    let upward_facing = dot(candidate_normal, -down) > 0.04f;
    let within_reach = normal_distance <= projected_major_radius
      + @@TIRE_SECTION_RADIUS@@ + @@CONTACT_REACH@@;
    if (upward_facing && within_reach
        && (peak_arc_penetration > 0.0f || temporal_hit.valid != 0u)) {
      surface_normal = candidate_normal;
      surface_point = ring_support_point
        - candidate_normal * dot(ring_support_point - plane_point, candidate_normal);
      support_position = evaluation_position;
      tire_radial_compression = peak_arc_penetration;
      radial_probes[lane * 15u] = contact_arc_angle;
      radial_probes[lane * 15u + 1u] = integrated_arc_penetration;
      radial_probes[lane * 15u + 2u] = integrated_arc_penetration
        / max(contact_arc_angle, 1.0e-8f);
      best_score = peak_arc_penetration;
    }
  }
  // A tire is a short capsule along its axle.  This radial closest-point
  // query lets the same tire touch wall faces and edges; drivetrain torque
  // then acts in the wall tangent, allowing a climb only when geometry,
  // available torque, normal load and Coulomb friction permit it.
  let wall_count = u32(terrain_parameters[1u]);
  for (var wall_index = 0u; wall_index < wall_count; wall_index += 1u) {
    let base = wall_index * 6u;
    let wall_minimum = vec3<f32>(wall_colliders[base], wall_colliders[base + 1u], wall_colliders[base + 2u]);
    let wall_maximum = vec3<f32>(wall_colliders[base + 3u], wall_colliders[base + 4u], wall_colliders[base + 5u]);
    let broadphase_reach = vec3<f32>(@@TIRE_RADIUS@@ + @@TIRE_WIDTH@@ + @@CONTACT_REACH@@);
    if (any(world_hub < wall_minimum - broadphase_reach) || any(world_hub > wall_maximum + broadphase_reach)) {
      continue;
    }
    for (var lateral_index = 0u; lateral_index < 3u; lateral_index += 1u) {
      let slice_hub = world_hub + axle * (@@TIRE_WIDTH@@ * LATERAL_FRACTIONS[lateral_index]);
      var point = clamp(slice_hub, wall_minimum, wall_maximum);
      let outside_delta = slice_hub - point; let outside_distance = length(outside_delta);
      var candidate_normal = safe_normalize(outside_delta, vec3<f32>(0.0f, 1.0f, 0.0f));
      var normal_distance = outside_distance;
      if (outside_distance <= 1.0e-6f) {
        var face_distance = slice_hub.x - wall_minimum.x;
        candidate_normal = vec3<f32>(-1.0f, 0.0f, 0.0f); point.x = wall_minimum.x;
        if (wall_maximum.x - slice_hub.x < face_distance) {
          face_distance = wall_maximum.x - slice_hub.x; candidate_normal = vec3<f32>(1.0f, 0.0f, 0.0f); point.x = wall_maximum.x;
        }
        if (slice_hub.y - wall_minimum.y < face_distance) {
          face_distance = slice_hub.y - wall_minimum.y; candidate_normal = vec3<f32>(0.0f, -1.0f, 0.0f); point = vec3<f32>(slice_hub.x, wall_minimum.y, slice_hub.z);
        }
        if (wall_maximum.y - slice_hub.y < face_distance) {
          face_distance = wall_maximum.y - slice_hub.y; candidate_normal = vec3<f32>(0.0f, 1.0f, 0.0f); point = vec3<f32>(slice_hub.x, wall_maximum.y, slice_hub.z);
        }
        if (slice_hub.z - wall_minimum.z < face_distance) {
          face_distance = slice_hub.z - wall_minimum.z; candidate_normal = vec3<f32>(0.0f, 0.0f, -1.0f); point = vec3<f32>(slice_hub.x, slice_hub.y, wall_minimum.z);
        }
        if (wall_maximum.z - slice_hub.z < face_distance) {
          face_distance = wall_maximum.z - slice_hub.z; candidate_normal = vec3<f32>(0.0f, 0.0f, 1.0f); point = vec3<f32>(slice_hub.x, slice_hub.y, wall_maximum.z);
        }
        normal_distance = -face_distance;
      }
      let penetration = @@TIRE_RADIUS@@ - normal_distance;
      let score = penetration + max(0.0f, dot(candidate_normal, -down)) * 0.003f;
      if (penetration >= -0.006f && score > best_score) {
        best_score = score; surface_point = point; surface_normal = candidate_normal;
        support_position = position; tire_radial_compression = max(0.0f, penetration);
      }
    }
  }
  let suspension_down = down;
  let suspension_alignment = clamp(-dot(suspension_down, surface_normal), 0.0f, 1.0f);
  let origin_to_surface = surface_point - support_position; let distance_along = dot(origin_to_surface, suspension_down);
  let geometric_compression = clamp(@@CLEARANCE_REST@@ - distance_along, 0.0f, @@SUSPENSION_TRAVEL@@)
    * suspension_alignment;
  if (best_score > -1.0e19f && (distance_along > 0.0f || tire_radial_compression > 0.0f)) {
    support = max(c2_unit(geometric_compression / 0.025f), c2_unit(tire_radial_compression / 0.012f));
  }
  let radial_out = safe_normalize(surface_point - world_hub, down);
  let rolling_raw = cross(axle, radial_out); let tangent_raw = rolling_raw - surface_normal * dot(rolling_raw, surface_normal);
  let rolling = safe_normalize(tangent_raw, rolling_axis); let right = safe_normalize(cross(rolling, surface_normal), axle);
  let attachment = surface_point - position - rotate_body(@@CENTER_OF_MASS@@, roll, pitch, yaw);
  let point_velocity = velocity + cross(angular, attachment);
  let tire_radial_velocity = dot(point_velocity, surface_normal);
  let slip_longitudinal = dot(point_velocity, rolling) - vehicle_feed[@@WHEEL_OMEGA_BASE@@u + lane] * @@TIRE_RADIUS@@;
  let slip_lateral = dot(point_velocity, right);
@@STORES@@
}'''
    # The cage patches in chassis-local space: every roll-cage node, then the
    # midpoint of every cage member, in the same order the contact axis lists
    # them.  A bar landing flat is caught by its middle and not only by the
    # joints at either end.
    node_positions = {
        node["identity"]: [float(value) for value in node["reference_position"]]
        for node in graph["nodes"]
    }
    cage_locals: list[list[float]] = []
    for kind, identity in contact_patch_lanes():
        if kind == "cage-node":
            cage_locals.append(node_positions[identity])
        elif kind == "cage-member":
            edge = next(item for item in graph["edges"] if item["identity"] == identity)
            first, second = node_positions[edge["a"]], node_positions[edge["b"]]
            cage_locals.append([(a + b) * .5 for a, b in zip(first, second)])
        elif kind == "body-shell-node":
            cage_locals.append(node_positions[identity])
    solid = source["solid_contact"]
    cage_corner_weight = float(load_audit["total_mass_kg"]) * abs(
        float(source["world"]["gravity"])) / max(1, len(cage_locals))
    # The cage rides the law's radial branch: alignment zero shuts the
    # suspension term off, so squash against the shell answers with the same
    # one-sided contact and the same saturating friction the tyres use.
    cage_rows = {
        # No active damper behind a cage tube: zero gains and a unit scale band
        # make that branch of the law the identity.
        "active_damping_body_velocity_gain_s_per_m": "0.0f",
        "active_damping_maximum_scale": "1.0f",
        "active_damping_minimum_scale": "1.0f",
        "active_damping_rebound_release_gain_s_per_m": "0.0f",
        "attachment_x": "arm.x", "attachment_y": "arm.y", "attachment_z": "arm.z",
        "chassis_velocity_y": vehicle("velocity_y"),
        "compression_velocity": "0.0f",
        "corner_front_sign": "select(-1.0f, 1.0f, patch_local.x >= 0.0f)",
        "corner_side_sign": "select(-1.0f, 1.0f, patch_local.z >= 0.0f)",
        "corner_weight": _wgsl_number(cage_corner_weight),
        "dt": _wgsl_number(1 / float(source["world"]["fixed_step_hz"])),
        "forward_x": "along.x", "forward_y": "along.y", "forward_z": "along.z",
        "geometric_compression": "0.0f",
        "linkage_motion_ratio": "1.0f",
        "load_sensitivity": "0.075f",
        "radial_carcass_loss": (f"select({_wgsl_number(float(solid['cage_contact_damping']))}, "
                                  f"{_wgsl_number(float(source['body_shell']['contact_damping_n_s_per_m']))}, is_shell)"),
        "tire_radial_effective_mass": _wgsl_number(max(1.0, cage_corner_weight / 9.81)),
        "sidewall_deformation_longitudinal": "dot(patch_velocity, along) / 50.0f",
        "sidewall_deformation_velocity_longitudinal": "0.0f",
        "sidewall_deformation_lateral": "dot(patch_velocity, across) / 50.0f",
        "sidewall_deformation_velocity_lateral": "0.0f",
        "sidewall_shear_stiffness_longitudinal": _wgsl_number(float(solid["cage_contact_stiffness"]) * 15),
        "sidewall_shear_stiffness_lateral": _wgsl_number(float(solid["cage_contact_stiffness"]) * 15),
        "sidewall_shear_damping": "0.0f",
        "maximum_compression_speed": "1.25f",
        "bump_stop_stiffness": "0.0f",
        "bump_stop_progressive_stiffness": "0.0f",
        "bump_stop_damping": "0.0f",
        "maximum_contact_area": "0.006f",
        "minimum_contact_area": "0.0008f",
        "mu_kinetic": _wgsl_number(float(solid["cage_kinetic_friction"])),
        "mu_static": _wgsl_number(float(solid["cage_static_friction"])),
        "normal_x": "normal.x", "normal_y": "normal.y", "normal_z": "normal.z",
        "pitch_velocity": vehicle("pitch_velocity"),
        "pneumatic_compression_damping": "0.0f",
        "pneumatic_efficiency": "0.96f",
        "pneumatic_rebound_damping": "0.0f",
        "previous_compression": "0.0f",
        "right_x": "across.x", "right_y": "across.y", "right_z": "across.z",
        "roll_velocity": vehicle("roll_velocity"),
        "slip_lateral": "dot(patch_velocity, across)",
        "slip_longitudinal": "dot(patch_velocity, along)",
        "slip_transition_speed": "0.42f",
        "spring_stiffness": "0.0f",
        "support": "touching",
        "suspension_alignment": "0.0f",
        "suspension_travel": _wgsl_number(suspension["travel"]),
        "tire_pressure": _wgsl_number(900000.0),
        "tire_major_radius": "0.025f",
        "tire_section_radius": "0.025f",
        "tire_effective_tread_width": "0.050f",
        "tire_reference_volume": "0.00031f",
        "tire_gas_polytropic_exponent": "1.0f",
        "tire_radial_compression": "min(max(0.0f, squash), 0.05f) * touching",
        "tire_radial_velocity": "clamp(dot(patch_velocity, normal), -2.0f, 2.0f)",
        "track_half_width": _wgsl_number(wheels["track_half_width"]),
        "wheelbase_half_length": _wgsl_number(wheels["wheelbase_half_length"]),
    }
    missing = [name for name in contact_inputs if name not in cage_rows]
    if missing:
        raise RuntimeError(f"cage contact row is missing inputs: {missing}")
    cage_stores = "\n".join(cage_store(name, cage_rows[name]) for name in contact_inputs)
    replacements = {
        "@@LANES@@": str(lanes),
        "@@WHEEL_LANES@@": str(len(WHEEL_NAMES)),
        "@@CAGE_RADIUS@@": _wgsl_number(float(solid["cage_contact_radius"])),
        "@@SHELL_RADIUS@@": _wgsl_number(float(source["body_shell"]["contact_radius_m"])),
        "@@SHELL_LANE_START@@": str(next(index for index, lane in enumerate(contact_patch_lanes())
                                           if lane[0] == "body-shell-node")),
        "@@CAGE_MAXIMUM_SQUASH@@": _wgsl_number(float(tires["radius"])),
        "@@CAGE_PATCH_LOCAL@@": ", ".join(
            "vec3<f32>(" + ", ".join(_wgsl_number(value) for value in local) + ")"
            for local in cage_locals
        ),
        "@@CAGE_PATCH_COUNT@@": str(len(cage_locals)),
        "@@CAGE_STORES@@": cage_stores,
        "@@CORNER_WEIGHTS@@": ", ".join(_wgsl_number(value) for value in corner_weights),
        "@@POSITION@@": ", ".join(vehicle(f"position_{axis}") for axis in "xyz"),
        "@@VELOCITY@@": ", ".join(vehicle(f"velocity_{axis}") for axis in "xyz"),
        "@@ROLL@@": vehicle("roll"), "@@PITCH@@": vehicle("pitch"), "@@YAW@@": vehicle("yaw"),
        "@@ROLL_VELOCITY@@": vehicle("roll_velocity"),
        "@@PITCH_VELOCITY@@": vehicle("pitch_velocity"),
        "@@YAW_VELOCITY@@": vehicle("yaw_velocity"),
        "@@COMPRESSION_BASE@@": str(vehicle_index["compression_front_left"]),
        "@@WHEEL_OMEGA_BASE@@": str(vehicle_index["wheel_omega_front_left"]),
        "@@HUB_CASES@@": "\n".join(hub_cases),
        "@@MAX_STEER@@": _wgsl_number(float(source["controls"]["maximum_steering_angle_degrees"]) * math.pi / 180),
        "@@TIRE_RADIUS@@": _wgsl_number(tires["radius"]),
        "@@TIRE_MAJOR_RADIUS@@": _wgsl_number(
            tires["radius"] - tires["toroid_section_radius_m"]),
        "@@TIRE_SECTION_RADIUS@@": _wgsl_number(
            tires["toroid_section_radius_m"]),
        "@@TIRE_WIDTH@@": _wgsl_number(tires["width"]),
        "@@CONTACT_REACH@@": _wgsl_number(tires["radius"] + suspension["travel"] + .025),
        "@@FIXED_DT@@": _wgsl_number(1 / (float(source["world"]["fixed_step_hz"]) * 3.0)),
        "@@CLEARANCE_REST@@": _wgsl_number(chassis["clearance"] + suspension["rest_length"]),
        "@@SUSPENSION_TRAVEL@@": _wgsl_number(suspension["travel"]),
        "@@CENTER_OF_MASS@@": "vec3<f32>(" + ", ".join(
            _wgsl_number(value) for value in load_audit["center_of_mass"]
        ) + ")",
        "@@STORES@@": stores,
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    if "@@" in template:
        raise RuntimeError("unresolved vehicle terrain WGSL template marker")
    return {
        "source_language": "python-graph-bake-plus-purpose-wgsl-indexed-gather",
        "authority": "gpu-height-field-and-json-mechanical-graph",
        "terrain_upload_policy": "initialization-or-authored-height-edit-only",
        "state_policy": "read-gpu-resident-packed-vehicle-feed",
        "shell_lane_start": next(index for index, lane in enumerate(contact_patch_lanes())
                                 if lane[0] == "body-shell-node"),
        "shell_lane_count": sum(1 for lane in contact_patch_lanes() if lane[0] == "body-shell-node"),
        "inputs": list(contact_inputs),
        "kernel": {
            "source": template, "entrypoint": "vehicle_terrain_contact_geometry",
            "workgroup_size": [lanes, 1, 1], "dispatch": [1, 1, 1], "invocations": lanes,
            "lane_mapping": [identity for _, identity in contact_patch_lanes()],
            "bindings": ["terrain_heights", "terrain_parameters", "vehicle_feed", "contact_feed", "controls",
            "wall_colliders", "radial_probes"],
        },
        "terrain_parameter_abi": [
            "field_count", "wall_count",
            "field[12]:origin_xyz,cell_xz,columns,rows,domain_xxzz,height_offset",
        ],
    }


def _vehicle_gpu_graph_adapters(
    contact_inputs: tuple[str, ...], contact_outputs: tuple[str, ...],
    vehicle_inputs: tuple[str, ...], vehicle_outputs: tuple[str, ...],
    vehicle_output_slots: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Generate storage-to-storage graph edges around compiler kernels."""

    ci = {name: index for index, name in enumerate(contact_inputs)}
    co = {name: index for index, name in enumerate(contact_outputs)}
    vi = {name: index for index, name in enumerate(vehicle_inputs)}
    lanes = contact_lane_count()
    vo = dict(vehicle_output_slots or {name: index for index, name in enumerate(vehicle_outputs)})
    wheel_lines: list[str] = []
    for lane, wheel in enumerate(WHEEL_NAMES):
        force = [f"contact_outputs[{co[f'chassis_force_{axis}']}u * {lanes}u + {lane}u]" for axis in "xyz"]
        normal = [f"contact_feed[{ci[f'normal_{axis}']}u * {lanes}u + {lane}u]" for axis in "xyz"]
        forward = [f"contact_feed[{ci[f'forward_{axis}']}u * {lanes}u + {lane}u]" for axis in "xyz"]
        dot_force = lambda basis: " + ".join(f"({force[index]} * {basis[index]})" for index in range(3))
        normal_load = f"max(0.0f, {dot_force(normal)})"
        force_squared = " + ".join(f"({value} * {value})" for value in force)
        tangent_squared = (
            f"max(0.0f, ({force_squared}) - "
            f"normal_load_{lane} * normal_load_{lane})"
        )
        mu = f"contact_feed[{ci['mu_static']}u * {lanes}u + {lane}u]"
        contact_radius = (
            f"(contact_feed[{ci['tire_major_radius']}u * {lanes}u + {lane}u] + "
            f"contact_feed[{ci['tire_section_radius']}u * {lanes}u + {lane}u])"
        )
        wheel_lines.extend([
            # The reduced contact wrench already carries the complete force
            # into the chassis graph.  Feeding the longitudinal projection a
            # second time would either be dead state or double-count contact;
            # only its moment at the tire radius belongs on the wheel graph.
            f"  vehicle_feed[{vi[f'tire_reaction_torque_{wheel}']}u] = finite_or(({dot_force(forward)}) * {contact_radius}, 0.0f);",
            f"  vehicle_feed[{vi[f'slip_longitudinal_{wheel}']}u] = "
            f"finite_or(contact_feed[{ci['slip_longitudinal']}u * {lanes}u + {lane}u], 0.0f);",
            f"  vehicle_feed[{vi[f'slip_lateral_{wheel}']}u] = "
            f"finite_or(contact_feed[{ci['slip_lateral']}u * {lanes}u + {lane}u], 0.0f);",
            f"  vehicle_feed[{vi[f'target_compression_{wheel}']}u] = "
            f"finite_or(contact_feed[{ci['geometric_compression']}u * {lanes}u + {lane}u], 0.0f);",
            f"  vehicle_feed[{vi[f'wheel_support_{wheel}']}u] = "
            f"finite_or(contact_feed[{ci['support']}u * {lanes}u + {lane}u], 0.0f);",
            f"  vehicle_feed[{vi[f'linkage_motion_ratio_{wheel}']}u] = "
            f"finite_or(contact_feed[{ci['linkage_motion_ratio']}u * {lanes}u + {lane}u], 1.0f);",
            f"  let normal_load_{lane} = {normal_load};",
            f"  vehicle_feed[{vi[f'contact_normal_force_{wheel}']}u] = finite_or(normal_load_{lane}, 0.0f);",
            f"  let normal_force_{lane} = vec3<f32>({normal[0]}, {normal[1]}, {normal[2]}) * normal_load_{lane};",
            f"  chassis_force -= normal_force_{lane};",
            f"  let attachment_{lane} = vec3<f32>("
            f"contact_feed[{ci['attachment_x']}u * {lanes}u + {lane}u], "
            f"contact_feed[{ci['attachment_y']}u * {lanes}u + {lane}u], "
            f"contact_feed[{ci['attachment_z']}u * {lanes}u + {lane}u]);",
            f"  chassis_torque -= cross(attachment_{lane}, normal_force_{lane});",
            f"  vehicle_feed[{vi[f'measured_friction_utilization_{wheel}']}u] = "
            f"finite_or(sqrt({tangent_squared}) / max(1.0e-5f, {mu} * normal_load_{lane}), 0.0f);",
        ])
    assembly = f'''@group(0) @binding(0) var<storage, read> contact_outputs: array<f32>;
@group(0) @binding(1) var<storage, read> reduced_wrench: array<f32>;
@group(0) @binding(2) var<storage, read> contact_feed: array<f32>;
@group(0) @binding(3) var<storage, read_write> vehicle_feed: array<f32>;
@group(0) @binding(4) var<storage, read> controls: array<f32>;
// NaN is the only float unequal to itself under IEEE-754, on every conformant
// GPU without exception -- unlike min()/max(), which rely on NaN-suppressing
// semantics some hardware does not implement.  This is the one place contact
// data crosses into the vehicle's own state; scrubbing here keeps a stray NaN
// on any one wheel from poisoning the whole integration.
fn finite_or(value: f32, fallback: f32) -> f32 {{
  return select(fallback, value, value == value);
}}
@compute @workgroup_size(1, 1, 1)
fn assemble_vehicle_graph_inputs(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x != 0u) {{ return; }}
  var chassis_force = vec3<f32>(finite_or(reduced_wrench[0u], 0.0f),
    finite_or(reduced_wrench[1u], 0.0f), finite_or(reduced_wrench[2u], 0.0f));
  var chassis_torque = vec3<f32>(finite_or(reduced_wrench[3u], 0.0f),
    finite_or(reduced_wrench[4u], 0.0f), finite_or(reduced_wrench[5u], 0.0f));
  vehicle_feed[{vi['yaw_cos']}u] = cos(vehicle_feed[{vi['yaw']}u]);
  vehicle_feed[{vi['yaw_sin']}u] = sin(vehicle_feed[{vi['yaw']}u]);
  vehicle_feed[{vi['throttle']}u] = controls[0u];
  vehicle_feed[{vi['brake']}u] = controls[2u];
  vehicle_feed[{vi['forward_gear_ratio']}u] = controls[3u];
  vehicle_feed[{vi['reverse_gear_ratio']}u] = controls[4u];
  vehicle_feed[{vi['transfer_case_ratio']}u] = controls[5u];
  vehicle_feed[{vi['front_differential_lock']}u] = controls[6u];
  vehicle_feed[{vi['rear_differential_lock']}u] = controls[13u];
  vehicle_feed[{vi['traction_control_enabled']}u] = controls[15u];
  vehicle_feed[{vi['abs_enabled']}u] = controls[16u];
  vehicle_feed[{vi['traction_control_authority']}u] = controls[19u];
  vehicle_feed[{vi['abs_authority']}u] = controls[20u];
  vehicle_feed[{vi['center_differential_lock']}u] = controls[21u];
  vehicle_feed[{vi['front_differential_brake']}u] = controls[22u];
  vehicle_feed[{vi['rear_differential_brake']}u] = controls[23u];
  vehicle_feed[{vi['total_force_x']}u] = controls[30u];
  vehicle_feed[{vi['total_force_y']}u] = controls[31u];
  vehicle_feed[{vi['total_force_z']}u] = controls[32u];
  vehicle_feed[{vi['total_torque_x']}u] = controls[33u];
  vehicle_feed[{vi['total_torque_y']}u] = controls[34u];
  vehicle_feed[{vi['total_torque_z']}u] = controls[35u];
  vehicle_feed[{vi['drive_direction']}u] = controls[7u];
  vehicle_feed[{vi['brake_lock_front_left']}u] = controls[8u];
  vehicle_feed[{vi['brake_lock_front_right']}u] = controls[9u];
  vehicle_feed[{vi['brake_lock_rear_left']}u] = controls[10u];
  vehicle_feed[{vi['brake_lock_rear_right']}u] = controls[11u];
  let front_share = select(clamp(controls[12u], 0.05f, 0.95f), 0.5f, controls[14u] > 0.5f);
  vehicle_feed[{vi['drive_fraction_front_left']}u] = front_share * 0.5f;
  vehicle_feed[{vi['drive_fraction_front_right']}u] = front_share * 0.5f;
  vehicle_feed[{vi['drive_fraction_rear_left']}u] = (1.0f - front_share) * 0.5f;
  vehicle_feed[{vi['drive_fraction_rear_right']}u] = (1.0f - front_share) * 0.5f;
{chr(10).join(wheel_lines)}
  vehicle_feed[{vi['contact_wrench_force_x']}u] = chassis_force.x;
  vehicle_feed[{vi['contact_wrench_force_y']}u] = chassis_force.y;
  vehicle_feed[{vi['contact_wrench_force_z']}u] = chassis_force.z;
  vehicle_feed[{vi['contact_wrench_torque_x']}u] = chassis_torque.x;
  vehicle_feed[{vi['contact_wrench_torque_y']}u] = chassis_torque.y;
  vehicle_feed[{vi['contact_wrench_torque_z']}u] = chassis_torque.z;
}}'''
    commits: list[str] = []
    direct = {
        **{f"position_{axis}_next": f"position_{axis}" for axis in "xyz"},
        **{f"velocity_{axis}_next": f"velocity_{axis}" for axis in "xyz"},
        "roll_next": "roll", "pitch_next": "pitch", "yaw_next": "yaw",
        "roll_velocity_next": "roll_velocity", "pitch_velocity_next": "pitch_velocity",
        "yaw_velocity_next": "yaw_velocity",
        **{f"wheel_omega_{wheel}_next": f"wheel_omega_{wheel}" for wheel in WHEEL_NAMES},
        **{f"wheel_angle_{wheel}_next": f"wheel_angle_{wheel}" for wheel in WHEEL_NAMES},
        **{f"slip_longitudinal_{wheel}_next": f"previous_slip_longitudinal_{wheel}"
           for wheel in WHEEL_NAMES},
        **{f"slip_sensor_velocity_{wheel}_next": f"slip_sensor_velocity_{wheel}"
           for wheel in WHEEL_NAMES},
        **{f"friction_utilization_{wheel}_next": f"friction_utilization_{wheel}"
           for wheel in WHEEL_NAMES},
        **{f"friction_utilization_sensor_velocity_{wheel}_next":
           f"friction_utilization_sensor_velocity_{wheel}" for wheel in WHEEL_NAMES},
        **{f"tire_deformation_longitudinal_{wheel}_next": f"tire_deformation_longitudinal_{wheel}"
           for wheel in WHEEL_NAMES},
        **{f"tire_deformation_velocity_longitudinal_{wheel}_next":
           f"tire_deformation_velocity_longitudinal_{wheel}" for wheel in WHEEL_NAMES},
        **{f"tire_deformation_lateral_{wheel}_next": f"tire_deformation_lateral_{wheel}"
           for wheel in WHEEL_NAMES},
        **{f"tire_deformation_velocity_lateral_{wheel}_next":
           f"tire_deformation_velocity_lateral_{wheel}" for wheel in WHEEL_NAMES},
        **{f"compression_{wheel}_next": f"compression_{wheel}" for wheel in WHEEL_NAMES},
        **{f"compression_velocity_{wheel}_next": f"compression_velocity_{wheel}"
           for wheel in WHEEL_NAMES},
        "engine_angular_speed_next": "engine_angular_speed",
    }
    for output, feed in direct.items():
        # This is the vehicle's own persistent state, read back as "now" by
        # every kernel next tick.  A single non-finite output here does not
        # cost one bad frame -- it becomes the truck's new reality forever,
        # since nothing downstream has any way to know it was garbage.
        # Holding the previous value on a bad frame is what makes a single
        # stray NaN survivable instead of permanent.
        commits.append(
            f"  vehicle_feed[{vi[feed]}u] = "
            f"finite_or(vehicle_outputs[{vo[output]}u], vehicle_feed[{vi[feed]}u]);"
        )
    commit = f'''@group(0) @binding(0) var<storage, read> vehicle_outputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> vehicle_feed: array<f32>;
fn finite_or(value: f32, fallback: f32) -> f32 {{
  return select(fallback, value, value == value);
}}
@compute @workgroup_size(1, 1, 1)
fn commit_vehicle_graph_state(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x != 0u) {{ return; }}
{chr(10).join(commits)}
}}'''
    return {
        "schema": "abstract-ui-vehicle-gpu-graph-adapters-v0",
        "authority": "compiler-published-packed-abis",
        "control_abi": ["throttle", "steering", "brake", "forward_gear_ratio",
                        "reverse_gear_ratio", "transfer_case_ratio", "front_differential_lock",
                        "drive_direction", "brake_lock_front_left", "brake_lock_front_right",
                        "brake_lock_rear_left", "brake_lock_rear_right", "front_drive_share",
                        "rear_differential_lock", "center_transfer_lock",
                        "traction_control_enabled", "abs_enabled",
                        "front_knuckle_steer_angle", "rear_knuckle_steer_angle",
                        "traction_control_authority", "abs_authority",
                        "center_differential_coupling",
                        "front_differential_brake", "rear_differential_brake", "body_shell_active",
                        "front_left_knuckle_angle", "front_right_knuckle_angle",
                        "rear_left_knuckle_angle", "rear_right_knuckle_angle", "tire_pressure_pa",
                        "outrigger_force_x", "outrigger_force_y", "outrigger_force_z",
                        "outrigger_torque_x", "outrigger_torque_y", "outrigger_torque_z"],
        "assembly": {
            "source": assembly, "entrypoint": "assemble_vehicle_graph_inputs",
            "workgroup_size": [1, 1, 1], "dispatch": [1, 1, 1],
            "bindings": ["contact_outputs", "reduced_wrench", "contact_feed", "vehicle_feed", "controls"],
        },
        "commit": {
            "source": commit, "entrypoint": "commit_vehicle_graph_state",
            "workgroup_size": [1, 1, 1], "dispatch": [1, 1, 1],
            "bindings": ["vehicle_outputs", "vehicle_feed"],
        },
        "dispatch_order": ["terrain_contact_geometry", "compiled_contact_law",
                           "backend_gemm_wrench_reduction", "assemble_vehicle_inputs",
                           "compiled_vehicle_transition", "commit_vehicle_state"],
        "host_awaits_between_nodes": 0,
    }


def vehicle_webgpu_program_model(
    config: VehicleConfiguration,
    engine_profiles: tuple[Mapping[str, Any], ...] = (),
) -> dict[str, Any]:
    total_milestones = 7
    _vehicle_build_progress("contact-wgsl", 0, total_milestones, "lowering wheel contact law")
    program = compile_wheel_contact_webgpu()
    _vehicle_build_progress("contact-runtime", 1, total_milestones, "building scalar contact oracle")
    runtime_contact = compile_wheel_contact_abstract_tensor(packed_outputs=False)
    _vehicle_build_progress("contact-tensor", 2, total_milestones, "packing contact tensor program")
    tensor_contact = compile_wheel_contact_abstract_tensor(packed_outputs=True)
    tensor_program = tensor_contact.artifacts[0]
    _vehicle_build_progress("wrench-reduction", 3, total_milestones, "lowering four-corner wrench reduction")
    reduction = compile_vehicle_wrench_reduction_webgpu()
    reduction_program = reduction.artifacts[0]
    _vehicle_build_progress("vehicle-symbolics", 4, total_milestones, "compiling one live-parametric equation set")
    vehicle_compilation = compile_symbolic_vehicle_physics_gpu_ssa()
    _vehicle_build_progress("vehicle-webgpu", 5, total_milestones, "lowering parametric integration stages")
    vehicle_stages = compile_symbolic_vehicle_physics_webgpu_stages()
    vehicle_inputs = tuple(vehicle_compilation.function.metadata["argument_names"])
    # Engine profiles remain durable selectable parameter records.  Do not bake the
    # entire vehicle graph once per engine/mode: every selection feeds this one ABI.
    engine_profile_variants: list[dict[str, Any]] = []
    vehicle_output_slots = {
        name: int(stage["output_offset_floats"]) + index
        for stage in vehicle_stages
        for index, name in enumerate(stage["outputs"])
    }
    vehicle_output_buffer_floats = max(vehicle_output_slots.values()) + 1
    contact_inputs = tuple(tensor_contact.argument_names)
    _vehicle_build_progress("graph-adapters", 6, total_milestones, "assembling resident buffer adapters")
    adapters = _vehicle_gpu_graph_adapters(
        contact_inputs, tuple(tensor_contact.output_names),
        vehicle_inputs, tuple(VEHICLE_STATE_OUTPUTS),
        vehicle_output_slots,
    )
    adapters["dispatch_order"] = ["terrain_contact_geometry", "compiled_contact_law",
                                  "backend_gemm_wrench_reduction", "assemble_vehicle_inputs",
                                  *(f"compiled_vehicle_{stage['identity']}" for stage in vehicle_stages),
                                  "commit_vehicle_state"]
    _vehicle_build_progress("vehicle-model", 7, total_milestones, "parametric vehicle program assembled")
    return {
        "schema": "abstract-ui-vehicle-webgpu-program-v0",
        "identity": f"{config.identity}/programs/wheel-contact-webgpu",
        "source_language": "sympy-equation-set",
        "lowering": ["sympy", "process-graph", "repository-ssa-f32", "webgpu-wgsl"],
        "configuration_digest": config.digest,
        "kernel": {"source": program.source, "entrypoint": "main",
                   "workgroup_size": list(program.launch_plan.workgroup_size),
                   "dispatch": list(program.launch_plan.groups),
                   "invocations": contact_lane_count(),
                   "lane_mapping": [identity for _, identity in contact_patch_lanes()]},
        "inputs": list(runtime_contact.argument_names),
        "outputs": list(runtime_contact.output_names),
        "tensor_contact_precompile": {
            "source_language": "sympy-cse-to-abstract-tensor-python",
            "abstract_tensor_source": tensor_contact.source,
            "inputs": list(tensor_contact.argument_names),
            "outputs": list(tensor_contact.output_names),
            "packed_outputs": tensor_contact.packed_outputs,
            "storage_rows": {name: index for index, name in enumerate(tensor_contact.output_names)},
            "reducer_output_order": list(CONTACT_PATCH_TENSOR_REDUCER_OUTPUTS),
            "precompile_output_translation": "reducer-publications-to-authored-contact-abi",
            "wrench_row_start": tensor_contact.output_names.index("chassis_force_x"),
            "wrench_row_count": 6,
            "kernel": {
                "source": tensor_program.source,
                "entrypoint": "main",
                "workgroup_size": list(tensor_program.launch_plan.workgroup_size),
                "dispatch": list(tensor_program.launch_plan.groups),
                "invocations": contact_lane_count(),
                "io": tensor_program.api.to_mapping()["metadata"]["io_layout"],
                "output_span": tensor_program.api.to_mapping()["metadata"]["output_span"],
            },
        },
        "terrain_contact_geometry": _vehicle_terrain_contact_webgpu(
            config, contact_inputs, vehicle_inputs,
        ),
        "graph_adapters": adapters,
        "wrench_reduction": {
            "source_language": "sympy-matrix-expression-to-abstract-tensor-python",
            "symbolic_source": reduction.symbolic_source,
            "abstract_tensor_source": reduction.source,
            "input_shapes": [list(shape) for shape in reduction.input_shapes],
            "output_shape": list(reduction.output_shape),
            "kernel": {
                "source": reduction_program.source,
                "entrypoint": "main",
                "workgroup_size": list(reduction_program.launch_plan.workgroup_size),
                "dispatch": list(reduction_program.launch_plan.groups),
                "backend_variant": reduction_program.api.metadata.get("variant"),
                "problem_shape": reduction_program.api.metadata.get("problem_shape"),
                "io": reduction_program.api.to_mapping()["metadata"]["io_layout"],
                "scalars": {"alpha": 1.0, "beta": 0.0},
            },
        },
        "vehicle_integration": {
            "source_language": "sympy-equation-set-to-repository-ssa",
            "inputs": list(vehicle_compilation.function.metadata["argument_names"]),
            "outputs": list(VEHICLE_STATE_OUTPUTS),
            "output_slots": vehicle_output_slots,
            "output_buffer_floats": vehicle_output_buffer_floats,
            "state_residency": "gpu-persistent-with-passive-presentation-snapshots",
            "shared_output_buffer": "one-allocation-disjoint-256-byte-aligned-storage-views",
            "host_transfers_between_stages": 0,
            "default_specialization": {
                "policy": "single-live-parametric-kernel/no-vehicle-profile-prebakes",
                "resident_feed_abi": list(vehicle_inputs), "fixed_inputs": [],
                "folded_inputs": [],
                "pipeline_swap_moves_state": False,
            },
            "engine_profile_variants": engine_profile_variants,
            "engine_profile_dispatch": "durable-profile-parameters-feed-single-live-parametric-kernel",
            "stages": [{
                "identity": stage["identity"], "outputs": list(stage["outputs"]),
                "output_offset_floats": int(stage["output_offset_floats"]),
                "kernel": {
                    "source": stage["kernel"].source,
                    "entrypoint": "main",
                    "workgroup_size": list(stage["kernel"].launch_plan.workgroup_size),
                    "dispatch": list(stage["kernel"].launch_plan.groups), "invocations": 1,
                    "io": stage["kernel"].api.to_mapping()["metadata"]["io_layout"],
                    "output_span": stage["kernel"].api.to_mapping()["metadata"]["output_span"],
                },
            } for stage in vehicle_stages],
        },
        "paired_force_rule": "wheel_force_is_exact_negative_of_published_chassis_force",
        "reduction": "AbstractTensor GEMM maps four wheel-wrench rows to one chassis wrench",
        "fallback": "compiled scalar WASM oracle uses the same JSON configuration",
    }


def _vehicle_mechanical_graph(config: VehicleConfiguration) -> dict[str, Any]:
    """Build the explicit mechanical authority graph from the compact JSON parameters.

    Nodes carry wrenches; edges are constraints or power-transfer laws.  The four
    suspension corners are genuine double-wishbone graphs (four control-arm links,
    an upright, hub constraint, coilover, contact patch, and half shaft), rather
    than labels attached to a scalar spring lane.
    """

    source = config.source
    chassis, wheels = source["chassis"], source["wheels"]
    powertrain, suspension = source["powertrain"], source["suspension"]
    drivetrain = source["drivetrain"]
    electrical = source["electrical"]
    service_lines = source["service_lines"]
    mass_properties = config.mass_properties()
    component_masses = {item["identity"]: item["mass_kg"]
                        for item in mass_properties["components"]}
    half_length = float(chassis["half_length"])
    half_width = float(chassis["half_width"])
    wheelbase = float(wheels["wheelbase_half_length"])
    axle_offset = float(wheels["axle_group_offset_x_m"])
    track = float(wheels["track_half_width"])
    hub_face_offset = float(wheels["hub_face_offset"])
    wheel_radius = float(source["tires"]["radius"])
    frame_y = float(chassis["height"]) * .72
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    membranes: list[dict[str, Any]] = []
    tire_skin_abi = balloon_tire_graph_abi(source)
    tire_topology = tire_skin_abi["topology"]
    wheel_mount_solution = solve_vehicle_wheel_placement_mounts(config)
    body_packaging = solve_vehicle_body_packaging(config)
    body_station = body_packaging["longitudinal_stations_m"]
    alignment_strain_relief_definition = {
        "identity": "force-limited-alignment-strain-relief-v1",
        "family": "linear-hydraulic-actuator",
        "kind": "backdrivable-series-elastic-alignment-actuator",
        "state": ["commanded_rest_length_m", "relief_stroke_m", "relief_velocity_m_s",
                  "axial_force_n", "dissipated_energy_j", "temperature_k", "health"],
        "linear_stiffness_n_per_m": float(
            suspension["alignment_strain_relief_stiffness_n_per_m"]),
        "linear_damping_n_s_per_m": float(
            suspension["alignment_strain_relief_damping_n_s_per_m"]),
        "holding_force_n": float(suspension["alignment_strain_relief_holding_force_n"]),
        "relief_force_n": float(suspension["alignment_strain_relief_relief_force_n"]),
        "maximum_relief_stroke_m": float(
            suspension["alignment_strain_relief_maximum_stroke_m"]),
        "maximum_relief_rate_m_per_s": float(
            suspension["alignment_strain_relief_maximum_rate_m_per_s"]),
        "recenter_force_n": float(suspension["alignment_strain_relief_recenter_force_n"]),
        "recenter_rate_m_per_s": float(
            suspension["alignment_strain_relief_recenter_rate_m_per_s"]),
        "mass_kg": float(suspension["alignment_strain_relief_actuator_mass_kg_each"]),
        "energy_law": (
            "commanded-length-plus-series-relief-stroke; pressure-relief work becomes heat; "
            "powered recentering draws from the alignment hydraulic manifold"),
        "high_current_position_command_supported": False,
        "high_current_reason": (
            "alignment links own toe/camber geometry and a sacrificial relief path; "
            "fast vertical trim belongs to the series coilover preload actuator"),
        "failure_mode": "hold-current-total-length-until-repaired-or-break-bushing-opens",
    }
    knuckle_break_bushing_definition = {
        "identity": "replaceable-knuckle-break-bushing-v1",
        "family": "sacrificial-six-axis-junction-cartridge",
        "kind": "controlled-failure-knuckle-mechanical-fuse",
        "state": ["elastic_deflection_m", "plastic_set_m", "combined_utilization",
                  "dissipated_energy_j", "failed"],
        "yield_force_n": float(suspension["knuckle_break_bushing_yield_force_n"]),
        "fracture_force_n": float(suspension["knuckle_break_bushing_fracture_force_n"]),
        "yield_displacement_m": float(
            suspension["knuckle_break_bushing_yield_displacement_m"]),
        "fracture_displacement_m": float(
            suspension["knuckle_break_bushing_fracture_displacement_m"]),
        "yield_moment_nm": float(suspension["knuckle_break_bushing_yield_moment_nm"]),
        "fracture_moment_nm": float(suspension["knuckle_break_bushing_fracture_moment_nm"]),
        "mass_kg": float(suspension["knuckle_break_bushing_mass_kg_each"]),
        "combined_wrench_law": (
            "sqrt((axial/yield-axial)^2+(shear/yield-shear)^2+"
            "(bending/yield-moment)^2+(torsion/yield-moment)^2)"),
        "calibration": "fracture-immediately-before-minimum-protected-member-plastic-onset",
        "failure_response": "connector-constraint-opens-and-knuckle-load-path-is-removed",
        "service": "replaceable-cartridge-may-be-carried-as-massed-cargo",
    }
    rotating_accessory_presets = {
        "differential-port-crawl-flywheel": {
            "mass_kg": 96.0, "radius_m": .42, "polar_inertia_kg_m2": 8.4672,
            "bearing_drag_nm": 2.8, "maximum_speed_rad_s": 180.0,
            "torsional_yield_torque_nm": 5200.0,
            "torsional_fracture_torque_nm": 7600.0,
        },
        "pre-clutch-crank-flywheel": {
            "mass_kg": 54.0, "radius_m": .31, "polar_inertia_kg_m2": 2.5947,
            "bearing_drag_nm": 1.6, "maximum_speed_rad_s": 520.0,
            "torsional_yield_torque_nm": 4600.0,
            "torsional_fracture_torque_nm": 6800.0,
        },
    }
    static_accessory_presets = {
        "configurable-barrel-tank": {
            "mass_kg": 34.0, "radius_m": .22, "cylindrical_length_m": .70,
            "wall_thickness_m": .003, "shell_density_kg_m3": 7850.0,
            "contents_density_kg_m3": 740.0, "fill_fraction": .80,
            "mount_interface": "generic-six-axis-wrench-attachment-v1",
            "state": ["contents_mass_kg", "contents_center_of_mass_local",
                      "contents_angular_momentum", "sloshing_mode_amplitude"],
            "contents_law": "optional-baffled-low-order-free-surface-slosh-appendage",
        },
        "industrial-high-pressure-gas-cylinder": {
            "empty_mass_kg": 58.7, "reference_gas_mass_kg": 11.5,
            "mass_kg": 70.2, "outside_radius_m": .115, "straight_length_m": 1.25,
            "wall_thickness_m": .007, "shell_density_kg_m3": 7850.0,
            "shell_yield_stress_pa": 620_000_000.0,
            "shell_fracture_stress_pa": 760_000_000.0,
            "minimum_burst_safety_factor": 2.25,
            "reference_pressure_pa": 20_000_000.0, "reference_temperature_k": 293.15,
            "molar_mass_kg_per_mol": .0280134,
            "permeability_mol_m_per_m2_s_pa": 0.0,
            "mount_interface": "two-generic-six-axis-wrench-attachment-v1",
            "state": ["gas_mass_kg", "pressure_pa", "temperature_k", "internal_energy_j",
                      "valve_open_fraction", "leak_area_m2", "shell_plastic_strain",
                      "restraint_damage"],
            "pressure_vessel_law": (
                "real-gas-state-plus-thick-wall-hoop-and-longitudinal-stress-with-"
                "temperature-dependent-shell-yield"),
            "rupture_disc_set_pressure_pa": 34_000_000.0,
            "failure_wrenches": {
                "leak": "compressible-choked-or-unchoked-jet-reaction-at-orifice",
                "valve_shear": "full-bore-jet-wrench-and-free-cylinder-rocket-state",
                "rupture": "stored-gas-expansion-energy-plus-shell-fragment-impulses",
            },
        },
        "direct-drive-high-pressure-compressor": {
            "mass_kg": 41.0, "polar_inertia_kg_m2": .086,
            "maximum_speed_rad_s": 420.0, "maximum_pressure_pa": 30_000_000.0,
            "volumetric_displacement_m3_per_rev": 7.5e-5,
            "isentropic_efficiency": .68, "mechanical_efficiency": .91,
            "mount_interface": "shared-engine-accessory-block-v1",
            "drive_interface": "alternator-cvt-shared-torque-summation-v1",
            "eligible_drive_ports": ["electrical.alternator_cvt"],
            "state": ["shaft_angle", "shaft_angular_velocity", "discharge_pressure_pa",
                      "head_temperature_k", "bearing_temperature_k", "wear"],
            "load_law": (
                "positive-displacement-compression-reaction-torque-is-summed-with-"
                "alternator-reaction-torque-before-the-shared-cvt-reacts-on-the-engine"),
        },
        "reversible-accessory-block-motor": {
            "mass_kg": 8.5, "rotor_inertia_kg_m2": .012,
            "continuous_power_w": 6_000.0, "peak_power_w": 11_000.0,
            "continuous_torque_nm": 38.0, "peak_torque_nm": 76.0,
            "maximum_speed_rad_s": 680.0, "drive_efficiency": .89,
            "regeneration_efficiency": .83,
            "mount_interface": "shared-engine-accessory-block-v1",
            "electrical_interface": "fused-bidirectional-dc-bus-v1",
            "state": ["rotor_angle", "rotor_angular_velocity", "winding_temperature_k",
                      "controller_temperature_k", "wear", "commanded_torque_nm"],
            "energy_law": (
                "electrical-bus-power-and-shaft-power-differ-by-directional-efficiency-"
                "and-loss-power-enters-motor-thermal-state"),
        },
        "four-lead-acid-battery-cube": {
            "mass_kg": 72.0, "cell_block_count": 4, "layout": "two-by-two-cube",
            "nominal_voltage_v": 48.0, "capacity_ah": 60.0,
            "capacity_j": 10_368_000.0, "maximum_discharge_current_a": 420.0,
            "maximum_charge_current_a": 120.0, "internal_resistance_ohm": .018,
            "mount_interface": "center-pan-battery-tray-six-axis-wrench-v1",
            "electrical_interface": "fused-bidirectional-dc-bus-v1",
            "state": ["charge_fraction", "terminal_voltage_v", "current_a",
                      "temperature_k", "sulfation", "plate_damage"],
            "energy_law": (
                "four-separately-stateful-lead-acid-blocks-with-busbar-contactor-"
                "and-fuse-losses-no-shaft-torque-without-equal-battery-energy-debit"),
        },
    }

    def node(identity: str, position: list[float], kind: str, **attributes: Any) -> None:
        nodes.append({"identity": identity, "kind": kind, "reference_position": position,
                      "wrench": {"force": [0.0, 0.0, 0.0], "moment": [0.0, 0.0, 0.0]},
                      **attributes})

    def edge(identity: str, a: str, b: str, constraint: str, *, radius: float = .012,
             palette: str = "rollbar-silver", **attributes: Any) -> None:
        pa = next(item["reference_position"] for item in nodes if item["identity"] == a)
        pb = next(item["reference_position"] for item in nodes if item["identity"] == b)
        rest_length = math.sqrt(sum((pa[axis] - pb[axis]) ** 2 for axis in range(3)))
        area = math.pi * radius ** 2
        decorative = constraint.startswith("rigid-lamp") or "headlamp" in identity
        routed_line = constraint in {
            "insulated-copper-wire", "pressure-rated-hydraulic-line",
            "pressure-rated-air-line", "rigid-brake-hard-line",
            "rigid-pneumatic-hard-line", "rigid-alignment-hydraulic-line",
            "flexible-hydraulic-hose", "flexible-brake-hose",
            "flexible-air-line", "flexible-pneumatic-hose",
            "flexible-alignment-hydraulic-hose",
            "drilled-hub-air-passage", "sheathed-parking-brake-cable",
            "pneumatic-valve-to-closed-volume", "annular-bearing-pneumatic-rotary-seal",
            "traditional-rim-service-valve", "tube-stem-to-rim-valve-install-binding",
            "hydraulic-caliper-service-port",
        }
        # Each physical edge owns the compliance and loss at both of its
        # junctions.  These are endpoint bushings, not an extra force applied
        # beside the graph: their damping power is evaluated from the solved
        # relative motion and accumulated by the resident worker.
        frame_mount = a.startswith("frame.") or b.startswith("frame.")
        # Static performance-polyurethane pack.  The values are calculated
        # from one declared annular mount geometry/material so every frame
        # junction receives the same auditable six-axis law rather than a
        # hand-tuned damping number.  They remain named graph parameters even
        # when the selected native build resolves them to constants.
        bushing_outer_radius = .024 if frame_mount else .018
        bushing_inner_radius = .010 if frame_mount else .008
        bushing_length = .038 if frame_mount else .032
        bushing_youngs_modulus = 65.0e6 if frame_mount else 48.0e6
        bushing_shear_modulus = 22.0e6 if frame_mount else 16.5e6
        bushing_area = math.pi * (bushing_outer_radius ** 2 - bushing_inner_radius ** 2)
        bushing_polar_second_moment = math.pi * (
            bushing_outer_radius ** 4 - bushing_inner_radius ** 4) / 2
        bushing_linear_stiffness = bushing_youngs_modulus * bushing_area / bushing_length
        bushing_angular_stiffness = (
            bushing_shear_modulus * bushing_polar_second_moment / bushing_length)
        bushing_damping_ratio = .22 if frame_mount else .18
        bushing_effective_mass = 18.0 if frame_mount else 7.5
        bushing_effective_rotary_inertia = .18 if frame_mount else .055
        bushing = None if decorative or routed_line else {
            "model": "six-axis-kelvin-voigt-junction",
            "parameter_pack": "performance-polyurethane-calculated-static-v2",
            "compile_policy": "parameterized-source-resolved-to-static-kernel-constants",
            "frame_mount": frame_mount,
            "material": {"shore_hardness_a": 92, "youngs_modulus_pa": bushing_youngs_modulus,
                         "shear_modulus_pa": bushing_shear_modulus,
                         "loss_factor": 2 * bushing_damping_ratio},
            "geometry_m": {"outer_radius": bushing_outer_radius,
                           "inner_radius": bushing_inner_radius, "length": bushing_length},
            "linear_stiffness_n_per_m": bushing_linear_stiffness,
            "linear_damping_n_s_per_m": 2 * bushing_damping_ratio * math.sqrt(
                bushing_linear_stiffness * bushing_effective_mass),
            "angular_stiffness_nm_per_rad": bushing_angular_stiffness,
            "angular_damping_nm_s_per_rad": 2 * bushing_damping_ratio * math.sqrt(
                bushing_angular_stiffness * bushing_effective_rotary_inertia),
            "preload_compression_m": .00035 if frame_mount else .00020,
            "preload_force_n": bushing_linear_stiffness * (.00035 if frame_mount else .00020),
            "yield_displacement_m": .0045 if frame_mount else .0060,
            "yield_force_n": bushing_linear_stiffness * (.0045 if frame_mount else .0060),
            "fracture_displacement_m": .014 if frame_mount else .018,
            "fracture_force_n": bushing_linear_stiffness * (.014 if frame_mount else .018),
            "static_friction_torque_nm": 1.8,
            "dissipation": "sum-c-linear-v-relative-squared-and-c-angular-omega-relative-squared",
        }
        damage = None if decorative else {
            "model": "elastic-plastic-member-with-shear-fracture",
            "natural_rest_length": rest_length,
            "plastic_strain_limit": .0025 if constraint != "spring-damper" else .035,
            "fracture_strain": .075 if constraint != "spring-damper" else .24,
            "axial_yield_force_n": area * 250_000_000 * .72,
            "shear_force_limit_n": area * 250_000_000 * .42,
            "failure_response": "constraint-opens-and-load-path-is-removed",
            "respawn_response": "restore-authored-natural-length-and-health",
        }
        edges.append({"identity": identity, "a": a, "b": b, "constraint": constraint,
                      "rest_length": rest_length, "radius": radius, "palette_role": palette,
                      **({"damage": damage} if damage else {}),
                      **({"joint_bushings": {"a": dict(bushing), "b": dict(bushing)}}
                         if bushing else {}), **attributes})

    # Rigid frame: four load nodes, perimeter rails, and both triangulating diagonals.
    for longitudinal, x in (("front", half_length), ("rear", -half_length)):
        for lateral, side in (("left", -1.0), ("right", 1.0)):
            node(f"frame.{longitudinal}_{lateral}", [x, frame_y, side * half_width * .78],
                 "chassis-load-node", fixed_to="chassis", structural_deformable=True,
                 longitudinal_authority="chassis-half-length",
                 chassis_reference_plane_corner=True,
                 reference_identity_persistent_through_deformation=True)
    frame_pairs = (
        ("front_left", "front_right"), ("front_right", "rear_right"),
        ("rear_right", "rear_left"), ("rear_left", "front_left"),
        ("front_left", "rear_right"), ("front_right", "rear_left"),
    )
    for index, (a, b) in enumerate(frame_pairs):
        edge(f"frame.member.{index}", f"frame.{a}", f"frame.{b}", "rigid-distance",
             radius=.018, load_path="chassis-wrench-reduction")

    # A short structural post at every frame corner is the invariant suspension
    # adapter. Suspension families replace complete subgraphs outside this
    # boundary; they do not rewrite the frame or invent unbound pickup points.
    for corner in WHEEL_NAMES:
        post = wheel_mount_solution["standard_corner_posts"][corner]
        lower_identity = f"suspension_mount_post.{corner}.lower"
        upper_identity = f"suspension_mount_post.{corner}.upper"
        node(lower_identity, list(post["lower"]), "modular-suspension-post-terminal",
             fixed_to="chassis", structural_deformable=True,
             compatible_architectures=wheel_mount_solution["compatible_architectures"])
        node(upper_identity, list(post["upper"]), "modular-suspension-post-terminal",
             fixed_to="chassis", structural_deformable=True,
             mass_kg=float(wheel_mount_solution["post_mass_kg_each"]), mass_in_total=True,
             mass_frame="chassis-sprung",
             compatible_architectures=wheel_mount_solution["compatible_architectures"])
        edge(f"suspension_mount_post.{corner}.lower_half", lower_identity,
             f"frame.{corner}", "elastic-plastic-structural-post", radius=float(
                 source["wheel_placement_demands"]["mount_synthesis"]["post_outer_radius_m"]),
             load_path="suspension-bracket-to-chassis-corner")
        edge(f"suspension_mount_post.{corner}.upper_half", f"frame.{corner}",
             upper_identity, "elastic-plastic-structural-post", radius=float(
                 source["wheel_placement_demands"]["mount_synthesis"]["post_outer_radius_m"]),
             load_path="suspension-bracket-to-chassis-corner")

    attachment_parameters = source["wrench_attachments"]
    def attachment_admission(payload: str, peak_force_n: float,
                             peak_moment_nm: float) -> dict[str, Any]:
        force_limit = float(attachment_parameters["maximum_force_n"])
        moment_limit = float(attachment_parameters["maximum_moment_nm"])
        if peak_force_n > force_limit or peak_moment_nm > moment_limit:
            raise ValueError(
                f"wrench attachment rejects {payload}: required force/moment "
                f"({peak_force_n:.6g} N, {peak_moment_nm:.6g} Nm) exceeds "
                f"({force_limit:.6g} N, {moment_limit:.6g} Nm)")
        return {
            "payload": payload, "peak_force_local_n": float(peak_force_n),
            "peak_moment_local_nm": float(peak_moment_nm),
            "attachment_force_limit_n": force_limit,
            "attachment_moment_limit_nm": moment_limit,
            "admitted": True,
        }
    if bool(attachment_parameters["enabled"]):
        for corner in WHEEL_NAMES:
            frame_identity = f"frame.{corner}"
            frame_position = next(item["reference_position"] for item in nodes
                                  if item["identity"] == frame_identity)
            boss = f"attachment.{corner}"
            node(boss, list(frame_position), "generic-six-axis-wrench-attachment",
                 mass_kg=component_masses[f"wrench_attachment_{corner}"], mass_in_total=True,
                 local_frame={"origin": list(frame_position),
                              "axes": {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}},
                 wrench_envelope={
                     "maximum_force_n": float(attachment_parameters["maximum_force_n"]),
                     "maximum_moment_nm": float(attachment_parameters["maximum_moment_nm"]),
                 },
                 accepts="any-mass-or-subgraph-whose-declared-wrench-envelope-fits")
            edge(f"attachment.mount.{corner}", frame_identity, boss,
                 "breakable-six-axis-braze-on", radius=float(
                     attachment_parameters["bolt_circle_radius_m"]),
                 load_path="generic-accessory-wrench-to-persistent-chassis-corner",
                 yield_force_n=float(attachment_parameters["yield_force_n"]),
                 fracture_force_n=float(attachment_parameters["fracture_force_n"]),
                 yield_moment_nm=float(attachment_parameters["yield_moment_nm"]),
                 fracture_moment_nm=float(attachment_parameters["fracture_moment_nm"]))

    # Front/rear bumpers and the four ballast hangers share the persistent
    # chassis-corner hardpoint identities. Their geometry is not a visual
    # offset: tube and ballast masses enter the same COM/inertia audit, and
    # each bumper shock is an explicit preload/compression/rebound force path.
    attachment_layout = mass_properties["chassis_attachment_layout"]
    bumper_parameters = source["bumpers"]
    bumper_by_name = {row["identity"]: row for row in attachment_layout["bumpers"]}
    for longitudinal, sign in (("front", 1.0), ("rear", -1.0)):
        row = bumper_by_name.get(longitudinal)
        if row is None:
            continue
        endpoint_names = []
        for lateral, side in (("left", -1.0), ("right", 1.0)):
            identity = f"bumper.{longitudinal}.{lateral}"
            endpoint_names.append(identity)
            position = [row["center"][0], row["center"][1],
                        side * row["cross_tube_length_m"] * .5]
            node(identity, position, "shock-mounted-heavy-bumper-end",
                 mass_kg=float(row["assembly_mass_kg"]) * .5, mass_in_total=True,
                 moves_with=f"frame.{longitudinal}_{lateral}",
                 collision_radius_m=float(bumper_parameters["cross_tube_outer_radius_m"]))
            edge(f"bumper.shock.{longitudinal}_{lateral}",
                 f"attachment.{longitudinal}_{lateral}", identity,
                 "preloaded-bumper-shock-absorber",
                 radius=float(bumper_parameters["mount_tube_outer_radius_m"]),
                 load_path="bumper-contact-through-shock-to-chassis-corner",
                 axis_local=[sign, 0.0, 0.0],
                 rest_extension_m=float(bumper_parameters["rest_extension_m"]),
                 maximum_compression_m=float(bumper_parameters["maximum_compression_m"]),
                 preload_force_n=float(bumper_parameters["preload_force_n"]),
                 compression_stiffness_n_per_m=float(
                     bumper_parameters["compression_stiffness_n_per_m"]),
                 compression_damping_n_s_per_m=float(
                     bumper_parameters["compression_damping_n_s_per_m"]),
                 rebound_damping_n_s_per_m=float(
                     bumper_parameters["rebound_damping_n_s_per_m"]),
                 maximum_force_n=float(bumper_parameters["maximum_force_n"]),
                 attachment_admission=attachment_admission(
                     f"bumper.{longitudinal}_{lateral}",
                     float(bumper_parameters["maximum_force_n"]) * .5,
                     float(bumper_parameters["maximum_force_n"]) * .5
                     * float(bumper_parameters["mount_tube_outer_radius_m"])),
                 passivity="damping-opposes-relative-velocity-preload-is-bounded-by-travel")
        edge(f"bumper.cross_tube.{longitudinal}", endpoint_names[0], endpoint_names[1],
             "rigid-distance", radius=float(bumper_parameters["cross_tube_outer_radius_m"]),
             load_path="bumper-cross-tube-distributes-contact-between-two-corner-shocks")

    ballast_parameters = source["ballast"]
    for row in attachment_layout["ballast"]:
        corner = str(row["identity"])
        frame_identity = f"frame.{corner}"
        attachment_identity = f"attachment.{corner}"
        frame_position = next(item["reference_position"] for item in nodes
                              if item["identity"] == frame_identity)
        hanger_component = f"ballast_hanger_{corner}"
        hanger_mass = component_masses[hanger_component]
        hanger_lower = f"ballast.{corner}.hanger_lower"
        node(hanger_lower,
             [frame_position[0], frame_position[1] - float(ballast_parameters["maximum_drop_m"]),
              frame_position[2]], "ballast-hanger-lower-eye",
             mass_kg=hanger_mass, mass_in_total=True, moves_with=frame_identity)
        edge(f"ballast.hanger.{corner}", attachment_identity, hanger_lower,
             "rigid-distance", radius=float(ballast_parameters["hanger_tube_outer_radius_m"]),
             load_path="ballast-weight-to-persistent-chassis-corner",
             density_kg_m3=float(ballast_parameters["hanger_material_density_kg_m3"]),
             maximum_drop_m=float(ballast_parameters["maximum_drop_m"]),
             attachment_admission=attachment_admission(
                 f"ballast.{corner}",
                 (float(row["requested_mass_kg"]) + hanger_mass)
                 * abs(float(source["world"]["gravity"])),
                 (float(row["requested_mass_kg"]) + hanger_mass)
                 * abs(float(source["world"]["gravity"]))
                 * float(ballast_parameters["maximum_drop_m"])))
        if float(row["requested_mass_kg"]) <= 0:
            continue
        block = f"ballast.{corner}.weight"
        node(block, list(map(float, row["center"])), "density-sized-ballast-block",
             mass_kg=float(row["requested_mass_kg"]), mass_in_total=True,
             material=str(ballast_parameters["material"]),
             density_kg_m3=float(row["density_kg_m3"]),
             volume_m3=float(row["volume_m3"]), dimensions_m=list(map(float, row["dimensions_m"])),
             collision="closed-convex-block")
        edge(f"ballast.weight_mount.{corner}", hanger_lower, block,
             "rigid-distance", radius=float(ballast_parameters["hanger_tube_outer_radius_m"]),
             load_path="density-sized-ballast-to-hanger-and-frame-corner")
    cage_front_x = float(body_station["cab_front"])
    cage_rear_x = float(body_station["cab_rear"])
    cage_z = half_width * .76
    cage_floor, cage_roof = .08, max(.42, float(chassis["height"]) + .24)
    for longitudinal, x in (("front", cage_front_x), ("rear", cage_rear_x)):
        for lateral, side in (("left", -1.0), ("right", 1.0)):
            node(f"cage.{longitudinal}_{lateral}.lower", [x, cage_floor, side * cage_z],
                 "roll-cage-node", fixed_to="chassis", structural_deformable=True,
                 longitudinal_authority="chassis-length")
            node(f"cage.{longitudinal}_{lateral}.upper", [x, cage_roof, side * cage_z],
                 "roll-cage-node", fixed_to="chassis", structural_deformable=True,
                 longitudinal_authority="chassis-length")
            edge(f"cage.pillar.{longitudinal}_{lateral}",
                 f"cage.{longitudinal}_{lateral}.lower", f"cage.{longitudinal}_{lateral}.upper",
                 "rigid-distance", radius=.018, load_path="occupant-cell-to-frame")
    for level in ("lower", "upper"):
        for index, (a, b) in enumerate((("front_left", "front_right"),
                                        ("front_right", "rear_right"),
                                        ("rear_right", "rear_left"),
                                        ("rear_left", "front_left"))):
            edge(f"cage.{level}.{index}", f"cage.{a}.{level}", f"cage.{b}.{level}",
                 "rigid-distance", radius=.018, load_path="occupant-cell-to-frame")
    for longitudinal in ("front", "rear"):
        for lateral in ("left", "right"):
            edge(f"cage.frame_mount.{longitudinal}_{lateral}",
                 f"cage.{longitudinal}_{lateral}.lower", f"frame.{longitudinal}_{lateral}",
                 "rigid-distance", radius=.016, load_path="roll-cage-to-lower-frame")

    # The removable cosmetic body is outside the occupant cell, but it still
    # owns collision samples and four sacrificial mounts. Each mount can yield
    # and open without pretending the shell is part of the rigid cage.
    shell_y = max(.24, float(chassis["height"]) + .10)
    for longitudinal, x in (("front", float(body_station["front_end"])),
                            ("rear", float(body_station["rear_end"]))):
        for lateral, side in (("left", -1.0), ("right", 1.0)):
            shell_node = f"body_shell.{longitudinal}_{lateral}"
            frame_node = f"frame.{longitudinal}_{lateral}"
            node(shell_node, [x, shell_y, side * half_width * 1.02], "body-shell-contact-mount")
            edge(f"body_shell.mount.{longitudinal}_{lateral}", shell_node, frame_node,
                 "breakable-body-shell-mount", radius=.008, palette="body-shell-glass",
                 load_path="cosmetic-shell-contact-to-chassis",
                 mount_yield_force_n=float(source["body_shell"]["mount_yield_force_n"]),
                 mount_fracture_force_n=float(source["body_shell"]["mount_fracture_force_n"]))
    for identity, position in (
        ("hood_left", [float(body_station["hood_center"]), .39, -half_width * 1.02]),
        ("hood_right", [float(body_station["hood_center"]), .39, half_width * 1.02]),
        ("cab_roof_left", [float(body_station["cab_center"]), .70, -half_width * .82]),
        ("cab_roof_right", [float(body_station["cab_center"]), .70, half_width * .82]),
        ("bed_left", [float(body_station["bed_center"]), .31, -half_width * 1.02]),
        ("bed_right", [float(body_station["bed_center"]), .31, half_width * 1.02]),
        ("nose_center", [float(body_station["front_end"]), .28, 0.0]),
        ("tail_center", [float(body_station["rear_end"]), .28, 0.0]),
    ):
        node(f"body_shell.sample.{identity}", position, "body-shell-contact-sample")

    # Six RC-style body pins are permanent structural interfaces, not floating
    # weapon props. Each pin locates the shell through a captive preloaded
    # spring and may also accept one independently recoiling clutch-gimbal.
    node("frame.center", [0.0, frame_y, 0.0], "chassis-load-node", fixed_to="chassis",
         structural_deformable=True, longitudinal_authority="chassis-length")
    for corner in ("front_left", "front_right", "rear_left", "rear_right"):
        edge(f"frame.center_brace.{corner}", "frame.center", f"frame.{corner}", "rigid-distance",
             radius=.014, load_path="body-assembly-wrench-distribution")
    body_pin_mounts = {
        "hood_left": ([float(body_station["hood_center"]), .45, -half_width * 1.02],
                      "frame.front_left", "body_shell.sample.hood_left"),
        "hood_right": ([float(body_station["hood_center"]), .45, half_width * 1.02],
                       "frame.front_right", "body_shell.sample.hood_right"),
        "cab_left": ([float(body_station["cab_center"]), .76, -half_width * .82],
                     "cage.front_left.upper", "body_shell.sample.cab_roof_left"),
        "cab_right": ([float(body_station["cab_center"]), .76, half_width * .82],
                      "cage.front_right.upper", "body_shell.sample.cab_roof_right"),
        "bed_left": ([float(body_station["bed_center"]), .39, -half_width * 1.02],
                     "frame.rear_left", "body_shell.sample.bed_left"),
        "bed_right": ([float(body_station["bed_center"]), .39, half_width * 1.02],
                      "frame.rear_right", "body_shell.sample.bed_right"),
    }
    for name, (position, frame_node, shell_node) in body_pin_mounts.items():
        base = f"body_pin.{name}.base"
        spring_seat = f"body_pin.{name}.spring_seat"
        lock = f"body_pin.{name}.lock"
        yaw = f"turret.{name}.yaw"
        pitch = f"turret.{name}.pitch"
        weapon = f"turret.{name}.weapon"
        node(base, [position[0], frame_y, position[2]], "structural-body-pin-frame-foot",
             fixed_to="chassis")
        node(spring_seat, [position[0], position[1] - .055, position[2]],
             "captive-body-pin-lower-spring-seat", fixed_to="chassis")
        node(lock, position, "rc-body-pin-lock-and-gimbal-interface",
             generalized_coordinate=f"body_pin_compression_{name}", moves_with=shell_node,
             mass_accounting_component="body_shell_mounts", body_retention=True,
             optional_payload_interface="actuated-damped-clutch-gimbal")
        node(yaw, position, "yaw-clutch-bearing", body_assembly_identity="six-body-pin-carrier")
        node(pitch, [position[0], position[1] + .04, position[2]], "pitch-bearing",
             body_assembly_identity="six-body-pin-carrier")
        node(weapon, [position[0] + .29, position[1] + .04, position[2]], "recoiling-weapon-mass",
             mass_kg=20.0, mass_in_total=False, body_assembly_identity="six-body-pin-carrier")
        edge(f"body_pin.frame_mount.{name}", base, frame_node,
             "breakable-six-axis-body-pin-foot", radius=.016,
             load_path="body-pin-payload-and-shell-wrench-to-frame",
             mount_yield_force_n=72_000.0, mount_fracture_force_n=118_000.0)
        edge(f"body_pin.shaft.{name}", base, spring_seat,
             "elastic-plastic-structural-body-pin", radius=.014,
             load_path="pin-bending-and-payload-recoil")
        edge(f"body_pin.retainer_spring.{name}", spring_seat, lock,
             "preloaded-captive-body-retainer-spring", radius=.012,
             stiffness_n_per_m=18_000.0, compression_damping_n_s_per_m=780.0,
             rebound_damping_n_s_per_m=1_150.0, preload_compression_m=.006,
             maximum_compression_m=.018, maximum_force_n=2_400.0,
             passivity="spring-stores-bounded-energy-damping-opposes-pin-relative-velocity")
        edge(f"body_pin.body_lock.{name}", lock, shell_node,
             "removable-body-clip-with-preloaded-spring", radius=.010,
             load_path="shell-contact-through-pin-spring-to-frame",
             release_authority="explicit-body-clip-removal")
        edge(f"turret.mount.{name}", lock, yaw, "actuated-damped-clutch-gimbal-base", radius=.028,
             load_path="optional-payload-wrench-through-body-pin-to-frame",
             body_assembly_identity="six-body-pin-carrier",
             clutch_engagement=1.0, angular_stiffness_nm_per_rad=8_200.0,
             angular_damping_nm_s_per_rad=680.0, holding_torque_nm=3_400.0,
             release_behavior="bounded-free-gimbal-after-commanded-clutch-release")
        edge(f"turret.gimbal.yaw.{name}", yaw, pitch, "gimbal-yaw-bearing", radius=.035,
             body_assembly_identity="six-body-pin-carrier")
        edge(f"turret.gimbal.pitch.{name}", pitch, weapon, "gimbal-pitch-bearing", radius=.030,
             body_assembly_identity="six-body-pin-carrier")
        edge(f"turret.recoil.{name}", weapon, pitch, "point-impulse-wrench-coupling", radius=.018,
             load_path="individual-shot-recoil-r-cross-impulse", body_assembly_identity="six-body-pin-carrier")
    node("turret.fire_control", [-.30, .36, 0.0], "independent-fire-control-computer", mass_kg=9.0,
         mass_in_total=False, body_assembly_identity="six-body-pin-carrier")
    node("turret.ammunition", [-.18, .30, 0.0], "mass-and-volume-limited-magazine", mass_kg=51.0,
         mass_in_total=False, capacity_volume_m3=.050, body_assembly_identity="six-body-pin-carrier")
    for index, (name, position, frame_node) in enumerate((
        ("front_left", [half_length * .91, .29, -half_width * 1.04], "frame.front_left"),
        ("front_right", [half_length * .91, .29, half_width * 1.04], "frame.front_right"),
        ("rear_left", [-half_length * .91, .29, -half_width * 1.04], "frame.rear_left"),
        ("rear_right", [-half_length * .91, .29, half_width * 1.04], "frame.rear_right"),
        ("mid_left_low", [0.0, .22, -half_width * 1.04], "frame.front_left"),
        ("mid_right_low", [0.0, .22, half_width * 1.04], "frame.front_right"),
        ("mid_left_high", [0.0, .58, -half_width * .79], "frame.rear_left"),
        ("mid_right_high", [0.0, .58, half_width * .79], "frame.rear_right"),
    )):
        armor_node = f"armor.skirt.{name}"
        node(armor_node, position, "segmented-steel-armor-contact-mount",
             body_assembly_identity="six-body-pin-carrier")
        edge(f"armor.mount.{index}", armor_node, frame_node, "breakable-six-axis-bolt-pattern", radius=.014,
             load_path="armor-contact-wrench-to-chassis", body_assembly_identity="six-body-pin-carrier",
             mount_yield_force_n=94_000.0, mount_fracture_force_n=156_000.0)
    for name, (position, frame_node) in {
        "front_left": ([half_length * .56, .20, -half_width * .88], "frame.front_left"),
        "front_right": ([half_length * .56, .20, half_width * .88], "frame.front_right"),
        "rear_left": ([-half_length * .56, .20, -half_width * .88], "frame.rear_left"),
        "rear_right": ([-half_length * .56, .20, half_width * .88], "frame.rear_right"),
    }.items():
        mount = f"outrigger.{name}.mount"
        foot = f"outrigger.{name}.foot"
        node(mount, position, "hydraulic-outrigger-trunnion", body_assembly_identity="six-body-pin-carrier")
        node(foot, position, "persistent-terrain-weld-foot", mass_kg=12.0, mass_in_total=False,
             body_assembly_identity="six-body-pin-carrier")
        edge(f"outrigger.mount.{name}", mount, frame_node, "breakable-six-axis-bolt-pattern", radius=.018,
             load_path="terrain-anchor-wrench-to-chassis", body_assembly_identity="six-body-pin-carrier")
        edge(f"outrigger.actuator.{name}", mount, foot, "telescopic-hydraulic-diagonal", radius=.030,
             load_path="bidirectional-hydraulic-anchor-force", body_assembly_identity="six-body-pin-carrier",
             maximum_extension_m=1.72, inboard_reserve_m=.72, minimum_structural_overlap_m=.24,
             extension_rate_m_s=.34, axial_stiffness_n_per_m=140_000.0,
             axial_damping_n_s_per_m=12_000.0, maximum_axial_force_n=85_000.0,
             force_authority="canonical-vehicle-total-wrench-input")

    # Close-set, fully round low lamps behind a black brush guard.  Their
    # chassis-local positions are also the authority for the moving light cones.
    lamp_x, lamp_y, lamp_spacing, lamp_radius = half_length + .045, .095, .14, .058
    electrical_points = {
        "battery": [.34, .15, -.22], "fusebox": [.30, .22, .18], "ecu": [.18, .26, -.18],
        "tcu": [.16, .24, .14],
        "front_junction": [half_length * .58, .16, 0.0],
        "rear_junction": [-half_length * .58, .18, 0.0],
        "starter": [float(powertrain["engine_position"][0]) + .08,
                    float(powertrain["engine_position"][1]), -.16],
        "alternator": [float(powertrain["engine_position"][0]) - .10,
                       float(powertrain["engine_position"][1]) + .08, .16],
        "alternator_cvt": [float(powertrain["engine_position"][0]) - .05,
                           float(powertrain["engine_position"][1]) + .08, .08],
        "horn": [half_length * .76, .12, 0.0],
        "ignition_driver": [float(powertrain["engine_position"][0]) + .02,
                            float(powertrain["engine_position"][1]) + .10, -.12],
        "imu": [0.0, .20, 0.0], "brake_switch": [.04, .18, -.18], "light_switch": [.06, .30, -.20],
        "hydraulic_pump": [-.10, .10, .24], "hydraulic_manifold": [-.18, .12, .18],
        "pneumatic_compressor": [-.28, .12, -.24], "pneumatic_accumulator": [-.42, .13, 0.0],
        "pneumatic_tire_manifold": [-.38, .15, .08],
        "brake_master_manifold": [.30, .20, -.08],
        "parking_brake_equalizer": [-.18, .16, 0.0],
        "alignment_manifold": [-.12, .18, .12],
        "steering_servo": [.08, .29, -.10],
    }
    electrical_component_names = {
        "battery": "starter_battery", "starter": "starter_motor",
        "alternator": "alternator", "alternator_cvt": "alternator_cvt",
        "ecu": "vehicle_computer", "tcu": "transmission_control_unit",
        "fusebox": "fusebox_relays", "steering_servo": "steering_servo",
        "hydraulic_pump": "hydraulic_pump", "pneumatic_compressor": "pneumatic_compressor",
        "pneumatic_accumulator": "pneumatic_accumulator",
        "pneumatic_tire_manifold": "pneumatic_tire_manifold",
        "brake_master_manifold": "brake_master_manifold",
        "parking_brake_equalizer": "parking_brake_equalizer",
        "alignment_manifold": "alignment_manifold",
    }
    for name, position in electrical_points.items():
        node(f"electrical.{name}", position,
             "vehicle-computer" if name == "ecu" else
             "transmission-control-unit" if name == "tcu" else
             "demand-smoothing-generator-cvt" if name == "alternator_cvt" else
             "shaft-generator-bank" if name == "alternator" else "electrical-junction",
             fixed_to="chassis", circuit_role=name,
             mass_kg=component_masses.get(electrical_component_names.get(name, ""), 0.0),
             mass_in_total=name in electrical_component_names,
             generator_count=(int(electrical["alternator_count"])
                              if name == "alternator" else None),
             rotor_inertia_kg_m2_each=(float(electrical["alternator_rotor_inertia_kg_m2_each"])
                                      if name == "alternator" else None),
             live_ratio_coordinate=("alternator_cvt_ratio_state"
                                    if name == "alternator_cvt" else None))
    for lateral, center_z in (("left", -lamp_spacing), ("right", lamp_spacing)):
        center_identity = f"lighting.headlamp.{lateral}.center"
        node(center_identity, [lamp_x, lamp_y, center_z], "round-headlamp-emitter",
             fixed_to="chassis", beam_axis_local=[1.0, 0.0, 0.0],
             beam_half_angle_degrees=18.0, beam_range_m=16.0)
        ring = []
        for index in range(12):
            angle = 2 * math.pi * index / 12
            identity = f"lighting.headlamp.{lateral}.ring_{index}"
            ring.append(identity)
            node(identity, [lamp_x, lamp_y + lamp_radius * math.cos(angle),
                            center_z + lamp_radius * math.sin(angle)],
                 "round-headlamp-lens-rim", fixed_to="chassis")
            edge(f"lighting.headlamp.{lateral}.spoke_{index}", center_identity, identity,
                 "rigid-lamp-lens", radius=.005, palette="active",
                 light_emission="warm-white")
        for index, identity in enumerate(ring):
            edge(f"lighting.headlamp.{lateral}.rim_{index}", identity,
                 ring[(index + 1) % len(ring)], "rigid-round-lamp-rim",
                 radius=.006, palette="active", light_emission="warm-white")
    tail_x, tail_y, tail_spacing, tail_radius = -half_length - .025, .18, .22, .048
    for lateral, center_z in (("left", -tail_spacing), ("right", tail_spacing)):
        center_identity = f"lighting.tail.{lateral}.center"
        node(center_identity, [tail_x, tail_y, center_z], "tail-brake-lamp-emitter",
             fixed_to="chassis", beam_axis_local=[-1.0, 0.0, 0.0], circuit="tail-and-brake")
        ring = []
        for index in range(8):
            angle = 2 * math.pi * index / 8
            identity = f"lighting.tail.{lateral}.ring_{index}"
            ring.append(identity)
            node(identity, [tail_x, tail_y + tail_radius * math.cos(angle),
                            center_z + tail_radius * math.sin(angle)],
                 "tail-brake-lamp-lens-rim", fixed_to="chassis")
            edge(f"lighting.tail.{lateral}.spoke_{index}", center_identity, identity,
                 "rigid-lamp-lens", radius=.0045, palette="active", light_emission="red")
        for index, identity in enumerate(ring):
            edge(f"lighting.tail.{lateral}.rim_{index}", identity,
                 ring[(index + 1) % len(ring)], "rigid-round-lamp-rim",
                 radius=.0055, palette="active", light_emission="red")
    wire_routes = (
        ("battery_feed", "electrical.battery", "electrical.fusebox", "battery-main", 80.0),
        ("ecu_feed", "electrical.fusebox", "electrical.ecu", "computer-and-ignition", 12.0),
        ("tcu_feed", "electrical.fusebox", "electrical.tcu", "transmission-control", 12.0),
        ("powertrain_can", "electrical.ecu", "electrical.tcu", "engine-transmission-can", 2.0),
        ("starter_feed", "electrical.fusebox", "electrical.starter", "starter-solenoid", 180.0),
        ("alternator_charge", "electrical.alternator", "electrical.battery", "alternator-charge", 95.0),
        ("front_harness", "electrical.fusebox", "electrical.front_junction", "front-lighting-horn", 20.0),
        ("rear_harness", "electrical.fusebox", "electrical.rear_junction", "rear-lighting", 12.0),
        ("horn_branch", "electrical.front_junction", "electrical.horn", "horn", 12.0),
        ("ignition_command", "electrical.ecu", "electrical.ignition_driver", "crank-cam-timed-ignition", 8.0),
        ("imu_bus", "electrical.imu", "electrical.ecu", "pitch-rate-and-acceleration-can", 1.0),
        ("brake_switch_bus", "electrical.brake_switch", "electrical.ecu", "brake-light-request", 1.0),
        ("light_switch_bus", "electrical.light_switch", "electrical.ecu", "lighting-request", 1.0),
        ("hydraulic_pump_feed", "electrical.fusebox", "electrical.hydraulic_pump", "suspension-hydraulics", 65.0),
        ("compressor_feed", "electrical.fusebox", "electrical.pneumatic_compressor", "tire-and-shock-air", 48.0),
        ("steering_servo_feed", "electrical.fusebox", "electrical.steering_servo", "steering-assist-power", 70.0),
        ("headlamp_left", "electrical.front_junction", "lighting.headlamp.left.center", "headlight", 8.0),
        ("headlamp_right", "electrical.front_junction", "lighting.headlamp.right.center", "headlight", 8.0),
        ("tail_left", "electrical.rear_junction", "lighting.tail.left.center", "tail-brake-light", 4.0),
        ("tail_right", "electrical.rear_junction", "lighting.tail.right.center", "tail-brake-light", 4.0),
    )
    for name, a, b, circuit, maximum_current in wire_routes:
        edge(f"electrical.wire.{name}", a, b, "insulated-copper-wire", radius=.0035,
             palette="active", circuit=circuit, maximum_current_a=maximum_current,
             electrical_authority="vehicle-computer-fusebox-relay-dispatch",
             routing="relaxed-multi-segment-harness", bundle_kind="electrical-loom",
             slack_ratio=1.08, bend_relaxation=.78, relaxation_rate_hz=4.0,
             minimum_bend_radius_m=.018)
    guard_points = {
        "lower_left": [lamp_x + .018, .015, -.29],
        "upper_left": [lamp_x + .018, .19, -.29],
        "upper_inner_left": [lamp_x + .018, .225, -.18],
        "upper_inner_right": [lamp_x + .018, .225, .18],
        "upper_right": [lamp_x + .018, .19, .29],
        "lower_right": [lamp_x + .018, .015, .29],
    }
    for name, position in guard_points.items():
        node(f"lighting.brush_guard.{name}", position, "brush-guard-node", fixed_to="chassis")
    guard_order = tuple(guard_points)
    for index in range(len(guard_order) - 1):
        edge(f"lighting.brush_guard.member_{index}",
             f"lighting.brush_guard.{guard_order[index]}",
             f"lighting.brush_guard.{guard_order[index + 1]}",
             "rigid-brush-guard", radius=.012, palette="drivetrain-black")

    for corner in WHEEL_NAMES:
        longitudinal, lateral = corner.split("_")
        x = axle_offset + (wheelbase if longitudinal == "front" else -wheelbase)
        side = -1.0 if lateral == "left" else 1.0
        nominal_motion_ratio = .78
        static_compression = min(float(suspension["travel"]), config.sprung_mass()
                                 * abs(float(source["world"]["gravity"]))
                                 * float(source["mass_distribution"][corner])
                                 / (float(suspension["stiffness"]) * nominal_motion_ratio ** 2))
        hub_y = -float(chassis["clearance"])
        hub_z = side * track
        knuckle_z = side * (track - hub_face_offset)
        prefix = f"suspension.{corner}"
        # A-arm pickup pairs define two chassis-fixed revolute axes.  Their paired
        # links converge at spherical ball joints on a rigid upright.
        synthesized_pickups = wheel_mount_solution["synthesized_pickups"][corner]
        points = {
            "upper_pickup_forward": list(synthesized_pickups["upper_pickup_forward"]),
            "upper_pickup_rear": list(synthesized_pickups["upper_pickup_rear"]),
            "lower_pickup_forward": list(synthesized_pickups["lower_pickup_forward"]),
            "lower_pickup_rear": list(synthesized_pickups["lower_pickup_rear"]),
            "upper_ball_joint": [x, hub_y + .085, knuckle_z],
            "lower_ball_joint": [x, hub_y - .085, knuckle_z],
            "upper_knuckle_socket": [x, hub_y + .085, knuckle_z],
            "lower_knuckle_socket": [x, hub_y - .085, knuckle_z],
            "knuckle": [x, hub_y, knuckle_z],
            "halfshaft_joint": [x, hub_y, knuckle_z],
            # The hub IS the wheel-area object: the former wheel_rim and
            # tire_carcass nodes double-represented this exact position and
            # are collapsed into it. The hub carries the wheel mass, the
            # spin inertia, the rigid bead-to-bead rim profile, and the
            # tire object (the tire_skin subgraph attaches here).
            "hub": [x, hub_y, hub_z],
            "brake_rotor": [x, hub_y, hub_z - side * .012],
            "brake_caliper": [x - .025, hub_y + .025, knuckle_z],
            "coilover_chassis": list(synthesized_pickups["coilover_chassis"]),
            "contact_patch": [x, hub_y - wheel_radius, hub_z],
        }
        rack_x = x - .08 if longitudinal == "front" else x + .08
        arm_x = x - .045 if longitudinal == "front" else x + .045
        points["steering_rack"] = [rack_x, hub_y + .025, side * half_width * .24]
        points["tie_rod_outer"] = [arm_x, hub_y + .025, knuckle_z - side * .025]
        points["steering_arm"] = [arm_x, hub_y + .025, knuckle_z - side * .025]
        moving_mass_components = {
            "knuckle": f"knuckle_upright_{corner}",
            "brake_caliper": f"brake_caliper_{corner}",
            "brake_rotor": f"brake_rotor_{corner}",
            "lower_ball_joint": f"coilover_unsprung_{corner}",
            "coilover_chassis": f"coilover_sprung_{corner}",
            "hub": f"wheel_{corner}",
        }
        for name, position in points.items():
            fixed = name.endswith("pickup_forward") or name.endswith("pickup_rear") \
                or name in {"coilover_chassis", "steering_rack"}
            node(f"{prefix}.{name}", position,
                 "contact-patch" if name == "contact_patch" else
                  "upright-knuckle" if name == "knuckle" else
                  "halfshaft-universal-joint" if name == "halfshaft_joint" else
                 "wheel-hub" if name == "hub" else
                 "brake-rotor" if name == "brake_rotor" else
                 "brake-caliper" if name == "brake_caliper" else
                 "replaceable-break-bushing-terminal" if name in {
                     "upper_knuckle_socket", "lower_knuckle_socket", "tie_rod_outer"} else
                 "spherical-joint" if name.endswith("ball_joint") else
                 "chassis-pickup", fixed_to="chassis" if fixed else None,
                 generalized_coordinate=None if fixed else f"compression_{corner}",
                 mass_kg=component_masses.get(moving_mass_components.get(name, ""), 0.0),
                 mass_in_total=name in moving_mass_components,
                 mass_frame=("chassis-sprung" if name == "coilover_chassis" else
                             "corner-unsprung" if name in moving_mass_components else None),
                 moves_with=(f"suspension.{corner}.knuckle" if name in {
                     "knuckle", "brake_caliper", "brake_rotor", "upper_knuckle_socket",
                     "lower_knuckle_socket", "steering_arm",
                     "hub"} else "chassis" if name == "coilover_chassis" else None))
        chassis_service_z = side * half_width * .62
        service_points = {
            "pneumatic_chassis_junction": [x, frame_y - .03, chassis_service_z],
            "pneumatic_service_loop": [x - .035, hub_y + .05, knuckle_z],
            "pneumatic_rotary_union": [x, hub_y, knuckle_z + side * .018],
            "pneumatic_bearing_rotor": [x, hub_y, knuckle_z + side * .010],
            "pneumatic_rim_seat_port_inner": [x, hub_y, hub_z - side * float(source["tires"]["width"]) * .42],
            "pneumatic_rim_seat_port_outer": [x, hub_y, hub_z + side * float(source["tires"]["width"]) * .42],
            "pneumatic_wheel_valve": [x, hub_y + wheel_radius * .34, hub_z],
            "pneumatic_tube_stem": [x, hub_y + float(source["wheels"]["rim_radius"]), hub_z],
            "brake_chassis_junction": [x + .025, frame_y - .05, chassis_service_z],
            "brake_service_port": [x - .035, hub_y + .045, knuckle_z],
            "alignment_chassis_junction": [x - .025, frame_y, chassis_service_z],
            "alignment_service_port": [x + .025, hub_y + .035, knuckle_z],
        }
        service_mass_components = {
            "pneumatic_chassis_junction": ("pneumatic_chassis_lines", .25),
            "pneumatic_service_loop": (f"pneumatic_service_loop_{corner}", 1.0),
            "pneumatic_rotary_union": (f"pneumatic_rotary_union_{corner}", 1.0),
            "pneumatic_wheel_valve": (f"pneumatic_wheel_valve_{corner}", 1.0),
            "brake_chassis_junction": ("brake_chassis_lines", .25),
            "brake_service_port": (f"brake_service_hose_{corner}", 1.0),
            "alignment_chassis_junction": ("alignment_chassis_lines", .25),
            "alignment_service_port": (f"alignment_service_loop_{corner}", 1.0),
        }
        for name, position in service_points.items():
            mass_name, mass_scale = service_mass_components.get(name, (None, 0.0))
            sprung = "chassis_junction" in name
            node(
                f"{prefix}.{name}", position,
                "chassis-service-junction" if sprung else
                "rotating-air-union" if name == "pneumatic_rotary_union" else
                "wheel-air-valve" if name == "pneumatic_wheel_valve" else
                "rotating-bearing-air-chamber" if name == "pneumatic_bearing_rotor" else
                "rim-bead-seat-air-port" if "rim_seat_port" in name else
                "tube-valve-stem" if name == "pneumatic_tube_stem" else
                "typed-moving-service-port",
                fixed_to="chassis" if sprung else None,
                moves_with=("chassis" if sprung else
                            f"{prefix}.hub" if name in {
                                "pneumatic_bearing_rotor", "pneumatic_rim_seat_port_inner",
                                "pneumatic_rim_seat_port_outer", "pneumatic_wheel_valve",
                                "pneumatic_tube_stem"} else
                            f"{prefix}.knuckle"),
                accepts=("pressurized-gas" if name.startswith("pneumatic") else
                         "brake-fluid-pressure" if name.startswith("brake") else
                         "alignment-hydraulic-pressure"),
                work_device="declared-by-future-loadout-not-created-by-service-network",
                mass_kg=(float(component_masses[mass_name]) * mass_scale
                         if mass_name is not None else 0.0),
                mass_in_total=mass_name is not None,
                mass_frame="chassis-sprung" if sprung else "corner-unsprung",
                assembly_custody=(
                    "pillar-then-knuckle-stationary-side" if name == "pneumatic_rotary_union" else
                    "hub-rotating-side" if name in {"pneumatic_bearing_rotor",
                        "pneumatic_rim_seat_port_inner", "pneumatic_rim_seat_port_outer",
                        "pneumatic_wheel_valve"} else
                    "tube-then-rim-service-valve-on-mount" if name == "pneumatic_tube_stem" else None),
                removed_hub_material_kg=(
                    float(source["service_lines"]["pneumatic_outer_valve_removed_hub_material_kg_each"])
                    if name == "pneumatic_wheel_valve" else 0.0),
                balance_mass_delta_kg=(
                    float(source["service_lines"]["pneumatic_wheel_valve_mass_kg_each"])
                    - float(source["service_lines"]["pneumatic_outer_valve_removed_hub_material_kg_each"])
                    if name == "pneumatic_wheel_valve" else 0.0),
            )
        if longitudinal == "rear":
            node(
                f"{prefix}.parking_brake_service_port",
                [x - .055, hub_y + .015, knuckle_z],
                "parking-brake-cable-terminal", moves_with=f"{prefix}.knuckle",
                accepts="tension-only-parking-brake-command",
                work_device="existing-rear-caliper-parking-brake-mechanism",
                mass_kg=float(component_masses[f"parking_brake_cable_{corner}"]),
                mass_in_total=True, mass_frame="corner-unsprung",
            )
        skin_vertex_names: list[str] = []
        vertex_mass = float(tire_skin_abi["parameters"]["vertex_mass_kg"])
        for vertex_index, local_position in enumerate(tire_topology.rest_positions):
            identity = f"{prefix}.tire_skin.vertex_{vertex_index}"
            skin_vertex_names.append(identity)
            node(
                identity,
                [x + local_position[0], hub_y + local_position[1], hub_z + local_position[2]],
                "balloon-tire-skin-vertex",
                generalized_coordinate=f"tire_skin_{corner}_vertex_{vertex_index}_xyz",
                mass_kg=vertex_mass,
                mass_in_total=True,
                mass_frame="corner-unsprung-deformable",
                collision_authority="deformed-skin-vertex-triangle-ccd",
            )
        face_identities: list[str] = []
        for face_index, ((ia, ib, ic), rest_data) in enumerate(zip(
            tire_topology.faces, tire_topology.face_rest_data, strict=True,
        )):
            face_identity = f"{prefix}.tire_skin.face_{face_index}"
            face_identities.append(face_identity)
            membranes.append({
                "identity": face_identity,
                "kind": "compiled-stvk-kelvin-membrane-face",
                "vertices": [skin_vertex_names[ia], skin_vertex_names[ib], skin_vertex_names[ic]],
                "rest_inverse": [list(rest_data[:2]), list(rest_data[2:4])],
                "rest_area_m2": rest_data[4],
                "kernel": "balloon_tire_membrane_face",
                "force_reduction": "scatter-three-face-wrenches-to-shared-skin-vertices",
                "pressure_source": f"{prefix}.tire_skin.closed_volume",
            })
        for ring_index, ring in enumerate(tire_topology.bead_rings):
            for bead_index, vertex_index in enumerate(ring):
                identity = f"{prefix}.tire_skin.bead_{ring_index}_{bead_index}"
                edges.append({
                    "identity": identity,
                    "a": f"{prefix}.hub",
                    "b": skin_vertex_names[vertex_index],
                    "constraint": "compiled-bead-to-rim-equal-opposite-wrench",
                    "rest_length": math.sqrt(sum(
                        value * value for value in tire_topology.rest_positions[vertex_index]
                    )),
                    "radius": .008,
                    "palette_role": "drivetrain-black",
                    "target_local_position": list(tire_topology.rest_positions[vertex_index]),
                    "kernel": "balloon_tire_bead_constraint",
                    "force_path": "skin-bead-force-and-opposite-rim-force-moment",
                    "stiffness_n_per_m": float(source["tire_skin"]["bead_stiffness_n_per_m"]),
                    "damping_n_s_per_m": float(source["tire_skin"]["bead_damping_n_s_per_m"]),
                })
        membranes.append({
            "identity": f"{prefix}.tire_skin.closed_volume",
            "kind": "compiled-closed-skin-polytropic-gas",
            "faces": face_identities,
            "kernel": "balloon_tire_gas",
            "reference_volume_m3": tire_topology.reference_volume_m3,
            "reference_pressure_pa": float(source["tires"]["pressure_pa"]),
            "pressure_force_path": "gas-pressure-times-closed-mesh-volume-gradient",
        })
        for level in ("upper", "lower"):
            for direction in ("forward", "rear"):
                edge(f"{prefix}.{level}_arm_{direction}",
                     f"{prefix}.{level}_pickup_{direction}", f"{prefix}.{level}_ball_joint",
                     "rigid-distance", palette="suspension-yellow", joint_a="revolute-axis-x",
                     joint_b="spherical", force_path="contact-to-chassis",
                     linear_actuator={
                         "authority": ("alignment-and-lvl-composed-rest-length-modifier"
                                       if level == "upper" else
                                       "lvl-hydraulic-link-length-modifier"),
                         "maximum_extension_m": .46, "fail_mode": "hold-current-length"},
                     **({"alignment_strain_relief_actuator": {
                         **alignment_strain_relief_definition,
                         "installed_identity": f"alignment_actuator_{corner}_{level}_{direction}",
                         "command_authority": ("camber-and-caster-rest-length-target"),
                         "mass_node": f"{prefix}.{level}_ball_joint",
                     }} if level == "upper" else {}))
            for direction in ("forward", "rear"):
                post_terminal = f"suspension_mount_post.{corner}.{'upper' if level == 'upper' else 'lower'}"
                edge(f"{prefix}.{level}_pickup_mount_{direction}",
                     f"{prefix}.{level}_pickup_{direction}", post_terminal,
                     "rigid-distance", radius=.013, palette="rollbar-silver",
                     force_path="wishbone-pickup-through-standard-post-to-frame")
        edge(f"{prefix}.upper_break_bushing", f"{prefix}.upper_ball_joint",
             f"{prefix}.upper_knuckle_socket", "replaceable-sacrificial-knuckle-bushing",
             radius=.018, palette="suspension-yellow",
             sacrificial_break_bushing={**knuckle_break_bushing_definition,
                 "installed_identity": f"knuckle_break_bushing_{corner}_upper",
                 "protected_paths": [f"{prefix}.upper_arm_forward", f"{prefix}.upper_arm_rear"]})
        edge(f"{prefix}.lower_break_bushing", f"{prefix}.lower_ball_joint",
             f"{prefix}.lower_knuckle_socket", "replaceable-sacrificial-knuckle-bushing",
             radius=.018, palette="suspension-yellow",
             sacrificial_break_bushing={**knuckle_break_bushing_definition,
                 "installed_identity": f"knuckle_break_bushing_{corner}_lower",
                 "protected_paths": [f"{prefix}.lower_arm_forward", f"{prefix}.lower_arm_rear"]})
        edge(f"{prefix}.upright", f"{prefix}.upper_knuckle_socket", f"{prefix}.lower_knuckle_socket",
             "rigid-distance", radius=.014, joint_a="spherical", joint_b="spherical",
             force_path="contact-to-control-arms")
        edge(f"{prefix}.hub_carrier", f"{prefix}.upper_knuckle_socket", f"{prefix}.knuckle",
             "rigid-offset", radius=.011, force_path="hub-to-upright")
        edge(f"{prefix}.wheel_bearing", f"{prefix}.knuckle", f"{prefix}.hub",
             "rotational-bearing", radius=.012, palette="rollbar-silver",
             polar_inertia_kg_m2=config.wheel_rotational_inertia(),
             gyroscopic_reaction="signed-omega-cross-angular-momentum-to-knuckle-and-chassis",
             structural_constraint="five-axis-support-one-axis-free-rotation",
             radial_stiffness_n_per_m=8.5e7, axial_stiffness_n_per_m=6.2e7,
             moment_stiffness_nm_per_rad=1.8e5, radial_yield_force_n=145_000.0,
             axial_yield_force_n=92_000.0, moment_yield_nm=18_000.0,
             force_path="upright-through-bearing-to-wheel")
        edge(f"{prefix}.outer_halfshaft_joint", f"{prefix}.halfshaft_joint", f"{prefix}.hub",
             "selectable-wheel-end-locking-hub", radius=.010, palette="drivetrain-black",
             polar_inertia_kg_m2=config.wheel_rotational_inertia(),
             generalized_coordinate=f"steering_angle_{longitudinal}",
             engagement_coordinate=f"hub_locker_engagement_{corner}",
             wear_coordinate=f"hub_locker_wear_{corner}",
             glaze_coordinate=f"hub_locker_glaze_{corner}",
             clutch="4140-steel-clutch-ring-and-inner-splined-drive-gear",
             bearing_support="wheel-hub-remains-supported-by-knuckle-wheel-bearing",
             disengaged_path="wheel-rotor-and-tire-free-on-bearing-halfshaft-isolated",
             reconnect_interlock=(
                 "zero-commanded-torque-and-near-zero-halfshaft-to-hub-relative-speed"),
             force_path="halfshaft-universal-joint-through-locking-clutch-to-wheel-hub")
        edge(f"{prefix}.coilover", f"{prefix}.coilover_chassis", f"{prefix}.lower_ball_joint",
             "spring-damper", radius=.017, palette="suspension-yellow",
             stiffness=float(suspension["stiffness"]),
             constitutive_models={
                 "selector": int(suspension["spring_model_selector"]),
                 "0": {"kind": "linear", "stiffness_n_per_m": float(suspension["stiffness"])},
                 "1": {"kind": "custom-parametric-progressive",
                       "linear_n_per_m": float(suspension["stiffness"]),
                       "quadratic_n_per_m2": float(suspension["spring_progressive_quadratic_n_per_m2"]),
                       "cubic_n_per_m3": float(suspension["spring_progressive_cubic_n_per_m3"])},
                 "2": {"kind": "composite-parametric-helix",
                       "primary": {name: float(suspension[f"spring_primary_{name}"])
                                   for name in ("wire_diameter_m", "mean_coil_diameter_m",
                                                "active_turns", "shear_modulus_pa")},
                       "secondary": {name: float(suspension[f"spring_secondary_{name}"])
                                     for name in ("wire_diameter_m", "mean_coil_diameter_m",
                                                  "active_turns", "shear_modulus_pa")},
                       "secondary_engagement_compression_m": float(
                           suspension["spring_secondary_engagement_compression_m"]),
                       "coupling_efficiency": float(
                           suspension["spring_composite_coupling_efficiency"])},
             },
             static_preload_compression_m=static_compression,
             visualization={
                 "kind": "helical-coilover",
                 "wire_radius_m": max(.0026, min(.0062, .0026 +
                     float(suspension["stiffness"]) / 45_000_000.0)),
                 "active_turns": max(5.5, min(10.5, 10.5 -
                     float(suspension["stiffness"]) / 45_000.0)),
                 "preload_collar": True,
                 "damper_shaft_radius_m": .006,
             },
             compression_damping=float(suspension["pneumatic_compression_damping"]),
             rebound_damping=float(suspension["pneumatic_rebound_damping"]),
             bump_stop={"role": "complementary-terminal-branch-no-parallel-double-count",
                        "start_compression_m": float(suspension["travel"])
                        * float(suspension["bump_stop_start_fraction_of_travel"]),
                        "linear_stiffness_n_per_m": float(suspension["bump_stop_stiffness_n_per_m"]),
                        "progressive_stiffness_n_per_m2": float(
                            suspension["bump_stop_progressive_stiffness_n_per_m2"]),
                        "damping_n_s_per_m": float(suspension["bump_stop_damping_n_s_per_m"]),
                        "maximum_compression_m": float(suspension["travel"])},
             force_path="lower-arm-to-chassis")
        edge(f"{prefix}.coilover_mount_bracket", f"{prefix}.coilover_chassis",
             f"suspension_mount_post.{corner}.upper", "rigid-distance", radius=.014,
             palette="rollbar-silver", force_path="coilover-through-standard-post-to-frame")
        edges[-1]["mass_distribution"] = {
            "total_mass_kg": float(suspension["coilover_mass_kg"]),
            "sprung_fraction": 1 - float(suspension["coilover_unsprung_fraction"]),
            "unsprung_fraction": float(suspension["coilover_unsprung_fraction"]),
            "upper_mass_node": f"{prefix}.coilover_chassis",
            "lower_mass_node": f"{prefix}.lower_ball_joint",
        }
        edge(f"{prefix}.droop_limit_strap", f"{prefix}.coilover_chassis",
             f"{prefix}.lower_ball_joint", "tension-limit-strap", radius=.006,
             palette="suspension-yellow", tension_only=True,
             maximum_length_m=math.sqrt(sum((points["coilover_chassis"][axis] -
                 points["lower_ball_joint"][axis]) ** 2 for axis in range(3))) +
                 float(suspension["travel"]) * .16,
             axial_stiffness_n_per_m=1.9e6, tensile_yield_force_n=42_000.0,
             failure_response="strap-fracture-allows-full-droop-until-other-member-limit",
             force_path="lower-arm-droop-stop-to-coilover-tower")
        edge(f"{prefix}.coilover_tower", f"{prefix}.coilover_chassis",
             f"cage.{longitudinal}_{lateral}.lower", "rigid-distance", radius=.015,
             palette="rollbar-silver", force_path="coilover-top-to-roll-cage-and-frame")
        edge(f"{prefix}.rotor_mount", f"{prefix}.hub", f"{prefix}.brake_rotor",
             "rigid-rotor-mount", radius=.009, palette="drivetrain-black",
             torsional_stiffness_nm_per_rad=2.4e5, torsional_yield_nm=24_000.0,
             force_path="wheel-hub-to-brake-rotor")
        edge(f"{prefix}.caliper_mount", f"{prefix}.knuckle", f"{prefix}.brake_caliper",
             "rigid-caliper-mount", radius=.009, palette="suspension-yellow",
             shear_stiffness_n_per_m=7.5e7, moment_stiffness_nm_per_rad=1.4e5,
             shear_yield_force_n=110_000.0, moment_yield_nm=16_000.0,
             force_path="brake-reaction-to-knuckle-upright-and-both-control-arms")
        edge(f"{prefix}.service_brake", f"{prefix}.brake_rotor", f"{prefix}.brake_caliper",
             "friction-brake-torque-couple", radius=.008, palette="suspension-yellow",
             torque_channel=f"brake_torque * brake * brake_scale_{corner}",
             reaction_path="caliper-to-knuckle-to-wishbones-to-chassis")
        edge(f"{prefix}.steering_arm", f"{prefix}.knuckle", f"{prefix}.steering_arm",
             "rigid-offset", radius=.009, palette="suspension-yellow",
             generalized_coordinate=f"steering_angle_{longitudinal}",
             force_path="steering-moment-to-upright")
        edge(f"{prefix}.tie_rod", f"{prefix}.steering_rack", f"{prefix}.tie_rod_outer",
             "steering-link", radius=.009, palette="suspension-yellow",
             generalized_coordinate=f"steering_angle_{longitudinal}",
             joint_a="spherical", joint_b="spherical",
             force_path="rack-to-upright-steering-moment",
             linear_actuator={"authority": "alignment-and-lvl-composed-rest-length-modifier",
                              "maximum_extension_m": .46, "fail_mode": "hold-current-length"},
             alignment_strain_relief_actuator={
                 **alignment_strain_relief_definition,
                 "installed_identity": f"alignment_actuator_{corner}_tie_rod",
                 "command_authority": "toe-rest-length-target",
                 "mass_node": f"{prefix}.tie_rod_outer",
             })
        edge(f"{prefix}.tie_rod_break_bushing", f"{prefix}.tie_rod_outer",
             f"{prefix}.steering_arm", "replaceable-sacrificial-knuckle-bushing",
             radius=.016, palette="suspension-yellow",
             sacrificial_break_bushing={**knuckle_break_bushing_definition,
                 "installed_identity": f"knuckle_break_bushing_{corner}_tie_rod",
                 "protected_paths": [f"{prefix}.tie_rod", f"{prefix}.steering_arm"]})

    edge("hydraulics.pump_to_manifold", "electrical.hydraulic_pump", "electrical.hydraulic_manifold",
         "pressure-rated-hydraulic-line", radius=.005, palette="suspension-yellow",
         pressure_authority="pose-controller-pump-and-relief-valve",
         routing="relaxed-multi-segment-harness", bundle_kind="hydraulic-harness",
         slack_ratio=1.035, bend_relaxation=.40, relaxation_rate_hz=10.0,
         minimum_bend_radius_m=.035, pressure_rating_pa=18_000_000.0)
    edge("pneumatics.compressor_to_accumulator", "electrical.pneumatic_compressor",
         "electrical.pneumatic_accumulator", "pressure-rated-air-line", radius=.005,
         palette="active", pressure_authority="regulated-tire-and-shock-reservoir",
         routing="relaxed-multi-segment-harness", bundle_kind="pneumatic-harness",
         slack_ratio=1.06, bend_relaxation=.62, relaxation_rate_hz=6.0,
         minimum_bend_radius_m=.025, pressure_rating_pa=1_200_000.0)
    edge("pneumatics.accumulator_to_tire_manifold", "electrical.pneumatic_accumulator",
         "electrical.pneumatic_tire_manifold", "pressure-rated-air-line",
         radius=float(service_lines["pneumatic_hard_line_radius_m"]), palette="active",
         pressure_authority="ecu-regulated-central-tire-inflation-manifold",
         routing="chassis-clipped-hard-line", bundle_kind="pneumatic-harness",
         pressure_rating_pa=1_200_000.0)
    for corner in WHEEL_NAMES:
        edge(f"hydraulics.line.{corner}", "electrical.hydraulic_manifold",
             f"suspension.{corner}.coilover_chassis", "flexible-hydraulic-hose", radius=.0045,
             palette="suspension-yellow", channel=f"corner_height_{corner}",
             routing="relaxed-multi-segment-harness", bundle_kind="hydraulic-harness",
             slack_ratio=1.035, bend_relaxation=.40, relaxation_rate_hz=10.0,
             minimum_bend_radius_m=.035, pressure_rating_pa=18_000_000.0)
        edge(f"pneumatics.shock_line.{corner}", "electrical.pneumatic_accumulator",
             f"suspension.{corner}.coilover_chassis", "flexible-air-line", radius=.0035,
             palette="active", channel=f"pneumatic_damping_{corner}",
             routing="relaxed-multi-segment-harness", bundle_kind="pneumatic-harness",
             slack_ratio=1.06, bend_relaxation=.62, relaxation_rate_hz=6.0,
             minimum_bend_radius_m=.025, pressure_rating_pa=1_200_000.0)
        prefix = f"suspension.{corner}"
        edge(f"pneumatics.tire_hard_line.{corner}", "electrical.pneumatic_tire_manifold",
             f"{prefix}.pneumatic_chassis_junction", "rigid-pneumatic-hard-line",
             radius=float(service_lines["pneumatic_hard_line_radius_m"]), palette="active",
             channel=f"tire_pressure_{corner}", routing="chassis-clipped-hard-line",
             bundle_kind="pneumatic-harness", pressure_rating_pa=1_200_000.0,
             mass_distribution={"total_mass_kg": float(
                 service_lines["pneumatic_chassis_line_mass_kg"]) / 4,
                 "mass_node": f"{prefix}.pneumatic_chassis_junction"})
        edge(f"pneumatics.tire_service_loop.{corner}",
             f"{prefix}.pneumatic_chassis_junction", f"{prefix}.pneumatic_service_loop",
             "flexible-pneumatic-hose", radius=float(service_lines["pneumatic_hose_radius_m"]),
             palette="active", channel=f"tire_pressure_{corner}",
             routing="suspension-travel-service-loop", bundle_kind="pneumatic-harness",
             slack_ratio=1.18, bend_relaxation=.62, relaxation_rate_hz=6.0,
             minimum_bend_radius_m=.035, pressure_rating_pa=1_200_000.0,
             mass_distribution={"total_mass_kg": float(
                 service_lines["pneumatic_service_loop_mass_kg_each"]),
                 "mass_node": f"{prefix}.pneumatic_service_loop"})
        edge(f"pneumatics.rotary_union_feed.{corner}", f"{prefix}.pneumatic_service_loop",
             f"{prefix}.pneumatic_rotary_union", "flexible-pneumatic-hose",
             radius=float(service_lines["pneumatic_hose_radius_m"]), palette="active",
             pressure_transfer="stationary-knuckle-side-to-rotating-wheel-side",
             rotary_seal_axis="wheel-bearing-axis", pressure_rating_pa=1_200_000.0,
             custody_transfer="pillar-stationary-fixture-to-installed-knuckle")
        edge(f"pneumatics.bearing_rotary_seal.{corner}",
             f"{prefix}.pneumatic_rotary_union", f"{prefix}.pneumatic_bearing_rotor",
             "annular-bearing-pneumatic-rotary-seal", radius=.006, palette="active",
             stationary_owner=f"{prefix}.knuckle", rotating_owner=f"{prefix}.hub",
             pressure_rating_pa=1_200_000.0)
        for seat in ("inner", "outer"):
            edge(f"pneumatics.hub_passage.{corner}.{seat}",
             f"{prefix}.pneumatic_bearing_rotor",
             f"{prefix}.pneumatic_rim_seat_port_{seat}", "drilled-hub-air-passage",
             radius=float(service_lines["pneumatic_hard_line_radius_m"]), palette="active",
             rotates_with=f"{prefix}.hub", pressure_rating_pa=1_200_000.0,
             terminal=f"{prefix}.tire_skin.closed_volume")
        edge(f"pneumatics.outer_service_valve.{corner}",
             f"{prefix}.pneumatic_wheel_valve",
             f"{prefix}.pneumatic_rim_seat_port_outer", "traditional-rim-service-valve",
             radius=float(service_lines["pneumatic_hard_line_radius_m"]), palette="active",
             terminal=f"{prefix}.tire_skin.closed_volume",
             state_channel=f"tire_pressure_{corner}", check_valve=True,
             user_accessible=True, rotates_with=f"{prefix}.hub")
        edge(f"pneumatics.tube_stem_install.{corner}",
             f"{prefix}.pneumatic_tube_stem", f"{prefix}.pneumatic_wheel_valve",
             "tube-stem-to-rim-valve-install-binding",
             radius=float(service_lines["pneumatic_hard_line_radius_m"]), palette="active",
             enabled_when="tire_skin.pneumatic_mode == tube",
             synchronization="tube-stem-becomes-outer-service-valve-on-install",
             terminal=f"{prefix}.tire_skin.closed_volume")

        edge(f"brakes.hard_line.{corner}", "electrical.brake_master_manifold",
             f"{prefix}.brake_chassis_junction", "rigid-brake-hard-line",
             radius=float(service_lines["brake_hard_line_radius_m"]), palette="actuator-yellow",
             channel=f"service_brake_pressure_{corner}", routing="chassis-clipped-hard-line",
             pressure_rating_pa=24_000_000.0,
             mass_distribution={"total_mass_kg": float(
                 service_lines["brake_chassis_line_mass_kg"]) / 4,
                 "mass_node": f"{prefix}.brake_chassis_junction"})
        edge(f"brakes.service_hose.{corner}", f"{prefix}.brake_chassis_junction",
             f"{prefix}.brake_service_port", "flexible-brake-hose",
             radius=float(service_lines["brake_hose_radius_m"]), palette="actuator-yellow",
             channel=f"service_brake_pressure_{corner}",
             routing="suspension-and-steering-service-loop", slack_ratio=1.20,
             minimum_bend_radius_m=.045, pressure_rating_pa=24_000_000.0,
             mass_distribution={"total_mass_kg": float(
                 service_lines["brake_service_hose_mass_kg_each"]),
                 "mass_node": f"{prefix}.brake_service_port"})
        edge(f"brakes.caliper_feed.{corner}", f"{prefix}.brake_service_port",
             f"{prefix}.brake_caliper", "hydraulic-caliper-service-port",
             radius=float(service_lines["brake_hose_radius_m"]), palette="actuator-yellow",
             work_law="existing-friction-brake-torque-couple",
             loadout_contract="replaceable-caliper-must-accept-brake-fluid-pressure-port")

        edge(f"alignment.hard_line.{corner}", "electrical.alignment_manifold",
             f"{prefix}.alignment_chassis_junction", "rigid-alignment-hydraulic-line",
             radius=float(service_lines["alignment_hard_line_radius_m"]),
             palette="suspension-yellow", routing="chassis-clipped-hard-line",
             channel=f"alignment_supply_{corner}", pressure_rating_pa=18_000_000.0,
             mass_distribution={"total_mass_kg": float(
                 service_lines["alignment_chassis_line_mass_kg"]) / 4,
                 "mass_node": f"{prefix}.alignment_chassis_junction"})
        edge(f"alignment.service_loop.{corner}", f"{prefix}.alignment_chassis_junction",
             f"{prefix}.alignment_service_port", "flexible-alignment-hydraulic-hose",
             radius=float(service_lines["alignment_hose_radius_m"]),
             palette="suspension-yellow", channel=f"alignment_supply_{corner}",
             routing="suspension-and-steering-service-loop", slack_ratio=1.22,
             minimum_bend_radius_m=.045, pressure_rating_pa=18_000_000.0,
             terminal_policy="typed-port-awaits-loadout-device",
             mass_distribution={"total_mass_kg": float(
                 service_lines["alignment_service_loop_mass_kg_each"]),
                 "mass_node": f"{prefix}.alignment_service_port"})

        if corner.startswith("rear"):
            edge(f"brakes.parking_cable.{corner}", "electrical.parking_brake_equalizer",
                 f"{prefix}.parking_brake_service_port", "sheathed-parking-brake-cable",
                 radius=float(service_lines["parking_brake_cable_radius_m"]),
                 palette="actuator-yellow", tension_only=True,
                 channel=f"parking_brake_tension_{corner}", routing="chassis-guides-with-axle-loop",
                 slack_ratio=1.12, failure_response="service-brake-hydraulics-remain-independent",
                 mass_distribution={"total_mass_kg": float(
                     service_lines["parking_brake_cable_mass_kg"]) / 2,
                     "mass_node": f"{prefix}.parking_brake_service_port"})

    # One torsion stabilizer per axle connects the two lower arms. It is an
    # ordinary load-path member and therefore shares the same plastic/shear
    # damage contract as the wishbones, shocks, frame and transfer rods.
    for axle in ("front", "rear"):
        edge(f"suspension.{axle}.anti_roll_bar",
             f"suspension.{axle}_left.lower_ball_joint",
             f"suspension.{axle}_right.lower_ball_joint", "torsion-stabilizer",
             radius=.012, palette="suspension-yellow",
             torsional_stiffness_nm_per_rad=float(suspension["stiffness"]) * .018,
             force_path="left-lower-arm-to-right-lower-arm")
        edge(f"suspension.{axle}.coilover_tower_cross_brace",
             f"suspension.{axle}_left.coilover_chassis",
             f"suspension.{axle}_right.coilover_chassis", "rigid-distance",
             radius=.016, palette="rollbar-silver",
             force_path="left-coilover-tower-to-right-coilover-tower-and-cage")

    # Complete driver-to-upright steering topology.  The black ring and column
    # are control/drivetrain hardware; the yellow tie rods remain suspension
    # links.  Every member shares the same steering-angle coordinate.
    steering_center = [0.10, .40, -half_width * .30]
    steering_radius = .105
    node("steering.wheel.center", steering_center, "steering-wheel-hub",
         generalized_coordinate="steering_angle")
    ring_nodes = []
    for index in range(8):
        angle = 2 * math.pi * index / 8
        identity = f"steering.wheel.ring_{index}"
        ring_nodes.append(identity)
        node(identity, [steering_center[0],
                        steering_center[1] + steering_radius * math.cos(angle),
                        steering_center[2] + steering_radius * math.sin(angle)],
             "steering-wheel-rim", generalized_coordinate="steering_angle")
        edge(f"steering.wheel.spoke_{index}", "steering.wheel.center", identity,
             "rigid-steering-wheel-spoke", radius=.006, palette="drivetrain-black",
             generalized_coordinate="steering_angle")
    for index, identity in enumerate(ring_nodes):
        edge(f"steering.wheel.rim_{index}", identity, ring_nodes[(index + 1) % len(ring_nodes)],
             "rigid-steering-wheel-rim", radius=.007, palette="drivetrain-black",
             generalized_coordinate="steering_angle")
    left_rack = next(item for item in nodes
                     if item["identity"] == "suspension.front_left.steering_rack")
    right_rack = next(item for item in nodes
                      if item["identity"] == "suspension.front_right.steering_rack")
    rack_center = [(left_rack["reference_position"][axis]
                    + right_rack["reference_position"][axis]) * .5 for axis in range(3)]
    column_lower = [0.34, .23, -half_width * .18]
    pinion = [rack_center[0] - .08, rack_center[1] + .035, rack_center[2]]
    node("steering.column.lower", column_lower, "steering-column-universal",
         generalized_coordinate="steering_angle")
    node("steering.pinion", pinion, "steering-pinion",
         generalized_coordinate="steering_angle")
    node("steering.rack.center", rack_center, "steering-rack",
         generalized_coordinate="steering_angle")
    rear_left_rack = next(item for item in nodes
                          if item["identity"] == "suspension.rear_left.steering_rack")
    rear_right_rack = next(item for item in nodes
                           if item["identity"] == "suspension.rear_right.steering_rack")
    rear_rack_center = [(rear_left_rack["reference_position"][axis]
                         + rear_right_rack["reference_position"][axis]) * .5 for axis in range(3)]
    proportioner = [column_lower[0] + .07, column_lower[1] - .035, column_lower[2] + .03]
    rear_pinion = [rear_rack_center[0] + .08, rear_rack_center[1] + .035, rear_rack_center[2]]
    node("steering.proportioner", proportioner, "front-rear-steering-proportion-gearbox",
         generalized_coordinate="steering_front_rear_proportion")
    node("steering.rear_pinion", rear_pinion, "steering-pinion",
         generalized_coordinate="steering_angle_rear")
    node("steering.rear_rack.center", rear_rack_center, "steering-rack",
         generalized_coordinate="steering_angle_rear")
    edge("steering.column.upper", "steering.wheel.center", "steering.column.lower",
         "steering-torque-shaft", radius=.009, palette="drivetrain-black",
         generalized_coordinate="steering_angle")
    edge("steering.column.lower", "steering.column.lower", "steering.proportioner",
         "universal-joint-steering-shaft", radius=.008, palette="drivetrain-black",
         generalized_coordinate="steering_angle")
    edge("steering.proportioner.front", "steering.proportioner", "steering.pinion",
         "steering-torque-shaft", radius=.008, palette="drivetrain-black",
         generalized_coordinate="steering_angle_front")
    edge("steering.proportioner.rear", "steering.proportioner", "steering.rear_pinion",
         "steering-torque-shaft", radius=.008, palette="drivetrain-black",
         generalized_coordinate="steering_angle_rear")
    edge("steering.rack_and_pinion", "steering.pinion", "steering.rack.center",
         "rack-and-pinion-angle-to-translation", radius=.009, palette="drivetrain-black",
         generalized_coordinate="steering_angle")
    edge("steering.rack.left", "steering.rack.center",
         "suspension.front_left.steering_rack", "rack-translation", radius=.008,
         palette="drivetrain-black", generalized_coordinate="steering_angle")
    edge("steering.rack.right", "steering.rack.center",
         "suspension.front_right.steering_rack", "rack-translation", radius=.008,
         palette="drivetrain-black", generalized_coordinate="steering_angle")
    edge("steering.rear_rack_and_pinion", "steering.rear_pinion", "steering.rear_rack.center",
         "rack-and-pinion-angle-to-translation", radius=.009, palette="drivetrain-black",
         generalized_coordinate="steering_angle_rear")
    edge("steering.rear_rack.left", "steering.rear_rack.center",
         "suspension.rear_left.steering_rack", "rack-translation", radius=.008,
         palette="drivetrain-black", generalized_coordinate="steering_angle_rear")
    edge("steering.rear_rack.right", "steering.rear_rack.center",
         "suspension.rear_right.steering_rack", "rack-translation", radius=.008,
         palette="drivetrain-black", generalized_coordinate="steering_angle_rear")
    edge("steering.assist.motor_to_column", "electrical.steering_servo", "steering.column.lower",
         "steering-assist-torque-coupling", radius=.012, palette="active",
         generalized_coordinate="steering_angle", torque_authority="powered-servo-or-zero",
         failure_response="manual-column-remains-connected")

    engine_position = [float(value) for value in powertrain["engine_position"]]
    power_nodes = {
        "powertrain.engine": engine_position,
        "powertrain.clutch": [engine_position[0] + .15, engine_position[1], 0.0],
        "powertrain.pre_clutch_flywheel_wrench": [engine_position[0] + .09,
                                                    engine_position[1], 0.0],
        "powertrain.transmission": [engine_position[0] + .29, engine_position[1] - .015, 0.0],
        "powertrain.transfer_case": [engine_position[0] + .39, engine_position[1] - .035, 0.0],
        "powertrain.direct_drive_bypass": [engine_position[0] + .34, engine_position[1] - .025, .045],
        "powertrain.center_shaft": [-.12, .06, 0.0],
        "powertrain.front_differential": [axle_offset + wheelbase, .065, 0.0],
        "powertrain.rear_differential": [axle_offset - wheelbase, .065, 0.0],
        "powertrain.front_differential_brake": [axle_offset + wheelbase + .105, .065, 0.0],
        "powertrain.rear_differential_brake": [axle_offset - wheelbase - .105, .065, 0.0],
        "mount.engine_left": [engine_position[0], .075, -half_width * .58],
        "mount.engine_right": [engine_position[0], .075, half_width * .58],
        "mount.transmission_left": [engine_position[0] + .28, .055, -half_width * .52],
        "mount.transmission_right": [engine_position[0] + .28, .055, half_width * .52],
        "mount.transfer_case_left": [engine_position[0] + .39, .045, -half_width * .45],
        "mount.transfer_case_right": [engine_position[0] + .39, .045, half_width * .45],
    }
    graph_mass_names = {
        "powertrain.engine": "engine", "powertrain.transmission": "transmission",
        "powertrain.transfer_case": "transfer_case",
        "powertrain.front_differential": "front_differential",
        "powertrain.rear_differential": "rear_differential",
    }
    for identity, position in power_nodes.items():
        differential_brake = identity.endswith("_differential_brake")
        node(identity, position, "powertrain-mount" if identity.startswith("mount.") else
             "pre-clutch-rotating-six-axis-wrench-port"
             if identity.endswith("pre_clutch_flywheel_wrench") else
             "differential-driveline-brake" if identity.endswith("_differential_brake") else "rotating-mass",
             fixed_to="chassis" if identity.startswith("mount.") else None,
             mass_kg=(22.0 if differential_brake else
                      component_masses.get(graph_mass_names.get(identity, ""), 0.0)),
             mass_in_total=identity in graph_mass_names,
             polar_inertia_kg_m2=(
                 float(drivetrain["differential_brake_rotor_inertia_kg_m2"])
                 if differential_brake else None),
             inertia_axis="axle-input-shaft" if differential_brake else None,
             mass_integration_status=(
                 "declared-part-budget-existing-lumped-vehicle-mass-remains-authoritative"
                 if differential_brake else None))
    edge("drivetrain.engine_to_alternator_cvt", "powertrain.engine",
         "electrical.alternator_cvt", "direct-torque-shaft", radius=.009,
         palette="drivetrain-black",
         torque_channels=["alternator_reaction_torque_nm",
                          "accessory_motor_engine_reaction_torque_nm",
                          "compressor_engine_reaction_torque_nm"],
         torque_reduction="signed-sum-at-shared-accessory-block-shaft",
         drive="no-belt")
    edge("drivetrain.engine_to_pre_clutch_flywheel_wrench", "powertrain.engine",
         "powertrain.pre_clutch_flywheel_wrench", "torque-shaft-wrench-extension",
         radius=.018, palette="drivetrain-black", frame="engine-crank-before-main-clutch",
         inertia_coordinate="external_engine_flywheel_inertia",
         transfer="force-moment-angular-position-and-angular-velocity")
    edge("electrical.wire.tcu_bypass_actuator", "electrical.tcu",
         "powertrain.direct_drive_bypass", "relaxed-insulated-copper-control-harness",
         radius=.0025, palette="drivetrain-black",
         command_coordinate="direct_drive_bypass_command",
         interlock_feedback=["engine_angular_speed", "differential_wrench_shaft_omega_front",
                             "differential_wrench_shaft_omega_rear", "clutch_torque"],
         reaction="electrical-command-only-mechanical-wrench-remains-in-dog-clutch-edge")
    edge("drivetrain.alternator_cvt_to_bank", "electrical.alternator_cvt",
         "electrical.alternator", "continuously-variable-torque-shaft", radius=.008,
         palette="drivetrain-black", ratio_coordinate="alternator_cvt_ratio_state",
         efficiency=float(electrical["alternator_cvt_efficiency"]),
         wear_coordinate="alternator_cvt_wear", glaze_coordinate="alternator_cvt_glaze",
         torque_channels=["alternator_reaction_torque_nm",
                          "accessory_motor_shaft_torque_nm",
                          "compressor_shaft_reaction_torque_nm"],
         bidirectional_motor_bus="accessory-battery-cube")
    for axle, sign in (("front", 1.0), ("rear", -1.0)):
        brake_position = next(item["reference_position"] for item in nodes
                              if item["identity"] == f"powertrain.{axle}_differential_brake")
        wrench_position = [brake_position[0] + sign * .14,
                           brake_position[1], brake_position[2]]
        node(f"powertrain.{axle}_differential_brake_wrench", wrench_position,
             "rotating-six-axis-drivetrain-wrench-port",
             generalized_coordinate=f"{axle}_differential_brake_shaft_angle",
             accepts="drivetrain-or-accessory-six-axis-wrench",
             frame="rotating-differential-brake-output-shaft",
             maximum_torque_nm=float(drivetrain["differential_brake_torque_nm"]),
             future_loadout_port=True)
        edge(f"drivetrain.{axle}_differential_brake_shaft_extension",
             f"powertrain.{axle}_differential_brake",
             f"powertrain.{axle}_differential_brake_wrench",
             "torque-shaft-wrench-extension", radius=.016, palette="drivetrain-black",
             torque_channel=f"{axle}_differential_brake_torque",
             transfer="force-moment-angular-position-and-angular-velocity",
             torsional_yield_torque_nm=5200.0, torsional_fracture_torque_nm=7600.0)
    torque_edges = (
        ("engine_to_clutch", "powertrain.engine", "powertrain.clutch", "engine_torque"),
        ("clutch_to_transmission", "powertrain.clutch", "powertrain.transmission", "clutch_torque"),
        ("transmission_to_transfer_case", "powertrain.transmission", "powertrain.transfer_case", "transmission_output_torque"),
        ("transfer_case_to_shaft", "powertrain.transfer_case", "powertrain.center_shaft", "driveline_torque"),
        ("shaft_to_front_diff", "powertrain.center_shaft", "powertrain.front_differential", "front_differential_torque"),
        ("shaft_to_rear_diff", "powertrain.center_shaft", "powertrain.rear_differential", "rear_differential_torque"),
    )
    for name, a, b, channel in torque_edges:
        edge(f"drivetrain.{name}", a, b, "torque-shaft", radius=.011,
             palette="drivetrain-black", torque_channel=channel)
    edge("drivetrain.direct_drive_bypass", "powertrain.engine", "powertrain.transmission",
         "synchronized-positive-dog-clutch-bypass", radius=.013, palette="drivetrain-black",
         engagement_coordinate="direct_drive_bypass_engagement",
         command_coordinate="direct_drive_bypass_command",
         tooth_health_coordinate="direct_drive_bypass_tooth_health",
         torque_channel="direct_drive_bypass_torque_nm",
         interlock="low-relative-speed-and-unloaded-dog-teeth-before-engagement",
         bypasses="main-friction-clutch-slip-path")
    for axle in ("front", "rear"):
        edge(f"drivetrain.{axle}_differential_brake",
             f"powertrain.{axle}_differential", f"powertrain.{axle}_differential_brake",
             "friction-brake-torque-couple", radius=.018, palette="suspension-yellow",
             torque_channel=f"{axle}_differential_brake_torque",
             command=f"{axle}_differential_brake",
             modulation="abs-authority-before-axle-differential",
             reaction_path="differential-housing-to-axle-and-chassis",
             rotor_radius_m=.082, rotor_mass_kg=22.0,
             rotor_polar_inertia_kg_m2=float(
                 drivetrain["differential_brake_rotor_inertia_kg_m2"]),
             angular_velocity_coordinate=f"({axle}_left_wheel_omega+{axle}_right_wheel_omega)/2",
             momentum_integration_status="integrated-live-in-canonical-driveline-mass-matrix")
    for axle in ("front", "rear"):
        for lateral in ("left", "right"):
            corner = f"{axle}_{lateral}"
            edge(f"drivetrain.{corner}_halfshaft", f"powertrain.{axle}_differential",
                 f"suspension.{corner}.halfshaft_joint", "constant-velocity-torque-shaft", radius=.009,
                 palette="drivetrain-black", torque_channel=f"wheel_torque_{corner}",
                 torsional_yield_torque_nm=4200.0, torsional_fracture_torque_nm=6500.0,
                 failure_response="open-halfshaft-requiring-locker-to-route-torque-to-intact-side")
    for component, mounts in (("engine", ("engine_left", "engine_right")),
                              ("transmission", ("transmission_left", "transmission_right")),
                              ("transfer_case", ("transfer_case_left", "transfer_case_right"))):
        for mount in mounts:
            edge(f"mount.{component}.{mount}", f"powertrain.{component}", f"mount.{mount}",
                 "six-axis-compliant-mount", radius=.012, palette="drivetrain-black",
                 transfer="force-and-moment-to-chassis")

    # A reusable routed tension actuator. The cable does not pretend to be a
    # rigid tie rod: its fixed guides define a bendable route, the inner cable
    # transmits tension only, and a small table maps pedal travel to throttle-
    # lever travel. This control-rate graph is deliberately outside the tire/
    # contact kernel; the already filtered throttle coordinate drives it.
    throttle_points = {
        "controls.throttle.pedal": [.03, .23, -half_width * .25],
        "controls.throttle.guide_cabin": [.18, .24, -half_width * .31],
        "controls.throttle.guide_firewall": [.30, .20, -half_width * .27],
        "powertrain.intake.plenum": [engine_position[0] - .015, .205, 0.0],
        "powertrain.intake.throttle_body": [engine_position[0] + .075, .205, -half_width * .13],
        "powertrain.intake.throttle_lever": [engine_position[0] + .075, .225, -half_width * .18],
    }
    for identity, position in throttle_points.items():
        node(identity, position,
             "cable-guide" if ".guide_" in identity else
             "table-actuator-input" if identity.endswith("pedal") else
             "throttle-lever" if identity.endswith("lever") else "intake-component",
             fixed_to="chassis", generalized_coordinate=(
                 "throttle_cable_travel" if identity.endswith(("pedal", "lever")) else None))
    cable_route = (
        "controls.throttle.pedal", "controls.throttle.guide_cabin",
        "controls.throttle.guide_firewall", "powertrain.intake.throttle_lever",
    )
    for index, (a, b) in enumerate(zip(cable_route, cable_route[1:])):
        edge(f"actuator.throttle.cable_{index}", a, b, "routed-tension-cable",
             radius=.004, palette="actuator-yellow", tension_only=True,
             stretch_coefficient_m_per_n=1.8e-5, bend_relaxation="guide-node-catenary-low-rate",
             force_path="pedal-inner-cable-to-throttle-return-spring")
    edge("actuator.throttle.lever", "powertrain.intake.throttle_body",
         "powertrain.intake.throttle_lever", "table-actuator-linear-link",
         radius=.006, palette="actuator-yellow", generalized_coordinate="throttle_cable_travel",
         joint_a="revolute", joint_b="cable-eye",
         travel_table=[[0.0, 0.0], [.18, .006], [.50, .019], [.78, .030], [1.0, .038]],
         return_spring_n_per_m=880.0, viscous_drag_n_s_per_m=3.2,
         force_law="positive-tension-minus-elastic-stretch-and-return-spring")
    energy_points = {
        "energy.storage": [-half_length * .48, .18, 0.0],
        "energy.delivery.guide_rear": [-half_length * .26, .14, -half_width * .30],
        "energy.delivery.guide_front": [engine_position[0] - .12, .16, -half_width * .30],
        "powertrain.fuel_rail": [engine_position[0], .19, -half_width * .18],
        "powertrain.intake.air_filter": [engine_position[0] - .12, .22, half_width * .25],
        "powertrain.exhaust.header_left": [engine_position[0], .14, -half_width * .27],
        "powertrain.exhaust.header_right": [engine_position[0], .14, half_width * .27],
    }
    for identity, position in energy_points.items():
        node(identity, position,
             "interchangeable-energy-storage" if identity == "energy.storage" else
             "delivery-route-guide" if ".guide_" in identity else
             "fuel-rail" if identity.endswith("fuel_rail") else
             "air-intake" if "air_filter" in identity else "exhaust-header",
             fixed_to="chassis")
    energy_route = ("energy.storage", "energy.delivery.guide_rear",
                    "energy.delivery.guide_front", "powertrain.fuel_rail")
    for index, (a, b) in enumerate(zip(energy_route, energy_route[1:])):
        edge(f"energy.delivery.segment_{index}", a, b, "routed-energy-line",
             radius=.005, palette="engine-accent", route_support="bend-guides",
             medium_rate_state="fuel-pressure-flow-or-voltage-current-selected-by-power-unit")
    edge("powertrain.intake.air_path", "powertrain.intake.air_filter",
         "powertrain.intake.plenum", "intake-flow-path", radius=.014,
         palette="engine-metal", medium_rate_state="air-mass-flow")
    for side in ("left", "right"):
        edge(f"powertrain.exhaust.{side}", "powertrain.engine",
             f"powertrain.exhaust.header_{side}", "exhaust-flow-path", radius=.012,
             palette="engine-accent", medium_rate_state="exhaust-pulse-pressure-and-temperature")
    corner_loads = {}
    gravity = abs(float(source["world"]["gravity"]))
    for corner in WHEEL_NAMES:
        coilover = next(item for item in edges if item["identity"] == f"suspension.{corner}.coilover")
        a = next(item["reference_position"] for item in nodes if item["identity"] == coilover["a"])
        b = next(item["reference_position"] for item in nodes if item["identity"] == coilover["b"])
        motion_ratio = abs(a[1] - b[1]) / max(1e-9, float(coilover["rest_length"]))
        mass_kg = config.sprung_mass() * float(source["mass_distribution"][corner])
        unsprung_mass_kg = config.unsprung_mass_per_corner()
        static_load = mass_kg * gravity
        corner_loads[corner] = {
            "design_supported_mass_kg": mass_kg,
            "design_static_load_n": static_load,
            "unsprung_mass_kg": unsprung_mass_kg,
            "design_terrain_load_n": (mass_kg + unsprung_mass_kg) * gravity,
            "linkage_motion_ratio": motion_ratio,
            "design_spring_compression_m": min(float(suspension["travel"]), static_load / (
                float(suspension["stiffness"]) * motion_ratio ** 2)),
        }
    configured_front = sum(float(source["mass_distribution"][name])
                           for name in ("front_left", "front_right"))
    load_audit = {
        **mass_properties,
        "configured_axle_fractions": {"front": configured_front, "rear": 1 - configured_front},
        "configured_vs_derived_front_fraction_error": configured_front - mass_properties[
            "derived_axle_fractions"]["front"],
        "corners": corner_loads,
        "spring_load_sum_n": sum(item["design_static_load_n"] for item in corner_loads.values()),
    }
    for member in edges:
        if member["palette_role"] == "suspension-yellow":
            member["subsystem_class"] = "articulation-linkage"
            member["end_behavior"] = "pivoting-or-spherical-as-declared"
        elif member["palette_role"] == "actuator-yellow":
            member["subsystem_class"] = "routed-control-actuator"
            member["end_behavior"] = "tension-only-cable-or-table-driven-lever"
        elif member["palette_role"] == "drivetrain-black":
            member["subsystem_class"] = "rotary-transmission"
        else:
            member["subsystem_class"] = "load-bearing-structure"
    # Keep the authored offsets needed to rebuild the graph when wheelbase or
    # chassis length changes.  Axle-related nodes translate with their axle;
    # all other equipment/body nodes retain a normalized chassis-longitudinal
    # location.  Runtime changes therefore rebuild rest geometry instead of
    # stretching a rendered shell independently of its load graph.
    for item in nodes:
        identity = str(item["identity"])
        axle = next((name for name in ("front", "rear")
                     if identity.startswith(f"suspension.{name}_") or
                     identity.startswith(f"powertrain.{name}_differential")), None)
        authority = item.get("longitudinal_authority")
        if axle or authority == "wheelbase":
            sign = 1.0 if (axle == "front" or ".front_" in identity) else -1.0
            item["longitudinal_parameterization"] = {
                "authority": "axle-group-offset-plus-wheelbase-half-length", "sign": sign,
                "offset_m": float(item["reference_position"][0]) - axle_offset - sign * wheelbase,
            }
        else:
            item["longitudinal_parameterization"] = {
                "authority": "chassis-half-length",
                "fraction": float(item["reference_position"][0]) / max(1e-9, half_length),
            }
    return {
        "schema": "abstract-ui-mechanical-wrench-graph-v1",
        "authority": "json-parameters-expanded-by-python-compiler",
        "coordinate_system": "chassis-local-x-forward-y-up-z-right",
        "state_law": "node-force-and-node-moment-reduced-through-edge-constraints",
        "wheel_placement_and_mounts": wheel_mount_solution,
        "body_packaging": body_packaging,
        "chassis_reference_plane": {
            "corner_identities": [f"frame.{corner}" for corner in WHEEL_NAMES],
            "identity_policy": "persistent-authored-hardpoints",
            "pose_policy": "plane-is-reconstructed-from-current-solved-corner-positions",
            "deformation_policy": "identities-do-not-move-to-a-different-node-when-frame-yields",
        },
        "wrench_attachment_api": {
            "schema": "generic-six-axis-wrench-attachment-v1",
            "attachment_identities": [f"attachment.{corner}" for corner in WHEEL_NAMES],
            "required_payload_fields": [
                "mass_kg", "center_of_mass_local", "inertia_kg_m2",
                "peak_force_local_n", "peak_moment_local_nm", "mount_transform",
            ],
            "admission": "payload-force-and-moment-envelope-must-fit-before-graph-link",
            "accounting": "accepted-payload-mass-com-and-inertia-enter-load-audit",
            "failure": "mount-yield-and-fracture-open-the-edge-without-erasing-payload-mass",
        },
        "service_port_api": {
            "schema": "typed-vehicle-service-port-v1",
            "device_policy": "routing-and-fitting-exist-now-work-device-is-declared-by-loadout",
            "networks": {
                "tire_pressure": {
                    "source": "electrical.pneumatic_tire_manifold",
                    "route": "chassis-hard-line-to-suspension-loop-to-knuckle-stator-to-bearing-rotor-to-dual-rim-seat-ports",
                    "terminals": [f"suspension.{corner}.pneumatic_wheel_valve"
                                  for corner in WHEEL_NAMES],
                    "bearing_terminals": [f"suspension.{corner}.pneumatic_bearing_rotor"
                                          for corner in WHEEL_NAMES],
                    "rim_seat_terminals": [
                        f"suspension.{corner}.pneumatic_rim_seat_port_{seat}"
                        for corner in WHEEL_NAMES for seat in ("inner", "outer")],
                    "tube_install_binding": "tube-stem-becomes-traditional-outer-rim-valve",
                    "tractor_default": "bearing-feed-and-user-accessible-outer-rim-valve-both-live",
                    "medium": "pressurized-gas",
                },
                "service_brake": {
                    "source": "electrical.brake_master_manifold",
                    "terminals": [f"suspension.{corner}.brake_service_port"
                                  for corner in WHEEL_NAMES],
                    "medium": "brake-fluid-pressure",
                },
                "parking_brake": {
                    "source": "electrical.parking_brake_equalizer",
                    "terminals": [f"suspension.rear_{side}.parking_brake_service_port"
                                  for side in ("left", "right")],
                    "medium": "tension-only-inner-cable",
                },
                "alignment": {
                    "source": "electrical.alignment_manifold",
                    "terminals": [f"suspension.{corner}.alignment_service_port"
                                  for corner in WHEEL_NAMES],
                    "medium": "alignment-hydraulic-pressure",
                },
            },
            "mass_accounting": "hard-lines-are-sprung-service-loops-unions-valves-and-terminals-are-unsprung",
        },
        "drivetrain_wrench_api": {
            "schema": "rotating-six-axis-drivetrain-wrench-port-v1",
            "ports": ["powertrain.pre_clutch_flywheel_wrench",
                      *(f"powertrain.{axle}_differential_brake_wrench"
                        for axle in ("front", "rear"))],
            "state": ["force_xyz", "moment_xyz", "shaft_angle", "shaft_angular_velocity"],
            "placement": "exposed-end-of-extended-differential-brake-shaft",
            "loadout_policy": "future-accessory-must-declare-mass-inertia-and-wrench-envelope",
            "external_hub_torque_inputs": [f"external_hub_torque_{corner}"
                                           for corner in WHEEL_NAMES],
            "hub_input_law": "external-torque-enters-wheel-inertia-never-commanded-wheel-speed",
            "differential_wrench_torque_inputs": [
                f"external_differential_wrench_torque_{axle}" for axle in ("front", "rear")],
            "differential_wrench_inertia_inputs": [
                f"external_differential_inertia_{axle}" for axle in ("front", "rear")],
            "wheel_end_disconnects": [f"hub_locker_engagement_{corner}"
                                       for corner in WHEEL_NAMES],
            "disconnect_law": (
                "locking-hub-clutch-ring-opens-halfshaft-to-wheel-torque-path-while-"
                "hub-side-wrench-and-service-brake-remain-on-wheel"),
            "field_sequence": [
                "remove-driveline-torque", "stop-vehicle-and-apply-service-brakes",
                "disengage-wheel-end-locking-hubs", "test-differential-side-wrench",
                "remove-test-torque-and-stop-halfshafts", "engage-locking-hubs-at-zero-slip",
                "release-service-brakes", "prove-wheel-drive-from-differential-wrench",
            ],
        },
        "friction_pack_state_api": {
            "schema": "vehicle-friction-pack-state-v1",
            "state_per_pack": ["glaze", "wear", "slip_power"],
            "thermal_state": "main-clutch-explicit;other-pack-thermal-capacity-is-next-material-detail",
            "packs": ["main_clutch", "alternator_cvt",
                      *(f"hub_locker_{corner}" for corner in WHEEL_NAMES),
                      *(f"differential_locker_{axle}" for axle in ("front", "rear", "center"))],
            "law": "slip-work-wears-pack-high-slip-power-builds-glaze-friction-capacity-falls",
        },
        "rotating_accessory_presets": rotating_accessory_presets,
        "static_accessory_presets": static_accessory_presets,
        "validator_default_accessory_loadout": {
            "purpose": "asymmetric-post-accessory-ballast-proof",
            "items": [
                {"preset": "industrial-high-pressure-gas-cylinder",
                 "identity": "air_mix_reserve_left",
                 "mount_ports": ["attachment.front_left", "attachment.rear_left"]},
                {"preset": "direct-drive-high-pressure-compressor",
                 "identity": "air_mix_compressor_right",
                 "mount_ports": ["electrical.alternator_cvt", "electrical.alternator"],
                 "drive_port": "electrical.alternator_cvt"},
                {"preset": "reversible-accessory-block-motor",
                 "identity": "alternator_block_motor",
                 "mount_ports": ["electrical.alternator_cvt", "electrical.alternator"],
                 "drive_port": "electrical.alternator_cvt"},
                {"preset": "reversible-accessory-block-motor",
                 "identity": "compressor_block_motor",
                 "mount_ports": ["electrical.alternator_cvt", "electrical.alternator"],
                 "drive_port": "electrical.alternator_cvt"},
                {"preset": "four-lead-acid-battery-cube",
                 "identity": "accessory_battery_cube",
                 "mount_ports": ["frame.center"]},
            ],
        },
        "leveling_controller": {
            "maximum_corner_offset_m": .62,
            "actuation": "per-edge-rest-length-through-existing-suspension-force-graph",
            "startup_enabled": False,
        },
        "nodes": nodes, "edges": edges, "membranes": membranes, "load_audit": load_audit,
        "steering_system": {
            "topology": "steering-wheel-column-proportioner-two-pinions-two-racks-four-knuckles",
            "input": "steering-wheel-wrench",
            "output": "rack-force-to-tie-rod-to-knuckle-moment",
            "axle_selection": "independent-front-and-rear-with-continuous-proportion",
            "disconnected_behavior": "knuckle-steer-coordinate-free-with-contact-caster-damping",
            "default": "both-axles-active-counter-phase",
            "electronic_assist": "ecu-speed-sensitive-column-command-rate-with-mechanical-fallback",
        },
        "actuator_family": {
            alignment_strain_relief_definition["identity"]: alignment_strain_relief_definition,
            knuckle_break_bushing_definition["identity"]: knuckle_break_bushing_definition,
        },
        "control_actuators": [{
            "identity": "actuator.throttle",
            "kind": "table-routed-tension-actuator",
            "input": "filtered-throttle-pedal-travel",
            "route": list(cable_route),
            "output": "powertrain.intake.throttle_lever",
            "evaluation_rate": "control-rate-low-cost",
            "constitutive_law": "tension-only-elastic-inner-cable-with-return-spring",
            "hydraulic_reuse_boundary": "route-geometry-only-pressure-flow-compliance-require-a-distinct-law",
        }],
        "wheel_end_system": {
            "topology": ("balloon-skin-vertices-faces-two-bead-rings-rim-rotor-hub-bearing-"
                         "knuckle-sockets-replaceable-break-bushings-control-arms"),
            "tire_model": "compiled-balloon-skin-v1",
            "tire_state_scalars_per_wheel": len(tire_skin_abi["state"]),
            "tire_collision_authority": tire_skin_abi["collision_authority"],
            "bearing_constraint": "five-axis-structural-support-with-one-free-spin-axis",
            "brake_reaction": "rotor-friction-pair-to-caliper-knuckle-upright-and-control-arms",
            "angular_momentum": "wheel-rotor-bearing-polar-inertia-reacts-gyroscopically-through-knuckle",
            "alignment_relief": alignment_strain_relief_definition["identity"],
            "knuckle_mechanical_fuse": knuckle_break_bushing_definition["identity"],
            "integration_status": "declared-graph-contract-current-chassis-kernel-uses-lumped-wheel-end-wrench",
        },
        "energy_system": {
            "topology": "interchangeable-storage-through-routed-delivery-to-power-unit",
            "storage_node": "energy.storage", "delivery_route": list(energy_route),
            "reservoir_manifold": {
                "supported_modes": ["primary", "reserve-sequence", "metered-blend"],
                "controls": ["primary-selection", "reserve-changeover", "carrier-mix-fraction"],
                "compatibility_rule": "carrier-properties-must-be-solved-before-flow-can-affect-cylinder-charge",
                "unsafe_example": "nitromethane-in-gasoline-calibration-causes-knock-or-failure-not-free-power",
            },
            "combustion_outputs": ["air-mass-flow", "fuel-mass-flow", "mixture-lambda",
                                   "cylinder-charge", "header-pulse-state"],
            "electric_outputs": ["bus-voltage", "phase-current", "inverter-loss", "shaft-torque"],
            "mass_rule": "storage-shell-and-live-carrier-contribute-to-chassis-mass-center-and-inertia",
        },
        "execution_bands": {
            "fast_critical_120_hz": ["contact-inequalities", "tire-force-integral",
                                      "wheel-and-driveline-angular-momentum", "chassis-wrench-step"],
            "medium_critical_30_to_60_hz": ["routed-control-actuators", "mixture-or-inverter-state",
                                             "cylinder-or-pole-events", "mount-vibration-envelope"],
            "audio_48_khz_non_authoritative": ["pcm-from-published-engine-event-state"],
            "slow_1_to_10_hz": ["fuel-or-charge-inventory", "thermal-state", "wear-and-workshop-state"],
            "handoff": "versioned-common-buffers-with-single-writer-fields-and-no-dom-authority",
        },
        "constraint_reduction": {
            "suspension": "double-wishbone-four-bar-plus-coilover-motion-ratio",
            "contact": "tire-patch-and-cage-node/member-midpoint-terrain-wrenches-to-chassis",
            "powertrain": "shaft-torque-and-six-axis-mount-reactions",
            "chassis": "sum-node-force-and-position-cross-force-plus-node-moment",
        },
    }


def vehicle_slot_model(root: str, actor: str) -> dict[str, Any]:
    config = load_default_car_configuration()
    chassis = config.source["chassis"]
    wheels = config.source["wheels"]
    axle_offset = float(wheels["axle_group_offset_x_m"])
    node_positions = {
        "front_left": [axle_offset + float(wheels["wheelbase_half_length"]), -float(chassis["clearance"]),
                       -float(wheels["track_half_width"])],
        "front_right": [axle_offset + float(wheels["wheelbase_half_length"]), -float(chassis["clearance"]),
                        float(wheels["track_half_width"])],
        "rear_left": [axle_offset - float(wheels["wheelbase_half_length"]), -float(chassis["clearance"]),
                       -float(wheels["track_half_width"])],
        "rear_right": [axle_offset - float(wheels["wheelbase_half_length"]), -float(chassis["clearance"]),
                       float(wheels["track_half_width"])],
    }
    member_pairs = (
        ("front_left", "front_right"), ("front_right", "rear_right"),
        ("rear_right", "rear_left"), ("rear_left", "front_left"),
        ("front_left", "rear_right"), ("front_right", "rear_left"),
    )
    chassis_structure = {
        "schema": "abstract-ui-stick-ball-chassis-v0",
        "model": "elastic-plastic-breakable-pipe-members-with-compliant-suspension-at-nodes",
        "material": {"name": "steel", "youngs_modulus_pa": 200_000_000_000.0,
                     "density_kg_m3": 7850.0, "yield_strength_pa": 250_000_000.0,
                     "solver_interpretation": "finite-axial-stiffness-yield-plastic-set-and-open-fracture"},
        "nodes": [{"identity": name, "local_position": position,
                   "contact_patch": name, "force_application_point": True}
                  for name, position in node_positions.items()],
        "members": [{"a": a, "b": b, "constraint": "rigid-distance",
                     "rest_length": sum((node_positions[a][axis] - node_positions[b][axis]) ** 2
                                        for axis in range(3)) ** .5}
                    for a, b in member_pairs],
        "spring_law": "compiled-hooke-plus-velocity-damping-per-contact-node",
        "pose_reduction": "sum-node-forces-and-r-cross-f-then-compiled-rigid-chassis-step",
    }
    mechanical_graph = _vehicle_mechanical_graph(config)
    chassis_member_edges = [edge for edge in mechanical_graph["edges"]
                            if edge["identity"].startswith(("frame.", "cage."))]

    def chassis_profile(identity: str, label: str, material: str, *, outer_diameter_mm: float,
                        wall_thickness_mm: float, density: float, youngs_modulus: float,
                        yield_strength: float, shear_strength: float) -> dict[str, Any]:
        outer = outer_diameter_mm / 1000
        wall = wall_thickness_mm / 1000
        inner = outer - 2 * wall
        area = math.pi * (outer ** 2 - inner ** 2) / 4
        second_moment = math.pi * (outer ** 4 - inner ** 4) / 64
        total_length = sum(float(edge["rest_length"]) for edge in chassis_member_edges)
        return {
            "identity": identity, "label": label, "material": material,
            "outer_diameter_m": outer, "wall_thickness_m": wall,
            "section_area_m2": area, "second_moment_area_m4": second_moment,
            "density_kg_m3": density, "youngs_modulus_pa": youngs_modulus,
            "yield_strength_pa": yield_strength, "shear_strength_pa": shear_strength,
            "joint_efficiency": .72,
            "member_length_m": total_length, "member_mass_kg": total_length * area * density,
            "axial_yield_force_n": area * yield_strength * .72,
            "shear_force_limit_n": area * shear_strength * .72,
            "solver_policy": "profile-swaps-physical-member-limits-mass-and-rigid-body-inertia",
        }

    chassis_profiles = [
        chassis_profile("chromoly-4130-38x2", "38.1 × 2.4 mm · 4130 chromoly", "AISI 4130 chromoly",
                        outer_diameter_mm=38.1, wall_thickness_mm=2.4, density=7850,
                        youngs_modulus=205_000_000_000, yield_strength=435_000_000,
                        shear_strength=260_000_000),
        chassis_profile("dom-44x3", "44.5 × 3.0 mm · DOM steel", "1020 DOM steel",
                        outer_diameter_mm=44.5, wall_thickness_mm=3.0, density=7850,
                        youngs_modulus=205_000_000_000, yield_strength=350_000_000,
                        shear_strength=210_000_000),
        chassis_profile("mild-51x3", "50.8 × 3.2 mm · mild steel", "A36 mild steel",
                        outer_diameter_mm=50.8, wall_thickness_mm=3.2, density=7850,
                        youngs_modulus=200_000_000_000, yield_strength=250_000_000,
                        shear_strength=145_000_000),
        chassis_profile("aluminum-6061-64x5", "63.5 × 4.8 mm · 6061-T6 aluminum", "6061-T6 aluminum",
                        outer_diameter_mm=63.5, wall_thickness_mm=4.8, density=2700,
                        youngs_modulus=69_000_000_000, yield_strength=276_000_000,
                        shear_strength=207_000_000),
    ]
    default_chassis_profile = next(item for item in chassis_profiles if item["identity"] == "dom-44x3")
    for edge in chassis_member_edges:
        edge["chassis_profile_member"] = True
        edge["radius"] = default_chassis_profile["outer_diameter_m"] / 2
        edge["mass_kg"] = (float(edge["rest_length"])
                           * default_chassis_profile["section_area_m2"]
                           * default_chassis_profile["density_kg_m3"])
        edge["mass_in_total"] = True
        edge["mass_accounting"] = "allocated-within-frame-cage-driver-residual"
        edge["broken_mass_policy"] = "mass-remains-split-between-endpoint-nodes-after-fracture"
        if edge.get("damage"):
            edge["damage"].update({
                "material": default_chassis_profile["material"],
                "section_area_m2": default_chassis_profile["section_area_m2"],
                "youngs_modulus_pa": default_chassis_profile["youngs_modulus_pa"],
                "axial_stiffness_n_per_m": default_chassis_profile["youngs_modulus_pa"]
                * default_chassis_profile["section_area_m2"] / max(1e-9, float(edge["rest_length"])),
                "yield_strength_pa": default_chassis_profile["yield_strength_pa"],
                "shear_strength_pa": default_chassis_profile["shear_strength_pa"],
                "axial_yield_force_n": default_chassis_profile["axial_yield_force_n"],
                "shear_force_limit_n": default_chassis_profile["shear_force_limit_n"],
            })
    def power_unit_preset(identity: str, label: str, kind: str, *, displacement: float,
                          bmep: float, braking_bmep: float, idle: float, torque_peak: float,
                          power_peak: float, redline: float, inertia: float, mass: float,
                          clutch_torque: float, combustion_efficiency: float,
                          coupling_efficiency: float,
                          preferred_fuel: str = "pump-gasoline-93",
                          compatible_fuels: tuple[str, ...] = ("pump-gasoline-93", "nitromethane-race"),
                          fuel_compatibility: dict[str, float] | None = None,
                          preferred_ignition: str = "gasoline-distributor",
                          ignition_compatibility: dict[str, float] | None = None,
                          mechanical_roles: tuple[str, ...] = ("propulsion",)) -> dict[str, Any]:
        rpm_to_rad_s = 2 * math.pi / 60
        return {
            "identity": identity, "label": label, "kind": kind,
            "curve_reference": f"springtail-power-unit-curves-v0:{identity}",
            "curve_storage": "immutable-parameter-pack-selected-by-reference",
            "mechanical_roles": list(mechanical_roles),
            "preferred_fuel_profile": preferred_fuel,
            "compatible_fuel_profiles": list(compatible_fuels),
            "fuel_compatibility": dict(fuel_compatibility or {name: 1.0 for name in compatible_fuels}),
            "preferred_ignition_profile": preferred_ignition,
            "ignition_compatibility": dict(ignition_compatibility or {
                "gasoline-distributor": 1.0, "nitromethane-ecu-race": 1.0,
                "nitromethane-magneto": 1.0,
            }),
            "configuration": {"displacement_liters": displacement,
                              "brake_mean_effective_pressure_pa": bmep,
                              "engine_braking_mean_effective_pressure_pa": braking_bmep,
                              "idle_rpm": idle, "torque_peak_rpm": torque_peak,
                              "power_peak_rpm": power_peak, "redline_rpm": redline,
                              "engine_rotating_inertia_kg_m2": inertia,
                              "engine_mass_kg": mass, "clutch_maximum_torque_nm": clutch_torque,
                              "combustion_efficiency": combustion_efficiency,
                              "clutch_efficiency": coupling_efficiency},
            "parameters": {"engine_displacement_m3": displacement / 1000,
                           "brake_mean_effective_pressure": bmep,
                           "engine_braking_mean_effective_pressure": braking_bmep,
                           "engine_idle_angular_speed": idle * rpm_to_rad_s,
                           "engine_torque_peak_angular_speed": torque_peak * rpm_to_rad_s,
                           "engine_power_peak_angular_speed": power_peak * rpm_to_rad_s,
                           "engine_redline_angular_speed": redline * rpm_to_rad_s,
                           "engine_rotating_inertia": inertia, "engine_mass": mass,
                           "clutch_maximum_torque": clutch_torque,
                           "combustion_efficiency": combustion_efficiency,
                           "clutch_efficiency": coupling_efficiency},
        }
    power_unit_presets = [
        power_unit_preset("amc-258-jeep-i6", "AMC-era Jeep 258 ci trail I6", "combustion",
                          displacement=4.227, bmep=835_000, braking_bmep=135_000, idle=650,
                          torque_peak=1800, power_peak=3200, redline=4400, inertia=.68,
                          mass=220, clutch_torque=235, combustion_efficiency=.84,
                          coupling_efficiency=.90),
        power_unit_preset("honda-style-commuter-i4-1500", "Honda-style 1.5 L commuter I4", "combustion",
                          displacement=1.5, bmep=1_160_000, braking_bmep=145_000, idle=750,
                          torque_peak=4200, power_peak=6000, redline=6800, inertia=.16,
                          mass=112, clutch_torque=175, combustion_efficiency=.91,
                          coupling_efficiency=.94),
        power_unit_preset("aircooled-flat-four-1584", "1584 cc air-cooled flat-four", "combustion",
                          displacement=1.584, bmep=720_000, braking_bmep=125_000, idle=850,
                          torque_peak=2800, power_peak=4100, redline=4800, inertia=.31,
                          mass=102, clutch_torque=155, combustion_efficiency=.82,
                          coupling_efficiency=.93),
        power_unit_preset("springtail-i4-1600", "Springtail 1.6 L trail I4", "combustion",
                          displacement=1.6, bmep=1_600_000, braking_bmep=165_000, idle=850,
                          torque_peak=3600, power_peak=5200, redline=6400, inertia=.22,
                          mass=142, clutch_torque=260, combustion_efficiency=.88,
                          coupling_efficiency=.94),
        power_unit_preset("superbike-i4-1340", "1340 cc superbike I4", "combustion",
                          displacement=1.34, bmep=1_720_000, braking_bmep=190_000, idle=1250,
                          torque_peak=7000, power_peak=9700, redline=11000, inertia=.085,
                          mass=82, clutch_torque=205, combustion_efficiency=.91,
                          coupling_efficiency=.95),
        power_unit_preset("gt-flat-six-4000", "4.0 L GT flat-six", "combustion",
                          displacement=4.0, bmep=1_650_000, braking_bmep=210_000, idle=900,
                          torque_peak=6250, power_peak=8400, redline=9000, inertia=.19,
                          mass=190, clutch_torque=610, combustion_efficiency=.92,
                          coupling_efficiency=.96),
        power_unit_preset("supercharged-drag-v8-8200", "8.2 L supercharged drag V8", "combustion",
                          displacement=8.2, bmep=6_200_000, braking_bmep=420_000, idle=1150,
                          torque_peak=6500, power_peak=8700, redline=9600, inertia=.46,
                          mass=338, clutch_torque=4_800, combustion_efficiency=.94,
                          coupling_efficiency=.97),
        power_unit_preset("monster-540-blown-methanol", "540 ci blown-methanol monster V8", "combustion",
                          displacement=8.849, bmep=2_850_000, braking_bmep=360_000, idle=1100,
                          torque_peak=5200, power_peak=7100, redline=8000, inertia=.55,
                          mass=345, clutch_torque=2_900, combustion_efficiency=.93,
                          coupling_efficiency=.97),
        power_unit_preset("monster-632-twin-turbo", "632 ci twin-turbo monster big-block", "combustion",
                          displacement=10.357, bmep=3_650_000, braking_bmep=410_000, idle=1050,
                          torque_peak=4400, power_peak=6500, redline=7300, inertia=.70,
                          mass=425, clutch_torque=4_500, combustion_efficiency=.94,
                          coupling_efficiency=.97),
        power_unit_preset("packard-merlin-v1650", "Packard / Rolls-Royce Merlin V-1650 aircraft V12", "combustion",
                          displacement=27.04, bmep=1_420_000, braking_bmep=235_000, idle=600,
                          torque_peak=2200, power_peak=3000, redline=3200, inertia=2.85,
                          mass=744, clutch_torque=3_600, combustion_efficiency=.86,
                          coupling_efficiency=.92, preferred_fuel="aviation-gasoline-100-130",
                          compatible_fuels=("aviation-gasoline-100-130", "pump-gasoline-93", "nitromethane-race"),
                          fuel_compatibility={"aviation-gasoline-100-130": 1.0, "pump-gasoline-93": .48,
                                              "nitromethane-race": .70},
                          preferred_ignition="aircraft-dual-magneto",
                          ignition_compatibility={"aircraft-dual-magneto": 1.0, "gasoline-distributor": .62,
                                                  "nitromethane-magneto": .55, "nitromethane-ecu-race": .42}),
        power_unit_preset("cat-c18-industrial-diesel", "18.1 L heavy-machine turbo diesel I6", "combustion",
                          displacement=18.1, bmep=2_540_000, braking_bmep=390_000, idle=600,
                          torque_peak=1400, power_peak=1900, redline=2200, inertia=5.8,
                          mass=1_673, clutch_torque=4_400, combustion_efficiency=.92,
                          coupling_efficiency=.91, preferred_fuel="ultra-low-sulfur-diesel",
                          compatible_fuels=("ultra-low-sulfur-diesel",),
                          fuel_compatibility={"ultra-low-sulfur-diesel": 1.0, "pump-gasoline-93": .035,
                                              "aviation-gasoline-100-130": .025, "nitromethane-race": .06},
                          preferred_ignition="diesel-injection-governor",
                          ignition_compatibility={"diesel-injection-governor": 1.0, "gasoline-distributor": .03,
                                                  "aircraft-dual-magneto": .02, "nitromethane-magneto": .02,
                                                  "nitromethane-ecu-race": .025}),
        power_unit_preset("dual-motor-ev-reference", "dual-motor EV drive unit", "electric",
                          displacement=4.0, bmep=1_420_000, braking_bmep=760_000, idle=120,
                          torque_peak=900, power_peak=9000, redline=18000, inertia=.12,
                          mass=205, clutch_torque=720, combustion_efficiency=.99,
                          coupling_efficiency=.98),
        power_unit_preset("servo-direct-drive-400", "400 Nm direct-drive servo", "servo-electric",
                          displacement=.80, bmep=6_300_000, braking_bmep=1_900_000, idle=60,
                          torque_peak=120, power_peak=4200, redline=8000, inertia=.075,
                          mass=52, clutch_torque=400, combustion_efficiency=.995,
                          coupling_efficiency=.985, mechanical_roles=("propulsion", "steering-assist",
                              "chassis-articulation", "auxiliary-actuation")),
    ]
    architecture_by_preset = {
        "amc-258-jeep-i6": ("inline-six", 6, 1, 0.0, [1, 5, 3, 6, 2, 4]),
        "honda-style-commuter-i4-1500": ("inline-four", 4, 1, 0.0, [1, 3, 4, 2]),
        "aircooled-flat-four-1584": ("flat-four", 4, 2, 180.0, [1, 4, 3, 2]),
        "springtail-i4-1600": ("inline-four", 4, 1, 0.0, [1, 3, 4, 2]),
        "superbike-i4-1340": ("inline-four", 4, 1, 0.0, [1, 2, 4, 3]),
        "gt-flat-six-4000": ("flat-six", 6, 2, 180.0, [1, 6, 2, 4, 3, 5]),
        "supercharged-drag-v8-8200": ("crossplane-v8", 8, 2, 90.0, [1, 8, 4, 3, 6, 5, 7, 2]),
        "monster-540-blown-methanol": ("crossplane-v8", 8, 2, 90.0, [1, 8, 4, 3, 6, 5, 7, 2]),
        "monster-632-twin-turbo": ("crossplane-v8", 8, 2, 90.0, [1, 8, 4, 3, 6, 5, 7, 2]),
        "packard-merlin-v1650": ("sixty-degree-v12", 12, 2, 60.0, [1, 6, 3, 5, 2, 4, 7, 12, 9, 11, 8, 10]),
        "cat-c18-industrial-diesel": ("inline-six", 6, 1, 0.0, [1, 5, 3, 6, 2, 4]),
        "dual-motor-ev-reference": ("dual-electric", 0, 2, 0.0, []),
        "servo-direct-drive-400": ("servo-electric", 0, 1, 0.0, []),
    }
    package_by_preset = {
        "amc-258-jeep-i6": ((1.05, .76, .62), (.72, .18, .48)),
        "honda-style-commuter-i4-1500": ((.69, .66, .58), (.46, .15, .39)),
        "packard-merlin-v1650": ((2.25, 1.02, .78), (1.34, .27, .58)),
        "cat-c18-industrial-diesel": ((1.43, 1.32, 1.01), (1.02, .31, .72)),
    }
    for preset in power_unit_presets:
        preset["kernel_selector"] = power_unit_presets.index(preset)
        layout, cylinders, banks, bank_angle, firing_order = architecture_by_preset[preset["identity"]]
        four_stroke_events = [index * 720.0 / max(1, cylinders) for index in range(cylinders)]
        electric = preset["kind"] != "combustion"
        methanol = "methanol" in preset["identity"]
        storage_mass = 340.0 if preset["kind"] == "electric" else 82.0 if electric else 64.0 if methanol else 48.0
        compression_ignition = preset["preferred_ignition_profile"] == "diesel-injection-governor"
        aircraft_magneto = preset["preferred_ignition_profile"] == "aircraft-dual-magneto"
        preset["architecture"] = {
            "schema": "springtail-engine-architecture-v0", "layout": layout,
            "cylinders": cylinders, "banks": banks, "bank_angle_degrees": bank_angle,
            "firing_order": firing_order, "cycle_degrees": 720.0 if cylinders else 360.0,
            "event_phases_degrees": four_stroke_events,
            "mount_vibration_reference": f"springtail-mount-vibration-v0:{preset['identity']}",
            "mount_harmonics": ([1.0, 2.0, cylinders / 2] if cylinders else [6.0, 12.0, 24.0]),
            "workshop_bake": "cylinder-pressure-impulses-crank-balance-and-mount-transfer-function",
        }
        default_scale = max(.55, min(1.35, float(preset["configuration"]["displacement_liters"]) ** (1 / 3) * .42))
        engine_envelope, pan_envelope = package_by_preset.get(
            preset["identity"],
            ((.78 * default_scale, .68 * default_scale, .62 * default_scale),
             (.52 * default_scale, .16 * default_scale, .42 * default_scale)),
        )
        _, chassis_fit = fit_vehicle_chassis_to_power_unit(
            config,
            engine_envelope_m=engine_envelope,
            oil_pan_envelope_m=pan_envelope,
            engine_mass_kg=float(preset["configuration"]["engine_mass_kg"]),
        )
        preset["package"] = {
            "engine_envelope_m": list(engine_envelope),
            "oil_pan_envelope_m": list(pan_envelope),
            "chassis_fit": chassis_fit,
        }
        preset["energy_system"] = {
            "schema": "springtail-power-energy-system-v0",
            "carrier": "electricity" if electric else preset["preferred_fuel_profile"],
            "storage_kind": "battery-pack" if electric else "baffled-fuel-tank",
            "installed_storage_mass_kg": storage_mass,
            "delivery_kind": "high-voltage-cables-and-inverter" if electric else "tank-pump-line-and-fuel-rail",
            "conversion": "inverter-phase-current-to-electromagnetic-torque" if electric else
                "intake-air-plus-metered-fuel-to-cylinder-charge-and-bmep",
            "air_path": None if electric else "filter-throttle-body-plenum-runners-intake-valves",
            "exhaust_path": None if electric else "exhaust-valves-headers-collectors",
            "mass_authority": "storage-shell-plus-live-carrier-mass-belongs-in-rigid-body-mass-and-inertia",
            "current_torque_authority": "compiled-bmep-curve-until-consumable-flow-state-is-linked",
            "reservoir_options": ([] if electric else [
                {"identity": "primary", "carrier": preset["preferred_fuel_profile"],
                 "selection": "default", "mass_and_inertia": "live-fill-state-required"},
                {"identity": "reserve", "carrier": "same-as-primary",
                 "selection": "sequenced-changeover", "mass_and_inertia": "live-fill-state-required"},
                {"identity": "auxiliary-oxidizer-or-fuel", "carrier": "user-selected",
                 "selection": "metered-blend", "mass_and_inertia": "live-fill-state-required"},
            ]),
            "mixture_control": (None if electric else {
                "authority": "medium-rate-manifold-flow-state",
                "required_inputs": ["carrier-properties", "mass-flow", "air-flow", "spark-timing",
                                    "compression-ratio", "charge-temperature"],
                "failure_modes": ["lean-misfire", "knock", "pre-ignition", "over-fueling", "stall"],
                "torque_status": "declared-hook-not-yet-authoritative",
            }),
            "ignition_system": (None if electric else {
                "parts": (["crank-cam-trigger", "high-pressure-injection-pump", "common-rail", "injectors"]
                          if compression_ignition else
                          ["dual-magnetos", "shielded-plug-leads", "two-spark-plugs-per-cylinder"]
                          if aircraft_magneto else
                          ["trigger-distributor", "coil-bank", "plug-leads", "spark-plugs"]),
                "event_source": ("governed-injection-start-by-cylinder-cycle-phase" if compression_ignition else
                                 "dual-magneto-spark-by-cylinder-cycle-phase" if aircraft_magneto else
                                 "architecture-firing-order-and-cycle-phase"),
                "timing_status": "authoritative-torque-derate-damage-and-audio-phase-input",
            }),
        }
    base_transmission = dict(config.source["transmission"])
    def transmission_preset(identity: str, label: str, ratios: list[float], reverse: float,
                            low_range: float, upshift: list[float], downshift: list[float]) -> dict[str, Any]:
        configuration = dict(base_transmission)
        configuration.update({"forward_ratios": ratios, "reverse_ratio": reverse,
                              "ultra_low_range_ratio": low_range,
                              "upshift_wheel_speed_rad_s": upshift,
                              "downshift_wheel_speed_rad_s": downshift,
                              "starting_gear": min(2, len(ratios))})
        return {"identity": identity, "label": label, "configuration": configuration,
                "runtime_policy": "live-ratio-control-no-symbolic-recompile"}
    transmission_presets = [
        transmission_preset("cj-wide-four-speed", "CJ wide-ratio 4-speed · 3.52 first",
                            list(base_transmission["forward_ratios"]), float(base_transmission["reverse_ratio"]),
                            float(base_transmission["ultra_low_range_ratio"]),
                            list(base_transmission["upshift_wheel_speed_rad_s"]),
                            list(base_transmission["downshift_wheel_speed_rad_s"])),
        transmission_preset("expedition-super-crawl", "expedition 6-speed · 12.80 crawler",
                            [12.8, 6.8, 3.8, 2.3, 1.4, 1.0], 10.4, 3.8,
                            [4.0, 7.5, 13.0, 21.0, 34.0, 55.0], [0.0, 2.5, 5.0, 9.0, 16.0, 27.0]),
        transmission_preset("close-ratio-six-speed", "close-ratio 6-speed",
                            [3.2, 2.2, 1.62, 1.26, 1.0, .82], 3.0, 2.1,
                            [12.0, 21.0, 31.0, 42.0, 55.0, 72.0], [0.0, 8.0, 15.0, 23.0, 33.0, 45.0]),
        transmission_preset("drag-three-speed", "drag 3-speed · reinforced",
                            [2.48, 1.56, 1.0], 2.2, 1.0,
                            [27.0, 52.0, 88.0], [0.0, 18.0, 38.0]),
    ]
    clutch_presets = [
        {"identity": "old-soft-organic", "label": "old soft organic clutch", "default": True,
         "mass_kg": 14.0, "driven_inertia_kg_m2": .060,
         "stiffness_nm_per_rad_s": 4.2, "maximum_torque_nm": 235.0, "efficiency": .90,
         "engagement": "long-progressive-cushion-spring-and-aged-organic-facing"},
        {"identity": "modern-organic-hd", "label": "modern heavy-duty organic clutch", "default": False,
         "mass_kg": 10.0, "driven_inertia_kg_m2": .025,
         "stiffness_nm_per_rad_s": 9.5, "maximum_torque_nm": 420.0, "efficiency": .95,
         "engagement": "medium-progressive-diaphragm-spring"},
        {"identity": "sintered-six-puck", "label": "sintered six-puck clutch", "default": False,
         "mass_kg": 8.0, "driven_inertia_kg_m2": .018,
         "stiffness_nm_per_rad_s": 16.0, "maximum_torque_nm": 700.0, "efficiency": .97,
         "engagement": "short-abrupt-unsprung-puck"},
        {"identity": "aircraft-heavy-multiplate", "label": "aircraft heavy multi-plate clutch", "default": False,
         "mass_kg": 31.0, "driven_inertia_kg_m2": .145,
         "stiffness_nm_per_rad_s": 31.0, "maximum_torque_nm": 3_800.0, "efficiency": .94,
         "engagement": "short-heavy-multiplate-with-high-pedal-force"},
        {"identity": "industrial-twin-disc", "label": "industrial twin-disc clutch", "default": False,
         "mass_kg": 58.0, "driven_inertia_kg_m2": .310,
         "stiffness_nm_per_rad_s": 38.0, "maximum_torque_nm": 4_800.0, "efficiency": .91,
         "engagement": "slow-heavy-sprung-hub-twin-disc"},
    ]
    wheel_parts = [
        {"identity": "balloon-black-current", "label": "big black balloon tires",
         "realization": "parametric-pneumatic-wheel-renderer", "default": False,
         "radius_scale": .58139535, "width_scale": 1.61290323, "rim_scale": .37735849,
         "carcass_profile": "high-sidewall-balloon", "cold_pressure_kpa": 82.0,
         "wheel_mass_kg": 17.0, "tire_mass_kg": 18.0, "rotational_inertia_scale": 1.6,
         "toroid_section_radius_m": .115, "effective_tread_width_fraction": .78,
         "radial_carcass_loss_n_s_per_m": 1320.0,
         "sidewall_shear_stiffness_longitudinal_n_per_m": 420000.0,
         "sidewall_shear_stiffness_lateral_n_per_m": 330000.0,
         "sidewall_shear_damping_n_s_per_m": 420.0,
         "longitudinal_deformation_mode_frequency_hz": 8.0,
         "lateral_deformation_mode_frequency_hz": 5.0,
         "sidewall_deformation_damping_ratio": 1.05, "maximum_sidewall_deformation_m": .055,
         "compound": "off-road-compliant", "dry_grip_scale": 1.0,
         "tire_color": "#202624", "tread_color": "#687672"},
        {"identity": "legacy-small-brown", "label": "tacky brown racing tires",
         "realization": "parametric-pneumatic-wheel-renderer", "default": False,
         "radius_scale": .41860465, "width_scale": 1.25806452, "rim_scale": .27169811,
         "carcass_profile": "low-sidewall-racing", "cold_pressure_kpa": 175.0,
         "wheel_mass_kg": 13.0, "tire_mass_kg": 12.0, "rotational_inertia_scale": 1.35,
         "toroid_section_radius_m": .055, "effective_tread_width_fraction": .82,
         "radial_carcass_loss_n_s_per_m": 1550.0,
         "sidewall_shear_stiffness_longitudinal_n_per_m": 690000.0,
         "sidewall_shear_stiffness_lateral_n_per_m": 610000.0,
         "sidewall_shear_damping_n_s_per_m": 610.0,
         "longitudinal_deformation_mode_frequency_hz": 11.0,
         "lateral_deformation_mode_frequency_hz": 9.0,
         "sidewall_deformation_damping_ratio": 1.08, "maximum_sidewall_deformation_m": .030,
         "compound": "tacky-race-rubber", "dry_grip_scale": 1.22,
         "tire_color": "#6b4b32", "tread_color": "#a77850"},
        {"identity": "tall-thin-tractor-steel-disc", "label": "towering tractor steel-disc wheels",
         "realization": "parametric-pneumatic-wheel-renderer", "default": True,
         "radius_scale": 1.0, "width_scale": 1.0, "rim_scale": 1.0,
         "carcass_profile": "thin-agricultural-steel-disc", "rim_profile": "solid-steel-plate",
         "tread_pattern": "agricultural-chevron", "cold_pressure_kpa": 135.0,
         "compound": "agricultural-cut-resistant", "dry_grip_scale": .92,
         "wheel_mass_kg": 68.0, "tire_mass_kg": 14.0, "rotational_inertia_scale": 1.35,
         "toroid_section_radius_m": .085, "effective_tread_width_fraction": .86,
         "gas_polytropic_exponent": 1.38, "radial_carcass_loss_n_s_per_m": 1900.0,
         "sidewall_shear_stiffness_longitudinal_n_per_m": 640000.0,
         "sidewall_shear_stiffness_lateral_n_per_m": 520000.0,
         "sidewall_shear_damping_n_s_per_m": 760.0,
         "longitudinal_deformation_mode_frequency_hz": 10.0,
         "lateral_deformation_mode_frequency_hz": 7.0,
         "sidewall_deformation_damping_ratio": 1.12,
         "maximum_sidewall_deformation_m": .035,
         "tire_color": "#171b18", "tread_color": "#343c35", "rim_color": "#8d9189"},
    ]
    body_shells = [
        {"identity": "clear-polycarbonate-rc", "label": "clear tinted polycarbonate RC shell",
         "realization": "chassis-relative-breakable-contact-shell", "physics": True, "default": True,
         "material_identity": "acrylic_pmma", "material_profile": "transmissive-phong-plastic",
         "palette_role": "body-shell-glass", "ior": 1.586, "transmission": .72, "opacity": .30},
        {"identity": "fiberglass-monster-pickup", "label": "fiberglass monster pickup",
         "realization": "chassis-relative-breakable-contact-shell", "physics": True, "default": False},
        {"identity": "bare-frame", "label": "bare frame",
         "realization": "no-cosmetic-shell", "physics": False, "default": False},
        {"identity": "six-body-pin-carrier", "label": "six body-pin armored carrier",
         "realization": "six-structural-body-pins-with-optional-clutch-gimbals", "physics": True, "default": False,
         "palette_role": "drivetrain-black", "assembly_mass_kg": 358.0,
         "center_of_mass_local": [-.03, .52, 0.0],
         "principal_inertia_kg_m2": [82.0, 119.0, 104.0],
         "armor": {"material": "quenched-steel", "thickness_m": .018,
                    "mass_kg": 126.0, "mount_count": 8,
                    "collision": "segmented-breakable-plastic-contact-shell"},
         "fire_control": {"computer_mass_kg": 9.0, "power_w": 185.0,
                          "target_authority": "active-focus-ray-surface-intersection",
                          "friendly_fire_interlock": "nearest-friendly-ray-entry-before-target",
                          "primary_fire_takeover_default": True},
         "ammunition": {"capacity_count": 60, "capacity_volume_m3": .050,
                        "capacity_mass_kg": 51.0, "round_mass_kg": .85,
                        "round_volume_m3": .00078, "initial_count": 60,
                        "muzzle_speed_m_s": 72.0, "recoil_impulse_n_s": 49.0},
         "body_pins": {"count": 6, "retains_body_without_weapon_payload": True,
                       "spring_stiffness_n_per_m": 18_000.0,
                       "compression_damping_n_s_per_m": 780.0,
                       "rebound_damping_n_s_per_m": 1_150.0,
                       "maximum_compression_m": .018},
         "turrets": [{
             "identity": name,
             "local_position": list(next(
                 node["reference_position"] for node in mechanical_graph["nodes"]
                 if node["identity"] == f"body_pin.{name}.lock")),
             "post_height_m": float(next(
                 node["reference_position"][1] for node in mechanical_graph["nodes"]
                 if node["identity"] == f"body_pin.{name}.lock"))
                 - float(chassis["height"]) * .72,
         } for name in ("hood_left", "hood_right", "cab_left", "cab_right",
                         "bed_left", "bed_right")],
         "weapon": {"count": 6, "mass_each_kg": 20.0, "gimbal_mass_each_kg": 4.0,
                    "yaw_limit_degrees": 175.0, "pitch_min_degrees": -28.0,
                    "pitch_max_degrees": 72.0, "slew_rate_degrees_s": 150.0},
         "outriggers": {"count": 4, "mass_kg": 48.0, "hydraulic_power_w": 820.0,
                        "extension_rate_m_s": .34, "maximum_extension_m": 1.72,
                        "inboard_reserve_m": .72, "minimum_structural_overlap_m": .24,
                        "axial_stiffness_n_per_m": 140_000.0,
                        "axial_damping_n_s_per_m": 12_000.0,
                        "maximum_axial_force_n": 85_000.0,
                        "hydraulic_accumulator_capacity_j": 12_000.0,
                        "hand_pump_displacement_m3_per_click": 4.5e-6,
                        "hand_pump_pressure_pa": 10_000_000.0,
                        "hand_pump_efficiency": .72,
                        "diagonal_direction_local": [.18, -.78, .60],
                        "contact": "one-sided-swept-foot-crossing",
                        "anchor": "persistent-six-axis-terrain-weld-until-retraction",
                        "feet": ["front_left", "front_right", "rear_left", "rear_right"]}},
    ]
    body_packaging_presets = [
        {"identity": "compact-race-pickup", "label": "compact improvised race pickup",
         "requested_cab_length_m": .72, "bed_length_m": .42,
         "front_clip_length_m": .30, "seat_rows": 1},
        {"identity": "long-flatbed", "label": "long configurable flatbed",
         "requested_cab_length_m": .88, "bed_length_m": 1.80,
         "front_clip_length_m": .38, "seat_rows": 1},
        {"identity": "bus-chassis", "label": "bus-length passenger chassis",
         "requested_cab_length_m": 4.80, "bed_length_m": 0.0,
         "front_clip_length_m": .55, "seat_rows": 6},
    ]
    accessory_presets = [{
        "identity": "generic-loadout-winch",
        "label": "generic drivetrain/electric winch",
        "kind": "wrench-producing-cable-accessory",
        "default_mounted": False,
        "mount_interface": "generic-six-axis-wrench-attachment-v1",
        "drive_interfaces": {
            "drivetrain": "rotating-six-axis-drivetrain-wrench-port-v1",
            "electrical": "fused-dc-motor-power-and-command-port",
        },
        "mass_properties_required": ["mass_kg", "center_of_mass_local", "inertia_kg_m2"],
        "cable_stepper": {
            "state": ["node_position_xyz", "node_velocity_xyz", "segment_strain",
                      "spool_length_m", "spool_angular_velocity"],
            "law": "tension-only-extensible-rope-with-bending-free-contact-segments",
            "integration": "shared-graph-substeps-selected-by-declared-cable-resolution",
            "compute_burden": "explicit-segment-count-times-substeps-never-hidden-in-accessory-metadata",
            "status": "contract-ready-integrator-deferred-until-loadout-device-work",
        },
        "winch_region": {
            "frame": "chassis-local",
            "shape": "loadout-declared-bounded-volume",
            "hook_wrench_position": "arbitrary-point-within-region",
        },
        "hook_wrench": {
            "kernel": "winch-cable-terminal-six-axis-wrench",
            "publishes": ["force_xyz", "moment_xyz", "application_point_xyz"],
            "reaction": "equal-and-opposite-through-spool-drive-and-accessory-mount",
        },
    }, {
        "identity": "differential-port-crawl-flywheel",
        "label": "massive differential-port crawl flywheel",
        "kind": "rotating-inertia-drivetrain-accessory",
        "default_mounted": False,
        "mount_interface": "rotating-six-axis-drivetrain-wrench-port-v1",
        "eligible_ports": [f"powertrain.{axle}_differential_brake_wrench"
                           for axle in ("front", "rear")],
        "parameters": dict(mechanical_graph["rotating_accessory_presets"][
            "differential-port-crawl-flywheel"]),
        "state_coupling": (
            "adds-polar-inertia-to-live-differential-wrench-shaft-state-and-"
            "returns-bearing-and-gyroscopic-reaction-through-port-wrench"),
        "guard": "overspeed-or-mount-fracture-opens-accessory-edge-with-angular-momentum-preserved",
    }, {
        "identity": "pre-clutch-crank-flywheel",
        "label": "massive pre-clutch crank flywheel",
        "kind": "rotating-inertia-drivetrain-accessory",
        "default_mounted": False,
        "mount_interface": "rotating-six-axis-drivetrain-wrench-port-v1",
        "eligible_ports": ["powertrain.pre_clutch_flywheel_wrench"],
        "parameters": dict(mechanical_graph["rotating_accessory_presets"][
            "pre-clutch-crank-flywheel"]),
        "state_coupling": (
            "adds-polar-inertia-to-external-engine-flywheel-inertia-before-main-clutch-"
            "and-direct-drive-bypass"),
        "guard": "overspeed-or-mount-fracture-opens-accessory-edge-with-angular-momentum-preserved",
    }, {
        "identity": "configurable-barrel-tank",
        "label": "configurable barrel tank",
        "kind": "generic-contained-material-accessory",
        "default_mounted": False,
        "mount_interface": "generic-six-axis-wrench-attachment-v1",
        "placement": "any-validated-frame-or-body-braze-on-wrench-port",
        "geometry_parameters": {
            "radius_m": .22, "cylindrical_length_m": .70,
            "wall_thickness_m": .003, "end_shape": "ellipsoidal",
        },
        "material_parameters": {
            "shell_density_kg_m3": 7850.0, "contents_density_kg_m3": 740.0,
            "fill_fraction": .80,
        },
        "state": ["contents_mass_kg", "contents_center_of_mass_local",
                  "contents_angular_momentum", "sloshing_mode_amplitude"],
        "mass_law": (
            "shell-and-contents-volume-integrals-publish-mass-center-of-mass-and-"
            "inertia-to-the-mount-wrench"),
        "liquid_model": "optional-baffled-low-order-free-surface-slosh-appendage",
        "guard": "mount-yield-or-shell-rupture-opens-the-corresponding-physical-graph-edge",
    }, {
        "identity": "industrial-high-pressure-gas-cylinder",
        "label": "industrial high-pressure gas torpedo cylinder",
        "kind": "certified-pressure-vessel-accessory",
        "default_mounted": False,
        "mount_interface": "generic-six-axis-wrench-attachment-v1",
        "placement": "validated-cylinder-cradle-and-two-independent-restraints-only",
        "geometry_parameters": {
            "outside_radius_m": .115, "straight_length_m": 1.25,
            "wall_thickness_m": .007, "end_shape": "hemispherical",
        },
        "material_parameters": {
            "shell_density_kg_m3": 7850.0, "shell_yield_stress_pa": 620_000_000.0,
            "shell_fracture_stress_pa": 760_000_000.0,
            "minimum_burst_safety_factor": 2.25,
        },
        "gas_parameters": {
            "species": "loadout-selected-real-gas-mixture",
            "reference_pressure_pa": 20_000_000.0,
            "reference_temperature_k": 293.15,
            "molar_mass_kg_per_mol": .0280134,
            "permeability_mol_m_per_m2_s_pa": 0.0,
        },
        "state": ["gas_mass_kg", "pressure_pa", "temperature_k", "internal_energy_j",
                  "valve_open_fraction", "leak_area_m2", "shell_plastic_strain",
                  "restraint_damage"],
        "pressure_vessel_law": (
            "real-gas-state-plus-thick-wall-hoop-and-longitudinal-stress-with-"
            "temperature-dependent-shell-yield"),
        "valve_and_relief": {
            "service_valve": True, "excess_flow_valve": True,
            "rupture_disc_set_pressure_pa": 34_000_000.0,
            "relief_discharge_direction": "declared-in-cylinder-local-frame",
        },
        "failure_wrenches": {
            "leak": "compressible-choked-or-unchoked-jet-reaction-at-orifice",
            "valve_shear": "full-bore-jet-wrench-and-free-cylinder-rocket-state",
            "rupture": "stored-gas-expansion-energy-plus-shell-fragment-impulses",
        },
        "mass_law": (
            "shell-integral-plus-live-gas-mass-publishes-total-mass-center-of-mass-"
            "and-inertia-through-both-restraint-wrenches"),
        "guard": (
            "reject-installation-unless-pressure-rating-temperature-rating-clearance-"
            "and-two-restraint-load-paths-pass-validator"),
    }, {
        "identity": "direct-drive-high-pressure-compressor",
        "label": "direct-drive high-pressure compressor",
        "kind": "shared-shaft-pressure-accessory",
        "default_mounted": True,
        "parameters": dict(mechanical_graph["static_accessory_presets"][
            "direct-drive-high-pressure-compressor"]),
    }, {
        "identity": "reversible-accessory-block-motor",
        "label": "reversible accessory-block motor",
        "kind": "bidirectional-electromechanical-accessory",
        "default_mounted": True,
        "parameters": dict(mechanical_graph["static_accessory_presets"][
            "reversible-accessory-block-motor"]),
    }, {
        "identity": "four-lead-acid-battery-cube",
        "label": "four lead-acid battery cube",
        "kind": "stateful-electrical-energy-storage-accessory",
        "default_mounted": True,
        "parameters": dict(mechanical_graph["static_accessory_presets"][
            "four-lead-acid-battery-cube"]),
    }]
    fuel_profiles = [
        {"identity": "pump-gasoline-93", "label": "93-octane gasoline",
         "carrier": "gasoline",
         "energy_density_j_per_kg": 44_000_000.0, "stoichiometric_air_fuel_ratio": 14.7,
         "mass_flow_scale": 1.0, "torque_scale": 1.0, "preferred_advance_degrees": 14.0,
         "preferred_rpm_advance_degrees": 12.0, "combustion_sharpness": 1.0, "default": True},
        {"identity": "nitromethane-race", "label": "nitromethane",
         "carrier": "nitromethane",
         "energy_density_j_per_kg": 11_300_000.0, "stoichiometric_air_fuel_ratio": 1.7,
         "mass_flow_scale": 5.8, "torque_scale": 1.82, "preferred_advance_degrees": 31.0,
         "preferred_rpm_advance_degrees": 4.0, "combustion_sharpness": .72, "default": False},
        {"identity": "aviation-gasoline-100-130", "label": "100/130 aviation gasoline",
         "carrier": "aviation-gasoline", "energy_density_j_per_kg": 43_500_000.0,
         "stoichiometric_air_fuel_ratio": 14.7, "mass_flow_scale": 1.02, "torque_scale": 1.0,
         "preferred_advance_degrees": 26.0, "preferred_rpm_advance_degrees": 8.0,
         "combustion_sharpness": .92, "default": False},
        {"identity": "ultra-low-sulfur-diesel", "label": "ultra-low-sulfur diesel",
         "carrier": "diesel", "energy_density_j_per_kg": 45_500_000.0,
         "stoichiometric_air_fuel_ratio": 14.5, "mass_flow_scale": .72, "torque_scale": 1.0,
         "preferred_advance_degrees": 12.0, "preferred_rpm_advance_degrees": 4.0,
         "combustion_sharpness": 1.28, "default": False},
    ]
    ignition_profiles = [
        {"identity": "gasoline-distributor", "label": "gasoline ECU timing",
         "advance_degrees": 14.0, "rpm_advance_degrees": 12.0, "load_retard_degrees": 2.0,
         "dispatch": "ecu-electronic", "default": True},
        {"identity": "nitromethane-ecu-race", "label": "nitromethane ECU timing",
         "advance_degrees": 31.0, "rpm_advance_degrees": 4.0, "load_retard_degrees": 5.0,
         "dispatch": "ecu-electronic", "default": False},
        {"identity": "nitromethane-magneto", "label": "nitromethane magneto timing",
         "advance_degrees": 31.0, "rpm_advance_degrees": 2.0, "load_retard_degrees": 0.0,
         "dispatch": "mechanical-magneto", "default": False},
        {"identity": "aircraft-dual-magneto", "label": "aircraft dual-magneto timing",
         "advance_degrees": 26.0, "rpm_advance_degrees": 8.0, "load_retard_degrees": 3.0,
         "dispatch": "mechanical-magneto", "default": False},
        {"identity": "diesel-injection-governor", "label": "diesel injection timing",
         "advance_degrees": 12.0, "rpm_advance_degrees": 4.0, "load_retard_degrees": 1.0,
         "dispatch": "compression-injection", "default": False},
    ]
    driving_modes = [
        {"identity": "trail", "label": "trail / crawl", "throttle_exponent": 2.15,
         "throttle_rate_scale": .72, "default": False},
        {"identity": "road", "label": "road", "throttle_exponent": 1.35,
         "throttle_rate_scale": 1.0, "default": True},
        {"identity": "sport", "label": "sport", "throttle_exponent": .72,
         "throttle_rate_scale": 1.45, "default": False},
    ]
    torque_graph = {
        "schema": "abstract-ui-vehicle-torque-graph-v0",
        "authority": "compiled-sympy-abstract-tensor-ssa-wgsl",
        "nodes": [
            {"identity": "engine", "kind": "engine",
             "mass_kg": config.source["powertrain"]["engine_mass_kg"],
             "local_position": list(config.source["powertrain"]["engine_position"]),
             "orientation_degrees": list(config.source["powertrain"]["engine_orientation_degrees"])},
            {"identity": "clutch", "kind": "clutch"},
            {"identity": "transmission", "kind": "selectable-ratio-transmission",
             "mass_kg": config.source["powertrain"]["transmission_mass_kg"],
             "default_mode": config.source["transmission"]["mode_default"],
             "starting_gear": config.source["transmission"]["starting_gear"],
             "crawler_gear": config.source["transmission"]["crawler_gear"],
             "forward_ratios": list(config.source["transmission"]["forward_ratios"]),
             "reverse_ratio": config.source["transmission"]["reverse_ratio"]},
            {"identity": "transfer_case", "kind": "three-range-transfer-case",
             "mass_kg": config.source["powertrain"]["transfer_case_mass_kg"],
             "high_range_ratio": 1.0,
             "low_range_ratio": config.source["transmission"]["low_range_ratio"],
             "ultra_low_range_ratio": config.source["transmission"]["ultra_low_range_ratio"],
             "efficiency": config.source["drivetrain"]["transfer_case_efficiency"],
             "drag_torque_nm": config.source["drivetrain"]["transfer_case_drag_torque_nm"]},
            {"identity": "final_drive", "kind": "final-drive"},
            {"identity": "front_differential", "kind": "driver-lockable-differential",
             "mass_kg": config.source["powertrain"]["front_differential_mass_kg"]},
            {"identity": "rear_differential", "kind": "driver-lockable-differential",
             "mass_kg": config.source["powertrain"]["rear_differential_mass_kg"]},
            *({"identity": wheel, "kind": "wheel-half-shaft"} for wheel in WHEEL_NAMES),
            *({"identity": f"brake_{wheel}", "kind": "rotor-caliper-torque-couple"}
              for wheel in WHEEL_NAMES),
            {"identity": "engine_mount", "kind": "six-axis-chassis-reaction"},
            {"identity": "chassis", "kind": "rigid-body-torque-sink"},
        ],
        "edges": [
            {"from": "engine", "to": "clutch", "channel": "engine_torque"},
            {"from": "clutch", "to": "transmission", "channel": "clutch_torque"},
            {"from": "transmission", "to": "transfer_case", "channel": "transmission_output_torque"},
            {"from": "transfer_case", "to": "final_drive", "channel": "driveline_torque",
             "loss_model": "smooth-drag-plus-parametric-efficiency"},
            {"from": "final_drive", "to": "front_differential", "channel": "front_differential_torque"},
            {"from": "final_drive", "to": "rear_differential", "channel": "rear_differential_torque"},
            *({"from": "front_differential" if wheel.startswith("front") else "rear_differential",
               "to": wheel, "channel": f"drive_fraction_{wheel}"} for wheel in WHEEL_NAMES),
            *({"from": wheel, "to": f"brake_{wheel}",
               "channel": f"brake_torque * brake * brake_scale_{wheel}"}
              for wheel in WHEEL_NAMES),
            *({"from": f"brake_{wheel}", "to": "chassis",
               "channel": "equal-and-opposite-caliper-reaction-through-knuckle"}
              for wheel in WHEEL_NAMES),
            {"from": "engine", "to": "engine_mount", "channel": "engine_acceleration_torque"},
            {"from": "engine_mount", "to": "chassis", "channel": "powertrain_reaction_torque_xyz"},
        ],
        "compiled_outputs": list(VEHICLE_STATE_OUTPUTS[-15:]),
        "conservation_rule": "chassis-local crank reaction plus engine-and-transmission r-cross-inertial-force is applied to chassis-frame torque",
    }
    # Engine definitions are durable selectable records; the vehicle physics is
    # lowered once with a live parameter ABI instead of once per profile/mode.
    webgpu = vehicle_webgpu_program_model(config, tuple(power_unit_presets))
    return {
        "schema": ABSTRACT_UI_VEHICLE_VERSION,
        "identity": f"{actor}/vehicle-slot",
        "owner": actor,
        "active": None,
        "initial_state": {
            "mounted_vehicle": f"{root}/vehicles/springtail",
            "placement": "at-player-spawn",
            "presentation": "full-viewport-driving",
            "browser_fullscreen": "user-invoked-only",
            "dismount_enabled": True,
        },
        "allowed_kinds": ["car"],
        "selection_operation": "mount-vehicle-slot",
        "release_operation": "dismount-vehicle-slot",
        "vehicles": [{
            "identity": f"{root}/vehicles/springtail", "archetype": config.identity,
            "name": config.name, "kind": "car", "configuration": config.to_data(),
            "configuration_defaults": config.parameter_defaults(),
            "power_unit_presets": power_unit_presets,
            "engine_kernel_switch": {
                "policy": "durable-engine-profile-records-over-one-live-parametric-vehicle-kernel",
                "selector_field": "engine_profile_selector_for_profile-data-only",
                "runtime_compilation": False,
                "equation_modes": ["linear-playable", "symbolic-fidelity"],
                "default_equation_mode": "linear-playable",
                "compiled_selector_rule": None,
                "cases": [{"selector": preset["kernel_selector"], "identity": preset["identity"],
                           "compiled_selectors": {"linear-playable": preset["kernel_selector"] * 2,
                                                  "symbolic-fidelity": preset["kernel_selector"] * 2 + 1},
                           "parameters": dict(preset["parameters"])}
                          for preset in power_unit_presets],
            },
            "feature_triage": {
                "stability_gate": "engine-off-20-second-headless-quiescence-at-120hz-three-substeps",
                "core_validation": ["spawn-solid-top-clearance", "swept-radial-tire-crossing",
                                    "gross-mass-refresh", "fixed-dt-substeps", "brake-lock-interlocks"],
                "live_parametric_selectors": ["power-unit", "wheel-part", "clutch", "chassis-profile",
                                               "chassis-geometry", "body-shell", "fuel", "ignition"],
                "default_off": ["chassis-leveling", "six-body-pin-carrier", "hydraulic-outriggers"],
                "quarantined": ["opaque-floating-skill-slabs"],
                "deferred_until_stability_gate": ["nonadhesive-loop-validation", "full-time-wheel-alignment"],
            },
            "transmission_presets": transmission_presets,
            "clutch_presets": clutch_presets,
            "clutch_preset": "old-soft-organic",
            "wheel_parts": wheel_parts,
            "wheel_part": "tall-thin-tractor-steel-disc",
            "body_shells": body_shells,
            "body_shell": "clear-polycarbonate-rc",
            "body_packaging_presets": body_packaging_presets,
            "body_packaging": solve_vehicle_body_packaging(config),
            "accessory_presets": accessory_presets,
            "body_assembly_interface": {
                "schema": "abstract-ui-chassis-mounted-body-wrench-v0",
                "mass_properties": ["mass_kg", "center_of_mass_local", "principal_inertia_kg_m2"],
                "mount_contract": "each-load-enters-through-an-identified-breakable-six-axis-mount",
                "wrench_contract": {"frame": "chassis-local", "force": "newtons",
                                    "moment": "newton-metres", "impulse": "newton-seconds",
                                    "application_point": "chassis-local-metres"},
                "dynamic_payloads": ["fuel", "ammunition", "detachable-panels"],
                "aggregation": "parallel-axis-mass-properties-plus-per-event-r-cross-force",
                "events": ["body-assembly-selected", "payload-mass-changed", "mount-failed",
                           "point-impulse-applied"],
            },
            "fuel_profiles": fuel_profiles,
            "fuel_profile": "pump-gasoline-93",
            "ignition_profiles": ignition_profiles,
            "ignition_profile": "gasoline-distributor",
            "rolling_start_system": {
                "combustion": {
                    "switch": "engine_enabled-is-live-ignition-or-injection-enable",
                    "sequence": ["starter-and-battery-unavailable", "ignition-switched",
                                 "hub-wrench-backdrives-locked-hub", "differential-shaft",
                                 "transfer-case", "selected-gear", "clutch", "crank-catches",
                                 "select-neutral-and-remove-rig-torque"],
                    "self_start_guard": "combustion-and-idle-torque-are-zero-at-zero-crank-speed",
                },
                "electric": {
                    "sequence": ["select-electric-power-unit", "set-target-charge",
                                 "backdrive-same-mechanical-path", "negative-clutch-power-regenerates",
                                 "stop-regeneration-at-target-charge"],
                    "state": "traction_battery_charge_fraction_next",
                    "power": "regenerative_charge_power_w",
                },
            },
            "driving_modes": driving_modes,
            "driving_mode": "road",
            "vehicle_computer": {
                "identity": "electrical.ecu", "kind": "electronic-control-unit",
                "dispatches": ["ignition-profile", "rpm-load-timing-map", "lighting-relays",
                               "brake-lights", "tilt-wheelie-suppression",
                               "velocity-sensitive-steering-rate"],
                "sensor_inputs": ["crank-speed", "engine-load", "wheel-speeds", "pitch-angle",
                                  "pitch-rate", "longitudinal-acceleration", "brake-switch",
                                  "light-switch", "battery-voltage", "vehicle-road-speed"],
                "ignition_rule": "electronic profiles require powered ECU; magneto profiles remain mechanical",
            },
            "transmission_control_unit": {
                "identity": "electrical.tcu", "kind": "dedicated-transmission-control-unit",
                "dispatches": ["gear-selection", "transfer-range", "clutch-open-request",
                               "engine-torque-cut-request", "speed-match-request",
                               "direct-drive-dog-command"],
                "shift_sequence": ["request-ecu-torque-cut", "open-main-friction-clutch",
                                   "isolate-live-drivetrain", "match-crank-and-input-speed",
                                   "engage-positive-dog-after-interlock", "restore-engine-torque"],
                "wiring": ["electrical.wire.tcu_feed", "electrical.wire.powertrain_can",
                           "electrical.wire.tcu_bypass_actuator"],
            },
            "wiring_harness": {
                "authority": "mechanical-graph insulated-copper-wire edges",
                "circuits": ["battery-main", "starter-solenoid", "alternator-charge",
                             "computer-and-ignition", "front-lighting-horn", "rear-lighting",
                             "tail-brake-light", "sensor-bus", "steering-assist-power",
                             "transmission-control", "engine-transmission-can",
                             "direct-drive-bypass-actuator"],
                "protection": "fusebox-and-relay-dispatch",
            },
            "alternator_bank": {
                "count": int(config.source["electrical"]["alternator_count"]),
                "drive": "direct-shaft-through-demand-smoothing-cvt-no-belt",
                "cvt_ratio_state": "alternator_cvt_ratio_state_next",
                "electrical_output": "alternator_generated_power_w",
                "mechanical_reaction": "alternator_reaction_torque_nm",
                "rotor_inertia": "count-times-each-rotor-inertia-reflected-through-live-cvt-ratio-squared",
                "installation_stage": "transfer-and-differentials",
            },
            "direct_drive_bypass": {
                "kind": "transfer-case-actuated-synchronized-positive-dog-clutch",
                "command": "direct_drive_bypass_command",
                "engagement_state": "direct_drive_bypass_engagement_next",
                "torque": "direct_drive_bypass_torque_nm",
                "interlock": "engage-only-at-low-relative-speed-and-unloaded-teeth",
                "disengagement": "immediate-commanded-open",
                "effect": "main-friction-clutch-torque-fades-out-as-positive-drive-path-engages",
                "damage": "direct_drive_bypass_tooth_health_next",
            },
            "transmission_internal_options": [{
                "identity": "dry-friction-clutch", "default": True,
                "torque": "clutch_torque", "fluidic": False,
            }, {
                "identity": "hydrodynamic-fluid-coupling", "default": False,
                "engagement": "optional_fluid_coupling_engagement",
                "torque": "optional_fluid_coupling_torque_nm",
                "installation": "optional-clutch-or-transmission-internal-part",
            }],
            "steering_control": {
                "front_axle_enabled": True, "rear_axle_enabled": True,
                "front_share": .5, "rear_phase": -1.0,
                "maximum_steering_wheel_torque_nm": 38.0,
                "column_torsional_stiffness_nm_per_rad": 92.0,
                "pinion_radius_m": .018, "rack_compliance_m_per_n": 1.8e-6,
                "knuckle_response_frequency_hz": 5.5, "knuckle_damping_ratio": .92,
                "free_knuckle_caster_frequency_hz": .75,
                "velocity_rate_control_enabled": True,
                "parking_steering_rate_per_s": 3.2,
                "highway_steering_rate_per_s": .85,
                "steering_rate_reference_speed_m_s": 22.0,
                "steering_rate_curve_exponent": 1.35,
                "mechanical_fallback_rate_per_s": 1.25,
                "maximum_human_steering_wheel_torque_nm": 38.0,
                "human_torque_per_normalized_error_nm": 52.0,
                "manual_steering_viscous_nm_s": 12.0,
                "manual_maximum_rate_per_s": 1.45,
                "assist_torque_multiplier": 3.4,
                "assist_without_ecu_maximum_rate_per_s": 2.25,
                "steering_ratio": 16.0,
                "tire_scrub_radius_m": .032,
                "manual_static_friction_estimate": .72,
                "caster_resistance_nm_per_m_s_squared": .012,
                "rate_authority": "powered-ecu-road-speed-map-with-column-mechanical-fallback",
                "input_authority": "steering-wheel-wrench-through-rotary-transmission-subgraph",
            },
            "chassis_profiles": chassis_profiles,
            "chassis_profile": default_chassis_profile["identity"],
            "chassis_geometry_parameters": {
                "authority": "mechanical-graph-reference-geometry-and-all-derived-presentation-collision-mass",
                "chassis_length_m": float(config.source["chassis"]["half_length"]) * 2,
                "wheelbase_m": float(config.source["wheels"]["wheelbase_half_length"]) * 2,
                "chassis_length_range_m": [1.10, 5.20],
                "wheelbase_range_m": [
                    config.source["wheel_placement_demands"]["longitudinal_wheel_distance_m"]["minimum"],
                    config.source["wheel_placement_demands"]["longitudinal_wheel_distance_m"]["maximum"],
                ],
                "wheel_placement_independent_of_chassis_and_body_length": True,
                "wheel_placement_demands": copy.deepcopy(
                    config.source["wheel_placement_demands"]),
                "mount_solution": copy.deepcopy(
                    mechanical_graph["wheel_placement_and_mounts"]),
                "rebuild_policy": (
                    "frame-and-post-nodes-follow-chassis-length;axle-nodes-follow-independent-"
                    "wheelbase-and-axle-group-offset;links-are-resolved-between-them"),
            },
            "chassis_profile_reference": {
                "identity": default_chassis_profile["identity"],
                "vehicle_mass_kg": float(config.source["mass"]),
                "member_mass_kg": default_chassis_profile["member_mass_kg"],
            },
            "chassis_leveling": {
                "authority": "compiler-owned-four-mode-hydraulic-coarse-plus-trim-controller",
                "enabled": False,
                "mode": "derived-pose",
                "target_ride_height_offset_m": 0.0,
                "target_roll_rad": 0.0,
                "target_pitch_rad": 0.0,
                "response_frequency_hz": 0.55,
                "damping_ratio": 1.05,
                "maximum_actuator_rate_m_s": 0.055,
                "pose_lerp_rate_m_s": 0.055,
                "hydraulic_authority": {
                    "piston_area_m2": float(suspension["leveling_actuator_piston_area_m2"]),
                    "manifold_pressure_pa": float(suspension["leveling_manifold_pressure_pa"]),
                    "maximum_flow_m3_s": float(suspension["leveling_maximum_flow_m3_s"]),
                    "efficiency": float(suspension["leveling_hydraulic_efficiency"]),
                },
                "trim_stage": {
                    "placement": "series-coilover-preload-collar",
                    "stroke_m": float(suspension["leveling_trim_stroke_m"]),
                    "maximum_rate_m_s": float(suspension["leveling_trim_rate_m_s"]),
                    "round_robin": True,
                    "alignment_actuators_are_trim_actuators": False,
                },
                "falling_policy_choices": {
                    "0": "hold-current-geometry",
                    "1": "symmetric-landing-ready-droop",
                    "2": "terrain-conformal-predicted-placement",
                    "force_and_crossweight_hunting_while_airborne": False,
                },
                "maximum_corner_offset_m": float(
                    mechanical_graph["leveling_controller"]["maximum_corner_offset_m"]),
                "standard_corner_offset_m": min(.12, float(config.source["suspension"]["travel"]) * .35),
                "high_clearance_corner_offset_m": .54,
                "suspension_link_actuators": {
                    "authority": "per-edge-rest-length-modifier-through-mechanical-graph",
                    "eligible_constraints": ["upper-a-arm", "lower-a-arm", "steering-tie-rod"],
                    "maximum_length_extension_m": .46,
                    "coordination": "corner-height-to-upper-lower-arm-and-tie-rod-geometry-solve",
                    "fail_safe": "freeze-at-current-length-on-hydraulic-power-loss",
                },
                "manual_corner_targets_m": {wheel: 0.0 for wheel in WHEEL_NAMES},
                "pose_presets": [
                    {"identity": "level", "label": "LEVEL", "corners": {wheel: 0.0 for wheel in WHEEL_NAMES}},
                    {"identity": "high", "label": "HIGH", "corners": {wheel: .54 for wheel in WHEEL_NAMES}},
                    {"identity": "low", "label": "LOW", "corners": {wheel: -.065 for wheel in WHEEL_NAMES}},
                    {"identity": "nose-up", "label": "NOSE", "corners": {
                        "front_left": .075, "front_right": .075, "rear_left": -.035, "rear_right": -.035}},
                    {"identity": "tail-up", "label": "TAIL", "corners": {
                        "front_left": -.035, "front_right": -.035, "rear_left": .075, "rear_right": .075}},
                    {"identity": "left-up", "label": "LEFT", "corners": {
                        "front_left": .07, "rear_left": .07, "front_right": -.03, "rear_right": -.03}},
                    {"identity": "right-up", "label": "RIGHT", "corners": {
                        "front_left": -.03, "rear_left": -.03, "front_right": .07, "rear_right": .07}},
                ],
                "programmable_slots": ["A", "B", "C"],
                "pose_law": "roll-pitch-height-error-to-four-corner-rest-length-targets",
                "force_law": "pose-changes-only-through-existing-spring-contact-wrenches",
            },
            "wheel_alignment": {
                "authority": "mechanical-control-arm-and-tie-rod-rest-length-actuators",
                "actuator_object": "force-limited-alignment-strain-relief-v1",
                "break_bushing_object": "replaceable-knuckle-break-bushing-v1",
                "installed_actuators_per_corner": ["upper-forward", "upper-rear", "tie-rod"],
                "installed_break_bushings_per_corner": ["upper-ball-joint", "lower-ball-joint",
                                                         "tie-rod-outer"],
                "parameter_units": "degrees",
                "compiled_parameter_pack": "performance-road-static-v1",
                "ranges": {"camber_deg": [-8.0, 5.0], "caster_deg": [-2.0, 12.0],
                           "toe_deg": [-3.0, 3.0]},
                "corners": {wheel: {"camber_deg": -1.0, "caster_deg": 5.5,
                                     "toe_deg": .08 if wheel.startswith("front") else .12}
                            for wheel in WHEEL_NAMES},
                "linked_editing": True,
                "actuator_mapping": {
                    "camber": "paired-upper-control-arm-length-offset",
                    "caster": "differential-upper-forward-rear-control-arm-length-offset",
                    "toe": "steering-tie-rod-rest-length-offset",
                },
                "strain_relief_law": (
                    "hold commanded alignment below relief force; backdrive a bounded series stroke "
                    "above relief force; dissipate relief work as heat; powered recenter draws pump energy"),
                "knuckle_fuse_law": (
                    "replaceable connector bushings open their graph constraints immediately before "
                    "the protected arm tie-rod or knuckle structure reaches plastic onset"),
                "auto_calibration": {
                    "requires_stationary_speed_below_m_s": .08,
                    "requires_wheel_speed_below_rad_s": .15,
                    "settled_ticks": 60,
                    "minimum_supported_wheels": 4,
                    "full_time_maximum_body_acceleration_m_s2": 2.5,
                    "full_time_maximum_friction_utilization": .35,
                    "correction_rate_m_per_s": .012,
                    "completion_tolerance_deg": .04,
                    "policy": "measure-settled-knuckle-geometry-and-trim-real-link-lengths-in-place",
                },
            },
            "pneumatic_system": {
                "compressor_node": "electrical.pneumatic_compressor",
                "accumulator_node": "electrical.pneumatic_accumulator",
                "tire_manifold_node": "electrical.pneumatic_tire_manifold",
                "consumers": ["four-tire-pressure-regulators", "four-pneumatic-shock-dampers"],
                "tire_route": "hard-line-service-loop-knuckle-rotary-union-drilled-hub-wheel-valve-cavity",
                "moving_interface": "rotary-union-is-knuckle-mounted-wheel-side-rotates-with-rim",
                "pressure_range_pa": [config.source["electrical"]["minimum_tire_pressure_pa"],
                                      config.source["electrical"]["maximum_tire_pressure_pa"]],
                "initial_tire_pressure_pa": config.source["tires"]["pressure_pa"],
            },
            "transmission_preset": "cj-wide-four-speed",
            "power_unit_preset": "amc-258-jeep-i6",
            "physics": {"authority": "resident-compiled-sympy-abstract-tensor-ssa-wgsl",
                        "parallel_spring_lanes": list(WHEEL_NAMES),
                        "webgpu_program": webgpu["identity"],
                        "mechanical_graph": mechanical_graph,
                        "contact_patch_shape": {
                            "law": "pressure-area-divided-by-load-sensitive-effective-tread-width",
                            "area_authority": "compiled-wheel-contact-kernel",
                            "width_range_of_declared_tread": [.65, .85],
                            "shape": "wide-short-balloon-tire-footprint",
                        },
                        "cage_contact": {
                            "authority": "lockstep-worker-wrench-reduction",
                            "samples": ["roll-cage-nodes", "cage-member-midpoints"],
                            "law": "spring-damper-normal-plus-static-kinetic-coulomb-friction",
                            "torque": "center-of-mass-r-cross-contact-force",
                            "projection": "none-one-sided-swept-contact-wrench-only",
                        },
                        "transmission_policy": {
                            "authority": "lockstep-worker-state",
                            "default": "automatic-second-gear-launch",
                            "crawler_entry": "torque-reserve-insufficient-in-second",
                            "manual_controls": ["automatic", "gear-down", "gear-up"],
                        },
                        "damage_model": {
                            "authority": "worker-owned-elastic-plastic-mechanical-graph-state",
                            "default_mode": "live-parametric-damage-graph/no-prebake",
                            "parameter_change_mode": "same-live-parameter-abi/no-kernel-rebake",
                            "spring": "progressive-bump-stop-then-plastic-set-then-fracture",
                            "natural_geometry": "deformed-rest-lengths-drive-node-and-vertex-solve",
                            "members": "axial-yield-plus-shear-fracture-on-frame-cage-rods-shocks-and-stabilizers",
                            "halfshafts": "torsional-fracture-opens-path-until-corresponding-axle-lock-reroutes",
                            "reset": "authored-natural-lengths-and-full-health-on-respawn",
                        },
                        "chassis_structure": chassis_structure,
                        "torque_graph": torque_graph},
            "control_mapping": {"move-forward/backward": "throttle", "strafe-left/right": "steering",
                                "run": "boost", "jump": "handbrake"},
            "pose": {"position": [0.0, float(chassis["clearance"]), 0.0],
                     "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        }],
        "terrain_sampling": {
            "service": f"{root}/physics/contact-surfaces",
            "consumer": "vehicle-contact-patch",
            "host_scope": "sample-the-shared-declared-surface-ABI-only",
        },
        "programs": [webgpu],
    }


__all__ = [
    "ABSTRACT_UI_VEHICLE_VERSION", "DEFAULT_CAR_CONFIG", "VEHICLE_STATE_OUTPUTS", "WHEEL_NAMES",
    "CONTACT_PATCH_OUTPUTS", "CONTACT_PATCH_TENSOR_REDUCER_OUTPUTS", "VehicleConfiguration", "compile_symbolic_vehicle_physics",
    "fit_vehicle_chassis_to_power_unit", "fit_vehicle_chassis_to_body_packaging",
    "fit_vehicle_wheelbase_under_body_mass", "solve_vehicle_body_packaging",
    "solve_vehicle_wheel_placement_mounts",
    "compile_wheel_contact_ssa", "compile_wheel_contact_wasm", "compile_wheel_contact_c", "compile_wheel_contact_abstract_tensor",
    "compile_wheel_contact_webgpu", "compile_vehicle_wrench_reduction_webgpu",
    "compile_sympy_matrix_to_abstract_tensor_backend", "extra_precision_closure",
    "compile_symbolic_vehicle_physics_wasm", "compile_symbolic_vehicle_physics_c", "compile_symbolic_vehicle_suspension_rig_wasm", "compile_symbolic_vehicle_suspension_rig_c",
    "SUSPENSION_RIG_OUTPUTS", "load_default_car_configuration",
    "symbolic_vehicle_equations", "symbolic_vehicle_physics_wasm_plugin",
    "symbolic_wheel_contact_equations", "symbolic_wheel_contact_wasm_plugin",
    "compile_symbolic_vehicle_physics_gpu_ssa", "compile_symbolic_vehicle_physics_webgpu",
    "compile_symbolic_vehicle_physics_webgpu_stages",
    "vehicle_webgpu_program_model",
    "set_vehicle_build_progress_sink",
    "vehicle_configuration_from_mapping", "vehicle_slot_model",
]
