"""JSON-configured vehicle slots and compiled parallel suspension physics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import sympy
from sympy.printing.pycode import PythonCodePrinter

from .abstract_ui_world import WorldWasmPlugin
from .ssa_wasm_backend import SSAWasmArtifact, emit_ssa_function_to_wasm
from .ssa_webgpu_backend import WGSLModule, emit_module as emit_webgpu_module
from .symbolic_equation_compiler import (
    SymbolicEquationCompilation,
    SymbolicPublication,
    compile_sympy_equations,
)


ABSTRACT_UI_VEHICLE_VERSION = "abstract-ui-vehicle-slot-v0"
DEFAULT_CAR_CONFIG = Path(__file__).parents[2] / "configs" / "vehicles" / "fun_car.json"
WHEEL_NAMES = ("front_left", "front_right", "rear_left", "rear_right")


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
    return tuple(lanes)


def contact_lane_count() -> int:
    return len(contact_patch_lanes())
VEHICLE_STATE_OUTPUTS = (
    "position_x_next", "position_y_next", "position_z_next",
    "velocity_x_next", "velocity_y_next", "velocity_z_next",
    "roll_next", "pitch_next", "yaw_next",
    "roll_velocity_next", "pitch_velocity_next", "yaw_velocity_next",
    *(f"wheel_omega_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"slip_longitudinal_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"slip_sensor_velocity_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"friction_utilization_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"friction_utilization_sensor_velocity_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"traction_scale_{wheel}" for wheel in WHEEL_NAMES),
    *(f"brake_scale_{wheel}" for wheel in WHEEL_NAMES),
    *(f"compression_{wheel}_next" for wheel in WHEEL_NAMES),
    *(f"spring_force_{wheel}" for wheel in WHEEL_NAMES),
    *(f"damper_scale_{wheel}" for wheel in WHEEL_NAMES),
    "engine_angular_speed_next", "engine_rpm",
    "engine_torque", "clutch_torque", "transmission_output_torque",
    "driveline_torque", "front_differential_torque", "rear_differential_torque",
    "engine_acceleration_torque", "engine_angular_acceleration",
    "powertrain_reaction_torque_x", "powertrain_reaction_torque_y",
    "powertrain_reaction_torque_z", "engine_mount_torque_x",
    "engine_mount_torque_y", "engine_mount_torque_z",
    "wheel_gyroscopic_reaction_torque_x", "wheel_gyroscopic_reaction_torque_y",
    "wheel_gyroscopic_reaction_torque_z",
)


def _number(mapping: Mapping[str, Any], name: str, *, positive: bool = False) -> float:
    value = mapping.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"vehicle field {name!r} must be numeric")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"vehicle field {name!r} must be positive")
    return result


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

    def mass_properties(self) -> dict[str, Any]:
        """Derive one conserved rigid-body mass model from the component layout."""

        source = self.source
        total = _number(source, "mass", positive=True)
        chassis, wheels, tires = source["chassis"], source["wheels"], source["tires"]
        powertrain, drivetrain = source["powertrain"], source["drivetrain"]
        wheelbase = _number(wheels, "wheelbase_half_length", positive=True)
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
             [wheelbase, .065, 0.0]),
            ("rear_differential", _number(powertrain, "rear_differential_mass_kg", positive=True),
             [-wheelbase, .065, 0.0]),
        ]
        suspension = source["suspension"]
        nominal_motion_ratio = .78
        for corner in WHEEL_NAMES:
            longitudinal, lateral = corner.split("_")
            static_load = total * abs(float(source["world"]["gravity"])) * float(
                source["mass_distribution"][corner])
            compression = min(float(suspension["travel"]), static_load / (
                float(suspension["stiffness"]) * nominal_motion_ratio ** 2))
            hub_y = (-float(chassis["clearance"]) - float(suspension["rest_length"])
                     + compression + radius)
            position = [wheelbase if longitudinal == "front" else -wheelbase,
                        hub_y, -track if lateral == "left" else track]
            component_specs.append((f"wheel_{corner}", wheel_mass, position))
            component_specs.append((f"tire_{corner}", tire_mass, list(position)))
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
        front_fraction = max(0.0, min(1.0, (center[0] + wheelbase) / (2 * wheelbase)))
        return {
            "total_mass_kg": total,
            "allocated_component_mass_kg": allocated,
            "residual_frame_cage_driver_mass_kg": residual,
            "center_of_mass": center,
            "inertia_kg_m2": {axis: inertias[index] for index, axis in enumerate(("roll", "pitch", "yaw"))},
            "derived_axle_fractions": {"front": front_fraction, "rear": 1 - front_fraction},
            "components": [{"identity": identity, "mass_kg": mass, "local_position": position}
                           for identity, mass, position in components],
        }

    def wheel_rotational_inertia(self) -> float:
        """Wheel, hub, and tire polar inertia derived from declared masses."""

        wheels, tires, drivetrain = (self.source["wheels"], self.source["tires"],
                                     self.source["drivetrain"])
        rim_radius = _number(wheels, "rim_radius", positive=True)
        tire_radius = _number(tires, "radius", positive=True)
        wheel_mass = _number(drivetrain, "wheel_mass_kg", positive=True)
        tire_mass = _number(drivetrain, "tire_mass_kg", positive=True)
        scale = _number(drivetrain, "rotational_inertia_scale", positive=True)
        # Rim is approximately a hoop; balloon tire is a thick annulus.
        return scale * (wheel_mass * rim_radius ** 2
                        + .5 * tire_mass * (rim_radius ** 2 + tire_radius ** 2))

    def parameter_defaults(self) -> dict[str, float]:
        suspension = self.source["suspension"]
        controls = self.source["controls"]
        drivetrain = self.source["drivetrain"]
        powertrain = self.source["powertrain"]
        chassis = self.source["chassis"]
        mass_properties = self.mass_properties()
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
        return {
            "inverse_mass": 1.0 / _number(self.source, "mass", positive=True),
            "gravity": _number(world, "gravity"),
            "suspension_rest_length": _number(suspension, "rest_length", positive=True),
            "suspension_travel": _number(suspension, "travel", positive=True),
            "chassis_clearance": _number(self.source["chassis"], "clearance", positive=True),
            "spring_stiffness": _number(suspension, "stiffness", positive=True),
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
            "track_half_width": _number(self.source["wheels"], "track_half_width", positive=True),
            **{f"linkage_motion_ratio_{wheel}": 1.0 for wheel in WHEEL_NAMES},
            **{f"inverse_inertia_{axis}": 1.0 / mass_properties["inertia_kg_m2"][axis]
               for axis in ("roll", "pitch", "yaw")},
            **{f"center_of_mass_{axis}": mass_properties["center_of_mass"][index]
               for index, axis in enumerate("xyz")},
            "engine_displacement_m3": _number(powertrain, "displacement_liters", positive=True) / 1000,
            "brake_mean_effective_pressure": _number(
                powertrain, "brake_mean_effective_pressure_pa", positive=True),
            "engine_braking_mean_effective_pressure": _number(
                powertrain, "engine_braking_mean_effective_pressure_pa", positive=True),
            **{f"engine_{name}_angular_speed": _number(powertrain, f"{name}_rpm", positive=True)
               * 2 * math.pi / 60 for name in ("idle", "torque_peak", "power_peak", "redline")},
            "engine_angular_speed": _number(powertrain, "idle_rpm", positive=True) * 2 * math.pi / 60,
            "drive_direction": 1.0,
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
            "aerodynamic_drag": _number(controls, "aerodynamic_drag", positive=True),
            **{name: _number(self.source["traction_control"], name, positive=True) for name in (
                "target_friction_utilization", "throttle_intervention_gain",
                "brake_intervention_gain", "slip_growth_gain",
                "slip_growth_reference_m_s2", "minimum_torque_fraction",
                "slip_sensor_frequency_hz", "slip_sensor_damping_ratio",
                "utilization_sensor_frequency_hz", "utilization_sensor_damping_ratio",
            )},
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
        "schema", "identity", "name", "kind", "mass", "mass_distribution", "chassis", "wheels", "tires",
        "suspension", "solid_contact", "drivetrain", "transmission", "powertrain", "controls", "traction_control",
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
        "wheels": {"track_half_width", "wheelbase_half_length", "rim_radius", "hub_face_offset"},
        "tires": {"radius", "pressure_pa", "width", "minimum_contact_area", "maximum_contact_area",
                  "static_friction", "kinetic_friction", "load_sensitivity",
                  "radial_stiffness_n_per_m", "radial_damping_n_s_per_m",
                  "longitudinal_stiffness", "lateral_stiffness", "slip_transition_speed"},
        "mass_distribution": set(WHEEL_NAMES),
        "suspension": {"rest_length", "travel", "stiffness", "pneumatic_compression_damping",
                       "pneumatic_rebound_damping", "pneumatic_efficiency",
                       "maximum_compression_speed", "active_damping_minimum_scale",
                       "active_damping_maximum_scale", "active_damping_body_velocity_gain_s_per_m",
                       "active_damping_rebound_release_gain_s_per_m"},
        "solid_contact": {"static_friction", "kinetic_friction", "restitution",
                          "penetration_bias", "maximum_correction_speed", "cage_contact_radius",
                          "cage_contact_stiffness", "cage_contact_damping", "cage_contact_maximum_force",
                          "cage_static_friction", "cage_kinetic_friction"},
        "drivetrain": {"brake_torque_nm", "front_drive_fraction", "rear_drive_fraction",
                       "rolling_resistance_torque_nm", "maximum_wheel_speed_rad_s",
                       "wheel_mass_kg", "tire_mass_kg", "rotational_inertia_scale",
                       "differential_lock_stiffness_nm_per_rad_s",
                       "differential_lock_maximum_torque_nm", "transfer_case_efficiency",
                       "transfer_case_drag_torque_nm"},
        "transmission": {"mode_default", "starting_gear", "crawler_gear", "forward_ratios", "reverse_ratio",
                         "ultra_low_range_ratio",
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
        "controls": {"maximum_steering_angle_degrees", "aerodynamic_drag", "angular_damping",
                     "throttle_rise_rate_per_s", "throttle_fall_rate_per_s",
                     "input_response_frequency_hz", "input_response_damping_ratio"},
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
    for section in ("chassis", "wheels", "tires", "suspension", "solid_contact", "drivetrain", "controls",
                    "traction_control",
                    "mass_distribution"):
        for name in expected[section]:
            _number(value[section], name, positive=True)
    for name in expected["powertrain"] - {"engine_position", "engine_orientation_degrees"}:
        _number(value["powertrain"], name, positive=True)
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
    for name in ("reverse_ratio", "ultra_low_range_ratio", "minimum_shift_interval_s", "upshift_torque_reserve",
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
                      + 4 * (float(value["drivetrain"]["wheel_mass_kg"])
                             + float(value["drivetrain"]["tire_mass_kg"])))
    if allocated_mass >= float(value["mass"]):
        raise ValueError("vehicle component masses must leave positive frame/cage/driver mass")
    if float(value["suspension"]["pneumatic_efficiency"]) > 1.0:
        raise ValueError("pneumatic efficiency cannot exceed one")
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
    return VehicleConfiguration(value, canonical, hashlib.sha256(canonical.encode()).hexdigest())


@lru_cache(maxsize=1)
def load_default_car_configuration() -> VehicleConfiguration:
    return vehicle_configuration_from_mapping(json.loads(DEFAULT_CAR_CONFIG.read_text(encoding="utf-8")))


def _symbols() -> dict[str, sympy.Symbol]:
    names = [
        "position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z",
        "roll", "pitch", "yaw", "roll_velocity", "pitch_velocity", "yaw_velocity",
        "dt", "throttle", "brake", "drive_direction", "yaw_cos", "yaw_sin",
        "inverse_mass", "gravity", "suspension_rest_length", "suspension_travel",
        "chassis_clearance",
        "spring_stiffness", "pneumatic_compression_damping", "pneumatic_rebound_damping",
        "pneumatic_efficiency", "maximum_compression_speed", "active_damping_minimum_scale",
        "active_damping_maximum_scale", "active_damping_body_velocity_gain_s_per_m",
        "active_damping_rebound_release_gain_s_per_m", "wheelbase_half_length", "track_half_width",
        "inverse_inertia_roll", "inverse_inertia_pitch",
        "inverse_inertia_yaw", "angular_damping", "aerodynamic_drag",
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
        "brake_torque", "wheel_inertia", "wheel_radius", "differential_lock",
        "differential_lock_stiffness", "differential_lock_maximum_torque",
        "rolling_resistance_torque", "maximum_wheel_speed",
        "target_friction_utilization", "throttle_intervention_gain", "brake_intervention_gain",
        "slip_growth_gain", "slip_growth_reference_m_s2", "minimum_torque_fraction",
        "slip_sensor_frequency_hz", "slip_sensor_damping_ratio",
        "utilization_sensor_frequency_hz", "utilization_sensor_damping_ratio",
        "total_force_x", "total_force_y", "total_force_z",
        "total_torque_x", "total_torque_y", "total_torque_z",
        "contact_wrench_force_x", "contact_wrench_force_y", "contact_wrench_force_z",
        "contact_wrench_torque_x", "contact_wrench_torque_y", "contact_wrench_torque_z",
    ]
    for wheel in WHEEL_NAMES:
        names.extend((f"compression_{wheel}", f"wheel_height_{wheel}", f"wheel_support_{wheel}",
                      f"target_compression_{wheel}",
                      f"wheel_omega_{wheel}", f"longitudinal_force_{wheel}",
                      f"slip_longitudinal_{wheel}", f"previous_slip_longitudinal_{wheel}",
                      f"slip_sensor_velocity_{wheel}",
                      f"measured_friction_utilization_{wheel}",
                      f"friction_utilization_{wheel}",
                      f"friction_utilization_sensor_velocity_{wheel}",
                      f"drive_fraction_{wheel}", f"linkage_motion_ratio_{wheel}"))
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


@lru_cache(maxsize=1)
def symbolic_vehicle_equations() -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """One transition with four independent spring lanes and one body reduction."""

    s = _symbols()
    dt = s["dt"]
    compressions: dict[str, sympy.Basic] = {}
    forces: dict[str, sympy.Basic] = {}
    damping_scales: dict[str, sympy.Basic] = {}
    for wheel_index, wheel in enumerate(WHEEL_NAMES):
        target = _c2_clamp(s[f"target_compression_{wheel}"], 0,
                           s["suspension_travel"], sympy.Float("0.012")) * s[f"wheel_support_{wheel}"]
        compressions[wheel] = target
        rate = _c2_clamp((target - s[f"compression_{wheel}"]) / dt,
                         -s["maximum_compression_speed"], s["maximum_compression_speed"],
                         sympy.Float("0.08"))
        front_sign = 1 if wheel_index < 2 else -1
        side_sign = -1 if wheel_index % 2 == 0 else 1
        corner_body_velocity = (s["velocity_y"]
                                + front_sign * s["pitch_velocity"] * s["wheelbase_half_length"]
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
        forces[wheel] = _c2_positive((s["spring_stiffness"] * target * motion_ratio
                                      + pneumatic) * motion_ratio, sympy.Float("60"))

    net_force = tuple(s[f"total_force_{axis}"] + s[f"contact_wrench_force_{axis}"]
                      for axis in "xyz")
    net_torque = tuple(s[f"total_torque_{axis}"] + s[f"contact_wrench_torque_{axis}"]
                       for axis in "xyz")
    velocity_x_next = s["velocity_x"] + dt * s["inverse_mass"] * (
        net_force[0] - s["aerodynamic_drag"] * _smooth_abs(s["velocity_x"]) * s["velocity_x"])
    velocity_y_next = s["velocity_y"] + dt * (s["gravity"] + s["inverse_mass"] * (
        net_force[1] - s["aerodynamic_drag"] * _smooth_abs(s["velocity_y"]) * s["velocity_y"]))
    velocity_z_next = s["velocity_z"] + dt * s["inverse_mass"] * (
        net_force[2] - s["aerodynamic_drag"] * _smooth_abs(s["velocity_z"]) * s["velocity_z"])
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
    combustion_torque = (throttle_magnitude * indicated_torque
                         * s["combustion_efficiency"] * torque_curve * redline_rolloff)
    idle_error = _c2_clamp((s["engine_idle_angular_speed"] - s["engine_angular_speed"])
                           / (s["engine_idle_angular_speed"] * sympy.Float("0.18")),
                           0, 1, sympy.Float("0.035"))
    idle_governor_torque = indicated_torque * sympy.Float("0.38") * idle_error
    engine_braking_torque = (s["engine_braking_mean_effective_pressure"]
                             * s["engine_displacement_m3"] / (4 * sympy.pi)
                             * (1 - throttle_magnitude) * s["engine_angular_speed"]
                             / sympy.sqrt(s["engine_angular_speed"] ** 2
                                          + s["engine_idle_angular_speed"] ** 2))
    direction = _c2_clamp(s["drive_direction"], -1, 1, sympy.Float("0.025"))
    signed_gear = ((1 + direction) / 2 * s["forward_gear_ratio"]
                   - (1 - direction) / 2 * s["reverse_gear_ratio"])
    mean_wheel_speed = sum(_smooth_abs(s[f"wheel_omega_{wheel}"]) for wheel in WHEEL_NAMES) / 4
    coupled_crank_speed = (mean_wheel_speed * _smooth_abs(signed_gear)
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
    )
    engine_torque = combustion_torque + idle_governor_torque - engine_braking_torque
    raw_clutch_torque = (clutch_engagement * s["clutch_maximum_torque"]
                         * sympy.tanh(s["clutch_stiffness"]
                                      * (s["engine_angular_speed"] - coupled_crank_speed)
                                      / s["clutch_maximum_torque"]))
    # During launch a friction clutch/torque converter cannot transmit more
    # positive torque than the crank is presently producing without stalling
    # it. Preserve negative overrun torque as a separate branch so engine
    # braking remains physical once road speed has engaged the overrun clutch.
    requested_drive_torque = _c2_positive(raw_clutch_torque, sympy.Float("0.01"))
    requested_overrun_torque = raw_clutch_torque - requested_drive_torque
    launch_torque_limit = (_c2_positive(engine_torque, sympy.Float("0.01"))
                           * sympy.Float("0.90"))
    transmitted_drive_torque = launch_torque_limit * sympy.tanh(
        requested_drive_torque / (launch_torque_limit + sympy.Float("0.001"))
    )
    clutch_torque = transmitted_drive_torque + requested_overrun_torque
    transmission_output_torque = clutch_torque * signed_gear * s["clutch_efficiency"]
    transfer_case_input_torque = transmission_output_torque * s["transfer_case_ratio"]
    transfer_case_direction = transfer_case_input_torque / _smooth_abs(transfer_case_input_torque)
    transfer_case_output_torque = transfer_case_direction * _c2_positive(
        _smooth_abs(transfer_case_input_torque) - s["transfer_case_drag_torque"],
        sympy.Float("0.5"))
    driveline_torque = (transfer_case_output_torque * s["transfer_case_efficiency"]
                        * s["final_drive_ratio"] * s["driveline_efficiency"])
    front_differential_torque = driveline_torque * (
        s["drive_fraction_front_left"] + s["drive_fraction_front_right"])
    rear_differential_torque = driveline_torque * (
        s["drive_fraction_rear_left"] + s["drive_fraction_rear_right"])
    transmitted_crank_load = clutch_torque
    engine_acceleration_torque = engine_torque - transmitted_crank_load
    engine_angular_acceleration = engine_acceleration_torque / s["engine_rotating_inertia"]
    engine_angular_speed_raw = (s["engine_angular_speed"] + dt * engine_angular_acceleration)
    engine_angular_speed_next = _c2_clamp(
        engine_angular_speed_raw, 0, s["engine_redline_angular_speed"] * sympy.Float("1.03"),
        sympy.Float("0.5"),
    )

    acceleration = (
        s["inverse_mass"] * (net_force[0]
                             - s["aerodynamic_drag"] * _smooth_abs(s["velocity_x"]) * s["velocity_x"]),
        s["gravity"] + s["inverse_mass"] * (net_force[1]
                                             - s["aerodynamic_drag"] * _smooth_abs(s["velocity_y"]) * s["velocity_y"]),
        s["inverse_mass"] * (net_force[2]
                             - s["aerodynamic_drag"] * _smooth_abs(s["velocity_z"]) * s["velocity_z"]),
    )
    acceleration_chassis = (
        acceleration[0] * s["yaw_cos"] + acceleration[2] * s["yaw_sin"],
        acceleration[1] - s["gravity"],
        -acceleration[0] * s["yaw_sin"] + acceleration[2] * s["yaw_cos"],
    )
    engine_inertial_force_local = tuple(-s["engine_mass"] * value for value in acceleration_chassis)
    transmission_inertial_force_local = tuple(
        -s["transmission_mass"] * value for value in acceleration_chassis)
    transfer_case_inertial_force_local = tuple(
        -s["transfer_case_mass"] * value for value in acceleration_chassis)
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
    wheel_angular_momentum = s["wheel_inertia"] * sum(
        s[f"wheel_omega_{wheel}"] for wheel in WHEEL_NAMES
    )
    # Signed Ω×H reaction of the spinning wheel/tire assemblies. H is along
    # chassis-right; reverse wheel rotation therefore reverses the couple.
    wheel_gyroscopic_reaction = (
        -s["yaw_velocity"] * wheel_angular_momentum,
        s["roll_velocity"] * wheel_angular_momentum,
        sympy.Integer(0),
    )
    roll_torque = (net_torque[0] * s["yaw_cos"] + net_torque[2] * s["yaw_sin"]
                   + powertrain_reaction[0] + wheel_gyroscopic_reaction[0])
    pitch_torque = (-net_torque[0] * s["yaw_sin"] + net_torque[2] * s["yaw_cos"]
                    + powertrain_reaction[2])
    yaw_torque = net_torque[1] + powertrain_reaction[1] + wheel_gyroscopic_reaction[1]
    roll_velocity_next = (s["roll_velocity"] + dt * roll_torque * s["inverse_inertia_roll"]) / (1 + dt * s["angular_damping"])
    pitch_velocity_next = (s["pitch_velocity"] + dt * pitch_torque * s["inverse_inertia_pitch"]) / (1 + dt * s["angular_damping"])
    yaw_velocity_next = (s["yaw_velocity"] + dt * yaw_torque * s["inverse_inertia_yaw"]) / (1 + dt * s["angular_damping"])

    wheel_omegas: dict[str, sympy.Basic] = {}
    traction_scales: dict[str, sympy.Basic] = {}
    brake_scales: dict[str, sympy.Basic] = {}
    drive_torque = driveline_torque
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
    for wheel in WHEEL_NAMES:
        omega = s[f"wheel_omega_{wheel}"]
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
        slip_magnitude_growth = (filtered_slip / _smooth_abs(filtered_slip)
                                 * slip_sensor_velocity)
        slip_growth = _c2_positive(
            slip_magnitude_growth / s["slip_growth_reference_m_s2"],
            sympy.Float("0.08"),
        )
        traction_target = 1 / (
            1 + s["throttle_intervention_gain"] * utilization_excess
            + s["slip_growth_gain"] * slip_growth
        )
        brake_target = 1 / (
            1 + s["brake_intervention_gain"] * utilization_excess
            + s["slip_growth_gain"] * slip_growth
        )
        traction_scales[wheel] = _c2_clamp(traction_target, s["minimum_torque_fraction"], 1,
                                            sympy.Float("0.035"))
        brake_scales[wheel] = _c2_clamp(brake_target, s["minimum_torque_fraction"], 1,
                                        sympy.Float("0.035"))
        smooth_direction = omega / _smooth_abs(omega)
        axle_torque = drive_torque * s[f"drive_fraction_{wheel}"] * traction_scales[wheel]
        lock_torque = (s["differential_lock"] * s["differential_lock_maximum_torque"]
                       * sympy.tanh(s["differential_lock_stiffness"]
                                    * (s[f"wheel_omega_{opposite_wheel[wheel]}"] - omega)
                                    / s["differential_lock_maximum_torque"]))
        tire_reaction = s[f"longitudinal_force_{wheel}"] * s["wheel_radius"]
        resisting = smooth_direction * (s["brake_torque"] * s["brake"] * brake_scales[wheel]
                                         + s["rolling_resistance_torque"])
        raw_omega = omega + dt * (axle_torque + lock_torque - tire_reaction - resisting) / s["wheel_inertia"]
        wheel_omegas[wheel] = (s["maximum_wheel_speed"] * raw_omega
                               / sympy.sqrt(s["maximum_wheel_speed"] ** 2 + raw_omega ** 2))

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
        **{f"slip_longitudinal_{wheel}_next": filtered_slips[wheel]
           for wheel in WHEEL_NAMES},
        **{f"slip_sensor_velocity_{wheel}_next": slip_sensor_velocities[wheel]
           for wheel in WHEEL_NAMES},
        **{f"friction_utilization_{wheel}_next": filtered_utilizations[wheel]
           for wheel in WHEEL_NAMES},
        **{f"friction_utilization_sensor_velocity_{wheel}_next":
           utilization_sensor_velocities[wheel] for wheel in WHEEL_NAMES},
        **{f"traction_scale_{wheel}": traction_scales[wheel] for wheel in WHEEL_NAMES},
        **{f"brake_scale_{wheel}": brake_scales[wheel] for wheel in WHEEL_NAMES},
        **{f"compression_{wheel}_next": compressions[wheel] for wheel in WHEEL_NAMES},
        **{f"spring_force_{wheel}": forces[wheel] for wheel in WHEEL_NAMES},
        **{f"damper_scale_{wheel}": damping_scales[wheel] for wheel in WHEEL_NAMES},
        "engine_angular_speed_next": engine_angular_speed_next,
        "engine_rpm": engine_angular_speed_next * 60 / (2 * sympy.pi),
        "engine_torque": engine_torque,
        "clutch_torque": clutch_torque,
        "transmission_output_torque": transmission_output_torque,
        "driveline_torque": driveline_torque,
        "front_differential_torque": front_differential_torque,
        "rear_differential_torque": rear_differential_torque,
        "engine_acceleration_torque": engine_acceleration_torque,
        "engine_angular_acceleration": engine_angular_acceleration,
        **{f"powertrain_reaction_torque_{axis}": powertrain_reaction[index]
           for index, axis in enumerate("xyz")},
        **{f"engine_mount_torque_{axis}": mount_torque[index]
           for index, axis in enumerate("xyz")},
        **{f"wheel_gyroscopic_reaction_torque_{axis}": wheel_gyroscopic_reaction[index]
           for index, axis in enumerate("xyz")},
    }
    equations = tuple(sympy.Eq(sympy.Symbol(name, real=True), expression, evaluate=False)
                      for name, expression in expressions.items())
    return equations, s


@lru_cache(maxsize=1)
def compile_symbolic_vehicle_physics() -> SymbolicEquationCompilation:
    equations, _ = symbolic_vehicle_equations()
    publications = tuple(SymbolicPublication(name, f"world.vehicle.{name}")
                         for name in VEHICLE_STATE_OUTPUTS)
    return compile_sympy_equations(equations, name="abstract_ui_vehicle_step", publications=publications)


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
    equations, _ = symbolic_vehicle_equations()
    publications = tuple(SymbolicPublication(name, f"world.vehicle.{name}")
                         for name in VEHICLE_STATE_OUTPUTS)
    return compile_sympy_equations(
        equations, name="abstract_ui_vehicle_step_gpu",
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


@lru_cache(maxsize=1)
def symbolic_wheel_contact_equations() -> tuple[tuple[sympy.Equality, ...], dict[str, sympy.Symbol]]:
    """Per-wheel mesh-contact, pneumatic patch, and Coulomb friction law."""

    names = (
        "dt support hub_height hub_velocity_y previous_compression geometric_compression surface_height "
        "chassis_velocity_y roll_velocity pitch_velocity wheelbase_half_length track_half_width "
        "corner_front_sign corner_side_sign active_damping_minimum_scale active_damping_maximum_scale "
        "active_damping_body_velocity_gain_s_per_m active_damping_rebound_release_gain_s_per_m "
        "normal_x normal_y normal_z forward_x forward_y forward_z right_x right_y right_z "
        "tire_radial_compression tire_radial_velocity suspension_alignment "
        "tire_radial_stiffness tire_radial_damping "
        "slip_longitudinal slip_lateral attachment_x attachment_y attachment_z "
        "corner_weight suspension_rest_length chassis_clearance suspension_travel "
        "spring_stiffness linkage_motion_ratio pneumatic_compression_damping pneumatic_rebound_damping pneumatic_efficiency "
        "maximum_compression_speed tire_pressure minimum_contact_area maximum_contact_area "
        "mu_static mu_kinetic load_sensitivity longitudinal_stiffness lateral_stiffness "
        "slip_transition_speed"
    )
    s = {name: sympy.Symbol(name, real=True) for name in names.split()}
    epsilon = sympy.Float("1e-5")
    compression = _c2_clamp(s["geometric_compression"], 0, s["suspension_travel"],
                            sympy.Float("0.012")) * s["support"]
    compression_rate = _c2_clamp((compression - s["previous_compression"]) / s["dt"],
                                 -s["maximum_compression_speed"], s["maximum_compression_speed"],
                                 sympy.Float("0.08"))
    corner_body_velocity = (s["chassis_velocity_y"]
                            + s["corner_front_sign"] * s["pitch_velocity"]
                            * s["wheelbase_half_length"]
                            - s["corner_side_sign"] * s["roll_velocity"]
                            * s["track_half_width"])
    raw_damping_scale = (1
                         + s["active_damping_body_velocity_gain_s_per_m"]
                         * _smooth_abs(corner_body_velocity)
                         - s["active_damping_rebound_release_gain_s_per_m"]
                         * _c2_positive(-compression_rate, sympy.Float("0.08")))
    damping_scale = _c2_clamp(raw_damping_scale, s["active_damping_minimum_scale"],
                              s["active_damping_maximum_scale"], sympy.Float("0.025"))
    pneumatic = s["pneumatic_efficiency"] * (
        damping_scale * s["pneumatic_compression_damping"]
        * _c2_positive(compression_rate, sympy.Float("0.08"))
        - damping_scale * s["pneumatic_rebound_damping"]
        * _c2_positive(-compression_rate, sympy.Float("0.08"))
    )
    suspension_load = _c2_positive(
        (s["spring_stiffness"] * compression * s["linkage_motion_ratio"] + pneumatic)
        * s["linkage_motion_ratio"], sympy.Float("60")) * s["support"]
    carcass_load = _c2_positive(
        s["tire_radial_stiffness"] * s["tire_radial_compression"]
        - s["tire_radial_damping"] * s["tire_radial_velocity"],
        sympy.Float("60")) * s["support"]
    normal_load = (s["suspension_alignment"] * suspension_load
                   + (1 - s["suspension_alignment"]) * carcass_load)
    contact_area = _c2_clamp(normal_load / s["tire_pressure"], s["minimum_contact_area"],
                             s["maximum_contact_area"], sympy.Float("0.0015")) * s["support"]
    reference_area = s["corner_weight"] / s["tire_pressure"]
    patch_scale = _c2_clamp(contact_area / (reference_area + epsilon), sympy.Float("0.62"),
                            sympy.Float("1.18"), sympy.Float("0.04"))
    overload = _c2_positive(normal_load / (s["corner_weight"] + epsilon) - 1,
                            sympy.Float("0.08"))
    load_scale = _c2_clamp(1 - s["load_sensitivity"] * overload, sympy.Float("0.58"), 1,
                           sympy.Float("0.04"))
    requested_long = -s["longitudinal_stiffness"] * s["slip_longitudinal"]
    requested_lateral = -s["lateral_stiffness"] * s["slip_lateral"]
    requested_magnitude = sympy.sqrt(requested_long ** 2 + requested_lateral ** 2 + epsilon ** 2)
    slip_speed = sympy.sqrt(s["slip_longitudinal"] ** 2 + s["slip_lateral"] ** 2
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
def compile_wheel_contact_ssa() -> SymbolicEquationCompilation:
    equations, _ = symbolic_wheel_contact_equations()
    publications = tuple(SymbolicPublication(name, f"world.vehicle.contact.{name}")
                         for name in CONTACT_PATCH_OUTPUTS)
    return compile_sympy_equations(
        equations, name="abstract_ui_wheel_contact", publications=publications, dtype="float32",
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


class _AbstractTensorPythonPrinter(PythonCodePrinter):
    """Spell supported SymPy intrinsics as native AbstractTensor methods."""

    def _print_Pow(self, expression):  # noqa: N802 - SymPy printer protocol
        if expression.exp == sympy.S.Half:
            return f"({self._print(expression.base)}).sqrt()"
        return super()._print_Pow(expression)

    def _print_tanh(self, expression):
        return f"({self._print(expression.args[0])}).tanh()"


_ABSTRACT_TENSOR_PYTHON_PRINTER = _AbstractTensorPythonPrinter()


def _abstract_tensor_python(expression: sympy.Basic) -> str:
    """Print scalar SymPy as elementwise AbstractTensor-friendly Python."""

    return _ABSTRACT_TENSOR_PYTHON_PRINTER.doprint(expression)


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

    equations, _ = symbolic_wheel_contact_equations()
    output_symbols = {equation.lhs for equation in equations}
    # Match ``compile_sympy_equations``' stable ABI rule directly.  Invoking
    # the complete scalar compiler here merely to rediscover this sorted set
    # used to make the tensor precompile pay for two independent compilations.
    argument_names = tuple(sorted({
        str(symbol)
        for equation in equations
        for symbol in equation.rhs.free_symbols - output_symbols
    }))
    replacements, reduced = sympy.cse(
        [equation.rhs for equation in equations],
        symbols=sympy.numbered_symbols("tensor_tmp_"),
        order="canonical",
    )
    lines = [f"def abstract_ui_wheel_contact_tensor({', '.join(argument_names)}):"]
    lines.extend(
        f"    {temporary} = {_abstract_tensor_python(expression)}"
        for temporary, expression in replacements
    )
    returned = ", ".join(_abstract_tensor_python(expression) for expression in reduced)
    lines.append(f"    return {returned}")
    source = "\n".join(lines) + "\n"
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

    lanes = contact_lane_count()

    def store(name: str, expression: str) -> str:
        return f"  contact_feed[{contact_index[name]}u * {lanes}u + lane] = {expression};"

    def cage_store(name: str, expression: str) -> str:
        return f"  contact_feed[{contact_index[name]}u * {lanes}u + lane] = {expression};"

    static_compressions = {
        wheel: min(
            float(suspension["travel"]),
            float(source["mass"]) * abs(float(source["world"]["gravity"]))
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
        "lateral_stiffness": tires["lateral_stiffness"],
        "load_sensitivity": tires["load_sensitivity"],
        "longitudinal_stiffness": tires["longitudinal_stiffness"],
        "maximum_compression_speed": suspension["maximum_compression_speed"],
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
        "tire_radial_stiffness": tires["radial_stiffness_n_per_m"],
        "tire_radial_damping": tires["radial_damping_n_s_per_m"],
        "track_half_width": wheels["track_half_width"],
        "wheelbase_half_length": wheels["wheelbase_half_length"],
    }
    expressions = {
        **{name: _wgsl_number(value) for name, value in constants.items()},
        "attachment_x": "attachment.x", "attachment_y": "attachment.y", "attachment_z": "attachment.z",
        "chassis_velocity_y": vehicle("velocity_y"),
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
        "support": "support",
    }
    missing = set(contact_inputs) - set(expressions)
    if missing:
        raise RuntimeError(f"terrain contact kernel does not populate contact inputs: {sorted(missing)}")
    stores = "\n".join(store(name, expressions[name]) for name in contact_inputs)
    template = r'''struct TerrainSample {
  height: f32,
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
fn terrain_sample(x: f32, z: f32) -> TerrainSample {
  let origin_x = terrain_parameters[0u]; let origin_y = terrain_parameters[1u];
  let origin_z = terrain_parameters[2u]; let cell_x = terrain_parameters[3u];
  let cell_z = terrain_parameters[4u]; let columns = u32(terrain_parameters[5u]);
  let rows = u32(terrain_parameters[6u]); let minimum_x = terrain_parameters[7u];
  let maximum_x = terrain_parameters[8u]; let minimum_z = terrain_parameters[9u];
  let maximum_z = terrain_parameters[10u];
  if (columns < 2u || rows < 2u || x < minimum_x || x > maximum_x || z < minimum_z || z > maximum_z) {
    return TerrainSample(0.0f, vec3<f32>(0.0f, 1.0f, 0.0f), 1u);
  }
  let u = clamp((x - origin_x) / cell_x, 0.0f, f32(columns - 1u));
  let v = clamp((z - origin_z) / cell_z, 0.0f, f32(rows - 1u));
  let column = min(columns - 2u, u32(floor(u))); let row = min(rows - 2u, u32(floor(v)));
  let tx = u - f32(column); let tz = v - f32(row);
  let h00 = terrain_heights[row * columns + column];
  let h10 = terrain_heights[row * columns + column + 1u];
  let h01 = terrain_heights[(row + 1u) * columns + column];
  let h11 = terrain_heights[(row + 1u) * columns + column + 1u];
  var height: f32; var gradient: vec2<f32>;
  if (tx >= tz) {
    height = h00 + (h10 - h00) * tx + (h11 - h10) * tz;
    gradient = vec2<f32>((h10 - h00) / cell_x, (h11 - h10) / cell_z);
  } else {
    height = h00 + (h11 - h01) * tx + (h01 - h00) * tz;
    gradient = vec2<f32>((h11 - h01) / cell_x, (h01 - h00) / cell_z);
  }
  return TerrainSample(origin_y + height, safe_normalize(vec3<f32>(-gradient.x, 1.0f, -gradient.y),
    vec3<f32>(0.0f, 1.0f, 0.0f)), 1u);
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
  let sample = terrain_sample(patch_world.x, patch_world.z);
  let squash = sample.height + @@CAGE_RADIUS@@ - patch_world.y;
  let touching = select(0.0f, 1.0f, sample.valid != 0u && squash > 0.0f
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
  let steer = select(0.0f, -controls[1u] * @@MAX_STEER@@, lane < 2u);
  let rolling_axis = forward_axis * cos(steer) + right_axis * sin(steer);
  let axle = right_axis * cos(steer) - forward_axis * sin(steer);
  var best_score = -1.0e20f; var surface_point = world_hub + down * @@TIRE_RADIUS@@;
  var surface_normal = vec3<f32>(0.0f, 1.0f, 0.0f); var support = 0.0f;
  var support_position = position; var tire_radial_compression = 0.0f;
  for (var radial_index = 0u; radial_index < 5u; radial_index += 1u) {
    let angle = RADIAL_ANGLES[radial_index];
    let radial = down * cos(angle) + rolling_axis * sin(angle);
    for (var lateral_index = 0u; lateral_index < 3u; lateral_index += 1u) {
      let probe = world_hub + radial * @@TIRE_RADIUS@@ + axle * (@@TIRE_WIDTH@@ * LATERAL_FRACTIONS[lateral_index]);
      let probe_velocity = velocity + cross(angular, probe - position); let next_probe = probe + probe_velocity * @@FIXED_DT@@;
      let sample = terrain_sample(probe.x, probe.z); let next_sample = terrain_sample(next_probe.x, next_probe.z);
      let hub_sample = terrain_sample(world_hub.x, world_hub.z);
      let current_clearance = probe.y - sample.height; let next_clearance = next_probe.y - next_sample.height;
      let hub_clearance = world_hub.y - hub_sample.height;
      // Spatial crossing recovers a tire whose outer probe is already below
      // the one-sided top skin while its hub is still above it.  Temporal
      // crossing prevents the same event before a fast probe reaches the
      // other side during this fixed step.
      let spatial_crossing = hub_clearance >= -0.006f && current_clearance <= 0.006f;
      let temporal_crossing = current_clearance >= -0.006f && next_clearance <= 0.006f
        && next_clearance < current_clearance;
      let crossed_skin = spatial_crossing || temporal_crossing;
      var point = vec3<f32>(probe.x, sample.height, probe.z); var candidate_normal = sample.normal;
      var evaluation_hub = world_hub; var evaluation_position = position;
      if (spatial_crossing) {
        let fraction = clamp(hub_clearance / max(1.0e-6f, hub_clearance - current_clearance), 0.0f, 1.0f);
        let crossing_probe = mix(world_hub, probe, fraction);
        let crossing_sample = terrain_sample(crossing_probe.x, crossing_probe.z);
        point = vec3<f32>(crossing_probe.x, crossing_sample.height, crossing_probe.z);
        candidate_normal = crossing_sample.normal;
      } else if (temporal_crossing) {
        let fraction = clamp(current_clearance / max(1.0e-6f, current_clearance - next_clearance), 0.0f, 1.0f);
        let crossing_probe = mix(probe, next_probe, fraction); let crossing_sample = terrain_sample(crossing_probe.x, crossing_probe.z);
        point = vec3<f32>(crossing_probe.x, crossing_sample.height, crossing_probe.z);
        candidate_normal = crossing_sample.normal;
        evaluation_hub = world_hub + hub_velocity * (@@FIXED_DT@@ * fraction);
        evaluation_position = position + velocity * (@@FIXED_DT@@ * fraction);
      }
      let hub_to_surface = evaluation_hub - point; let radial_distance = max(1.0e-8f, length(hub_to_surface));
      let normal_distance = dot(hub_to_surface, candidate_normal); let alignment = normal_distance / radial_distance;
      let penetration = @@TIRE_RADIUS@@ - normal_distance; let score = penetration + alignment * 0.003f;
      if (sample.valid != 0u && (crossed_skin || normal_distance <= @@CONTACT_REACH@@) && alignment >= 0.12f && score > best_score) {
        best_score = score; surface_point = point; surface_normal = candidate_normal;
        support_position = evaluation_position; tire_radial_compression = max(0.0f, penetration);
      }
    }
  }
  // A tire is a short capsule along its axle.  This radial closest-point
  // query lets the same tire touch wall faces and edges; drivetrain torque
  // then acts in the wall tangent, allowing a climb only when geometry,
  // available torque, normal load and Coulomb friction permit it.
  let wall_count = u32(terrain_parameters[11u]);
  for (var wall_index = 0u; wall_index < wall_count; wall_index += 1u) {
    let base = wall_index * 6u;
    let wall_minimum = vec3<f32>(wall_colliders[base], wall_colliders[base + 1u], wall_colliders[base + 2u]);
    let wall_maximum = vec3<f32>(wall_colliders[base + 3u], wall_colliders[base + 4u], wall_colliders[base + 5u]);
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
  // GROUND IS A SOLID, NOT A SKIN.  Every probe above asks "did this tyre
  // cross the surface this step"; a hub already beneath the height field
  // crossed nothing, its alignment goes negative, and every candidate is
  // rejected -- so a buried wheel finds no contact at all and falls forever.
  // When nothing was found, the field answers directly from below, reported at
  // the bottom of the carcass (reporting at the hub reads as a fully bottomed
  // suspension and the law answers with tens of kilonewtons).
  if (best_score <= -1.0e19f) {
    let buried_sample = terrain_sample(world_hub.x, world_hub.z);
    let buried_depth = buried_sample.height - world_hub.y;
    if (buried_sample.valid != 0u && buried_depth > -@@TIRE_RADIUS@@) {
      let squash = clamp(buried_depth + @@TIRE_RADIUS@@, 0.0f, 0.06f);
      surface_point = vec3<f32>(world_hub.x, world_hub.y - @@TIRE_RADIUS@@ + squash, world_hub.z);
      surface_normal = buried_sample.normal;
      support_position = position;
      tire_radial_compression = squash;
      best_score = 0.0f;
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
        "corner_front_sign": "select(-1.0f, 1.0f, patch_local.x >= 0.0f)",
        "corner_side_sign": "select(-1.0f, 1.0f, patch_local.z >= 0.0f)",
        "corner_weight": _wgsl_number(cage_corner_weight),
        "dt": _wgsl_number(1 / float(source["world"]["fixed_step_hz"])),
        "forward_x": "along.x", "forward_y": "along.y", "forward_z": "along.z",
        "geometric_compression": "0.0f",
        "lateral_stiffness": _wgsl_number(float(solid["cage_contact_stiffness"]) * .3),
        "linkage_motion_ratio": "1.0f",
        "load_sensitivity": "0.075f",
        "longitudinal_stiffness": _wgsl_number(float(solid["cage_contact_stiffness"]) * .3),
        "maximum_compression_speed": "1.25f",
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
        "tire_radial_compression": "min(max(0.0f, squash), 0.05f) * touching",
        # A tube can end a step buried; the law must see a bounded squash or the
        # radial spring answers with tens of kilonewtons and the shell launches.
        "tire_radial_damping": _wgsl_number(float(solid["cage_contact_damping"])),
        "tire_radial_stiffness": _wgsl_number(float(solid["cage_contact_stiffness"])),
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
        "@@TIRE_WIDTH@@": _wgsl_number(tires["width"]),
        "@@CONTACT_REACH@@": _wgsl_number(tires["radius"] + suspension["travel"] + .025),
        "@@FIXED_DT@@": _wgsl_number(1 / float(source["world"]["fixed_step_hz"])),
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
        "inputs": list(contact_inputs),
        "kernel": {
            "source": template, "entrypoint": "vehicle_terrain_contact_geometry",
            "workgroup_size": [lanes, 1, 1], "dispatch": [1, 1, 1], "invocations": lanes,
            "lane_mapping": [identity for _, identity in contact_patch_lanes()],
            "bindings": ["terrain_heights", "terrain_parameters", "vehicle_feed", "contact_feed", "controls",
                         "wall_colliders"],
        },
        "terrain_parameter_abi": [
            "origin_x", "origin_y", "origin_z", "cell_x", "cell_z", "columns", "rows",
            "minimum_x", "maximum_x", "minimum_z", "maximum_z",
            "wall_count",
        ],
    }


def _vehicle_gpu_graph_adapters(
    contact_inputs: tuple[str, ...], contact_outputs: tuple[str, ...],
    vehicle_inputs: tuple[str, ...], vehicle_outputs: tuple[str, ...],
) -> dict[str, Any]:
    """Generate storage-to-storage graph edges around compiler kernels."""

    ci = {name: index for index, name in enumerate(contact_inputs)}
    co = {name: index for index, name in enumerate(contact_outputs)}
    vi = {name: index for index, name in enumerate(vehicle_inputs)}
    lanes = contact_lane_count()
    vo = {name: index for index, name in enumerate(vehicle_outputs)}
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
        wheel_lines.extend([
            f"  vehicle_feed[{vi[f'longitudinal_force_{wheel}']}u] = finite_or({dot_force(forward)}, 0.0f);",
            f"  vehicle_feed[{vi[f'slip_longitudinal_{wheel}']}u] = "
            f"finite_or(contact_feed[{ci['slip_longitudinal']}u * {lanes}u + {lane}u], 0.0f);",
            f"  vehicle_feed[{vi[f'target_compression_{wheel}']}u] = "
            f"finite_or(contact_feed[{ci['geometric_compression']}u * {lanes}u + {lane}u], 0.0f);",
            f"  vehicle_feed[{vi[f'wheel_support_{wheel}']}u] = "
            f"finite_or(contact_feed[{ci['support']}u * {lanes}u + {lane}u], 0.0f);",
            f"  vehicle_feed[{vi[f'linkage_motion_ratio_{wheel}']}u] = "
            f"finite_or(contact_feed[{ci['linkage_motion_ratio']}u * {lanes}u + {lane}u], 1.0f);",
            f"  let normal_load_{lane} = {normal_load};",
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
  vehicle_feed[{vi['contact_wrench_force_x']}u] = finite_or(reduced_wrench[0u], 0.0f);
  vehicle_feed[{vi['contact_wrench_force_y']}u] = finite_or(reduced_wrench[1u], 0.0f);
  vehicle_feed[{vi['contact_wrench_force_z']}u] = finite_or(reduced_wrench[2u], 0.0f);
  vehicle_feed[{vi['contact_wrench_torque_x']}u] = finite_or(reduced_wrench[3u], 0.0f);
  vehicle_feed[{vi['contact_wrench_torque_y']}u] = finite_or(reduced_wrench[4u], 0.0f);
  vehicle_feed[{vi['contact_wrench_torque_z']}u] = finite_or(reduced_wrench[5u], 0.0f);
  vehicle_feed[{vi['yaw_cos']}u] = cos(vehicle_feed[{vi['yaw']}u]);
  vehicle_feed[{vi['yaw_sin']}u] = sin(vehicle_feed[{vi['yaw']}u]);
  vehicle_feed[{vi['throttle']}u] = controls[0u];
  vehicle_feed[{vi['brake']}u] = controls[2u];
  vehicle_feed[{vi['forward_gear_ratio']}u] = controls[3u];
  vehicle_feed[{vi['reverse_gear_ratio']}u] = controls[4u];
  vehicle_feed[{vi['transfer_case_ratio']}u] = controls[5u];
  vehicle_feed[{vi['differential_lock']}u] = controls[6u];
  vehicle_feed[{vi['drive_direction']}u] = controls[7u];
{chr(10).join(wheel_lines)}
}}'''
    commits: list[str] = []
    direct = {
        **{f"position_{axis}_next": f"position_{axis}" for axis in "xyz"},
        **{f"velocity_{axis}_next": f"velocity_{axis}" for axis in "xyz"},
        "roll_next": "roll", "pitch_next": "pitch", "yaw_next": "yaw",
        "roll_velocity_next": "roll_velocity", "pitch_velocity_next": "pitch_velocity",
        "yaw_velocity_next": "yaw_velocity",
        **{f"wheel_omega_{wheel}_next": f"wheel_omega_{wheel}" for wheel in WHEEL_NAMES},
        **{f"slip_longitudinal_{wheel}_next": f"previous_slip_longitudinal_{wheel}"
           for wheel in WHEEL_NAMES},
        **{f"slip_sensor_velocity_{wheel}_next": f"slip_sensor_velocity_{wheel}"
           for wheel in WHEEL_NAMES},
        **{f"friction_utilization_{wheel}_next": f"friction_utilization_{wheel}"
           for wheel in WHEEL_NAMES},
        **{f"friction_utilization_sensor_velocity_{wheel}_next":
           f"friction_utilization_sensor_velocity_{wheel}" for wheel in WHEEL_NAMES},
        **{f"compression_{wheel}_next": f"compression_{wheel}" for wheel in WHEEL_NAMES},
        "engine_angular_speed_next": "engine_angular_speed",
    }
    for output, feed in direct.items():
        commits.append(f"  vehicle_feed[{vi[feed]}u] = vehicle_outputs[{vo[output]}u];")
    commit = f'''@group(0) @binding(0) var<storage, read> vehicle_outputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> vehicle_feed: array<f32>;
@compute @workgroup_size(1, 1, 1)
fn commit_vehicle_graph_state(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x != 0u) {{ return; }}
{chr(10).join(commits)}
}}'''
    return {
        "schema": "abstract-ui-vehicle-gpu-graph-adapters-v0",
        "authority": "compiler-published-packed-abis",
        "control_abi": ["throttle", "steering", "brake", "forward_gear_ratio",
                        "reverse_gear_ratio", "transfer_case_ratio", "differential_lock",
                        "drive_direction"],
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


def vehicle_webgpu_program_model(config: VehicleConfiguration) -> dict[str, Any]:
    program = compile_wheel_contact_webgpu()
    runtime_contact = compile_wheel_contact_abstract_tensor(packed_outputs=False)
    tensor_contact = compile_wheel_contact_abstract_tensor(packed_outputs=True)
    tensor_program = tensor_contact.artifacts[0]
    reduction = compile_vehicle_wrench_reduction_webgpu()
    reduction_program = reduction.artifacts[0]
    vehicle_compilation = compile_symbolic_vehicle_physics_gpu_ssa()
    vehicle_program = compile_symbolic_vehicle_physics_webgpu()
    contact_inputs = tuple(tensor_contact.argument_names)
    vehicle_inputs = tuple(vehicle_compilation.function.metadata["argument_names"])
    adapters = _vehicle_gpu_graph_adapters(
        contact_inputs, tuple(tensor_contact.output_names),
        vehicle_inputs, tuple(VEHICLE_STATE_OUTPUTS),
    )
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
            "state_residency": "gpu-persistent-with-passive-presentation-snapshots",
            "kernel": {
                "source": vehicle_program.source,
                "entrypoint": "main",
                "workgroup_size": list(vehicle_program.launch_plan.workgroup_size),
                "dispatch": list(vehicle_program.launch_plan.groups),
                "invocations": 1,
                "io": vehicle_program.api.to_mapping()["metadata"]["io_layout"],
                "output_span": vehicle_program.api.to_mapping()["metadata"]["output_span"],
            },
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
    mass_properties = config.mass_properties()
    component_masses = {item["identity"]: item["mass_kg"]
                        for item in mass_properties["components"]}
    half_length = float(chassis["half_length"])
    half_width = float(chassis["half_width"])
    wheelbase = float(wheels["wheelbase_half_length"])
    track = float(wheels["track_half_width"])
    hub_face_offset = float(wheels["hub_face_offset"])
    wheel_radius = float(source["tires"]["radius"])
    frame_y = float(chassis["height"]) * .72
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def node(identity: str, position: list[float], kind: str, **attributes: Any) -> None:
        nodes.append({"identity": identity, "kind": kind, "reference_position": position,
                      "wrench": {"force": [0.0, 0.0, 0.0], "moment": [0.0, 0.0, 0.0]},
                      **attributes})

    def edge(identity: str, a: str, b: str, constraint: str, *, radius: float = .012,
             palette: str = "rollbar-silver", **attributes: Any) -> None:
        pa = next(item["reference_position"] for item in nodes if item["identity"] == a)
        pb = next(item["reference_position"] for item in nodes if item["identity"] == b)
        rest_length = math.sqrt(sum((pa[axis] - pb[axis]) ** 2 for axis in range(3)))
        edges.append({"identity": identity, "a": a, "b": b, "constraint": constraint,
                      "rest_length": rest_length, "radius": radius, "palette_role": palette,
                      **attributes})

    # Rigid frame: four load nodes, perimeter rails, and both triangulating diagonals.
    for longitudinal, x in (("front", wheelbase), ("rear", -wheelbase)):
        for lateral, side in (("left", -1.0), ("right", 1.0)):
            node(f"frame.{longitudinal}_{lateral}", [x, frame_y, side * half_width * .78],
                 "chassis-load-node", fixed_to="chassis")
    frame_pairs = (
        ("front_left", "front_right"), ("front_right", "rear_right"),
        ("rear_right", "rear_left"), ("rear_left", "front_left"),
        ("front_left", "rear_right"), ("front_right", "rear_left"),
    )
    for index, (a, b) in enumerate(frame_pairs):
        edge(f"frame.member.{index}", f"frame.{a}", f"frame.{b}", "rigid-distance",
             radius=.018, load_path="chassis-wrench-reduction")
    cage_x, cage_z = half_length * .68, half_width * .76
    cage_floor, cage_roof = .08, max(.42, float(chassis["height"]) + .24)
    for longitudinal, x in (("front", cage_x), ("rear", -cage_x)):
        for lateral, side in (("left", -1.0), ("right", 1.0)):
            node(f"cage.{longitudinal}_{lateral}.lower", [x, cage_floor, side * cage_z],
                 "roll-cage-node", fixed_to="chassis")
            node(f"cage.{longitudinal}_{lateral}.upper", [x, cage_roof, side * cage_z],
                 "roll-cage-node", fixed_to="chassis")
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

    # Close-set, fully round low lamps behind a black brush guard.  Their
    # chassis-local positions are also the authority for the moving light cones.
    lamp_x, lamp_y, lamp_spacing, lamp_radius = half_length + .045, .095, .14, .058
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
        x = wheelbase if longitudinal == "front" else -wheelbase
        side = -1.0 if lateral == "left" else 1.0
        nominal_motion_ratio = .78
        static_compression = min(float(suspension["travel"]), float(source["mass"])
                                 * abs(float(source["world"]["gravity"]))
                                 * float(source["mass_distribution"][corner])
                                 / (float(suspension["stiffness"]) * nominal_motion_ratio ** 2))
        hub_y = (-float(chassis["clearance"]) - float(suspension["rest_length"])
                 + static_compression + wheel_radius)
        hub_z = side * track
        knuckle_z = side * (track - hub_face_offset)
        prefix = f"suspension.{corner}"
        # A-arm pickup pairs define two chassis-fixed revolute axes.  Their paired
        # links converge at spherical ball joints on a rigid upright.
        points = {
            "upper_pickup_forward": [x + .09, frame_y + .055, side * half_width * .43],
            "upper_pickup_rear": [x - .09, frame_y + .055, side * half_width * .43],
            "lower_pickup_forward": [x + .11, frame_y - .205, side * half_width * .38],
            "lower_pickup_rear": [x - .11, frame_y - .205, side * half_width * .38],
            "upper_ball_joint": [x, hub_y + .085, knuckle_z],
            "lower_ball_joint": [x, hub_y - .085, knuckle_z],
            "knuckle": [x, hub_y, knuckle_z],
            "hub": [x, hub_y, hub_z],
            "wheel_rim": [x, hub_y, hub_z],
            "tire_carcass": [x, hub_y, hub_z],
            "brake_rotor": [x, hub_y, hub_z - side * .012],
            "brake_caliper": [x - .025, hub_y + .025, knuckle_z],
            "coilover_chassis": [x - .025, frame_y + .13, side * half_width * .30],
            "contact_patch": [x, hub_y - wheel_radius, hub_z],
        }
        if longitudinal == "front":
            points["steering_rack"] = [x - .08, hub_y + .025, side * half_width * .24]
            points["steering_arm"] = [x - .045, hub_y + .025, knuckle_z - side * .025]
        for name, position in points.items():
            fixed = name.endswith("pickup_forward") or name.endswith("pickup_rear") \
                or name in {"coilover_chassis", "steering_rack"}
            node(f"{prefix}.{name}", position,
                 "contact-patch" if name == "contact_patch" else
                 "upright-knuckle" if name == "knuckle" else
                 "wheel-hub" if name == "hub" else
                 "wheel-rim" if name == "wheel_rim" else
                 "pneumatic-tire-carcass" if name == "tire_carcass" else
                 "brake-rotor" if name == "brake_rotor" else
                 "brake-caliper" if name == "brake_caliper" else
                 "spherical-joint" if name.endswith("ball_joint") else
                 "chassis-pickup", fixed_to="chassis" if fixed else None,
                 generalized_coordinate=None if fixed else f"compression_{corner}",
                 mass_kg=(component_masses[f"wheel_{corner}"] if name == "wheel_rim" else
                          component_masses[f"tire_{corner}"] if name == "tire_carcass" else 0.0),
                 mass_in_total=name in {"wheel_rim", "tire_carcass"})
        for level in ("upper", "lower"):
            for direction in ("forward", "rear"):
                edge(f"{prefix}.{level}_arm_{direction}",
                     f"{prefix}.{level}_pickup_{direction}", f"{prefix}.{level}_ball_joint",
                     "rigid-distance", palette="suspension-yellow", joint_a="revolute-axis-x",
                     joint_b="spherical", force_path="contact-to-chassis")
            for direction in ("forward", "rear"):
                edge(f"{prefix}.{level}_pickup_mount_{direction}",
                     f"{prefix}.{level}_pickup_{direction}", f"frame.{longitudinal}_{lateral}",
                     "rigid-distance", radius=.013, palette="rollbar-silver",
                     force_path="wishbone-pickup-to-lower-frame")
        edge(f"{prefix}.upright", f"{prefix}.upper_ball_joint", f"{prefix}.lower_ball_joint",
             "rigid-distance", radius=.014, joint_a="spherical", joint_b="spherical",
             force_path="contact-to-control-arms")
        edge(f"{prefix}.hub_carrier", f"{prefix}.upper_ball_joint", f"{prefix}.knuckle",
             "rigid-offset", radius=.011, force_path="hub-to-upright")
        edge(f"{prefix}.wheel_bearing", f"{prefix}.knuckle", f"{prefix}.hub",
             "rotational-bearing", radius=.012, palette="rollbar-silver",
             polar_inertia_kg_m2=config.wheel_rotational_inertia(),
             gyroscopic_reaction="signed-omega-cross-angular-momentum-to-knuckle-and-chassis",
             force_path="upright-through-bearing-to-wheel")
        edge(f"{prefix}.coilover", f"{prefix}.coilover_chassis", f"{prefix}.lower_ball_joint",
             "spring-damper", radius=.017, palette="suspension-yellow",
             stiffness=float(suspension["stiffness"]),
             compression_damping=float(suspension["pneumatic_compression_damping"]),
             rebound_damping=float(suspension["pneumatic_rebound_damping"]),
             force_path="lower-arm-to-chassis")
        edge(f"{prefix}.coilover_tower", f"{prefix}.coilover_chassis",
             f"cage.{longitudinal}_{lateral}.lower", "rigid-distance", radius=.015,
             palette="rollbar-silver", force_path="coilover-top-to-roll-cage-and-frame")
        edge(f"{prefix}.hub_to_wheel", f"{prefix}.hub", f"{prefix}.wheel_rim",
             "rigid-wheel-mount", radius=.01, palette="rollbar-silver",
             polar_inertia_kg_m2=config.wheel_rotational_inertia(), force_path="halfshaft-to-wheel")
        edge(f"{prefix}.rotor_mount", f"{prefix}.wheel_rim", f"{prefix}.brake_rotor",
             "rigid-rotor-mount", radius=.009, palette="drivetrain-black",
             force_path="wheel-to-brake-rotor")
        edge(f"{prefix}.caliper_mount", f"{prefix}.knuckle", f"{prefix}.brake_caliper",
             "rigid-caliper-mount", radius=.009, palette="suspension-yellow",
             force_path="brake-reaction-to-upright")
        edge(f"{prefix}.service_brake", f"{prefix}.brake_rotor", f"{prefix}.brake_caliper",
             "friction-brake-torque-couple", radius=.008, palette="suspension-yellow",
             torque_channel=f"brake_torque * brake * brake_scale_{corner}",
             reaction_path="caliper-to-knuckle-to-wishbones-to-chassis")
        edge(f"{prefix}.wheel_to_tire", f"{prefix}.wheel_rim", f"{prefix}.tire_carcass",
             "bead-seat", radius=.01, palette="drivetrain-black", force_path="wheel-to-tire-carcass")
        edge(f"{prefix}.tire", f"{prefix}.tire_carcass", f"{prefix}.contact_patch",
             "pneumatic-contact", radius=.01, palette="drivetrain-black",
             force_path="terrain-wrench-entry")
        if longitudinal == "front":
            edge(f"{prefix}.steering_arm", f"{prefix}.hub", f"{prefix}.steering_arm",
                 "rigid-offset", radius=.009, palette="suspension-yellow",
                 force_path="steering-moment-to-upright")
            edge(f"{prefix}.tie_rod", f"{prefix}.steering_rack", f"{prefix}.steering_arm",
                 "steering-link", radius=.009, palette="suspension-yellow",
                 generalized_coordinate="steering_angle",
                 force_path="rack-to-upright-steering-moment")

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
    edge("steering.column.upper", "steering.wheel.center", "steering.column.lower",
         "steering-torque-shaft", radius=.009, palette="drivetrain-black",
         generalized_coordinate="steering_angle")
    edge("steering.column.lower", "steering.column.lower", "steering.pinion",
         "universal-joint-steering-shaft", radius=.008, palette="drivetrain-black",
         generalized_coordinate="steering_angle")
    edge("steering.rack_and_pinion", "steering.pinion", "steering.rack.center",
         "rack-and-pinion-angle-to-translation", radius=.009, palette="drivetrain-black",
         generalized_coordinate="steering_angle")
    edge("steering.rack.left", "steering.rack.center",
         "suspension.front_left.steering_rack", "rack-translation", radius=.008,
         palette="drivetrain-black", generalized_coordinate="steering_angle")
    edge("steering.rack.right", "steering.rack.center",
         "suspension.front_right.steering_rack", "rack-translation", radius=.008,
         palette="drivetrain-black", generalized_coordinate="steering_angle")

    engine_position = [float(value) for value in powertrain["engine_position"]]
    power_nodes = {
        "powertrain.engine": engine_position,
        "powertrain.clutch": [engine_position[0] + .15, engine_position[1], 0.0],
        "powertrain.transmission": [engine_position[0] + .29, engine_position[1] - .015, 0.0],
        "powertrain.transfer_case": [engine_position[0] + .39, engine_position[1] - .035, 0.0],
        "powertrain.center_shaft": [-.12, .06, 0.0],
        "powertrain.front_differential": [wheelbase, .065, 0.0],
        "powertrain.rear_differential": [-wheelbase, .065, 0.0],
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
        node(identity, position, "powertrain-mount" if identity.startswith("mount.") else "rotating-mass",
             fixed_to="chassis" if identity.startswith("mount.") else None,
             mass_kg=component_masses.get(graph_mass_names.get(identity, ""), 0.0),
             mass_in_total=identity in graph_mass_names)
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
    for axle in ("front", "rear"):
        for lateral in ("left", "right"):
            corner = f"{axle}_{lateral}"
            edge(f"drivetrain.{corner}_halfshaft", f"powertrain.{axle}_differential",
                 f"suspension.{corner}.hub", "constant-velocity-torque-shaft", radius=.009,
                 palette="drivetrain-black", torque_channel=f"wheel_torque_{corner}")
    for component, mounts in (("engine", ("engine_left", "engine_right")),
                              ("transmission", ("transmission_left", "transmission_right")),
                              ("transfer_case", ("transfer_case_left", "transfer_case_right"))):
        for mount in mounts:
            edge(f"mount.{component}.{mount}", f"powertrain.{component}", f"mount.{mount}",
                 "six-axis-compliant-mount", radius=.012, palette="drivetrain-black",
                 transfer="force-and-moment-to-chassis")
    corner_loads = {}
    gravity = abs(float(source["world"]["gravity"]))
    for corner in WHEEL_NAMES:
        coilover = next(item for item in edges if item["identity"] == f"suspension.{corner}.coilover")
        a = next(item["reference_position"] for item in nodes if item["identity"] == coilover["a"])
        b = next(item["reference_position"] for item in nodes if item["identity"] == coilover["b"])
        motion_ratio = abs(a[1] - b[1]) / max(1e-9, float(coilover["rest_length"]))
        mass_kg = float(source["mass"]) * float(source["mass_distribution"][corner])
        static_load = mass_kg * gravity
        corner_loads[corner] = {
            "design_supported_mass_kg": mass_kg,
            "design_static_load_n": static_load,
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
    return {
        "schema": "abstract-ui-mechanical-wrench-graph-v1",
        "authority": "json-parameters-expanded-by-python-compiler",
        "coordinate_system": "chassis-local-x-forward-y-up-z-right",
        "state_law": "node-force-and-node-moment-reduced-through-edge-constraints",
        "nodes": nodes, "edges": edges, "load_audit": load_audit,
        "constraint_reduction": {
            "suspension": "double-wishbone-four-bar-plus-coilover-motion-ratio",
            "contact": "tire-patch-and-cage-node/member-midpoint-terrain-wrenches-to-chassis",
            "powertrain": "shaft-torque-and-six-axis-mount-reactions",
            "chassis": "sum-node-force-and-position-cross-force-plus-node-moment",
        },
    }


def vehicle_slot_model(root: str, actor: str) -> dict[str, Any]:
    config = load_default_car_configuration()
    webgpu = vehicle_webgpu_program_model(config)
    chassis = config.source["chassis"]
    wheels = config.source["wheels"]
    node_positions = {
        "front_left": [float(wheels["wheelbase_half_length"]), -float(chassis["clearance"]),
                       -float(wheels["track_half_width"])],
        "front_right": [float(wheels["wheelbase_half_length"]), -float(chassis["clearance"]),
                        float(wheels["track_half_width"])],
        "rear_left": [-float(wheels["wheelbase_half_length"]), -float(chassis["clearance"]),
                      -float(wheels["track_half_width"])],
        "rear_right": [-float(wheels["wheelbase_half_length"]), -float(chassis["clearance"]),
                       float(wheels["track_half_width"])],
    }
    member_pairs = (
        ("front_left", "front_right"), ("front_right", "rear_right"),
        ("rear_right", "rear_left"), ("rear_left", "front_left"),
        ("front_left", "rear_right"), ("front_right", "rear_left"),
    )
    chassis_structure = {
        "schema": "abstract-ui-stick-ball-chassis-v0",
        "model": "rigid-distance-members-with-compliant-suspension-at-nodes",
        "material": {"name": "steel", "youngs_modulus_pa": 200_000_000_000.0,
                     "density_kg_m3": 7850.0, "yield_strength_pa": 250_000_000.0,
                     "solver_interpretation": "rigid-limit-not-explicit-high-frequency-spring"},
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
            {"identity": "transfer_case", "kind": "two-range-transfer-case",
             "mass_kg": config.source["powertrain"]["transfer_case_mass_kg"],
             "high_range_ratio": 1.0,
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
    return {
        "schema": ABSTRACT_UI_VEHICLE_VERSION,
        "identity": f"{actor}/vehicle-slot",
        "owner": actor,
        "active": None,
        "initial_state": {
            "mounted_vehicle": f"{root}/vehicles/springtail",
            "placement": "at-player-spawn",
            "presentation": "full-viewport-driving",
            "browser_fullscreen": "request-on-first-user-gesture",
            "dismount_enabled": True,
        },
        "allowed_kinds": ["car"],
        "selection_operation": "mount-vehicle-slot",
        "release_operation": "dismount-vehicle-slot",
        "vehicles": [{
            "identity": f"{root}/vehicles/springtail", "archetype": config.identity,
            "name": config.name, "kind": "car", "configuration": config.to_data(),
            "configuration_defaults": config.parameter_defaults(),
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
                            "projection": "bounded-post-step-tunneling-rejection-only",
                        },
                        "transmission_policy": {
                            "authority": "lockstep-worker-state",
                            "default": "automatic-second-gear-launch",
                            "crawler_entry": "torque-reserve-insufficient-in-second",
                            "manual_controls": ["automatic", "gear-down", "gear-up"],
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
    "compile_wheel_contact_ssa", "compile_wheel_contact_wasm", "compile_wheel_contact_abstract_tensor",
    "compile_wheel_contact_webgpu", "compile_vehicle_wrench_reduction_webgpu",
    "compile_sympy_matrix_to_abstract_tensor_backend", "extra_precision_closure",
    "compile_symbolic_vehicle_physics_wasm", "load_default_car_configuration",
    "symbolic_vehicle_equations", "symbolic_vehicle_physics_wasm_plugin",
    "symbolic_wheel_contact_equations", "symbolic_wheel_contact_wasm_plugin",
    "compile_symbolic_vehicle_physics_gpu_ssa", "compile_symbolic_vehicle_physics_webgpu",
    "vehicle_webgpu_program_model",
    "vehicle_configuration_from_mapping", "vehicle_slot_model",
]
