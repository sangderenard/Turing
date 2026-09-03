"""Reusable mechanical boundary contracts for validators and game artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class BearingRuleSet:
    """Serializable aspirations for one bearing force-transfer transformer."""

    bore_m: float
    outer_diameter_m: float
    width_m: float
    radial_clearance_m: float
    axial_float_m: float
    contact_angle_deg: float
    radial_stiffness_n_m: float
    axial_stiffness_n_m: float
    tilt_stiffness_nm_rad: float
    radial_damping_n_s_m: float
    axial_damping_n_s_m: float
    tilt_damping_nm_s_rad: float
    rolling_friction_coefficient: float
    seal_drag_nm: float
    viscous_drag_nm_s_rad: float
    stribeck_speed_rad_s: float
    radial_static_limit_n: float
    radial_dynamic_limit_n: float
    axial_limit_n: float
    speed_limit_rad_s: float
    temperature_limit_k: float
    wear_limit: float = 1.0

    def as_mapping(self) -> dict[str, float]:
        values = {name: float(value) for name, value in asdict(self).items()}
        if not (0.0 < values["bore_m"] < values["outer_diameter_m"]):
            raise ValueError("bearing bore must be positive and smaller than its OD")
        if values["width_m"] <= 0.0:
            raise ValueError("bearing width must be positive")
        for name in ("radial_static_limit_n", "radial_dynamic_limit_n",
                     "axial_limit_n", "speed_limit_rad_s",
                     "temperature_limit_k"):
            if values[name] <= 0.0:
                raise ValueError(f"bearing {name} must be positive")
        return values


def generic_rotational_torque_port(
    identity: str,
    owner: str,
    *,
    axis: tuple[float, float, float],
    rated_torque_nm: float,
    rated_speed_rad_s: float,
    flange: dict[str, Any],
    role: str = "input",
) -> dict[str, Any]:
    """Describe a source-neutral rotating wrench boundary.

    A driveshaft, hand crank, brake, dynamometer, motor, or another compiled
    mechanical graph may connect here.  The port never invents a speed source:
    applied torque enters the owner's declared inertia and returns the exact
    equal/opposite reaction wrench.
    """

    if role not in {"input", "output", "bidirectional"}:
        raise ValueError("rotational torque port role must be input, output, or bidirectional")
    return {
        "schema": "rotating-six-axis-drivetrain-wrench-port-v1",
        "identity": identity,
        "owner": owner,
        "kind": f"generic-rotational-torque-{role}",
        "role": role,
        "axis": list(axis),
        "state": ["shaft_angle_rad", "shaft_angular_velocity_rad_s"],
        "inputs": ["applied_force_xyz_n", "applied_moment_xyz_nm",
                   "connected_inertia_kg_m2"],
        "outputs": ["reaction_force_xyz_n", "reaction_moment_xyz_nm",
                    "shaft_angle_rad", "shaft_angular_velocity_rad_s"],
        "rated_torque_nm": float(rated_torque_nm),
        "rated_speed_rad_s": float(rated_speed_rad_s),
        "flange": dict(flange),
        "law": (
            "port-torque-enters-declared-rotor-inertia-with-signed-flow-direction; "
            "angular-velocity-is-observed-state-never-a-command; reaction-wrench-"
            "is-equal-and-opposite"
        ),
        "fixture_policy": (
            "validator-dynamometer-is-temporary-custody-and-never-becomes-part-of-owner"
        ),
    }


def bearing_interface_node(
    identity: str,
    *,
    structural_owner: str,
    rotating_owner: str,
    axis: tuple[float, float, float],
    bearing_kind: str,
    reference_position: tuple[float, float, float],
    rule_set: BearingRuleSet,
    preloaded: bool = False,
    assembly_stage: str | None = None,
) -> dict[str, Any]:
    """Create a bearing node shared by structural and drivetrain graphs."""

    rules = rule_set.as_mapping()
    node = {
        "schema": "mechanical-bearing-interface-node-v1",
        "identity": identity,
        "kind": bearing_kind,
        "mass_kg": 0.0,
        "reference_position": list(reference_position),
        "interface_roles": ["structural-reaction", "drivetrain-rotation"],
        "structural_owner": structural_owner,
        "rotating_owner": rotating_owner,
        "axis": list(axis),
        "race_ports": {
            "stationary": {
                "owner": structural_owner,
                "accepts_edge_classes": [
                    "load-bearing-structure", "bearing-seat", "fixture"],
            },
            "rotating": {
                "owner": rotating_owner,
                "accepts_edge_classes": [
                    "drivetrain", "shaft", "hub", "gear-carrier"],
            },
        },
        "constraint": {
            "free_coordinate": "rotation-about-axis",
            "degrees_of_freedom": {
                "rotation-about-axis": {
                    "kind": "free-periodic", "range_rad": None},
                "axial-translation": {
                    "kind": "bounded-compliant",
                    "range_m": [-rules["axial_float_m"],
                                rules["axial_float_m"]],
                },
                "radial-translation": {
                    "kind": "clearance-then-compliant",
                    "range_m": [0.0, rules["radial_clearance_m"]],
                },
                "tilt": {"kind": "compliant-constrained"},
            },
            "constrained_coordinates": [
                "translation-x", "translation-y", "translation-z",
                "rotation-about-transverse-a", "rotation-about-transverse-b",
            ],
            "preloaded": bool(preloaded),
        },
        "geometry": {
            name: rules[name] for name in (
                "bore_m", "outer_diameter_m", "width_m",
                "contact_angle_deg")
        },
        "force_transfer_transformer": {
            "schema": "bearing-wrench-transformer-v1",
            "input": "relative-six-axis-race-motion-and-temperature",
            "output": "equal-and-opposite-six-axis-race-wrenches",
            "freedom": "rotation-about-axis-with-bounded-bearing-drag",
            "reaction_law": {
                "radial": "clearance-deadband-then-stiffness-plus-damping",
                "axial": "float-deadband-then-stiffness-plus-damping",
                "tilt": "angular-stiffness-plus-damping",
            },
            "friction_law": (
                "stribeck-blend-of-seal-drag-rolling-friction-and-viscous-drag; "
                "friction-work-enters-bearing-temperature"
            ),
            "failure_law": (
                "overload-and-overtemperature-accumulate-wear; clearance-and-"
                "drag-rise-with-wear; seizure-locks-free-rotation; fracture-"
                "disconnects-race-reaction-constraint"
            ),
            "parameters": rules,
        },
        "state": [
            "relative_angle_rad", "relative_angular_velocity_rad_s",
            "radial_reaction_n", "axial_reaction_n", "drag_torque_nm",
            "temperature_k", "lubricant_state", "clearance_m", "wear",
            "break_state",
        ],
        "failure_rule": (
            "bearing-break-disconnects-race-constraint-while-preserving-"
            "both-former-owners-as-separate-graph-nodes"
        ),
    }
    if assembly_stage is not None:
        node["assembly_stage"] = assembly_stage
    node["presentation"] = {
        "source": "abstract-ui-part-geometry",
        "primitive": "bearing-races",
    }
    return node


def rotating_hub_node(
    identity: str,
    *,
    reference_position: tuple[float, float, float],
    axis: tuple[float, float, float],
    mass_kg: float,
    flange_radius_m: float,
    barrel_radius_m: float,
    width_m: float,
    assembly_stage: str,
) -> dict[str, Any]:
    """Create an axle hub, distinct from its bearing and attached wheels."""

    return {
        "schema": "abstract-ui-rotating-hub-node-v1",
        "identity": identity,
        "kind": "full-floating-wheel-mounting-hub",
        "abstract_ui_role": "axle-hub",
        "mass_kg": float(mass_kg),
        "reference_position": list(reference_position),
        "axis": list(axis),
        "assembly_stage": assembly_stage,
        "geometry": {
            "primitive": "wheel-mounting-hub",
            "flange_radius_m": float(flange_radius_m),
            "barrel_radius_m": float(barrel_radius_m),
            "width_m": float(width_m),
            "axis": list(axis),
        },
        "presentation": {
            "source": "abstract-ui-part-geometry",
            "primitive": "wheel-mounting-hub",
        },
        "ports": {
            "bearing_rotating_race": f"{identity}:bearing-rotating-race",
            "wheel_mounting_face": f"{identity}:wheel-mounting-face",
            "shaft_spline": f"{identity}:shaft-spline",
        },
    }


def bearing_race_edge(
    identity: str,
    *,
    bearing: Mapping[str, Any],
    race: str,
    connected_node: str,
    edge_class: str,
    kind: str,
) -> dict[str, Any]:
    """Connect either graph domain to its physical race on a bearing node.

    A bearing is deliberately one node shared by the structural and drivetrain
    graphs.  Race selection says which side of its relative-rotation constraint
    owns an incident edge; it does not split the bearing into parallel graphs.
    """

    if bearing.get("schema") != "mechanical-bearing-interface-node-v1":
        raise ValueError("bearing race edges require a bearing interface node")
    ports = bearing["race_ports"]
    if race not in ports:
        raise ValueError(f"unknown bearing race {race!r}")
    accepted = ports[race]["accepts_edge_classes"]
    if edge_class not in accepted:
        raise ValueError(
            f"{edge_class!r} cannot connect to the {race} bearing race")
    bearing_identity = str(bearing["identity"])
    nodes = ([connected_node, bearing_identity] if race == "stationary"
             else [bearing_identity, connected_node])
    return {
        "identity": identity,
        "kind": kind,
        "edge_class": edge_class,
        "bearing_race": race,
        "nodes": nodes,
    }


__all__ = ["BearingRuleSet", "bearing_interface_node", "bearing_race_edge",
           "generic_rotational_torque_port", "rotating_hub_node"]
