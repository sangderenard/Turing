"""Composable subject class for the living mechanical-creature map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


MECHANICAL_CREATURE_VERSION = "mechanical-creature-v0"


@dataclass(frozen=True, slots=True)
class Structure:
    """Force-bearing nodes and constraints that own pose and damage geometry."""

    graph_identity: str
    material_profile: str


@dataclass(frozen=True, slots=True)
class Actuator:
    """A bounded device that changes force or preferred mechanical geometry."""

    identity: str
    target: str
    rate_limit: float


@dataclass(frozen=True, slots=True)
class Stabilizer:
    """A passive or active device that opposes relative motion without owning pose."""

    identity: str
    coupled_members: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContactSurfaces:
    """Shared local surface queries and equal/opposite contact-wrench laws."""

    service_identity: str
    evaluation_scope: str = "entity-local-candidates-only"


@dataclass(frozen=True, slots=True)
class ParametricEngine:
    """A selectable immutable curve pack feeding a resident compiled power stage."""

    preset_identity: str
    curve_reference: str


@dataclass(frozen=True, slots=True)
class TickEnvelope:
    """One deterministic time assignment shared by world-owned objects."""

    tick: int
    dt: float
    subdt: float
    substeps: int


@dataclass(frozen=True, slots=True)
class VehicleInstance:
    """Stable reference to a car owned by the world object graph."""

    identity: str


@dataclass(frozen=True, slots=True)
class ValidatorRigObject:
    """Stable reference to the world's validator/assembly rig."""

    identity: str


class MechanicalCreatureWorld:
    """The class whose members become the explorable Living Data Map.

    The host may call :meth:`tick` from the website cycle or a standalone
    native driver.  Owned objects do no work before receiving that assignment.
    """

    initial_vehicle: VehicleInstance
    validator_rig: ValidatorRigObject

    def tick(self, tick: int, dt: float, subdt: float, substeps: int) -> None:
        """Assign one envelope to the rig, validator, and car in stable order."""


class MechanicalCreature:
    """Root subject projected into the map; the truck is its first instance.

    This class is deliberately a domain object rather than a rendering class.
    ``AbstractUI`` remains the neutral packet envelope produced by the map
    compiler.  The fields below are the meta-objects a player can inspect and
    alter without inventing a car-specific second physics system.
    """

    structure: Structure
    actuators: tuple[Actuator, ...]
    stabilizers: tuple[Stabilizer, ...]
    contact_surfaces: ContactSurfaces
    parametric_engine: ParametricEngine

    def request_preferred_pose(self, ride_height: float, roll: float, pitch: float) -> None:
        """Set a rate-limited second-order target; never assign body pose directly."""

    def reduce_contact_wrenches(self) -> None:
        """Integrate local tire/cage samples into one conserved structure wrench."""

    def dispatch_resident_stages(self) -> None:
        """Run structure, actuator/contact, and engine passes over common buffers."""

    def select_material_profile(self, identity: str) -> None:
        """Swap tube mass, stiffness, yield and shear limits as one physical profile."""


def mechanical_creature_model(
    vehicle: Mapping[str, Any], *, world_identity: str, contact_surface_identity: str,
) -> dict[str, Any]:
    """Describe one vehicle as a composition of general mechanical meta-objects."""

    graph = vehicle["physics"]["mechanical_graph"]
    edges = graph["edges"]
    actuator_members = [
        edge["identity"] for edge in edges
        if edge["constraint"] in {"spring-damper", "steering-link", "rack-translation",
                                  "routed-tension-cable", "table-actuator-linear-link"}
    ]
    stabilizer_members = [
        edge["identity"] for edge in edges
        if edge["constraint"] == "torsion-stabilizer"
    ]
    identity = f"{vehicle['identity']}/mechanical-creature"
    return {
        "schema": MECHANICAL_CREATURE_VERSION,
        "identity": identity,
        "subject_class": "src.compiler.mechanical_creature.MechanicalCreature",
        "active_instance": vehicle["identity"],
        "instances": [vehicle["identity"]],
        "current_scope": "truck-only-first-instance",
        "extension_policy": "new-creatures-compose-the-same-meta-object-roles",
        "meta_objects": {
            "structure": {
                "identity": f"{identity}/structure",
                "authority": "mechanical-wrench-graph",
                "source": graph["schema"],
                "members": [edge["identity"] for edge in edges
                            if edge.get("chassis_profile_member")],
                "material_profiles": [profile["identity"]
                                      for profile in vehicle["chassis_profiles"]],
            },
            "actuators": {
                "identity": f"{identity}/actuators",
                "members": actuator_members,
                "preferred_pose": vehicle["chassis_leveling"],
                "pose_authority": "spring-contact-wrenches-not-direct-transform",
                "control_rate_members": graph.get("control_actuators", []),
            },
            "stabilizers": {
                "identity": f"{identity}/stabilizers",
                "members": stabilizer_members,
                "active_damping": "four-independent-compiled-damper-scales",
            },
            "contact_surfaces": {
                "identity": f"{identity}/contact-surfaces",
                "service": contact_surface_identity,
                "scope": "local-candidates-near-entity-only",
                "reduction": graph["constraint_reduction"]["contact"],
            },
            "parametric_engine": {
                "identity": f"{identity}/parametric-engine",
                "active_preset": vehicle["power_unit_preset"],
                "presets": [preset["identity"] for preset in vehicle["power_unit_presets"]],
                "execution": "eager-fixed-default-then-resident-parametric-stage-swap",
            },
        },
        "sequence_flow": [
            "parametric_engine", "actuators", "stabilizers", "contact_surfaces", "structure",
        ],
        "buffer_policy": "common-resident-locations-no-inter-stage-host-transfer",
        "execution_bands": graph.get("execution_bands", {}),
        "world": world_identity,
    }


__all__ = [
    "MECHANICAL_CREATURE_VERSION", "Actuator", "ContactSurfaces", "MechanicalCreature",
    "MechanicalCreatureWorld", "ParametricEngine", "Stabilizer", "Structure",
    "TickEnvelope", "ValidatorRigObject", "VehicleInstance", "mechanical_creature_model",
]
