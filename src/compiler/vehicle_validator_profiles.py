"""Data-driven assembly profiles consumed by the existing vehicle validator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .abstract_ui_dually_axle import (
    DUALLY_VALIDATOR_STAGES, DUALLY_WHEELS, roadside_dually_axle_assembly,
)
from .vehicle_native_assembly import (
    WheelFixturePlan, assembled_point_mass_properties,
    infer_structural_grasp_frame, negotiate_wheel_fixture,
)
from .vehicle_native_graph_program import (
    VehicleGraphConstants, vehicle_graph_constants_from_model,
)


@dataclass(frozen=True, slots=True)
class ValidatorAssemblyProfile:
    """One loaded artifact and its negotiated use of the common validator."""

    identity: str
    model: Mapping[str, Any]
    wheel_names: tuple[str, ...]
    stages: tuple[str, ...]
    fixture_plan: WheelFixturePlan
    graph_constants: VehicleGraphConstants
    wheel_attachments: tuple[tuple[float, float, float], ...]
    structural_support_positions: tuple[tuple[float, float, float], ...]
    tire_dimensions: tuple[float, float, float, float, float, float]
    tire_pneumatic_mode: str
    tire_material_profile: str
    rated_pressure_pa: float
    mass_properties: Mapping[str, Any]


def dually_validator_profile() -> ValidatorAssemblyProfile:
    """Load the four-wheel solid axle as a real validator input artifact."""

    assembly = roadside_dually_axle_assembly(
        "validator:loaded", center_x=0.0, center_z=0.0)
    model = assembly.model
    fixture = negotiate_wheel_fixture(model)
    grasp = infer_structural_grasp_frame(model)
    wheels = tuple(model["wheels"])
    if tuple(identity.rsplit("/", 1)[-1]
             for identity in fixture.wheel_identities) != DUALLY_WHEELS:
        raise ValueError("dually fixture and authored wheel axes disagree")
    components = tuple({
        "identity": str(node["identity"]),
        "mass_kg": float(node.get("mass_kg", 0.0)),
        "local_position": tuple(float(value) for value in node.get(
            "reference_position", (0.0, 0.0, 0.0))),
    } for node in model["mechanical_graph"]["nodes"]
                       if float(node.get("mass_kg", 0.0)) > 0.0)
    first = wheels[0]
    section_radius = 0.5 * float(first["section_width_m"])
    return ValidatorAssemblyProfile(
        identity="commercial-dually-axle",
        model=model,
        wheel_names=DUALLY_WHEELS,
        stages=DUALLY_VALIDATOR_STAGES,
        fixture_plan=fixture,
        graph_constants=vehicle_graph_constants_from_model(
            model, len(fixture.structural_support_identities)),
        wheel_attachments=tuple(
            pillar.eventual_installation_position
            for pillar in fixture.pillars),
        structural_support_positions=tuple(
            tuple(float(value) for value in corner["position"])
            for corner in grasp.support_corners),
        tire_dimensions=(
            float(first["radius_m"]), section_radius,
            float(first["section_width_m"]), float(first["tire_mass_kg"]),
            float(first["rated_pressure_pa"]), float(first["rim_radius_m"])),
        tire_pneumatic_mode=str(first["pneumatic_mode"]),
        tire_material_profile=str(first["material_profile"]),
        rated_pressure_pa=float(first["rated_pressure_pa"]),
        mass_properties=assembled_point_mass_properties(components),
    )


__all__ = ["ValidatorAssemblyProfile", "dually_validator_profile"]
