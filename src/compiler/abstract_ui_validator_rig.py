"""Portable AbstractUI contract for the in-world vehicle validator rig.

The rig is a world object, not a page, modal, scheduler, or second physics
runtime.  This module describes ownership and custody transitions so document,
native, WebAssembly, and graphics adapters can all realize the same object
graph without inventing identities of their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .abstract_ui_world import WorldObject


VALIDATOR_RIG_SCHEMA = "abstract-ui-vehicle-validator-rig-v0"


@dataclass(frozen=True, slots=True)
class ValidatorRigAssembly:
    """The rig model together with its canonical world-registry objects."""

    model: Mapping[str, Any]
    world_objects: tuple[WorldObject, ...]


def validator_rig_assembly(
    root: str,
    actor: str,
    initial_vehicle: Mapping[str, Any],
    projectile_archetype: Mapping[str, Any],
    *,
    center_x: float,
    center_z: float,
) -> ValidatorRigAssembly:
    """Define one rig and its initially funded backup vehicle build.

    ``initial_vehicle`` remains the one object mounted by the vehicle slot.  A
    separate backup identity is held by the rig until assembly and the existing
    qualification program both complete.
    """

    rig_identity = f"{root}/validator-rig"
    initial_vehicle_identity = str(initial_vehicle["identity"])
    backup_vehicle_identity = f"{root}/vehicles/springtail-rig-build-001"
    projectile_identity = str(projectile_archetype["identity"])
    stage_identity = f"{rig_identity}/construction-stage"
    hopper_identity = f"{rig_identity}/hopper"

    part_specs = (
        ("frame-and-cage", 14),
        ("suspension-and-steering", 10),
        ("wheels-and-tires", 12),
        ("powertrain", 14),
        ("body-and-glazing", 8),
        ("controls-and-wiring", 6),
    )
    parts = []
    for index, (name, material_units) in enumerate(part_specs):
        identity = f"{backup_vehicle_identity}/parts/{name}"
        parts.append({
            "identity": identity,
            "name": name,
            "sequence": index,
            "material_units": material_units,
            "state": "projected" if index == 0 else "awaiting-material",
            "custody": "rig-projection" if index == 0 else "rig-recipe",
            "presentation": {
                "projected": {
                    "program": f"{root}/shader-programs/construction-line",
                    "background_alpha": 0.0,
                    "line_alpha": 0.18,
                    "source": "vehicle-part-line-work",
                },
                "installed": {
                    "program": f"{root}/shader-programs/pluck-phong",
                    "material_source": "vehicle-configuration-material-binding",
                },
            },
            "handoff": (
                "material-ball->hopper-ledger->line-projection->"
                "rig-actuator->vehicle-installed-phong"
            ),
        })

    total_material_units = sum(item[1] for item in part_specs)
    model: dict[str, Any] = {
        "schema": VALIDATOR_RIG_SCHEMA,
        "identity": rig_identity,
        "kind": "validator-assembly-rig",
        "owner": root,
        "operator": actor,
        "world_properties": {
            "initial_vehicle": initial_vehicle_identity,
            "validator_rig": rig_identity,
        },
        "identity_contract": {
            "initial_vehicle": initial_vehicle_identity,
            "room_discovery_reference": initial_vehicle_identity,
            "mounted_vehicle_reference": initial_vehicle_identity,
            "copy_policy": "references-only-never-synthesize-or-copy-the-vehicle",
        },
        "transform": {
            "position": [center_x, 0.0, center_z],
            "yaw_degrees": 0.0,
        },
        "form": {
            "kind": "box-rig-with-side-stage",
            "rig_half_extent": [1.25, 1.15, 1.0],
            "stage_identity": stage_identity,
            "stage_offset": [2.25, 0.06, 0.0],
            "stage_half_extent": [1.7, 0.06, 1.25],
        },
        "hopper": {
            "identity": hopper_identity,
            "kind": "projectile-material-intake",
            "placement": "inconvenient-top-rear-corner",
            "local_center": [-0.82, 1.17, -0.66],
            "radius_m": 0.22,
            "accepted_archetypes": [projectile_identity],
            "contact": "existing-projectile-cycle-overlap-event",
            "consumption": "one-projectile-becomes-one-material-unit",
        },
        "material_supply": {
            "unit": "physics-ball-material-unit",
            "recipe_units": total_material_units,
            "staged_at_start": total_material_units,
            "hopper_buffer_at_start": 1,
            "consumed_at_start": 1,
            "remaining_staged_at_start": total_material_units - 1,
            "sufficiency": "one-complete-backup-vehicle",
            "delivery": "player-feeds-staged-balls-through-hopper",
            "authority": "rig-ledger-not-projectile-renderer",
        },
        "construction": {
            "vehicle": backup_vehicle_identity,
            "archetype": initial_vehicle.get("archetype"),
            "state": "building",
            "current_part": parts[0]["identity"],
            "parts": parts,
            "actuators": {
                "owner": rig_identity,
                "workspace": stage_identity,
                "motion_source": "tick-envelope",
                "scheduler": "none-in-assigned-mode",
            },
        },
        "execution": {
            "mode": "assigned-ticks",
            "available_modes": ["assigned-ticks", "autonomous"],
            "assigned_ticks": {
                "caller": f"python:{__name__.rsplit('.', 1)[0]}.mechanical_creature.MechanicalCreatureWorld/member:method:tick",
                "envelope": ["tick", "dt", "subdt", "substeps"],
                "dispatch_order": ["validator_rig", "validator", "initial_vehicle"],
                "clock_ownership": "world-class-host",
                "inactive_until_called": True,
            },
            "autonomous": {
                "purpose": "standalone-native-qualification-or-tool-host",
                "clock_ownership": "explicit-host-adapter",
                "internal_scheduler": False,
                "same_tick_entrypoint": True,
            },
            "determinism": "all-consumers-receive-the-same-tick-envelope",
        },
        "projection": {
            "origin": f"{rig_identity}/emitter",
            "destination": stage_identity,
            "trail": "faint-scattered-laser-drawing",
            "program": f"{root}/shader-programs/construction-line",
            "background_alpha": 0.0,
            "line_alpha": 0.18,
            "depth": "world-occluded",
        },
        "qualification": {
            "authority": "existing-vehicle-qualification-worker-and-report",
            "input": backup_vehicle_identity,
            "state": "waiting-for-assembly",
            "release_condition": "assembly-complete-and-qualification-passed",
            "failure_condition": "qualification-failed-or-installed-part-broken",
        },
        "release": {
            "operation": "publish-qualified-vehicle-to-existing-vehicle-slot",
            "vehicle": backup_vehicle_identity,
            "identity_policy": "release-the-same-built-object-never-a-render-copy",
        },
        "breakage": {
            "drivability": "unusable-when-required-part-is-broken",
            "recovery": "craft-and-qualify-another-rig-owned-vehicle",
        },
    }

    rig_object = WorldObject(
        rig_identity,
        "validator-assembly-rig",
        root,
        "Vehicle validator and assembly rig",
        model["transform"],
        model["form"],
        material_bindings={"cabinet": "building-wall", "laser": "active"},
        capabilities=("accept-material-projectile", "assemble-vehicle",
                      "qualify-vehicle", "release-vehicle", "tick"),
        physics={"body": "static", "collider": "box-and-hopper-trigger",
                 "cycle": "assigned-world-class-tick-envelope"},
        persistence={"scope": "world", "fields": ["material_supply", "construction"]},
        extensions={"model": rig_identity},
    )
    stage_object = WorldObject(
        stage_identity,
        "validator-construction-stage",
        rig_identity,
        "Projected vehicle construction stage",
        {"local_offset": model["form"]["stage_offset"]},
        {"kind": "thin-stage", "half_extent": model["form"]["stage_half_extent"]},
        material_bindings={"surface": "ground"},
        capabilities=("receive-line-projection", "hold-actuator-work"),
        physics={"body": "static", "collider": "solid-contact-surface"},
    )
    initial_vehicle_object = WorldObject(
        initial_vehicle_identity,
        "car",
        root,
        str(initial_vehicle.get("name", "Initial vehicle")),
        initial_vehicle.get("pose", {}),
        {"kind": "vehicle-instance", "source": "vehicle-slot"},
        capabilities=("discover", "mount", "drive", "damage"),
        physics={"authority": "vehicle-worker", "usable": True},
        persistence={"identity": "stable-world-property"},
        extensions={"vehicle_slot_reference": initial_vehicle_identity},
    )
    backup_vehicle_object = WorldObject(
        backup_vehicle_identity,
        "car-under-construction",
        rig_identity,
        "Rig-built backup vehicle",
        {"stage": stage_identity},
        {"kind": "vehicle-instance", "source": "rig-assembly"},
        capabilities=("receive-installed-part", "qualify", "release"),
        physics={"authority": "vehicle-worker", "usable": False},
        persistence={"identity": "stable-from-first-projection-through-release"},
        extensions={"recipe_archetype": initial_vehicle.get("archetype")},
    )
    return ValidatorRigAssembly(
        model,
        (rig_object, stage_object, initial_vehicle_object, backup_vehicle_object),
    )


def validator_rig_geometry_boxes(assembly: ValidatorRigAssembly) -> tuple[dict[str, Any], ...]:
    """Realize the rig as ordinary child geometry without changing its identity."""

    model = assembly.model
    rig_identity = str(model["identity"])
    stage_identity = str(model["form"]["stage_identity"])
    x, _, z = (float(value) for value in model["transform"]["position"])
    stage_dx, stage_y, stage_dz = (
        float(value) for value in model["form"]["stage_offset"]
    )
    hopper_x, hopper_y, hopper_z = (
        float(value) for value in model["hopper"]["local_center"]
    )

    def solid(
        identity: str,
        parent: str,
        label: str,
        center: list[float],
        half_extent: list[float],
        height: float,
        palette_role: str,
        *,
        elevation: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "identity": identity,
            "kind": "validator-rig-fixture",
            "label": label,
            "parent_identity": parent,
            "center": center,
            "half_extent": half_extent,
            "height": height,
            "floor_height": height,
            "wall_thickness": 0.04,
            "palette_role": palette_role,
            "wall_palette_role": palette_role,
            "geometry_mode": "solid",
            "openings": [],
            "placement": {"custody": "rig-owned", "elevation": elevation,
                          "yaw_degrees": 0.0},
            "physics": {"body": "static", "collider": "solid-contact-surface"},
            "artifact": {"capabilities": ["inspect", "publish-mesh"]},
        }

    cabinet = solid(
        f"{rig_identity}/geometry/cabinet",
        rig_identity,
        "Validator rig cabinet",
        [x, z],
        [1.25, 1.0],
        2.3,
        "building-wall",
    )
    stage = solid(
        f"{stage_identity}/geometry/platform",
        stage_identity,
        "Vehicle construction stage",
        [x + stage_dx, z + stage_dz],
        [1.7, 1.25],
        0.12,
        "ground",
        elevation=stage_y,
    )
    # Four narrow rails leave a real opening rather than painting a dark circle
    # onto the cabinet. The trigger volume itself is semantic and non-solid.
    mouth_x, mouth_z = x + hopper_x, z + hopper_z
    radius = float(model["hopper"]["radius_m"])
    rail = 0.055
    hopper_rails = (
        solid(f"{model['hopper']['identity']}/geometry/north", rig_identity,
              "Hopper north rim", [mouth_x, mouth_z + radius],
              [radius + rail, rail], 0.14, "active", elevation=hopper_y),
        solid(f"{model['hopper']['identity']}/geometry/south", rig_identity,
              "Hopper south rim", [mouth_x, mouth_z - radius],
              [radius + rail, rail], 0.14, "active", elevation=hopper_y),
        solid(f"{model['hopper']['identity']}/geometry/west", rig_identity,
              "Hopper west rim", [mouth_x - radius, mouth_z],
              [rail, radius - rail], 0.14, "active", elevation=hopper_y),
        solid(f"{model['hopper']['identity']}/geometry/east", rig_identity,
              "Hopper east rim", [mouth_x + radius, mouth_z],
              [rail, radius - rail], 0.14, "active", elevation=hopper_y),
    )
    return (cabinet, stage, *hopper_rails)


__all__ = [
    "VALIDATOR_RIG_SCHEMA", "ValidatorRigAssembly", "validator_rig_assembly",
    "validator_rig_geometry_boxes",
]
