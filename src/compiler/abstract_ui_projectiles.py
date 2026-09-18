"""Gun tools and compiled-physics projectile archetypes for AbstractUI worlds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .abstract_ui_tools import AbstractUITool, ToolHook, ToolMode


ABSTRACT_UI_PROJECTILE_VERSION = "abstract-ui-projectiles-v0"


@dataclass(frozen=True, slots=True)
class ProjectileArchetype:
    identity: str
    name: str = "Physics ball"
    radius: float = 0.11
    mass: float = 0.12
    launch_speed: float = 8.5
    linear_drag: float = 0.045
    lifetime: float = 18.0
    maximum_active: int = 48
    palette_role: str = "projectile-ball"

    def __post_init__(self) -> None:
        if min(self.radius, self.mass, self.launch_speed, self.lifetime) <= 0:
            raise ValueError("projectile dimensions, mass, speed, and lifetime must be positive")
        if self.maximum_active < 1:
            raise ValueError("projectile archetype requires an active capacity")

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": ABSTRACT_UI_PROJECTILE_VERSION,
            "identity": self.identity, "name": self.name,
            "geometry": {"kind": "sphere", "radius": self.radius},
            "physics": {
                "program": "world.physics.compiled-sympy-wasm",
                "body": "dynamic", "collider": "sphere",
                "mass": self.mass, "linear_drag": self.linear_drag,
                "restitution": 0.78, "ball_restitution": 0.9,
            },
            "launch_speed": self.launch_speed, "lifetime": self.lifetime,
            "maximum_active": self.maximum_active,
            "palette_role": self.palette_role,
            "identity_policy": "mint-on-fire-retain-across-sleep-and-event-transition",
            "entity_contract": {
                "archetype": "physics-ball-entity",
                "controller": "compiled-projectile-physics",
                "coordinate_space": "data-world",
                "top_down_marker_scale": 0.45,
                "rest_state": "entity-sleeps-outside-active-physics-membership",
                "wake_conditions": ["collision-touch", "physics-field-change"],
                "sleep_speed": 0.32,
                "sleep_delay": 1.15,
                "event_transitions": {
                    "pickup": "entity-to-static-world-pickup",
                    "explode": "entity-to-explosion-event",
                    "deposit": "entity-to-world-material-deposit",
                    "illuminate": "entity-to-light-emitter",
                },
                "pickup_operation": "projectile-ammunition",
            },
        }


def gun_tool(identity: str, projectile: str) -> AbstractUITool:
    return AbstractUITool(
        identity, "Physics-ball gun", (
            ToolHook("primary-action", "fire-projectile", projectile),
            ToolHook("secondary-action", "mode-secondary-hold", projectile),
        ),
        modes=(
            ToolMode(f"{identity}/modes/normal", "normal",
                     "hold secondary to charge exit velocity; fire on release",
                     "charge-projectile-exit-velocity", "fire-projectile"),
            ToolMode(f"{identity}/modes/attractor", "attractor",
                     "primary pulls the crosshair target; hold secondary for a broad field",
                     "grow-projectile-attractor-strength", "pull-crosshair-projectile"),
        ),
        default_mode="normal",
    )


def projectile_system_model(root: str) -> dict[str, Any]:
    archetype = ProjectileArchetype(f"{root}/archetypes/physics-ball")
    return {
        "schema": ABSTRACT_UI_PROJECTILE_VERSION,
        "identity": f"{root}/projectiles",
        "organization": f"{root}/entities/organizations/projectiles",
        "cycle": "entities.projectile-compiled-physics",
        "archetype": archetype.to_data(),
        "instances": [],
        "stages": [
            "spawn-entity", "compiled-physics", "pose-publication",
            "sleep-outside-physics-membership", "wake-on-contact-or-field-change",
            "event-transition", "collect-to-ammunition",
        ],
    }


__all__ = [
    "ABSTRACT_UI_PROJECTILE_VERSION", "ProjectileArchetype", "gun_tool",
    "projectile_system_model",
]
