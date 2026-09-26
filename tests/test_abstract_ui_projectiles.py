"""Projectile archetype and gun-tool contracts."""

import pytest

from src.compiler.abstract_ui_projectiles import (
    ProjectileArchetype, gun_tool, projectile_system_model,
)


def test_physics_ball_uses_compiled_world_physics_and_spherical_identity():
    data = ProjectileArchetype("world/balls/default").to_data()
    assert data["geometry"] == {"kind": "sphere", "radius": 0.11}
    assert data["physics"]["program"] == "world.physics.compiled-sympy-wasm"
    assert data["identity_policy"] == (
        "mint-on-fire-retain-across-sleep-and-event-transition"
    )
    assert data["entity_contract"]["controller"] == "compiled-projectile-physics"
    assert data["entity_contract"]["rest_state"] == (
        "entity-sleeps-outside-active-physics-membership"
    )
    assert data["entity_contract"]["wake_conditions"] == [
        "collision-touch", "physics-field-change",
    ]
    assert data["entity_contract"]["event_transitions"]["pickup"] == (
        "entity-to-static-world-pickup"
    )
    assert {"explode", "deposit", "illuminate"} <= set(
        data["entity_contract"]["event_transitions"]
    )
    assert data["entity_contract"]["pickup_operation"] == "projectile-ammunition"
    assert data["physics"]["restitution"] == 0.78
    assert data["physics"]["ball_restitution"] == 0.9


def test_gun_routes_primary_to_projectile_spawn():
    data = gun_tool("world/tools/gun", "world/balls/default").to_data()
    assert data["name"] == "Physics-ball gun"
    assert data["hooks"][0] == {
        "action": "primary-action", "operation": "fire-projectile",
        "destination": "world/balls/default",
    }
    assert data["hooks"][1]["operation"] == "mode-secondary-hold"
    assert data["default_mode"] == "normal"
    assert [mode["name"] for mode in data["modes"]] == ["normal", "attractor"]
    assert data["modes"][0]["secondary_behavior"] == (
        "charge-projectile-exit-velocity"
    )
    assert data["modes"][1]["secondary_behavior"] == (
        "grow-projectile-attractor-strength"
    )
    assert data["modes"][0]["primary_behavior"] == "fire-projectile"
    assert data["modes"][1]["primary_behavior"] == "pull-crosshair-projectile"


def test_projectile_system_has_bounded_active_cycle_and_instance_archive():
    data = projectile_system_model("world")
    assert data["archetype"]["maximum_active"] == 48
    assert data["instances"] == []
    assert data["stages"] == [
        "spawn-entity", "compiled-physics", "pose-publication",
        "sleep-outside-physics-membership", "wake-on-contact-or-field-change",
        "event-transition", "collect-to-ammunition",
    ]


def test_projectile_archetype_rejects_nonpositive_physics_values():
    with pytest.raises(ValueError, match="must be positive"):
        ProjectileArchetype("bad", radius=0)
