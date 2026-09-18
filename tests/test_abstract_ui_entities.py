"""Entity mezzanine, native actor, NPC, and cycle contract tests."""

import pytest

from src.compiler.abstract_ui_entities import (
    ENTITY_CYCLE_PHASES,
    POINTER_ARCHETYPE,
    ControlInputFrame,
    EntityCyclePolicy,
    EntityInteraction,
    EntityTick,
    entity_mezzanine_model,
    run_entity_cycle,
    spawn_pointer_and_follower,
    spawn_pointer_and_followers,
)


def test_native_pointer_and_npc_are_spawned_from_the_identical_archetype():
    mezzanine = spawn_pointer_and_follower()
    pointer, npc = mezzanine.entities
    assert pointer.archetype is POINTER_ARCHETYPE
    assert npc.archetype is POINTER_ARCHETYPE
    assert pointer.archetype.geometry == npc.archetype.geometry
    assert pointer.archetype.texture == npc.archetype.texture
    assert pointer.controller.kind == "native-input"
    assert npc.controller.kind == "second-order-follow"
    assert npc.controller.parameter("target") == pointer.identity
    geometry = dict(pointer.archetype.geometry.parameters)
    assert geometry["radius"] == 1.75
    assert geometry["embodiment_scale"] == 0.25


def test_entities_are_namespaced_under_system_root_and_corralled_together():
    mezzanine = spawn_pointer_and_follower(system_root="world")
    assert mezzanine.identity == "world/entities"
    group = mezzanine.organizations[0]
    assert group.identity == "world/entities/organizations/pointer-beings"
    assert group.members == tuple(entity.identity for entity in mezzanine.entities)
    assert len(set(group.members)) == 2


def test_control_phase_moves_mouse_actor_then_second_order_cycle_pursues_it():
    mezzanine = spawn_pointer_and_follower()
    pointer, npc = mezzanine.entities
    result = run_entity_cycle(
        mezzanine,
        EntityTick(1, 1.0, 0.1),
        inputs=(ControlInputFrame("mouse.primary", 1, 1.0, (100.0, 80.0, 0.0)),),
    )
    moved_pointer = result.mezzanine.entity(pointer.identity)
    moved_npc = result.mezzanine.entity(npc.identity)
    assert moved_pointer.pose.position == (100.0, 80.0, 0.0)
    assert moved_npc.pose.position[0] > npc.pose.position[0]
    assert moved_npc.pose.position[1] > npc.pose.position[1]
    assert moved_npc.pose.velocity != (0.0, 0.0, 0.0)
    assert moved_pointer.pose.facing[0] > 0
    assert moved_pointer.pose.facing[1] > 0


def test_cycle_policy_can_move_as_a_whole_without_changing_semantics():
    mezzanine = spawn_pointer_and_follower()
    tick = EntityTick(1, 1.0, 0.05)
    frames = (ControlInputFrame("mouse.primary", 1, 1.0, (50.0, 25.0, 0.0)),)
    inline = run_entity_cycle(
        mezzanine, tick, inputs=frames, policy=EntityCyclePolicy("inline"),
    )
    worker = run_entity_cycle(
        mezzanine, tick, inputs=frames, policy=EntityCyclePolicy("worker"),
    )
    assert inline.mezzanine.entities == worker.mezzanine.entities
    assert inline.presentation == worker.presentation
    assert inline.policy.execution == "inline"
    assert worker.policy.execution == "worker"
    assert inline.policy.phases == ENTITY_CYCLE_PHASES


def test_interactions_are_conceptual_entity_records_not_callbacks():
    mezzanine = spawn_pointer_and_follower()
    pointer, npc = mezzanine.entities
    interaction = EntityInteraction(pointer.identity, "approach", npc.identity)
    result = run_entity_cycle(
        mezzanine, EntityTick(1, 0.0, 0.0), interactions=(interaction,),
    )
    assert result.interactions == (interaction,)
    with pytest.raises(KeyError, match="interaction actor is missing"):
        run_entity_cycle(
            mezzanine,
            EntityTick(2, 0.1, 0.1),
            interactions=(EntityInteraction("missing", "inspect", npc.identity),),
        )


def test_transport_model_exposes_mezzanine_cycle_controllers_and_embodiment():
    model = entity_mezzanine_model(spawn_pointer_and_follower())
    assert model["system_root"] == "system-root"
    assert model["cycle"]["phases"] == list(ENTITY_CYCLE_PHASES)
    assert model["cycle"]["execution"] == ["inline", "worker"]
    assert [entity["controller"]["kind"] for entity in model["entities"]] == [
        "native-input", "second-order-follow",
    ]
    assert model["entities"][0]["geometry"] == model["entities"][1]["geometry"]
    assert model["entities"][0]["texture"] == model["entities"][1]["texture"]


def test_first_through_fourth_order_followers_share_body_and_have_distinct_colors():
    mezzanine = spawn_pointer_and_followers()
    pointer, *followers = mezzanine.entities
    assert [entity.controller.kind for entity in mezzanine.entities] == [
        "native-input", "first-order-follow", "second-order-follow",
        "third-order-follow", "fourth-order-follow",
    ]
    assert all(entity.archetype is POINTER_ARCHETYPE for entity in mezzanine.entities)
    colors = [dict(entity.traits)["color"] for entity in mezzanine.entities]
    assert len(colors) == len(set(colors)) == 5
    initial_distances = {
        entity.identity: abs(200.0 - entity.pose.position[0]) for entity in followers
    }
    for sequence in range(1, 201):
        inputs = (
            (ControlInputFrame("mouse.primary", sequence, sequence * 0.01, (200.0, 120.0, 0.0)),)
            if sequence == 1 else ()
        )
        mezzanine = run_entity_cycle(
            mezzanine, EntityTick(sequence, sequence * 0.01, 0.01), inputs=inputs,
        ).mezzanine
    for entity in mezzanine.entities[1:]:
        assert abs(200.0 - entity.pose.position[0]) < initial_distances[entity.identity]
    model = entity_mezzanine_model(mezzanine)
    assert [entity["color"] for entity in model["entities"]] == colors
    assert all(len(entity["pose"]["facing"]) == 3 for entity in model["entities"])
