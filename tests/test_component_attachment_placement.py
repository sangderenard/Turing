import pytest

from src.compiler.abstract_ui_archetypes import LivingDocument, LivingNode
from src.compiler.abstract_ui_placement import (
    PlacementPayload, PlacementTransform, mark_attachment_point, connect_component,
    apply_component_placement_action,
)
from src.compiler.abstract_ui_actions import IssuedAction


def test_component_connections_use_named_ports_and_owner_local_points():
    initial = LivingDocument("world", nodes=(LivingNode("rig", "object", "Rig"),))
    document = initial
    for i in range(4):
        document = mark_attachment_point(document, identity=f"point/{i}", owner="rig",
                                          pose=PlacementTransform(position=(i, 0, 0)))
    assert len(initial.nodes) == 1
    payload = PlacementPayload("divider", "rig", "inventory", {
        "archetype": "power-divider", "connection_ports": ("input", "left", "right", "mount"),
    })
    bindings = {"mount": "point/3", "right": "point/2", "left": "point/1", "input": "point/0"}
    result = connect_component(document, payload, bindings)
    connections = [e for e in result.edges if e.relationship == "connected-at"]
    assert [dict(e.properties)["port"] for e in connections] == ["input", "left", "right", "mount"]
    assert [e.target for e in connections] == [f"point/{i}" for i in range(4)]
    assert all(e.source == payload.identity for e in connections)
    assert dict(result.nodes[1].properties)["coordinate_space"] == "owner-local"
    assert dict(result.nodes[-1].properties)["placement"]["custody"] == "placed"
    assert payload.custody == "inventory"
    assert result.revision == 5 and document.revision == 4
    with pytest.raises(ValueError, match="exactly"):
        connect_component(document, payload, {"input": "point/0"})
    with pytest.raises(KeyError, match="marked"):
        connect_component(document, payload, {**bindings, "mount": "rig"})
    with pytest.raises(ValueError, match="already exists"):
        connect_component(result, payload, bindings)
    # Selection order or actor-specific UI code cannot alter port routing.
    assert connect_component(document, payload, dict(reversed(list(bindings.items())))) == result


def test_human_and_machine_placement_actions_share_edits_and_reject_stale_revision():
    initial = LivingDocument("world", nodes=(LivingNode("rig", "object", "Rig"),))
    results = []
    for actor in ("player", "validator"):
        document = initial
        for i in range(2):
            action = IssuedAction(f"{actor}/mark/{i}", actor, "mark-attachment-point",
                                  "world", "placement", 1.0, (
                ("expected_revision", document.revision), ("identity", f"point/{i}"),
                ("owner", "rig"), ("pose", PlacementTransform(position=(i, 0, 0))),
            ))
            previous = document
            document, edit = apply_component_placement_action(document, action)
            assert edit.actor == actor and edit.identity == action.identity
            assert edit.before_revision == previous.revision
            assert edit.added_nodes == document.nodes[-1:]
            with pytest.raises(ValueError, match="stale"):
                apply_component_placement_action(document, action)
        payload = PlacementPayload("actuator", "rig", "stock", {
            "archetype": "linear-actuator", "connection_ports": ("base", "rod"),
        })
        action = IssuedAction(f"{actor}/install", actor, "connect-component", "world",
                              "placement", 2.0, (
            ("expected_revision", document.revision), ("payload", payload),
            ("bindings", {"base": "point/0", "rod": "point/1"}),
        ))
        document, edit = apply_component_placement_action(document, action)
        assert edit.actor == actor and len(edit.added_edges) == 3
        results.append(document)
    assert results[0] == results[1]
    assert len(initial.nodes) == 1 and initial.revision == 0


def test_placed_support_drives_existing_rig_law_and_returns_reaction():
    import numpy as np
    from src.common.tensors import AbstractTensor
    from src.compiler.mechanical_ports import bind_placed_rig_point
    from src.compiler.vehicle_native_graph_program import VEHICLE_NATIVE_GRAPH_VECTOR_SOURCE

    document = LivingDocument("world", nodes=(
        LivingNode("world", "world", "World"), LivingNode("body", "object", "Body"),
    ))
    for identity, owner, position in (("body-point", "body", (1, 0, 0)),
                                      ("target", "world", (1, 2, 0))):
        document, _ = apply_component_placement_action(document, IssuedAction(
            f"mark/{identity}", "validator", "mark-attachment-point", "world", "placement", 0,
            (("expected_revision", document.revision), ("identity", identity),
             ("owner", owner), ("pose", PlacementTransform(position=position))),
        ))
    payload = PlacementPayload("support", "world", "stock", {
        "connection_ports": ("body", "world"), "mechanical_operator": "vehicle_rig_points_vector",
    })
    document, _ = apply_component_placement_action(document, IssuedAction(
        "install/support", "validator", "connect-component", "world", "placement", 0,
        (("expected_revision", document.revision), ("payload", payload),
         ("bindings", {"body": "body-point", "world": "target"})),
    ))
    command = dict(enabled=1, mode=1, stiffness=(10, 10, 10), damping=(0, 0, 0),
                   maximum_force=100, target_velocity=(0, 0, 0), force=(0, 0, 0))
    binding = bind_placed_rig_point(document, "support", "body", command)
    assert (binding.identity, binding.body_point, binding.world_point) == ("support", "body-point", "target")
    namespace = {"AbstractTensor": AbstractTensor}
    exec(VEHICLE_NATIVE_GRAPH_VECTOR_SOURCE, namespace)
    with AbstractTensor.use_backend("numpy"):
        zero = AbstractTensor.tensor(np.zeros((1, 3)), dtype="float64")
        force, moment, reaction = namespace["vehicle_rig_points_vector"](
            zero, zero, zero, zero,
            AbstractTensor.tensor(np.array(binding.values).reshape(1, 1, 21), dtype="float64"))
    np.testing.assert_allclose(force.data, [[0, 20, 0]])
    np.testing.assert_allclose(moment.data, [[0, 0, 20]])
    np.testing.assert_allclose(reaction.data, [[[0, -20, 0, 0, 0, -20]]])
    with pytest.raises(ValueError, match="port owners"):
        bind_placed_rig_point(document, "support", "another-body", command)
