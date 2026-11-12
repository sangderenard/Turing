"""Storybook authoring stays inside canonical class introspection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku.storybook_environment import (
    StorybookEnvironment, LanternCottage, WhisperingWood, MoonBridge,
    ClockworkTower, build_storybook_world,
)
from src.compiler.abstract_ui_introspection import build_introspective_world


def test_storybook_preserves_canonical_identity_order_and_edges():
    authored = build_introspective_world(StorybookEnvironment, depth_down=2)
    story = build_storybook_world()
    assert story.identity == authored.identity
    assert len(story.buildings()) == 5
    assert [b.identity for b in story.buildings()] == [b.identity for b in authored.buildings()]
    assert [(t.source, t.target, t.relationship) for t in story.tracks] == [
        (t.source, t.target, t.relationship) for t in authored.tracks]
    assert len(story.tracks) == 7
    for expected, actual in zip(authored.buildings(), story.buildings()):
        assert [(r.identity, r.member_kind) for r in actual.rooms] == [
            (r.identity, r.member_kind) for r in expected.rooms]


def test_storybook_narrative_is_retained_as_intention_data():
    world = build_storybook_world()
    for subject in (StorybookEnvironment, LanternCottage, WhisperingWood,
                    MoonBridge, ClockworkTower):
        building = world.building(subject.__name__)
        for name, value in vars(subject).items():
            if not name.startswith("_") and isinstance(value, str):
                assert building.room(name).intentions[-1].to_data()["arguments"] == [value]
    root = world.building("StorybookEnvironment")
    assert root.room("lantern_cottage").member_kind == "nested_class"


def test_storybook_is_deterministic_and_does_not_construct_classes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Introspection must not instantiate the story")
    monkeypatch.setattr(StorybookEnvironment, "__new__", forbidden)
    first = build_storybook_world()
    assert first == build_storybook_world()
    other = build_storybook_world(seed="another-edition")
    assert [b.identity for b in first.buildings()] == [b.identity for b in other.buildings()]
    assert [(t.source, t.target, t.relationship) for t in first.tracks] == [
        (t.source, t.target, t.relationship) for t in other.tracks]



def test_compact_graph_is_exact_full_projection_input(monkeypatch):
    import pytest
    from src.compiler import abstract_ui_div_map as projection
    from meowdoku.storybook_environment import storybook_graph
    world = build_storybook_world()
    compact = storybook_graph()
    class ReachedGeometry(Exception):
        pass
    def capture(model):
        assert model == compact
        raise ReachedGeometry
    monkeypatch.setattr(projection, "document_geometry", capture)
    # Stop at the first geometry call, before any kernel compilation. This is
    # the actual full producer, not a second expected-value serializer.
    with pytest.raises(ReachedGeometry):
        projection.world_div_map_model(world)
    assert "world" not in compact and "loop_deployment" not in compact


from tests.test_meowdoku_profile_dashboard import server, _cookie, _call


def test_storybook_actual_rest_walk_retains_authored_graph(server):
    from copy import deepcopy
    from meowdoku.storybook_environment import storybook_graph
    world = build_storybook_world()
    model = storybook_graph()
    original = deepcopy(model)
    root = world.building("StorybookEnvironment").identity
    headers = _cookie(server, "storywalker", "operator")
    base = "/api/living-data/sessions"
    status, _, state = _call(server.port, base, "POST", headers,
                           {"map": model, "at": root, "request_id": "storybook"})
    assert status == 201
    url = base + "/" + state["session_id"]
    # Follow only references authored in the actual class graph, in reverse
    # through the chapter chain. No new pathfinding or movement rule is added.
    for revision, name in enumerate(("ClockworkTower", "MoonBridge", "WhisperingWood", "LanternCottage")):
        target = world.building(name).identity
        edge = next(e for e in state["exits"] if e["to"] == target and e["relationship"] == "contains-type")
        status, _, state = _call(server.port, url + "/move", "POST", headers,
            {**edge, "expected_revision": revision, "request_id": f"chapter-{revision}"})
        assert status == 200
        assert state["at"]["identity"] == target and name in state["text"]
        assert not state["simulation_advanced"] and state["frame"] is None
    assert model == original
    assert server.sessions.tables == {}
    assert _call(server.port, url + "/close", "POST", headers,
                 {"expected_revision": 4})[0] == 200


def test_storybook_projection_is_static_by_default(monkeypatch):
    from meowdoku.storybook_environment import project_storybook_map
    from src.compiler import abstract_ui_div_map
    captured = {}
    def project(world, **kwargs):
        captured.update(kwargs)
        return world
    monkeypatch.setattr(abstract_ui_div_map, "project_world_to_div_map", project)
    project_storybook_map()
    assert captured["static_only"] is True
