"""The aspirational AbstractUI boneyard remains executable and inspectable."""

import pytest

from src.compiler.abstract_ui_vocabulary import (
    ABSTRACT_UI_VOCABULARY,
    UIIntention,
    building,
    door,
    interior,
    open_,
    player,
    region,
    reveal,
    room,
    vocabulary_manifest,
    walk_intentions,
    when,
    world,
)


def test_boneyard_exposes_a_large_finite_declared_vocabulary():
    manifest = vocabulary_manifest()
    assert len(manifest) >= 400
    assert len(manifest) == len(ABSTRACT_UI_VOCABULARY)
    assert manifest[0]["name"] == "abstract_ui"
    assert {entry["domain"] for entry in manifest} >= {
        "context", "existence", "navigation", "action", "narrative",
        "program", "style", "world", "event", "accessibility",
    }
    assert {"filesystem", "readme", "scratch_file", "source_file", "test_file",
            "weld", "unweld", "custody", "placement", "gimbal", "skybox",
            "additive", "subtractive", "gun", "projectile", "physics_ball",
            "fire_projectile"} <= set(ABSTRACT_UI_VOCABULARY)


def test_obvious_nested_world_language_builds_serializable_intentions():
    specimen = world(
        region(
            "foundry",
            building(
                "pump house",
                interior(room("control room")),
                enterable=True,
            ),
        ),
    )
    data = specimen.to_data()
    assert data["word"] == "world"
    assert data["arguments"][0]["word"] == "region"
    pump_house = data["arguments"][0]["arguments"][1]
    assert pump_house["word"] == "building"
    assert pump_house["traits"] == {"enterable": True}


def test_fluent_and_perl_like_composition_records_intention_not_host_behavior():
    operator = player("operator")
    east_door = door("east")
    flow = when(operator.does(open_(east_door))) >> reveal(room("pump"))
    assert isinstance(flow, UIIntention)
    assert flow.word == "then"
    words = [node.word for node in walk_intentions(flow)]
    assert words[:4] == ["then", "when", "does", "player"]
    assert "open" in words
    assert "reveal" in words


def test_logic_operators_are_structural_and_truthiness_is_rejected():
    left = door("east").enabled(True)
    right = room("pump").visible(True)
    assert (left & right).word == "all_of"
    assert (left | right).word == "any_of"
    assert (~left).word == "not"
    with pytest.raises(TypeError, match="no Python truth value"):
        bool(left)


def test_unknown_fluent_human_phrase_is_retained_as_open_vocabulary():
    phrase = building("archive").remembers("every visitor", tenderly=True)
    assert phrase.word == "remembers"
    assert phrase.domain == "open"
    assert phrase.arguments[0].word == "building"
    assert phrase.to_data()["traits"] == {"tenderly": True}
