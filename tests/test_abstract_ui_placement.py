"""Placement keeps graph identity separate from representation custody."""

import pytest

from src.compiler.abstract_ui_placement import (
    PlacementPayload, PlacementPolicy, PlacementRecipe,
    default_placement_recipes, placement_tool,
)


def test_inventory_custody_does_not_rewrite_owner_or_source_container():
    payload = PlacementPayload("object:note", "class:Panel", "room:members", {})
    preview = payload.with_custody("preview")
    placed = preview.with_custody("placed")
    assert placed.semantic_owner == payload.semantic_owner
    assert placed.source_container == payload.source_container
    assert placed.to_data()["preserved_relations"] == [
        "owned-by", "filesystem-contained-by",
    ]


def test_placement_policy_publishes_gimbal_snap_and_subtractive_contracts():
    data = PlacementPolicy("world/placement").to_data()
    assert data["gimbal"]["translation_axes"] == ["x", "y", "z"]
    assert "object-face" in data["snap_modes"]
    assert data["subtractive_contract"]["realization"] == "ordered-boundary-opening"
    assert data["portal_contract"] == {
        "set_size": "many-to-many",
        "primary_action_role": "in",
        "secondary_action_role": "out",
        "directionality": "in-to-out",
        "target": "rendered-mesh-triangle-splat",
        "division": "triangle-barycentric-subdomains",
        "mapping": "local-manifold-frame",
        "backing": "probabilistic-tube-graph",
        "backing_graph": "world/placement/port-graphs/default",
        "distribution": "normalized-spatial-gaussian",
        "intermediary_manifold": "directed-tube-edge",
        "path_model": "relaxed-quaternion-cubic",
        "modes": {
            "standard": {
                "aperture_class": "person", "aperture_scale": 1.0,
                "tube_scale": 1.0, "handle_scale": 1.0,
            },
            "mega": {
                "aperture_class": "vehicle", "aperture_scale": 4.0,
                "tube_scale": 4.0, "handle_scale": 4.0,
            },
        },
        "future_backing": "graph-defined-port-set",
    }


def test_default_opening_recipes_have_minecraft_style_available_counts():
    recipes = default_placement_recipes("world/placement")
    assert [(item.opening_kind, item.stock) for item in recipes] == [
        ("door", 8), ("window", 12), ("gate", 4), ("portal", 12),
    ]
    assert all(item.to_data()["count_semantics"] == "available-unplaced-instances"
               for item in recipes)


def test_subtractive_recipe_rejects_unknown_opening_kind():
    with pytest.raises(ValueError, match="canonical opening"):
        PlacementRecipe("recipe:bad", "Bad", "subtractive", 1,
                        opening_kind="dent")


def test_placement_tool_routes_primary_and_secondary_hooks():
    tool = placement_tool("tool:placement").to_data()
    assert [hook["operation"] for hook in tool["hooks"]] == [
        "placement-primary", "placement-secondary",
    ]
    assert tool["default_mode"] == "standard"
    assert [mode["name"] for mode in tool["modes"]] == ["standard", "mega"]
    assert "Vehicle-scale" in tool["modes"][1]["description"]
