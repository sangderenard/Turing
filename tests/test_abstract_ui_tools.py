"""Color selector, entity inventory, and active tool tests."""

import pytest

from src.compiler.abstract_ui_primitives import UIColor
from src.compiler.abstract_ui_tools import (
    EntityInventory,
    Hotbar,
    InventoryItem,
    color_selector,
    depth_map_tool,
    form_tool,
)


def test_color_selector_is_a_color_input_with_semantic_destination():
    selector = color_selector(
        "tools/color", destination="entity.pointer.first.color", value="#ff6b6b",
    )
    properties = dict(selector.properties)
    assert selector.archetype == "color-selector"
    assert properties["input_kind"] == "color"
    assert properties["interaction"] == "set-color"
    assert properties["destination"] == "entity.pointer.first.color"
    assert selector.effective_palette().bg == UIColor("#ff6b6b")


def test_inventory_reaches_entities_and_tracks_one_active_tool():
    inventory = EntityInventory("inventory:pointer", "pointer.primary")
    inventory = inventory.add(InventoryItem(
        "inventory:pointer/color", "selector:color", "Color selector",
        is_tool=True, color=UIColor("#c77dff"),
    ))
    inventory = inventory.add(InventoryItem(
        "inventory:pointer/orb", "entity:orb", "Orb", is_tool=False,
    ))
    equipped = inventory.equip("inventory:pointer/color")
    assert equipped.active_tool.entity == "selector:color"
    assert equipped.active_tool.inventory == inventory.identity

    primitive = equipped.to_primitive()
    assert primitive.archetype == "entity-inventory"
    assert dict(primitive.properties)["active_tool"] == "inventory:pointer/color"
    assert [child.archetype for child in primitive.children] == [
        "inventory-item", "inventory-item",
    ]


def test_non_tool_cannot_be_equipped_and_removing_active_tool_clears_it():
    inventory = EntityInventory("inventory", "actor")
    inventory = inventory.add(InventoryItem("tool", "entity:tool", "Tool", True))
    inventory = inventory.add(InventoryItem("thing", "entity:thing", "Thing", False))
    with pytest.raises(ValueError, match="not a tool"):
        inventory.equip("thing")
    assert inventory.equip("tool").remove("tool").active_tool is None


def test_hotbar_is_an_ordered_view_of_first_ten_inventory_slots():
    inventory = EntityInventory("inventory", "actor").add(InventoryItem(
        "tool", "entity:tool", "Form tool", True, slot=1,
    )).equip("tool")
    hotbar = Hotbar.from_inventory(inventory)
    assert len(hotbar.slots) == 10
    assert hotbar.slots[0].key == "Digit1"
    assert hotbar.slots[0].item == "tool"
    assert hotbar.slots[9].key == "Digit0"
    assert hotbar.active_slot == 1


def test_inventory_rejects_duplicate_and_zero_slots():
    inventory = EntityInventory("inventory", "actor").add(InventoryItem(
        "one", "entity:one", "One", True, slot=1,
    ))
    with pytest.raises(ValueError, match="occupied"):
        inventory.add(InventoryItem("two", "entity:two", "Two", True, slot=1))
    with pytest.raises(ValueError, match="one-based"):
        EntityInventory("other", "actor").add(InventoryItem(
            "zero", "entity:zero", "Zero", True, slot=0,
        ))


def test_form_tool_is_an_abstract_ui_object_with_input_hooks_and_dialogue():
    tool = form_tool("tools/form").to_data()
    assert tool["schema"] == "abstract-ui-tool-v0"
    assert [(hook["action"], hook["operation"]) for hook in tool["hooks"]] == [
        ("primary-action", "open-dialogue"),
        ("secondary-action", "toggle-focus-context"),
    ]
    dialogue = tool["dialogue"]
    assert dialogue["focus"] == "dialogue"
    assert dialogue["response_required"]
    assert [item["name"] for item in dialogue["properties"]] == [
        "face_color", "wall_color", "height", "wall_thickness", "radius",
    ]
    assert [preset["name"] for preset in dialogue["presets"]] == [
        "Verdant", "Warm", "Stone",
    ]


def test_depth_map_tool_has_sculpt_and_middle_modes():
    tool = depth_map_tool("tools/depth-map").to_data()
    assert tool["default_mode"] == "sculpt"
    assert [(hook["action"], hook["operation"]) for hook in tool["hooks"]] == [
        ("primary-action", "depth-map-primary"),
        ("secondary-action", "depth-map-secondary"),
    ]
    assert [(mode["name"], mode["primary_behavior"], mode["secondary_behavior"])
            for mode in tool["modes"]] == [
        ("sculpt", "lower-depth", "raise-depth"),
        ("middle", "relax-to-middle", "grow-texture"),
    ]
