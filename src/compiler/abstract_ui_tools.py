"""Color selectors, entity inventories, and active-tool state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .abstract_ui_primitives import UIColor, UIPrimitive, div, input_, palette


ABSTRACT_UI_TOOL_VERSION = "abstract-ui-tools-v0"


def color_selector(
    identity: str,
    *,
    destination: str,
    value: str | UIColor,
    locked: bool = False,
) -> UIPrimitive:
    """Construct a neutral color-valued input, not a browser color widget."""

    color = UIColor(value) if isinstance(value, str) else value
    primitive = input_(
        identity,
        interaction="set-color",
        destination=destination,
        value=color.value,
        input_kind="color",
        palette=palette(bg=color, locked=locked),
    )
    return replace(primitive, archetype="color-selector")


@dataclass(frozen=True, slots=True)
class InventoryItem:
    identity: str
    entity: str
    name: str
    is_tool: bool = False
    color: UIColor | None = None
    properties: tuple[tuple[str, Any], ...] = ()
    slot: int | None = None
    quantity: int = 1
    maximum_stack: int = 1
    stack_key: str | None = None

    def __post_init__(self) -> None:
        if self.quantity < 0 or self.quantity > self.maximum_stack:
            raise ValueError("inventory quantity must fit its declared stack")


@dataclass(frozen=True, slots=True)
class ActiveTool:
    inventory: str
    item: str
    entity: str


@dataclass(frozen=True, slots=True)
class ToolHook:
    action: str
    operation: str
    destination: str

    def to_data(self) -> dict[str, str]:
        return {
            "action": self.action,
            "operation": self.operation,
            "destination": self.destination,
        }


@dataclass(frozen=True, slots=True)
class ToolMode:
    identity: str
    name: str
    description: str
    secondary_behavior: str
    primary_behavior: str = "default-tool-primary"

    def to_data(self) -> dict[str, str]:
        return {
            "identity": self.identity, "name": self.name,
            "description": self.description,
            "primary_behavior": self.primary_behavior,
            "secondary_behavior": self.secondary_behavior,
        }


@dataclass(frozen=True, slots=True)
class AestheticProperty:
    name: str
    label: str
    input_kind: str
    default: str | float
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None

    def to_data(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name, "label": self.label,
            "input_kind": self.input_kind, "default": self.default,
        }
        for name in ("minimum", "maximum", "step"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class AestheticPreset:
    identity: str
    name: str
    values: tuple[tuple[str, str | float], ...]

    def to_data(self) -> dict[str, Any]:
        return {"identity": self.identity, "name": self.name, "values": dict(self.values)}


@dataclass(frozen=True, slots=True)
class ToolDialogue:
    identity: str
    title: str
    properties: tuple[AestheticProperty, ...]
    presets: tuple[AestheticPreset, ...]

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": "abstract-ui-tool-dialogue-v0",
            "identity": self.identity,
            "title": self.title,
            "focus": "dialogue",
            "response_required": True,
            "properties": [item.to_data() for item in self.properties],
            "presets": [item.to_data() for item in self.presets],
        }


@dataclass(frozen=True, slots=True)
class AbstractUITool:
    identity: str
    name: str
    hooks: tuple[ToolHook, ...]
    dialogue: ToolDialogue | None = None
    modes: tuple[ToolMode, ...] = ()
    default_mode: str | None = None

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": "abstract-ui-tool-v0",
            "identity": self.identity,
            "kind": "tool",
            "name": self.name,
            "hooks": [hook.to_data() for hook in self.hooks],
            "dialogue": None if self.dialogue is None else self.dialogue.to_data(),
            "modes": [mode.to_data() for mode in self.modes],
            "default_mode": self.default_mode,
        }


def form_tool(identity: str) -> AbstractUITool:
    dialogue_identity = f"{identity}/aesthetic-dialogue"
    return AbstractUITool(
        identity,
        "Form tool",
        (
            ToolHook("primary-action", "open-dialogue", dialogue_identity),
            ToolHook("secondary-action", "toggle-focus-context", "focused-identity"),
        ),
        ToolDialogue(
            dialogue_identity,
            "Aesthetic form",
            (
                AestheticProperty("face_color", "Face color", "color", "#68b89c"),
                AestheticProperty("wall_color", "Wall color", "color", "#41675b"),
                AestheticProperty("height", "Wall height", "range", 1.0, 0.04, 4.0, 0.04),
                AestheticProperty("wall_thickness", "Wall thickness", "range", 0.04, 0.01, 0.3, 0.01),
                AestheticProperty("radius", "Corner radius", "range", 8.0, 0.0, 32.0, 1.0),
            ),
            (
                AestheticPreset("preset:verdant", "Verdant", (
                    ("face_color", "#68b89c"), ("wall_color", "#41675b"),
                    ("wall_thickness", 0.04), ("radius", 8.0),
                )),
                AestheticPreset("preset:warm", "Warm", (
                    ("face_color", "#c9865b"), ("wall_color", "#f0bd73"),
                    ("wall_thickness", 0.08), ("radius", 12.0),
                )),
                AestheticPreset("preset:stone", "Stone", (
                    ("face_color", "#66706d"), ("wall_color", "#a8b0aa"),
                    ("wall_thickness", 0.14), ("radius", 2.0),
                )),
            ),
        ),
    )


def depth_map_tool(identity: str) -> AbstractUITool:
    return AbstractUITool(
        identity,
        "Depth-map tool",
        (
            ToolHook("primary-action", "depth-map-primary", "focused-height-field"),
            ToolHook("secondary-action", "depth-map-secondary", "focused-height-field"),
        ),
        modes=(
            ToolMode(f"{identity}/modes/sculpt", "sculpt",
                     "Left lowers terrain; right raises terrain.",
                     "raise-depth", "lower-depth"),
            ToolMode(f"{identity}/modes/middle", "middle",
                     "Left relaxes terrain toward its middle height; right grows texture scale.",
                     "grow-texture", "relax-to-middle"),
        ),
        default_mode="sculpt",
    )


@dataclass(frozen=True, slots=True)
class EntityInventory:
    """An immutable corral of reachable entity references and equipped state."""

    identity: str
    owner: str
    items: tuple[InventoryItem, ...] = ()
    active_tool: ActiveTool | None = None

    def add(self, item: InventoryItem) -> "EntityInventory":
        if any(existing.identity == item.identity for existing in self.items):
            raise ValueError(f"inventory item already exists: {item.identity}")
        if any(existing.entity == item.entity for existing in self.items):
            raise ValueError(f"entity is already inventoried: {item.entity}")
        if item.slot is not None:
            if item.slot < 1:
                raise ValueError("inventory slots are one-based")
            if any(existing.slot == item.slot for existing in self.items):
                raise ValueError(f"inventory slot is occupied: {item.slot}")
        return replace(self, items=(*self.items, item))

    def remove(self, item_identity: str) -> "EntityInventory":
        item = self.item(item_identity)
        active = self.active_tool
        if active is not None and active.item == item.identity:
            active = None
        return replace(
            self,
            items=tuple(existing for existing in self.items if existing != item),
            active_tool=active,
        )

    def item(self, identity: str) -> InventoryItem:
        for item in self.items:
            if item.identity == identity:
                return item
        raise KeyError(identity)

    def equip(self, item_identity: str) -> "EntityInventory":
        item = self.item(item_identity)
        if not item.is_tool:
            raise ValueError(f"inventory item is not a tool: {item_identity}")
        return replace(
            self,
            active_tool=ActiveTool(self.identity, item.identity, item.entity),
        )

    def clear_tool(self) -> "EntityInventory":
        return replace(self, active_tool=None)

    def to_primitive(self) -> UIPrimitive:
        children = tuple(
            replace(
                div(
                    item.identity,
                    name=item.name,
                    entity=item.entity,
                    is_tool=item.is_tool,
                    color=None if item.color is None else item.color.value,
                    slot=item.slot,
                    quantity=item.quantity,
                    maximum_stack=item.maximum_stack,
                    stack_key=item.stack_key,
                ),
                archetype="inventory-item",
            )
            for item in self.items
        )
        return replace(
            div(
                self.identity,
                *children,
                owner=self.owner,
                active_tool=(
                    None if self.active_tool is None else self.active_tool.item
                ),
            ),
            archetype="entity-inventory",
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": ABSTRACT_UI_TOOL_VERSION,
            "identity": self.identity,
            "owner": self.owner,
            "items": [
                {
                    "identity": item.identity,
                    "entity": item.entity,
                    "name": item.name,
                    "is_tool": item.is_tool,
                    "slot": item.slot,
                    "color": None if item.color is None else item.color.value,
                    "properties": dict(item.properties),
                    "quantity": item.quantity,
                    "maximum_stack": item.maximum_stack,
                    "stack_key": item.stack_key,
                }
                for item in self.items
            ],
            "active_tool": None if self.active_tool is None else {
                "item": self.active_tool.item,
                "entity": self.active_tool.entity,
            },
        }


@dataclass(frozen=True, slots=True)
class HotbarSlot:
    number: int
    key: str
    item: str | None = None


@dataclass(frozen=True, slots=True)
class Hotbar:
    """An editable view onto the first ten one-based inventory slots."""

    identity: str
    inventory: str
    slots: tuple[HotbarSlot, ...]
    active_slot: int | None = None

    @staticmethod
    def from_inventory(inventory: EntityInventory, *, size: int = 10) -> "Hotbar":
        if size != 10:
            raise ValueError("the initial numeric hotbar has exactly ten slots")
        by_slot = {item.slot: item.identity for item in inventory.items if item.slot is not None}
        slots = tuple(
            HotbarSlot(number, f"Digit{number % 10}", by_slot.get(number))
            for number in range(1, 11)
        )
        active_item = None if inventory.active_tool is None else inventory.active_tool.item
        active_slot = None if active_item is None else next(
            (slot.number for slot in slots if slot.item == active_item), None,
        )
        return Hotbar(f"{inventory.identity}/hotbar", inventory.identity, slots, active_slot)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": "abstract-ui-hotbar-v0",
            "identity": self.identity,
            "inventory": self.inventory,
            "active_slot": self.active_slot,
            "slots": [
                {"number": slot.number, "label": str(slot.number % 10),
                 "key": slot.key, "item": slot.item}
                for slot in self.slots
            ],
            "relationship": "view-of-inventory-slots-1-through-10",
        }


__all__ = [
    "ABSTRACT_UI_TOOL_VERSION",
    "AbstractUITool",
    "ActiveTool",
    "AestheticPreset",
    "AestheticProperty",
    "EntityInventory",
    "Hotbar",
    "HotbarSlot",
    "InventoryItem",
    "ToolDialogue",
    "ToolHook",
    "ToolMode",
    "color_selector",
    "form_tool", "depth_map_tool",
]
