"""Identity-preserving inventory custody and world-placement contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from .abstract_ui_tools import AbstractUITool, ToolHook, ToolMode


ABSTRACT_UI_PLACEMENT_VERSION = "abstract-ui-placement-v0"
SUBTRACTIVE_KINDS = ("door", "gate", "portal", "window")


@dataclass(frozen=True, slots=True)
class PlacementTransform:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def to_data(self) -> dict[str, list[float]]:
        return {
            "position": list(self.position), "rotation": list(self.rotation),
            "scale": list(self.scale),
        }


@dataclass(frozen=True, slots=True)
class PlacementPayload:
    """A movable representation whose authored graph relations remain intact."""

    identity: str
    semantic_owner: str
    source_container: str
    representation: Mapping[str, Any]
    custody: str = "inventory"
    transform: PlacementTransform = PlacementTransform()
    placement_kind: str = "additive"

    def __post_init__(self) -> None:
        if self.custody not in {"inventory", "preview", "placed"}:
            raise ValueError(f"unknown placement custody {self.custody!r}")
        if self.placement_kind not in {"additive", "subtractive"}:
            raise ValueError(f"unknown placement kind {self.placement_kind!r}")
        if not self.identity or not self.semantic_owner or not self.source_container:
            raise ValueError("placement payload requires identity, owner, and container")

    def with_custody(self, custody: str) -> "PlacementPayload":
        return replace(self, custody=custody)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": ABSTRACT_UI_PLACEMENT_VERSION,
            "identity": self.identity, "semantic_owner": self.semantic_owner,
            "source_container": self.source_container,
            "custody": self.custody, "placement_kind": self.placement_kind,
            "transform": self.transform.to_data(),
            "representation": dict(self.representation),
            "preserved_relations": ["owned-by", "filesystem-contained-by"],
        }


@dataclass(frozen=True, slots=True)
class PlacementRecipe:
    identity: str
    name: str
    placement_kind: str
    stock: int
    maximum_stack: int = 64
    opening_kind: str | None = None
    width: float = 0.8
    height: float = 1.1

    def __post_init__(self) -> None:
        if self.stock < 0 or self.stock > self.maximum_stack:
            raise ValueError("placement recipe stock exceeds its stack")
        if self.placement_kind == "subtractive" and self.opening_kind not in SUBTRACTIVE_KINDS:
            raise ValueError("subtractive recipes require a canonical opening kind")

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity, "name": self.name,
            "placement_kind": self.placement_kind, "stock": self.stock,
            "maximum_stack": self.maximum_stack, "opening_kind": self.opening_kind,
            "width": self.width, "height": self.height,
            "count_semantics": "available-unplaced-instances",
        }


@dataclass(frozen=True, slots=True)
class PlacementPolicy:
    identity: str
    snap_modes: tuple[str, ...] = (
        "free", "grid", "object-face", "object-center", "opening-track",
    )
    grid_step: float = 0.25
    snap_distance: float = 0.55

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": ABSTRACT_UI_PLACEMENT_VERSION,
            "identity": self.identity,
            "custody_states": ["inventory", "preview", "placed"],
            "semantic_owner_policy": "preserve-unless-explicit-transfer",
            "gimbal": {
                "translation_axes": ["x", "y", "z"],
                "rotation_axes": ["yaw", "pitch", "roll"],
                "scale_axes": ["uniform"],
            },
            "snap_modes": list(self.snap_modes), "grid_step": self.grid_step,
            "snap_distance": self.snap_distance,
            "subtractive_contract": {
                "host_required": True,
                "host_surface": "boundary-wall",
                "realization": "ordered-boundary-opening",
                "identity": "opening-object-retains-independent-identity",
            },
            "portal_contract": {
                "set_size": "many-to-many",
                "primary_action_role": "in",
                "secondary_action_role": "out",
                "directionality": "in-to-out",
                "target": "rendered-mesh-triangle-splat",
                "division": "triangle-barycentric-subdomains",
                "mapping": "local-manifold-frame",
                "backing": "probabilistic-tube-graph",
                "backing_graph": f"{self.identity}/port-graphs/default",
                "distribution": "normalized-spatial-gaussian",
                "intermediary_manifold": "directed-tube-edge",
                "path_model": "relaxed-quaternion-cubic",
                "modes": {
                    "standard": {
                        "aperture_class": "person",
                        "aperture_scale": 1.0,
                        "tube_scale": 1.0,
                        "handle_scale": 1.0,
                    },
                    "mega": {
                        "aperture_class": "vehicle",
                        "aperture_scale": 4.0,
                        "tube_scale": 4.0,
                        "handle_scale": 4.0,
                    },
                },
                "future_backing": "graph-defined-port-set",
            },
        }


def placement_tool(identity: str) -> AbstractUITool:
    return AbstractUITool(
        identity, "Placement tool", (
            ToolHook("primary-action", "placement-primary", "focused-identity"),
            ToolHook("secondary-action", "placement-secondary", "focused-identity"),
        ),
        modes=(
            ToolMode(
                f"{identity}/modes/standard", "standard",
                "Person-scale portals and quaternion tubes.",
                "place-output-portal", "place-input-portal",
            ),
            ToolMode(
                f"{identity}/modes/mega", "mega",
                "Vehicle-scale portals, tubes, path handles, and trumpet mouths.",
                "place-mega-output-portal", "place-mega-input-portal",
            ),
        ),
        default_mode="standard",
    )


def default_placement_recipes(root: str) -> tuple[PlacementRecipe, ...]:
    return (
        PlacementRecipe(f"{root}/recipes/door", "Door", "subtractive", 8,
                        opening_kind="door", width=0.82, height=1.15),
        PlacementRecipe(f"{root}/recipes/window", "Window", "subtractive", 12,
                        opening_kind="window", width=0.72, height=0.58),
        PlacementRecipe(f"{root}/recipes/gate", "Gate", "subtractive", 4,
                        opening_kind="gate", width=1.2, height=1.5),
        PlacementRecipe(f"{root}/recipes/portal", "Portal", "manifold", 12,
                        opening_kind="portal", width=0.9, height=1.35),
    )


__all__ = [
    "ABSTRACT_UI_PLACEMENT_VERSION", "SUBTRACTIVE_KINDS",
    "PlacementPayload", "PlacementPolicy", "PlacementRecipe",
    "PlacementTransform", "default_placement_recipes", "placement_tool",
]
