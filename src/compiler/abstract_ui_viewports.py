"""Viewer-camera and shader-viewport objects for AbstractUI projections."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .abstract_ui_dynamics import DeviceMonitor, DynamicsSpace
from .abstract_ui_primitives import UIBBox, UIPalette


ABSTRACT_UI_VIEWPORT_VERSION = "abstract-ui-viewport-v0"


def _vector3(value: tuple[float, float, float], label: str) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must be a finite three-vector")
    return result


@dataclass(frozen=True, slots=True)
class ViewerCamera:
    identity: str
    subject: str
    position: tuple[float, float, float] = (0.0, 1.2, 0.0)
    facing: tuple[float, float, float] = (0.0, 0.0, -1.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    field_of_view: float = 70.0
    tracking_actor: str | None = None
    facing_rule: str = "actor-motion"
    embodiment_scale: float = 1.0
    eye_height: float = 1.15
    collision_radius: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector3(self.position, "camera position"))
        object.__setattr__(self, "facing", _vector3(self.facing, "camera facing"))
        object.__setattr__(self, "up", _vector3(self.up, "camera up"))
        if not 1.0 <= self.field_of_view < 180.0:
            raise ValueError("camera field of view must be in [1, 180)")
        if self.embodiment_scale <= 0 or self.eye_height <= 0 or self.collision_radius <= 0:
            raise ValueError("camera embodiment dimensions must be positive")

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "kind": "viewer-camera",
            "subject": self.subject,
            "position": list(self.position),
            "facing": list(self.facing),
            "up": list(self.up),
            "field_of_view": self.field_of_view,
            "tracking_actor": self.tracking_actor,
            "facing_rule": self.facing_rule,
            "embodiment_scale": self.embodiment_scale,
            "eye_height": self.eye_height,
            "collision_radius": self.collision_radius,
        }


@dataclass(frozen=True, slots=True)
class FragmentOperation:
    identity: str
    operation: str
    inputs: tuple[str, ...]
    output: str

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "operation": self.operation,
            "inputs": list(self.inputs),
            "output": self.output,
        }


@dataclass(frozen=True, slots=True)
class ShaderProgramChoice:
    """One selectable graphics program and its explicit adapter contract."""

    identity: str
    label: str
    vertex_source: str
    fragment_source: str
    origin: str
    adapter: str
    resource_bindings: tuple[Mapping[str, Any], ...] = ()

    def to_data(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "label": self.label,
            "vertex_source": self.vertex_source,
            "fragment_source": self.fragment_source,
            "origin": self.origin,
            "adapter": self.adapter,
            "resource_bindings": [dict(item) for item in self.resource_bindings],
        }


@dataclass(frozen=True, slots=True)
class ViewportControlBinding:
    """One semantic actor action and the native inputs which may express it."""

    action: str
    inputs: tuple[str, ...]

    def to_data(self) -> dict[str, Any]:
        return {"action": self.action, "inputs": list(self.inputs)}


DEFAULT_VIEWPORT_CONTROL_BINDINGS = (
    ViewportControlBinding("move-forward", ("keyboard:KeyW", "gamepad:left-y-negative")),
    ViewportControlBinding("move-backward", ("keyboard:KeyS", "gamepad:left-y-positive")),
    ViewportControlBinding("strafe-left", ("keyboard:KeyA", "gamepad:left-x-negative")),
    ViewportControlBinding("strafe-right", ("keyboard:KeyD", "gamepad:left-x-positive")),
    ViewportControlBinding("run", ("keyboard:ShiftLeft", "keyboard:ShiftRight")),
    ViewportControlBinding("jump", ("keyboard:Space",)),
    ViewportControlBinding("look", ("pointer:relative-motion", "gamepad:right-stick")),
    ViewportControlBinding("primary-action", ("pointer:button-0", "gamepad:button-0")),
    ViewportControlBinding("secondary-action", ("pointer:button-2", "gamepad:button-1")),
)


@dataclass(frozen=True, slots=True)
class ViewportControlPolicy:
    """Backend-neutral rules for temporarily routing input to a viewport actor."""

    identity: str
    actor: str | None
    activation: str = "highlight"
    release: tuple[str, ...] = ("escape", "focus-loss")
    captures: tuple[str, ...] = ("keyboard", "pointer", "gamepad")
    pointer_mode: str = "relative-when-available"
    gamepad_selection: str = "first-connected"
    move_speed: float = 3.2
    run_multiplier: float = 2.0
    jump_speed: float = 3.6
    look_sensitivity: float = 0.0025
    bindings: tuple[ViewportControlBinding, ...] = DEFAULT_VIEWPORT_CONTROL_BINDINGS

    def __post_init__(self) -> None:
        if self.activation not in {"highlight", "focus", "explicit"}:
            raise ValueError("viewport control activation must be highlight, focus, or explicit")
        if (self.move_speed <= 0 or self.run_multiplier < 1 or self.jump_speed <= 0
                or self.look_sensitivity <= 0):
            raise ValueError("viewport movement and look rates must be positive")

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": "abstract-ui-viewport-controls-v0",
            "identity": self.identity,
            "actor": self.actor,
            "activation": self.activation,
            "release": list(self.release),
            "captures": list(self.captures),
            "pointer_mode": self.pointer_mode,
            "gamepad_selection": self.gamepad_selection,
            "move_speed": self.move_speed,
            "run_multiplier": self.run_multiplier,
            "jump_speed": self.jump_speed,
            "look_sensitivity": self.look_sensitivity,
            "bindings": [binding.to_data() for binding in self.bindings],
        }


@dataclass(frozen=True, slots=True)
class Viewport:
    """A bounded view onto a subject, independent of its presentation backend."""

    identity: str
    system_root: str
    subject: str
    bounds: UIBBox
    camera: ViewerCamera
    palette: UIPalette
    presentation_target: str = "abstract-surface"
    control_policy: ViewportControlPolicy | None = None
    device_monitor: DeviceMonitor | None = None
    dynamics_space: DynamicsSpace | None = None

    def to_data(self) -> dict[str, Any]:
        data = {
            "schema": ABSTRACT_UI_VIEWPORT_VERSION,
            "identity": self.identity,
            "kind": "viewport",
            "name": "Viewer camera",
            "system_root": self.system_root,
            "subject": self.subject,
            "bounds": {
                "x": self.bounds.x,
                "y": self.bounds.y,
                "width": self.bounds.width,
                "height": self.bounds.height,
                "coordinate_space": self.bounds.coordinate_space,
            },
            "camera": self.camera.to_data(),
            "presentation_target": self.presentation_target,
            "interaction": {"type": "inspect", "destination": self.identity},
            "dependencies": [
                {"relationship": "contained-by", "target": self.system_root},
                {"relationship": "views", "target": self.subject},
            ],
        }
        if self.control_policy is not None:
            data["control_policy"] = self.control_policy.to_data()
            if self.control_policy.actor is not None:
                data["dependencies"].append({
                    "relationship": "routes-controls-to",
                    "target": self.control_policy.actor,
                })
        if self.device_monitor is not None:
            data["device_monitor"] = self.device_monitor.to_data()
            data["dependencies"].append({
                "relationship": "monitors", "target": self.device_monitor.identity,
            })
        if self.dynamics_space is not None:
            data["dynamics_space"] = self.dynamics_space.to_data()
            data["dependencies"].append({
                "relationship": "presents-dynamics", "target": self.dynamics_space.identity,
            })
        return data


@dataclass(frozen=True, slots=True)
class ShaderViewport(Viewport):
    """A viewport whose presentation target is filled by a fragment chain."""

    fragment_chain: tuple[FragmentOperation, ...] = ()
    geometry_source: str = "document-geometry"
    backend_candidates: tuple[str, ...] = ("webgl2", "canvas2d")
    shader_choices: tuple[ShaderProgramChoice, ...] = ()
    default_shader: str = "living-map-default"
    presentation_target: str = "shader-surface"

    def to_data(self) -> dict[str, Any]:
        data = Viewport.to_data(self)
        data.update({
            "kind": "shader-viewport",
            "name": "First-person living data map",
            "geometry_source": self.geometry_source,
            "fragment_chain": [item.to_data() for item in self.fragment_chain],
            "backend_candidates": list(self.backend_candidates),
            "shader_choices": [item.to_data() for item in self.shader_choices],
            "default_shader": self.default_shader,
        })
        return data


def document_geometry(world: Mapping[str, Any]) -> dict[str, Any]:
    """Realize containment with deliberately nonlinear conceptual distance."""

    room_pitch = 3.2
    building_clearance = 3.5
    region_records: list[dict[str, Any]] = []
    cursor_x = 0.0
    maximum_depth = 1.0
    for region in world.get("regions", ()):
        building_records: list[dict[str, Any]] = []
        region_start = cursor_x
        region_depth = 4.0
        for building in region.get("buildings", ()):
            rooms = tuple(building.get("rooms", ()))
            columns = max((int(room["position"]["column"]) for room in rooms), default=0) + 1
            rows = max((int(room["position"]["row"]) for room in rooms), default=0) + 1
            width = columns * room_pitch + 1.4
            depth = rows * room_pitch + 1.4
            building_x = cursor_x + width * 0.5
            building_z = building_clearance + depth * 0.5
            building_boxes: list[dict[str, Any]] = [{
                "identity": building["identity"], "kind": "building",
                "label": building.get("name", building["identity"]),
                "parent_identity": region["identity"], "hierarchy_depth": 2,
                "center": [building_x, building_z], "half_extent": [width * 0.5, depth * 0.5],
                "height": 1.8, "floor_height": 0.035,
                "wall_thickness": 0.12, "radius": 8.0,
                "palette_role": "building-face", "wall_palette_role": "building-wall",
                "openings": [{
                    "identity": f"{building['identity']}/opening:entry",
                    "kind": "door", "side": "south", "offset": 0.0,
                    "width": 0.72, "height": 0.72,
                }],
            }]
            for room in rooms:
                position = room["position"]
                room_x = cursor_x + 0.7 + (int(position["column"]) + 0.5) * room_pitch
                room_z = building_clearance + 0.7 + (int(position["row"]) + 0.5) * room_pitch
                height = {
                    "method": 1.35, "property": 1.05, "field": 0.82,
                    "nested-class": 1.7,
                }.get(str(room.get("member_kind")), 0.95)
                building_boxes.append({
                    "identity": room["identity"], "kind": "room",
                    "label": room.get("name", room["identity"]),
                    "member_kind": room.get("member_kind"),
                    "parent_identity": building["identity"], "hierarchy_depth": 3,
                    "center": [room_x, room_z], "half_extent": [0.82, 0.82],
                    "height": height, "floor_height": 0.025,
                    "wall_thickness": 0.065, "radius": 8.0,
                    "palette_role": "room-face", "wall_palette_role": "room-wall",
                    "openings": [{
                        "identity": f"{room['identity']}/opening:door",
                        "kind": "door", "side": "south", "offset": 0.0,
                        "width": 0.42, "height": min(0.58, height * 0.78),
                    }],
                })
            building_records.extend(building_boxes)
            # Sibling separation grows faster than the depicted building span.
            cursor_x += width + building_clearance + math.pow(max(width, depth), 0.72)
            region_depth = max(region_depth, depth + building_clearance * 2.0)
        occupied_width = max(3.0, cursor_x - region_start - building_clearance)
        courtyard_margin = 4.0 + math.sqrt(occupied_width * region_depth) * 0.45
        region_width = occupied_width + courtyard_margin * 2.0
        region_total_depth = region_depth + courtyard_margin * 2.0
        region_center = [
            region_start + occupied_width * 0.5,
            region_total_depth * 0.5 - courtyard_margin * 0.25,
        ]
        courtyard = {
            "identity": region["identity"], "kind": "courtyard",
            "label": region.get("name", region["identity"]),
            "parent_identity": f"{world['identity']}/representation:global",
            "hierarchy_depth": 1,
            "center": region_center,
            "half_extent": [region_width * 0.5, region_total_depth * 0.5],
            "height": 3.2, "floor_height": 0.02,
            "wall_thickness": 0.20, "radius": 12.0,
            "palette_role": "courtyard-face", "wall_palette_role": "courtyard-wall",
            "openings": [{
                "identity": f"{region['identity']}/opening:gate",
                "kind": "gate", "side": "south", "offset": 0.0,
                "width": 1.1, "height": 0.82,
            }],
            "metaphor": "defensive class compound",
        }
        region_records.append({"courtyard": courtyard, "contents": building_records})
        cursor_x = region_start + region_width + 6.0 + math.pow(region_width, 0.78)
        maximum_depth = max(maximum_depth, region_total_depth)

    structural_boxes = [
        item
        for record in region_records
        for item in (record["courtyard"], *record["contents"])
    ]
    if structural_boxes:
        minimum_x = min(box["center"][0] - box["half_extent"][0] for box in structural_boxes)
        maximum_x = max(box["center"][0] + box["half_extent"][0] for box in structural_boxes)
        minimum_z = min(box["center"][1] - box["half_extent"][1] for box in structural_boxes)
        maximum_z = max(box["center"][1] + box["half_extent"][1] for box in structural_boxes)
    else:
        minimum_x, maximum_x, minimum_z, maximum_z = -2.0, 2.0, -2.0, 2.0
    occupied_span = max(maximum_x - minimum_x, maximum_z - minimum_z, 4.0)
    global_margin = 12.0 + math.pow(occupied_span, 1.18) * 0.55
    envelope_identity = f"{world['identity']}/representation:global"
    envelope = {
        "identity": envelope_identity, "kind": "world-envelope",
        "label": "Global scope / sky floor", "parent_identity": world["identity"],
        "hierarchy_depth": 0,
        "center": [(minimum_x + maximum_x) * 0.5, (minimum_z + maximum_z) * 0.5],
        "half_extent": [
            (maximum_x - minimum_x) * 0.5 + global_margin,
            (maximum_z - minimum_z) * 0.5 + global_margin,
        ],
        "height": 12.0, "floor_height": 0.018, "wall_thickness": 0.04,
        "radius": 0.0, "palette_role": "world-face",
        "wall_palette_role": "world-map-horizon",
        "openings": [],
        "skybox": {
            "floor": True, "horizon": True, "global_content": True,
            "always_visible": True,
            "semantic_role": "parent-world-map-boundary",
            "transition": "cross-horizon-to-parent-context-map",
            "layers": ["sky-gradient", "world-map-horizon", "return-portal"],
        },
    }
    boxes = [envelope, *structural_boxes]
    extent_x = envelope["half_extent"][0] * 2.0
    extent_z = envelope["half_extent"][1] * 2.0
    return {
        "schema": "abstract-ui-document-geometry-v0",
        "coordinate_space": "data-world",
        "boxes": boxes,
        "extent": [extent_x, extent_z],
        "origin": [
            envelope["center"][0] - envelope["half_extent"][0],
            envelope["center"][1] - envelope["half_extent"][1],
        ],
        "relationships": [
            "world-envelope-contains-courtyard",
            "courtyard-contains-building", "building-contains-room",
        ],
        "hierarchy_space": {
            "policy": "nonlinear-containment-distance-v0",
            "room_pitch": room_pitch,
            "building_clearance": building_clearance,
            "sibling_gap": "extent^0.72",
            "region_gap": "extent^0.78",
            "global_margin": "12+extent^1.18*0.55",
            "meaning": "conceptual-boundaries-expand-faster-than-local-form",
        },
        "representation_boundary": {
            "identity": f"{envelope_identity}/boundary:outside",
            "inside": envelope_identity,
            "crossing_operation": "switch-map-representation",
            "outside_representation": "parent-context-map",
            "entry_representation": "defensive-courtyard",
            "visualization": "persistent-outer-skybox-world-map-horizon",
        },
        "boundary_semantics": {
            "source": "dom-border",
            "meaning": "wall",
            "height_parameter": "boxes[].height",
            "thickness": 0.04,
            "floor": "mandatory-slab",
            "interior": "hollow",
            "ceiling": {"when": "height-at-absolute-maximum", "absolute_maximum": 4.0},
            "openings": "boxes[].openings",
            "opening_order": "document-order",
            "future_composition": "boundary-union-minus-openings",
        },
    }


__all__ = [
    "ABSTRACT_UI_VIEWPORT_VERSION", "DEFAULT_VIEWPORT_CONTROL_BINDINGS",
    "FragmentOperation", "ShaderProgramChoice", "ShaderViewport", "ViewerCamera", "Viewport",
    "ViewportControlBinding", "ViewportControlPolicy", "document_geometry",
]
