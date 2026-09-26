"""Tile-authored building graphs for the Living Data Map build experience.

The builder publishes ordinary world objects and graph records.  It owns no
physics loop: rigid, atmosphere, thermal, electrical, damage, and other engines
may select the resulting identities through their existing graph adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .abstract_ui_world import WorldObject


HOUSE_BUILDER_VERSION = "abstract-ui-house-builder-v0"
WALL_SIDES = ("north", "east", "south", "west")
SERVICE_BOUNDARIES = ("pre-breaker", "post-breaker")


@dataclass(frozen=True, slots=True)
class TileRectangle:
    x: int
    z: int
    width: int
    depth: int
    level: int = 0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.depth <= 0:
            raise ValueError("a room needs a positive tile footprint")


@dataclass(frozen=True, slots=True)
class RoomOpening:
    identity: str
    side: str
    offset_tiles: float
    width_tiles: float
    height_m: float
    kind: str = "door"
    sill_height_m: float = 0.0
    connects_to: str = "outside"
    seal_model: str = "semi-sealed"
    closed_leakage_area_m2: float | None = None

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("an opening needs an identity")
        if self.side not in WALL_SIDES:
            raise ValueError(f"unknown wall side {self.side!r}")
        if self.kind not in {"door", "gate", "window"}:
            raise ValueError(f"unknown architectural opening {self.kind!r}")
        if self.width_tiles <= 0.0 or self.height_m <= 0.0:
            raise ValueError("an opening needs positive width and height")
        if self.sill_height_m < 0.0:
            raise ValueError("an opening sill cannot be below its room floor")
        if self.kind == "door" and self.seal_model != "semi-sealed":
            raise ValueError("ordinary room doors use the semi-sealed portal contract")


@dataclass(frozen=True, slots=True)
class ElectricalServiceBoundary:
    source_identity: str
    side: str = "west"
    offset_tiles: float = 0.0
    height_m: float = 1.4
    delivery: str = "pre-breaker"
    panel_identity: str | None = None
    upstream_breaker_identity: str | None = None
    service: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_identity:
            raise ValueError("an electrical service needs a source identity")
        if self.side not in WALL_SIDES:
            raise ValueError(f"unknown service wall {self.side!r}")
        if self.delivery not in SERVICE_BOUNDARIES:
            raise ValueError(f"unknown service boundary {self.delivery!r}")
        if self.delivery == "post-breaker" and not self.upstream_breaker_identity:
            raise ValueError("post-breaker service requires its upstream breaker")

    @property
    def branch_source_identity(self) -> str | None:
        if self.delivery == "post-breaker":
            return self.upstream_breaker_identity
        return self.panel_identity


@dataclass(frozen=True, slots=True)
class WallOutlet:
    identity: str
    side: str
    offset_tiles: float
    height_m: float = 0.40
    service: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("an outlet needs an identity")
        if self.side not in WALL_SIDES:
            raise ValueError(f"unknown outlet wall {self.side!r}")
        if self.height_m < 0.0:
            raise ValueError("an outlet cannot be below its room floor")


@dataclass(frozen=True, slots=True)
class RoomSpec:
    identity: str
    parent: str
    tiles: TileRectangle
    tile_size_m: float = 1.0
    wall_height_m: float = 2.4
    wall_thickness_m: float = 0.14
    floor_thickness_m: float = 0.16
    foundation_height_m: float = 0.20
    openings: tuple[RoomOpening, ...] = ()
    electrical: ElectricalServiceBoundary | None = None
    outlets: tuple[WallOutlet, ...] = ()

    def __post_init__(self) -> None:
        if not self.identity or not self.parent:
            raise ValueError("a room needs identity and parent")
        if min(self.tile_size_m, self.wall_height_m, self.wall_thickness_m,
               self.floor_thickness_m) <= 0.0:
            raise ValueError("room dimensions must be positive")
        if self.foundation_height_m < 0.0:
            raise ValueError("foundation height cannot be negative")
        if self.outlets and self.electrical is None:
            raise ValueError("placed outlets require an electrical service boundary")
        if self.outlets and self.electrical.branch_source_identity is None:
            raise ValueError(
                "pre-breaker service requires a real panel before outlets can be wired"
            )


@dataclass(frozen=True, slots=True)
class RoomGraph:
    identity: str
    nodes: tuple[Mapping[str, Any], ...]
    edges: tuple[Mapping[str, Any], ...]
    world_objects: tuple[WorldObject, ...]
    geometry_boxes: tuple[Mapping[str, Any], ...]
    requirements: tuple[Mapping[str, Any], ...] = ()

    def to_data(self) -> dict[str, Any]:
        return {
            "schema": HOUSE_BUILDER_VERSION,
            "identity": self.identity,
            "nodes": [dict(node) for node in self.nodes],
            "edges": [dict(edge) for edge in self.edges],
            "world_objects": [item.to_data() for item in self.world_objects],
            "geometry_boxes": [dict(box) for box in self.geometry_boxes],
            "requirements": [dict(item) for item in self.requirements],
        }


def _position_on_wall(spec: RoomSpec, side: str, offset_tiles: float,
                      height_m: float) -> tuple[float, float, float]:
    width_m = spec.tiles.width * spec.tile_size_m
    depth_m = spec.tiles.depth * spec.tile_size_m
    center_x = (spec.tiles.x + spec.tiles.width / 2.0) * spec.tile_size_m
    center_z = (spec.tiles.z + spec.tiles.depth / 2.0) * spec.tile_size_m
    along = offset_tiles * spec.tile_size_m
    floor_y = spec.tiles.level * spec.wall_height_m + spec.foundation_height_m
    if side == "north":
        return center_x + along, floor_y + height_m, center_z + depth_m / 2.0
    if side == "south":
        return center_x + along, floor_y + height_m, center_z - depth_m / 2.0
    if side == "east":
        return center_x + width_m / 2.0, floor_y + height_m, center_z + along
    return center_x - width_m / 2.0, floor_y + height_m, center_z + along


def build_room_graph(spec: RoomSpec) -> RoomGraph:
    """Build one room as an ordinary object graph and world realization."""

    width_m = spec.tiles.width * spec.tile_size_m
    depth_m = spec.tiles.depth * spec.tile_size_m
    center_x = (spec.tiles.x + spec.tiles.width / 2.0) * spec.tile_size_m
    center_z = (spec.tiles.z + spec.tiles.depth / 2.0) * spec.tile_size_m
    base_y = spec.tiles.level * spec.wall_height_m
    floor_y = base_y + spec.foundation_height_m
    half_x, half_z = width_m / 2.0, depth_m / 2.0
    room = spec.identity
    foundation = f"{room}.foundation"
    floor = f"{room}.floor"
    volume = f"{room}.air-volume"
    wall_ids = {side: f"{room}.wall.{side}" for side in WALL_SIDES}

    nodes: list[dict[str, Any]] = [{
        "identity": room, "kind": "room", "parent": spec.parent,
        "tile_rectangle": {
            "x": spec.tiles.x, "z": spec.tiles.z,
            "width": spec.tiles.width, "depth": spec.tiles.depth,
            "level": spec.tiles.level, "tile_size_m": spec.tile_size_m,
        },
    }, {
        "identity": foundation, "kind": "foundation", "parent": room,
        "reference_position": [center_x, base_y + spec.foundation_height_m / 2.0, center_z],
        "body_half_extent_m": [half_x, spec.foundation_height_m / 2.0, half_z],
        "body": "static-rigid", "material": "unresolved-foundation-material",
    }, {
        "identity": floor, "kind": "floor-slab", "parent": room,
        "reference_position": [center_x, floor_y + spec.floor_thickness_m / 2.0, center_z],
        "body_half_extent_m": [half_x, spec.floor_thickness_m / 2.0, half_z],
        "body": "static-rigid", "material": "unresolved-floor-material",
    }, {
        "identity": volume, "kind": "enclosed-air-volume", "parent": room,
        "reference_position": [center_x, floor_y + spec.wall_height_m / 2.0, center_z],
        "volume_m3": width_m * depth_m * spec.wall_height_m,
        "bounds_half_extent_m": [half_x, spec.wall_height_m / 2.0, half_z],
        "state_owner": "atmosphere-engine", "spatial_model": "room-volume",
    }]

    for side in WALL_SIDES:
        north_south = side in {"north", "south"}
        wall_half = [
            half_x if north_south else spec.wall_thickness_m / 2.0,
            spec.wall_height_m / 2.0,
            spec.wall_thickness_m / 2.0 if north_south else half_z,
        ]
        wall_position = _position_on_wall(spec, side, 0.0, spec.wall_height_m / 2.0)
        nodes.append({
            "identity": wall_ids[side], "kind": "wall-body", "parent": room,
            "side": side, "reference_position": list(wall_position),
            "body_half_extent_m": wall_half, "body": "static-rigid",
            "material": "unresolved-wall-assembly",
        })

    edges: list[dict[str, Any]] = []
    for child in (foundation, floor, volume, *wall_ids.values()):
        edges.append({
            "identity": f"{room}.contains.{child.rsplit('.', 1)[-1]}",
            "a": room, "b": child, "relationship": "contains",
        })
    edges.append({
        "identity": f"{floor}.supported-by-foundation", "a": foundation,
        "b": floor, "relationship": "supports",
    })
    for wall in wall_ids.values():
        edges.append({
            "identity": f"{wall}.fixed-to-floor", "a": floor, "b": wall,
            "relationship": "fixed-joint",
        })
        edges.append({
            "identity": f"{wall}.bounds-air", "a": wall, "b": volume,
            "relationship": "bounds-volume",
        })

    opening_data: list[dict[str, Any]] = []
    for opening in spec.openings:
        wall_length_tiles = spec.tiles.width if opening.side in {"north", "south"} else spec.tiles.depth
        if abs(opening.offset_tiles) + opening.width_tiles / 2.0 > wall_length_tiles / 2.0:
            raise ValueError(f"{opening.identity}: opening falls outside {opening.side} wall")
        if opening.sill_height_m + opening.height_m > spec.wall_height_m:
            raise ValueError(f"{opening.identity}: opening exceeds wall height")
        record = {
            "identity": opening.identity, "kind": opening.kind,
            "parent": wall_ids[opening.side], "side": opening.side,
            "offset": opening.offset_tiles * spec.tile_size_m,
            "width": opening.width_tiles * spec.tile_size_m,
            "bottom": opening.sill_height_m, "height": opening.height_m,
            "connects": [volume, opening.connects_to],
            "seal_model": opening.seal_model,
            "closed_leakage_area_m2": opening.closed_leakage_area_m2,
            "leakage_calibration": (
                "authored" if opening.closed_leakage_area_m2 is not None else "unresolved"
            ),
        }
        opening_data.append(record)
        nodes.append(record)
        edges.extend(({
            "identity": f"{opening.identity}.cuts-wall", "a": opening.identity,
            "b": wall_ids[opening.side], "relationship": "boundary-opening",
        }, {
            "identity": f"{opening.identity}.connects-volume", "a": volume,
            "b": opening.connects_to, "relationship": "portal-adjacency",
            "seal_model": opening.seal_model,
        }))

    requirements: list[dict[str, Any]] = []
    if spec.electrical is not None:
        service_id = f"{room}.electrical-service"
        service_position = _position_on_wall(
            spec, spec.electrical.side, spec.electrical.offset_tiles,
            spec.electrical.height_m,
        )
        nodes.append({
            "identity": service_id, "kind": "electrical-service-hookup",
            "parent": wall_ids[spec.electrical.side],
            "reference_position": list(service_position),
            "delivery_boundary": spec.electrical.delivery,
            "source_identity": spec.electrical.source_identity,
            "service": dict(spec.electrical.service),
        })
        edges.append({
            "identity": f"{service_id}.supplied-by", "a": spec.electrical.source_identity,
            "b": service_id, "relationship": "electrical-service-boundary",
            "delivery": spec.electrical.delivery,
        })
        if spec.electrical.delivery == "pre-breaker":
            requirements.append({
                "identity": f"{room}.requires-breaker-panel",
                "kind": "required-object",
                "required_kind": "electrical-distribution-panel",
                "satisfied_by": spec.electrical.panel_identity,
                "before": "branch-circuit-or-outlet-energization",
            })

        branch_source = spec.electrical.branch_source_identity
        for outlet in spec.outlets:
            outlet_position = _position_on_wall(
                spec, outlet.side, outlet.offset_tiles, outlet.height_m,
            )
            nodes.append({
                "identity": outlet.identity, "kind": "electrical-outlet-box",
                "parent": wall_ids[outlet.side], "side": outlet.side,
                "reference_position": list(outlet_position),
                "service": dict(outlet.service or spec.electrical.service),
            })
            edges.append({
                "identity": f"{outlet.identity}.conduit", "a": branch_source,
                "b": outlet.identity, "relationship": "electrical-conduit",
                "host_wall": wall_ids[outlet.side], "concealed_in_wall": True,
                "conductors_explicit": True,
            })

    semantic_parts = tuple({
        "identity": identity, "role": role, "material_role": material,
    } for identity, role, material in (
        (foundation, "foundation", "foundation"),
        (floor, "floor", "floor"),
        *((wall_ids[side], f"wall-{side}", "wall") for side in WALL_SIDES),
        (volume, "enclosed-volume", "void"),
    )) + tuple({
        "identity": opening["identity"], "role": "opening",
        "opening_kind": opening["kind"], "side": opening["side"],
        "material_role": "void",
    } for opening in opening_data)

    room_object = WorldObject(
        identity=room, kind="room", parent=spec.parent, label=room.rsplit(".", 1)[-1],
        transform={
            "position": [center_x, floor_y, center_z],
            "coordinate_space": "tile-addressed-world",
            "tile_rectangle": dict(nodes[0]["tile_rectangle"]),
        },
        form={
            "recipe": "boundary-floor-with-openings",
            "half_extent": [half_x, half_z], "height": spec.wall_height_m,
            "floor_height": spec.floor_thickness_m,
            "wall_thickness": spec.wall_thickness_m,
            "foundation_height": spec.foundation_height_m,
            "openings": opening_data,
        },
        material_bindings={
            "foundation": "foundation", "floor": "floor", "wall": "wall",
        },
        capabilities=(
            "inspect", "move", "resize-by-tile", "set-wall-height",
            "set-foundation-height", "receive-openings", "receive-wall-services",
        ),
        semantic_parts=semantic_parts,
        physics={
            "collision_authority": "world-physics", "body": "static-building",
            "enclosed_volume": volume, "enabled": True,
        },
        persistence={"authority": "living-document", "revision": 0},
        extensions={"house_builder.room_graph": f"{room}.graph"},
    )
    foundation_object = WorldObject(
        identity=foundation, kind="foundation", parent=room,
        label=f"{room_object.label} foundation",
        transform={"position": [center_x, base_y, center_z]},
        form={
            "recipe": "solid-box", "half_extent": [half_x, half_z],
            "height": spec.foundation_height_m,
        },
        material_bindings={"body": "foundation"},
        capabilities=("inspect", "publish-mesh"),
        physics={"body": "static", "enabled": spec.foundation_height_m > 0.0},
        persistence={"authority": "living-document", "revision": 0},
    )
    volume_object = WorldObject(
        identity=volume, kind="enclosed-air-volume", parent=room,
        label=f"{room_object.label} air volume",
        transform={"position": [center_x, floor_y, center_z]},
        form={
            "recipe": "volume-box", "half_extent": [half_x, half_z],
            "height": spec.wall_height_m,
        },
        capabilities=("inspect", "exchange-through-openings"),
        physics={"state_owner": "atmosphere-engine", "collision": False},
        persistence={"authority": "living-document", "revision": 0},
    )

    room_box = {
        "identity": room, "kind": "room", "label": room_object.label,
        "parent_identity": spec.parent, "center": [center_x, center_z],
        "half_extent": [half_x, half_z], "height": spec.wall_height_m,
        "floor_height": spec.floor_thickness_m,
        "wall_thickness": spec.wall_thickness_m,
        "palette_role": "room-face", "wall_palette_role": "room-wall",
        "geometry_mode": "boundary", "openings": opening_data,
        "placement": {
            "custody": "placed", "elevation": floor_y,
            "yaw_degrees": 0.0,
        },
        "tile_rectangle": dict(nodes[0]["tile_rectangle"]),
        "room_graph_identity": f"{room}.graph",
    }
    foundation_box = {
        "identity": foundation, "kind": "foundation", "label": foundation_object.label,
        "parent_identity": room, "center": [center_x, center_z],
        "half_extent": [half_x, half_z], "height": spec.foundation_height_m,
        "floor_height": spec.foundation_height_m,
        "wall_thickness": spec.wall_thickness_m,
        "palette_role": "artifact-source", "wall_palette_role": "artifact-source",
        "geometry_mode": "solid", "openings": [],
        "placement": {"custody": "placed", "elevation": base_y, "yaw_degrees": 0.0},
        "physics": {"body": "static", "enabled": spec.foundation_height_m > 0.0},
    }
    return RoomGraph(
        f"{room}.graph", tuple(nodes), tuple(edges),
        (room_object, foundation_object, volume_object),
        (room_box, foundation_box), tuple(requirements),
    )


def house_builder_model(root: str) -> dict[str, Any]:
    """Publish editor defaults without claiming they are physical constants."""

    return {
        "schema": HOUSE_BUILDER_VERSION,
        "identity": f"{root}/house-builder",
        "authority": "living-document-world-object-graph",
        "interaction_host": "living-data-map-placement-tool",
        "tile_size_m": 1.0,
        "defaults": {
            "width_tiles": 4, "depth_tiles": 4,
            "wall_height_m": 2.4, "wall_thickness_m": 0.14,
            "floor_thickness_m": 0.16, "foundation_height_m": 0.20,
            "service_delivery": "pre-breaker",
        },
        "wall_height_presets_m": [2.4, 3.0, 4.0],
        "service_delivery_options": list(SERVICE_BOUNDARIES),
        "graph": {"nodes": [], "edges": [], "requirements": []},
        "rooms": [],
        "solver_binding": "declarative-unbound",
    }


__all__ = [
    "HOUSE_BUILDER_VERSION", "SERVICE_BOUNDARIES", "WALL_SIDES",
    "ElectricalServiceBoundary", "RoomGraph", "RoomOpening", "RoomSpec",
    "TileRectangle", "WallOutlet", "build_room_graph", "house_builder_model",
]
