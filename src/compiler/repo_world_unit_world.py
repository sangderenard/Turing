"""One sub-addressed unit of a repository world, lifted into the living map's own world records.

This is gap D of docs/REPO_WORLD_MAP_DESIGN.md, per unit (§5a): ``build_unit_world(index, address)`` returns an
``AbstractUIWorld`` that ``abstract_ui_div_map.world_div_map_graph`` turns into the ``abstract-ui-div-map-v0`` model.
That one model is what the living page draws and what the text navigator walks (``living_data_query.GraphView``:
regions contain buildings, ``tracks`` are the passages), so a unit is walkable as text and drawable as a page from
the same record.

The records, following the owner's design (2026-11-10/12):
  * the unit is one courtyard (``AbstractUIRegion``) whose identity is the unit's address -- the folder, or the
    window ``…@<offset>`` of a folder too long to stand in whole; the world keeps the folder's own identity, so
    every window of one courtyard is the same island (``island_of``: ``world:<folder>``);
  * every item is a lot (``AbstractUIBuilding``) that looks like every other: one metaphor (``ruin``, the raw state
    every place starts in), no size, no rank; a file lot has no rooms yet (its interior is its own island, not
    built), a folder lot has a door: an island door into the subfolder, or a treadmill door when it is too long;
  * the courtyard's passage leads back out to the window of the parent that shows this folder's lot;
  * an import between two lots of the unit is a track; an import that leaves the unit is a track to the room at the
    other end, which the navigator can name but not enter until the unit holding it is loaded
    (``UnitIndex.unit_of``). Each such track carries ``label`` = the other end's unit address.
"""

from __future__ import annotations

from typing import Any, Callable

from . import abstract_ui_vocabulary as words
from .abstract_ui_doorways import declare_island_door, declare_island_passage, declare_treadmill_door
from .abstract_ui_introspection import (
    AbstractUIBuilding, AbstractUIRegion, AbstractUITrack, AbstractUIWorld, GridPosition, _grid_positions,
)
from .repo_world_units import Treadmill, UnitIndex, split_address

def island_of(region: str) -> str:
    """The world (island) identity of a folder's courtyard; every window of it is the same island. Distinct from
    the courtyard region's own identity (the unit address), which is the place a walker stands on."""

    return "world:" + region


LOT_METAPHOR = "ruin"
"""Every lot starts raw: someone else's structure, a little ruined (owner, 2026-11-10). Decoration is an occupant's."""


def _name(index: UnitIndex, item: str) -> tuple[str, str, str]:
    """(name, path, source kind) of a file or folder item."""

    if item in index.regions:
        region = index.regions[item]
        return region["name"] + "/", region["path"], "folder"
    room = index.rooms[item]
    return room["name"], room["path"], "file"


def build_unit_world(index: UnitIndex, address: str, *, href: Callable[[str], str] | None = None) -> AbstractUIWorld:
    """The living world of one unit. ``href(address)`` names the document that serves another unit, when there is one."""

    unit = index.unit(address)
    link = href or (lambda _address: None)
    positions = _grid_positions(len(unit.items))
    lots = []
    for item, position in zip(unit.items, positions):
        name, path, kind = _name(index, item)
        lots.append(AbstractUIBuilding(
            item, name, path, kind, LOT_METAPHOR, position, (),
            words.building(name, words.identity(item), source_kind=kind, metaphor=LOT_METAPHOR),
        ))
    folder = index.regions[unit.region]
    courtyard_name = (folder["path"] or ".") + ("" if unit.offset is None else f" @ {unit.offset}")
    courtyard = AbstractUIRegion(
        unit.address, courtyard_name, GridPosition(0, 0), tuple(lots),
        words.region(courtyard_name, *(lot.intention for lot in lots), words.identity(unit.address)),
    )
    tracks = []
    for pair_identity in unit.inner_portals:
        pair = next(p for p in index.portals if p["identity"] == pair_identity)
        tracks.append(AbstractUITrack(
            pair["source_room"], pair["target_room"], pair["reference_kind"], pair["specifier"],
            words.track(words.identity(pair["source_room"]), words.identity(pair["target_room"]),
                        relationship=pair["reference_kind"], label=pair["specifier"]),
        ))
    for portal in unit.portals:
        source, target = (portal.inside, portal.outside) if portal.direction == "out" else (portal.outside, portal.inside)
        tracks.append(AbstractUITrack(
            source, target, portal.reference_kind, portal.outside_unit,
            words.track(words.identity(source), words.identity(target),
                        relationship=portal.reference_kind, label=portal.outside_unit),
        ))
    # The doors as the text navigator sees them. The page crosses a door by its openings (declared below); the
    # navigator walks only dependencies and tracks, so each door is also a track from its lot to the address it
    # opens on (a treadmill opens at offset 0), and the way out is a track from the courtyard to the parent window.
    # The target is another unit: named here, entered by loading it (UnitIndex.unit).
    for sub, door in unit.doors.items():
        opens = door.address(0) if isinstance(door, Treadmill) else door
        tracks.append(AbstractUITrack(sub, opens, "door", opens,
                                      words.track(words.identity(sub), words.identity(opens), relationship="door", label=opens)))
    if unit.way_out is not None:
        tracks.append(AbstractUITrack(unit.address, unit.way_out, "way-out", unit.way_out,
                                      words.track(words.identity(unit.address), words.identity(unit.way_out),
                                                  relationship="way-out", label=unit.way_out)))
    world = AbstractUIWorld(
        island_of(unit.region), (courtyard,), tuple(tracks),
        words.world(courtyard.intention, words.identity(island_of(unit.region)),
                    words.describe(f"The courtyard of {courtyard_name}")),
    )
    for sub, door in unit.doors.items():
        if isinstance(door, Treadmill):
            world = declare_treadmill_door(world, sub, island_of(sub), length=door.length, window=door.window,
                                           href=link(door.address(0)))
        else:
            world = declare_island_door(world, sub, island_of(sub), href=link(door))
    if unit.way_out is not None:
        parent, _offset = split_address(unit.way_out)
        world = declare_island_passage(world, unit.address, lot=unit.region, outside=island_of(parent),
                                       href=link(unit.way_out))
    return world


def unit_graph(index: UnitIndex, address: str, *, href: Callable[[str], str] | None = None) -> dict[str, Any]:
    """The unit's ``abstract-ui-div-map-v0`` graph: what the navigator walks and the page draws."""

    from .abstract_ui_div_map import world_div_map_graph
    from .abstract_ui_wilderness import DIMENSIONS, position

    graph = world_div_map_graph(build_unit_world(index, address, href=href))
    # The map's place in the wilderness: the folder's island, so every window of one courtyard stands at one point.
    # The embedding is blank (abstract_ui_wilderness); a reader that does not know the field ignores it.
    graph["wilderness"] = {"identity": graph["identity"], "dimensions": DIMENSIONS,
                           "position": list(position(graph["identity"], DIMENSIONS))}
    return graph


__all__ = ["LOT_METAPHOR", "build_unit_world", "island_of", "unit_graph"]
