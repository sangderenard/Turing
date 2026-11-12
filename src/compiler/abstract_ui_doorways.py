"""Declare a door between a lot and an interior in one Living Data Map world.

A LOCAL portal is the game's existing tube ride through the same space. A NONLOCAL portal is a door: a material
sheet in a doorway, crossed at once. This module only DECLARES a door, as four openings on records the engine
already has (``AbstractUIBuilding.openings`` and ``AbstractUIRegion.openings``). The engine does the rest:
``document_geometry`` places each sheet in the doorway it already cut (``coincident_with``), the browser turns
``kind: "portal"`` openings into portal splats, links each declared pair to its own partner only, and moves a
body across a ``transition: "door"`` pair without a tube.

    lot (a building)                      interior (a region)
    entry doorway, facing out             its one passage (the gate), facing in
      enter:in  ----- pair "enter" ----->   enter:out     walk in the door  -> arrive just inside the passage
      exit:out  <---- pair "exit"  ------   exit:in       walk out the passage -> arrive outside the door

Nothing here lays out, sizes or decorates either side, and nothing says where the interior is: a declared pair is
certain at any distance. An interior kept as its own island, in its own file, is not built; when it is, the same
four openings describe its door.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .abstract_ui_introspection import AbstractUIWorld

DOOR_TRANSITION = "door"


def door_identity(lot: str, interior: str) -> str:
    return f"{lot}/door:{interior}"


def door_openings(lot: str, interior: str) -> dict[str, tuple[dict[str, Any], ...]]:
    """The four declared openings of one door, keyed by the record that carries them (``lot``, ``interior``)."""

    door = door_identity(lot, interior)
    entry, passage = f"{lot}/opening:entry", f"{interior}/opening:gate"

    def end(pair: str, role: str, anchor: str, **extra: Any) -> dict[str, Any]:
        return {"identity": f"{door}/{pair}:{role}", "kind": "portal", "port_role": role,
                "pair": f"{door}/{pair}", "transition": DOOR_TRANSITION, "coincident_with": anchor, **extra}

    return {
        "lot": (end("enter", "in", entry), end("exit", "out", entry)),
        "interior": (end("enter", "out", passage, facing="inward"), end("exit", "in", passage, facing="inward")),
    }


def declare_door(world: AbstractUIWorld, lot: str, interior: str) -> AbstractUIWorld:
    """Return ``world`` with a door declared from the building ``lot`` into the region ``interior``.

    Raises ``KeyError`` when either identity is not in the world and ``ValueError`` when the interior is the
    region the lot stands in, or when this door is already declared.
    """

    regions = {region.identity: region for region in world.regions}
    if interior not in regions:
        raise KeyError(f"no region {interior!r} in world {world.identity!r}")
    home = next((region for region in world.regions
                 if any(building.identity == lot for building in region.buildings)), None)
    if home is None:
        raise KeyError(f"no building {lot!r} in world {world.identity!r}")
    if home.identity == interior:
        raise ValueError(f"{lot!r} stands in {interior!r}; a door leads somewhere else")
    openings = door_openings(lot, interior)
    declared = {opening["identity"] for opening in openings["lot"] + openings["interior"]}
    for record in (*(b for b in home.buildings if b.identity == lot), regions[interior]):
        if declared & {opening.get("identity") for opening in record.openings}:
            raise ValueError(f"the door {door_identity(lot, interior)!r} is already declared")

    def with_door(region):
        if region.identity == home.identity:
            region = replace(region, buildings=tuple(
                replace(building, openings=(*building.openings, *openings["lot"]))
                if building.identity == lot else building for building in region.buildings))
        if region.identity == interior:
            region = replace(region, openings=(*region.openings, *openings["interior"]))
        return region

    return replace(world, regions=tuple(with_door(region) for region in world.regions))


# ---- a door between ISLANDS: two worlds, each in its own document -------------------------------------------
#
# An island is an independent world. Its door is the same four ends, split across two worlds: the lot's two
# ends are declared in the outside world, the passage's two ends in the island's own world. Each `in` end has no
# partner in its own world; it names the island its partner lives in (``island``) and, when known, the document
# to open (``href``). The browser leaves through such a sheet (`leaveThroughIslandDoor`) and arrives at the
# `out` end that carries the same pair in the other world (`completeIslandArrival`).


def _island_end(door: str, pair: str, role: str, anchor: str, island: str | None, href: str | None,
                **extra: Any) -> dict[str, Any]:
    opening = {"identity": f"{door}/{pair}:{role}", "kind": "portal", "port_role": role,
               "pair": f"{door}/{pair}", "transition": DOOR_TRANSITION, "coincident_with": anchor, **extra}
    if role == "in":
        opening["island"] = island
        if href:
            opening["href"] = href
    return opening


def declare_island_door(world: AbstractUIWorld, lot: str, island: str, *, href: str | None = None) -> AbstractUIWorld:
    """Declare, in the OUTSIDE world, the lot's side of a door into the island ``island`` (another world)."""

    home = next((region for region in world.regions
                 if any(building.identity == lot for building in region.buildings)), None)
    if home is None:
        raise KeyError(f"no building {lot!r} in world {world.identity!r}")
    if island == world.identity:
        raise ValueError("an island door leads to another world")
    door, entry = door_identity(lot, island), f"{lot}/opening:entry"
    ends = (_island_end(door, "enter", "in", entry, island, href),
            _island_end(door, "exit", "out", entry, None, None))
    building = next(b for b in home.buildings if b.identity == lot)
    if {end["identity"] for end in ends} & {opening.get("identity") for opening in building.openings}:
        raise ValueError(f"the door {door!r} is already declared")
    return replace(world, regions=tuple(
        replace(region, buildings=tuple(
            replace(b, openings=(*b.openings, *ends)) if b.identity == lot else b for b in region.buildings))
        if region.identity == home.identity else region for region in world.regions))


def declare_island_passage(island_world: AbstractUIWorld, region: str, *, lot: str, outside: str,
                           href: str | None = None) -> AbstractUIWorld:
    """Declare, in the ISLAND's own world, its side of the door from ``lot`` (a building of the world ``outside``).

    ``region`` is the island's region whose gate is its one passage. The pair identities are the ones
    ``declare_island_door(outside_world, lot, island_world.identity)`` uses, so the two halves meet.
    """

    regions = {candidate.identity: candidate for candidate in island_world.regions}
    if region not in regions:
        raise KeyError(f"no region {region!r} in world {island_world.identity!r}")
    if outside == island_world.identity:
        raise ValueError("an island's passage leads to another world")
    door, gate = door_identity(lot, island_world.identity), f"{region}/opening:gate"
    ends = (_island_end(door, "enter", "out", gate, None, None, facing="inward"),
            _island_end(door, "exit", "in", gate, outside, href, facing="inward"))
    if {end["identity"] for end in ends} & {opening.get("identity") for opening in regions[region].openings}:
        raise ValueError(f"the passage of {door!r} is already declared")
    return replace(island_world, regions=tuple(
        replace(candidate, openings=(*candidate.openings, *ends)) if candidate.identity == region else candidate
        for candidate in island_world.regions))


# ---- a DIAL door: one doorway, several islands behind it ----------------------------------------------------
#
# Owner's design (2026-11-12): a courtyard too big to stand in whole opens with a dial door, like a room's
# contention dial. The lot keeps ONE doorway; beside it, a dial picks which island the doorway leads into. Each
# position is an ordinary island door (the same four ends, ``declare_island_door``), all coincident with the
# one entry, and every `in` end of the lot's side names the dial it belongs to (``dial``), its place on it
# (``dial_position``) and what the dial shows there (``dial_label``). The browser crosses the end of the position
# the dial is turned to; the rest of the dial is just records. Each island declares its own half with
# ``declare_island_passage`` as before, so the way back out needs no dial. Positions are the call's order.
# For a repository folder the positions are its sub-addressed units (``repo_world_units.UnitIndex.door``).


def dial_identity(lot: str) -> str:
    return f"{lot}/dial"


def declare_dial_door(world: AbstractUIWorld, lot: str, positions: Any) -> AbstractUIWorld:
    """Declare, in the OUTSIDE world, a dial door from ``lot`` into several islands.

    ``positions`` is a sequence of mappings ``{"island": <world identity>, "href": <document>|None, "label": str}``.
    One position is an ordinary island door, still marked with its dial so a reader sees one shape.
    """

    positions = list(positions)
    if not positions:
        raise ValueError("a dial door needs at least one position")
    islands = [p["island"] for p in positions]
    if len(set(islands)) != len(islands):
        raise ValueError("each dial position leads to a different island")
    dial = dial_identity(lot)
    for index, position in enumerate(positions):
        world = declare_island_door(world, lot, position["island"], href=position.get("href"))
        door = door_identity(lot, position["island"])

        def mark(opening, index=index, position=position, door=door):
            if opening.get("pair", "").startswith(door + "/"):
                return {**opening, "dial": dial, "dial_position": index, "dial_positions": len(positions),
                        "dial_label": str(position.get("label") or position["island"])}
            return opening

        world = replace(world, regions=tuple(
            replace(region, buildings=tuple(
                replace(b, openings=tuple(mark(o) for o in b.openings)) if b.identity == lot else b
                for b in region.buildings))
            for region in world.regions))
    return world


# ---- a TREADMILL door: a courtyard too long to stand in whole -----------------------------------------------
#
# Owner's design (2026-11-12): a courtyard is one contiguous separation, so it is not dialled in positions like a
# room's contention dial. Its door opens on a window that scrolls continuously over an implied continuation that
# is never allocated: a treadmill, as fast to run through as scrolling a document. It is ONE island door (one
# pair, one island: the courtyard itself, whose identity every window keeps), and the lot's ends carry the
# treadmill: how many items the courtyard holds (``length``), how many one window shows (``window``) and the
# offset the walker arrives at (``offset``, 0 unless the call says). The browser scrolls the offset and asks for
# the window there; the island declares its half with ``declare_island_passage`` as for any island door. For a
# repository folder ``length``/``window`` come from ``repo_world_units.UnitIndex.door`` (a ``Treadmill``).


def declare_treadmill_door(world: AbstractUIWorld, lot: str, island: str, *, length: int, window: int,
                           offset: int = 0, href: str | None = None) -> AbstractUIWorld:
    """Declare, in the OUTSIDE world, a treadmill door from ``lot`` into the courtyard island ``island``."""

    if not (isinstance(length, int) and isinstance(window, int) and isinstance(offset, int)):
        raise ValueError("length, window and offset are whole numbers of items")
    if window < 1 or length <= window:
        raise ValueError("a treadmill is for a courtyard longer than one window; a shorter one has a plain door")
    if not 0 <= offset < length:
        raise ValueError(f"offset {offset} is off the treadmill (0 to {length - 1})")
    world = declare_island_door(world, lot, island, href=href)
    door = door_identity(lot, island)
    treadmill = {"length": length, "window": window, "offset": offset}

    def mark(opening):
        return {**opening, "treadmill": dict(treadmill)} if opening.get("pair", "").startswith(door + "/") else opening

    return replace(world, regions=tuple(
        replace(region, buildings=tuple(
            replace(b, openings=tuple(mark(o) for o in b.openings)) if b.identity == lot else b
            for b in region.buildings))
        for region in world.regions))
