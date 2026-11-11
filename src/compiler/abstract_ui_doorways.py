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
