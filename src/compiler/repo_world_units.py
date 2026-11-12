"""Sub-addressed units: the walkable pieces of a repository world, and the treadmill doors that open them.

Owner's design (2026-11-12). A folder is an inner courtyard and its items (its files and its subfolders) are lots
in it. A root too big to stand in whole -- turing's 5,800 files, a folder of thousands -- is not laid out at once.
Its courtyard is ONE CONTIGUOUS SEQUENCE, so its door is not a dial of separate positions (that is the rooms'
contention dial, ``abstract_ui_doorways.declare_dial_door``): it is a TREADMILL. The door opens on a window of the
courtyard that scrolls continuously over an implied continuation which is never allocated, so dialling through
thousands of lots is no slower than scrolling a document. Only the window in view is a unit; every other window is
an offset nobody has asked for.

Addresses are the scan's own identities (``repo_world_scan.region_identity``), so they are stable for a commit and
never collide across commits:

    repo:<H>/dir:<path>             a folder that fits in one window: a plain door, one unit
    repo:<H>/dir:<path>@<offset>    the window of a folder that does not fit, starting at item ``offset``

A folder's items are its subfolders, then its files, each in path order; a window holds ``capacity`` consecutive
items from its offset (fewer at the end). Any offset is a valid window: there are no slices and no positions to
allocate. Nothing is ranked, weighed or sized: the outside of every lot looks the same, and where an item sits on
the treadmill follows only from its name.

What leaves a unit is named by address, never by loading the other side:
  * the way out: the window that shows this folder's own lot in its parent (none for the repository root);
  * a subfolder lot's door: the subfolder's plain address, or its ``Treadmill`` when it does not fit;
  * a portal pair whose other end is outside: that end's room and the window that shows it.

    index = UnitIndex(scan, capacity=UNIT_ITEMS)
    index.door(region_identity)        -> the plain address, or the Treadmill behind that folder's door
    index.unit_of(item_identity)       -> the window that shows a file or a subfolder lot (starting at it)
    index.unit(address)                -> Unit: its items, its way out, its doors and its portals out of the window
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

UNIT_ITEMS = 1024
"""Items per unit unless the call says otherwise. A unit and the exits it names stay well inside one Living Data
cursor (4,096 places); this is a capacity, not a meaning, and a caller may ask for less."""

WINDOW_SEPARATOR = "@"


def window_address(region: str, offset: int) -> str:
    return f"{region}{WINDOW_SEPARATOR}{int(offset)}"


def split_address(address: str) -> tuple[str, int | None]:
    """(region identity, window offset or None)."""

    region, sep, offset = address.rpartition(WINDOW_SEPARATOR)
    if sep and offset.isdigit() and "/dir:" in region:
        return region, int(offset)
    return address, None


@dataclass(frozen=True, slots=True)
class Treadmill:
    """The door of a courtyard that does not fit: one contiguous sequence of ``length`` items, seen ``window`` at a
    time from any offset. Nothing past the window in view is materialized."""

    region: str
    length: int
    window: int

    def address(self, offset: int) -> str:
        if not 0 <= int(offset) < self.length:
            raise KeyError(f"offset {offset} is off the treadmill of {self.region!r} (0 to {self.length - 1})")
        return window_address(self.region, offset)

    def to_data(self) -> dict[str, Any]:
        return {"region": self.region, "length": self.length, "window": self.window,
                "address": self.region + WINDOW_SEPARATOR + "{offset}"}


@dataclass(frozen=True, slots=True)
class UnitPortal:
    """A portal pair with one end inside the unit and the other outside it."""

    pair: str
    inside: str          # the room in this unit
    outside: str         # the room at the other end
    outside_unit: str    # the unit holding it
    direction: str       # "out" when the inside room references the outside one, "in" otherwise
    reference_kind: str

    def to_data(self) -> dict[str, Any]:
        return {"pair": self.pair, "inside": self.inside, "outside": self.outside, "outside_unit": self.outside_unit,
                "direction": self.direction, "reference_kind": self.reference_kind}


@dataclass(frozen=True, slots=True)
class Unit:
    address: str
    region: str                     # the folder whose courtyard this is
    offset: int | None              # where the window starts on the treadmill, None for a folder that fits
    length: int                     # how many items the whole courtyard holds
    items: tuple[str, ...]          # subfolder region identities and file room identities, in order
    way_out: str | None             # the window that shows this folder's lot in its parent
    doors: Mapping[str, Any]        # subfolder region identity -> its plain address or its Treadmill
    portals: tuple[UnitPortal, ...] # pairs crossing this unit's boundary
    inner_portals: tuple[str, ...]  # pairs with both ends in this unit

    def to_data(self) -> dict[str, Any]:
        return {"address": self.address, "region": self.region, "offset": self.offset,
                "length": self.length, "items": list(self.items), "way_out": self.way_out,
                "doors": {k: (v.to_data() if isinstance(v, Treadmill) else v) for k, v in self.doors.items()},
                "portals": [p.to_data() for p in self.portals], "inner_portals": list(self.inner_portals)}


def _records(scan: Any) -> tuple[Iterable, Iterable, Iterable]:
    """Regions, rooms and portal pairs from a RepoWorldScan or its ``to_data()`` form, as plain dicts."""

    if isinstance(scan, Mapping):
        return scan["regions"], scan["rooms"], scan["portals"]
    return ([r.to_data() for r in scan.regions], [r.to_data() for r in scan.rooms],
            [p.to_data() for p in scan.portals])


class UnitIndex:
    """Every unit of one repository scan. Cheap to build: no layout, only the containment and the portals."""

    def __init__(self, scan: Any, *, capacity: int = UNIT_ITEMS):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive whole number of items")
        regions, rooms, portals = _records(scan)
        self.capacity = capacity
        self.regions = {r["identity"]: r for r in regions}
        self.rooms = {r["identity"]: r for r in rooms}
        children: dict[str, list[dict]] = {k: [] for k in self.regions}
        files: dict[str, list[dict]] = {k: [] for k in self.regions}
        for region in self.regions.values():
            if region["parent"] is not None:
                children[region["parent"]].append(region)
        for room in self.rooms.values():
            files[room["region"]].append(room)
        self.items: dict[str, tuple[str, ...]] = {
            key: tuple(r["identity"] for r in sorted(children[key], key=lambda r: r["path"]))
                 + tuple(r["identity"] for r in sorted(files[key], key=lambda r: r["path"]))
            for key in self.regions}
        self._home: dict[str, str] = {}
        for region, items in self.items.items():
            long = len(items) > capacity
            for index, item in enumerate(items):
                self._home[item] = window_address(region, index) if long else region
        self.portals = tuple(portals)

    # ---- addresses -------------------------------------------------------------------------------------------------
    def door(self, region: str) -> Any:
        """What a folder's door opens on: its plain address when it fits, else its Treadmill."""

        if region not in self.regions:
            raise KeyError(f"no folder {region!r} in this scan")
        length = len(self.items[region])
        return region if length <= self.capacity else Treadmill(region, length, self.capacity)

    def unit_of(self, item: str) -> str:
        """The window that shows a file's or a subfolder's lot: the folder itself when it fits, else the window
        starting at the item. The repository root holds no lot; it is entered by its door."""

        try:
            return self._home[item]
        except KeyError:
            raise KeyError(f"{item!r} is not an item of any folder in this scan (the root is entered by its door)") from None

    # ---- one unit -------------------------------------------------------------------------------------------------
    def unit(self, address: str) -> Unit:
        region, offset = split_address(address)
        if region not in self.regions:
            raise KeyError(f"no folder {region!r} in this scan")
        everything = self.items[region]
        long = len(everything) > self.capacity
        if long != (offset is not None) or (offset is not None and not 0 <= offset < len(everything)):
            raise KeyError(f"{address!r} is not a unit: {region!r} "
                           + (f"is a treadmill of {len(everything)} items (offsets 0 to {len(everything) - 1})"
                              if long else "fits whole and has no treadmill"))
        items = everything if offset is None else everything[offset:offset + self.capacity]
        inside = set(items)
        parent = self.regions[region]["parent"]
        doors = {item: self.door(item) for item in items if item in self.regions}
        crossing, inner = [], []
        for pair in self.portals:
            source, target = pair["source_room"], pair["target_room"]
            ins, outs = source in inside, target in inside
            if ins and outs:
                inner.append(pair["identity"])
            elif ins or outs:
                here, there = (source, target) if ins else (target, source)
                crossing.append(UnitPortal(pair["identity"], here, there, self.unit_of(there),
                                           "out" if ins else "in", pair["reference_kind"]))
        return Unit(address, region, offset, len(everything), items,
                    None if parent is None else self.unit_of(region), doors, tuple(crossing), tuple(inner))


__all__ = ["Treadmill", "UNIT_ITEMS", "Unit", "UnitIndex", "UnitPortal", "WINDOW_SEPARATOR", "split_address",
           "window_address"]
