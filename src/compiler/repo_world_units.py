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

VIEWS (dev pages ticket 3/8, research case 810bfd48). A view is a courtyard that is no folder on disk: a unit whose
items come from a filter -- the files a project contract's scope selects (``project_view``). The index may hold
several scans (a project's roots may span a branch and a space; every identity carries its commit, so scans never
collide), and named views beside the folders:

    view:<account>/<project>/<view name>            a view that fits one window (owner, 2026-11-27)
    view:<account>/<project>/<view name>@<offset>   the window of one that does not: a treadmill, like a folder

A view lot keeps its real identity. Its home is the window that shows it in its real folder (``unit_of``), named in
the unit's ``homes``; a view has no way out of its own, and never becomes any item's home.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

UNIT_ITEMS = 1024
"""Items per unit unless the call says otherwise. A unit and the exits it names stay well inside one Living Data
cursor (4,096 places); this is a capacity, not a meaning, and a caller may ask for less."""

WINDOW_SEPARATOR = "@"
VIEW_PREFIX = "view:"


def window_address(region: str, offset: int) -> str:
    return f"{region}{WINDOW_SEPARATOR}{int(offset)}"


def view_address(account: str, project: str, name: str) -> str:
    """A view's address (owner, 2026-11-27): the project is part of it, so every project's views are distinct."""

    for part in (account, project, name):
        if not isinstance(part, str) or not part or "/" in part or WINDOW_SEPARATOR in part:
            raise ValueError(f"a view address part must be a plain name, not {part!r}")
    return f"{VIEW_PREFIX}{account}/{project}/{name}"


def split_address(address: str) -> tuple[str, int | None]:
    """(region identity, window offset or None)."""

    region, sep, offset = address.rpartition(WINDOW_SEPARATOR)
    if sep and offset.isdigit() and ("/dir:" in region or region.startswith(VIEW_PREFIX)):
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
    homes: Mapping[str, str] = None # a view's lots: item -> the window that shows it in its real folder

    def to_data(self) -> dict[str, Any]:
        return {"address": self.address, "region": self.region, "offset": self.offset,
                "length": self.length, "items": list(self.items), "way_out": self.way_out,
                "doors": {k: (v.to_data() if isinstance(v, Treadmill) else v) for k, v in self.doors.items()},
                "portals": [p.to_data() for p in self.portals], "inner_portals": list(self.inner_portals),
                "homes": dict(self.homes or {})}


def _records(scan: Any) -> tuple[Iterable, Iterable, Iterable]:
    """Regions, rooms and portal pairs from a RepoWorldScan or its ``to_data()`` form, as plain dicts; from several
    scans, all of their records (each identity carries its commit)."""

    if isinstance(scan, (list, tuple)):
        parts = [_records(one) for one in scan]
        return tuple(tuple(r for part in parts for r in part[k]) for k in range(3))
    if isinstance(scan, Mapping):
        return scan["regions"], scan["rooms"], scan["portals"]
    return ([r.to_data() for r in scan.regions], [r.to_data() for r in scan.rooms],
            [p.to_data() for p in scan.portals])


def _unique(records: Iterable, kind: str) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for record in records:
        held = found.setdefault(record["identity"], record)
        if held is not record and held != record:
            raise ValueError(f"two scans disagree about the {kind} {record['identity']!r}")
    return found


class UnitIndex:
    """Every unit of one repository scan, or of several, and the views named over them. Cheap to build: no layout,
    only the containment and the portals."""

    def __init__(self, scan: Any, *, capacity: int = UNIT_ITEMS, views: Mapping[str, Iterable[str]] | None = None):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive whole number of items")
        regions, rooms, portals = _records(scan)
        self.capacity = capacity
        self.regions = _unique(regions, "folder")
        self.rooms = _unique(rooms, "file")
        portals = tuple(_unique(portals, "portal pair").values())
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
        self.views: dict[str, tuple[str, ...]] = {}
        for address, items in (views or {}).items():
            if not isinstance(address, str) or not address.startswith(VIEW_PREFIX) or \
                    split_address(address)[1] is not None or WINDOW_SEPARATOR in address:
                raise ValueError(f"a view's address is {VIEW_PREFIX}<account>/<project>/<name>, not {address!r}")
            items = tuple(items)
            unknown = [item for item in items if item not in self.rooms]
            if unknown:
                raise KeyError(f"view {address!r} names files that are in no scan: {unknown[:5]}")
            if len(set(items)) != len(items):
                raise ValueError(f"view {address!r} names one file twice")
            self.views[address] = tuple(sorted(items, key=lambda item: (self.rooms[item]["path"], item)))

    def _courtyard(self, region: str) -> tuple[str, ...]:
        if region in self.views:
            return self.views[region]
        if region in self.regions:
            return self.items[region]
        raise KeyError(f"no folder or view {region!r} in this index")

    # ---- addresses -------------------------------------------------------------------------------------------------
    def door(self, region: str) -> Any:
        """What a folder's or a view's door opens on: its plain address when it fits, else its Treadmill."""

        length = len(self._courtyard(region))
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
        everything = self._courtyard(region)
        long = len(everything) > self.capacity
        if long != (offset is not None) or (offset is not None and not 0 <= offset < len(everything)):
            raise KeyError(f"{address!r} is not a unit: {region!r} "
                           + (f"is a treadmill of {len(everything)} items (offsets 0 to {len(everything) - 1})"
                              if long else "fits whole and has no treadmill"))
        items = everything if offset is None else everything[offset:offset + self.capacity]
        inside = set(items)
        view = region in self.views
        parent = None if view else self.regions[region]["parent"]
        doors = {item: self.door(item) for item in items if item in self.regions}
        homes = {item: self.unit_of(item) for item in items} if view else {}
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
                    None if parent is None else self.unit_of(region), doors, tuple(crossing), tuple(inner), homes)


def project_view(contract: Any, scans: Any, *, capacity: int = UNIT_ITEMS) -> tuple[UnitIndex, str]:
    """The index over a project's scans with its one view: every file the contract's scope admits
    (``ProjectContract.includes``), from any scan. Returns the index and the view's address."""

    index = UnitIndex(scans, capacity=capacity)
    address = view_address(contract.account, contract.project, contract.view)
    selected = [identity for identity, room in index.rooms.items() if contract.includes(room["path"])]
    return UnitIndex(scans, capacity=capacity, views={address: selected}), address


__all__ = ["Treadmill", "UNIT_ITEMS", "Unit", "UnitIndex", "UnitPortal", "VIEW_PREFIX", "WINDOW_SEPARATOR",
           "project_view", "split_address", "view_address", "window_address"]
