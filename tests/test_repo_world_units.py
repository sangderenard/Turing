"""Sub-addressed units of a repository world: a courtyard too long to stand in whole is a treadmill of windows; a
room's contention is a dial of separate positions."""

import pytest

from src.compiler.abstract_ui_div_map import world_div_map_model
from src.compiler.abstract_ui_doorways import (
    declare_dial_door, declare_island_door, declare_treadmill_door, dial_identity, door_identity,
)
from src.compiler.abstract_ui_introspection import build_introspective_world
from src.compiler.abstract_ui_viewports import document_geometry
from src.compiler.repo_world_scan import region_identity, room_identity, scan_files
from src.compiler.repo_world_units import Treadmill, UnitIndex, split_address, window_address

H = "a" * 40
FILES = {
    "README.md": b"hi\n",
    "pkg/__init__.py": b"",
    "pkg/a.py": b"from pkg import b\n",
    "pkg/b.py": b"",
    "pkg/c.py": b"",
    "pkg/d.py": b"import tools.t\n",
    "pkg/e.py": b"",
    "pkg/sub/x.py": b"",
    "tools/t.py": b"",
}
PKG = region_identity(H, "pkg")
# pkg holds sub/, then __init__.py, a.py ... e.py, in path order: 7 items
PKG_ITEMS = (region_identity(H, "pkg/sub"),) + tuple(room_identity(H, f"pkg/{n}") for n in
                                                      ("__init__.py", "a.py", "b.py", "c.py", "d.py", "e.py"))


def _scan():
    return scan_files(H, None, sorted(FILES), FILES.__getitem__)


def test_a_folder_that_fits_is_one_unit_behind_a_plain_door():
    index = UnitIndex(_scan(), capacity=100)
    root = region_identity(H, ".")
    assert index.door(root) == root
    unit = index.unit(root)
    assert unit.offset is None and unit.length == 3 and unit.way_out is None
    assert unit.items == (PKG, region_identity(H, "tools"), room_identity(H, "README.md"))
    assert unit.doors == {PKG: PKG, region_identity(H, "tools"): region_identity(H, "tools")}


def test_a_folder_that_does_not_fit_is_a_treadmill_any_offset_of_which_is_a_window():
    index = UnitIndex(_scan(), capacity=3)
    assert index.door(PKG) == Treadmill(PKG, 7, 3)
    assert index.door(PKG).to_data()["address"] == PKG + "@{offset}"
    # every offset is a window: consecutive items, no slices, no positions
    for offset in range(7):
        assert index.unit(window_address(PKG, offset)).items == PKG_ITEMS[offset:offset + 3]
    unit = index.unit(window_address(PKG, 2))
    assert (unit.offset, unit.length, unit.way_out) == (2, 7, region_identity(H, "."))
    # the window that shows an item starts at it; the parent's door to pkg is the treadmill
    assert index.unit_of(room_identity(H, "pkg/d.py")) == window_address(PKG, 5)
    assert index.unit(region_identity(H, ".")).doors[PKG] == Treadmill(PKG, 7, 3)
    assert split_address(window_address(PKG, 6)) == (PKG, 6) and split_address(PKG) == (PKG, None)
    with pytest.raises(KeyError):
        index.unit(PKG)                        # a treadmill is only ever seen through a window
    with pytest.raises(KeyError):
        index.unit(window_address(PKG, 7))     # off the end
    with pytest.raises(KeyError):
        index.door(PKG).address(7)
    with pytest.raises(KeyError):
        index.unit(window_address(region_identity(H, "tools"), 0))


def test_portals_name_the_window_on_the_other_side_without_loading_it():
    index = UnitIndex(_scan(), capacity=3)
    first = index.unit(window_address(PKG, 0))   # sub/, __init__.py, a.py
    assert [(p.inside, p.outside, p.outside_unit, p.direction) for p in first.portals] == [
        (room_identity(H, "pkg/a.py"), room_identity(H, "pkg/b.py"), window_address(PKG, 3), "out")]
    later = index.unit(window_address(PKG, 3))   # b.py, c.py, d.py
    assert {(p.inside, p.outside, p.outside_unit, p.direction) for p in later.portals} == {
        (room_identity(H, "pkg/b.py"), room_identity(H, "pkg/a.py"), window_address(PKG, 2), "in"),
        (room_identity(H, "pkg/d.py"), room_identity(H, "tools/t.py"), region_identity(H, "tools"), "out"),
    }
    both = index.unit(window_address(PKG, 2))     # a.py, b.py, c.py hold both ends of a -> b
    assert len(both.inner_portals) == 1
    whole = UnitIndex(_scan(), capacity=100).unit(PKG)
    assert len(whole.inner_portals) == 1 and len(whole.portals) == 1


class Lot:
    plaque: str


def _lot_openings(world, lot):
    boxes = {box["identity"]: box for box in document_geometry(world_div_map_model(world, static_only=True))["boxes"]}
    return boxes[lot]["openings"]


def test_a_treadmill_door_is_one_island_door_that_carries_its_scroll():
    outside = build_introspective_world(Lot, seed="treadmill-door-test")
    lot = outside.regions[0].buildings[0].identity
    island = "island:owner/name/pkg"
    with_door = declare_treadmill_door(outside, lot, island, length=7, window=3, offset=5, href="/islands/pkg.html")
    ends = [o for o in _lot_openings(with_door, lot) if o.get("treadmill")]
    door = door_identity(lot, island)
    assert {o["identity"] for o in ends} == {f"{door}/enter:in", f"{door}/exit:out"}
    assert all(o["treadmill"] == {"length": 7, "window": 3, "offset": 5} for o in ends)
    entry = next(o for o in ends if o["port_role"] == "in")
    assert (entry["island"], entry["href"]) == (island, "/islands/pkg.html")
    for bad in ({"length": 3, "window": 3}, {"length": 7, "window": 0}, {"length": 7, "window": 3, "offset": 7}):
        with pytest.raises(ValueError):
            declare_treadmill_door(outside, lot, island, **bad)


def test_a_dial_door_is_one_doorway_with_an_island_door_per_contender():
    outside = build_introspective_world(Lot, seed="dial-door-test")
    lot = outside.regions[0].buildings[0].identity
    positions = [{"island": f"island:owner/name/x.py~{who}", "href": f"/islands/x-{who}.html", "label": who}
                 for who in ("ann", "bo", "cy")]
    with_dial = declare_dial_door(outside, lot, positions)
    ends = [o for o in _lot_openings(with_dial, lot) if o.get("dial")]
    assert len(ends) == 6 and {o["dial"] for o in ends} == {dial_identity(lot)}
    entries = sorted((o for o in ends if o["port_role"] == "in"), key=lambda o: o["dial_position"])
    assert [(o["dial_position"], o["island"], o["href"], o["dial_label"]) for o in entries] == [
        (k, p["island"], p["href"], p["label"]) for k, p in enumerate(positions)]
    assert all(o["dial_positions"] == 3 and o["coincident_with"] == f"{lot}/opening:entry" for o in ends)
    with pytest.raises(ValueError):
        declare_dial_door(outside, lot, [])
    with pytest.raises(ValueError):
        declare_dial_door(outside, lot, [positions[0], positions[0]])
    with pytest.raises(ValueError):           # a position is an island door, so it cannot be declared twice
        declare_dial_door(declare_island_door(outside, lot, positions[0]["island"]), lot, positions)
