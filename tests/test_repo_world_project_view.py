"""A project's view courtyard (dev pages ticket 3/8, research case 810bfd48): a unit whose lots are exactly the
sources a project contract's scope selects, across a project's scans, each with a door home to its real folder."""

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku.living_data_query import GraphView, query  # noqa: E402
from src.compiler.project_contract import parse_project_contract  # noqa: E402
from src.compiler.repo_world_scan import region_identity, room_identity, scan_files  # noqa: E402
from src.compiler.repo_world_unit_world import build_unit_world, island_of, measure_unit, unit_graph  # noqa: E402
from src.compiler.repo_world_units import (  # noqa: E402
    Treadmill, UnitIndex, project_view, split_address, view_address, window_address,
)

BRANCH, SPACE = "b" * 40, "c" * 40
BRANCH_FILES = {
    "examples/garden/world.py": b"import examples.garden.helpers\n",
    "examples/garden/helpers.py": b"",
    "examples/garden/uses_other.py": b"import examples.other.x\n",
    "examples/garden/shaders/sky.frag": b"void main() {}\n",
    "examples/garden/build/world.py": b"",
    "examples/garden/page.wasm": b"\0asm",
    "examples/garden/index.html": b"<p>the published page</p>\n",
    "examples/garden.py": b"",
    "examples/other/x.py": b"",
    "README.md": b"",
}
SPACE_FILES = {"examples/garden/extra.py": b"", "notes.txt": b""}
CONTRACT = yaml.safe_dump({
    "schema": "project-contract-v0", "identity": "ann/garden",
    "placement": {"mode": "branch", "branch": {"repository": "turing", "name": "ann/garden"}},
    "scope": {"roots": ["examples/garden"], "include": {"languages": ["python", "glsl"]}, "exclude": ["build/"]},
    "view": {"name": "sources"},
})
VIEW = "view:ann/garden/sources"


def scans():
    return [scan_files(BRANCH, None, sorted(BRANCH_FILES), BRANCH_FILES.__getitem__),
            scan_files(SPACE, None, sorted(SPACE_FILES), SPACE_FILES.__getitem__)]


def built(capacity=1024):
    return project_view(parse_project_contract(CONTRACT), scans(), capacity=capacity)


def b(path):
    return room_identity(BRANCH, path)


SELECTED = {b("examples/garden/world.py"), b("examples/garden/helpers.py"), b("examples/garden/uses_other.py"),
            b("examples/garden/shaders/sky.frag"), room_identity(SPACE, "examples/garden/extra.py")}


def exits(graph):
    return {(e["source"], e["relationship"], e["target"]) for e in GraphView(graph).edges
            if e["relationship"] != "contained-by"}


def test_the_view_selects_exactly_the_scoped_sources_from_every_scan():
    index, address = built()
    assert address == VIEW == view_address("ann", "garden", "sources")
    unit = index.unit(address)
    assert set(unit.items) == SELECTED, "no product, page, excluded folder, or file outside a root"
    assert [index.rooms[item]["path"] for item in unit.items] == sorted(index.rooms[item]["path"] for item in unit.items)
    assert unit.way_out is None and unit.doors == {}
    assert index.unit_of(b("examples/garden/world.py")) == region_identity(BRANCH, "examples/garden"), \
        "a view never becomes an item's home"


def test_every_lot_keeps_its_real_identity_and_has_a_door_home():
    index, address = built()
    unit = index.unit(address)
    assert unit.homes == {item: index.unit_of(item) for item in SELECTED}
    graph = unit_graph(index, address)
    homes = {(source, target) for source, relationship, target in exits(graph) if relationship == "home"}
    assert homes == set(unit.homes.items())
    world = build_unit_world(index, address, href=lambda unit_address: "/unit?address=" + unit_address)
    lots = {lot.identity: lot for region in world.regions for lot in region.buildings}
    assert set(lots) == SELECTED and {lot.metaphor for lot in lots.values()} == {"ruin"}
    for item in SELECTED:
        islands = {(o.get("island"), o.get("href")) for o in lots[item].openings if o.get("island")}
        assert islands == {(island_of(index.rooms[item]["region"]), "/unit?address=" + unit.homes[item])}


def test_imports_inside_are_tracks_and_imports_out_are_labelled_with_the_other_end():
    index, address = built()
    graph = unit_graph(index, address)
    found = exits(graph)
    assert (b("examples/garden/world.py"), "python-import", b("examples/garden/helpers.py")) in found
    assert (b("examples/garden/uses_other.py"), "python-import", b("examples/other/x.py")) in found
    out = [p for p in index.unit(address).portals if p.outside == b("examples/other/x.py")]
    assert [(p.direction, p.outside_unit) for p in out] == [("out", region_identity(BRANCH, "examples/other"))]
    labels = {t.target: t.label for t in build_unit_world(index, address).tracks if t.relationship == "python-import"}
    assert labels[b("examples/other/x.py")] == region_identity(BRANCH, "examples/other")


def test_the_view_is_a_walkable_graph_with_its_own_place_in_the_wilderness():
    index, address = built()
    graph = unit_graph(index, address)
    assert graph["schema"] == "abstract-ui-div-map-v0" and graph["identity"] == island_of(VIEW)
    walked = query({"map": graph, "at": VIEW, "radius": 1})
    assert {n["identity"] for n in walked["nodes"]} >= SELECTED
    folder = unit_graph(index, region_identity(BRANCH, "examples/garden"))
    assert graph["wilderness"]["position"] != folder["wilderness"]["position"]
    assert len(graph["wilderness"]["position"]) == graph["wilderness"]["dimensions"]


def test_a_long_view_is_a_treadmill_of_windows():
    index, address = built(capacity=2)
    door = index.door(address)
    assert isinstance(door, Treadmill) and door.length == len(SELECTED) and door.window == 2
    assert split_address(window_address(VIEW, 2)) == (VIEW, 2)
    with pytest.raises(KeyError, match="is a treadmill"):
        index.unit(VIEW)
    windows = [index.unit(door.address(offset)) for offset in range(0, door.length, 2)]
    assert [len(w.items) for w in windows] == [2, 2, 1]
    assert set().union(*(w.items for w in windows)) == SELECTED
    assert len({unit_graph(index, w.address)["identity"] for w in windows}) == 1, "every window is one island"


def test_the_size_is_measured_before_any_html():
    index, address = built()
    size = measure_unit(index, address)
    graph = unit_graph(index, address)
    assert size["items"] == len(SELECTED) and size["tracks"] == len(build_unit_world(index, address).tracks)
    assert size["graph_bytes"] == len(json.dumps(graph, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    assert size["openings"] >= len(SELECTED), "at least one door home per lot"


def test_views_are_named_and_checked():
    for parts in (("ann", "garden", "two/parts"), ("ann", "", "sources"), ("ann", "garden", "a@1")):
        with pytest.raises(ValueError, match="plain name"):
            view_address(*parts)
    with pytest.raises(ValueError, match="a view's address is"):
        UnitIndex(scans(), views={"garden": []})
    with pytest.raises(KeyError, match="in no scan"):
        UnitIndex(scans(), views={VIEW: [b("missing.py")]})
    with pytest.raises(ValueError, match="names one file twice"):
        UnitIndex(scans(), views={VIEW: [b("examples/garden/world.py")] * 2})
    one = scans()[0]
    assert UnitIndex([one, one]).rooms == UnitIndex(one).rooms, "the same scan twice is the same records"
