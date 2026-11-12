"""A unit of a repository world as one living model: walked by the text navigator, drawn by the page; and the blank
wilderness embedding that gives every map its place on the hypersphere."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku.living_data_query import GraphView, query  # noqa: E402
from src.compiler.abstract_ui_div_map import world_div_map_model  # noqa: E402
from src.compiler.abstract_ui_viewports import document_geometry  # noqa: E402
from src.compiler.abstract_ui_wilderness import cosine, golden_ratio, position  # noqa: E402
from src.compiler.repo_world_scan import region_identity, room_identity, scan_files  # noqa: E402
from src.compiler.repo_world_unit_world import build_unit_world, island_of, unit_graph  # noqa: E402
from src.compiler.repo_world_units import UnitIndex, window_address  # noqa: E402

H = "b" * 40
FILES = {
    "README.md": b"", "pkg/__init__.py": b"", "pkg/a.py": b"from pkg import b\n", "pkg/b.py": b"", "pkg/c.py": b"",
    "pkg/d.py": b"import tools.t\n", "pkg/e.py": b"", "pkg/sub/x.py": b"", "tools/t.py": b"",
}
ROOT, PKG, TOOLS = region_identity(H, "."), region_identity(H, "pkg"), region_identity(H, "tools")


def _index(capacity):
    return UnitIndex(scan_files(H, None, sorted(FILES), FILES.__getitem__), capacity=capacity)


def _exits(graph):
    return {(e["source"], e["relationship"], e["target"]) for e in GraphView(graph).edges if e["relationship"] != "contained-by"}


def test_the_root_courtyard_holds_every_item_as_a_lot_and_its_doors_are_exits():
    graph = unit_graph(_index(3), ROOT)
    view = GraphView(graph)
    assert graph["identity"] == island_of(ROOT)
    assert {view.node(i)["kind"] for i in (PKG, TOOLS, room_identity(H, "README.md"))} == {"building"}
    # pkg is too long for one window: its door opens on the treadmill at offset 0; tools fits and opens on itself
    assert _exits(graph) == {(PKG, "door", window_address(PKG, 0)), (TOOLS, "door", TOOLS)}
    walked = query({"map": graph, "at": ROOT, "radius": 1})
    assert {n["identity"] for n in walked["nodes"]} >= {PKG, TOOLS, room_identity(H, "README.md")}


def test_a_window_names_its_way_out_its_doors_and_the_imports_that_leave_it():
    graph = unit_graph(_index(3), window_address(PKG, 0))    # sub/, __init__.py, a.py
    assert _exits(graph) == {
        (region_identity(H, "pkg/sub"), "door", region_identity(H, "pkg/sub")),
        (room_identity(H, "pkg/a.py"), "python-import", room_identity(H, "pkg/b.py")),
        (window_address(PKG, 0), "way-out", ROOT),
    }
    later = unit_graph(_index(3), window_address(PKG, 3))     # b.py, c.py, d.py
    assert (room_identity(H, "pkg/d.py"), "python-import", room_identity(H, "tools/t.py")) in _exits(later)
    # every window of one courtyard is the same island, at the same place in the wilderness
    assert later["identity"] == graph["identity"] == island_of(PKG)
    assert later["wilderness"] == graph["wilderness"]


def test_the_page_lays_out_the_same_world_and_its_doors():
    world = build_unit_world(_index(3), ROOT, href=lambda address: "/unit?address=" + address)
    boxes = {b["identity"]: b for b in document_geometry(world_div_map_model(world, static_only=True))["boxes"]}
    assert boxes[ROOT]["kind"] == "courtyard" and boxes[PKG]["kind"] == "building"
    treadmill = [o for o in boxes[PKG].get("openings", []) if o.get("treadmill")]
    assert treadmill and all(o["treadmill"] == {"length": 7, "window": 3, "offset": 0} for o in treadmill)
    plain = [o for o in boxes[TOOLS].get("openings", []) if o.get("island") == island_of(TOOLS)]
    assert [o["href"] for o in plain] == ["/unit?address=" + TOOLS]


def test_the_blank_embedding_is_a_stable_even_spread_on_the_sphere():
    assert math.isclose(golden_ratio(1), (1 + 5 ** 0.5) / 2)
    a = position("world:repo:x/dir:.")
    assert a == position("world:repo:x/dir:.") and math.isclose(sum(c * c for c in a), 1.0)
    assert len(position("x", 3)) == 3 and len(a) == 8
    points = [position(f"world:repo:x/dir:{k}", 3) for k in range(400)]
    mean = [sum(p[i] for p in points) / len(points) for i in range(3)]
    assert math.sqrt(sum(m * m for m in mean)) < 0.1            # no pole the points crowd toward
    nearest = min(math.acos(min(1.0, cosine(points[0], p))) for p in points[1:])
    assert nearest > 0.01                                          # no two of these land on one spot
    with pytest.raises(ValueError):
        position("x", 1)
    with pytest.raises(ValueError):
        position("", 3)
