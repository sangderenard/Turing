"""Pure tests for src/compiler/portal_record.py (no scanner import)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.compiler.portal_record import (
    PortalEnd, PortalFrameMap, PortalPair, PortalPlacement, PortalProvenance,
    PortalTransmission, ends_by_owner, from_graph_edge, from_repo_scan_pair,
)


def _scan_pair(src="repo:H/file:a.py", dst="repo:H/file:b.py", line=3):
    pid = f"repo:H/portal:{src}:{line}->{dst}"
    return SimpleNamespace(
        identity=pid, source_room=src, target_room=dst, source_line=line,
        reference_kind="python-import", specifier="b",
        out_end=SimpleNamespace(identity=pid + "/end:out", room=src, role="out",
                                pair=pid, transmission=None),
        in_end=SimpleNamespace(identity=pid + "/end:in", room=dst, role="in",
                               pair=pid, transmission=None))


def test_round_trip_full_record():
    pair = from_graph_edge("A", "B", kind="call", provenance={"specifier": "f", "source_line": 7})
    out_e = PortalEnd(pair.source.identity, "A", "out", pair.identity,
                      PortalPlacement(face="north", center=(0, 1, 0), extent=(0.5, 1.0)),
                      PortalTransmission(spectral=((4e-7, 0.9), (7e-7, 0.5)),
                                         refractive_index=(1.0, 1.5), pass_probability=0.25,
                                         maps_velocity=True),
                      "person", 1.0)
    in_e = PortalEnd(pair.target.identity, "B", "in", pair.identity,
                     PortalPlacement(normal=(0, 0, 1), radius=0.45),
                     PortalTransmission(scalar=0.5))
    full = PortalPair(pair.identity, out_e, in_e, pair.provenance,
                      frame_map=PortalFrameMap.from_yaw((0, 0, 0), (5, 0, 1), 0.0, 1.0),
                      weight=0.3, path_model="relaxed-quaternion-cubic")
    data = json.loads(json.dumps(full.to_data()))
    assert data["schema"] == "portal-record.v1"
    assert PortalPair.from_data(data) == full
    assert PortalPair.from_data(data).to_data() == full.to_data()


def test_schema_tag_required():
    data = from_graph_edge("A", "B", kind="call").to_data()
    data["schema"] = "other"
    with pytest.raises(ValueError):
        PortalPair.from_data(data)


@pytest.mark.parametrize("make", [
    lambda: PortalEnd("e", "o", "sideways", "p"),
    lambda: PortalTransmission(scalar=1.5),
    lambda: PortalTransmission(spectral=((5e-7, -0.1),)),
    lambda: PortalTransmission(spectral=((5e-7, 0.5), (4e-7, 0.5))),
    lambda: PortalTransmission(spectral=((5e-7, 0.5), (5e-7, 0.5))),
    lambda: PortalTransmission(spectral=((5e-7, 0.5),), scalar=0.5),
    lambda: PortalTransmission(refractive_index=(1.0, 0.0)),
    lambda: PortalTransmission(pass_probability=2.0),
    lambda: PortalPlacement(),
    lambda: PortalPlacement(normal=(0, 0, 2)),
    lambda: PortalFrameMap((0, 0, 0), (0, 0, 0), ((1, 0, 0), (0, 1, 0), (0, 0, -1))),
    lambda: PortalFrameMap((0, 0, 0), (0, 0, 0), ((2, 0, 0), (0, 1, 0), (0, 0, 1))),
])
def test_validation_failures(make):
    with pytest.raises(ValueError):
        make()


def test_pair_rules():
    pair = from_graph_edge("A", "B", kind="call")
    with pytest.raises(ValueError):  # same roles
        PortalPair(pair.identity, pair.source, PortalEnd("x", "B", "out", pair.identity), pair.provenance)
    with pytest.raises(ValueError):  # frame map before placement
        PortalPair(pair.identity, pair.source, pair.target, pair.provenance,
                   frame_map=PortalFrameMap.from_yaw((0, 0, 0), (0, 0, 0), 1.0, 0.0))
    with pytest.raises(ValueError):
        from_graph_edge("A", "A", kind="call")


def test_from_repo_scan_pair_by_hand():
    scan = _scan_pair()
    pair = from_repo_scan_pair(scan)
    assert pair.identity == scan.identity
    assert (pair.source.owner, pair.source.role) == (scan.source_room, "out")
    assert (pair.target.owner, pair.target.role) == (scan.target_room, "in")
    assert pair.provenance == PortalProvenance("python-import", "b", 3)
    assert pair.direction == "source-to-target"
    assert pair.frame_map is None


def test_transmission_stays_none_from_scanner():
    pair = from_repo_scan_pair(_scan_pair())
    assert pair.source.transmission is None and pair.target.transmission is None
    assert pair.source.placement is None and pair.target.placement is None


def test_ends_by_owner_grouping_and_determinism():
    pairs = [from_repo_scan_pair(_scan_pair(dst=f"repo:H/file:{n}.py", line=i + 1))
             for i, n in enumerate("bcb")]
    pairs.append(from_repo_scan_pair(_scan_pair(src="repo:H/file:b.py", dst="repo:H/file:a.py", line=9)))
    grouped = ends_by_owner(pairs)
    assert list(grouped) == sorted(grouped)
    assert len(grouped["repo:H/file:a.py"]) == 4  # three out, one in
    assert sum(len(v) for v in grouped.values()) == 2 * len(pairs)
    assert ends_by_owner(list(reversed(pairs))) == grouped
    for ends in grouped.values():
        assert [e.identity for e in ends] == sorted(e.identity for e in ends)


def test_frame_map_yaw_matches_living_map_law():
    fm = PortalFrameMap.from_yaw((1, 0, 0), (10, 0, 0), 0.0, 1.0)
    # physics: x' = tx + cos*rx - sin*rz ; z' = tz + sin*rx + cos*rz
    rx, rz = 2.0, 0.0
    x = fm.target_anchor[0] + fm.rotation[0][0] * rx + fm.rotation[0][2] * rz
    z = fm.target_anchor[2] + fm.rotation[2][0] * rx + fm.rotation[2][2] * rz
    assert (x, z) == (10.0, 2.0)
