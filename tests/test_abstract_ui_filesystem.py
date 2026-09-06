"""Human artifact ownership, placement, attachment, and filesystem graphs."""

import pytest

from src.compiler.abstract_ui_filesystem import (
    ArtifactAttachment,
    advance_attachment,
    artifact_geometry_boxes,
    canonical_path,
    filesystem_graph_for_world,
)


def _world():
    return {
        "identity": "world",
        "regions": [{"buildings": [{
            "identity": "python:pkg.Panel", "module": "pkg.panel", "name": "Panel",
        }]}],
    }


def _geometry():
    return {"boxes": [
        {"identity": "world/representation:global", "kind": "world-envelope",
         "center": [0.0, 0.0], "half_extent": [30.0, 30.0]},
        {"identity": "region:pkg", "kind": "courtyard",
         "center": [0.0, 1.0], "half_extent": [8.0, 7.0]},
    ]}


def test_filesystem_ownership_and_world_placement_are_independent_edges():
    graph = filesystem_graph_for_world(_world(), actor="player")
    data = graph.to_data()
    assert [artifact["kind"] for artifact in data["artifacts"]] == [
        "source", "test", "readme", "annotation", "scratch",
    ]
    source = data["artifacts"][0]
    assert source["path"] == "/pkg/panel.py"
    assert source["owner"] == "python:pkg.Panel"
    assert source["placed_in"] == "world/representation:global"
    assert source["physics"]["body"] == "dynamic"
    assert data["ownership_edges"][0]["target"] == "python:pkg.Panel"
    assert data["placement_edges"][0]["target"] != "python:pkg.Panel"


def test_close_slow_attachment_settles_then_welds_and_resets_if_disturbed():
    attachment = ArtifactAttachment("owner")
    settling = advance_attachment(
        attachment, distance=0.2, relative_speed=0.05, dt=0.3,
    )
    assert settling.state == "settling"
    disturbed = advance_attachment(
        settling, distance=0.8, relative_speed=0.05, dt=0.1,
    )
    assert (disturbed.state, disturbed.settle_time) == ("loose", 0.0)
    welded = advance_attachment(
        settling, distance=0.2, relative_speed=0.05, dt=0.4,
    )
    assert welded.state == "welded"
    assert advance_attachment(welded, distance=100, relative_speed=100, dt=1) == welded


def test_internal_native_web_and_wasm_backends_share_one_logical_graph():
    data = filesystem_graph_for_world(_world()).to_data()
    assert data["path_semantics"] == "canonical-posix-logical-path"
    assert set(data["backend_contracts"]) == {"internal", "native-c", "web", "wasm"}
    assert data["backend_contracts"]["web"]["boundary"] == (
        "browser-origin-sandbox-no-host-filesystem-claim"
    )
    assert data["backend_contracts"]["native-c"]["boundary"] == (
        "explicit-root-capability-no-ambient-path-access"
    )


def test_artifacts_realize_as_small_solid_bodies_without_changing_owner():
    graph = filesystem_graph_for_world(_world())
    boxes = artifact_geometry_boxes(graph, _geometry())
    assert len(boxes) == 5
    assert all(box["geometry_mode"] == "solid" for box in boxes)
    assert boxes[0]["parent_identity"] == "python:pkg.Panel"
    assert boxes[0]["spatial_container"] == "world/representation:global"
    assert boxes[0]["physics"]["body"] == "dynamic"


def test_logical_paths_are_canonical_and_cannot_escape_root():
    assert canonical_path(r"src\pkg\module.py") == "/src/pkg/module.py"
    with pytest.raises(ValueError, match="escape"):
        canonical_path("../outside")
