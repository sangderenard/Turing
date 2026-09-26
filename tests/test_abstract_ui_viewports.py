"""Neutral viewport inheritance and document-geometry extrusion."""

import pytest

from src.compiler.abstract_ui_primitives import bbox, palette
from src.compiler.abstract_ui_viewports import (
    FragmentOperation,
    ShaderViewport,
    ViewerCamera,
    Viewport,
    ViewportControlPolicy,
    document_geometry,
)


def _world():
    return {
        "identity": "world",
        "regions": [{
            "identity": "region",
            "buildings": [{
                "identity": "building",
                "rooms": [
                    {"identity": "field", "member_kind": "field", "position": {"column": 0, "row": 0}},
                    {"identity": "method", "member_kind": "method", "position": {"column": 1, "row": 0}},
                ],
            }],
        }],
    }


def test_shader_viewport_inherits_bounded_view_subject_and_camera():
    camera = ViewerCamera("camera", "world", tracking_actor="pointer")
    stage = FragmentOperation("extrude", "extrude-document-geometry", ("map",), "boxes")
    viewport = ShaderViewport(
        "viewer", "system-root", "world", bbox(0, 0, 640, 360), camera,
        palette(fg="#fff", bg="#000"), fragment_chain=(stage,),
    )

    assert isinstance(viewport, Viewport)
    data = viewport.to_data()
    assert data["kind"] == "shader-viewport"
    assert data["camera"]["tracking_actor"] == "pointer"
    assert data["camera"]["embodiment_scale"] == 1.0
    assert data["fragment_chain"][0]["operation"] == "extrude-document-geometry"
    assert data["dependencies"][-1] == {"relationship": "views", "target": "world"}


def test_document_geometry_preserves_courtyard_building_room_hierarchy():
    geometry = document_geometry(_world())
    assert [box["kind"] for box in geometry["boxes"]] == [
        "world-envelope", "courtyard", "building", "room", "room",
    ]
    envelope, courtyard, building, field, method = geometry["boxes"]
    assert method["height"] > field["height"]
    assert envelope["half_extent"][0] > courtyard["half_extent"][0]
    assert courtyard["height"] > building["height"]
    assert courtyard["metaphor"] == "defensive class compound"
    assert geometry["hierarchy_space"]["policy"] == "nonlinear-containment-distance-v0"
    assert geometry["representation_boundary"]["crossing_operation"] == (
        "switch-map-representation"
    )
    assert geometry["relationships"] == [
        "world-envelope-contains-courtyard",
        "courtyard-contains-building", "building-contains-room",
    ]


def test_viewer_camera_validates_projection_facts():
    with pytest.raises(ValueError, match="field of view"):
        ViewerCamera("camera", "world", field_of_view=180)


def test_viewer_camera_validates_embodiment_dimensions():
    with pytest.raises(ValueError, match="embodiment dimensions"):
        ViewerCamera("camera", "world", embodiment_scale=0)


def test_viewport_control_policy_is_backend_neutral_and_routes_to_actor():
    camera = ViewerCamera("camera", "world", tracking_actor="actor")
    controls = ViewportControlPolicy("viewer/controls", "actor")
    viewport = Viewport(
        "viewer", "system-root", "world", bbox(0, 0, 640, 360), camera,
        palette(fg="#fff", bg="#000"), control_policy=controls,
    )

    data = viewport.to_data()
    assert data["control_policy"]["activation"] == "highlight"
    assert data["control_policy"]["captures"] == ["keyboard", "pointer", "gamepad"]
    assert data["control_policy"]["gamepad_selection"] == "first-connected"
    assert data["control_policy"]["run_multiplier"] == 2.0
    assert data["control_policy"]["jump_speed"] == 3.6
    assert data["control_policy"]["bindings"][0] == {
        "action": "move-forward",
        "inputs": ["keyboard:KeyW", "gamepad:left-y-negative"],
    }
    assert {binding["action"]: binding["inputs"] for binding in data["control_policy"]["bindings"]}[
        "run"
    ] == ["keyboard:ShiftLeft", "keyboard:ShiftRight"]
    assert {binding["action"]: binding["inputs"] for binding in data["control_policy"]["bindings"]}[
        "jump"
    ] == ["keyboard:Space"]
    assert data["dependencies"][-1] == {
        "relationship": "routes-controls-to", "target": "actor",
    }


def test_viewport_control_policy_rejects_invalid_rates():
    with pytest.raises(ValueError, match="rates must be positive"):
        ViewportControlPolicy("controls", "actor", move_speed=0)
