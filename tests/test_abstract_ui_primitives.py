"""Canonical div/input/palette/bbox/graph primitive tests."""

import pytest

from src.compiler.abstract_ui_primitives import (
    DEFAULT_PALETTE,
    UIDecoration,
    UIFont,
    UIInsets,
    UILength,
    UIPrimitiveEdge,
    UIRadii,
    bbox,
    div,
    graph,
    input_,
    palette,
)


def test_div_contains_constructible_primitives_in_appearance_order():
    label = div("panel/label", text="Gain")
    control = input_(
        "panel/gain", interaction="set-value", destination="program.gain", value=0.5,
    )
    panel = div("panel", label, control)
    assert panel.archetype == "div"
    assert [child.identity for child in panel.children] == ["panel/label", "panel/gain"]
    assert dict(control.properties)["interaction"] == "set-value"
    assert dict(control.properties)["destination"] == "program.gain"


def test_palette_layers_fg_bg_margins_radii_font_decoration_visibility_and_lock():
    base = palette(
        fg="#eeeeee",
        bg="#111111",
        margins=UIInsets.all(8),
        radii=UIRadii.all(6),
        font=UIFont(("Inter", "sans-serif"), UILength(15), weight=500),
        decoration=UIDecoration(("border", "shadow")),
        visible=True,
        locked=False,
    )
    local = palette(fg="#ffcc66", radii=UIRadii.all(12))
    resolved = base.overlay(local)
    assert resolved.fg.value == "#ffcc66"
    assert resolved.bg.value == "#111111"
    assert resolved.margins == UIInsets.all(8)
    assert resolved.radii == UIRadii.all(12)
    assert resolved.font.families == ("Inter", "sans-serif")
    assert resolved.decoration.names == ("border", "shadow")
    assert resolved.visible and not resolved.locked


def test_palette_named_colors_overlay_for_css_and_shader_consumers():
    base = palette(colors={"room": "#112233", "sky": "#010203"})
    local = palette(colors={"room": "#445566"})
    resolved = base.overlay(local)
    assert {name: color.value for name, color in resolved.colors} == {
        "room": "#445566", "sky": "#010203",
    }


def test_locked_is_an_edit_rule_not_only_a_rendering_hint():
    locked = div("locked", palette=palette(locked=True))
    with pytest.raises(PermissionError, match="locked"):
        locked.with_(div("child"))
    with pytest.raises(PermissionError, match="locked"):
        locked.styled(palette(fg="red"))
    with pytest.raises(PermissionError, match="locked"):
        locked.placed(bbox(0, 0, 10, 10))


def test_bbox_has_explicit_coordinate_space_and_geometry_operations():
    outer = bbox(0, 0, 100, 80, coordinate_space="viewport")
    inner = bbox(10, 10, 20, 20, coordinate_space="viewport")
    assert outer.contains(50, 30)
    assert outer.intersects(inner)
    with pytest.raises(ValueError, match="coordinate space"):
        outer.intersects(bbox(0, 0, 1, 1, coordinate_space="world"))


def test_graph_validates_identity_and_endpoint_integrity():
    source = div("graph/source")
    destination = input_(
        "graph/destination", interaction="inspect", destination="graph/source",
    )
    edge = UIPrimitiveEdge(source.identity, destination.identity, "flows-to")
    value = graph("graph", source, destination, edges=(edge,))
    assert value.edges == (edge,)
    assert [item.identity for item in value.objects()] == [
        "graph", "graph/source", "graph/destination",
    ]
    with pytest.raises(KeyError, match="unknown objects"):
        graph("bad", source, edges=(UIPrimitiveEdge(source.identity, "missing", "to"),))


def test_primitive_transport_resolves_palette_and_preserves_graph_edges():
    node = div(
        "node",
        palette=palette(bg="#223344", visible=False),
        bbox=bbox(1, 2, 30, 40),
    ).connect("node", "node", "self")
    data = node.to_data()
    assert data["palette"]["fg"] == DEFAULT_PALETTE.fg.value
    assert data["palette"]["bg"] == "#223344"
    assert data["palette"]["visible"] is False
    assert data["bbox"] == {
        "x": 1.0, "y": 2.0, "width": 30.0, "height": 40.0,
        "coordinate_space": "layout",
    }
    assert data["edges"][0]["relationship"] == "self"
