"""Producer-declared openings (portal ends) travel from the world dataclasses to the geometry boxes.

Class mode declares none, so its model and boxes are unchanged; a producer that declares a portal end on a room or
a building sees it on that box after the door the geometry mints, which is where the living map already converts
``kind: "portal"`` openings into portal splats (``migrateLegacyPortalOpenings``).
"""

from dataclasses import replace

from src.compiler.abstract_ui_div_map import DIV_MAP_JAVASCRIPT, world_div_map_model
from src.compiler.abstract_ui_introspection import build_introspective_world
from src.compiler.abstract_ui_viewports import document_geometry


class Lot:
    plaque: str

    def knock(self, times: int) -> bool:
        return times > 0


def _world():
    return build_introspective_world(Lot, seed="declared-openings-test")


def _boxes(model):
    return {box["identity"]: box for box in document_geometry(model)["boxes"]}


def test_class_mode_declares_nothing_and_is_unchanged():
    world = _world()
    model = world_div_map_model(world, static_only=True)
    building = model["regions"][0]["buildings"][0]
    assert "openings" not in building
    assert all("openings" not in room for room in building["rooms"])
    boxes = _boxes(model)
    assert [o["kind"] for o in boxes[building["identity"]]["openings"]] == ["door"]
    for room in building["rooms"]:
        assert [o["identity"] for o in boxes[room["identity"]]["openings"]] == [f"{room['identity']}/opening:door"]


def test_declared_portal_ends_reach_the_boxes_after_the_minted_door():
    world = _world()
    region = world.regions[0]
    building = region.buildings[0]
    room = building.rooms[0]
    doorway = {"identity": f"{building.identity}/portal:interior/end:in", "kind": "portal", "port_role": "in",
               "pair": f"{building.identity}/portal:interior", "side": "north", "offset": 0.0,
               "width": 0.6, "height": 0.7}
    reference = {"identity": f"{room.identity}/portal:ref:1/end:in", "kind": "portal", "port_role": "in",
                 "pair": f"{room.identity}/portal:ref:1", "side": "east", "offset": 0.1, "width": 0.3, "height": 0.4,
                 "reference": {"source_line": 12, "reference_kind": "python-import", "specifier": "pkg.mod"},
                 "transmission": None}
    declared_room = replace(room, openings=(reference,))
    declared_building = replace(building, rooms=(declared_room, *building.rooms[1:]), openings=(doorway,))
    declared_world = replace(world, regions=(replace(region, buildings=(declared_building, *region.buildings[1:])),
                                             *world.regions[1:]))

    model = world_div_map_model(declared_world, static_only=True)
    model_building = model["regions"][0]["buildings"][0]
    assert model_building["openings"] == [doorway]
    assert model_building["rooms"][0]["openings"] == [reference]
    assert all("openings" not in other for other in model_building["rooms"][1:])

    boxes = _boxes(model)
    building_openings = boxes[building.identity]["openings"]
    assert [o["kind"] for o in building_openings] == ["door", "portal"]
    assert building_openings[1] == doorway
    room_openings = boxes[room.identity]["openings"]
    assert [o["kind"] for o in room_openings] == ["door", "portal"]
    assert room_openings[1]["pair"] == reference["pair"] and room_openings[1]["port_role"] == "in"
    assert room_openings[1]["transmission"] is None
    # The declaration is copied, never aliased: editing a box cannot rewrite the producer's record.
    room_openings[1]["width"] = 9.0
    assert reference["width"] == 0.3


def test_browser_template_honours_a_declared_pair():
    assert "function portalDeclaredPair(splat)" in DIV_MAP_JAVASCRIPT
    assert "portalDeclaredPair(target) === portalDeclaredPair(source)" in DIV_MAP_JAVASCRIPT
    assert "...(opening.pair ? {pair: opening.pair} : {})" in DIV_MAP_JAVASCRIPT


# ---- a door: a lot's doorway leading into an interior (src/compiler/abstract_ui_doorways.py) ----

import pytest

from src.compiler.abstract_ui_doorways import declare_door, door_identity


class Interior:
    hearth: str

    def rest(self) -> None:
        return None


def _lot_and_interior():
    outside = _world()
    inside = build_introspective_world(Interior, seed="declared-openings-test")
    interior_region = replace(inside.regions[0], identity="region:interior-of-lot")
    world = replace(outside, regions=(*outside.regions, interior_region))
    return world, outside.regions[0].buildings[0].identity, interior_region.identity


def test_a_door_is_four_declared_openings_placed_in_the_doorways_the_geometry_cut():
    world, lot, interior = _lot_and_interior()
    with_door = declare_door(world, lot, interior)
    door = door_identity(lot, interior)

    boxes = _boxes(world_div_map_model(with_door, static_only=True))
    lot_openings = boxes[lot]["openings"]
    entry = lot_openings[0]
    assert [o["identity"] for o in lot_openings] == [f"{lot}/opening:entry", f"{door}/enter:in", f"{door}/exit:out"]
    for sheet in lot_openings[1:]:
        assert sheet["kind"] == "portal" and sheet["transition"] == "door" and "facing" not in sheet
        assert {key: sheet[key] for key in ("side", "offset", "width", "height")} == \
            {key: entry[key] for key in ("side", "offset", "width", "height")}

    passage_openings = boxes[interior]["openings"]
    gate = passage_openings[0]
    assert [o["identity"] for o in passage_openings] == [f"{interior}/opening:gate", f"{door}/enter:out", f"{door}/exit:in"]
    for sheet in passage_openings[1:]:
        assert sheet["facing"] == "inward" and sheet["transition"] == "door"
        assert (sheet["side"], sheet["width"], sheet["height"]) == (gate["side"], gate["width"], gate["height"])

    ends = {o["identity"]: o for o in lot_openings[1:] + passage_openings[1:]}
    assert ends[f"{door}/enter:in"]["port_role"] == "in" and ends[f"{door}/enter:out"]["port_role"] == "out"
    assert ends[f"{door}/enter:in"]["pair"] == ends[f"{door}/enter:out"]["pair"] == f"{door}/enter"
    assert ends[f"{door}/exit:in"]["pair"] == ends[f"{door}/exit:out"]["pair"] == f"{door}/exit"

    # The world without the door is untouched, and every other box is exactly what it was.
    before = _boxes(world_div_map_model(world, static_only=True))
    assert [o["kind"] for o in before[lot]["openings"]] == ["door"]
    for identity, box in before.items():
        if identity not in (lot, interior):
            assert boxes[identity] == box


def test_a_door_refuses_what_it_cannot_mean():
    world, lot, interior = _lot_and_interior()
    with pytest.raises(KeyError):
        declare_door(world, lot, "region:nowhere")
    with pytest.raises(KeyError):
        declare_door(world, "building:nobody", interior)
    with pytest.raises(ValueError):
        declare_door(world, lot, world.regions[0].identity)       # the region the lot stands in
    with_door = declare_door(world, lot, interior)
    with pytest.raises(ValueError):
        declare_door(with_door, lot, interior)                    # already declared


def test_a_sheet_cannot_claim_a_doorway_that_was_never_cut():
    world, lot, _interior = _lot_and_interior()
    region = world.regions[0]
    building = region.buildings[0]
    bad = replace(building, openings=({"identity": f"{lot}/x", "kind": "portal", "port_role": "in",
                                       "coincident_with": f"{lot}/opening:window"},))
    broken = replace(world, regions=(replace(region, buildings=(bad, *region.buildings[1:])), *world.regions[1:]))
    with pytest.raises(ValueError, match="not an opening minted"):
        document_geometry(world_div_map_model(broken, static_only=True))
