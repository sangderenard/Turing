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
