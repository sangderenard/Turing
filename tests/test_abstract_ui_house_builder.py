import pytest

from src.compiler.abstract_ui_house_builder import (
    ElectricalServiceBoundary,
    RoomOpening,
    RoomSpec,
    TileRectangle,
    WallOutlet,
    build_room_graph,
    house_builder_model,
)


def test_room_is_a_tile_addressed_object_graph_with_real_boundaries():
    graph = build_room_graph(RoomSpec(
        "garage.room", "lot.house", TileRectangle(2, 3, 6, 5),
        openings=(RoomOpening(
            "garage.room.door", "south", 0.0, 1.0, 2.05,
            connects_to="outside.air",
        ),),
        electrical=ElectricalServiceBoundary(
            "utility.service", delivery="pre-breaker",
        ),
    ))
    nodes = {node["identity"]: node for node in graph.nodes}
    assert nodes["garage.room.foundation"]["body"] == "static-rigid"
    assert nodes["garage.room.wall.north"]["kind"] == "wall-body"
    assert nodes["garage.room.air-volume"]["state_owner"] == "atmosphere-engine"
    assert nodes["garage.room.door"]["seal_model"] == "semi-sealed"
    assert nodes["garage.room.door"]["closed_leakage_area_m2"] is None
    assert graph.requirements[0]["required_kind"] == "electrical-distribution-panel"
    room = graph.world_objects[0]
    assert room.form["recipe"] == "boundary-floor-with-openings"
    assert room.transform["tile_rectangle"]["width"] == 6
    assert len([node for node in graph.nodes if node["kind"] == "wall-body"]) == 4


def test_pre_breaker_service_cannot_silently_feed_an_outlet_without_a_panel():
    with pytest.raises(ValueError, match="requires a real panel"):
        RoomSpec(
            "room", "lot", TileRectangle(0, 0, 3, 3),
            electrical=ElectricalServiceBoundary("utility", delivery="pre-breaker"),
            outlets=(WallOutlet("room.outlet", "north", 0.0),),
        )


def test_post_breaker_outlet_routes_conduit_from_declared_breaker():
    graph = build_room_graph(RoomSpec(
        "room", "lot", TileRectangle(0, 0, 3, 3),
        electrical=ElectricalServiceBoundary(
            "house.panel", delivery="post-breaker",
            upstream_breaker_identity="house.panel.breaker.1",
        ),
        outlets=(WallOutlet("room.outlet", "north", 0.0),),
    ))
    conduit = next(edge for edge in graph.edges if edge["identity"] == "room.outlet.conduit")
    assert conduit["a"] == "house.panel.breaker.1"
    assert conduit["host_wall"] == "room.wall.north"
    assert conduit["concealed_in_wall"] is True


def test_house_builder_publishes_living_map_authority_and_no_solver():
    model = house_builder_model("world")
    assert model["interaction_host"] == "living-data-map-placement-tool"
    assert model["authority"] == "living-document-world-object-graph"
    assert model["solver_binding"] == "declarative-unbound"
