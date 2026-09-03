"""Identity, custody, and tick contracts for the in-world validator rig."""

from src.compiler.abstract_ui_introspection import build_introspective_world
from src.compiler.abstract_ui_validator_rig import (
    validator_rig_assembly,
    validator_rig_geometry_boxes,
)
from src.compiler.mechanical_creature import MechanicalCreatureWorld


def _assembly():
    return validator_rig_assembly(
        "world:probe",
        "actor:probe",
        {
            "identity": "world:probe/vehicles/free-car",
            "name": "Free car",
            "archetype": "vehicle:probe",
            "pose": {"position": [0.0, 1.0, 0.0]},
        },
        {"identity": "world:probe/archetypes/physics-ball"},
        center_x=4.0,
        center_z=8.0,
    )


def test_rig_preserves_initial_car_identity_and_owns_backup_until_release():
    assembly = _assembly()
    model = assembly.model
    objects = {item.identity: item for item in assembly.world_objects}

    initial = model["world_properties"]["initial_vehicle"]
    backup = model["construction"]["vehicle"]
    assert model["identity_contract"]["room_discovery_reference"] == initial
    assert model["identity_contract"]["mounted_vehicle_reference"] == initial
    assert objects[initial].parent == "world:probe"
    assert objects[backup].parent == model["identity"]
    assert objects[backup].physics["usable"] is False
    assert model["release"]["vehicle"] == backup


def test_part_installation_transfers_custody_from_line_projection_to_phong():
    model = _assembly().model
    parts = model["construction"]["parts"]
    assert sum(part["material_units"] for part in parts) == model["material_supply"]["recipe_units"]
    assert all(part["presentation"]["projected"]["background_alpha"] == 0.0 for part in parts)
    assert all(part["presentation"]["installed"]["program"].endswith("/pluck-phong") for part in parts)
    assert all(part["handoff"].endswith("vehicle-installed-phong") for part in parts)


def test_rig_geometry_is_child_realization_and_hopper_mouth_remains_open():
    assembly = _assembly()
    boxes = validator_rig_geometry_boxes(assembly)
    rig_identity = assembly.model["identity"]
    assert boxes[0]["parent_identity"] == rig_identity
    assert boxes[1]["parent_identity"] == assembly.model["form"]["stage_identity"]
    hopper = boxes[2:]
    assert len(hopper) == 4
    assert all(box["geometry_mode"] == "solid" for box in hopper)
    assert {box["identity"].rsplit("/", 1)[-1] for box in hopper} == {
        "north", "south", "west", "east",
    }


def test_world_class_exposes_owned_objects_as_rooms_and_assigns_time_explicitly():
    world = build_introspective_world(MechanicalCreatureWorld, seed="rig-world")
    building = world.building("MechanicalCreatureWorld")
    assert {room.name for room in building.rooms} >= {
        "initial_vehicle", "validator_rig", "tick",
    }
    tick_room = building.room("tick")
    assert tick_room.parameters == ("tick", "dt", "subdt", "substeps")

    execution = _assembly().model["execution"]
    assert execution["mode"] == "assigned-ticks"
    assert execution["assigned_ticks"]["inactive_until_called"] is True
    assert execution["assigned_ticks"]["envelope"] == ["tick", "dt", "subdt", "substeps"]
    assert execution["autonomous"]["internal_scheduler"] is False
