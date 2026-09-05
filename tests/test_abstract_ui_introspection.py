"""Pangram-style tests for AbstractUI class-to-world interpretation."""

from dataclasses import dataclass

import pytest

from src.common.tensors.accelerator_backends.dual_ir_shell import DualIRShell
from src.compiler.abstract_ui_introspection import (
    DEFAULT_ROOM_PALETTE,
    AbstractUIBuilding,
    AbstractUIRegion,
    AbstractUIRoom,
    AbstractUITrack,
    AbstractUIWorld,
    build_introspective_world,
)
from src.compiler.abstract_ui_vocabulary import building, remembers, sentence
from src.compiler.class_emission_plan import (
    ClassEmission,
    ClassFieldEmission,
    ClassMethodEmission,
    MethodParameterEmission,
)


def test_real_repository_class_becomes_module_region_class_building_and_member_rooms():
    world = build_introspective_world(DualIRShell, seed="pangram")

    assert isinstance(world, AbstractUIWorld)
    assert len(world.regions) == 1
    region = world.regions[0]
    assert isinstance(region, AbstractUIRegion)
    assert region.name == DualIRShell.__module__

    shell = world.building("DualIRShell")
    assert isinstance(shell, AbstractUIBuilding)
    assert shell.source_kind == "python"
    assert shell.room("compiled_shell_program").member_kind == "field"
    assert shell.room("rollup_profile").member_kind == "method"
    assert shell.room("rollup_log").member_kind == "method"

    positions = [room.position for room in shell.rooms]
    assert len(positions) == len(set(positions))
    assert [room.name for room in shell.rooms[:3]] == [
        "compiled_shell_program", "shell_control_program", "map_ir",
    ]


def test_room_metaphors_are_fluid_but_deterministic_and_identity_is_not():
    first = build_introspective_world(DualIRShell, seed="one")
    repeated = build_introspective_world(DualIRShell, seed="one")
    alternate = build_introspective_world(DualIRShell, seed="two")
    first_rooms = first.building("DualIRShell").rooms
    repeated_rooms = repeated.building("DualIRShell").rooms
    alternate_rooms = alternate.building("DualIRShell").rooms

    assert [(room.identity, room.metaphor) for room in first_rooms] == [
        (room.identity, room.metaphor) for room in repeated_rooms
    ]
    assert [room.identity for room in first_rooms] == [
        room.identity for room in alternate_rooms
    ]
    assert any(
        left.metaphor != right.metaphor
        for left, right in zip(first_rooms, alternate_rooms)
    )
    assert all(
        room.metaphor in DEFAULT_ROOM_PALETTE[room.member_kind]
        for room in first_rooms
    )


def test_selected_up_and_down_recursion_creates_buildings_and_navigation_tracks():
    class Foundation:
        foundation_value: int

    class Archive(Foundation):
        class Annex:
            shelf_count: int

        annex: Annex

        def catalog(self, query: str) -> int:
            return len(query)

    world = build_introspective_world(
        Archive, depth_up=1, depth_down=1, seed="recursive",
    )
    names = {building.name for building in world.buildings()}
    assert names == {"Archive", "Foundation", "Annex"}
    assert {(track.relationship, world.building(track.target).name) for track in world.tracks} == {
        ("inherits", "Foundation"),
        ("contains-type", "Annex"),
    }
    assert all(isinstance(track, AbstractUITrack) for track in world.tracks)


def test_free_form_language_adorns_a_python_member_and_implies_direct_code():
    adornment = sentence(
        building("log archive"),
        remembers("the last complete launch"),
    )
    world = build_introspective_world(
        DualIRShell,
        adornments={"rollup_log": (adornment,)},
    )
    room = world.building("DualIRShell").room("rollup_log")
    assert isinstance(room, AbstractUIRoom)
    assert room.intentions[-1] is adornment
    assert room.implied_code[0].dialect == "python-expression"
    assert room.implied_code[0].source == "instance.rollup_log()"
    assert room.implied_code[0].executable


def test_ssa_class_definition_enters_the_same_world_objects_with_ssa_receipts():
    emission = ClassEmission(
        identity="demo.Pump",
        fields=(ClassFieldEmission("pressure", 0, type_name="float"),),
        methods=(ClassMethodEmission(
            name="set_pressure",
            function_reference=17,
            function_name="demo.Pump.set_pressure",
            qualified_name="demo.Pump.set_pressure",
            kind="method",
            is_static=False,
            receiver_position=0,
            receiver_fields=(),
            receiver_evidence="test",
            parameters=(MethodParameterEmission(
                "value", 1, 2, type_name="float",
            ),),
            body_available=True,
        ),),
        origin_language="python",
    )
    world = build_introspective_world(emission, seed="ssa")
    pump = world.building("Pump")
    assert pump.source_kind == "ssa"
    assert pump.room("pressure").implied_code[0].source == (
        "%value = load_field %receiver, slot 0"
    )
    method_code = pump.room("set_pressure").implied_code[0]
    assert method_code.dialect == "repository-ssa-intent"
    assert method_code.source == "%result = call @17(%receiver, %value)"
    assert not method_code.executable


def test_ssa_recursion_refuses_to_invent_missing_class_relationships():
    emission = ClassEmission("demo.Empty", (), ())
    with pytest.raises(ValueError, match="correlated class plan"):
        build_introspective_world(emission, depth_down=1)


def test_world_reports_exact_abstract_ui_objects_it_instantiates():
    world = build_introspective_world(DualIRShell)
    objects = world.objects()
    assert objects[0] is world
    assert sum(isinstance(item, AbstractUIRegion) for item in objects) == 1
    assert sum(isinstance(item, AbstractUIBuilding) for item in objects) == 1
    assert sum(isinstance(item, AbstractUIRoom) for item in objects) == len(
        world.building("DualIRShell").rooms
    )

