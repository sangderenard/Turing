import math
from types import SimpleNamespace

import pytest

from examples.chamber_raincloud_live import OpenAirPort


class _Geometry:
    identity = "lab.dewar.closed_top"
    part = "lab.dewar"
    position = (0.0, 1.0, 0.0)
    direction = (0.0, 1.0, 0.0)
    radius_m = 0.0


def _puncture(*, through=True, radius=0.08):
    return SimpleNamespace(
        through=through,
        radius_m=radius,
        entry_position_m=(0.0, 1.05, 0.0),
        exit_position_m=(0.0, 1.0, 0.0),
    )


def test_through_damage_becomes_one_conserved_port_across_four_voxel_faces():
    demo = SimpleNamespace(WATER={"R_a": 287.0, "R_v": 461.5,
                                  "cv_a": 718.0, "cv_v": 1410.0})
    boundary = OpenAirPort(
        demo, cell_volume_m3=0.25 ** 3, shape=(4, 4, 4),
        bounds_min_m=(-0.5, 0.0, -0.5), bounds_max_m=(0.5, 1.0, 0.5),
        relative_humidity=0.6, geometry=_Geometry())
    damage = SimpleNamespace(
        graph={"nodes": [{"identity": "wall", "damage_opens_to": "chamber"}]},
        state=SimpleNamespace(part_damage={
            "wall": SimpleNamespace(punctures=[_puncture()])}))

    boundary.sync_damage_ports(damage, "chamber")

    assert boundary.damage_port_count == 1
    assert len(boundary.face_emitters) == 4
    assert boundary.damage_area_m2 == pytest.approx(math.pi * 0.08 ** 2)
    assert len({voxel for voxel, _ in boundary.face_emitters}) == 4


def test_blind_damage_does_not_change_the_atmosphere_boundary():
    demo = SimpleNamespace(WATER={"R_a": 287.0, "R_v": 461.5,
                                  "cv_a": 718.0, "cv_v": 1410.0})
    boundary = OpenAirPort(
        demo, cell_volume_m3=1.0, shape=(1, 1, 1),
        bounds_min_m=(-0.5, 0.0, -0.5), bounds_max_m=(0.5, 1.0, 0.5),
        relative_humidity=0.6, geometry=_Geometry())
    damage = SimpleNamespace(
        graph={"nodes": [{"identity": "wall", "damage_opens_to": "chamber"}]},
        state=SimpleNamespace(part_damage={
            "wall": SimpleNamespace(punctures=[_puncture(through=False)])}))

    boundary.sync_damage_ports(damage, "chamber")

    assert boundary.damage_port_count == 0
    assert boundary.face_emitters == ()
