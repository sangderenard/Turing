import pytest

from examples.chamber_raincloud_live import LiveChamber


@pytest.fixture(scope="module")
def live():
    return LiveChamber(
        frame_seconds=0.02, port_radius_m=0.08,
        ambient_rh=0.60, shape=(1, 1, 1))


def test_every_dewar_top_level_sim_uses_the_managed_dt_boundary(live):
    assert live.sim._registration
    assert live.machine_system._registration
    assert live.cycle_engine._registration
    assert live.fluid_system._registration
    assert live.electrical_system._registration
    assert live.thermal_system._registration


def test_real_dewar_participants_can_slip_independently(live):
    live.thermal_system.causal_ceiling_dt = lambda: 0.005

    live.step()

    assert live.sim.world_time == pytest.approx(0.02)
    assert live.machine_system.world_time == pytest.approx(0.02)
    assert live.cycle_engine.world_time == pytest.approx(0.02)
    assert live.fluid_system.world_time == pytest.approx(0.02)
    assert live.electrical_system.world_time == pytest.approx(0.02)
    assert live.thermal_system.world_time == pytest.approx(0.005)
    thermal = next(row for row in live.last_dt_profile
                   if row["name"] == "thermal-system")
    assert thermal["advanced_s"] == pytest.approx(0.005)
    assert thermal["slip_s"] == pytest.approx(0.015)
