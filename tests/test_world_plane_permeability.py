import numpy as np
from src.common.dt_system.solids.api import WorldPlane, MATERIAL_SLIPPERY

def test_striped_permeability():
    plane = WorldPlane(
        normal=np.array([1.0, 0.0, 0.0]),
        offset=0.0,
        material=MATERIAL_SLIPPERY,
        fluid_mode="wrap",
        permeability=(4, 0),
    )
    assert plane.is_fluid_permeable(np.array([0.0, 0.1, 0.0]))
    assert not plane.is_fluid_permeable(np.array([0.0, 0.3, 0.0]))

def test_checkered_permeability():
    plane = WorldPlane(
        normal=np.array([1.0, 0.0, 0.0]),
        offset=0.0,
        material=MATERIAL_SLIPPERY,
        fluid_mode="wrap",
        permeability=(4, 4),
    )
    assert plane.is_fluid_permeable(np.array([0.0, 0.1, 0.1]))
    assert not plane.is_fluid_permeable(np.array([0.0, 0.1, 0.3]))
    assert plane.is_fluid_permeable(np.array([0.0, 0.3, 0.3]))


def test_world_plane_solver_warp():
    from src.common.dt_system.classic_mechanics.engines import DemoState, MetaCollisionEngine
    from src.common.dt_system.solids.api import WorldConfinement

    state = DemoState(
        pos=[(-0.1, 0.1)],
        vel=[(0.0, 0.0)],
        acc=[(0.0, 0.0)],
        mass=[1.0],
        springs=[],
        rest_len={},
        k_spring={},
        pneu_damp={},
    )
    plane = WorldPlane(
        normal=np.array([1.0, 0.0, 0.0]),
        offset=0.0,
        material=MATERIAL_SLIPPERY,
        fluid_mode="wrap",
        permeability=(4, 0),
    )
    world = WorldConfinement(planes=[plane])
    eng = MetaCollisionEngine([state], world=world)
    eng._resolve_world_planes()
    assert state.pos[0][0] > 0.0
