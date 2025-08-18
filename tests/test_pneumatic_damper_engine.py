import numpy as np
from src.common.dt_system.classic_mechanics.engines import DemoState, PneumaticDamperEngine
from src.common.dt_system.state_table import StateTable

def test_pneumatic_damper_slows_relative_motion():
    s = DemoState(
        pos=[(0.0, 0.0), (1.0, 0.0)],
        vel=[(0.0, 0.0), (1.0, 0.0)],
        acc=[(0.0, 0.0), (0.0, 0.0)],
        mass=[1.0, 1.0],
        springs=[(0, 1)],
        rest_len={(0, 1): 1.0},
        k_spring={(0, 1): 0.0},
        pneu_damp={(0, 1): (1.0, 1.0)},
        ground_k=0.0,
    )
    table = StateTable()
    eng = PneumaticDamperEngine(s, state_table=table)
    eng.step(0.1, state_table=table)
    u0, u1 = eng.uuids
    assert table.identity_registry[u0]['acc'][0] > 0
    assert table.identity_registry[u1]['acc'][0] < 0
