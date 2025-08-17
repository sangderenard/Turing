import numpy as np
from src.common.dt_system.classic_mechanics.engines import DemoState, PneumaticDamperEngine

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
    eng = PneumaticDamperEngine(s)
    eng.step(0.1)
    assert s.acc[0][0] > 0
    assert s.acc[1][0] < 0
