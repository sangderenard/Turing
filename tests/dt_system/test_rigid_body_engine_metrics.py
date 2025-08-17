import numpy as np
import pytest

from src.common.dt_system.classic_mechanics.rigid_body_engine import (
    RigidBodyEngine,
    WorldAnchor,
    WorldObjectLink,
    COM,
)
from src.common.dt_system.state_table import StateTable
from src.common.dt_system.dt_scaler import Metrics


def _make_engine():
    table = StateTable()
    # Single vertex registered in group "body0"
    uuid_v = table.register_identity((1.2, 0.0), 1.0)
    table.register_group("body0", {uuid_v})
    link = WorldObjectLink(
        world_anchor=WorldAnchor(position=(0.0, 0.0)),
        object_anchor=(0, "body0", COM, 1.0),
        link_type="spring",
        properties={"rest_length": 1.0, "k": 10.0},
    )
    eng = RigidBodyEngine([link], state_table=table, rigid_body_groups=[{"label": "body0", "vertices": {uuid_v}}])
    eng.velocities = np.array([[0.5, 0.0]])
    return eng, table


@pytest.mark.dt
@pytest.mark.fast
def test_rigid_body_engine_metrics_accumulate():
    eng, table = _make_engine()

    ok1, m1, _ = eng.step(0.1, state=None, state_table=table)
    assert ok1 and isinstance(m1, Metrics)
    assert m1.max_vel == pytest.approx(0.5)
    assert m1.div_inf > 0.0
    assert m1.dt_limit and m1.dt_limit > 0.0

    ok2, m2, _ = eng.step(0.1, state=None, state_table=table)
    assert ok2 and isinstance(m2, Metrics)
    assert m2.div_inf > m1.div_inf

