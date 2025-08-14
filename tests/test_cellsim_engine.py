import pytest

from src.cells.cellsim.engine.saline import SalineEngine
from src.cells.cellsim.data.state import Cell, Bath


def test_engine_step_adapts_dt_and_conserves_nonneg():
    cells = [
        Cell(V=10.0, n={"Imp": 10.0}, base_pressure=1e4),
        Cell(V=10.0, n={"Imp": 0.0}, base_pressure=1e4),
    ]
    bath = Bath(V=100.0, n={"Na": 1500.0 * 100.0}, pressure=1e4)
    eng = SalineEngine(cells, bath)

    dt = 1e-3
    for _ in range(10):
        dt = eng.step(dt)
        assert dt > 0
        for c in cells:
            assert c.V >= 0
            for v in c.n.values():
                assert v >= 0
        assert bath.V >= 0
        for v in bath.n.values():
            assert v >= 0
