from src.common.dt_system.classic_mechanics.demo import run_demo
from src.common.dt_system.state_table import StateTable


def test_demo_runs_and_prints_schedule(capsys):
    table = StateTable()
    run_demo(0.01, steps=1, state_table=table)
    out = capsys.readouterr().out
    assert "Process graph nodes" in out
    assert "ASAP levels" in out
    assert "Interference edges" in out
    assert "Frame 0 advanced" in out
    assert len(table.identity_registry) > 0
