import io
import sys

from src.common.dt_system.classic_mechanics.demo import run_demo


def test_demo_runs_and_prints_schedule(capsys):
    run_demo(0.01, steps=1)
    out = capsys.readouterr().out
    assert "Process graph nodes" in out
    assert "ASAP levels" in out
    assert "Interference edges" in out
    assert "Frame 0 advanced" in out
