import pytest

from src.transmogrifier.cells.cell_consts import Cell
from src.transmogrifier.cells.simulator import Simulator
from src.transmogrifier.cells.simulator_methods import visualization as viz


def test_visualization_runs_without_pygame(monkeypatch):
    """visualise_step should execute even when pygame is unavailable."""
    # Simulate pygame being unavailable
    monkeypatch.setattr(viz, "pygame", None)
    monkeypatch.setattr(viz, "VISUALISE", False)
    viz._vis = None

    cells = [
        Cell(left=0, right=128, label="0", len=128, stride=128),
        Cell(left=128, right=256, label="1", len=128, stride=64),
        Cell(left=256, right=512, label="2", len=256, stride=32),
        Cell(left=512, right=768, label="3", len=256, stride=16),
    ]
    sim = Simulator(cells)

    viz.visualise_step(sim, cells)

    # nothing should have created a visualiser instance
    assert viz._vis is None
