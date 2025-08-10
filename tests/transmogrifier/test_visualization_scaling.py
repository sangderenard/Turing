import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from src.transmogrifier.cells.cell_consts import Cell
from src.transmogrifier.cells.simulator import Simulator
from src.transmogrifier.cells.simulator_methods.visualization import _LCVisual


def test_scale_updates_with_cell_resize():
    cells = [
        Cell(stride=10, left=0, right=100, label="A"),
        Cell(stride=10, left=100, right=200, label="B"),
    ]
    sim = Simulator(cells)

    vis = _LCVisual(sim)
    initial = vis.scale_x

    cells[1].right = 400
    vis.draw()
    updated = vis.scale_x

    assert updated == pytest.approx(1200 / (400 - 0))
    assert updated < initial

    pygame.quit()
