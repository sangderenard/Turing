import pytest


def test_scale_updates_with_cell_resize(monkeypatch):
    # The dummy driver belongs only to this headless visualization test.  A
    # module-level assignment leaked through pytest collection and disabled
    # every later OpenGL/GLSL test in the same process.
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    from src.cells.cell_consts import Cell
    from src.cells.simulator import Simulator
    from src.cells.simulator_methods.visualization import _LCVisual

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
