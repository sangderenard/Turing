import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

pygame = pytest.importorskip("pygame")

from src.visualizations.reel_demo_shell import ReelDemoShell


def test_physical_activity_callback_moves_reels_without_inventing_position():
    pygame.init()
    try:
        shell = ReelDemoShell(pygame.Rect(0, 0, 720, 480))
        graphics = shell.reel_graphics
        graphics.total_tape = 100
        left_angle = graphics.left.angle
        right_angle = graphics.right.angle

        shell.update_status((80, 20), 0.025, True, False, "read")
        shell.update(1 / 60)

        assert graphics.position_driven
        assert graphics.playing and not graphics.recording
        assert (graphics.left_tape, graphics.right_tape) == (80, 20)
        assert graphics.left.angle != left_angle
        assert graphics.right.angle != right_angle

        read_angle = graphics.left.angle
        shell.update_status((60, 40), 0.050, False, False, "seek")
        shell.update(1 / 60)

        assert graphics.seeking
        assert (graphics.left_tape, graphics.right_tape) == (60, 40)
        assert graphics.left.angle != read_angle
        surface = pygame.Surface((720, 480))
        graphics.draw(surface)
        assert pygame.surfarray.array3d(surface).any()
    finally:
        pygame.quit()
