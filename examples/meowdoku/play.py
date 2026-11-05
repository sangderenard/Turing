"""Play Meowdoku: the Python version, in a window.

    python examples/meowdoku/play.py [--n 5] [--seed 0] [--size 600]

The game is ``MeowdokuSession`` (one move is one dt-system round); the
board is the authored per-pixel law of ``shader.py`` evaluated on numpy,
the same law the WebGL shader is compiled from.  This window is the Python
shell of the control record: the pointer resolves to a board cell
(``pointer-cell-row``/``pointer-cell-col``), the left button is
``primary-action`` (place a cat), the right button ``secondary-action``
(toggle an X), and each click hands one ``MeowdokuControls`` to the round.

Keys: U undo, R reset, N new puzzle, Esc quit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT), str(ROOT / "examples")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from meowdoku.program import MeowdokuControls  # noqa: E402
from meowdoku.session import MeowdokuSession  # noqa: E402
from meowdoku.shader import render_reference, session_feeds  # noqa: E402

HUD = 44  # pixels of text under the board


class Game:
    """The window's state: a session, its rendered board, and the shell's
    sampling of the pointer and buttons into control records."""

    def __init__(self, n: int, seed: int, size: int):
        self.n, self.seed, self.size = n, seed, size
        self.session = MeowdokuSession(n=n, seed=seed)
        self.image: np.ndarray | None = None
        self.message = ""

    # -- the shell's sampling of its devices -----------------------------
    def pointer_cell(self, x: float, y: float) -> tuple[int, int] | None:
        if not (0 <= x < self.size and 0 <= y < self.size):
            return None
        cell = self.size / self.n
        return int(y // cell), int(x // cell)

    def click(self, x: float, y: float, button: int) -> bool:
        """One click: sample the pointer cell and the button into a control
        record and hand it to the round.  Returns whether a round ran."""
        where = self.pointer_cell(x, y)
        if where is None or self.session.won or self.session.lost:
            return False
        row, col = where
        controls = MeowdokuControls(
            select_row=float(row), select_col=float(col),
            primary_action=1.0 if button == 1 else 0.0,
            secondary_action=1.0 if button == 3 else 0.0,
        )
        if not (controls.primary_action or controls.secondary_action):
            return False
        hearts = self.session.hearts
        self.session.hand(controls)
        self.image = None
        if self.session.hearts < hearts:
            self.message = "wrong: a heart lost"
        elif controls.primary_action and self.session.cat[row, col] == 1.0:
            self.message = "a cat sits"
        else:
            self.message = ""
        return True

    def key(self, name: str) -> None:
        if name == "u":
            self.session.undo()
        elif name == "r":
            self.session.reset()
        elif name == "n":
            self.seed += 1
            self.session = MeowdokuSession(n=self.n, seed=self.seed)
        else:
            return
        self.image = None
        self.message = ""

    # -- the board, from the same law the shader is compiled from --------
    def board(self) -> np.ndarray:
        if self.image is None:
            feeds = session_feeds(self.session, self.size, self.size)
            self.image = (render_reference(feeds) * 255.0).astype(np.uint8)
        return self.image

    def status(self) -> str:
        s = self.session
        if s.won:
            return f"solved with {s.hearts} heart(s) left  ·  N new puzzle"
        if s.lost:
            return "no hearts left  ·  R reset, N new puzzle"
        return (f"hearts {s.hearts}  ·  cats to place {s.residual}  ·  "
                f"left: cat  right: X  ·  U undo  R reset  N new" +
                (f"  ·  {self.message}" if self.message else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play Meowdoku (Python version)")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=600)
    args = parser.parse_args(argv)

    import pygame

    game = Game(args.n, args.seed, args.size)
    pygame.init()
    screen = pygame.display.set_mode((args.size, args.size + HUD))
    pygame.display.set_caption("Meowdoku")
    font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    game.key(pygame.key.name(event.key))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                game.click(event.pos[0], event.pos[1], event.button)
        surface = pygame.surfarray.make_surface(np.transpose(game.board(), (1, 0, 2)))
        screen.fill((16, 24, 39))
        screen.blit(surface, (0, 0))
        screen.blit(font.render(game.status(), True, (232, 236, 244)), (10, args.size + 12))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
