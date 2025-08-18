"""Pygame-based renderer for point, edge and triangle primitives.

This module keeps the rendering API tiny so higher-level systems can
convert arbitrary state tables into simple shape lists before drawing.
When pygame is not available (e.g. headless CI) the renderer can still be
imported but attempting to instantiate :class:`PygameRenderer` will raise a
:class:`RuntimeError`.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple, Optional

__all__ = ["PygameRenderer", "is_available"]

try:  # pragma: no cover - tolerate headless environments
    import pygame
except Exception:  # noqa: BLE001 - missing SDL libs
    pygame = None  # type: ignore


def is_available() -> bool:
    """Return ``True`` if pygame imported successfully."""
    return pygame is not None


class PygameRenderer:
    """Draw primitive shapes onto a pygame window.

    Parameters
    ----------
    width, height:
        Window size in pixels.
    screen:
        Optional existing :class:`pygame.Surface` to draw into.  When ``None``
        the renderer will create its own window.  This allows a higher level
        orchestrator to provide a window/context that downstream renderers can
        share.
    """

    def __init__(self, width: int, height: int, screen: Optional["pygame.Surface"] = None) -> None:
        if pygame is None:  # pragma: no cover - runtime guard
            raise RuntimeError("pygame is not available")
        if screen is None:
            pygame.init()
            screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Turing Renderer")
        self.screen = screen
        self.width = width
        self.height = height

    # -- drawing ---------------------------------------------------------
    def clear(self) -> None:
        if pygame is None:  # pragma: no cover - runtime guard
            return
        self.screen.fill((0, 0, 0))

    def draw(self, state: Dict[str, Iterable]) -> None:
        """Draw ``state`` then update the display.

        ``state`` may contain ``"points"``, ``"edges"`` and ``"triangles"``
        entries whose coordinates are in screen space (pixels).
        """

        if pygame is None:  # pragma: no cover - runtime guard
            return
        for x, y in state.get("points", []):
            pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), 3)
        for (x0, y0), (x1, y1) in state.get("edges", []):
            pygame.draw.line(
                self.screen, (200, 200, 200), (int(x0), int(y0)), (int(x1), int(y1)), 1
            )
        for tri in state.get("triangles", []):
            pts = [(int(px), int(py)) for px, py in tri]
            pygame.draw.polygon(self.screen, (120, 150, 200), pts, 1)
        pygame.display.flip()

    def close(self) -> None:
        if pygame is not None:  # pragma: no cover - runtime guard
            pygame.display.quit()
