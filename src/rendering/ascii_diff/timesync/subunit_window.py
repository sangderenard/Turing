#!/usr/bin/env python3
"""Pygame window for displaying pixel subunit grids."""
from __future__ import annotations

import numpy as np
# --- END HEADER ---

try:  # Optional dependency
    import pygame
    _HAS_PYGAME = True
except Exception:  # pragma: no cover - may be missing
    pygame = None  # type: ignore
    _HAS_PYGAME = False

class SubunitWindow:
    """Display a grid of pixel data using Pygame."""

    def __init__(self, grid_shape: tuple[int, int], subunit_size: int = 10) -> None:
        """Create a window sized for ``grid_shape``."""
        if not _HAS_PYGAME:
            raise RuntimeError(
                "pygame is required for SubunitWindow. Install timesync[gui]."
            )
        pygame.init()
        self.subunit_size = max(1, subunit_size)
        width = grid_shape[1] * self.subunit_size
        height = grid_shape[0] * self.subunit_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Subunit Grid")

    def update_subunits(
        self,
        changes: list[tuple[int, int, np.ndarray]],
    ) -> None:
        """Update the window with ``changes`` in-place."""
        if not _HAS_PYGAME:
            raise RuntimeError(
                "pygame is required for SubunitWindow. Install timesync[gui]."
            )
        for y, x, sub in changes:
            surf = pygame.surfarray.make_surface(sub.swapaxes(0, 1))
            surf = pygame.transform.scale(
                surf,
                (sub.shape[1] * self.subunit_size, sub.shape[0] * self.subunit_size),
            )
            self.screen.blit(surf, (x * self.subunit_size, y * self.subunit_size))
        pygame.display.update()

    def close(self) -> None:
        if _HAS_PYGAME:
            pygame.quit()
