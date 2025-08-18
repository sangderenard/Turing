"""Renderer selection and state-table translation utilities.

``RenderChooser`` inspects the runtime environment and chooses between an
OpenGL, pygame or ASCII renderer.  Inputs are simple mappings describing
primitive shapes (points, edges, triangles) in screen space.  The chooser
translates these into the selected backend's preferred format.
"""

from __future__ import annotations

import queue
import select
import sys
import threading
from typing import Any, Dict, Iterable, Tuple, List, Set

__all__ = ["RenderChooser"]


class RenderChooser:
    """Select the most capable renderer available at runtime."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.mode = "ascii"
        self.renderer: Any
        self.screen = None

        # Try OpenGL first (uses pygame for window/context)
        try:  # pragma: no cover - best effort in headless CI
            from opengl_render import GLRenderer
            import pygame
            from pygame.locals import DOUBLEBUF, OPENGL

            pygame.init()
            pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
            if GLRenderer is not None:
                # TODO: GLRenderer currently creates its own window; we simply
                # establish a context here so that future versions can reuse it.
                self.renderer = GLRenderer(size=(width, height))  # type: ignore[call-arg]
                self.mode = "opengl"
            else:
                raise RuntimeError
        except Exception:
            # Fall back to pygame
            try:  # pragma: no cover - headless environments
                from pygame_render import PygameRenderer, is_available
                import pygame

                if is_available():
                    pygame.init()
                    self.screen = pygame.display.set_mode((width, height))
                    self.renderer = PygameRenderer(width, height, self.screen)
                    self.mode = "pygame"
                else:
                    raise RuntimeError
            except Exception:
                # Final fallback: ASCII
                from ascii_render import AsciiRenderer

                self.renderer = AsciiRenderer(width, height)
                self.mode = "ascii"

        # Input and rendering thread state
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._events: List[str] = []
        self._keys: Set[str] = set()
        self._lock = threading.Lock()
        self._running = True

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def render(self, state: Dict[str, Any]) -> None:
        """Queue ``state`` for rendering on the worker thread."""

        self._queue.put(state)

    # ------------------------------------------------------------------
    def poll_input(self) -> Tuple[Set[str], List[str]]:
        """Return currently held keys and recent discrete events."""

        with self._lock:
            keys = set(self._keys)
            events = list(self._events)
            self._events.clear()
            if self.mode == "ascii":
                # Without key-up events the best we can do is treat presses as
                # momentary; clear after each poll.
                self._keys.clear()
        return keys, events

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Stop the worker thread and release any backend resources."""

        self._running = False
        self._thread.join(timeout=1.0)
        try:
            close_fn = getattr(self.renderer, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass
        if self.mode in ("pygame", "opengl"):
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        """Worker thread body: poll input and process render requests."""

        while self._running:
            self._poll_input()
            try:
                state = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if self.mode == "opengl":  # pragma: no cover - requires GL context
                self._render_opengl(state)
            elif self.mode == "pygame":
                self.renderer.clear()
                self.renderer.draw(state)
            else:
                self._render_ascii(state)

    # ------------------------------------------------------------------
    def _poll_input(self) -> None:
        if self.mode in ("pygame", "opengl"):
            try:  # pragma: no cover - headless environments
                import pygame
            except Exception:
                return
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    with self._lock:
                        self._events.append("quit")
                elif event.type == pygame.KEYDOWN:
                    name = pygame.key.name(event.key)
                    with self._lock:
                        self._keys.add(name)
                        self._events.append(name)
                elif event.type == pygame.KEYUP:
                    name = pygame.key.name(event.key)
                    with self._lock:
                        self._keys.discard(name)
        else:
            try:
                if select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1)
                    with self._lock:
                        self._keys.add(ch)
                        self._events.append(ch)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _render_ascii(self, state: Dict[str, Any]) -> None:
        r = self.renderer
        r.clear()
        for x, y in state.get("points", []):
            r.point(int(x), int(y))
        for (x0, y0), (x1, y1) in state.get("edges", []):
            r.line(int(x0), int(y0), int(x1), int(y1))
        for tri in state.get("triangles", []):
            r.triangle(
                (int(tri[0][0]), int(tri[0][1])),
                (int(tri[1][0]), int(tri[1][1])),
                (int(tri[2][0]), int(tri[2][1])),
            )
        print(r.to_ascii())

    # ------------------------------------------------------------------
    def _render_opengl(self, state: Dict[str, Any]) -> None:
        try:  # pragma: no cover - requires GL libs
            import numpy as np
            from opengl_render import pack_points, draw_layers

            pts = np.array([[x, y, 0.0] for x, y in state.get("points", [])], np.float32)
            layers = {}
            if pts.size:
                layers["points"] = pack_points(pts)
            if layers:
                draw_layers(self.renderer, layers)  # type: ignore[arg-type]
        except Exception:
            # Best-effort: silently ignore when OpenGL not functional
            pass
