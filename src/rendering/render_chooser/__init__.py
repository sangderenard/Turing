"""Renderer selection and state-table translation utilities.

``RenderChooser`` inspects the runtime environment and chooses between an
OpenGL, pygame or ASCII renderer.  Inputs are simple mappings describing
primitive shapes (points, edges, triangles) in screen space.  The chooser
translates these into the selected backend's preferred format and exchanges
frames with a :class:`~src.common.double_buffer.DoubleBuffer` to keep data
and rendering concerns on separate threads.
"""

from __future__ import annotations

import os
import select
import sys
import threading
import time
from typing import Any, Dict, Iterable, Tuple, List, Set
import numpy as np
from src.common.double_buffer import DoubleBuffer
from src.rendering.ascii_diff import (
    ThreadedAsciiDiffPrinter,
    full_clear_and_reset_cursor,
)

__all__ = ["RenderChooser"]


class RenderChooser:
    """Select a renderer backend, defaulting to ASCII."""

    def __init__(self, width: int, height: int, mode: str | None = None) -> None:
        self.width = width
        self.height = height
        self.mode = "ascii"
        self.renderer: Any
        self.screen = None
        self._ascii_printer: ThreadedAsciiDiffPrinter | None = None
        self._ascii_queue = None
        self._sync_event = threading.Event()

        preferred = (mode or os.environ.get("TURING_RENDERER", "ascii")).lower()

        if preferred == "auto":
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
                    return
            except Exception:
                pass
            try:  # pragma: no cover - headless environments
                from pygame_render import PygameRenderer, is_available
                import pygame

                if is_available():
                    pygame.init()
                    self.screen = pygame.display.set_mode((width, height))
                    self.renderer = PygameRenderer(width, height, self.screen)
                    self.mode = "pygame"
                    return
            except Exception:
                pass
            preferred = "ascii"

        if preferred == "opengl":
            try:  # pragma: no cover - best effort in headless CI
                from opengl_render import GLRenderer
                import pygame
                from pygame.locals import DOUBLEBUF, OPENGL

                pygame.init()
                pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
                if GLRenderer is not None:
                    self.renderer = GLRenderer(size=(width, height))  # type: ignore[call-arg]
                    self.mode = "opengl"
                    return
            except Exception:
                preferred = "pygame"

        if preferred == "pygame":
            try:  # pragma: no cover - headless environments
                from pygame_render import PygameRenderer, is_available
                import pygame

                if is_available():
                    pygame.init()
                    self.screen = pygame.display.set_mode((width, height))
                    self.renderer = PygameRenderer(width, height, self.screen)
                    self.mode = "pygame"
                    return
            except Exception:
                preferred = "ascii"

    from src.ascii_render import AsciiRenderer

    # Allow char cell and color settings to be passed via environment or args in future
    self.char_cell_pixel_height = 1
    self.char_cell_pixel_width = 1
    self.enable_fg_color = False
    self.enable_bg_color = False
    self.renderer = AsciiRenderer(width, height,
                     char_cell_pixel_height=self.char_cell_pixel_height,
                     char_cell_pixel_width=self.char_cell_pixel_width,
                     enable_fg_color=self.enable_fg_color,
                     enable_bg_color=self.enable_bg_color)
    self.mode = "ascii"
    self._ascii_printer = ThreadedAsciiDiffPrinter()
    self._ascii_queue = self._ascii_printer.get_queue()
    full_clear_and_reset_cursor()

    # Input and rendering thread state
    self._buffer = DoubleBuffer()
    self._events = []
    self._keys = set()
    self._lock = threading.Lock()
    self._running = True

    self._thread = threading.Thread(target=self._loop, daemon=True)
    self._thread.start()

    # ------------------------------------------------------------------
    def render(self, state: Dict[str, Any]) -> None:
        """Queue ``state`` for rendering on the worker thread."""

        self._buffer.write_frame(state)

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
        if self._ascii_printer is not None:
            self._ascii_printer.stop()
        if self.mode in ("pygame", "opengl"):
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def sync(self) -> None:
        """Block until pending frames have been rendered and printed."""
        if self.mode != "ascii" or self._ascii_queue is None:
            return
        self._sync_event.clear()
        # Send a sentinel frame that causes the worker thread to signal when
        # all previous frames have been processed.
        self.render({"__sync__": True})
        self._sync_event.wait()
        self._ascii_queue.join()

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        """Worker thread body: poll input and process render requests."""

        while self._running:
            self._poll_input()
            state = self._buffer.read_frame(agent_idx=1)
            if state is None:
                time.sleep(0.05)
                continue
            if state.get("__sync__"):
                self._sync_event.set()
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
        image = state.get("image")
        if image is not None:
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = arr[..., None]
            r.paint(arr)
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
        ascii_out = r.to_ascii_diff(
            char_cell_pixel_height=getattr(r, 'char_cell_pixel_height', 1),
            char_cell_pixel_width=getattr(r, 'char_cell_pixel_width', 1),
            enable_fg_color=getattr(r, 'enable_fg_color', False),
            enable_bg_color=getattr(r, 'enable_bg_color', False),
        )
        if ascii_out and self._ascii_queue is not None:
            self._ascii_queue.put(ascii_out)

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
