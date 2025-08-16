"""Thread-backed helpers for the OpenGL renderer."""

from __future__ import annotations

from collections import deque
import queue
import threading
import time
from typing import Callable, Mapping


class GLRenderThread:
    """Run a renderer in its own thread with a frame queue and history.

    Parameters
    ----------
    renderer_factory:
        Callable that returns an object with :func:`print_layers` or OpenGL
        ``draw`` methods. The renderer is constructed inside the render
        thread to ensure proper OpenGL context affinity.
    viewport:
        Tuple ``(width, height)`` describing the viewport in pixels.  When
        omitted, the renderer's internal default is used by the draw API.
    history:
        Maximum number of past frames to retain. ``0`` keeps an unbounded
        history.
    loop_mode:
        Behaviour when the input queue is empty. ``"idle"`` re-draws the last
        known frame. ``"loop"`` replays the stored history from start to end.
        ``"bounce"`` replays history forwards then backwards (ping-pong).
    """

    def __init__(
        self,
        renderer_factory: Callable[[], object],
        *,
        viewport: tuple[int, int] | None = None,
        history: int = 32,
        loop_mode: str = "idle",
        ghost_trail: bool = True,
    ) -> None:
        self._renderer_factory = renderer_factory
        self.renderer: object | None = None
        # May be None; draw_layers will fall back to renderer default
        self.viewport = viewport
        maxlen = history if history > 0 else None
        self.history: deque[Mapping[str, object]] = deque(maxlen=maxlen)
        self.queue: "queue.Queue[Mapping[str, object] | None]" = queue.Queue()
        norm = loop_mode.lower()
        if norm == "none":
            norm = "idle"
        if norm not in {"idle", "loop", "bounce"}:
            raise ValueError("loop_mode must be 'idle', 'loop' or 'bounce'")
        self.loop_mode = norm
        self.ghost_trail = ghost_trail
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # Public API -----------------------------------------------------
    def submit(self, layers: Mapping[str, object]) -> None:
        """Enqueue a new frame to be drawn by the thread."""
        self.queue.put(layers)

    def get_submit_hook(self) -> Callable[[Mapping[str, object]], None]:
        """Return a function that enqueues frames."""

        def hook(layers: Mapping[str, object]) -> None:
            self.submit(layers)

        return hook

    def stop(self) -> None:
        """Signal the thread to exit and wait for completion."""
        self._stop.set()
        self.queue.put(None)
        self._thread.join()

    # Internal worker ------------------------------------------------
    def _run(self) -> None:  # pragma: no cover - thread loop
        from .api import draw_layers, rainbow_history_points  # local import
        try:  # pragma: no cover - tolerate missing OpenGL libs
            from .renderer import PointLayer
        except Exception:  # noqa: BLE001
            PointLayer = None  # type: ignore

        def _pump_events() -> None:
            """Best-effort event pump to keep window responsive."""
            try:  # pragma: no cover - headless environments
                import pygame
                pygame.event.pump()
            except Exception:
                pass

        while not self._stop.is_set():
            # Lazily construct the renderer in this thread before any drawing
            if self.renderer is None:
                try:
                    self.renderer = self._renderer_factory()
                except Exception:
                    # If creation fails, sleep briefly and retry loop; allows
                    # environments without GL libs to progress/exit cleanly.
                    time.sleep(0.05)
                    continue
            try:
                item = self.queue.get(timeout=0.01)
            except queue.Empty:
                item = None

            if item is None:
                if self._stop.is_set():
                    break
                # queue empty or sentinel; replay history if requested
                if self.loop_mode in {"loop", "bounce"} and self.history:
                    seq = list(self.history)
                    if self.loop_mode == "bounce" and len(seq) > 1:
                        seq = seq + seq[-2:0:-1]
                    for frame in seq:
                        if self._stop.is_set():
                            break
                        _pump_events()
                        draw_layers(self.renderer, frame, self.viewport)  # type: ignore[arg-type]
                        time.sleep(1.0 / 60.0)
                else:
                    _pump_events()
                    if self.renderer is not None and hasattr(self.renderer, "draw"):
                        try:
                            self.renderer.draw(self.viewport)  # type: ignore[call-arg]
                        except Exception:
                            pass
                    elif self.history:
                        frame = self.history[-1]
                        draw_layers(self.renderer, frame, self.viewport)  # type: ignore[arg-type]
                    time.sleep(1.0 / 60.0)
                continue

            # Normal frame: draw and store in history
            self.history.append(item)
            frame = item
            if self.ghost_trail and PointLayer is not None:
                pts_hist = []
                for past in list(self.history)[:-1]:
                    pts = past.get("fluid") or past.get("points")
                    if isinstance(pts, PointLayer):
                        pts_hist.append(pts.positions)
                if pts_hist:
                    ghost = rainbow_history_points(pts_hist)
                    frame = dict(item)
                    frame["ghost"] = ghost
            draw_layers(self.renderer, frame, self.viewport)  # type: ignore[arg-type]
            self.queue.task_done()
        time.sleep(0.01)
