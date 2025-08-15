"""Thread-backed helpers for the OpenGL renderer."""

from __future__ import annotations

from collections import deque
import queue
import threading
import time
from typing import Callable, Mapping, Tuple


class GLRenderThread:
    """Run a renderer in its own thread with a frame queue and history.

    Parameters
    ----------
    renderer:
        Object with :func:`print_layers` or OpenGL ``draw`` methods.
    viewport:
        Tuple ``(width, height)`` describing the viewport in pixels.
    history:
        Maximum number of past frames to retain. ``0`` keeps an unbounded
        history.
    loop:
        When ``True`` the stored history is replayed whenever the input queue is
        empty.
    bounce:
        When ``True`` and :paramref:`loop` is enabled, the history is played
        forwards then backwards (ping-pong).
    """

    def __init__(
        self,
        renderer: object,
        *,
        viewport: Tuple[int, int],
        history: int = 32,
        loop: bool = False,
        bounce: bool = False,
        ghost_trail: bool = True,
    ) -> None:
        self.renderer = renderer
        self.viewport = viewport
        maxlen = history if history > 0 else None
        self.history: deque[Mapping[str, object]] = deque(maxlen=maxlen)
        self.queue: "queue.Queue[Mapping[str, object] | None]" = queue.Queue()
        self.loop = loop
        self.bounce = bounce
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

        while not self._stop.is_set():
            try:
                item = self.queue.get(timeout=0.01)
            except queue.Empty:
                item = None

            if item is None:
                # queue empty or sentinel; replay history if requested
                if self.loop and self.history:
                    seq = list(self.history)
                    if self.bounce and len(seq) > 1:
                        seq = seq + seq[-2:0:-1]
                    for frame in seq:
                        if self._stop.is_set():
                            break
                        draw_layers(self.renderer, frame, self.viewport)
                        time.sleep(0.01)
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
            draw_layers(self.renderer, frame, self.viewport)
            self.queue.task_done()
        time.sleep(0.01)