"""Thread-backed helpers for the OpenGL renderer."""

from __future__ import annotations

from collections import deque
import queue
import threading
import time
from typing import Callable, Mapping


class GLRenderThread:
    """Run a renderer with a non-blocking latest-frame mailbox and history.

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
        # A single latest-frame mailbox prevents render lag and makes submit
        # non-blocking. Stale frames have no semantic value; history is formed
        # only from frames the renderer actually consumes.
        self.queue: "queue.Queue[Mapping[str, object] | None]" = queue.Queue(maxsize=1)
        norm = loop_mode.lower()
        if norm == "none":
            norm = "idle"
        if norm not in {"idle", "loop", "bounce"}:
            raise ValueError("loop_mode must be 'idle', 'loop' or 'bounce'")
        self.loop_mode = norm
        self.ghost_trail = ghost_trail
        self._stop = threading.Event()
        self.submitted_frames = 0
        self.dropped_frames = 0
        self.presented_frames = 0
        self.completed_hz = 0.0
        self._rate_started = time.perf_counter()
        self._rate_frame = 0
        self.last_error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=False)
        self._thread.start()

    # Public API -----------------------------------------------------
    @staticmethod
    def _release_layers(layers: Mapping[str, object] | None) -> None:
        if layers is None:
            return
        for layer in layers.values():
            release = getattr(layer, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass

    @staticmethod
    def _has_lease(layers: Mapping[str, object]) -> bool:
        return any(callable(getattr(layer, "release", None)) for layer in layers.values())

    def submit(self, layers: Mapping[str, object]) -> None:
        """Publish the newest frame without ever waiting for the renderer."""
        self.submitted_frames += 1
        try:
            self.queue.put_nowait(layers)
            return
        except queue.Full:
            pass
        self.dropped_frames += 1
        try:
            superseded = self.queue.get_nowait()
            self.queue.task_done()
            self._release_layers(superseded)
        except queue.Empty:
            pass
        try:
            self.queue.put_nowait(layers)
        except queue.Full:
            # Another producer won the latest-frame slot; either frame is
            # newer than the one currently being rendered, so never wait.
            self._release_layers(layers)

    def get_submit_hook(self) -> Callable[[Mapping[str, object]], None]:
        """Return a function that enqueues frames."""

        def hook(layers: Mapping[str, object]) -> None:
            self.submit(layers)

        return hook

    def stop(self, timeout: float | None = None) -> None:
        """Signal exit and wait for native renderer cleanup to complete."""
        self._stop.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout)

    # Lifecycle helpers ------------------------------------------------
    def is_alive(self) -> bool:
        """Return whether the internal render thread is alive."""
        return self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        """Block until the render thread exits (user closes the window)."""
        self._thread.join(timeout)

    # Internal worker ------------------------------------------------
    def _run(self) -> None:  # pragma: no cover - thread loop
        try:
            self._run_frames()
        except BaseException as exc:
            self.last_error = exc
            self._stop.set()
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        while True:
            try:
                pending = self.queue.get_nowait()
            except queue.Empty:
                break
            self._release_layers(pending)
            self.queue.task_done()
        if self.renderer is not None and hasattr(self.renderer, "dispose"):
            try:
                self.renderer.dispose()  # type: ignore[call-arg]
            except Exception:
                pass
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass

    def _run_frames(self) -> None:
        from .api import draw_layers, rainbow_history_points  # local import
        try:  # pragma: no cover - tolerate missing OpenGL libs
            from .renderer import PointLayer
        except Exception:  # noqa: BLE001
            PointLayer = None  # type: ignore

        def _record_present() -> None:
            self.presented_frames += 1
            now = time.perf_counter()
            elapsed = now - self._rate_started
            if elapsed >= 0.5:
                self.completed_hz = (
                    self.presented_frames - self._rate_frame
                ) / elapsed
                self._rate_started = now
                self._rate_frame = self.presented_frames

        def _pump_events() -> None:
            """Process pygame events and handle window closure."""
            try:  # pragma: no cover - headless environments
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._stop.set()
                        pygame.quit()
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
                        _pump_events()
                        if self._stop.is_set():
                            break
                        draw_layers(self.renderer, frame, self.viewport)  # type: ignore[arg-type]
                        _record_present()
                        time.sleep(1.0 / 60.0)
                else:
                    _pump_events()
                    if self._stop.is_set():
                        break
                    if self.renderer is not None and hasattr(self.renderer, "draw"):
                        try:
                            self.renderer.draw(self.viewport)  # type: ignore[call-arg]
                            _record_present()
                        except Exception:
                            pass
                    elif self.history:
                        frame = self.history[-1]
                        draw_layers(self.renderer, frame, self.viewport)  # type: ignore[arg-type]
                        _record_present()
                    time.sleep(1.0 / 60.0)
                continue

            # A leased CUDA observation page remains alive only through the
            # device-to-device copy. It is never retained as replay history.
            leased = self._has_lease(item)
            if not leased:
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
            try:
                draw_layers(self.renderer, frame, self.viewport)  # type: ignore[arg-type]
                _record_present()
            finally:
                if leased:
                    self._release_layers(item)
                self.queue.task_done()
