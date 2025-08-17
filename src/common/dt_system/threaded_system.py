"""
Threaded system nodes for the dt-graph runner.

This module provides a wrapper that runs a DtCompatibleEngine inside a
background thread. The main dt-graph continues to control time (dt slices);
each step request is sent to the worker, which advances the engine, collects
metrics, and optionally emits a renderer-compatible "data tape" frame to an
output queue or a provided draw hook.

Design goals
------------
- Preserve dt-controller semantics: step(dt) still returns (ok, Metrics).
- Non-blocking frame emission with backpressure via a bounded queue.
- Minimal coupling: the capture function decides how to pack frames (plain
  numpy arrays for debug, or opengl_render.api layer dataclasses for GL).
- Clean shutdown: stop() signals the worker and joins the thread.
"""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
from typing import Any, Callable, Mapping, Optional

from .dt_scaler import Metrics
from .engine_api import DtCompatibleEngine
from .debug import dbg, is_enabled, pretty_metrics


CaptureFn = Callable[[], Mapping[str, Any]]


@dataclass
class _StepRequest:
    dt: float
    reply: "queue.Queue[tuple[bool, Metrics]]"


class ThreadedSystemEngine(DtCompatibleEngine):
    """Wrap a DtCompatibleEngine to run steps on a worker thread.

    Parameters
    ----------
    engine:
        The underlying engine implementing ``step(dt) -> (ok, Metrics)``.
    capture:
        Zero-arg callable invoked on the worker after each successful step to
        fetch a frame. The returned mapping is typically suitable for
        ``opengl_render.api.draw_layers`` (e.g., contains MeshLayer/PointLayer
        instances) but can be any consumer-defined dictionary for headless
        uses.
    draw_hook:
        Optional callable that consumes frames. When provided, frames are
        forwarded directly to this hook from the worker. When omitted, frames
        are written to ``output_queue`` for external consumption.
    max_queue:
        Maximum number of frames buffered in ``output_queue``. When full, the
        oldest frame is dropped to keep the queue responsive.
    """

    def __init__(
        self,
        engine: DtCompatibleEngine,
        *,
        capture: Optional[CaptureFn] = None,
        draw_hook: Optional[Callable[[Mapping[str, Any]], None]] = None,
        max_queue: int = 4,
        realtime: bool = False,
    ) -> None:
        self._engine = engine
        self._capture = capture or (lambda: {})
        self._draw_hook = draw_hook
        self._requests: "queue.Queue[_StepRequest | None]" = queue.Queue()
        self._outputs: "queue.Queue[Mapping[str, Any]]" = queue.Queue(maxsize=max(1, int(max_queue)))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._realtime = bool(realtime)
    # DtCompatibleEngine interface -------------------------------
    def step(self, dt: float) -> tuple[bool, Metrics]:
        """Synchronously request a worker step and wait for Metrics.

        The worker performs the compute and may enqueue a frame.
        """
        rep: "queue.Queue[tuple[bool, Metrics]]" = queue.Queue(maxsize=1)
        self._requests.put(_StepRequest(float(dt), rep))
        if is_enabled():
            dbg("threaded").debug(f"request: dt={float(dt):.6g}")
        try:
            ok, metrics = rep.get(timeout=10.0)
        except queue.Empty:
            # Treat timeout as a failed step with punitive metrics
            return False, Metrics(max_vel=0.0, max_flux=0.0, div_inf=1e9, mass_err=1e9)
        if is_enabled():
            dbg("threaded").debug(f"reply: ok={ok} metrics=({pretty_metrics(metrics)})")
        return ok, metrics

    def step_with_state(self, state, dt, *, realtime = False):
        return super().step_with_state(state, dt, realtime=realtime)

    # Lifecycle ---------------------------------------------------
    def stop(self) -> None:
        """Signal the worker to exit and wait for completion."""
        self._stop.set()
        try:
            self._requests.put_nowait(None)
        except Exception:
            pass
        self._thread.join(timeout=2.0)

    # Frame consumption ------------------------------------------
    @property
    def output_queue(self) -> "queue.Queue[Mapping[str, Any]]":
        """Queue of frames produced by the worker when no draw_hook is set."""
        return self._outputs

    # Internal worker --------------------------------------------
    def _emit_frame(self, frame: Mapping[str, Any]) -> None:
        if self._draw_hook is not None:
            try:
                self._draw_hook(frame)
            except Exception:
                pass
            return
        # Bounded queue with drop-oldest policy
        try:
            self._outputs.put_nowait(frame)
        except queue.Full:
            try:
                _ = self._outputs.get_nowait()
            except Exception:
                pass
            try:
                self._outputs.put_nowait(frame)
            except Exception:
                pass

    def _run(self) -> None:  # pragma: no cover - thread loop
        while not self._stop.is_set():
            try:
                req = self._requests.get(timeout=0.05)
            except queue.Empty:
                continue
            if req is None:
                break
            ok, metrics = False, Metrics(0.0, 0.0, 0.0, 0.0)
            try:
                if is_enabled():
                    dbg("threaded").debug(f"worker: step dt={req.dt:.6g}")
                # If in realtime mode and the engine exposes a realtime path,
                # prefer it to avoid nested superstep control.
                if self._realtime and hasattr(self._engine, "step_realtime"):
                    try:
                        ok, metrics = getattr(self._engine, "step_realtime")(req.dt)  # type: ignore[misc]
                    except Exception:
                        ok, metrics = self._engine.step(req.dt)
                else:
                    ok, metrics = self._engine.step(req.dt)
                # Emit frame best-effort regardless of ok
                frame = self._capture()
                if isinstance(frame, dict):
                    self._emit_frame(frame)
            except Exception:
                # On error, emit nothing and flag failure
                ok, metrics = False, Metrics(0.0, 0.0, 1e9, 1e9)
            finally:
                try:
                    req.reply.put_nowait((ok, metrics))
                except Exception:
                    pass


__all__ = ["ThreadedSystemEngine", "CaptureFn"]


# ---------------- Convenience capture builders ------------------

def capture_points(get_positions: Callable[[], Any], *, colors=None, sizes=None, default_size: float = 6.0) -> CaptureFn:
    """Return a capture() that packs a PointLayer from a positions getter.

    ``get_positions`` can return an array-like shaped (N,3) or (N,2);
    packing promotes to (N,3) float32. Colors/sizes are forwarded to
    opengl_render.api.pack_points.
    """

    def _cap() -> Mapping[str, Any]:
        from src.opengl_render.api import pack_points  # local import

        pos = get_positions()
        return {"points": pack_points(pos, colors=colors, sizes=sizes, default_size=default_size)}

    return _cap


def capture_cellsim_and_fluid(*, h=None, fluid_engine=None, rainbow: bool = False) -> CaptureFn:
    """Return a capture() that packs cellsim and/or fluid layers for OpenGL.

    ``h`` is a cellsim hierarchy object exposing ``cells`` with X/F. If you have
    a provider, pass ``h=provider._h``. ``fluid_engine`` can be any engine with
    a ``p`` attribute (particle positions) or a voxel grid; see fluid_layers.
    """

    def _cap() -> Mapping[str, Any]:
        from src.opengl_render.api import cellsim_layers, fluid_layers  # local import

        layers: dict[str, Any] = {}
        if h is not None:
            try:
                layers.update(cellsim_layers(h, rainbow=rainbow))
            except Exception:
                pass
        if fluid_engine is not None:
            try:
                layers.update(fluid_layers(fluid_engine, rainbow=rainbow))
            except Exception:
                pass
        return layers

    return _cap

