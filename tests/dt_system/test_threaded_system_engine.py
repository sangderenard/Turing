import time
import numpy as np

from src.common.dt_system.engine_api import DtCompatibleEngine
from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.threaded_system import ThreadedSystemEngine


class _PointsEngine(DtCompatibleEngine):
    def __init__(self):
        self.t = 0.0
        self.pos = np.zeros((8, 3), dtype=np.float32)

    def step(self, dt: float):
        self.t += float(dt)
        # Circle in XY for first 4 pts; static others
        n = self.pos.shape[0]
        k = np.arange(n // 2, dtype=np.float32)
        ang = self.t + k * 0.25
        self.pos[: n // 2, 0] = np.cos(ang)
        self.pos[: n // 2, 1] = np.sin(ang)
        vmax = float(np.max(np.abs(self.pos[: n // 2, :2]))) if n > 0 else 0.0
        return True, Metrics(max_vel=vmax, max_flux=vmax, div_inf=0.0, mass_err=0.0)


def test_threaded_engine_emits_frames():
    eng = _PointsEngine()

    def capture():
        from src.opengl_render.api import pack_points

        # Use pack_points to ensure compatibility with renderer pipeline
        return {"points": pack_points(eng.pos)}

    syseng = ThreadedSystemEngine(eng, capture=capture, draw_hook=None, max_queue=2)
    try:
        # Drive a few steps synchronously; worker will capture frames
        for _ in range(3):
            ok, m = syseng.step(0.1)
            assert ok
            assert m.max_vel >= 0.0
        # Allow worker to enqueue
        time.sleep(0.05)
        got = 0
        while not syseng.output_queue.empty():
            frame = syseng.output_queue.get_nowait()
            assert isinstance(frame, dict)
            # Should contain a PointLayer-compatible object under 'points'
            pts = frame.get("points")
            # Avoid importing GL types in headless env; check attributes expected by draw_layers
            assert hasattr(pts, "positions")
            assert getattr(pts, "positions").shape[1] == 3
            got += 1
        assert got > 0
    finally:
        syseng.stop()
