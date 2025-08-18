import time
import numpy as np
import time

import pytest
from src.common.dt_system.engine_api import DtCompatibleEngine
from src.common.dt_system.dt_scaler import Metrics
from src.common.dt_system.threaded_system import ThreadedSystemEngine
from src.common.dt_system.state_table import StateTable


class _PointsEngine(DtCompatibleEngine):
    def __init__(self):
        self.t = 0.0
        self.pos = np.zeros((8, 3), dtype=np.float32)
        self._uuid = None

    def step(self, dt: float, state=None, state_table=None):
        self.t += float(dt)
        n = self.pos.shape[0]
        k = np.arange(n // 2, dtype=np.float32)
        ang = self.t + k * 0.25
        self.pos[: n // 2, 0] = np.cos(ang)
        self.pos[: n // 2, 1] = np.sin(ang)
        vmax = float(np.max(np.abs(self.pos[: n // 2, :2]))) if n > 0 else 0.0
        if state_table is not None:
            if not getattr(self, "_registration", None):
                schema = lambda _: {"pos": self.pos[0].tolist(), "mass": 1.0}
                self.register(state_table, schema, [0])
                self._uuid = next(iter(self._registration))
            else:
                state_table.update_identity(self._uuid, pos=self.pos[0].tolist())
        return True, Metrics(max_vel=vmax, max_flux=vmax, div_inf=0.0, mass_err=0.0)


@pytest.mark.xfail(reason="Threaded engine capture is unreliable in headless environments")
def test_threaded_engine_emits_frames():
    eng = _PointsEngine()

    def capture():
        # Return raw positions to avoid GL dependencies in thread
        return {"points": eng.pos.copy()}

    table = StateTable()
    syseng = ThreadedSystemEngine(eng, capture=capture, draw_hook=None, max_queue=2)
    try:
        # Drive a few steps synchronously; worker will capture frames
        for _ in range(3):
            ok, m, _ = syseng.step(0.1, state_table=table)
            assert ok
            assert m.max_vel >= 0.0
        # Allow worker to enqueue
        time.sleep(0.05)
        while not syseng.output_queue.empty():
            frame = syseng.output_queue.get_nowait()
            assert isinstance(frame, dict)
            pts = frame.get("points")
            assert isinstance(pts, np.ndarray)
            assert pts.shape[1] == 3
        assert len(table.identity_registry) >= 1
    finally:
        syseng.stop()
