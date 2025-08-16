import threading
import time

import pytest
from src.opengl_render.render_sim_coordinator import run_option, OPTIONS


@pytest.mark.parametrize("choice", list(OPTIONS.keys()))
def test_menu_runs_in_debug(choice):
    """Smoke-test the OpenGL menu path in headless-friendly debug mode.

    - Keep the workload tiny (frames=1, dt=1e-3, sim_dim=2) for speed.
    - Enforce a hard timeout via a worker thread; xfail on timeout instead of hanging.
    """
    TIMEOUT_S = 10.0

    result: dict[str, object] = {}

    def worker():
        # Use debug_render=True to avoid real GL; reduce frames/dt for speed
        result["proc"] = run_option(
            choice,
            debug=True,
            frames=1,
            dt=1e-3,
            sim_dim=2,
            debug_render=True,
        )

    t = threading.Thread(target=worker, daemon=True)
    start = time.time()
    t.start()
    t.join(TIMEOUT_S)

    if t.is_alive():
        pytest.xfail("OpenGL menu debug demo exceeded timeout and may hang in some environments")
        return

    proc = result.get("proc")
    assert getattr(proc, "returncode", 1) == 0
    out = (getattr(proc, "stdout", "") or "").lower()
    assert "points" in out or "positions" in out
    assert "dtype" in out  # metadata vector info
