import time
import numpy as np

from src.opengl_render.api import make_threaded_draw_hook


class Recorder:
    """Simple stand-in for DebugRenderer that records frames."""

    def __init__(self):
        self.frames: list[dict] = []

    def print_layers(self, layers):
        self.frames.append(layers)


def test_threaded_glrenderer_collects_history():
    rec = Recorder()
    hook, thread = make_threaded_draw_hook(rec, (1, 1), history=2)
    f1 = {"points": np.zeros((1, 3), np.float32)}
    f2 = {"points": np.ones((1, 3), np.float32)}
    hook(f1)
    hook(f2)
    thread.queue.join()
    thread.stop()
    assert list(thread.history) == [f1, f2]
    assert rec.frames == [f1, f2]


def test_threaded_glrenderer_loops():
    rec = Recorder()
    hook, thread = make_threaded_draw_hook(rec, (1, 1), history=1, loop=True)
    frame = {"points": np.zeros((1, 3), np.float32)}
    hook(frame)
    thread.queue.join()
    # Give the thread a moment to replay history
    time.sleep(0.05)
    thread.stop()
    assert len(rec.frames) >= 2
