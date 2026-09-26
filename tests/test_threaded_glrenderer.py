import time
import numpy as np

from src.opengl_render.api import make_threaded_draw_hook


class Recorder:
    """Simple stand-in for DebugRenderer that records frames."""

    def __init__(self):
        self.frames: list[dict] = []

    def print_layers(self, layers):
        self.frames.append(layers)


class FailingRecorder:
    def __init__(self):
        self.disposed = False

    def print_layers(self, layers):
        raise RuntimeError("draw failed")

    def dispose(self):
        self.disposed = True


def test_threaded_glrenderer_collects_history():
    rec = Recorder()
    hook, thread = make_threaded_draw_hook(rec, (1, 1), history=2)
    f1 = {"points": np.zeros((1, 3), np.float32)}
    f2 = {"points": np.ones((1, 3), np.float32)}
    hook(f1)
    thread.queue.join()
    hook(f2)
    thread.queue.join()
    thread.stop()
    assert list(thread.history) == [f1, f2]
    assert rec.frames == [f1, f2]


def test_threaded_glrenderer_loops():
    rec = Recorder()
    hook, thread = make_threaded_draw_hook(rec, (1, 1), history=1, loop_mode="loop")
    frame = {"points": np.zeros((1, 3), np.float32)}
    hook(frame)
    thread.queue.join()
    # Give the thread a moment to replay history
    time.sleep(0.05)
    thread.stop()
    assert len(rec.frames) >= 2


def test_threaded_glrenderer_bounces():
    rec = Recorder()
    hook, thread = make_threaded_draw_hook(rec, (1, 1), history=2, loop_mode="bounce")
    f1 = {"points": np.zeros((1, 3), np.float32)}
    f2 = {"points": np.ones((1, 3), np.float32)}
    hook(f1)
    thread.queue.join()
    hook(f2)
    thread.queue.join()
    time.sleep(0.05)
    thread.stop()
    # initial frames then bounced back to f1
    assert rec.frames[:2] == [f1, f2]
    assert len(rec.frames) >= 3 and rec.frames[2] == f1


def test_threaded_glrenderer_cleans_up_after_draw_failure():
    rec = FailingRecorder()
    hook, thread = make_threaded_draw_hook(rec, (1, 1), history=1)
    hook({"points": np.zeros((1, 3), np.float32)})
    thread.join(timeout=1.0)
    thread.stop()

    assert not thread.is_alive()
    assert isinstance(thread.last_error, RuntimeError)
    assert str(thread.last_error) == "draw failed"
    assert rec.disposed
