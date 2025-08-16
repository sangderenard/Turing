import src.opengl_render.api as api
from src.cells.softbody.demo.numpy_sim_coordinator import DtStats
import pytest


class DummyRenderer:
    def __init__(self) -> None:
        self.overlay = None

    def set_overlay_text(self, text):
        self.overlay = text

    def draw(self, viewport):  # pragma: no cover - side-effect free
        pass

    def set_mesh(self, mesh):  # pragma: no cover - side-effect free
        pass

    def set_lines(self, lines):  # pragma: no cover - side-effect free
        pass

    def set_points(self, pts):  # pragma: no cover - side-effect free
        pass

    def set_mvp(self, mvp):  # pragma: no cover - side-effect free
        pass


def test_hud_text_callable_updates_real_time(monkeypatch: pytest.MonkeyPatch):
    """``hud_text`` callables should be evaluated during drawing."""

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

    # Replace OpenGL-dependent layer classes with simple stand-ins so the
    # helper can run in this headless test environment.
    monkeypatch.setattr(api, "MeshLayer", _Dummy, raising=False)
    monkeypatch.setattr(api, "LineLayer", _Dummy, raising=False)
    monkeypatch.setattr(api, "PointLayer", _Dummy, raising=False)

    stats = DtStats()
    renderer = DummyRenderer()
    assert stats.real_t == 0.0
    api.draw_layers(renderer, {"hud_text": stats.lines})
    assert stats.real_t > 0.0
    assert renderer.overlay is not None
    assert any(line.startswith("real t:") for line in renderer.overlay)
