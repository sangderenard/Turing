"""Host boundary checks; GL calls are recorded without opening a window."""

import builtins

import pytest

from src.rendering.opengl_render import renderer as gl


def test_external_context_renders_without_pygame_and_propagates_present_failure(monkeypatch):
    events = []
    original_import = builtins.__import__

    def deny_pygame(name, *args, **kwargs):
        if name == "pygame" or name.startswith("pygame."):
            raise AssertionError("external host imported pygame")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_pygame)
    monkeypatch.delenv("TURING_GL_CAPTURE_PATH", raising=False)
    monkeypatch.setattr(gl, "_link_program", lambda *args: 1)
    for name in tuple(vars(gl)):
        if name.startswith("gl") and callable(getattr(gl, name)):
            monkeypatch.setattr(
                gl, name, lambda *args, _name=name: events.append((_name, args))
            )

    def present():
        events.append(("present", ()))

    host = gl.RendererHost(
        present, lambda: 0.0,
        lambda lines, size: events.append(("overlay", (lines, size))),
    )
    renderer = gl.GLRenderer(host=host)
    renderer.set_overlay_text(["attempt 17"])
    renderer.draw((800, 600))
    assert events[-2:] == [
        ("overlay", (("attempt 17",), (800, 600))), ("present", ()),
    ]
    assert ("glEnable", (gl.GL_POINT_SPRITE,)) not in events
    assert ("glEnable", (gl.GL_POINT_SMOOTH,)) not in events

    def fail_present():
        raise RuntimeError("window closed before presentation")

    renderer = gl.GLRenderer(host=gl.RendererHost(fail_present, lambda: 0.0))
    with pytest.raises(RuntimeError, match="window closed before presentation"):
        renderer.draw((800, 600))
    renderer.set_overlay_text(["must remain visible"])
    with pytest.raises(RuntimeError, match="does not support text overlays"):
        renderer.draw((800, 600))
