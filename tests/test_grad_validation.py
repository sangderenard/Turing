import threading
import types
import sys
import pytest


def _stub_dependencies():
    stub_bridge = types.ModuleType("bridge_v2")
    def _dummy(*args, **kwargs):
        return [], [], []
    stub_bridge.push_impulses_from_op_v2 = _dummy
    stub_bridge.push_impulses_from_ops_batched = _dummy
    pkg = types.ModuleType("integration")
    pkg.bridge_v2 = stub_bridge
    sys.modules.setdefault("src.common.tensors.autoautograd.integration", pkg)
    sys.modules.setdefault(
        "src.common.tensors.autoautograd.integration.bridge_v2", stub_bridge
    )

    wbc = types.ModuleType("whiteboard_cache")
    class WhiteboardCache:
        pass
    wbc.WhiteboardCache = WhiteboardCache
    sys.modules.setdefault(
        "src.common.tensors.autoautograd.whiteboard_cache", wbc
    )
    open_gl = types.ModuleType("OpenGL")
    open_gl.__path__ = []
    gl_mod = types.ModuleType("GL")
    gl_mod.__path__ = []
    shaders_mod = types.ModuleType("shaders")
    def _noop(*args, **kwargs):
        return None
    shaders_mod.compileProgram = _noop
    shaders_mod.compileShader = _noop
    gl_mod.shaders = shaders_mod
    gl_mod.GL_DYNAMIC_DRAW = 0
    sys.modules.setdefault("OpenGL", open_gl)
    sys.modules.setdefault("OpenGL.GL", gl_mod)
    sys.modules.setdefault("OpenGL.GL.shaders", shaders_mod)
    open_gl.GL = gl_mod
    pygame = types.ModuleType("pygame")
    pygame.__path__ = []
    locals_mod = types.ModuleType("locals")
    for name in ["DOUBLEBUF", "OPENGL", "RESIZABLE", "VIDEORESIZE", "QUIT", "KEYDOWN", "K_SPACE"]:
        setattr(locals_mod, name, 0)
    pygame.locals = locals_mod
    pygame.display = types.SimpleNamespace(set_mode=lambda *a, **k: None)
    pygame.event = types.SimpleNamespace(get=lambda: [])
    pygame.init = lambda *a, **k: None
    pygame.quit = lambda *a, **k: None
    sys.modules.setdefault("pygame", pygame)
    sys.modules.setdefault("pygame.locals", locals_mod)
    # matplotlib stubs
    mpl = types.ModuleType("matplotlib")
    mpl.__path__ = []
    mpl.pyplot = types.ModuleType("pyplot")
    mpl.animation = types.ModuleType("animation")
    mpl.colors = types.ModuleType("colors")
    sys.modules.setdefault("matplotlib", mpl)
    sys.modules.setdefault("matplotlib.pyplot", mpl.pyplot)
    sys.modules.setdefault("matplotlib.animation", mpl.animation)
    sys.modules.setdefault("matplotlib.colors", mpl.colors)


_stub_dependencies()

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.autoautograd.spring_async_toy import _stack_grads_per_source


def test_stack_grads_width_mismatch_raises():
    g1 = AbstractTensor.get_tensor([1.0, 2.0])
    g2 = AbstractTensor.get_tensor([1.0])
    with pytest.raises(ValueError) as e:
        _stack_grads_per_source("add", 3, [1, 2], [g1, g2])
    msg = str(e.value)
    assert "add" in msg and "output 3" in msg
    assert "source index 1" in msg


def test_stack_grads_missing_gradient_raises():
    g1 = AbstractTensor.get_tensor([1.0, 2.0])
    with pytest.raises(ValueError) as e:
        _stack_grads_per_source("mul", 5, [1, 2], [g1])
    msg = str(e.value)
    assert "mul" in msg and "output 5" in msg
    assert "source index 1" in msg
