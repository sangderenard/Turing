import threading
import types
import sys
import time
from collections import defaultdict

import pytest

pytestmark = pytest.mark.xfail(
    reason="spring_async_toy transitioning to FluxSpring wrappers",
    strict=False,
)


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
    for name in [
        "DOUBLEBUF",
        "OPENGL",
        "RESIZABLE",
        "VIDEORESIZE",
        "QUIT",
        "KEYDOWN",
        "K_SPACE",
    ]:
        setattr(locals_mod, name, 0)
    pygame.locals = locals_mod
    pygame.display = types.SimpleNamespace(set_mode=lambda *a, **k: None)
    pygame.event = types.SimpleNamespace(get=lambda: [])
    pygame.init = lambda *a, **k: None
    pygame.quit = lambda *a, **k: None
    sys.modules.setdefault("pygame", pygame)
    sys.modules.setdefault("pygame.locals", locals_mod)

    mpl = types.ModuleType("matplotlib")
    mpl.__path__ = []
    mpl.pyplot = types.ModuleType("pyplot")
    mpl.animation = types.ModuleType("animation")
    mpl.colors = types.ModuleType("colors")
    sys.modules.setdefault("matplotlib", mpl)
    sys.modules.setdefault("matplotlib.pyplot", mpl.pyplot)
    sys.modules.setdefault("matplotlib.animation", mpl.animation)
    sys.modules.setdefault("matplotlib.colors", mpl.colors)


def test_geometry_residual_no_feature_impulse():
    _stub_dependencies()
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.autoautograd.spring_async_toy import (
        Experiencer,
        Node,
        BoundaryPort,
        SpringRepulsorSystem,
    )

    import src.common.tensors.autoautograd.integration.bridge_v2 as bridge
    import src.common.tensors.autoautograd.spring_async_toy as toy_mod

    def push_stub(sys, specs, weight=None, scale=1.0):
        y = AbstractTensor.get_tensor([0.0, 0.0])
        g_list = [AbstractTensor.get_tensor([1.0, 0.0])]
        return [y], [g_list], []

    bridge.push_impulses_from_ops_batched = push_stub
    toy_mod.filtered_poisson = lambda rhs, **kwargs: rhs

    zero = AbstractTensor.get_tensor([0.0, 0.0])
    node1 = Node(id=1, param=AbstractTensor.get_tensor(0.0), p=zero, v=zero, sphere=zero)
    node2 = Node(
        id=2,
        param=AbstractTensor.get_tensor(0.0),
        p=AbstractTensor.get_tensor([1.0, 0.0]),
        v=zero,
        sphere=zero,
    )
    sys_obj = SpringRepulsorSystem(nodes=[node1, node2], edges=[])
    port = BoundaryPort(nid=2, alpha=1.0, target_fn=lambda t: zero)
    sys_obj.add_boundary(port)
    sys_obj.edges = {}
    sys_obj.feedback_edges = []
    sys_obj.edge_locks = defaultdict(threading.Lock)

    calls = []

    def impulse_batch(src_ids, dst_id, op_id, g_scalars):
        calls.append((src_ids, dst_id, op_id, g_scalars))

    sys_obj.impulse_batch = impulse_batch

    stop = threading.Event()
    exp = Experiencer(
        sys_obj,
        stop,
        {},
        schedule_hz=100.0,
        ops_program=[("noop", [1], 2, None, None)],
    )

    exp.start()
    time.sleep(0.05)
    stop.set()
    exp.join()

    assert calls == []

