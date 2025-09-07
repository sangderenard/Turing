import threading
import types
import sys


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
    sys.modules.setdefault(
        "src.common.tensors.autoautograd.whiteboard_cache", types.ModuleType("whiteboard_cache")
    )
    open_gl = types.ModuleType("OpenGL")
    open_gl.GL = types.ModuleType("GL")
    sys.modules.setdefault("OpenGL", open_gl)
    sys.modules.setdefault("OpenGL.GL", open_gl.GL)


def test_broken_op_log_emitted_once(capsys):
    _stub_dependencies()
    from src.common.tensors.abstraction import AbstractTensor
    from src.common.tensors.autoautograd.spring_async_toy import Experiencer

    class DummySys:
        pass

    exp = Experiencer(DummySys(), threading.Event(), {}, ops_program=[])
    g_list = [AbstractTensor.get_tensor([])]
    g_stack = AbstractTensor.stack(list(g_list), dim=0)
    exp._warn_broken_op("noop", 7, g_stack, g_list)
    exp._warn_broken_op("noop", 7, g_stack, g_list)
    out = capsys.readouterr().out
    assert out.count("[BROKEN-OP]") == 1
    assert "op=noop" in out and "out=7" in out
