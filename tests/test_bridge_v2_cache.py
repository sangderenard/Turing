from src.common.tensors.autoautograd.integration import bridge_v2


class DummyNode:
    def __init__(self):
        self.p = 0
        self.param = (0, 0, 0)
        self.version = 0


class DummySys:
    def __init__(self):
        self.nodes = {0: DummyNode(), 1: DummyNode()}

    def impulse(self, *args, **kwargs):
        pass


def _stub_run_cached(sys, op_name, src_ids, **kwargs):
    return 0, tuple(0 for _ in src_ids)


def test_preactivation_cached(monkeypatch):
    monkeypatch.setattr(bridge_v2, "run_op_and_grads_cached", _stub_run_cached)
    calls = []

    def _fake_preactivate(sys, nid):
        calls.append(nid)
        return 0, {}

    monkeypatch.setattr(bridge_v2, "preactivate_src", _fake_preactivate)

    sys = DummySys()
    specs = [("noop", [0], 1, None, None), ("noop", [0, 1], 2, None, None)]
    bridge_v2.push_impulses_from_ops_batched(sys, specs)
    assert calls.count(0) == 1
    assert calls.count(1) == 1
