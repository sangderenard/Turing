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


def _stub_batched_vjp(*, sys, jobs, op_args, op_kwargs, backend):
    class _Batch:
        def __init__(self, n):
            self.ys = [0] * n
            self.grads_full = [[0] * len(job.src_ids) for job in jobs]

    return _Batch(len(jobs))


def test_preactivation_cached(monkeypatch):
    monkeypatch.setattr(bridge_v2, "run_batched_vjp", _stub_batched_vjp)
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
