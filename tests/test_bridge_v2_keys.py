from src.common.tensors.autoautograd.integration import bridge_v2


class DummyNode:
    def __init__(self):
        self.param = 0


class DummySys:
    def __init__(self):
        self.nodes = {0: DummyNode(), 1: DummyNode()}


def _stub_batched_vjp(*, sys, jobs, op_args, op_kwargs, get_attr, backend):
    class _Batch:
        def __init__(self, n):
            self.ys = [1] * n
            self.grads_per_source = [[0]] * n

    return _Batch(len(jobs))


def test_batched_forward_handles_list_kwargs(monkeypatch):
    monkeypatch.setattr(bridge_v2, "run_batched_vjp", _stub_batched_vjp)
    sys = DummySys()
    specs = [("noop", [0], 1, None, {"foo": [1, 2]})]
    assert bridge_v2.batched_forward_v2(sys, specs) == [1]
