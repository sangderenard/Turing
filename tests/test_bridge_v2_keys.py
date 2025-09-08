from src.common.tensors.autoautograd.integration import bridge_v2


class DummyNode:
    def __init__(self):
        self.param = 0
        self.sphere = 0


class DummySys:
    def __init__(self):
        self.nodes = {0: DummyNode(), 1: DummyNode()}


def _stub_run_cached(*args, **kwargs):
    return 1, (0,), {}


def test_batched_forward_handles_list_kwargs(monkeypatch):
    monkeypatch.setattr(bridge_v2, "run_op_and_grads_cached", _stub_run_cached)
    sys = DummySys()
    specs = [("noop", [0], 1, None, {"foo": [1, 2]})]
    assert bridge_v2.batched_forward_v2(sys, specs) == [1]


def test_batched_forward_batches_equivalent_slices(monkeypatch):
    calls = []

    def _recording_stub(*args, **kwargs):
        calls.append(1)
        return 1, (0,), {}

    monkeypatch.setattr(bridge_v2, "run_op_and_grads_cached", _recording_stub)
    sys = DummySys()
    specs = [
        ("noop", [0], 1, None, {"foo": slice(0, 10, 2)}),
        ("noop", [0], 1, None, {"foo": slice(0, 10, 2)}),
    ]
    assert bridge_v2.batched_forward_v2(sys, specs) == [1, 1]
    assert len(calls) == 1
