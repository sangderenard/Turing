from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.autoautograd.whiteboard_runtime import (
    BatchSlices,
    BatchVJPResult,
    run_op_and_grads_cached,
)


class _Node:
    def __init__(self):
        self.sphere = AbstractTensor.zeros(3, float)
        self.p = AbstractTensor.zeros(1, float)
        self.version = 0


class _Sys:
    def __init__(self):
        self.nodes = {0: _Node()}


def test_run_op_and_grads_cached_none_grads(monkeypatch):
    def _fake_batched_vjp(**_kwargs):
        return BatchVJPResult(
            slices=BatchSlices(index_of={}, job_ids=("j",)),
            ys=(AbstractTensor.zeros(3, float),),
            grads_full=(None,),
            grads_per_source=((0.0,),),
        )

    import src.common.tensors.autoautograd.whiteboard_runtime as wr

    monkeypatch.setattr(wr, "run_batched_vjp", _fake_batched_vjp)
    sys = _Sys()
    _y, g_param, _meta = run_op_and_grads_cached(sys, "noop", [0], grad_mode="param")
    g_param = AbstractTensor.get_tensor(g_param)
    assert getattr(g_param, "shape", None) == (1, 2)
    assert float(g_param.sum()) == 0.0
