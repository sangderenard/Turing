from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.autoautograd.whiteboard_runtime import run_batched_vjp


class _Node:
    def __init__(self, val):
        from src.common.tensors.abstraction import AbstractTensor

        self.sphere = AbstractTensor.get_tensor(val)
        self.p = AbstractTensor.zeros(1, float)
        self.version = 0


class _Sys:
    def __init__(self):
        self.nodes = {0: _Node([1.0, 2.0, 3.0]), 1: _Node([4.0, 5.0, 6.0])}


def test_run_batched_vjp_returns_tensor_per_source():
    import types
    import pytest
    from src.common.tensors.autoautograd.whiteboard_runtime import run_batched_vjp

    sys_obj = _Sys()
    j0 = types.SimpleNamespace(job_id="j0", op="__neg__", src_ids=(0,), residual=None)
    j1 = types.SimpleNamespace(job_id="j1", op="__neg__", src_ids=(1,), residual=None)
    res = run_batched_vjp(sys=sys_obj, jobs=(j0, j1))
    g = res.grads_per_source_tensor
    assert getattr(g, "shape", None)[0] == 2
    with pytest.raises(ValueError):
        if g:
            pass


def test_run_batched_vjp_no_jobs_has_none_tensor():
    from src.common.tensors.autoautograd.whiteboard_runtime import run_batched_vjp

    sys_obj = _Sys()
    res = run_batched_vjp(sys=sys_obj, jobs=())
    assert res.grads_per_source_tensor is None

