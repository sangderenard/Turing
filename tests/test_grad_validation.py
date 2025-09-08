import types
import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.autoautograd.whiteboard_runtime import run_batched_vjp


class _Node:
    def __init__(self, val):
        self.sphere = AbstractTensor.get_tensor(val)
        self.p = AbstractTensor.zeros(1, float)
        self.version = 0


class _Sys2:
    def __init__(self):
        self.nodes = {0: _Node([1.0, 2.0, 3.0]), 1: _Node([4.0, 5.0, 6.0])}


class _Sys3:
    def __init__(self):
        self.nodes = {
            0: _Node([1.0, 2.0, 3.0]),
            1: _Node([4.0, 5.0, 6.0]),
            2: _Node([7.0, 8.0, 9.0]),
        }


def test_grads_per_source_tensor_matches_reduction():
    import types

    sys = _Sys2()
    j0 = types.SimpleNamespace(job_id="j0", op="__neg__", src_ids=(0,), residual=None)
    j1 = types.SimpleNamespace(job_id="j1", op="__neg__", src_ids=(1,), residual=None)
    res = run_batched_vjp(sys=sys, jobs=(j0, j1))
    g = res.grads_per_source_tensor
    sums = tuple(float(x) for x in g.sum(dim=1))
    assert sums[0] == res.grads_per_source[0][0]
    assert sums[1] == res.grads_per_source[1][0]


def test_grads_full_consistent_with_stacked_tensor():
    import types

    sys = _Sys3()
    j0 = types.SimpleNamespace(job_id="j0", op="__neg__", src_ids=(0,), residual=None)
    j1 = types.SimpleNamespace(job_id="j1", op="__neg__", src_ids=(1,), residual=None)
    j2 = types.SimpleNamespace(job_id="j2", op="__neg__", src_ids=(2,), residual=None)
    res = run_batched_vjp(sys=sys, jobs=(j0, j1, j2))
    g = res.grads_per_source_tensor
    g0 = g[0]
    g1 = g[1]
    g2 = g[2]
    assert float((g0 - res.grads_full[0]).sum()) == 0.0
    assert float((g1 - res.grads_full[1]).sum()) == 0.0
    assert float((g2 - res.grads_full[2]).sum()) == 0.0

