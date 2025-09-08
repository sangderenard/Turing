import pytest

from src.common.tensors.abstraction import AbstractTensor
from src.common.tensors.autoautograd.whiteboard_runtime import run_op_and_grads_cached


class _Node:
    def __init__(self, value, param):
        self.p = AbstractTensor.get_tensor([value, value, value])
        self.param = AbstractTensor.get_tensor(param)
        self.sphere = AbstractTensor.concat([self.p, self.param], dim=0)
        self.version = 0


class _Sys:
    def __init__(self):
        self.nodes = {
            0: _Node(1.0, [0.0, 1.0, 1.0]),
            1: _Node(2.0, [0.0, 2.0, 1.0]),
        }


def test_gather_and_param_grads():
    sys = _Sys()
    indices = list(range(2))
    fn_specs = [
        (AbstractTensor.__mul__, slice(1, None, 3)),
        (AbstractTensor.__add__, slice(2, None, 3)),
    ]
    params_vec = AbstractTensor.concat(
        [sys.nodes[0].param.flatten(), sys.nodes[1].param.flatten()], dim=0
    ).flatten()
    param_lens = [len(sys.nodes[0].param.flatten()), len(sys.nodes[1].param.flatten())]
    _, g_param, _ = run_op_and_grads_cached(
        sys,
        "gather_and",
        [0, 1],
        op_args=(indices, fn_specs, params_vec),
        op_kwargs={"dim": 0},
        grad_mode="param",
        param_lens=param_lens,
    )
    g = AbstractTensor.get_tensor(g_param)
    assert getattr(g, "shape", None) == (2, 3)


def test_gather_and_dim_first_param_grads():
    sys = _Sys()
    indices = list(range(2))
    fn_specs = [
        (AbstractTensor.__mul__, slice(1, None, 3)),
        (AbstractTensor.__add__, slice(2, None, 3)),
    ]
    params_vec = AbstractTensor.concat(
        [sys.nodes[0].param.flatten(), sys.nodes[1].param.flatten()], dim=0
    ).flatten()
    param_lens = [len(sys.nodes[0].param.flatten()), len(sys.nodes[1].param.flatten())]
    _, g_param, _ = run_op_and_grads_cached(
        sys,
        "gather_and",
        [0, 1],
        op_args=(0, indices, fn_specs, params_vec),
        grad_mode="param",
        param_lens=param_lens,
    )
    g = AbstractTensor.get_tensor(g_param)
    assert getattr(g, "shape", None) == (2, 3)
