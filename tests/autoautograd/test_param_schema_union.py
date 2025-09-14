from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autoautograd.whiteboard_runtime import run_batched_vjp, _WBJob
import pytest


class _Node:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, AT.tensor(v))
        self.version = 0


class _Sys:
    def __init__(self):
        self.nodes = {
            0: _Node(alpha=1.0, w=2.0, b=3.0, kappa=0.0, l0=0.0),
            1: _Node(alpha=0.0, w=0.0, b=0.0, kappa=4.0, l0=5.0),
        }


def test_param_schema_union_grads():
    sys = _Sys()
    job1 = _WBJob(
        job_id="j1",
        op=None,
        src_ids=(0,),
        residual=AT.tensor(1.0),
        fn=lambda a, w, b, s, bias: s * (a * w + b) + bias,
        param_schema=("alpha", "w", "b"),
        fn_args=(AT.tensor(2.0),),
        fn_kwargs={"bias": AT.tensor(1.0)},
    )
    job2 = _WBJob(
        job_id="j2",
        op=None,
        src_ids=(1,),
        residual=AT.tensor(1.0),
        fn=lambda k, l, offset: k * l + offset,
        param_schema=("kappa", "l0"),
        fn_kwargs={"offset": AT.tensor(1.0)},
    )
    res = run_batched_vjp(sys=sys, jobs=(job1, job2))
    g = AT.get_tensor(res.grads_per_source_tensor)
    assert g.shape == (2, 5)
    # residuals are non-zero so each attribute in the schema receives a gradient
    assert all(float(g[0][i].item()) != 0.0 for i in range(3))
    assert all(float(g[1][i].item()) != 0.0 for i in range(3, 5))
    assert float(g[0][0].item()) == pytest.approx(8.0)
    assert float(g[0][1].item()) == pytest.approx(4.0)
    assert float(g[0][2].item()) == pytest.approx(4.0)
    assert float(g[0][3].item()) == pytest.approx(0.0)
    assert float(g[0][4].item()) == pytest.approx(0.0)
    assert float(g[1][0].item()) == pytest.approx(0.0)
    assert float(g[1][1].item()) == pytest.approx(0.0)
    assert float(g[1][2].item()) == pytest.approx(0.0)
    assert float(g[1][3].item()) == pytest.approx(10.0)
    assert float(g[1][4].item()) == pytest.approx(8.0)
