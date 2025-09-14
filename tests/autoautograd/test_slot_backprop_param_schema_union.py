import pytest

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autoautograd.fluxspring import ParamWheel
from src.common.tensors.autoautograd.slot_backprop import SlotBackpropQueue
from src.common.tensors.autoautograd.whiteboard_runtime import (
    BatchSlices,
    BatchVJPResult,
    _WBJob,
)


def test_slot_backprop_param_schema_union():
    # five scalar parameters, each with its own attribute
    params = [AT.tensor(v) for v in (1.0, 2.0, 3.0, 4.0, 5.0)]
    for p in params:
        p.requires_grad_(True)
    wheels = [
        ParamWheel(params[0], lambda t: None, slots=1, label="n.ctrl.alpha"),
        ParamWheel(params[1], lambda t: None, slots=1, label="n.ctrl.w"),
        ParamWheel(params[2], lambda t: None, slots=1, label="n.ctrl.b"),
        ParamWheel(params[3], lambda t: None, slots=1, label="n.ctrl.kappa"),
        ParamWheel(params[4], lambda t: None, slots=1, label="n.ctrl.l0"),
    ]
    for w in wheels:
        w.rotate(); w.bind_slot()
    mgr = SlotBackpropQueue(wheels)

    job1 = _WBJob(
        job_id="j1",
        op=None,
        src_ids=(0, 1, 2),
        residual=AT.tensor(1.0),
        fn=lambda alpha, w_, b_: alpha * w_ + b_,
        param_schema=("alpha", "w", "b"),
    )
    job2 = _WBJob(
        job_id="j2",
        op=None,
        src_ids=(3, 4),
        residual=AT.tensor(1.0),
        fn=lambda kappa, l0: kappa * l0,
        param_schema=("kappa", "l0"),
    )
    mgr.queue_job(0, job1, param_schema=("alpha", "w", "b"))
    mgr.queue_job(0, job2, param_schema=("kappa", "l0"))

    def _stub_vjp(*, sys, jobs, **_kw):
        assert jobs[0].param_schema == ("alpha", "w", "b")
        assert jobs[1].param_schema == ("kappa", "l0")
        g = AT.tensor([2.0, 1.0, 1.0, 5.0, 4.0])
        return BatchVJPResult(
            slices=BatchSlices(
                index_of={j.job_id: i for i, j in enumerate(jobs)},
                job_ids=tuple(j.job_id for j in jobs),
            ),
            ys=tuple(AT.tensor(0.0) for _ in jobs),
            grads_full=tuple(AT.tensor(0.0) for _ in jobs),
            grads_per_source=tuple(() for _ in jobs),
            grads_per_source_tensor=g,
            param_grads_full=tuple(),
            param_grads_tensor=None,
        )

    mgr.process_slot(0, sys=None, lr=1.0, run_vjp=_stub_vjp)

    # Gradients should have been applied per wheel
    assert wheels[0].params[0].item() == pytest.approx(-1.0)
    assert wheels[1].params[0].item() == pytest.approx(1.0)
    assert wheels[2].params[0].item() == pytest.approx(2.0)
    assert wheels[3].params[0].item() == pytest.approx(-1.0)
    assert wheels[4].params[0].item() == pytest.approx(1.0)

