import logging
import pytest
from types import SimpleNamespace

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autoautograd.fluxspring import ParamWheel, spiral_slot
from src.common.tensors.autoautograd.slot_backprop import SlotBackpropQueue
from src.common.tensors.autoautograd.whiteboard_runtime import _WBJob, BatchVJPResult, BatchSlices


def test_gradients_apply_to_matching_rows():
    p0 = AT.tensor(1.0); p0.requires_grad_(True)
    p1 = AT.tensor(2.0); p1.requires_grad_(True)

    w0 = ParamWheel(p0, lambda t: None, slots=2)
    w1 = ParamWheel(p1, lambda t: None, slots=2)
    for w in (w0, w1):
        w.rotate(); w.bind_slot()

    mgr = SlotBackpropQueue([w0, w1])
    tick = 5
    row_slots = [spiral_slot(tick, r, mgr.slots) for r in range(len(mgr.wheels))]

    def _fn(*_args):
        return AT.tensor(0.0)

    for row_idx in range(len(mgr.wheels)):
        job = _WBJob(
            job_id=f"j{row_idx}",
            op=None,
            src_ids=(row_idx,),
            residual=None,
            fn=_fn,
            param_schema=("p",),
        )
        mgr.add_residual(tick=tick, row_idx=row_idx, main=AT.tensor(1.0))
        mgr.queue_job(None, job, tick=tick, row_idx=row_idx, param_schema=("p",))

    def _stub_vjp(*, sys, jobs, **_kw):
        g = AT.zeros(len(mgr.wheels))
        for j in jobs:
            for sid in j.src_ids:
                g[int(sid)] = float(sid) + 1.0
        return BatchVJPResult(
            slices=BatchSlices(index_of={j.job_id: i for i, j in enumerate(jobs)}, job_ids=tuple(j.job_id for j in jobs)),
            ys=tuple(AT.tensor(0.0) for _ in jobs),
            grads_full=tuple(AT.tensor(0.0) for _ in jobs),
            grads_per_source=tuple(() for _ in jobs),
            grads_per_source_tensor=g,
            param_grads_full=tuple(),
            param_grads_tensor=None,
        )

    for slot in set(row_slots):
        mgr.process_slot(slot, sys=None, lr=1.0, run_vjp=_stub_vjp)

    assert w0.params[row_slots[0]].item() == pytest.approx(0.0)
    assert w1.params[row_slots[1]].item() == pytest.approx(0.0)
