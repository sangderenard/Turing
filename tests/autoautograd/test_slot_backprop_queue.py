import types

import pytest

from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autoautograd.fluxspring import ParamWheel
from src.common.tensors.autoautograd.slot_backprop import SlotBackpropQueue
from src.common.tensors.autoautograd.whiteboard_runtime import BatchSlices, BatchVJPResult


def test_slot_backprop_queue_applies_gradients_per_slot():
    # Two parameters each with two slots
    p0 = AT.tensor(1.0)
    p0.requires_grad_(True)
    p1 = AT.tensor(2.0)
    p1.requires_grad_(True)

    w0 = ParamWheel(p0, lambda t: None, slots=2)
    w1 = ParamWheel(p1, lambda t: None, slots=2)
    for w in (w0, w1):
        w.rotate(); w.bind_slot()  # activate slot 0

    mgr = SlotBackpropQueue([w0, w1])
    # Seed residual buffers for slot 0
    mgr.add_residual(0, main=AT.tensor(0.5))
    mgr.add_residual(0, spectral=AT.tensor(0.2))

    # Queue simple jobs for slot 0
    jobs = [types.SimpleNamespace(job_id=f"p{i}", op="__neg__", src_ids=(i,), residual=1.0) for i in range(2)]
    for j in jobs:
        mgr.queue_job(0, j)

    # Stub run_batched_vjp to return deterministic gradients
    def _stub_vjp(*, sys, jobs, **_kw):
        g = AT.tensor([2.0, 3.0])
        return BatchVJPResult(
            slices=BatchSlices(index_of={j.job_id: i for i, j in enumerate(jobs)}, job_ids=tuple(j.job_id for j in jobs)),
            ys=tuple(AT.tensor(0.0) for _ in jobs),
            grads_full=tuple(AT.tensor(0.0) for _ in jobs),
            grads_per_source=tuple(() for _ in jobs),
            grads_per_source_tensor=g,
            param_grads_full=tuple(),
            param_grads_tensor=None,
        )

    # Rotate wheels to slot 1, evicting slot 0
    ev0 = w0.rotate(); w0.bind_slot()
    ev1 = w1.rotate(); w1.bind_slot()
    assert ev0 == ev1 == 0

    mgr.process_slot(ev0, sys=None, lr=1.0, run_vjp=_stub_vjp)

    # Slot 0 parameters should have been updated; slot 1 untouched
    assert float(AT.get_tensor(w0.params[0])) == pytest.approx(-1.0)
    assert float(AT.get_tensor(w1.params[0])) == pytest.approx(-1.0)
    assert float(AT.get_tensor(w0.params[1])) == pytest.approx(1.0)
    assert float(AT.get_tensor(w1.params[1])) == pytest.approx(2.0)

    # Residual buffers and job queue for slot 0 should be cleared
    assert mgr.main_residuals[0] is None
    assert mgr.spectral_residuals[0] is None
    assert mgr.jobs[0] == []
