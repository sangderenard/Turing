import pytest
import logging


from src.common.tensors.abstraction import AbstractTensor as AT
from src.common.tensors.autoautograd.fluxspring import ParamWheel
from src.common.tensors.autoautograd.slot_backprop import SlotBackpropQueue
from src.common.tensors.autoautograd.whiteboard_runtime import (
    BatchSlices,
    BatchVJPResult,
    _WBJob,
)


def test_slot_backprop_queue_applies_gradients_per_slot(tmp_path):
    log_file = tmp_path / "wb.log"
    handler = logging.FileHandler(log_file)
    root_logger = logging.getLogger("src.common.tensors.autoautograd")
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

    try:
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

        # Queue composite jobs for slot 0 with residuals baked in
        def _route_fn(_):
            return AT.tensor(0.0)

        def _fft_fn(_):
            return AT.tensor(0.0)

        jobs = [
            _WBJob(job_id="route", op=None, src_ids=(0,), residual=AT.tensor(0.5), fn=_route_fn),
            _WBJob(job_id="fft", op=None, src_ids=(1,), residual=AT.tensor(0.2), fn=_fft_fn),
        ]
        mgr.queue_job(0, jobs[0], param_schema=("p",), fn_args=(1,), fn_kwargs={"bias": 2})
        mgr.queue_job(0, jobs[1], kind="spectral", param_schema=("p",), fn_args=(3,), fn_kwargs={"bias": 4})

        # Stub run_batched_vjp to emulate composite ops and return gradients
        def _stub_vjp(*, sys, jobs, **_kw):
            assert callable(jobs[0].fn) and callable(jobs[1].fn)
            assert jobs[0].residual.item() == pytest.approx(0.5)
            assert jobs[1].residual.item() == pytest.approx(0.2)
            assert jobs[0].fn_args == (1,)
            assert jobs[0].fn_kwargs == {"bias": 2}
            assert jobs[1].fn_args == (3,)
            assert jobs[1].fn_kwargs == {"bias": 4}
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
        assert w0.params[0].item() == pytest.approx(-1.0)
        assert w1.params[0].item() == pytest.approx(-1.0)
        assert w0.params[1].item() == pytest.approx(1.0)
        assert w1.params[1].item() == pytest.approx(2.0)

        # Residual buffers and job queue for slot 0 should be cleared
        assert mgr.main_residuals[0] is None
        assert mgr.spectral_residuals[0] is None
        assert mgr.jobs[0] == []
        assert mgr.spectral_jobs[0] == []
    finally:
        root_logger.removeHandler(handler)

    log_text = log_file.read_text()
    assert "enqueue main job=route" in log_text
    assert "enqueue spectral job=fft" in log_text
    assert "g_tensor" in log_text
    assert "apply idx=0" in log_text


def test_slot_backprop_queue_slot_keying():
    """Jobs map via (tick - row_idx) % W."""

    p = AT.tensor(0.0)
    p.requires_grad_(True)
    w = ParamWheel(p, lambda t: None, slots=4)
    w.rotate(); w.bind_slot()

    mgr = SlotBackpropQueue([w])

    tick = 5
    row_main = 2
    row_spec = 1

    job_main = _WBJob(job_id="jm", op=None, src_ids=(0,), residual=AT.tensor(1.0), fn=lambda _: AT.tensor(0.0))
    job_spec = _WBJob(job_id="js", op=None, src_ids=(0,), residual=AT.tensor(2.0), fn=lambda _: AT.tensor(0.0))

    mgr.queue_job(None, job_main, tick=tick, row_idx=row_main, param_schema=("p",))
    mgr.queue_job(None, job_spec, tick=tick, row_idx=row_spec, kind="spectral", param_schema=("p",))

    slot_main = (tick - row_main) % 4
    slot_spec = (tick - row_spec) % 4

    assert mgr.jobs[slot_main][0] is job_main
    assert mgr.spectral_jobs[slot_spec][0] is job_spec
