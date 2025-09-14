import logging
import src.common.tensors.autoautograd.whiteboard_runtime as wr
from src.common.tensors.abstraction import AbstractTensor


class _Node:
    def __init__(self):
        self.sphere = AbstractTensor.zeros(1, float)
        self.version = 0


class _Sys:
    def __init__(self):
        self.nodes = {0: _Node()}


def test_whiteboard_runs_probes(monkeypatch, caplog):
    sys = _Sys()
    job = wr._WBJob(
        job_id="j",
        op=None,
        src_ids=(0,),
        residual=AbstractTensor.ones(1),
        fn=lambda x, residual=None, **kw: x,
    )
    monkeypatch.setenv("WHITEBOARD_PROBES", "1")
    with caplog.at_level(logging.INFO):
        wr.run_batched_vjp(sys=sys, jobs=(job,))
    assert any("running autograd probes" in r.message for r in caplog.records)
