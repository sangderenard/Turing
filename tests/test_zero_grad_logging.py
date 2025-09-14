import logging
import src.common.tensors.autoautograd.whiteboard_runtime as wr
from src.common.tensors.abstraction import AbstractTensor


class _Node:
    def __init__(self):
        self.sphere = AbstractTensor.zeros(1, float)
        self.version = 0


class _Sys:
    def __init__(self, n):
        self.nodes = {i: _Node() for i in range(n)}


def test_zero_grad_warning_truncates_union_ids(caplog):
    sys = _Sys(10)
    job = wr._WBJob(
        job_id="j",
        op=None,
        src_ids=tuple(range(10)),
        residual=None,
        fn=lambda x, residual=None, **kw: x,
    )
    with caplog.at_level(logging.DEBUG):
        wr.run_batched_vjp(sys=sys, jobs=(job,))
    messages = [r.message for r in caplog.records if "WARNING zero grads" in r.message]
    assert messages, "expected zero-grads warning"
    msg = messages[0]
    assert "union_ids_len=10" in msg
    assert "union_ids_first=(0, 1, 2, 3, 4)" in msg
