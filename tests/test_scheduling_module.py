from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, List, Tuple

import pytest

from src.common.tensors.scheduling.runner import BulkOpRunner
from src.common.tensors.scheduling.triage import TriageEngine
from src.common.tensors.scheduling.op_queue import OpJob, OpQueue
from src.common.tensors.scheduling.results import QueueResultSink
from src.common.tensors.autoautograd.whiteboard_runtime import BatchVJPResult, BatchSlices


@dataclass
class _Node:
    param: Any
    sphere: Any
    version: int = 0


class _Sys:
    def __init__(self, params: List[Any], versions: List[int] | None = None) -> None:
        versions = versions or [0] * len(params)
        self.nodes = [_Node(param=t, sphere=t, version=v) for t, v in zip(params, versions)]


def _mk_jobs(n: int, *, k: int = 2, op: str = "sum_k", weight: str = "w0", tag: Any = None) -> List[OpJob]:
    jobs: List[OpJob] = []
    for j in range(n):
        jobs.append(
            OpJob(
                op=op,
                src_ids=tuple(range(k)),
                out_id=j,
                scale=1.0,
                residual=0.5,
                weight=weight,
                job_id=f"job-{op}-{j}",
                backend_tag=tag,
            )
        )
    return jobs


def _stub_batched_vjp(*, sys, jobs, op_args=(), op_kwargs=None, backend=None) -> BatchVJPResult:
    # Deterministic fake values per job
    ys = tuple(f"y:{j.job_id}" for j in jobs)
    grads_per_source = tuple(tuple(float(i) for i in range(len(j.src_ids))) for j in jobs)
    slices = BatchSlices(index_of={j.job_id: i for i, j in enumerate(jobs)}, job_ids=tuple(j.job_id for j in jobs))
    return BatchVJPResult(
        slices=slices,
        ys=ys,
        grads_full=tuple(None for _ in jobs),
        grads_per_source=grads_per_source,
        param_grads_full=tuple(None for _ in jobs),
        param_grads_tensor=None,
    )


def test_runner_cache_probe_and_update(monkeypatch):
    # Arrange a tiny system with scalar features (shape ())
    sys = _Sys([type("_S", (), {"shape": ()})(), type("_S", (), {"shape": ()})()])
    def get_attr(i: int):
        return sys.nodes[i].sphere
    def get_version(i: int) -> int:
        return sys.nodes[i].version

    runner = BulkOpRunner()
    # monkeypatch batched VJP to a stub
    import src.common.tensors.scheduling.runner as runner_mod
    monkeypatch.setattr(runner_mod, "run_batched_vjp", _stub_batched_vjp)

    # Single job
    job = _mk_jobs(1)[0]

    # Miss on empty cache
    assert runner.try_cached(job, get_attr=get_attr, get_attr_version=get_version, backend=None) is None

    # Compute via batched path (size 1)
    out = runner.run_bin(job.op, [job], get_attr=get_attr, get_attr_version=get_version, backend=None)
    assert len(out) == 1
    y, grads = out[0]
    assert y == f"y:{job.job_id}"
    assert grads == tuple(float(i) for i in range(len(job.src_ids)))

    # Now probe should hit and return the same package
    hit = runner.try_cached(job, get_attr=get_attr, get_attr_version=get_version, backend=None)
    assert hit == (y, grads)


def test_triage_bins_and_backend_scoping(monkeypatch):
    # System with vector features (shape (F,)) for F=3; triage infers F from shape
    sys = _Sys([type("_S", (), {"shape": (3,)})(), type("_S", (), {"shape": (3,)})()])
    def get_attr(i: int):
        return sys.nodes[i].sphere
    def get_version(i: int) -> int:
        return sys.nodes[i].version

    # Prepare runner and patch batched VJP, capturing backends used per call
    runner = BulkOpRunner()
    used_backends: List[Any] = []
    def _stub_with_backend(**kw):
        used_backends.append(kw.get("backend"))
        return _stub_batched_vjp(**kw)
    import src.common.tensors.scheduling.runner as runner_mod
    monkeypatch.setattr(runner_mod, "run_batched_vjp", _stub_with_backend)

    triage = TriageEngine(whiteboard_runner=runner, max_bin_size=16)
    sink = QueueResultSink()

    # Make two bins: same op/k/F but different backend_tag
    jobs_a = _mk_jobs(2, k=2, op="sum_k", weight="w0", tag="backend-A")
    jobs_b = _mk_jobs(3, k=2, op="sum_k", weight="w0", tag="backend-B")
    all_jobs = jobs_a + jobs_b

    class _Backend:
        def __init__(self, name): self.name = name
        @contextlib.contextmanager
        def scope(self):
            yield

    def resolve_backend(tag):
        return _Backend(tag)

    triage.process(
        all_jobs,
        get_attr=get_attr,
        get_attr_version=get_version,
        result_sink=sink,
        resolve_backend=resolve_backend,
        observe_shape=None,
    )

    # Results: one per job in order of publication
    results = sink.get_batch(10, timeout=0.1)
    assert len(results) == len(all_jobs)
    # Ensure each job_id appears once and has grads of expected length
    seen = {r.job_id for r in results}
    assert seen == {j.job_id for j in all_jobs}
    for r in results:
        assert isinstance(r.grads, tuple)
        assert len(r.grads) == len(all_jobs[0].src_ids)

    # Two batched runs (two bins) with distinct backends
    assert len(used_backends) == 2
    names = {b.name for b in used_backends}
    assert names == {"backend-A", "backend-B"}


def test_results_queue_basic():
    sink = QueueResultSink(maxsize=0)
    # Publish three results
    from src.common.tensors.scheduling.results import OpResult
    sink.publish(OpResult(job_id="j1", out_id=1, y=123, grads=(1.0,)))
    sink.publish(OpResult(job_id="j2", out_id=2, y=456, grads=(2.0, 3.0)))
    sink.publish(OpResult(job_id="j3", out_id=3, y=789, grads=()))
    batch = sink.get_batch(10, timeout=0.05)
    assert [r.job_id for r in batch] == ["j1", "j2", "j3"]


def test_op_queue_put_get_close():
    q = OpQueue(maxsize=0)
    jobs = _mk_jobs(3)
    for j in jobs:
        q.put(j)
    batch = q.get_batch(10, timeout=0.05)
    assert [j.job_id for j in batch] == [j.job_id for j in jobs]
    q.close()
    # After close and drain, further get_batch should return [] promptly
    assert q.get_batch(1, timeout=0.05) == []

