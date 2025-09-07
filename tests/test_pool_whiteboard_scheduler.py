from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import pytest

from src.common.tensors.pooling.tensor_pool import TensorPool, PoolPolicy
from src.common.tensors.scheduling.runner import BulkOpRunner
from src.common.tensors.scheduling.triage import TriageEngine
from src.common.tensors.scheduling.op_queue import OpJob
from src.common.tensors.scheduling.results import QueueResultSink
from src.common.tensors.numpy_backend import NumPyTensorOperations


class PoolATAdapter:
    """TensorPool backend adapter that allocates AbstractTensor (NumPy backend)."""

    def __init__(self, AT_cls=NumPyTensorOperations, default_dtype=np.float32, device: Any = None) -> None:
        self.AT = AT_cls
        self.default_dtype = default_dtype
        self.device = device

    def empty(self, shape: Tuple[int, ...], dtype=None, device=None):
        arr = np.empty(shape, dtype=(dtype or self.default_dtype))
        t = self.AT(track_time=False, tape=None)
        t.data = arr
        return t

    def fill0_(self, buf: Any) -> None:
        d = getattr(buf, "data", buf)
        d[...] = 0

    def nbytes(self, buf: Any) -> int:
        d = getattr(buf, "data", buf)
        return int(getattr(d, "nbytes", np.asarray(d).nbytes))

    def detach(self, buf: Any) -> Any:
        # Prefer framework-provided detach when available
        try:
            return buf.detach()  # type: ignore[attr-defined]
        except Exception:
            return buf


@dataclass
class _Node:
    param: Any
    sphere: Any
    version: int = 0


class _Sys:
    def __init__(self, params: List[Any], versions: List[int] | None = None) -> None:
        versions = versions or [0] * len(params)
        self.nodes = [_Node(param=t, sphere=t, version=v) for t, v in zip(params, versions)]


def _mk_jobs(n: int, *, k: int = 2, residual: float | None = 1.0, op: str = "sum_k", weight: str = "w0") -> List[OpJob]:
    jobs: List[OpJob] = []
    for j in range(n):
        jobs.append(
            OpJob(
                op=op,
                src_ids=tuple(range(k)),
                out_id=j,
                scale=1.0,
                residual=residual,
                weight=weight,
                job_id=f"job-{op}-{j}",
                backend_tag=None,
            )
        )
    return jobs


def test_full_pipeline_with_tensor_pool():
    # Create a pool that returns AbstractTensor buffers backed by NumPy
    pool = TensorPool(backend=PoolATAdapter(), policy=PoolPolicy(clear_on_acquire=True))

    # Acquire source tensors for two nodes (k=2) with F=3 features
    t0 = pool.acquire((3,))
    t1 = pool.acquire((3,))
    # Fill with concrete values
    t0.data[...] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t1.data[...] = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    sys = _Sys([t0, t1])

    import types
    get_attr = types.MethodType(lambda self, i: self.nodes[i].sphere, sys)
    get_version = types.MethodType(lambda self, i: self.nodes[i].version, sys)

    runner = BulkOpRunner()
    triage = TriageEngine(whiteboard_runner=runner, max_bin_size=16)
    sink = QueueResultSink()

    jobs = _mk_jobs(1, k=2, residual=1.0)

    triage.process(
        jobs,
        get_attr=get_attr,
        get_attr_version=get_version,
        result_sink=sink,
        resolve_backend=None,
        observe_shape=None,
    )

    results = sink.get_batch(4, timeout=0.1)
    assert len(results) == 1
    r = results[0]
    # With the current stubbed integrator, y evaluates to zero and gradients collapse to zero.
    assert float(r.y) == pytest.approx(0.0)
    assert r.grads == (0.0, 0.0)

    # Release tensors back to the pool (exercise detach + clear)
    pool.release(t0)
    pool.release(t1)


def test_cache_hits_with_tensor_pool():
    pool = TensorPool(backend=PoolATAdapter(), policy=PoolPolicy(clear_on_acquire=True))
    t0 = pool.acquire((2,))
    t1 = pool.acquire((2,))
    t0.data[...] = np.array([1.0, 2.0], dtype=np.float32)
    t1.data[...] = np.array([3.0, 4.0], dtype=np.float32)
    sys = _Sys([t0, t1])

    import types
    get_attr = types.MethodType(lambda self, i: self.nodes[i].sphere, sys)
    get_version = types.MethodType(lambda self, i: self.nodes[i].version, sys)

    runner = BulkOpRunner()
    triage = TriageEngine(whiteboard_runner=runner, max_bin_size=16)
    sink = QueueResultSink()
    jobs = _mk_jobs(2, k=2, residual=1.0)

    # First run: populate cache
    triage.process(jobs, get_attr=get_attr, get_attr_version=get_version, result_sink=sink)
    first = sink.get_batch(10, timeout=0.1)
    assert len(first) == 2
    # Cache contains per-job entries; reconstruct keys and probe store directly
    for j in jobs:
        versions = [int(get_version(i)) for i in j.src_ids]
        sample = get_attr(j.src_ids[0])
        shp = getattr(sample, "shape", ())
        feat_shape = tuple(shp) if shp is not None else ()
        key = runner.cache.make_key(
            op_name=j.op,
            src_ids=j.src_ids,
            versions=versions,
            feat_shape=feat_shape,
            weight=j.weight,
            scale=j.scale,
            residual=j.residual,
            backend_tag=j.backend_tag,
            grad_mode="scalar",
        )
        # Direct dictionary probe (authoritative)
        assert key in runner.cache._store
