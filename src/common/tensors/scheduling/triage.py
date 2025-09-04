# scheduling/triage.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Optional, Callable
from .op_queue import OpJob
from .results import OpResult, ResultSink

@dataclass(frozen=True)
class BinKey:
    """Shape-equivalence class for batching (include backend)."""
    op: str
    k: int
    F: int
    weight: str
    backend_tag: Optional[Any]

class TriageEngine:
    """
    Cache hits → results; misses → bins → whiteboard, largest-first.
    Runner API:
      try_cached(job, get_attr=..., get_attr_version=..., backend=None) -> Optional[(y, grads)]
      run_bin(op, jobs, get_attr=..., get_attr_version=..., backend=None) -> List[(y, grads)]
    """
    def __init__(self, *, whiteboard_runner: Any, max_bin_size: int = 256) -> None:
        self.runner = whiteboard_runner
        self.max_bin_size = int(max_bin_size)

    def process(
        self,
        jobs: Sequence[OpJob],
        *,
        get_attr: Callable[[int], Any],
        get_attr_version: Optional[Callable[[int], Optional[int]]],
        result_sink: ResultSink,
        resolve_backend: Optional[Callable[[Optional[Any]], Any]] = None,
        observe_shape: Optional[Callable[[Tuple[int, ...]], None]] = None,
    ) -> None:
        # 1) cache fast path
        misses: List[OpJob] = []
        for job in jobs:
            backend = resolve_backend(job.backend_tag) if resolve_backend else None
            hit = self.runner.try_cached(job, get_attr=get_attr, get_attr_version=get_attr_version, backend=backend)
            if hit is not None:
                y, grads = hit
                result_sink.publish(OpResult(job_id=job.job_id, out_id=job.out_id, y=y, grads=grads))
            else:
                misses.append(job)
        if not misses:
            return

        # 2) bin by (op, k, F, weight, backend_tag)
        bins: Dict[BinKey, List[OpJob]] = {}
        for job in misses:
            k = len(job.src_ids)
            F = self._infer_F(get_attr(job.src_ids[0]))
            key = BinKey(job.op, k, F, job.weight, job.backend_tag)
            bins.setdefault(key, []).append(job)

        # 3) process bins largest-first; cap batch size
        for key, bucket in sorted(bins.items(), key=lambda kv: len(kv[1]), reverse=True):
            backend = resolve_backend(key.backend_tag) if resolve_backend else None
            for start in range(0, len(bucket), self.max_bin_size):
                sub = bucket[start:start + self.max_bin_size]
                if observe_shape is not None:
                    B = len(sub); shape = (B, key.k) if key.F == 1 else (B, key.k, key.F)
                    observe_shape(shape)
                results = self.runner.run_bin(key.op, sub, get_attr=get_attr, get_attr_version=get_attr_version, backend=backend)
                if len(results) != len(sub):
                    raise RuntimeError("runner.run_bin returned mismatched result count")
                for job, (y, grads) in zip(sub, results):
                    result_sink.publish(OpResult(job_id=job.job_id, out_id=job.out_id, y=y, grads=grads))

    @staticmethod
    def _infer_F(sample_attr: Any) -> int:
        shp = getattr(sample_attr, "shape", None)
        if shp is None: return 1
        if isinstance(shp, tuple): return 1 if len(shp) == 0 else int(shp[-1] or 1)
        return 1
