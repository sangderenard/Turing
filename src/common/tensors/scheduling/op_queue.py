# scheduling/op_queue.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Any, TYPE_CHECKING
from collections import deque
import threading
import time
import traceback

if TYPE_CHECKING:
    from ..abstraction import AbstractTensor

@dataclass(frozen=True)
class OpJob:
    """Unit of work for the experiencer → worker path."""
    op: str
    src_ids: Tuple[int, ...]
    out_id: int
    scale: float
    residual: Optional[float]
    weight: str
    job_id: str
    # NEW: used for binning & runner backend scope (must be hashable)
    backend_tag: Optional[Any] = None

@dataclass(frozen=True)
class ThreadedOpJob(OpJob):
    """A job that runs inside its own dedicated thread."""
    backend_override: Optional['AbstractTensor'] = None
    refresh_rate: float = 60.0
    greedy: bool = True
    fn: Callable[['ThreadedOpJob'], None] = field(default=None)
    thread: Optional[threading.Thread] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        # auto-derive tag if not supplied
        if self.backend_tag is None and self.backend_override is not None:
            object.__setattr__(self, "backend_tag", id(self.backend_override))

class OpQueue:
    """Thread-safe FIFO / MPMC queue API for ``OpJob`` instances."""
    def __init__(self, *, maxsize: int = 0) -> None:
        self._q: deque[OpJob] = deque()
        self._cv = threading.Condition()
        self._maxsize = maxsize
        self._closed = False
    def put(self, job: OpJob) -> None:
        with self._cv:
            if self._closed:
                raise RuntimeError("OpQueue is closed")
            while self._maxsize and len(self._q) >= self._maxsize and not self._closed:
                self._cv.wait()
            if self._closed:
                raise RuntimeError("OpQueue is closed")
            self._q.append(job); self._cv.notify_all()
    def get_batch(self, max_n: int, *, timeout: Optional[float] = None) -> List[OpJob]:
        if max_n <= 0: raise ValueError("max_n must be positive")
        deadline = None if timeout is None else time.perf_counter() + timeout
        with self._cv:
            while not self._q and not self._closed:
                if timeout is None: self._cv.wait()
                else:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0: return []
                    self._cv.wait(timeout=remaining)
                    if not self._q and (deadline - time.perf_counter()) <= 0: return []
            if not self._q and self._closed: return []
            out: List[OpJob] = []
            for _ in range(min(max_n, len(self._q))):
                out.append(self._q.popleft())
            self._cv.notify_all()
            return out
    def close(self) -> None:
        with self._cv:
            self._closed = True; self._cv.notify_all()
    def empty(self) -> bool:
        with self._cv: return not self._q
    def qsize(self) -> int:
        with self._cv: return len(self._q)
    def is_closed(self) -> bool:
        with self._cv: return self._closed

class OpDaemon(OpQueue):
    """Background worker manager: drains queue + runs dedicated threaded jobs."""
    def __init__(self, *, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self._stop_evt = threading.Event()
        self._workers: list[threading.Thread] = []
        self._threaded_jobs: list[ThreadedOpJob] = []

    def start_workers(self, handler: Callable[[OpJob], None], *, num_workers: int = 1, batch: int = 1) -> None:
        if num_workers <= 0 or batch <= 0: raise ValueError
        def _loop(idx: int) -> None:
            try:
                while not self._stop_evt.is_set():
                    jobs = self.get_batch(batch, timeout=0.05)
                    if not jobs and self.is_closed(): break
                    for job in jobs:
                        try: handler(job)
                        except Exception: traceback.print_exc()
            finally: return
        for i in range(num_workers):
            t = threading.Thread(target=_loop, args=(i,), name=f"op-worker-{i}", daemon=True)
            t.start(); self._workers.append(t)

    def spawn_threaded(self, job: ThreadedOpJob) -> None:
        if job.fn is None: raise ValueError("ThreadedOpJob.fn must be provided")
        period = 1.0 / max(1e-6, float(job.refresh_rate))
        def _loop() -> None:
            next_t = time.perf_counter()
            try:
                while not self._stop_evt.is_set():
                    if job.greedy:
                        try: job.fn(job)
                        except Exception: traceback.print_exc()
                    else:
                        clone = OpJob(
                            op=job.op, src_ids=job.src_ids, out_id=job.out_id,
                            scale=job.scale, residual=job.residual, weight=job.weight,
                            job_id=job.job_id, backend_tag=job.backend_tag,
                        )
                        try: self.put(clone)
                        except RuntimeError: break
                    next_t += period
                    sleep_for = next_t - time.perf_counter()
                    if sleep_for > 0: time.sleep(sleep_for)
                    else: next_t = time.perf_counter()
            finally: return
        t = threading.Thread(target=_loop, name=f"op-threaded-{job.job_id}", daemon=True)
        object.__setattr__(job, "thread", t); t.start(); self._threaded_jobs.append(job)

    def stop(self) -> None:
        self._stop_evt.set(); self.close()
    def join(self, *, timeout: Optional[float] = None) -> None:
        for t in list(self._workers): t.join(timeout=timeout)
        for j in list(self._threaded_jobs):
            if j.thread is not None: j.thread.join(timeout=timeout)
