from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, Callable, Dict, Optional, List
from collections import deque
import threading
import time

@dataclass(frozen=True)
class OpResult:
    """Id-tagged result with forward value and local gradients."""
    job_id: str
    out_id: int
    y: Any
    grads: Any  # shape (k, F) aligned to src_ids

class ResultSink:
    """Result delivery: queue OR callback registry."""
    def publish(self, r: OpResult) -> None:  # pragma: no cover - interface
        raise NotImplementedError

class QueueResultSink(ResultSink):
    def __init__(self, maxsize: int = 0) -> None:
        self._q: deque[OpResult] = deque()
        self._cv = threading.Condition()
        self._maxsize = maxsize
        self._closed = False

    def publish(self, r: OpResult) -> None:
        with self._cv:
            if self._closed:
                return
            while self._maxsize and len(self._q) >= self._maxsize and not self._closed:
                self._cv.wait()
            if self._closed:
                return
            self._q.append(r)
            self._cv.notify_all()

    def get_batch(self, max_n: int, *, timeout: Optional[float] = None) -> List[OpResult]:
        deadline = None if timeout is None else (time.perf_counter() + timeout)
        with self._cv:
            while not self._q and not self._closed:
                if timeout is None:
                    self._cv.wait()
                else:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        return []
                    self._cv.wait(timeout=remaining)
                    if not self._q and (deadline - time.perf_counter()) <= 0:
                        return []
            if not self._q and self._closed:
                return []
            out: List[OpResult] = []
            for _ in range(min(max_n, len(self._q))):
                out.append(self._q.popleft())
            self._cv.notify_all()
            return out

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

class CallbackResultSink(ResultSink):
    def __init__(self) -> None:
        self._callbacks: Dict[str, Callable[[OpResult], None]] = {}

    def register(self, job_id: str, cb: Callable[[OpResult], None]) -> None:
        self._callbacks[job_id] = cb

    def publish(self, r: OpResult) -> None:
        cb = self._callbacks.get(r.job_id)
        if cb is not None:
            cb(r)
