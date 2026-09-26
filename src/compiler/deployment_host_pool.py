"""Persistent host worker pool executing deployment frames.

The ``pool`` strategy of ``deployment_lowering`` for the python and llvm
backends: a set of daemon threads that start once, park on a condition
variable between frames ("already up and waiting for jobs"), and claim work
by advancing a shared cursor -- the degenerate, provably-correct form of
work stealing for frames whose jobs are all known at deploy time.  Every
job is claimed exactly once because cursor advancement is atomic under the
pool lock; the join is a completion count on the same condition.

This gives real parallelism for the LLVM path today: an ``LLVMExecution``
entry is a ctypes call over ``(void**, int32*)`` buffers, and ctypes
releases the GIL for the duration, so lanes genuinely overlap.  Pure-Python
lanes still interleave under the GIL -- the pool stays correct there, just
not faster; that is the honest boundary, stated rather than hidden.

Join semantics follow ``DeploymentJoin`` exactly:

- BARRIER / INDEXED : wait for all lanes; results indexed by lane.
- REDUCE(op)        : fold lane results with the canonical associative
  operator.  Without ``allow_reassociation`` the fold is performed strictly
  in lane-index order (equal to the serial schedule, bit-for-bit for
  floats); with the license, the fold may combine partials in completion
  order.  Both are correct; only the licensed form is order-free.
- PRODUCT           : refused visibly -- it needs a dedicated consumer,
  not a generic pool (the same stance llvm_simd_deployment takes toward
  joins outside its validated set).

A lane that raises does not poison the pool: the exception is captured,
the frame completes, and ``deploy`` re-raises the first failure with its
lane index after all lanes have settled -- so no worker is left running
against torn state and the error names its lane.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import reduce as _fold
from typing import Any, Callable, Sequence

from .deployment_frame import DeploymentJoin, DeploymentJoinMode

_REDUCERS: dict[str, Callable[[Any, Any], Any]] = {
    "Add": lambda left, right: left + right,
    "Mul": lambda left, right: left * right,
    "Min": min,
    "Max": max,
    "And": lambda left, right: bool(left) and bool(right),
    "Or": lambda left, right: bool(left) or bool(right),
}


class LaneExecutionError(RuntimeError):
    """A lane raised; carries the lane index and the original exception."""

    def __init__(self, lane_index: int, original: BaseException):
        super().__init__(
            f"deployment lane {lane_index} raised "
            f"{type(original).__name__}: {original}"
        )
        self.lane_index = int(lane_index)
        self.original = original


@dataclass
class _Frame:
    jobs: Sequence[Callable[[], Any]]
    cursor: int = 0
    completed: int = 0
    results: list = None
    failures: list = None

    def __post_init__(self) -> None:
        self.results = [None] * len(self.jobs)
        self.failures = []


class HostDeploymentPool:
    """Persistent pool; one frame in flight at a time, caller participates.

    The caller thread drains jobs alongside the workers, which makes the
    pool correct even with ``workers=0`` (pure serial execution through the
    identical code path -- the serial fallback is not a separate
    implementation that could drift).
    """

    def __init__(self, workers: int | None = None):
        count = (
            max(0, (os.cpu_count() or 2) - 1)
            if workers is None else max(0, int(workers))
        )
        self._condition = threading.Condition()
        self._frame: _Frame | None = None
        # Workers park on the generation, not the frame object, so a
        # drained-but-not-yet-retired frame is never re-entered (which
        # would busy-spin) -- the same rule the C runtime documents.
        self._generation = 0
        self._closing = False
        self._workers = [
            threading.Thread(
                target=self._worker_loop,
                name=f"turing-deploy-worker-{index}",
                daemon=True,
            )
            for index in range(count)
        ]
        for worker in self._workers:
            worker.start()

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    def _worker_loop(self) -> None:
        seen_generation = 0
        while True:
            with self._condition:
                while not self._closing and (
                    self._frame is None
                    or self._generation == seen_generation
                ):
                    self._condition.wait()
                if self._closing:
                    return
                seen_generation = self._generation
                frame = self._frame
            self._drain(frame)

    def _drain(self, frame: _Frame) -> None:
        while True:
            with self._condition:
                if frame.cursor >= len(frame.jobs):
                    return
                index = frame.cursor
                frame.cursor += 1
            try:
                result = frame.jobs[index]()
            except BaseException as error:  # captured, settled at the join
                with self._condition:
                    frame.failures.append((index, error))
                    frame.completed += 1
                    if frame.completed == len(frame.jobs):
                        self._condition.notify_all()
                continue
            with self._condition:
                frame.results[index] = result
                frame.completed += 1
                if frame.completed == len(frame.jobs):
                    self._condition.notify_all()

    def deploy(
        self,
        lanes: Sequence[Callable[[], Any]],
        *,
        join: DeploymentJoin = DeploymentJoin(),
    ) -> Any:
        """Run every lane, then honor the join. Blocks until settled."""

        if join.mode is DeploymentJoinMode.PRODUCT:
            raise ValueError(
                "a PRODUCT join needs a dedicated consumer, not a generic "
                "worker pool; lower it explicitly"
            )
        jobs = list(lanes)
        if not jobs:
            return ()
        frame = _Frame(jobs=jobs)
        with self._condition:
            if self._closing:
                raise RuntimeError("pool is closed")
            if self._frame is not None:
                raise RuntimeError(
                    "one deployment frame at a time; nested deploys must "
                    "compose into the frame's own lanes"
                )
            self._frame = frame
            self._generation += 1
            self._condition.notify_all()
        # The caller is a worker too; with zero pool threads this line IS
        # the serial fallback.
        self._drain(frame)
        with self._condition:
            while frame.completed < len(frame.jobs):
                self._condition.wait()
            self._frame = None
            self._condition.notify_all()
        if frame.failures:
            frame.failures.sort(key=lambda item: item[0])
            lane_index, error = frame.failures[0]
            raise LaneExecutionError(lane_index, error) from error
        return self._join(frame, join)

    @staticmethod
    def _join(frame: _Frame, join: DeploymentJoin) -> Any:
        if join.mode in (
            DeploymentJoinMode.BARRIER, DeploymentJoinMode.INDEXED,
        ):
            return tuple(frame.results)
        reducer = _REDUCERS[join.reduction_operator]
        # Lane-index order: with reassociation licensed this order is one
        # of the permitted trees; without it, it is THE serial order.
        return _fold(reducer, frame.results)

    def deploy_span(
        self,
        kernel: Callable[[int, int], None],
        total: int,
        *,
        chunk: int | None = None,
    ) -> int:
        """Element-range splitting: run ``kernel(start, stop)`` over chunks.

        The BARRIER-join special case for one pointwise lane over a known
        extent -- the shape ``deployment_classification`` admits under its
        pointwise veto.  Returns the number of chunks executed.
        """

        total = int(total)
        if total <= 0:
            return 0
        width = max(1, int(chunk) if chunk else -(-total // max(
            1, (self.worker_count + 1) * 2
        )))
        spans = [
            (start, min(start + width, total))
            for start in range(0, total, width)
        ]
        self.deploy([
            (lambda span=span: kernel(span[0], span[1]))
            for span in spans
        ])
        return len(spans)

    def close(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        for worker in self._workers:
            worker.join(timeout=5.0)

    def __enter__(self) -> "HostDeploymentPool":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "HostDeploymentPool",
    "LaneExecutionError",
]
