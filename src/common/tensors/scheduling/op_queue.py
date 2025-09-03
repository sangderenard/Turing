from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence


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


class OpQueue:
    """Thread-safe FIFO / MPMC queue API for ``OpJob`` instances."""

    def put(self, job: OpJob) -> None:
        """Enqueue a job for later processing."""
        # TODO: Implement enqueue semantics.
        raise NotImplementedError

    def get_batch(self, max_n: int) -> List[OpJob]:
        """Retrieve up to ``max_n`` jobs in FIFO order."""
        # TODO: Implement batched dequeue semantics.
        raise NotImplementedError

