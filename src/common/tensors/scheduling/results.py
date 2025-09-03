from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class OpResult:
    """Id-tagged result with forward value and local gradients."""

    job_id: str
    out_id: int
    y: Any
    grads: Any  # shape (k, F) aligned to src_ids


class ResultSink:
    """Result delivery: queue OR callback registry."""

    def publish(self, r: OpResult) -> None:
        """Deliver a result to its consumer."""
        # TODO: Implement publication to a queue or callback.
        raise NotImplementedError

