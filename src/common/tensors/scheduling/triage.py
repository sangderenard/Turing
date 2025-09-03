from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .op_queue import OpJob
from .results import OpResult, ResultSink


@dataclass(frozen=True)
class BinKey:
    """Shape-equivalence class for batching."""

    op: str
    k: int
    F: int
    weight: str


class TriageEngine:
    """Triage: cache hits → immediate results; misses → bins."""

    def __init__(self, *, whiteboard_runner: Any) -> None:
        """Create a triage engine with a whiteboard execution helper."""
        # TODO: Store the runner and prepare any caches.
        raise NotImplementedError

    def process(
        self,
        jobs: Sequence[OpJob],
        *,
        get_attr,
        get_attr_version,
        result_sink: ResultSink,
    ) -> None:
        """Process a batch of jobs, routing hits and misses appropriately."""
        # TODO: Implement batching, cache lookup, and execution.
        raise NotImplementedError

