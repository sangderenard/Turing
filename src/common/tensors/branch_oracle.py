from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BranchOracle:
    """Global, minimal predicate override queue.

    - force_sequence: push a sequence of (op, outcome) pairs that will be
      consumed in order by elementwise predicate ops (equal/less/greater/and/or/xor).
    - maybe_mask: called by predicate implementations; if the next queued
      override matches op, returns the forced boolean outcome and consumes it.
    """

    _queue: List[Tuple[str, bool]] = field(default_factory=list)

    def reset(self) -> None:
        self._queue.clear()

    def force_sequence(self, seq: List[Tuple[str, bool]]) -> None:
        self._queue.extend([(str(op), bool(val)) for op, val in seq])

    def maybe_mask(self, op: str) -> Optional[bool]:
        if not self._queue:
            return None
        op = str(op)
        # Consume from the front only if op matches; otherwise do not force
        next_op, val = self._queue[0]
        if next_op == op:
            self._queue.pop(0)
            return val
        return None


# Global instance
BRANCH_ORACLE = BranchOracle()

