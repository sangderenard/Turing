"""Small Python FIFO matching the Nodus edge-runtime sequencing shape."""

from __future__ import annotations

from collections import deque
from typing import Any


class DeploymentFIFO:
    """Bounded single-reader FIFO with publish/consume/unread semantics."""

    def __init__(self, slots: int = 64) -> None:
        slots = int(slots)
        if slots <= 0:
            raise ValueError("FIFO slots must be positive")
        self.slots = slots
        self._items: deque[Any] = deque()
        self.write_sequence = 0
        self.read_sequence = 0

    @property
    def unread(self) -> int:
        return self.write_sequence - self.read_sequence

    def publish(self, value: Any) -> bool:
        """Publish without overwriting unread data; false means full."""

        if len(self._items) >= self.slots:
            return False
        self._items.append(value)
        self.write_sequence += 1
        return True

    def consume(self) -> tuple[bool, Any]:
        """Return ``(available, value)`` and advance only when available."""

        if not self._items:
            return False, None
        value = self._items.popleft()
        self.read_sequence += 1
        return True, value


__all__ = ["DeploymentFIFO"]
