"""Central issuer for compiler-created representational IDs."""

from __future__ import annotations

from threading import Lock


class MonotonicIdSource:
    """A single non-repeating source of compiler-created integer IDs."""

    def __init__(self, first_id: int = 1_000_000_000) -> None:
        self._lock = Lock()
        self._next = int(first_id)

    def mint(self) -> int:
        with self._lock:
            value_id = self._next
            self._next += 1
            return value_id

    def mint_block(self, count: int) -> tuple[int, ...]:
        return tuple(self.mint() for _ in range(int(count)))

    def peek(self) -> int:
        with self._lock:
            return self._next


GLOBAL_MONOTONIC_IDS = MonotonicIdSource()

