from __future__ import annotations

from typing import Any, Tuple


class TensorPool:
    """Preallocator keyed by ``(shape, dtype, device)``."""

    def acquire(self, shape: Tuple[int, ...], *, dtype=None, device=None) -> Any:
        """Return a tensor buffer with the requested specification."""
        # TODO: Implement pool lookup or allocate a new buffer.
        raise NotImplementedError

    def release(self, buf: Any) -> None:
        """Return a buffer to the pool."""
        # TODO: Implement buffer recycling.
        raise NotImplementedError

    def observe(self, shape: Tuple[int, ...], *, dtype=None, device=None) -> None:
        """Record allocation statistics for pre-warming."""
        # TODO: Track allocation patterns.
        raise NotImplementedError

