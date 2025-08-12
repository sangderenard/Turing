from __future__ import annotations
from typing import Iterable, Iterator, TypeVar
from contextlib import contextmanager

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback if tqdm missing
    tqdm = None  # type: ignore

T = TypeVar("T")

class ProgressBar:
    """Utility to display nested progress bars.

    Uses :mod:`tqdm` if available; otherwise acts as a no-op iterator.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and tqdm is not None
        self._level = 0

    def iterate(self, iterable: Iterable[T], **kwargs) -> Iterator[T]:
        """Yield items from *iterable* while displaying a progress bar.

        Nested calls automatically stack using distinct ``position`` values.
        If ``tqdm`` is not available or progress bars are disabled this
        simply yields from the iterable without decoration.
        """
        if not self.enabled:
            for item in iterable:
                yield item
            return

        kwargs.setdefault("leave", True)
        bar = tqdm(iterable, position=self._level, **kwargs)
        self._level += 1
        try:
            for item in bar:
                yield item
        finally:
            self._level -= 1
            bar.close()

# Global instance used by all cellsim operations
progress = ProgressBar()
