from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Optional, Sequence, Tuple


class WhiteboardMode:
    """Single-backward, non-accumulating tape scope."""

    def __init__(self) -> None:  # pragma: no cover - trivial
        """Prepare any context state."""
        # TODO: Initialize resources required for whiteboard execution.
        raise NotImplementedError

    def __enter__(self) -> "WhiteboardMode":  # pragma: no cover - context stub
        """Enter the whiteboard mode."""
        # TODO: Establish the environment for a single backward pass.
        raise NotImplementedError

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - context stub
        """Tear down the whiteboard mode."""
        # TODO: Clean up allocated resources.
        raise NotImplementedError


def run_op_and_grads_cached(
    *,
    op: str,
    src_ids: Sequence[int],
    weight: str,
    scale: float,
    residual: float,
    get_attr: Callable[[int], Any],
    get_attr_version: Optional[Callable[[int], Optional[int]]] = None,
    scalarize: Optional[Callable[[Any], Any]] = None,
) -> Tuple[Any, Any]:
    """Return forward value and local gradients, using a cache when possible.

    The implementation should consult a ``WhiteboardCache`` instance. On a cache
    miss, it must perform a single forward and backward pass within a
    ``WhiteboardMode`` context. The returned gradients align with ``src_ids``.
    """
    # TODO: Implement cache lookup and whiteboard execution.
    raise NotImplementedError

