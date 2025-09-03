from __future__ import annotations

from typing import Any, Callable, Dict


class OpRegistry:
    """Maps op symbol → callable and its scalarization rule for gradients."""

    def __init__(self) -> None:
        """Create an empty operator registry."""
        # TODO: Initialize internal storage for ops and scalarizers.
        raise NotImplementedError

    def register(self, name: str, fn: Callable[..., Any], *, scalarize: Callable[[Any], Any]) -> None:
        """Register a new operator and its scalarization rule."""
        # TODO: Store the function and scalarization callable.
        raise NotImplementedError

    def get(self, name: str) -> Callable[..., Any]:
        """Return the operator callable associated with ``name``."""
        # TODO: Retrieve the operator implementation.
        raise NotImplementedError

    def get_scalarize(self, name: str) -> Callable[[Any], Any]:
        """Return the scalarization rule for ``name``."""
        # TODO: Retrieve the scalarization function.
        raise NotImplementedError

