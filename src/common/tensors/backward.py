"""Registry and utilities for backward algorithm methods.

This module provides a lightweight registry where tensor abstraction
can submit backward implementations for primitive operators.  The
registry can then assemble callable pipelines that execute these
backward operators in sequence, allowing simple composition of local
backward rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Any


@dataclass
class BackwardPipeline:
    """Callable wrapper that runs a series of backward operators.

    The functions are executed in the order they were provided.  Each
    function receives the result of the previous one.  The initial
    arguments supplied when calling the pipeline are forwarded to the
    first function.
    """

    functions: List[Callable[..., Any]]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result: Any = args
        for fn in self.functions:
            if isinstance(result, tuple):
                result = fn(*result, **kwargs)
            else:
                result = fn(result, **kwargs)
        return result


class BackwardRegistry:
    """Maintain a mapping of primitive names to backward callables."""

    def __init__(self) -> None:
        self._methods: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a backward implementation under ``name``."""
        self._methods[name] = fn

    def register_from_module(self, module: Any) -> "BackwardRegistry":
        """Discover and register all ``bw_*`` functions from ``module``.

        Returns the registry itself to allow chaining or storage by the
        caller.
        """
        for attr in dir(module):
            if attr.startswith("bw_"):
                self.register(attr[3:], getattr(module, attr))
        return self

    def build(self, sequence: Iterable[str]) -> BackwardPipeline:
        """Create a :class:`BackwardPipeline` for ``sequence`` of names."""
        funcs = [self._methods[name] for name in sequence if name in self._methods]
        return BackwardPipeline(funcs)


# Global registry used by the tensor abstraction
BACKWARD_REGISTRY = BackwardRegistry()
