from __future__ import annotations
from typing import Callable, Dict, Any

class OpRegistry:
    def __init__(self) -> None:
        self._ops: Dict[str, Callable[[Any], Any]] = {}
    def register(self, name: str, fn: Callable[[Any], Any]) -> None:
        self._ops[name] = fn
    def get(self, name: str) -> Callable[[Any], Any]:
        return self._ops[name]

registry = OpRegistry()

registry.register("add", lambda x: x.sum())
registry.register("mul", lambda x: x.prod())
registry.register("sum", lambda x: x.sum())
