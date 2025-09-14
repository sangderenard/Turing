"""Registry and utilities for backward algorithm methods.

This module provides a lightweight registry where tensor abstraction
can submit backward implementations for primitive operators.  The
registry can then assemble callable pipelines that execute these
backward operators in sequence, allowing simple composition of local
backward rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
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

# in backward.py

class BackwardRegistry:
    def register_from_backward_rules(self, rules: dict):
        """Register backward functions from BACKWARD_RULES using the 'python' key only."""
        # import here to avoid cycles
        from . import backward_registry as br

        # Everything the code snippets may reference:
        helper_globs = {
            # tensor surface
            "AbstractTensor": br.AbstractTensor,

            # registry helpers
            "unbroadcast": br.unbroadcast,
            "expand_to": br.expand_to,
            "indicator": br.indicator,
            "eps": br.eps,
            "T": br.T,

            # optional helper used by 'trace'
            "I_like": getattr(br, "I_like", None),
        }

        for opname, rule in rules.items():
            python_dict = rule.get("python", {})
            python_code = python_dict.get("body", "")
            if not python_code:
                continue

            parameters = python_dict.get("parameters", [])
            parameter_string = ", ".join(parameters)
            func_code = f"def bw_{opname}({parameter_string}):\n"
            for line in python_code.split(';'):
                func_code += f"    {line.strip()}\n"

            local_env = {}
            try:
                # <-- inject helpers here
                exec(func_code, helper_globs, local_env)
                fn = local_env[f"bw_{opname}"]
                self.register(f"{opname}", fn)
            except Exception as e:
                raise RuntimeError(f"Failed to register backward function for {opname}: {e}")

    def __init__(self) -> None:
        self._methods: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a backward implementation under ``name``."""

        from . import backward_registry as br

        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            checked = [br.coerce_to_tensor(a) for a in args]
            return fn(*checked, **kwargs)

        self._methods[name] = wrapped

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


from .backward_registry import BACKWARD_RULES
BACKWARD_REGISTRY.register_from_backward_rules(BACKWARD_RULES)

# Expose permute backward function for direct imports
bw_permute = BACKWARD_REGISTRY._methods["permute"]
