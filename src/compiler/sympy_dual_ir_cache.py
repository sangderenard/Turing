"""Persistent content-addressed cache for SymPy programs and repository IR.

The cache has two deliberately separate layers:

``solved-equations``
    The result of an expensive symbolic solve/reduction.  Callers supply the
    semantic solve record and a zero-argument producer.

``dual-ir``
    The backend-neutral repository compilation produced from a canonical
    equation program.  Backend emission is intentionally not part of this
    layer, so C, LLVM, WebAssembly, and GPU backends all reuse the same result.

Only compiler-owned objects are unpickled, from a repository-local cache whose
identity includes the implementation digest.  Writes use the existing atomic
AOT checkpoint store; corrupt or stale entries are misses and are rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from src.common.tensors.accelerator_backends.aot_checkpoint import (
    AOTCheckpointStore,
)


_CACHE_SCHEMA = "turing-sympy-dual-ir-v1"
_T = TypeVar("_T")


def _disabled() -> bool:
    return os.environ.get("TURING_DISABLE_SYMPY_DUAL_IR_CACHE", "").casefold() in {
        "1", "true", "yes", "on",
    }


def _configured_root() -> Path | None:
    value = os.environ.get("TURING_SYMPY_DUAL_IR_CACHE_DIR")
    return Path(value).expanduser().resolve() if value else None


@dataclass(frozen=True, slots=True)
class SympyCacheResult:
    """One cache lookup, including enough state for build diagnostics."""

    value: Any
    identity: str
    layer: str
    hit: bool
    status: str


class SympyDualIRCache:
    """Cache solved SymPy programs and their shared repository IR separately."""

    def __init__(
        self,
        implementation: str,
        *,
        root: str | Path | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.implementation = str(implementation)
        self.root = Path(root).expanduser().resolve() if root else _configured_root()
        self.enabled = not _disabled() if enabled is None else bool(enabled)

    def get_or_compute(
        self,
        layer: str,
        record: Mapping[str, Any],
        compute: Callable[[], _T],
    ) -> SympyCacheResult:
        """Load ``layer`` or atomically persist the value produced on a miss."""

        semantic_record = {
            "sympy_cache_schema": _CACHE_SCHEMA,
            "layer": str(layer),
            **dict(record),
        }
        store = AOTCheckpointStore(semantic_record, root=self.root)
        if not self.enabled:
            return SympyCacheResult(
                compute(), store.identity, str(layer), False, "disabled",
            )
        value = store.load(str(layer), self.implementation)
        if value is not None:
            return SympyCacheResult(
                value, store.identity, str(layer), True, store.last_load_status,
            )
        miss_status = store.last_load_status
        value = compute()
        store.store(str(layer), self.implementation, value)
        return SympyCacheResult(
            value, store.identity, str(layer), False, miss_status,
        )

    def solved_equations(
        self,
        record: Mapping[str, Any],
        solve: Callable[[], _T],
    ) -> SympyCacheResult:
        """Cache a solved/reduced equation-set program before IR lowering."""

        return self.get_or_compute("solved-equations", record, solve)

    def dual_ir(
        self,
        record: Mapping[str, Any],
        lower: Callable[[], _T],
    ) -> SympyCacheResult:
        """Cache repository IR independently of every eventual backend."""

        return self.get_or_compute("dual-ir", record, lower)


__all__ = ["SympyCacheResult", "SympyDualIRCache"]
