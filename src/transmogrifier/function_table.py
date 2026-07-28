"""Shared function references for ProcessGraph and SSA compilation.

The table owns function identity; individual IRs retain their own function
body representation.  A ProcessGraph call therefore carries a
``FunctionReference`` instead of embedding or copying its callee.  Backends may
later inline, specialize, or preserve that reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping


@dataclass(frozen=True, order=True)
class FunctionReference:
    """Opaque address of one entry in a :class:`FunctionTable`."""

    address: int


class FunctionResolutionState(str, Enum):
    """Resolution state used to expose recursive call backedges safely."""

    DECLARED = "declared"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    EXTERNAL = "external"


@dataclass
class FunctionEntry:
    """One named function and its available representations."""

    reference: FunctionReference
    name: str
    qualified_name: str
    state: FunctionResolutionState = FunctionResolutionState.DECLARED
    graph: Any = None
    python_callable: Callable[..., Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    implementations: dict[str, Callable[..., Any]] = field(
        default_factory=dict
    )
    recursive: bool = False


class StaticFunctionSlot:
    """Forward callable declaration used for recursive static definitions."""

    def __init__(self, reference: FunctionReference) -> None:
        self.reference = reference
        self._target: Callable[..., Any] | None = None

    def bind(self, target: Callable[..., Any]) -> None:
        self._target = target

    @property
    def bound(self) -> bool:
        return self._target is not None

    def __call__(self, *args, **kwargs):
        if self._target is None:
            raise RuntimeError(
                f"function reference {self.reference.address} was called "
                "before its static definition was assembled"
            )
        return self._target(*args, **kwargs)


class FunctionTable:
    """Addressable, recursion-aware function registry shared by compiler IRs."""

    def __init__(self) -> None:
        self._entries: dict[FunctionReference, FunctionEntry] = {}
        self._bindings: dict[str, FunctionReference] = {}
        self._qualified: dict[str, FunctionReference] = {}
        self._next_address = 0

    def declare(
        self,
        name: str,
        *,
        qualified_name: str | None = None,
        external: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> FunctionReference:
        """Declare a function or return its existing stable reference."""

        local_name = str(name)
        qualified = str(qualified_name or local_name)
        reference = self._qualified.get(qualified)
        if reference is None:
            reference = FunctionReference(self._next_address)
            self._next_address += 1
            self._entries[reference] = FunctionEntry(
                reference=reference,
                name=local_name,
                qualified_name=qualified,
                state=(
                    FunctionResolutionState.EXTERNAL
                    if external
                    else FunctionResolutionState.DECLARED
                ),
                metadata=dict(metadata or {}),
            )
            self._qualified[qualified] = reference
        else:
            entry = self._entries[reference]
            entry.metadata.update(dict(metadata or {}))
            if external and entry.state == FunctionResolutionState.DECLARED:
                entry.state = FunctionResolutionState.EXTERNAL
        self._bindings[local_name] = reference
        return reference

    def bind(self, name: str, reference: FunctionReference) -> None:
        """Bind another source-language name to an existing function."""

        self.entry(reference)
        self._bindings[str(name)] = reference

    def reference(self, name: str) -> FunctionReference | None:
        """Return the function reference bound to ``name``, if any."""

        return self._bindings.get(str(name))

    def entry(
        self,
        reference_or_name: FunctionReference | int | str,
    ) -> FunctionEntry:
        """Return one entry by opaque reference, integer address, or name."""

        if isinstance(reference_or_name, str):
            reference = self._bindings.get(reference_or_name)
            if reference is None:
                reference = self._qualified.get(reference_or_name)
            if reference is None:
                raise KeyError(f"unknown function {reference_or_name!r}")
        elif isinstance(reference_or_name, FunctionReference):
            reference = reference_or_name
        else:
            reference = FunctionReference(int(reference_or_name))
        try:
            return self._entries[reference]
        except KeyError as exc:
            raise KeyError(
                f"unknown function reference {reference.address}"
            ) from exc

    def begin_resolution(
        self,
        reference_or_name: FunctionReference | int | str,
    ) -> bool:
        """Mark resolution active; return false for a recursive backedge."""

        entry = self.entry(reference_or_name)
        if entry.state == FunctionResolutionState.RESOLVING:
            entry.recursive = True
            return False
        if entry.state == FunctionResolutionState.RESOLVED:
            return True
        entry.state = FunctionResolutionState.RESOLVING
        return True

    def resolve_graph(
        self,
        reference_or_name: FunctionReference | int | str,
        graph: Any,
    ) -> FunctionEntry:
        """Attach a graph body and complete structural resolution."""

        entry = self.entry(reference_or_name)
        entry.graph = graph
        entry.state = FunctionResolutionState.RESOLVED
        return entry

    def resolve_callable(
        self,
        reference_or_name: FunctionReference | int | str,
        function: Callable[..., Any],
    ) -> FunctionEntry:
        """Attach a contextual Python implementation."""

        entry = self.entry(reference_or_name)
        entry.python_callable = function
        if entry.graph is not None:
            entry.state = FunctionResolutionState.RESOLVED
        return entry

    def install_implementation(
        self,
        reference_or_name: FunctionReference | int | str,
        target: str,
        function: Callable[..., Any],
    ) -> None:
        """Cache one compiled target implementation for a function."""

        self.entry(reference_or_name).implementations[str(target)] = function

    def implementation(
        self,
        reference_or_name: FunctionReference | int | str,
        target: str,
    ) -> Callable[..., Any] | None:
        """Return a compiled implementation, if one has been installed."""

        return self.entry(reference_or_name).implementations.get(str(target))

    def __iter__(self):
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)


__all__ = [
    "FunctionEntry",
    "FunctionReference",
    "FunctionResolutionState",
    "FunctionTable",
    "StaticFunctionSlot",
]
