"""One inspectable repository-SSA source for tensor implementations.

An :class:`SSATensorCodeReference` is compilation input.  It owns ordinary
repository SSA functions and states which entrypoints implement each canonical
tensor operation.  It has no invocation API: consumers copy/link the referenced
SSA and continue ordinary SSA compilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .ssa import Function, IRModule
from .ssa_registry import Handler


@dataclass(frozen=True, slots=True)
class SSATensorOperationReference:
    canonical_op: str
    entrypoints: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.canonical_op or not self.entrypoints:
            raise ValueError("SSA tensor operation references require an op and entrypoint")


@dataclass(frozen=True, slots=True)
class SSATensorCodeReference:
    """A complete, fully inspectable tensor implementation in repository SSA."""

    name: str
    module: IRModule
    operations: Mapping[str, SSATensorOperationReference]
    source_identity: str
    external_primitives: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.name or not self.source_identity:
            raise ValueError("SSA tensor code references require source identity")
        legal = {handler.value for handler in Handler}
        missing = {
            entrypoint
            for reference in self.operations.values()
            for entrypoint in reference.entrypoints
            if entrypoint not in self.module.functions
        }
        if missing:
            raise ValueError(
                f"SSA tensor code reference has missing entrypoints: {sorted(missing)!r}"
            )
        illegal = {
            str(instruction.op)
            for function in self.module.functions.values()
            for block in function.blocks.values()
            for instruction in block.instrs
            if str(instruction.op) not in legal
        }
        if illegal:
            raise ValueError(
                f"SSA tensor code reference uses non-repository operations: {sorted(illegal)!r}"
            )
        unresolved = {
            str(callee)
            for function in self.module.functions.values()
            for block in function.blocks.values()
            for instruction in block.instrs
            if (callee := instruction.attributes.get("callee"))
            and str(callee) not in self.module.functions
            and str(callee) not in self.external_primitives
        }
        if unresolved:
            raise ValueError(
                "SSA tensor code reference has undeclared source dependencies: "
                f"{sorted(unresolved)!r}"
            )

    def operation(self, canonical_op: str) -> SSATensorOperationReference | None:
        return self.operations.get(str(canonical_op))

    @property
    def primitive_entrypoints(self) -> tuple[str, ...]:
        """The finite backend basis, distinct from public operation aliases."""

        return tuple(dict.fromkeys(
            entrypoint
            for operation in self.operations.values()
            for entrypoint in operation.entrypoints
        ))

    def dependency_closure(self, *roots: str) -> dict[str, Function]:
        """Return all ordinary SSA functions reachable from ``roots``."""

        reachable: set[str] = set()
        pending = list(map(str, roots))
        while pending:
            name = pending.pop()
            if name in reachable or name not in self.module.functions:
                continue
            reachable.add(name)
            function = self.module.functions[name]
            for block in function.blocks.values():
                for instruction in block.instrs:
                    callee = instruction.attributes.get("callee")
                    if callee in self.module.functions and callee not in reachable:
                        pending.append(str(callee))
        return {
            name: self.module.functions[name]
            for name in self.module.functions
            if name in reachable
        }


__all__ = ["SSATensorCodeReference", "SSATensorOperationReference"]
