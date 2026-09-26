"""Shared semantic identities across Turing's program representations.

This module does not lower programs.  It gives existing lowering edges one
authoritative vocabulary and an explicit exactness contract.  A common family
identity (for example ``arithmetic.add``) never discards representation-
specific facts: width/flags/memory on a machine operation and shape/layout on
a tensor operation remain facets.  Crossing an edge without preserving all
required facets produces a typed residual instead of a numerical projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class SemanticRepresentation(str, Enum):
    DUAL_IR = "dual-ir"
    PROCESS_GRAPH = "process-graph"
    MACHINE_GRAPH = "machine-program-graph"
    MACHINE_SSA = "machine-state-ssa"
    REPOSITORY_SSA = "repository-ssa"


@dataclass(frozen=True, slots=True)
class SemanticOperationIdentity:
    family: str
    representation: SemanticRepresentation
    spelling: str
    facets: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", str(self.family))
        object.__setattr__(self, "representation", SemanticRepresentation(
            self.representation,
        ))
        object.__setattr__(self, "spelling", str(self.spelling))
        object.__setattr__(self, "facets", MappingProxyType(dict(self.facets)))

    def attributes(self) -> dict[str, Any]:
        return {
            "semantic_family": self.family,
            "semantic_representation": self.representation.value,
            "semantic_spelling": self.spelling,
            "semantic_facets": dict(self.facets),
        }


@dataclass(frozen=True, slots=True)
class SemanticTranslationResidual:
    family: str
    source: SemanticRepresentation
    target: SemanticRepresentation
    missing_facets: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class SemanticTranslationProof:
    source: SemanticOperationIdentity
    target: SemanticOperationIdentity
    preserved_facets: tuple[str, ...]


_ALIASES = {
    # Scalar/tensor and repository spellings.
    "add": "arithmetic.add", "+": "arithmetic.add",
    "sub": "arithmetic.subtract", "subtract": "arithmetic.subtract",
    "mul": "arithmetic.multiply", "multiply": "arithmetic.multiply",
    "div": "arithmetic.divide", "truediv": "arithmetic.divide",
    "floordiv": "arithmetic.floor-divide", "mod": "arithmetic.modulo",
    "pow": "arithmetic.power", "neg": "arithmetic.negate",
    "abs": "arithmetic.absolute",
    "and": "bitwise.and", "bitwise_and": "bitwise.and",
    "or": "bitwise.or", "bitwise_or": "bitwise.or",
    "xor": "bitwise.xor", "bitwise_xor": "bitwise.xor",
    "not": "bitwise.not", "invert": "bitwise.not",
    "shl": "bitwise.shift-left", "lshift": "bitwise.shift-left",
    "shr": "bitwise.shift-right", "rshift": "bitwise.shift-right",
    "eq": "comparison.equal", "equality": "comparison.equal",
    "ne": "comparison.not-equal", "unequality": "comparison.not-equal",
    "lt": "comparison.less", "le": "comparison.less-equal",
    "gt": "comparison.greater", "ge": "comparison.greater-equal",
    "load": "memory.load", "store": "memory.store",
    "alloca": "memory.allocate", "getelementptr": "memory.address",
    "const": "value.constant", "input": "value.input",
    "call": "control.call", "return": "control.return", "ret": "control.return",
    "br": "control.branch", "condbr": "control.conditional-branch",
    "indirectbr": "control.indirect-branch", "phi": "control.phi",
    "where": "selection.where", "select": "selection.where",
    "reshape": "tensor.reshape", "broadcast_to": "tensor.broadcast",
    "sum": "tensor.reduce.sum", "mean": "tensor.reduce.mean",
    "prod": "tensor.reduce.product", "matmul": "tensor.matmul",
    "exp": "transcendental.exp", "log": "transcendental.log",
    "sin": "transcendental.sin", "cos": "transcendental.cos",
    "sqrt": "transcendental.sqrt",
}

_MACHINE_FAMILIES = {
    "INTEGER_ADD": "arithmetic.add",
    "INTEGER_SUBTRACT": "arithmetic.subtract",
    "INTEGER_SUBTRACT_WITH_BORROW": "arithmetic.subtract-with-borrow",
    "INTEGER_MULTIPLY": "arithmetic.multiply",
    "INTEGER_MULTIPLY_UNSIGNED": "arithmetic.multiply",
    "INTEGER_DIVIDE": "arithmetic.divide",
    "INTEGER_DIVIDE_SIGNED": "arithmetic.divide",
    "INTEGER_NEGATE": "arithmetic.negate",
    "INTEGER_INCREMENT": "arithmetic.increment",
    "INTEGER_DECREMENT": "arithmetic.decrement",
    "INTEGER_COMPARE": "comparison.compare",
    "INTEGER_TEST": "comparison.test",
    "BITWISE_AND": "bitwise.and",
    "BITWISE_OR": "bitwise.or",
    "BITWISE_XOR": "bitwise.xor",
    "BITWISE_NOT": "bitwise.not",
    "SHIFT_LEFT": "bitwise.shift-left",
    "SHIFT_RIGHT_LOGICAL": "bitwise.shift-right-logical",
    "SHIFT_RIGHT_ARITHMETIC": "bitwise.shift-right-arithmetic",
    "REGISTER_OR_MEMORY_READ": "memory.load",
    "REGISTER_OR_MEMORY_WRITE": "memory.store",
    "REGISTER_WRITE_IMMEDIATE": "memory.store",
    "DIRECT_RELATIVE_CALL": "control.call",
    "INDIRECT_CALL": "control.call",
    "DIRECT_RELATIVE_JUMP": "control.branch",
    "CONDITIONAL_RELATIVE_JUMP": "control.conditional-branch",
    "INDIRECT_JUMP": "control.indirect-branch",
    "RETURN": "control.return",
    "NO_OPERATION": "control.no-operation",
}


def semantic_family(
    spelling: str,
    representation: SemanticRepresentation | str,
) -> str:
    """Return a stable family without claiming that its facets are equal."""

    active = SemanticRepresentation(representation)
    raw = str(spelling)
    if active in {
        SemanticRepresentation.MACHINE_GRAPH,
        SemanticRepresentation.MACHINE_SSA,
    }:
        return _MACHINE_FAMILIES.get(raw.upper(), f"machine.{raw.lower()}")
    return _ALIASES.get(raw.casefold(), f"operation.{raw.casefold()}")


def semantic_identity(
    spelling: str,
    representation: SemanticRepresentation | str,
    *,
    facets: Mapping[str, Any] | None = None,
) -> SemanticOperationIdentity:
    return SemanticOperationIdentity(
        semantic_family(spelling, representation),
        SemanticRepresentation(representation),
        spelling,
        facets or {},
    )


def prove_exact_translation(
    source: SemanticOperationIdentity,
    target: SemanticOperationIdentity,
    *,
    required_facets: tuple[str, ...] | None = None,
) -> SemanticTranslationProof | SemanticTranslationResidual:
    """Prove one semantic edge exact, or retain its missing information."""

    if source.family != target.family:
        return SemanticTranslationResidual(
            source.family, source.representation, target.representation, (),
            f"semantic family changed to {target.family!r}",
        )
    required = tuple(required_facets or source.facets.keys())
    missing = tuple(
        facet for facet in required
        if facet not in target.facets or target.facets[facet] != source.facets[facet]
    )
    if missing:
        return SemanticTranslationResidual(
            source.family, source.representation, target.representation,
            missing, "required semantic facets were not preserved",
        )
    return SemanticTranslationProof(source, target, required)


__all__ = [
    "SemanticOperationIdentity", "SemanticRepresentation",
    "SemanticTranslationProof", "SemanticTranslationResidual",
    "prove_exact_translation", "semantic_family", "semantic_identity",
]
