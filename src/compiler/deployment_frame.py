"""Backend-neutral deploy/join framing shared by ProcessGraph, Control and SSA.

Deploy and join are structural operators, not numerical tensor operations.
The framed body continues to use the established operator vocabulary; the
join names an existing reducer when concurrent lane results must be combined.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DeploymentJoinMode(str, Enum):
    BARRIER = "barrier"
    INDEXED = "indexed"
    PRODUCT = "product"
    REDUCE = "reduce"


_REDUCTION_ALIASES = {
    "add": "Add",
    "sum": "Add",
    "mul": "Mul",
    "prod": "Mul",
    "min": "Min",
    "max": "Max",
    "and": "And",
    "all": "And",
    "or": "Or",
    "any": "Or",
}


def canonical_reduction_operator(operator: str) -> str:
    """Return the existing operator identity used by a reduction join."""

    spelling = str(operator).strip()
    canonical = _REDUCTION_ALIASES.get(spelling.lower(), spelling)
    if canonical not in frozenset(_REDUCTION_ALIASES.values()):
        raise ValueError(
            "join reducer must use an existing associative operator, not "
            f"{operator!r}"
        )
    return canonical


@dataclass(frozen=True)
class DeploymentJoin:
    """How completion and values leave one deployment frame."""

    mode: DeploymentJoinMode = DeploymentJoinMode.BARRIER
    reduction_operator: str | None = None
    # Floating add/multiply cannot change association unless the author or a
    # proof pass grants permission. Integer and logical reducers do not need
    # this flag merely to use multiple physical lanes.
    allow_reassociation: bool = False

    def __post_init__(self) -> None:
        mode = DeploymentJoinMode(self.mode)
        object.__setattr__(self, "mode", mode)
        if mode is DeploymentJoinMode.REDUCE:
            if self.reduction_operator is None:
                raise ValueError("a reduction join requires an operator")
            object.__setattr__(
                self,
                "reduction_operator",
                canonical_reduction_operator(self.reduction_operator),
            )
        elif self.reduction_operator is not None:
            raise ValueError(
                "only a reduction join may name a reduction operator"
            )


@dataclass(frozen=True)
class DeploymentFrame:
    """Validated structural contract between a deploy and its matching join."""

    region_id: int
    scale: int = 1
    join: DeploymentJoin = DeploymentJoin()

    def __post_init__(self) -> None:
        if int(self.region_id) < 0:
            raise ValueError("deployment region ID must be non-negative")
        if int(self.scale) < 1:
            raise ValueError("deployment scale must be positive")
        object.__setattr__(self, "region_id", int(self.region_id))
        object.__setattr__(self, "scale", int(self.scale))


__all__ = [
    "DeploymentFrame",
    "DeploymentJoin",
    "DeploymentJoinMode",
    "canonical_reduction_operator",
]
