"""Class-bound state-machine contract for managed AbstractTensor engines.

This module deliberately defines no scheduler and no time source.  A marked
class is advanced only when ``dt_system`` calls its ``step`` boundary with an
admitted timestep.  The marker gives Python AST ingestion a semantic identity
that ordinary classes do not have; downstream compilation still uses the
existing ControlProgram/StateMachineTick and numeric operator vocabulary.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any, ClassVar, Mapping

from .dt_system.dt_scaler import Metrics
from .dt_system.engine_api import DtCompatibleEngine


StateDimension = int | str


@dataclass(frozen=True, slots=True)
class TensorStateField:
    """One class-owned tensor field in a state-machine memory contract."""

    name: str
    shape: tuple[StateDimension, ...]
    dtype: str
    scope: str = "engine"
    mutable: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError("tensor state field name must be an identifier")
        if not self.dtype:
            raise ValueError("tensor state field dtype is required")
        if not self.scope:
            raise ValueError("tensor state field scope is required")
        for dimension in self.shape:
            if isinstance(dimension, int):
                if dimension < 0:
                    raise ValueError("tensor state dimensions cannot be negative")
            elif not dimension or not dimension.isidentifier():
                raise ValueError(
                    "symbolic tensor state dimensions must be identifiers"
                )


class AbstractTensorStateMachine(DtCompatibleEngine, ABC):
    """AST-visible marker and managed engine contract.

    Subclasses declare their class-bound tensor storage in ``state_fields``
    and implement one transaction-safe ``transition``.  ``step`` validates
    the externally admitted dt and delegates without subdividing, sleeping,
    reading a clock, or otherwise changing its meaning.

    Mutable state outside the supplied state object must be covered by the
    required ``snapshot``/``restore`` pair so rejected dt-system attempts are
    indistinguishable from attempts that never ran.
    """

    __abstract_tensor_state_machine__: ClassVar[bool] = True
    state_fields: ClassVar[tuple[TensorStateField, ...]] = ()

    @classmethod
    def tensor_state_schema(cls) -> tuple[TensorStateField, ...]:
        """Return the validated, class-owned tensor memory schema."""

        fields = tuple(cls.state_fields)
        names = tuple(field.name for field in fields)
        if len(set(names)) != len(names):
            raise ValueError(
                f"{cls.__name__} tensor state field names must be unique"
            )
        return fields

    def step(
        self,
        dt: float,
        state: Any = None,
        state_table: Any = None,
    ) -> tuple[bool, Metrics, Any]:
        """Run exactly one dt-system-admitted transition."""

        admitted_dt = float(dt)
        if not math.isfinite(admitted_dt) or admitted_dt <= 0.0:
            raise ValueError("state-machine dt must be finite and positive")
        if state_table is None:
            raise ValueError("state-machine transition requires StateTable")
        result = self.transition(state, admitted_dt, state_table=state_table)
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError(
                "state-machine transition must return (ok, Metrics, state)"
            )
        ok, metrics, next_state = result
        if not isinstance(metrics, Metrics):
            raise TypeError("state-machine transition must return Metrics")
        return bool(ok), metrics, next_state

    @abstractmethod
    def transition(
        self,
        state: Any,
        dt: float,
        *,
        state_table: Any,
    ) -> tuple[bool, Metrics, Any]:
        """Apply one admitted transition without choosing another dt."""

    @abstractmethod
    def get_state(self, state: Any = None) -> Any:
        """Return or update the state object owned by this machine."""

    @abstractmethod
    def snapshot(self) -> Any:
        """Capture all engine-owned mutable state for scientific rollback."""

    @abstractmethod
    def restore(self, snapshot: Any) -> None:
        """Restore a snapshot in place, preserving externally held identity."""


def is_abstract_tensor_state_machine(value: type[Any]) -> bool:
    """Return whether ``value`` implements the canonical runtime marker."""

    return bool(
        isinstance(value, type)
        and issubclass(value, AbstractTensorStateMachine)
        and getattr(value, "__abstract_tensor_state_machine__", False)
    )


__all__ = [
    "AbstractTensorStateMachine",
    "TensorStateField",
    "is_abstract_tensor_state_machine",
]

