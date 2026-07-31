"""Backend-neutral semantic loop representation.

Loops are programs, not syntax strings.  Frontends describe their domain,
termination, state, effects, and scheduling constraints here.  A planner then
selects a realization; Python, C, GLSL, tape, and future backends merely lower
that selected realization.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .hierarchical_plan import PlanClosure


class LoopDomainKind(str, Enum):
    RANGE = "range"
    ITERABLE = "iterable"
    CONDITION = "condition"
    STATE_MACHINE = "state_machine"


class LoopRealization(str, Enum):
    AUTO = "auto"
    UNROLL = "unroll"
    NATIVE = "native"
    STATE_MACHINE = "state_machine"
    KPN = "kpn"
    DISPATCH = "dispatch"


class IterableAccess(str, Enum):
    STATIC = "static"
    RESIDENT = "resident"
    CLOSURE_AGGREGATE = "closure_aggregate"
    GENERATOR = "generator"


class LoopStateEffectMode(str, Enum):
    """Backend-neutral realization of a source mutation."""

    OPAQUE = "opaque"
    INDEXED_PUBLICATION = "indexed_publication"


@dataclass(frozen=True)
class LoopValue:
    """A reference to a graph value, optionally with a known literal."""

    value_id: int | None = None
    literal: object | None = None

    @property
    def resolved(self) -> bool:
        return self.literal is not None


@dataclass(frozen=True)
class RangeDomain:
    start: LoopValue
    stop: LoopValue
    step: LoopValue


@dataclass(frozen=True)
class IterableDomain:
    iterable: LoopValue
    targets: tuple[tuple[str, int], ...]
    access: IterableAccess = IterableAccess.RESIDENT
    source_value_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ConditionDomain:
    condition_value_id: int


@dataclass(frozen=True)
class StateMachineDomain:
    state_value_id: int
    terminal_states: tuple[int, ...] = ()


LoopDomain = RangeDomain | IterableDomain | ConditionDomain | StateMachineDomain


@dataclass(frozen=True)
class LoopCarriedState:
    name: str
    initial_value_id: int
    next_value_id: int


@dataclass(frozen=True)
class LoopStateEffect:
    """One body effect and the loop-state transition it causes."""

    state_name: str
    operator: str
    state_input_id: int
    effect_node_id: int
    state_output_id: int | None = None
    loop_result_id: int | None = None
    argument_value_ids: tuple[int, ...] = ()
    mode: LoopStateEffectMode = LoopStateEffectMode.OPAQUE


@dataclass(frozen=True)
class LoopIterationOutput:
    """A source-defined per-iteration value and its materialized loop result."""

    value_id: int
    result_value_id: int
    materializer_node_id: int


@dataclass(frozen=True)
class LoopEffects:
    break_value_ids: tuple[int, ...] = ()
    continue_value_ids: tuple[int, ...] = ()
    yield_value_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class LoopPolicy:
    """Planner inputs; these are constraints/preferences, not source syntax."""

    realization: LoopRealization = LoopRealization.AUTO
    unroll_limit: int | None = None
    max_iterations: int | None = None
    greedy_input: bool = True
    backpressure: bool = False
    batch_to_downstream_capacity: bool = True
    allow_parallel_iterations: bool = False
    require_resident_state: bool = True


@dataclass(frozen=True)
class SemanticLoop:
    loop_id: int
    domain_kind: LoopDomainKind
    domain: LoopDomain
    body_node_ids: tuple[int, ...]
    body_closure: PlanClosure | None = None
    carried: tuple[LoopCarriedState, ...] = ()
    state_effects: tuple[LoopStateEffect, ...] = ()
    iteration_outputs: tuple[LoopIterationOutput, ...] = ()
    effects: LoopEffects = LoopEffects()
    policy: LoopPolicy = LoopPolicy()

    def __post_init__(self) -> None:
        expected = {
            LoopDomainKind.RANGE: RangeDomain,
            LoopDomainKind.ITERABLE: IterableDomain,
            LoopDomainKind.CONDITION: ConditionDomain,
            LoopDomainKind.STATE_MACHINE: StateMachineDomain,
        }[self.domain_kind]
        if not isinstance(self.domain, expected):
            raise TypeError(
                f"{self.domain_kind.value} loop requires {expected.__name__}"
            )
        if isinstance(self.domain, RangeDomain):
            if self.domain.step.resolved and int(self.domain.step.literal) == 0:
                raise ValueError("range loop step cannot be zero")


__all__ = [
    "ConditionDomain",
    "IterableDomain",
    "IterableAccess",
    "LoopCarriedState",
    "LoopDomain",
    "LoopDomainKind",
    "LoopEffects",
    "LoopPolicy",
    "LoopRealization",
    "LoopIterationOutput",
    "LoopStateEffect",
    "LoopStateEffectMode",
    "LoopValue",
    "RangeDomain",
    "SemanticLoop",
    "StateMachineDomain",
]
