"""Per-participant, per-step time contracts: exchange_time, and how it negotiates.

The dt system's job across several coupled simulations is to pick the next
step.  Today it takes the most constrained participant and imposes that on
everyone -- ``min(dt_limit)``, and a fold that SUMS energy and power across
pieces before dividing.  Both throw away the thing a coordinator needs: a
small stiff simulation is hidden inside a large slow one's totals, and an
inconsequential participant that is only watching its inputs drags the
entire system to its own cadence for no physical reason.

A participant publishes, per step, its own exchange time ``exchange_time_s``
-- stored energy over exchange power, E/P, the energy-stability metric that
schedules dt -- and a CONTRACT saying how that exchange time participates.
(Vocabulary, decided 2026-09-22: this quantity was formerly called ``tau``.
``tau`` now means TIME VELOCITY, the world time a participant actually
advanced over the world time asked of it; it is measured, never scheduled
on.  Wall cost is a separate, information-only ledger.)

``BIND``      exchange_time bounds the step; resolving my dynamics requires it.
``DILATE``    do not pin anyone.  I am in an energy well, characterized
              well enough to be evaluated at any elapsed interval in
              closed form: ``x(t) = x_eq + (x0 - x_eq) * exp(-dt/exchange_time)``.
              exchange_time here is the well's relaxation time -- the SAME quantity
              as in BIND, used to evaluate rather than to bound.  I am not
              running behind and owe no catch-up; there is nothing to
              replay.
``SUBCYCLE``  do not pin anyone.  I am INDEPENDENT: I run on my own clock
              (my own dispatched task and dt system), and whoever needs me
              consults my latest publication without waiting for me --
              stamped with the world time it is true at, read lagged
              otherwise.  My tau (time velocity) and my slip behind the
              consulting system are measured, never set.  See
              ``examples/llvm_dt_system.Subcycle``.
``HOLD``      I observed nothing that says a larger step is safe.  Do not
              grow.  (This is what ``power_w == 0`` used to mean by
              arithmetic accident, and it is now said outright.)

Absence is carried as a presence mask, never as a zero.  "Published no
exchange_time" and "published exchange_time = 0" are different claims and the controller's
behaviour genuinely differs between them; a dense span that let absence
read as zero would silently move dt.

``DILATE`` is only sound while the participant's motion stays in a
coordinate orthogonal to the coupled dynamics.  That is not taken on
trust: the dt system owns energy accounting, so a participant exchanging
energy it did not declare shows up as a conservation discrepancy.  The
claim is auditable where the evidence already lives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

HOLD = 0
BIND = 1
DILATE = 2
SUBCYCLE = 3

CONTRACT_NAMES = {
    HOLD: "hold",
    BIND: "bind",
    DILATE: "dilate",
    SUBCYCLE: "subcycle",
}


class ParticipantRegistry:
    """Monotonic ids for participants, assigned once at declaration.

    BUILD TIME ONLY, exactly like the channel registry: this is the one
    place a participant is looked up by name, so that nothing on the step
    path ever has to.  The id IS the index into every span below.
    """

    def __init__(self) -> None:
        self._names: list[str] = []
        self._by_name: dict[str, int] = {}

    def declare(self, name: str) -> int:
        key = str(name)
        existing = self._by_name.get(key)
        if existing is not None:
            return existing
        identity = len(self._names)
        self._names.append(key)
        self._by_name[key] = identity
        return identity

    def name_of(self, identity: int) -> str:
        """Reporting only -- never on the step path."""
        return self._names[int(identity)]

    def declared(self) -> tuple[str, ...]:
        return tuple(self._names)

    def __len__(self) -> int:
        return len(self._names)


@dataclass(frozen=True)
class StepContracts:
    """What every participant published this step, as aligned spans.

    One row per participant id; ``exchange_time_present`` is what keeps "did not
    publish" distinct from "published zero".
    """

    exchange_time_s: Any
    exchange_time_present: Any
    contract: Any

    @classmethod
    def of(
        cls,
        registry: ParticipantRegistry,
        published: Mapping[str, tuple[float | None, int]],
    ) -> "StepContracts":
        """Build the spans from what each named participant published.

        A participant that published nothing this step is absent from
        ``published`` and lands as ``HOLD`` with no exchange_time -- the honest
        reading of silence, and not the same as claiming exchange_time = 0.
        """
        from ...common.tensors import AbstractTensor

        exchange_times: list[float] = []
        present: list[bool] = []
        contracts: list[int] = []
        for name in registry.declared():
            entry = published.get(name)
            if entry is None:
                exchange_times.append(0.0)
                present.append(False)
                contracts.append(HOLD)
                continue
            exchange_time, contract = entry
            exchange_times.append(0.0 if exchange_time is None else float(exchange_time))
            present.append(exchange_time is not None)
            contracts.append(int(contract))
        return cls(
            exchange_time_s=AbstractTensor.tensor(exchange_times),
            exchange_time_present=AbstractTensor.tensor(present),
            contract=AbstractTensor.tensor(contracts),
        )


def binding_mask(contracts: StepContracts):
    """Participants whose exchange_time actually bounds this step.

    An exchange time only pins when its owner said ``BIND`` AND actually published
    one.  Every other contract is deliberately excluded here -- that
    exclusion is the whole point: it is what lets a dilating participant
    keep its own accurate cadence without imposing it.
    """
    return (contracts.contract == BIND) * contracts.exchange_time_present


def holds_dt(contracts: StepContracts) -> bool:
    """Whether any participant said it has no evidence for a larger step."""
    return bool((contracts.contract == HOLD).any().item())


def bound_dt(
    contracts: StepContracts,
    fraction: float,
    dt_proposed: float,
    dt_current: float | None = None,
) -> float:
    """The step after every binding participant's exchange_time is applied.

    ``dt <= fraction * exchange_time`` for each binding participant, taken as a
    minimum over them -- the same law the energy/power pin already applied,
    but per participant instead of over a blended total, so the stiff one
    is visible rather than averaged away.

    A ``HOLD`` anywhere prevents GROWTH beyond the current step without
    itself shrinking anything: no participant claimed a larger step is
    unsafe, they claimed not to know.
    """
    binding = binding_mask(contracts)
    limit = float(dt_proposed)
    if bool(binding.any().item()):
        from ...common.tensors import AbstractTensor

        # Non-binding rows must not win the minimum, so they are lifted to
        # the proposed step rather than left at zero -- a masked minimum,
        # not a filtered list, so this stays one vectorized reduction.
        pinned = AbstractTensor.where(
            binding,
            contracts.exchange_time_s * float(fraction),
            AbstractTensor.tensor([limit] * len(binding.tolist())),
        )
        limit = min(limit, float(pinned.min().item()))
    if dt_current is not None and holds_dt(contracts):
        limit = min(limit, float(dt_current))
    return limit


def dilated_state(
    x0: float, x_eq: float, exchange_time_s: float, elapsed_s: float,
) -> float:
    """Where a dilating participant is after ``elapsed_s``, in closed form.

    ``x(t) = x_eq + (x0 - x_eq) * exp(-t / exchange_time)`` -- relaxation toward the
    well minimum.  This is why ``DILATE`` owes no catch-up: its state at
    any elapsed interval is evaluated, not replayed, so no debt accrues no
    matter how long the coordinator leaves it alone.
    """
    import math

    if exchange_time_s <= 0.0:
        raise ValueError("a dilating participant needs a positive exchange_time")
    return x_eq + (x0 - x_eq) * math.exp(-float(elapsed_s) / float(exchange_time_s))


def crossing_time(
    x0: float, x_eq: float, exchange_time_s: float, barrier: float,
) -> float | None:
    """When this participant's relaxation reaches ``barrier``, or None.

    Inverting the closed form: ``t = -exchange_time * ln((barrier - x_eq) /
    (x0 - x_eq))``.  Returns None when the barrier is unreachable -- it
    lies beyond the well minimum the participant is relaxing toward, so
    no elapsed interval ever arrives there and the coordinator is free to
    step over the whole span.

    This is what lets a dilating reactor be left alone safely: the
    coordinator can ask when the condition would be met instead of
    stepping small enough not to miss it.
    """
    import math

    if exchange_time_s <= 0.0:
        raise ValueError("a dilating participant needs a positive exchange_time")
    span = x0 - x_eq
    if span == 0.0:
        return 0.0 if barrier == x0 else None
    ratio = (float(barrier) - x_eq) / span
    if ratio <= 0.0 or ratio > 1.0:
        # Beyond the minimum, or already behind us: never reached by
        # relaxation from here.
        return None
    return -float(exchange_time_s) * math.log(ratio)
