"""The multi-participant step, as spans.

``dt_system`` has always taken several coupled simulations and produced one
step.  It did that by folding first -- ``min(dt_limit)`` over everybody, metrics
maxed, channels summed -- and judging afterwards, which loses what a
coordinator needs: a small stiff participant hides inside a large slow one's
totals, and a participant that is only watching its inputs drags the whole
system to its own cadence for no physical reason.

This is the fold in the other order.  What is genuinely system-wide is
amalgamated; what is genuinely individual is gated:

    system state energy and conservation  ->  SUMMED   (extensive; a total)
    stability, and any custom limit       ->  GATED    (individual; a trip)
    individual power and conservation     ->  GATED too, when asked for

A participant trips on its own measure against its own limit, and that trip
stays its own: it does not become everyone's step.  The system total is a sum
because energy and conservation discrepancy are extensive, and a maximum would
report one participant's share as the system's.

FORM
----
Everything on the step path is a span, and every question is one masked
operation over the whole array.  Concretely:

* a participant id indexes the participant axis, a channel id indexes the
  channel axis.  Ids ARE indices, so nothing on this path resolves a name.
* absence is always its own span, never a sentinel value.  "Published no tau"
  and "published tau = 0" are different claims and the step behaves differently
  between them, so a dense span that let absence read as zero would silently
  move dt.
* names appear at the reporting boundary only -- a log line, a refusal, an
  attempt record -- because those are read by a person.

CAUSAL ORDER
------------
The participant axis is in CAUSAL ORDER.  When no symbolic reduction has been
performed the participants genuinely run in sequence, and participant ``i`` may
consume what ``i-1`` published this same step -- the chamber's air publishes the
face flows that its species and aerosol then ride.  So:

* amalgamation and judging vectorise freely: they happen after everyone has
  published, and a sum or a masked compare over the whole array is order
  independent by construction.
* reductions that must be exact use extended precision rather than a
  reordering, so accumulating along the participant axis in causal order costs
  nothing in accuracy.  Two limbs make the sum independent of order anyway;
  the order is kept because it is the physical one, not because the arithmetic
  needs it.
* masking may only remove work in a way that respects that order.

A probabilistic or stoichiometric ordering is a later option, not the default:
the default is the one that matches how the code is written and read.

WHY THIS SURVIVES THE MERGED REDUCTION
--------------------------------------
A participant is a DECLARED IDENTITY, never a callable unit.  Nothing here
knows whether the rows were filled by one kernel call per participant or by a
single fused kernel that had every law symbolically reduced together and writes
all the rows itself.  When the laws merge, what changes is who fills the rows.
That is also why publication is by identity and not by call site: a call site
disappears under reduction, an identity does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .error_channels import declared_channels
from .time_contracts import BIND, HOLD, ParticipantRegistry

#: Limbs used for a reduction that has to be exact.  Two is enough to make a
#: sum independent of the order it is accumulated in, which is what lets the
#: participant axis stay in causal order without the arithmetic caring.
EXACT_LIMBS = 2


@dataclass(frozen=True)
class Publication:
    """What one participant published this step.  A BUILDER, not the path.

    This exists so a hand-written simulation can say what it measured in
    readable terms; :meth:`StepSpans.of` turns a set of these into the spans
    the step actually consumes.  A compiled step skips it entirely and fills
    the spans directly.

    Every field is optional because silence is a real answer: a participant
    that published no ``dt_limit`` has not claimed an unlimited step, and one
    that published no channel has not measured zero.
    """

    channels: Mapping[str, float] = field(default_factory=dict)
    dt_limit: float | None = None
    tau_s: float | None = None
    contract: int = HOLD
    #: limits applied to THIS participant instead of the defaults
    limits: Mapping[str, float] | None = None


@dataclass(frozen=True)
class StepSpans:
    """One step's publications, as aligned spans.

    Per participant, shape ``(P,)``::

        tau_s              its own time constant
        tau_present        whether it published one at all
        contract           HOLD / BIND / DILATE / SUBCYCLE
        dt_limit           its own stability floor
        dt_limit_present   whether it published one at all

    Per participant and channel, shape ``(P, C)``::

        values / present           what it measured, and whether it did
        limits / limits_present    what it is judged against, and whether it is

    ``P`` is in causal order.  ``C`` is the channel registry, whole, so a
    channel id keeps being its index no matter who publishes.
    """

    tau_s: Any
    tau_present: Any
    contract: Any
    dt_limit: Any
    dt_limit_present: Any
    values: Any
    present: Any
    limits: Any
    limits_present: Any

    @classmethod
    def of(
        cls,
        registry: ParticipantRegistry,
        published: Mapping[str, Publication],
        *,
        default_limits: Mapping[str, float] | None = None,
    ) -> "StepSpans":
        """Build the spans from what each declared participant published.

        Participants come out in the registry's declaration order, which is the
        causal order.  A declared participant absent from ``published`` is
        silence: ``HOLD``, no tau, no measures, no limit -- not a row of zeros.
        """
        from ...common.tensors import AbstractTensor

        channels = declared_channels() or ("",)
        default_limits = default_limits or {}

        tau_s: list[float] = []
        tau_present: list[bool] = []
        contract: list[int] = []
        dt_limit: list[float] = []
        dt_limit_present: list[bool] = []
        values: list[list[float]] = []
        present: list[list[bool]] = []
        limits: list[list[float]] = []
        limits_present: list[list[bool]] = []

        for name in registry.declared():
            entry = published.get(name)
            tau = None if entry is None else entry.tau_s
            tau_s.append(0.0 if tau is None else float(tau))
            tau_present.append(tau is not None)
            contract.append(HOLD if entry is None else int(entry.contract))

            floor = None if entry is None else entry.dt_limit
            dt_limit.append(0.0 if floor is None else float(floor))
            dt_limit_present.append(floor is not None)

            measured = {} if entry is None else dict(entry.channels or {})
            declared = dict(default_limits)
            if entry is not None and entry.limits:
                declared.update(entry.limits)

            values.append([float(measured.get(c, 0.0)) for c in channels])
            present.append([c in measured for c in channels])
            limits.append([float(declared.get(c, 0.0)) for c in channels])
            limits_present.append([c in declared for c in channels])

        if not tau_s:   # no participants declared at all
            tau_s, tau_present, contract = [0.0], [False], [HOLD]
            dt_limit, dt_limit_present = [0.0], [False]
            values, present = [[0.0] * len(channels)], [[False] * len(channels)]
            limits, limits_present = [[0.0] * len(channels)], [[False] * len(channels)]

        tensor = AbstractTensor.tensor
        return cls(
            tau_s=tensor(tau_s), tau_present=tensor(tau_present),
            contract=tensor(contract),
            dt_limit=tensor(dt_limit), dt_limit_present=tensor(dt_limit_present),
            values=tensor(values), present=tensor(present),
            limits=tensor(limits), limits_present=tensor(limits_present),
        )

    @property
    def participants(self) -> int:
        return int(self.tau_s.shape[0])


# --------------------------------------------------------------------- summed


def system_totals(spans: StepSpans, *, limbs: int = EXACT_LIMBS):
    """Extensive totals per channel, and whether anybody reported each.

    Accumulated along the participant axis in causal order and in extended
    precision, so the total is exact and the order it was taken in cannot show
    up in the answer.  The precision is dropped on the way out: the leading limb
    is the correctly-rounded sum.

    Returns ``(totals, reported)``, both shape ``(C,)``.  ``reported`` is why
    this is not simply a sum: a channel nobody published is not a total of zero.
    """
    from ...common.tensors import AbstractTensor
    from ...common.tensors.extended_precision import add_expansions

    contributed = spans.values * spans.present
    accumulator = [AbstractTensor.zeros_like(contributed[0])
                   for _ in range(max(1, int(limbs)))]
    for index in range(spans.participants):
        accumulator = add_expansions(accumulator, [contributed[index]], limbs)
    reported = spans.present.sum(dim=0) > 0.0
    return accumulator[0], reported


# ---------------------------------------------------------------------- gated


def judged(spans: StepSpans):
    """Where a measure meets a limit: ``(P, C)``.

    A measure with no limit is not a failure to judge -- nobody asked for it to
    be judged.  A limit with no measure is not a violation either; the step did
    not report on it.  Only the intersection is judged.
    """
    return spans.present * spans.limits_present


def penalties(spans: StepSpans):
    """``measure / limit`` per participant per channel, zero where unjudged.

    One masked division over the whole set: the individual law applied to
    everybody at once, without folding first.
    """
    from ...common.tensors import AbstractTensor

    safe = AbstractTensor.where(spans.limits != 0.0, spans.limits,
                                AbstractTensor.ones_like(spans.limits))
    return AbstractTensor.where(judged(spans), spans.values / safe,
                                AbstractTensor.zeros_like(spans.values))


def tripped(spans: StepSpans):
    """Which individual gates opened: ``(P, C)`` of bool."""
    return penalties(spans) > 1.0


def any_tripped(spans: StepSpans):
    """Whether each participant tripped anything: ``(P,)`` of bool."""
    return tripped(spans).sum(dim=1) > 0.0


def worst_penalty(spans: StepSpans, floor: float = 1.0):
    """The largest judged ratio anywhere, never below ``floor``.

    This is the softening factor the proposal divides by: the worst offender
    across every participant and every channel, in one reduction over the whole
    array instead of a hash per channel per attempt.

    ``floor`` is 1.0 because a step that is inside every limit must not be
    rewarded with a longer one -- the ratio only ever shortens.  Note that an
    unpublished channel cannot affect this even though the dict form it replaces
    read absence as a measure of zero: zero loses to the floor.  That is luck
    rather than design, and it is why absence is masked here instead of being
    given a value.
    """
    from ...common.tensors import AbstractTensor

    ratios = penalties(spans)
    if int(ratios.shape[0]) == 0:
        return AbstractTensor.tensor(float(floor))
    return AbstractTensor.maximum(ratios.max(), AbstractTensor.tensor(float(floor)))


# ------------------------------------------------------------------ the step


def tau_bound(spans: StepSpans, fraction: float, dt_proposed, dt_current=None):
    """The step after every BINDing participant's tau is applied.

    ``dt <= fraction * tau`` per binding participant, as a masked minimum over
    them -- the same law the blended energy/power pin applied, per participant
    instead of over a total, so the stiff one is visible rather than averaged
    away.  A tau only pins when its owner said ``BIND`` and actually published
    one; every other contract is deliberately excluded, and that exclusion is
    what lets a dilating participant keep its own cadence without imposing it.

    A ``HOLD`` anywhere prevents GROWTH beyond the current step without itself
    shrinking anything: nobody claimed a larger step is unsafe, they claimed not
    to know.
    """
    from ...common.tensors import AbstractTensor

    binding = (spans.contract == BIND) * spans.tau_present
    limit = AbstractTensor.tensor(float(dt_proposed))
    if bool(binding.any().item()):
        # non-binding rows are lifted to the proposal so they cannot win the
        # minimum: a masked reduction, not a filtered list
        pinned = AbstractTensor.where(
            binding, spans.tau_s * float(fraction),
            AbstractTensor.ones_like(spans.tau_s) * float(dt_proposed),
        )
        limit = AbstractTensor.minimum(limit, pinned.min())
    if dt_current is not None and bool((spans.contract == HOLD).any().item()):
        limit = AbstractTensor.minimum(limit,
                                       AbstractTensor.tensor(float(dt_current)))
    return limit


def stability_gates(spans: StepSpans):
    """Each participant's own stability floor: ``((P,), (P,))``.

    Deliberately NOT reduced.  A participant's floor is its own gate, and
    imposing the stiffest one on everybody is what the time field replaces by
    giving each scope its own window at a shared substep count.  The caller
    receives the span and the presence mask and decides what to do per
    participant.
    """
    return spans.dt_limit, spans.dt_limit_present


# ------------------------------------------------------------------ reporting


def trip_report(
    spans: StepSpans,
    participant_names: Sequence[str],
    channel_names: Sequence[str] | None = None,
) -> tuple[tuple[str, str, float], ...]:
    """Every gate that opened, as words.  REPORTING ONLY.

    Ids become names here and nowhere else: a refusal, a log line or an attempt
    record is read by a person, so it resolves indices back to words at the
    boundary rather than carrying strings through the step.
    """
    names = list(channel_names) if channel_names is not None else list(declared_channels())
    ratios = penalties(spans).tolist()
    opened: list[tuple[str, str, float]] = []
    for index, row in enumerate(ratios):
        who = (participant_names[index] if index < len(participant_names)
               else str(index))
        for channel, ratio in enumerate(row):
            if float(ratio) > 1.0:
                label = names[channel] if channel < len(names) else str(channel)
                opened.append((who, label, float(ratio)))
    return tuple(opened)


def totals_report(spans: StepSpans) -> dict[str, float]:
    """The system totals, as words.  REPORTING ONLY."""
    totals, reported = system_totals(spans)
    names = declared_channels()
    values = totals.tolist()
    flags = reported.tolist()
    return {names[index]: float(values[index])
            for index in range(min(len(names), len(values)))
            if bool(flags[index])}
