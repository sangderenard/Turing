"""Error channels as id-indexed spans, not a string-keyed dict.

``Metrics.error_channels`` was a ``dict[str, float]``. A dict does not
lower well: it becomes a keyed store addressed by string tokens, and every
consultation on the step path is a hash.  Worse, those tokens are fnv1a-**64**
values, which do not survive a float64 column (53-bit mantissa) -- the exact
defect the compiler's ``key-column-not-integral`` finding now reports.

Dense monotonic ids avoid all of it.  An id is assigned once, at declaration,
stays far below 2**53 so it is exact in any numeric column, and IS the index
into the spans -- so the loop that judges a step indexes and never hashes.

Three things are deliberately separated here, because the old dict carried
all three under one name and their needs conflict:

* An **error measure** is compared against a limit and produces a penalty.
  That is what this module carries, and it is the only genuinely dynamic,
  plug-and-play part: a new simulator declares a channel, publishes it, and
  consumers that never heard of it are unaffected.
* A **physical observable** feeding a named law (``exchange_time_s``, ``shadow_growth``)
  is fixed at build time and belongs in ``time_contracts``; it was never an
  error and nothing compares it to a limit.
* A **controller diagnostic** (``dt_unresolved``, ``superstep_*``) is written
  by the controller for reporting and never judged at all.

Absence is a presence mask, never a zero: a channel nobody published and a
channel published as 0.0 are different claims, and the controller's bands
behave differently between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

_CHANNEL_NAMES: list[str] = []
_CHANNEL_BY_NAME: dict[str, int] = {}

# The shared dt program's ABI, explicitly ordered and independent of declaration
# history. Programs extending this layout append their own columns and declare
# the resulting extent in their extraction contract. Never use the process-wide
# registry to reconstruct a compiled artifact's column order.
DT_CHANNEL_NAMES = (
    "energy_j", "power_w", "shadow_growth", "div_inf", "mass_err",
    "height_positivity", "tracer_bounds", "maximum_substep_displacement_m",
    "causal_dt_excess", "time_slip", "spring_causal_dt_excess",
    "world_sparse_shape", "columnar_material_unit_error", "columnar_nonfinite",
    "damping_factor",
)
ENERGY_J, POWER_W, SHADOW_GROWTH = 0, 1, 2


def empty_channels():
    from ..tensors import AbstractTensor

    return AbstractTensor.zeros((len(DT_CHANNEL_NAMES),))


def channel_fields(published, *, names=DT_CHANNEL_NAMES, limits=False):
    """Build-time adapter for named configuration; never called by a step.

    Return constructor fields with explicit values and presence. Unknown names
    are errors, not silently dropped columns. The caller owns the layout.
    """
    if tuple(names[:len(DT_CHANNEL_NAMES)]) != DT_CHANNEL_NAMES:
        raise ValueError("dt channel layouts must retain the shared ABI prefix")
    unknown = set(published).difference(names)
    if unknown:
        raise ValueError(f"undeclared channels: {sorted(unknown)}")
    spans = ChannelSpans.of(published, names=names)
    if limits:
        return {"error_limits": spans.values, "error_limits_present": spans.present}
    return {"error_channels": spans.values, "error_present": spans.present}


def channel_report(values, present, names=DT_CHANNEL_NAMES):
    """Resolve column names only at the reporting boundary."""
    return {name: float(value) for name, value, flag in
            zip(names, values.tolist(), present.tolist()) if flag}


def declare_channel(name: str) -> int:
    """The monotonic id for ``name``, assigning one on first sight.

    BUILD TIME ONLY.  This is the one function that looks a channel up by
    string; it exists so that nothing on the step path ever has to.
    """
    key = str(name)
    existing = _CHANNEL_BY_NAME.get(key)
    if existing is not None:
        return existing
    identity = len(_CHANNEL_NAMES)
    _CHANNEL_NAMES.append(key)
    _CHANNEL_BY_NAME[key] = identity
    return identity


def channel_name(channel_id: int) -> str:
    """The declared name of a channel.  Reporting only -- never on the path."""
    return _CHANNEL_NAMES[int(channel_id)]


def declared_channels() -> tuple[str, ...]:
    """Every channel declared so far, in id order."""
    return tuple(_CHANNEL_NAMES)


def packed_channel_names(ids) -> tuple[bytes, tuple[int, ...]]:
    """``(blob, offsets)`` for these channel ids: the names as one minimal
    byte array plus one start offset per id, the last offset being the end.

    This is the whole textual surface of the channel system, and it is what
    a log line or a refusal reads.  It is built once and carried alongside;
    a step never touches it.
    """
    chunks: list[bytes] = []
    offsets: list[int] = [0]
    for channel_id in ids:
        chunks.append(_CHANNEL_NAMES[int(channel_id)].encode("utf-8"))
        offsets.append(offsets[-1] + len(chunks[-1]))
    return b"".join(chunks), tuple(offsets)


def unpack_channel_name(blob: bytes, offsets, index: int) -> str:
    """One name back out of ``packed_channel_names``.  Reporting only."""
    start = int(offsets[int(index)])
    stop = int(offsets[int(index) + 1])
    return blob[start:stop].decode("utf-8")


@dataclass(frozen=True)
class ChannelSpans:
    """What one step published, as aligned spans over the channel registry.

    ``values[i]`` is channel ``i``'s measure and ``present[i]`` says whether
    it was published at all.  The spans cover the whole registry rather than
    only what this participant published, so the id keeps being the index no
    matter who publishes -- which is what lets the controller inject its own
    channels without renumbering anyone.
    """

    values: Any
    present: Any

    @classmethod
    def of(cls, published: Mapping[str, float] | None = None, *, names=None) -> "ChannelSpans":
        from ...common.tensors import AbstractTensor

        published = published or {}
        values: list[float] = []
        present: list[bool] = []
        for name in (_CHANNEL_NAMES if names is None else names):
            measure = published.get(name)
            values.append(0.0 if measure is None else float(measure))
            present.append(measure is not None)
        return cls(
            values=AbstractTensor.tensor(values or [0.0]),
            present=AbstractTensor.tensor(present or [False]),
        )

    def has(self, channel_id: int) -> bool:
        """Whether this channel was published -- never "is it nonzero"."""
        return bool(self.present[int(channel_id)].item())

    def value(self, channel_id: int) -> float | None:
        """This channel's measure, or None when it was not published."""
        index = int(channel_id)
        if not self.has(index):
            return None
        return float(self.values[index].item())


@dataclass(frozen=True)
class ChannelLimits:
    """The limits a channel is judged against, indexed the same way.

    This is ``Targets.error_limits`` as spans.  It must be indexed by the
    same registry as the measures, or the pairing that made the dict work --
    measure and limit share a name -- is lost.
    """

    limits: Any
    present: Any

    @classmethod
    def of(cls, declared: Mapping[str, float] | None = None) -> "ChannelLimits":
        from ...common.tensors import AbstractTensor

        declared = declared or {}
        limits: list[float] = []
        present: list[bool] = []
        for name in _CHANNEL_NAMES:
            limit = declared.get(name)
            limits.append(0.0 if limit is None else float(limit))
            present.append(limit is not None)
        return cls(
            limits=AbstractTensor.tensor(limits or [0.0]),
            present=AbstractTensor.tensor(present or [False]),
        )


@dataclass(frozen=True)
class ParticipantChannels:
    """What EVERY participant published this step: participants x channels.

    ``ChannelSpans`` carries one participant's worth.  This carries the whole
    set, because the amalgamation and the gating want different things from it
    and both want them at once:

    * **summed** across participants for a system extensive -- total stored
      energy, total power, total conservation discrepancy.  The system view is
      a sum because those quantities are extensive; a max would report one
      participant's share as the system's.
    * **per participant** for a trip.  A stability limit or a custom limit is
      an individual gate: a participant trips on its own measure against its
      own limit, and it does not become everyone's step.  That is the whole
      difference from folding first and judging afterwards, where seven quiet
      participants and one loud one are indistinguishable.

    The participant axis is inside the arrays rather than outside them, so the
    judging stays one masked reduction over everything and the channel id keeps
    being the index along the second axis.  Nothing hashes, and the shape is the
    same shape the state columns already have.
    """

    values: Any    # (participants, channels)
    present: Any   # (participants, channels)

    @classmethod
    def of(cls, published: Iterable[Mapping[str, float] | None]) -> "ParticipantChannels":
        from ...common.tensors import AbstractTensor

        rows_values: list[list[float]] = []
        rows_present: list[list[bool]] = []
        for entry in published:
            entry = entry or {}
            values: list[float] = []
            present: list[bool] = []
            for name in _CHANNEL_NAMES:
                measure = entry.get(name)
                values.append(0.0 if measure is None else float(measure))
                present.append(measure is not None)
            rows_values.append(values or [0.0])
            rows_present.append(present or [False])
        if not rows_values:
            rows_values, rows_present = [[0.0]], [[False]]
        return cls(values=AbstractTensor.tensor(rows_values),
                   present=AbstractTensor.tensor(rows_present))


@dataclass(frozen=True)
class ParticipantLimits:
    """Each participant's own limits, indexed the same way.

    A participant may be judged against the defaults or against its own: a
    custom limit is how one participant is held to a standard the others are
    not, which is what makes the gate individual rather than a property of the
    step.  ``None`` for a participant means "the defaults apply to it".
    """

    limits: Any    # (participants, channels)
    present: Any   # (participants, channels)

    @classmethod
    def of(
        cls,
        per_participant: Iterable[Mapping[str, float] | None],
        default: Mapping[str, float] | None = None,
    ) -> "ParticipantLimits":
        from ...common.tensors import AbstractTensor

        default = default or {}
        rows_limits: list[list[float]] = []
        rows_present: list[list[bool]] = []
        for entry in per_participant:
            declared = dict(default)
            if entry:
                declared.update(entry)
            limits: list[float] = []
            present: list[bool] = []
            for name in _CHANNEL_NAMES:
                limit = declared.get(name)
                limits.append(0.0 if limit is None else float(limit))
                present.append(limit is not None)
            rows_limits.append(limits or [0.0])
            rows_present.append(present or [False])
        if not rows_limits:
            rows_limits, rows_present = [[0.0]], [[False]]
        return cls(limits=AbstractTensor.tensor(rows_limits),
                   present=AbstractTensor.tensor(rows_present))


def system_totals(channels: ParticipantChannels):
    """Summed across participants: one extensive total per channel.

    Also returns which channels anybody published at all, because a channel no
    participant reported is not a total of zero.
    """
    published = channels.present
    contributed = channels.values * published
    return contributed.sum(dim=0), (published.sum(dim=0) > 0.0)


def participant_penalties(channels: ParticipantChannels,
                          limits: ParticipantLimits):
    """``measure / limit`` per participant per channel, zero where unjudged.

    One masked division over the whole set -- the same law ``penalties`` applies
    to a single participant, applied to all of them without folding first.
    """
    from ...common.tensors import AbstractTensor

    mask = channels.present * limits.present
    safe = AbstractTensor.maximum(limits.limits, 1e-30)
    return AbstractTensor.where(mask, channels.values / safe,
                                AbstractTensor.zeros_like(channels.values))


def trips(channels: ParticipantChannels, limits: ParticipantLimits,
          names: Iterable[str] | None = None) -> tuple[tuple[Any, str, float], ...]:
    """Every individual gate that opened: ``(participant, channel, ratio)``.

    A trip belongs to the participant that tripped.  ``names`` labels the
    participants for reporting; without it they are their own indices.  Names
    appear here and nowhere else on this path.
    """
    ratios = participant_penalties(channels, limits).tolist()
    labels = list(names) if names is not None else None
    opened: list[tuple[Any, str, float]] = []
    for index, row in enumerate(ratios):
        who = labels[index] if labels is not None and index < len(labels) else index
        for channel_id, ratio in enumerate(row):
            if float(ratio) > 1.0:
                opened.append((who, channel_name(channel_id), float(ratio)))
    return tuple(opened)


def judged_mask(spans: ChannelSpans, limits: ChannelLimits):
    """Channels that are BOTH published and limited.

    A measure with no limit is not a failure to judge -- nobody asked for it
    to be judged.  A limit with no measure is likewise not a violation; the
    step simply did not report on it.  Only the intersection is judged, and
    that intersection is what the dict's ``.get(name, 0.0)`` silently got
    wrong: it turned "not published" into a measure of zero.
    """
    return spans.present * limits.present


def penalties(spans: ChannelSpans, limits: ChannelLimits):
    """``measure / limit`` for every judged channel, zero elsewhere.

    One vectorized division over the whole registry, masked -- the same
    quantity the controller's generator expression computed by hashing a
    name per channel per attempt.
    """
    from ...common.tensors import AbstractTensor

    mask = judged_mask(spans, limits)
    safe = AbstractTensor.maximum(limits.limits, 1e-30)
    return AbstractTensor.where(
        mask, spans.values / safe,
        AbstractTensor.zeros_like(spans.values),
    )


def exceeded(spans: ChannelSpans, limits: ChannelLimits) -> tuple[str, ...]:
    """Every judged channel whose measure is over its limit, by NAME.

    Names appear here and nowhere else on this path: a refusal, a log line
    or an attempt record is read by a person, so it resolves ids back to
    words at the boundary rather than carrying strings through the step.
    """
    ratios = penalties(spans, limits).tolist()
    mask = judged_mask(spans, limits).tolist()
    return tuple(
        channel_name(index)
        for index, (ratio, judged) in enumerate(zip(ratios, mask))
        if judged and float(ratio) > 1.0
    )
