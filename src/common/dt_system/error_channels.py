"""Error channels as id-indexed spans, not a string-keyed dict.

``Metrics.error_channels`` is a ``dict[str, float]``, and a dict does not
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
* A **physical observable** feeding a named law (``tau``, ``shadow_growth``)
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
    def of(cls, published: Mapping[str, float] | None = None) -> "ChannelSpans":
        from ...common.tensors import AbstractTensor

        published = published or {}
        values: list[float] = []
        present: list[bool] = []
        for name in _CHANNEL_NAMES:
            measure = published.get(name)
            values.append(0.0 if measure is None else float(measure))
            present.append(measure is not None)
        return cls(
            values=AbstractTensor.tensor(values or [0.0]),
            present=AbstractTensor.tensor(present or [False]),
        )

    def has(self, channel_id: int) -> bool:
        """Whether this channel was published -- never "is it nonzero"."""
        return bool(self.present.tolist()[int(channel_id)])

    def value(self, channel_id: int) -> float | None:
        """This channel's measure, or None when it was not published."""
        index = int(channel_id)
        if not self.has(index):
            return None
        return float(self.values.tolist()[index])


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
    safe = AbstractTensor.where(
        limits.limits != 0.0, limits.limits,
        AbstractTensor.tensor([1.0] * len(limits.limits.tolist())),
    )
    return AbstractTensor.where(
        mask, spans.values / safe,
        AbstractTensor.tensor([0.0] * len(spans.values.tolist())),
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
