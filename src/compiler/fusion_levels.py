"""How much of an authored program survives compilation, as a stated level.

``precompile_only`` names *when in the pipeline you stopped*, not *what you
got back*, which is the thing a caller actually reasons about. This module
introduces that second vocabulary without moving any pipeline stage: the
boolean keeps working and keeps meaning what it meant, while a level says what
the result is allowed to have lost.

The levels are ordered from most preserving to most collapsed::

    PRESERVE   no merging and no elimination -- one authored operator in,
               one operator out
    NO_FUSION  no merging of adjacent operators; elimination still permitted
    REGIONS    region boundaries survive; their interiors may fuse freely
    FUSED      no guarantee

Each carries a **testable invariant**, not a description, so a level is a
contract a run can be checked against rather than a word in a signature. That
matters here specifically: this tree's compilation failures are characteristically
silent and plausible -- a lowering reporting "complete" over zero instructions,
a captured training step whose gradient is frozen, a capture returning an empty
program and zero shortfalls. A level with an invariant is a thing that can be
falsified.

TWO OPEN QUESTIONS, recorded rather than quietly decided
--------------------------------------------------------

**Is this one ladder or two axes?** Merging adjacent operators and eliminating
operators outright are different operations, and the ladder is only sound if
each level strictly disables everything above it plus more. The distinction is
not hypothetical: the ``loop_carried`` shortfall (see
``tests/test_loop_carried_producers.py``) comes from region *elision*, which is
elimination -- ``NO_FUSION`` would not affect it and only ``PRESERVE`` reaches
it. If the nesting turns out not to hold, this becomes two independent knobs.
The test that settles it is whether ``NO_FUSION`` output is always a subset of
``PRESERVE`` output under the same source.

**``precompile_only`` currently controls two unrelated things.** Per
``aot_compile``'s module docstring, ``emit_glsl`` -- whether GLSL source is
produced at all -- is gated purely by ``precompile_only``. So the boolean means
both "how much collapsing" and "which artifacts are emitted". Mapping a level
onto it inherits that conflation: asking for ``PRESERVE`` would silently also
mean "no GLSL", for no principled reason. Separating emission is most of the
value of ever replacing the boolean, and is deliberately NOT done here.

WHAT IS ACTUALLY HONORED TODAY
------------------------------

``REGIONS`` and ``FUSED`` are the two behaviours the pipeline already has;
they are what ``precompile_only`` True/False select. ``PRESERVE`` and
``NO_FUSION`` are declared but not yet implemented, and asking for one RAISES.
It would be easy, and wrong, to let them fall back to ``REGIONS``: a caller
who asked for no collapsing and silently got collapsing is the exact failure
shape this vocabulary exists to make impossible.
"""

from __future__ import annotations

from enum import Enum


class FusionLevel(str, Enum):
    """How much collapsing a compilation is permitted to do."""

    PRESERVE = "preserve"
    NO_FUSION = "no_fusion"
    REGIONS = "regions"
    FUSED = "fused"


# Most preserving first. Membership order is the ladder; ``rank`` below reads
# it, so adding a level in the right position is the only edit a new level
# needs.
LADDER: tuple[FusionLevel, ...] = (
    FusionLevel.PRESERVE,
    FusionLevel.NO_FUSION,
    FusionLevel.REGIONS,
    FusionLevel.FUSED,
)


# What each level PROMISES, phrased so a test can fail it. These are the
# contract; the prose in the docstring is only commentary on them.
INVARIANTS: dict[FusionLevel, str] = {
    FusionLevel.PRESERVE: (
        "every authored operator appears in the emitted IR, in source order; "
        "operator count is preserved"
    ),
    FusionLevel.NO_FUSION: (
        "no emitted step carries more than one authored operator"
    ),
    FusionLevel.REGIONS: (
        "region boundaries present in the plan are present in the emitted IR; "
        "region interiors are unconstrained"
    ),
    FusionLevel.FUSED: (
        "no guarantee; the compiler may merge and eliminate freely"
    ),
}


# The levels the pipeline can actually deliver right now.
IMPLEMENTED: frozenset[FusionLevel] = frozenset(
    {FusionLevel.REGIONS, FusionLevel.FUSED}
)


# Why each unimplemented level is not available, so the refusal says something
# useful instead of "not supported".
_UNIMPLEMENTED_REASON: dict[FusionLevel, str] = {
    FusionLevel.PRESERVE: (
        "region formation currently elides a subgraph whose output traces "
        "solely to its own loop-carried input, and nothing turns that off. "
        "That elision is what PRESERVE would have to disable; see "
        "tests/test_loop_carried_producers.py for the shape it takes"
    ),
    FusionLevel.NO_FUSION: (
        "contiguous fusing has no off switch distinct from REGIONS; the "
        "plan-level distinction exists but is not plumbed to a caller"
    ),
}


def rank(level: FusionLevel) -> int:
    """Position on the ladder: 0 is the most preserving."""

    return LADDER.index(FusionLevel(level))


def preserves_at_least(level: FusionLevel, other: FusionLevel) -> bool:
    """True when ``level`` keeps everything ``other`` keeps, and possibly more."""

    return rank(level) <= rank(other)


def from_precompile_only(precompile_only: bool) -> FusionLevel:
    """The level today's boolean actually selects.

    This is a description of current behaviour, not an aspiration: the boolean
    chooses between the two implemented levels and nothing else.
    """

    return FusionLevel.REGIONS if precompile_only else FusionLevel.FUSED


def to_precompile_only(level: FusionLevel) -> bool:
    """The boolean that delivers ``level``, or a refusal naming why not."""

    resolved = resolve(level)
    return resolved is FusionLevel.REGIONS


def resolve(level: FusionLevel | str) -> FusionLevel:
    """Normalize ``level``, refusing one the pipeline cannot actually deliver.

    Raises rather than degrading. A caller who asked for no collapsing and
    silently received collapsing has been given a plausible wrong answer, which
    is the failure shape this vocabulary exists to prevent.
    """

    try:
        chosen = FusionLevel(level)
    except ValueError:
        raise ValueError(
            f"unknown fusion level {level!r}; expected one of "
            f"{[member.value for member in LADDER]}"
        ) from None
    if chosen not in IMPLEMENTED:
        raise NotImplementedError(
            f"fusion level {chosen.value!r} is declared but not yet delivered: "
            f"{_UNIMPLEMENTED_REASON[chosen]}. Implemented today: "
            f"{sorted(member.value for member in IMPLEMENTED)}"
        )
    return chosen


__all__ = [
    "IMPLEMENTED",
    "INVARIANTS",
    "LADDER",
    "FusionLevel",
    "from_precompile_only",
    "preserves_at_least",
    "rank",
    "resolve",
    "to_precompile_only",
]
