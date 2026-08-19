"""The recorded frontier: how far a program of given complexity actually gets.

``tools/translation_scorecard.py`` scores journeys through
Python -> SSA -> Python -> execution -> equivalence. This pins the result, so
the frontier is a fact under test rather than something rediscovered.

Both directions are informative. A level that regresses fails here. A level
that gets FIXED also fails here, and the expectation should then be moved
forward -- deliberately, with the fix -- rather than loosened.

Note what the stages mean. Every defect recorded below except level 7 reaches
EXECUTE: those programs lower without a shortfall, materialize, and run. Only
comparing against the authored Python catches them. That is why the scorecard
scores equivalence and not compilation.
"""
from __future__ import annotations

import pytest

from tools.translation_scorecard import JOURNEYS, STAGES, score


# The frontier as measured. Update deliberately, alongside a fix.
EXPECTED: dict[int, str] = {
    0: "PASSED",       # straight-line arithmetic
    1: "PASSED",       # two parameters
    2: "PASSED",       # fixed: storage formals declared, leased at entry
    3: "PASSED",       # fixed: regions now scheduled in dependency order
    4: "PASSED",       # loop, one carried value
    5: "PASSED",       # loop, compound body
    6: "PASSED",       # two carried values, fixed in topological_reducer
    7: "PASSED",       # adam-shaped triple carry: the goal shape, compiled
    8: "PASSED",       # fixed: carried slots seeded in the preheader
    9: "PASSED",       # fixed: callsites are schedulable statements now
    # --- the symbol-coordination ladder, measured 2026-08-19 ---
    10: "PASSED",      # keyword arguments bind by name, not call order
    11: "PASSED",      # a default fills its slot; an override displaces it
    12: "PASSED",      # declaration order survives against the alphabet
    13: "PASSED",      # transposed arguments stay distinct (anti-symmetric)
    14: "MATERIALIZE", # a thrice-rebound parameter name does not materialize
    15: "PASSED",      # int literals + int parameter meeting float arithmetic
    16: "MATERIALIZE", # authored if: materializer CRASHES (NoneType .id) --
                       # a defect in its refusal path, not a designed refusal
    17: "PASSED",      # while: FIXED same-day, twice -- the double-published
                       # scalar (stale pre-loop identity resurrected by the
                       # recovery pass) and the latch guard testing the
                       # pre-update carried value (one extra iteration)
    18: "MATERIALIZE", # any() over a generator: predicate region unrendered
}

#: Levels 0-9 are the original corpus whose frontier was cleared; levels 10+
#: are the symbol-coordination ladder whose stalls are the CURRENT frontier.
ORIGINAL_FRONTIER = tuple(range(10))


@pytest.mark.parametrize("journey", JOURNEYS, ids=[j.name for j in JOURNEYS])
def test_each_journey_reaches_its_recorded_stage(journey):
    stage, detail = score(journey)
    assert stage == EXPECTED[journey.level], (
        f"level {journey.level} ({journey.name}) reached {stage}, "
        f"expected {EXPECTED[journey.level]}: {detail}"
    )


def test_every_journey_has_a_recorded_expectation():
    """A new journey must be scored, not silently unmeasured."""

    assert {journey.level for journey in JOURNEYS} == set(EXPECTED)


def test_the_stages_are_ordered_from_earliest_to_latest_failure():
    assert STAGES == ("LOWER", "MATERIALIZE", "EXECUTE", "EQUIVALENT")


def test_the_original_corpus_frontier_stays_cleared():
    """All ten original journeys pass -- a dated fact (2026-08-19 still
    true). The corpus was then EXTENDED, per this test's own doctrine, with
    the symbol-coordination ladder (levels 10+), whose stalls are the
    current frontier and are pinned per-level above. If an original level
    regresses, the per-level pins fail loudly and this note is again false.
    """

    failing = [
        level for level in ORIGINAL_FRONTIER
        if EXPECTED[level] != "PASSED"
    ]
    assert failing == [], f"the original frontier regressed: {failing}"


def test_the_silent_failure_class_is_named_where_it_lives():
    """Runs-and-is-wrong is extinct in the original corpus, and level 17
    demonstrated the ladder's purpose the day it was added: the first
    while rung ever scored executed and was wrong twice over (a
    double-published scalar, then a latch guard lagging one iteration),
    and both defects were found and fixed because equivalence was scored.
    """

    assert all(
        EXPECTED[level] == "PASSED" for level in ORIGINAL_FRONTIER
    )
    assert EXPECTED[17] == "PASSED"
