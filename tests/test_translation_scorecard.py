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
}


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


def test_the_corpus_records_when_the_frontier_is_cleared():
    """All ten journeys pass. The guard flips from "not all green" to a
    dated fact, and the next frontier must come from EXTENDING the corpus --
    conditionals, nested loops, multiple entrypoints, dynamic storage -- not
    from trimming it. If a level regresses, the per-level pins above fail
    loudly and this note is again false.
    """

    failing = [level for level, stage in EXPECTED.items() if stage != "PASSED"]
    assert failing == [], f"the frontier regressed: {failing}"


def test_the_silent_failure_class_is_extinct_in_this_corpus():
    """Every journey that once compiled-and-was-wrong now computes correctly.

    The classes this corpus caught -- frozen carried value, formal/region
    collision, undeclared storage formal, mis-bound anchored call, elided
    body -- are individually pinned in their own suites so a regression in
    any one fails by name rather than by a wrong number here.
    """

    assert all(stage == "PASSED" for stage in EXPECTED.values())
