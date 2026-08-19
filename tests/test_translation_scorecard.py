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
    2: "EXECUTE",      # a value becomes both a formal and a region output
    3: "PASSED",       # fixed: regions now scheduled in dependency order
    4: "PASSED",       # loop, one carried value
    5: "PASSED",       # loop, compound body
    6: "PASSED",       # two carried values, fixed in topological_reducer
    7: "PASSED",       # adam-shaped triple carry: the goal shape, compiled
    8: "MATERIALIZE",  # linked call binds the updated id; use-before-def, refused
    9: "LOWER",        # carried value round-tripped through a call
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


def test_the_frontier_is_not_silently_all_green():
    """If everything passes, the corpus stopped being a measurement.

    The point of a scorecard is to sit at the edge. A corpus where every level
    passes has either been fixed -- in which case extend it -- or been trimmed
    to what already works, which is the failure mode this guards.
    """

    failing = [level for level, stage in EXPECTED.items() if stage != "PASSED"]
    assert failing, "extend the corpus past the current frontier"


def test_the_recorded_failures_are_not_all_compilation_failures():
    """Part of the frontier lowers cleanly and is simply wrong.

    The frontier has shrunk twice today (frozen carried value, region
    scheduling), but the dead-frame-storage journey still compiles, runs, and
    is wrong with no stage reporting anything. That is the defect class the
    round trip exists to detect, and the scorecard keeps at least one such
    journey until the class is extinct.
    """

    reached_execution = [
        level
        for level, stage in EXPECTED.items()
        if stage in {"EXECUTE", "EQUIVALENT"}
    ]
    assert len(reached_execution) >= 1
