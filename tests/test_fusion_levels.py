"""The fusion ladder as a contract, including its refusals.

Two properties matter more than the vocabulary itself. A level must map onto
what the pipeline really does today -- not onto what it should do -- and an
unimplemented level must REFUSE rather than quietly behave like a neighbouring
one. Silently degrading is the failure shape this whole scale exists to make
impossible: a caller who asks for no collapsing and receives collapsing has
been handed a plausible wrong answer.
"""
from __future__ import annotations

import pytest

from src.compiler.fusion_levels import (
    IMPLEMENTED,
    INVARIANTS,
    LADDER,
    FusionLevel,
    from_precompile_only,
    preserves_at_least,
    rank,
    resolve,
    to_precompile_only,
)


def test_the_ladder_runs_from_most_preserving_to_most_collapsed():
    assert LADDER == (
        FusionLevel.PRESERVE,
        FusionLevel.NO_FUSION,
        FusionLevel.REGIONS,
        FusionLevel.FUSED,
    )
    assert rank(FusionLevel.PRESERVE) < rank(FusionLevel.FUSED)
    assert preserves_at_least(FusionLevel.PRESERVE, FusionLevel.FUSED)
    assert not preserves_at_least(FusionLevel.FUSED, FusionLevel.PRESERVE)


def test_every_level_states_an_invariant_that_could_be_falsified():
    """A level without a testable promise is a word, not a contract."""

    assert set(INVARIANTS) == set(LADDER)
    for level, invariant in INVARIANTS.items():
        assert invariant.strip(), f"{level} states no invariant"


def test_the_boolean_maps_onto_the_two_levels_that_exist():
    """This describes current behaviour, so it must not describe the goal."""

    assert from_precompile_only(True) is FusionLevel.REGIONS
    assert from_precompile_only(False) is FusionLevel.FUSED
    assert to_precompile_only(FusionLevel.REGIONS) is True
    assert to_precompile_only(FusionLevel.FUSED) is False


def test_only_the_delivered_levels_are_marked_implemented():
    assert IMPLEMENTED == frozenset({FusionLevel.REGIONS, FusionLevel.FUSED})


@pytest.mark.parametrize(
    "level", [FusionLevel.PRESERVE, FusionLevel.NO_FUSION]
)
def test_an_undelivered_level_refuses_rather_than_degrading(level):
    """The whole point: no silent fallback to a neighbouring level."""

    with pytest.raises(NotImplementedError) as raised:
        resolve(level)
    message = str(raised.value)
    assert level.value in message
    # The refusal has to say what is missing, or it is just "no".
    assert len(message) > 80
    with pytest.raises(NotImplementedError):
        to_precompile_only(level)


def test_preserve_names_the_elision_it_would_have_to_disable():
    """Ties the level to the concrete behaviour blocking it."""

    with pytest.raises(NotImplementedError) as raised:
        resolve(FusionLevel.PRESERVE)
    assert "loop-carried" in str(raised.value)


def test_an_unknown_level_is_a_value_error_not_a_default():
    with pytest.raises(ValueError, match="unknown fusion level"):
        resolve("mostly")


def test_strings_resolve_so_callers_need_not_import_the_enum():
    assert resolve("regions") is FusionLevel.REGIONS
    assert resolve("fused") is FusionLevel.FUSED
