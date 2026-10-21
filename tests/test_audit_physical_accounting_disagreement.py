"""The audit treats only ``None`` as an unset physical field.

Wrong record: the physical check used ``resident in (None, False)``, and
``0 == False``, so a scalar's accounting ``program_abi_rank=0`` was invisible:
never reported unposted when the fact had no rank, and never compared when the
fact posted rank 1.  Rule: rank 0 is a value; ``sequence_arena`` alone is a
bool whose False (not an arena member) equals the fact's default.
"""
from __future__ import annotations

from tests.test_structural_concordance import _audited


def test_rank_zero_accounting_without_posted_rank_is_unposted():
    found = _audited({"program_abi_rank": 0})
    assert [f.kind for f in found] == ["ssa-physical-unposted"]
    assert "program_abi_rank=0" in found[0].detail


def test_rank_zero_accounting_against_posted_rank_one_disagrees():
    found = _audited({"program_abi_rank": 0}, rank=1)
    assert [f.kind for f in found] == ["ssa-publication-type-disagreement"]
    assert "program_abi_rank=0" in found[0].detail


def test_rank_zero_accounting_matching_posted_rank_zero_is_quiet():
    assert not _audited({"program_abi_rank": 0}, rank=0)


def test_arena_false_is_unset_for_that_key_only():
    assert not _audited({"sequence_arena": False})
    found = _audited({"sequence_arena": True})
    assert [f.kind for f in found] == ["ssa-physical-unposted"]
