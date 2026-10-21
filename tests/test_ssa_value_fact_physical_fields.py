"""SSAValueFact physical fields (stage 1): defaults, pickle, carry-forward."""
import dataclasses
import pickle

import pytest

from src.compiler.concordance_declarations import (
    KEEP_PHYSICAL, REVOKE_PHYSICAL, SSAValueFact, SSAValueOrigin,
    rebuilt_ssa_value_fact,
)
from src.compiler.identity_concordance import ConcordanceRefusal

PHYSICAL = dict(physical_dtype="float64", storage="span", rank=2,
                arena=True, device="cpu")


def test_old_three_field_pickle_restores_defaults():
    fact = SSAValueFact("float64", (3,), SSAValueOrigin.MINTED)
    state = dict(fact.__dict__)
    for name in PHYSICAL:  # an OLD pickle's __dict__ has only the 3 keys
        state.pop(name, None)
    old = SSAValueFact.__new__(SSAValueFact)
    object.__setattr__(old, "__dict__", state)
    loaded = pickle.loads(pickle.dumps(old))
    assert (loaded.dtype, loaded.shape, loaded.origin) == (
        "float64", (3,), SSAValueOrigin.MINTED)
    assert (loaded.physical_dtype, loaded.storage, loaded.rank,
            loaded.arena, loaded.device) == (None, None, None, False, None)
    assert loaded == fact
    assert pickle.loads(pickle.dumps(fact)) == fact


def test_replace_round_trip_keeps_physical_fields():
    fact = SSAValueFact("float64", (3,), SSAValueOrigin.MINTED, **PHYSICAL)
    loaded = pickle.loads(pickle.dumps(fact))
    assert loaded == fact
    again = dataclasses.replace(loaded, dtype="int64")
    assert again.dtype == "int64"
    assert all(getattr(again, k) == v for k, v in PHYSICAL.items())


def test_storage_is_validated():
    with pytest.raises(ValueError):
        SSAValueFact("int64", (), SSAValueOrigin.MINTED, storage="bogus")
    for storage in ("scalar", "span", "reference", "record", "sequence",
                    "table", "keyed", None):
        SSAValueFact("int64", (), SSAValueOrigin.MINTED, storage=storage)


def test_rebuild_keeps_by_default_and_revoke_is_explicit():
    recorded = SSAValueFact("float64", (3,), SSAValueOrigin.MINTED, **PHYSICAL)
    kept = rebuilt_ssa_value_fact(recorded, "int64", (), recorded.origin)
    assert kept.dtype == "int64" and kept.shape == ()
    assert all(getattr(kept, k) == v for k, v in PHYSICAL.items())
    assert rebuilt_ssa_value_fact(
        recorded, "float64", (3,), recorded.origin, KEEP_PHYSICAL) == recorded
    with pytest.raises(ConcordanceRefusal):
        rebuilt_ssa_value_fact(recorded, "float64", (3,), recorded.origin,
                               {"physical_dtype": None})
    revoked = rebuilt_ssa_value_fact(
        recorded, "float64", (3,), recorded.origin, REVOKE_PHYSICAL)
    assert (revoked.physical_dtype, revoked.storage, revoked.rank,
            revoked.arena, revoked.device) == (None, None, None, False, None)
    plain = SSAValueFact("float64", (3,), SSAValueOrigin.MINTED)
    set_ = rebuilt_ssa_value_fact(plain, "float64", (3,), plain.origin,
                                  {"storage": "scalar", "rank": 0})
    assert set_.storage == "scalar" and set_.rank == 0
    assert pickle.loads(pickle.dumps(KEEP_PHYSICAL)) is KEEP_PHYSICAL
