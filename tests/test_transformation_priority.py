"""Rewrite convergence is a proof property, independent of elapsed time."""
import pytest

from src.compiler.transformation_priority import (
    TransformationConflict, TransformationLedger, TransformationRule,
    frame_transformation_ledger,
)


def test_declared_rule_cycle_rejected_before_execution():
    with pytest.raises(TransformationConflict, match="split -> join"):
        TransformationLedger((TransformationRule("join", 1), TransformationRule("split", 2)),
                             {"join": ("split",), "split": ("join",)})


def test_fresh_allocation_ids_cannot_restart_join_split_cycle():
    ledger = frame_transformation_ledger()
    identity = ("caller", 7, "callee", 3)
    assert ledger.propose(identity, "receiver_field", ("state", "limit"), before=19, after=4)
    assert ledger.propose(identity, "distinct_owner", ("record", "second"), before=4, after=20)
    for new_allocation in range(21, 121):
        assert not ledger.propose(identity, "receiver_field", ("state", "limit"),
                                  before=new_allocation, after=4)
    assert len(ledger.events) == 3
    assert ledger.events[-1]["retained"] == ("distinct_owner", ("record", "second"))


def test_equal_priority_retains_incumbent_and_records_challenger():
    ledger = frame_transformation_ledger()
    ledger.propose(("call", 7), "distinct_owner", "first")
    assert not ledger.propose(("call", 7), "distinct_owner", "second")
    assert ledger.events[-1]["retained"] == ("distinct_owner", "first")
    assert ledger.events[-1]["proof"] == "second"
    assert ledger.events[-1]["reason"] == "incumbent_tie"
    assert ledger.propose(("call", 7), "distinct_owner", "first")


def test_repeated_proof_is_idempotent_and_identities_stay_independent():
    ledger = frame_transformation_ledger()
    for identity in (1, 2):
        for _ in range(4):
            assert ledger.propose(identity, "receiver_field", identity)
    assert len(ledger.events) == 2
    assert ledger.propose(1, "distinct_result", "output")
    assert not ledger.propose(1, "distinct_owner", "temporary")


def test_equal_priority_different_rules_keep_physical_incumbent():
    ledger = TransformationLedger((TransformationRule("a", 1), TransformationRule("b", 1)), {})
    assert ledger.propose("field", "a", "first", before=0, after=12)
    assert not ledger.propose("field", "b", "second", before=12, after=13)
    assert ledger.incumbent_target("field") == 12


def test_same_proof_cannot_replace_incumbent_by_allocating_again():
    ledger = frame_transformation_ledger()
    assert ledger.propose("field", "distinct_owner", "owner", before=0, after=12)
    for allocation in range(13, 30):
        assert not ledger.propose("field", "distinct_owner", "owner", before=0, after=allocation)
        assert ledger.incumbent_target("field") == 12
    assert len(ledger.events) == 2


def test_exact_argument_binding_prevents_anonymous_owner_split():
    ledger = frame_transformation_ledger()
    identity = ("caller", 80, "column", 0)
    assert ledger.propose(
        identity, "distinct_owner", ("value", ("row", 0)),
        before=3, after=101,
    )
    assert ledger.propose(
        identity, "exact_argument_binding", (114, 0),
        before=101, after=3,
    )
    assert ledger.incumbent_target(identity) == 3
    assert ledger.events[-1]["rule"] == "exact_argument_binding"
    assert ledger.events[-1]["accepted"] is True


def test_increasing_but_undeclared_transition_rejected():
    ledger = TransformationLedger((TransformationRule("a", 1), TransformationRule("b", 2)), {})
    ledger.propose(1, "a", "proof")
    with pytest.raises(TransformationConflict, match="Undeclared"):
        ledger.propose(1, "b", "proof")
