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
    assert ledger.propose(identity, "linked_record_member", ("state", "limit"), before=19, after=4)
    assert ledger.propose(identity, "distinct_owner", ("record", "second"), before=4, after=20)
    for new_allocation in range(21, 121):
        assert not ledger.propose(identity, "linked_record_member", ("state", "limit"),
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
            assert ledger.propose(identity, "linked_record_member", identity)
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


def test_repeated_rejection_is_found_by_its_book_key_at_scale():
    """A rejection is a keyed fact: recorded once, found without a scan.

    The ledger used to decide "already rejected?" by scanning every event,
    and each event access rebuilt the event log from the book: 200
    identities took 87 s and the woodshop dt system's call-result
    settlement (3892 call operands) stalled in its first round.
    """
    import time

    ledger = frame_transformation_ledger()
    count = 5000
    started = time.perf_counter()
    for index in range(count):
        ledger.propose(("call", index), "distinct_owner", ("first", index))
    for _repeat in range(2):
        for index in range(count):
            assert not ledger.propose(
                ("call", index), "distinct_owner", ("second", index),
            )
    elapsed = time.perf_counter() - started
    rejected = [event for event in ledger.events if not event["accepted"]]
    assert len(ledger.events) == 2 * count
    assert len(rejected) == count
    assert ledger._rejection_page.latest((
        ledger.scope, ("call", 7), "distinct_owner", ("second", 7),
        "distinct_owner", ("first", 7),
    )) == count + 7
    assert elapsed < 10.0, elapsed


def test_ledger_rows_are_sourced_without_the_proposer_naming_cells():
    """A decision stands on the incumbent decision it supersedes (its first
    claim, with no proof cell on the book, is the chain's NOVEL root); an
    event stands on the decision cell it records; a rejection stands on the
    retained decision.  No row is a raw primitive."""
    ledger = frame_transformation_ledger()
    book = ledger.book
    identity = ("caller", 9, "callee", 1)
    assert ledger.propose(identity, "linked_record_member", "p1", after=4)
    first = ledger.decision_cell(identity)
    assert ledger.propose(identity, "distinct_owner", "p2", after=20)
    second = ledger.decision_cell(identity)
    assert not ledger.propose(identity, "linked_record_member", "p1", after=4)
    sources = lambda ref: {src.key for src, _ in book.edges_into(ref)}
    assert not sources(first)  # NOVEL root: a mint row, no inbound edge
    assert first.key in sources(second)
    event_page = book.page("transformation_event")
    refs = [book.latest_ref(book.registry.pages["transformation_event"], row)
            for row in event_page.scope_rows(ledger.scope)]
    assert sources(refs[0]) == {first.key}
    assert sources(refs[1]) == {second.key}
    assert sources(refs[2]) == {second.key}
