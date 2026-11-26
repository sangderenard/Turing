"""Bundles and redelegations hold only flagged work; unflagged reviews pass by them as before (review_bundles.py)."""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku import branch_review, review_bundles  # noqa: E402

ADMIN, OPERATOR, ALICE, BOB = 1, 2, 3, 4


@pytest.fixture
def db(tmp_path):
    connection = sqlite3.connect(tmp_path / "reviews.db", isolation_level=None)
    connection.execute("CREATE TABLE account_roles (user_id INTEGER PRIMARY KEY, role TEXT NOT NULL)")
    connection.execute("INSERT INTO account_roles VALUES (1, 'admin'), (2, 'operator')")
    for user_id, name, branch in ((ALICE, "alice", "a"), (BOB, "bob", "b")):
        branch_review.register(connection, user_id=user_id, username=name, repository="turing", branch=branch,
                               known=["turing"], now=10.0)
    branch_review.register(connection, user_id=ALICE, username="alice", repository="turing", branch="other",
                           known=["turing"], now=10.0)
    yield connection
    connection.close()


def submit(db, user_id, username, branch, head, now=100.0):
    return branch_review.submit(db, user_id=user_id, username=username, repository="turing", branch=branch,
                                base="main", head=head, timeout_s=30.0, delete_branch=True, known=["turing"], now=now)


def bundle(db, now=110.0):
    return review_bundles.flag(db, user_id=OPERATOR, username="operator", repository="turing", branches=["a", "b"],
                               reason="the same files, two versions", known=["turing"], now=now)


def test_unflagged_reviews_pass_as_before(db):
    review = submit(db, ALICE, "alice", "other", "c" * 40)
    assert branch_review.get(db, review["id"], now=500.0)["standing"] == "accepted"
    assert [record["id"] for record in branch_review.queue(db, now=500.0)] == [review["id"]]


def test_flagging_is_for_operators_and_names_checked_out_branches(db):
    with pytest.raises(branch_review.ReviewError) as denied:
        review_bundles.flag(db, user_id=ALICE, username="alice", repository="turing", branches=["a", "b"],
                            reason="mine", known=["turing"])
    assert denied.value.status == 403
    with pytest.raises(branch_review.ReviewError, match="not checked out"):
        review_bundles.flag(db, user_id=OPERATOR, username="operator", repository="turing", branches=["a", "zz"],
                            reason="x", known=["turing"])
    told = bundle(db)
    assert told["tell"] == ["alice", "bob"]
    assert "single merge" in told["bundle"]["notice"] and told["bundle"]["id"] in told["bundle"]["notice"]
    with pytest.raises(branch_review.ReviewError, match="already in bundle"):
        bundle(db)
    rows = {row["branch"]: row for row in review_bundles.checked_out(db)}
    assert rows["a"]["bundle"] == told["bundle"]["id"] and rows["other"]["bundle"] is None


def test_bundled_reviews_consolidate_into_one_meta_review(db):
    first = submit(db, ALICE, "alice", "a", "a" * 40)
    second = submit(db, BOB, "bob", "b", "b" * 40)
    made = bundle(db)["bundle"]
    for review in (first, second):
        assert branch_review.get(db, review["id"], now=500.0)["standing"] == "consolidating"
    assert branch_review.queue(db, now=500.0) == []
    assert not branch_review.accepted_head(db, "turing", "main", "a" * 40, now=500.0)
    with pytest.raises(branch_review.ReviewError, match="consolidating"):
        branch_review.decide(db, first["id"], user_id=ADMIN, username="admin", head="a" * 40, decision="approved",
                             now=500.0)
    # Alice merges b into a and resubmits a: the meta review.
    meta = submit(db, ALICE, "alice", "a", "d" * 40, now=600.0)
    tips = {"a": "d" * 40, "b": "b" * 40}
    with pytest.raises(branch_review.ReviewError, match="does not contain"):
        review_bundles.consolidate(db, made["id"], meta["id"], user_id=ALICE, username="alice",
                                   contains=lambda repository, branch, head: branch == "a", now=610.0)
    with pytest.raises(branch_review.ReviewError) as denied:
        review_bundles.consolidate(db, made["id"], meta["id"], user_id=99, username="stranger",
                                   contains=lambda *_: True, now=610.0)
    assert denied.value.status == 403
    done = review_bundles.consolidate(db, made["id"], meta["id"], user_id=ALICE, username="alice",
                                      contains=lambda repository, branch, head: tips[branch] in (head, "b" * 40),
                                      now=620.0)
    assert done["bundle"]["status"] == "merged" and done["bundle"]["meta_review_id"] == meta["id"]
    assert done["tell"] == ["bob"]
    assert branch_review.get(db, second["id"], now=700.0)["standing"] == "withdrawn"
    assert branch_review.get(db, meta["id"], now=700.0)["standing"] == "accepted"


def test_dissolve_lets_reviews_flow_again(db):
    review = submit(db, ALICE, "alice", "a", "a" * 40)
    made = bundle(db)["bundle"]
    assert branch_review.get(db, review["id"], now=500.0)["standing"] == "consolidating"
    review_bundles.dissolve(db, made["id"], user_id=ADMIN, username="admin", reason="not related after all", now=510.0)
    assert branch_review.get(db, review["id"], now=520.0)["standing"] == "accepted"


def test_redelegating_a_review_opens_a_ticket_of_paths(db):
    review = submit(db, ALICE, "alice", "a", "a" * 40)
    opened = []

    def open_ticket(title, description, diagnostics):
        opened.append((title, description, diagnostics))
        return "t" * 32
    paths = [{"title": "Split the schema change", "detail": "tables only"}, {"title": "The page", "detail": ""}]
    with pytest.raises(branch_review.ReviewError, match="2 to 12"):
        review_bundles.redelegate(db, kind="review", source_id=review["id"], paths=paths[:1], note="", user_id=OPERATOR,
                                  username="operator", open_ticket=open_ticket, now=200.0)
    told = review_bundles.redelegate(db, kind="review", source_id=review["id"], paths=paths, note="too much at once",
                                     user_id=OPERATOR, username="operator", open_ticket=open_ticket, now=200.0)
    assert told["redelegation"]["ticket_id"] == "t" * 32 and told["tell"] == ["alice"]
    title, description, diagnostics = opened[0]
    assert "1. Split the schema change" in description and "2. The page" in description
    assert diagnostics["source_id"] == review["id"]
    assert branch_review.get(db, review["id"], now=500.0)["standing"] == "redelegated"
    assert branch_review.queue(db, now=500.0) == []
    # The work comes back as a new review of a branch: it is not redelegated.
    again = submit(db, ALICE, "alice", "a", "e" * 40, now=600.0)
    assert branch_review.get(db, again["id"], now=700.0)["standing"] == "accepted"


def test_redelegating_a_bundle_settles_it(db):
    first = submit(db, ALICE, "alice", "a", "a" * 40)
    made = bundle(db)["bundle"]
    told = review_bundles.redelegate(db, kind="bundle", source_id=made["id"],
                                     paths=[{"title": "one"}, {"title": "two"}], note="", user_id=ADMIN,
                                     username="admin", open_ticket=lambda *_: "x" * 32, now=300.0)
    assert sorted(told["tell"]) == ["alice", "bob"]
    assert review_bundles.get(db, made["id"])["status"] == "redelegated"
    assert branch_review.get(db, first["id"], now=500.0)["standing"] == "redelegated"
