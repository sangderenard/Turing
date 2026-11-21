"""A held review cannot pass its timeout or be replaced before an admin releases it."""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku import branch_review  # noqa: E402


@pytest.fixture
def db(tmp_path):
    connection = sqlite3.connect(tmp_path / "reviews.db", isolation_level=None)
    connection.execute("CREATE TABLE account_roles (user_id INTEGER PRIMARY KEY, role TEXT NOT NULL)")
    connection.execute("INSERT INTO account_roles VALUES (1, 'admin')")
    yield connection
    connection.close()


def submit(db, head="a" * 40, now=100.0):
    return branch_review.submit(
        db, user_id=2, username="author", repository="turing", branch="feature",
        base="main", head=head, timeout_s=30.0, delete_branch=True,
        known=["turing"], now=now)


def test_hold_blocks_timeout_queue_and_same_branch_replacement(db):
    review = submit(db)
    held = branch_review.hold(db, review["id"], user_id=1, username="admin",
                              reason="needs another pass", now=110.0)
    assert held["standing"] == "held" and held["held_by"] == "admin"
    assert branch_review.get(db, review["id"], now=500.0)["standing"] == "held"
    assert branch_review.queue(db, now=500.0) == []
    assert not branch_review.accepted_head(db, "turing", "main", "a" * 40, now=500.0)
    with pytest.raises(branch_review.ReviewError, match="held"):
        submit(db, "b" * 40, now=500.0)
    with pytest.raises(branch_review.ReviewError):
        branch_review.decide(db, review["id"], user_id=1, username="admin",
                             head="a" * 40, decision="approved", now=500.0)
    with pytest.raises(branch_review.ReviewError):
        branch_review.withdraw(db, review["id"], user_id=2, now=500.0)


def test_release_grants_fresh_window_and_keeps_audit(db):
    review = submit(db)
    with pytest.raises(branch_review.ReviewError) as denied:
        branch_review.hold(db, review["id"], user_id=2, username="author",
                           reason="not my decision", now=105.0)
    assert denied.value.status == 403
    branch_review.hold(db, review["id"], user_id=1, username="admin",
                       reason="needs another pass", now=110.0)
    resumed = branch_review.release(db, review["id"], user_id=1, username="admin",
                                    note="review resumed", now=500.0)
    assert resumed["standing"] == "waiting"
    assert resumed["accept_after"] == 500.0 + branch_review.RELEASE_WINDOW_S
    assert [event["action"] for event in resumed["hold_events"]] == ["hold", "release"]
    assert branch_review.queue(db, now=500.0) == []
    assert branch_review.get(db, review["id"], now=resumed["accept_after"])["standing"] == "accepted"
    decided = branch_review.decide(db, review["id"], user_id=1, username="admin",
                                   head="a" * 40, decision="approved", now=510.0)
    assert decided["standing"] == "approved"
    with pytest.raises(branch_review.ReviewError):
        branch_review.hold(db, review["id"], user_id=1, username="admin",
                           reason="too late", now=520.0)
