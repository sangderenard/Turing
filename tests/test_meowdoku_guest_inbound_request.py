"""A look preview can ask for inbound access without gaining it or flooding review."""
import time
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku import guest_review  # noqa: E402


def test_inbound_look_request_waits_and_is_bounded(tmp_path):
    path = tmp_path / "accounts.db"
    db = guest_review._connect(path)
    try:
        db.executescript("""
            CREATE TABLE users(id INTEGER PRIMARY KEY, username TEXT);
            CREATE TABLE temporary_accounts(user_id INTEGER PRIMARY KEY, expires_at REAL);
            CREATE TABLE preview_recovery(user_id INTEGER PRIMARY KEY, created_at REAL, frozen_at REAL);
        """)
        now = time.time()
        db.execute("INSERT INTO users VALUES(1,'look_1')")
        db.execute("INSERT INTO temporary_accounts VALUES(1,?)", (now + 604800,))
        db.execute("INSERT INTO preview_recovery VALUES(1,?,NULL)", (now,))
        who = SimpleNamespace(user_id=1, role="trial")
        status = guest_review._inbound_status(db, who)
        assert status["eligible_at"] > now
        with pytest.raises(guest_review.GuestReviewError) as waiting:
            guest_review._inbound_request(db, who, {"purpose": "I want a durable account to research the system."})
        assert waiting.value.status == 429
        assert db.execute("SELECT COUNT(*) FROM guest_inbound_requests").fetchone()[0] == 0

        db.execute("UPDATE preview_recovery SET created_at=? WHERE user_id=1", (now - 181,))
        receipt = guest_review._inbound_request(db, who, {"purpose": "I want a durable account to research the system."})
        assert receipt["state"] == "requested"
        assert guest_review._inbound_status(db, who)["request_id"] == receipt["id"]
        with pytest.raises(guest_review.GuestReviewError) as repeated:
            guest_review._inbound_request(db, who, {"purpose": "I want a second durable account for research."})
        assert repeated.value.status == 409

        for user_id in range(2, guest_review.INBOUND_DAILY + 1):
            db.execute("INSERT INTO users VALUES(?,?)", (user_id, f"look_{user_id}"))
            db.execute("INSERT INTO temporary_accounts VALUES(?,?)", (user_id, now + 604800))
            db.execute("INSERT INTO preview_recovery VALUES(?,?,NULL)", (user_id, now - 181))
            guest_review._inbound_request(db, SimpleNamespace(user_id=user_id, role="trial"),
                                          {"purpose": "I want a durable account to research the system."})
        last = guest_review.INBOUND_DAILY + 1
        db.execute("INSERT INTO users VALUES(?,?)", (last, f"look_{last}"))
        db.execute("INSERT INTO temporary_accounts VALUES(?,?)", (last, now + 604800))
        db.execute("INSERT INTO preview_recovery VALUES(?,?,NULL)", (last, now - 181))
        with pytest.raises(guest_review.GuestReviewError) as full:
            guest_review._inbound_request(db, SimpleNamespace(user_id=last, role="trial"),
                                          {"purpose": "I want a durable account to research the system."})
        assert full.value.status == 429
        queue = guest_review.inbound_request_queue(path)
        assert len(queue) == guest_review.INBOUND_DAILY
        assert queue[0]["untrusted"]["purpose"]
    finally:
        db.close()
