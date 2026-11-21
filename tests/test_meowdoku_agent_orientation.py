"""[ORIENT] Each account's orientation: shared basics, then its own profile; first today in full, later a reminder."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import agent_workspace as W  # noqa: E402

DAY = 1_794_960_000.0                                    # a fixed UTC moment


def _world(tmp_path):
    path = tmp_path / "accounts.db"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password_hash TEXT, display_name TEXT, created_at REAL)")
    raw.execute("CREATE TABLE temporary_accounts (user_id INTEGER PRIMARY KEY, expires_at REAL)")
    raw.executemany("INSERT INTO users VALUES (?,?,'x',?,0)", [(1, "boss", "boss"), (5, "claude", "claude"), (9, "nobody", "nobody")])
    raw.commit(); raw.close()
    db = W.connect(path)
    basics = W.save_document(db, 1, {"kind": "orientation", "title": "SpeakToMe basics", "body": "What the site is."})
    persona = W.save_document(db, 1, {"kind": "personality", "title": "Engineer", "body": "Help; triage."})
    short = W.save_document(db, 1, {"kind": "guidance", "title": "Returning", "body": "Carry on."})
    profile = W.save_profile(db, {"title": "Claude", "individual": "claude", "document_ids": [persona["id"]]})
    return db, basics, persona, short, profile


def test_first_interaction_today_gets_everything_and_later_ones_a_reminder(tmp_path):
    db, basics, persona, _, _ = _world(tmp_path)
    first = W.orientation(db, 5, now=DAY)
    assert first["first_today"] and first["branch"] == W.FIRST_TODAY
    assert [d["title"] for d in first["documents"]] == ["SpeakToMe basics", "Engineer"], "the shared basics come first"
    assert "What the site is." in first["preamble"] and "Help; triage." in first["preamble"]
    again = W.orientation(db, 5, now=DAY + 60)
    assert not again["first_today"] and "Help; triage." not in again["preamble"] and "Engineer" in again["preamble"]
    assert "Help; triage." in W.orientation(db, 5, now=DAY + 120, full=True)["preamble"], "full=1 always gives the text"
    assert W.orientation(db, 5, now=DAY + 86400)["first_today"], "a new UTC day starts over"


def test_a_profile_may_shape_each_branch_with_its_jobs(tmp_path):
    db, _, persona, short, profile = _world(tmp_path)
    W.save_profile_job(db, {"profile_id": profile["id"], "job_key": W.RETURNING, "title": "Returning",
                            "replacements": {persona["id"]: short["id"]}, "additions": []})
    W.orientation(db, 5, now=DAY)
    again = W.orientation(db, 5, now=DAY + 60)
    assert "Carry on." in again["preamble"] and "Help; triage." not in again["preamble"]
    assert again["decision"][1] == "profile: yes, job returning"


def test_an_account_without_a_profile_still_gets_the_basics(tmp_path):
    db, *_ = _world(tmp_path)
    got = W.orientation(db, 9, now=DAY)
    assert [d["title"] for d in got["documents"]] == ["SpeakToMe basics"] and got["decision"][1] == "profile: none"
