"""Loan sync ownership and dry-run checks without invoking Git."""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku import branch_gate, branch_review  # noqa: E402


def test_sync_loan_guards_and_dry_run(tmp_path, monkeypatch):
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL)")
    db.executemany("INSERT INTO users VALUES (?,?)", [(1, "ann"), (2, "bob")])
    branch_review.register(db, user_id=1, username="ann", repository="work",
                           branch="codex/example", known=["work"])
    db.execute("CREATE TABLE checkouts (repository TEXT, branch TEXT, user_id INTEGER, path TEXT)")
    folder = tmp_path / "loan"
    folder.mkdir()
    db.execute("INSERT INTO checkouts VALUES (?,?,?,?)", ("work", "codex/example", 1, str(folder)))
    policy = {**branch_gate.DEFAULTS, "repositories": {"work": {"path": str(tmp_path), "default": "main"}}}
    with pytest.raises(branch_gate.GateError, match="registered"):
        branch_gate.sync_loan(policy, db, "work", "codex/example", "bob", apply=True)
    with pytest.raises(branch_gate.GateError, match="default branch"):
        branch_gate.sync_loan(policy, db, "work", "main", "ann", apply=True)

    calls = []

    def fake_git(path, *args, **kwargs):
        calls.append(args)
        if args == ("rev-parse", "--show-toplevel"):
            return str(folder)
        if args == ("symbolic-ref", "-q", "--short", "HEAD"):
            return "codex/example"
        if args == ("rev-parse", "--verify", "refs/remotes/origin/main^{commit}"):
            return "upstream"
        if args == ("rev-parse", "HEAD"):
            return "loan"
        return ""

    monkeypatch.setattr(branch_gate, "git", fake_git)
    result = branch_gate.sync_loan(policy, db, "work", "codex/example", "ann")
    assert result["state"] == "ready"
    assert not any(args[0] in ("fetch", "merge") for args in calls)
