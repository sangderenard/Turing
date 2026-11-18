"""[CAPACITY] Each character's declared capacity: stateful or stateless (case 46d6ad2dc4b14ce497e53d5fba151b8d, stage 2)."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import agent_workspace as W  # noqa: E402

DAY = 1_794_960_000.0
LLAMA = {"state": "stateless", "context_tokens": 4096, "tools": "runner", "session": "none",
         "note": "1B model; the runner reads and posts for it"}


def _world(tmp_path):
    path = tmp_path / "accounts.db"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password_hash TEXT, display_name TEXT, created_at REAL)")
    raw.execute("CREATE TABLE temporary_accounts (user_id INTEGER PRIMARY KEY, expires_at REAL)")
    raw.executemany("INSERT INTO users VALUES (?,?,'x',?,0)", [(1, "boss", "boss"), (7, "llama", "llama")])
    raw.commit(); raw.close()
    db = W.connect(path)
    persona = W.save_document(db, 1, {"kind": "personality", "title": "Helper", "body": "Be helpful."})
    profile = W.save_profile(db, {"title": "Llama", "individual": "llama", "document_ids": [persona["id"]]})
    return db, profile


def test_a_declaration_is_saved_read_with_the_character_and_replaced(tmp_path):
    db, profile = _world(tmp_path)
    assert profile["capacity"] is None, "nothing is declared until an operator declares it"
    saved = W.save_capacity(db, 1, {"id": profile["id"], **LLAMA})
    assert saved["capacity"]["state"] == "stateless" and saved["capacity"]["context_tokens"] == 4096
    assert saved["capacity"]["tools"] == "runner" and saved["capacity"]["declared_by"] == 1
    again = W.save_capacity(db, 1, {"id": profile["id"], **LLAMA, "state": "stateful", "session": "resumable"})
    assert again["capacity"]["state"] == "stateful" and W.profile(db, profile["id"])["capacity"]["session"] == "resumable"
    assert db.execute("SELECT COUNT(*) FROM agent_capacity").fetchone()[0] == 1


@pytest.mark.parametrize("change,error", [
    ({"state": "sometimes"}, "state must be"), ({"tools": "both"}, "tools must be"), ({"session": "kept"}, "session must be"),
    ({"context_tokens": 100}, "context_tokens"), ({"context_tokens": "4096"}, "context_tokens"),
    ({"context_tokens": True}, "context_tokens"),
])
def test_only_the_declared_words_are_accepted(tmp_path, change, error):
    db, profile = _world(tmp_path)
    with pytest.raises(W.WorkspaceError, match=error):
        W.save_capacity(db, 1, {"id": profile["id"], **LLAMA, **change})


def test_an_unknown_character_is_refused_and_a_deleted_one_takes_its_declaration(tmp_path):
    db, profile = _world(tmp_path)
    with pytest.raises(W.WorkspaceError) as refused:
        W.save_capacity(db, 1, {"id": "0" * 32, **LLAMA})
    assert refused.value.status == 404
    W.save_capacity(db, 1, {"id": profile["id"], **LLAMA})
    W.delete_profile(db, profile["id"])
    assert db.execute("SELECT COUNT(*) FROM agent_capacity").fetchone()[0] == 0


def test_the_agent_is_told_how_it_is_run_first_in_every_branch(tmp_path):
    db, profile = _world(tmp_path)
    plain = W.orientation(db, 7, now=DAY)
    assert plain["capacity"] is None and plain["decision"][-1] == "how you are run: not declared"
    W.save_capacity(db, 1, {"id": profile["id"], **LLAMA})
    full = W.orientation(db, 7, now=DAY + 86400)
    assert full["preamble"].startswith(W.CAPACITY_LINES["stateless"]) and "Be helpful." in full["preamble"]
    assert full["decision"][2] == "how you are run: stateless, 4096 tokens, tools by the runner, session none"
    # Stage 4: a stateless agent remembers nothing, so it is never handed the short reminder: its full documents until a
    # reduction is cached (test_meowdoku_agent_guidance.py).
    again = W.orientation(db, 7, now=DAY + 86400 + 60)
    assert again["preamble"].startswith(W.CAPACITY_LINES["stateless"]) and "Be helpful." in again["preamble"]


def test_the_capacity_is_its_own_schema_module(tmp_path):
    db, _ = _world(tmp_path)
    modules = dict(db.execute("SELECT module, version FROM schema_version"))
    assert modules["agent_capacity"] == 1 and modules["agent_workspace"] == 2
