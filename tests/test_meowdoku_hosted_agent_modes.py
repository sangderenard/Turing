"""Scoped last-word controls and audit for hosted agent accounts."""
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku import hosted_agent_modes as modes


def test_last_word_scope_precedence_and_actor_gate(tmp_path):
    db = sqlite3.connect(tmp_path / "accounts.db", isolation_level=None)
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE)")
    db.executemany("INSERT INTO users(id,username) VALUES(?,?)",
                   [(1, "boss"), (2, "llama"), (3, "alice")])
    modes.prepare(db)
    assert modes.setting(db, 2, "global", "")["effective"] is False
    with pytest.raises(modes.ModeError, match="operator or admin"):
        modes.set_mode(db, agent_id=2, actor_id=3, actor_role="member",
                       scope="global", key="", mode="on")
    modes.set_mode(db, agent_id=2, actor_id=1, actor_role="operator",
                   scope="global", key="", mode="on")
    assert modes.setting(db, 2, "conversation", "alice")["effective"] is True
    modes.set_mode(db, agent_id=2, actor_id=1, actor_role="operator",
                   scope="conversation", key="alice", mode="off")
    assert modes.setting(db, 2, "conversation", "alice")["effective"] is False
    modes.set_mode(db, agent_id=2, actor_id=1, actor_role="operator",
                   scope="conversation", key="alice", mode="inherit")
    assert modes.setting(db, 2, "conversation", "alice")["effective"] is True
    assert db.execute("SELECT count(*) FROM hosted_agent_mode_audit").fetchone()[0] == 3
    with pytest.raises(modes.ModeError, match="global mode"):
        modes.set_mode(db, agent_id=2, actor_id=1, actor_role="operator",
                       scope="global", key="", mode="inherit")
    db.close()
