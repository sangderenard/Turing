"""[SESSIONS] Kept bell sessions for the stateful service (case 46d6ad2dc4b14ce497e53d5fba151b8d, stage 5)."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples"), str(ROOT / "examples" / "meowdoku" / "deploy")]

from meowdoku import agent_guidance as G  # noqa: E402
from meowdoku import agent_workspace as W  # noqa: E402

DAY = 1_794_960_000.0
RESUMABLE = {"state": "stateless", "context_tokens": 200000, "tools": "own", "session": "resumable"}


def _world(tmp_path):
    path = tmp_path / "accounts.db"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password_hash TEXT, display_name TEXT, created_at REAL)")
    raw.execute("CREATE TABLE temporary_accounts (user_id INTEGER PRIMARY KEY, expires_at REAL)")
    raw.executemany("INSERT INTO users VALUES (?,?,'x',?,0)", [(1, "boss", "boss"), (7, "llama", "llama"), (8, "claude", "claude"),
                                                             (9, "nobody", "nobody")])
    raw.commit(); raw.close()
    db = W.connect(path)
    rules = W.save_document(db, 1, {"kind": "guidance", "title": "Claude rules", "body": "Review only diffs you read."})
    claude = W.save_profile(db, {"title": "Claude", "individual": "claude", "document_ids": [rules["id"]]})
    W.save_capacity(db, 1, {"id": claude["id"], **RESUMABLE})
    llama = W.save_profile(db, {"title": "Llama", "individual": "llama", "document_ids": []})
    W.save_capacity(db, 1, {"id": llama["id"], **RESUMABLE, "session": "none"})
    return db, {"claude": claude["id"], "llama": llama["id"], "rules": rules}


def test_a_kept_session_reads_stateful_while_it_lives_and_the_declaration_is_untouched(tmp_path):
    db, ids = _world(tmp_path)
    digest = G.sources(db, ids["claude"])["digest"]
    assert W.keep_session(db, 8, {"action": "keep"}, DAY)["state"] == "stateful"
    running = W.running_capacity(db, ids["claude"], DAY + 1)
    assert running["state"] == "stateful" and running["declared_state"] == "stateless" and running["kept_since"] == DAY
    assert W.capacity(db, ids["claude"])["state"] == "stateless", "the operator's declaration is never rewritten"
    assert G.sources(db, ids["claude"])["digest"] == digest, "keeping a session is not a change of the sources"
    W.orientation(db, 8, now=DAY + 2)
    again = W.orientation(db, 8, now=DAY + 3)
    assert again["capacity"]["state"] == "stateful" and again["preamble"].startswith(W.CAPACITY_LINES["stateful"])
    assert again["guidance"] is None and "You were oriented earlier today" in again["preamble"], "it was briefed in its session"
    W.keep_session(db, 8, {"action": "keep"}, DAY + 300)
    assert W.running_capacity(db, ids["claude"], DAY + 400)["kept_since"] == DAY, "a heartbeat keeps the same session"
    assert W.running_capacity(db, ids["claude"], DAY + 300 + W.KEPT_SECONDS + 1)["state"] == "stateless", "a stale heartbeat"
    W.keep_session(db, 8, {"action": "keep"}, DAY + 2000)
    assert W.running_capacity(db, ids["claude"], DAY + 2001)["kept_since"] == DAY + 2000, "after a lapse, a new session"
    W.keep_session(db, 8, {"action": "drop", "error": "SeatError: the agent's process ended"}, DAY + 2002)
    assert W.running_capacity(db, ids["claude"], DAY + 2003)["state"] == "stateless"
    assert W.profile(db, ids["claude"])["session"]["error"] == "SeatError: the agent's process ended"


def test_only_a_character_declared_resumable_keeps_a_session(tmp_path):
    db, _ = _world(tmp_path)
    with pytest.raises(W.WorkspaceError) as llama:
        W.keep_session(db, 7, {"action": "keep"}, DAY)
    assert llama.value.status == 409
    with pytest.raises(W.WorkspaceError) as nobody:
        W.keep_session(db, 9, {"action": "keep"}, DAY)
    assert nobody.value.status == 409, "no character at all"
    with pytest.raises(W.WorkspaceError):
        W.keep_session(db, 8, {"action": "resume"}, DAY)


def test_the_orientation_names_the_digest_a_session_is_briefed_again_on(tmp_path):
    db, ids = _world(tmp_path)
    first = W.orientation(db, 8, now=DAY)["source_digest"]
    assert first == G.sources(db, ids["claude"])["digest"]
    W.save_document(db, 1, {**ids["rules"], "body": "Review only diffs you have read in full."})
    assert W.orientation(db, 8, now=DAY + 1)["source_digest"] != first
    assert W.orientation(db, 9, now=DAY)["source_digest"] is None, "no character, no digest"


def test_sessions_are_their_own_schema_module(tmp_path):
    db, _ = _world(tmp_path)
    assert dict(db.execute("SELECT module, version FROM schema_version"))["agent_sessions"] == 1


def test_the_launcher_reads_whether_a_session_is_resumable(tmp_path):
    import profile_prompt
    db, _ = _world(tmp_path)
    db.close()
    args = ["--accounts-db", str(tmp_path / "accounts.db"), "--resumable"]
    assert profile_prompt.main(args + ["--individual", "claude"]) == 0
    assert profile_prompt.main(args + ["--individual", "llama"]) == 2
    assert profile_prompt.main(args + ["--individual", "nobody"]) == 2


def test_a_session_adapter_keeps_its_chat_tools_and_a_seat_does_not():
    import roundtable_seat as S
    seen = []
    S.Claude(program="claude", popen=lambda command, **_: seen.append(command) or _Proc())
    S.Claude(program="claude", popen=lambda command, **_: seen.append(command) or _Proc(), disallowed=())
    assert "mcp__speaktome__chat_post" in seen[0] and "--disallowedTools" not in seen[1]


class _Proc:
    class _Out:
        def readline(self):
            return b""
    stdout, stdin = _Out(), None


# ---------------------------------------------------------------- the session's loop, with fakes
class _Rest:
    def __init__(self, digests):
        self.digests, self.kept = list(digests), []

    def call(self, path, body=None, **_):
        if path.startswith("/api/agent-workspace/orientation"):
            digest = self.digests.pop(0) if len(self.digests) > 1 else self.digests[0]
            return {"preamble": "You are {agent}, signed in as {login}.", "source_digest": digest}
        self.kept.append(body)
        return {"state": "stateful"}


class _Agent:
    made = []

    def __init__(self, fail_on=None):
        self.asked, self.closed, self.fail_on = [], False, fail_on
        _Agent.made.append(self)

    def ask(self, text, seconds):
        self.asked.append(text)
        if self.fail_on is not None and len(self.asked) == self.fail_on:
            from roundtable_seat import SeatError
            raise SeatError("the agent's process ended")
        return "done"

    def close(self):
        self.closed = True


def _session(rest, looks, agent=lambda: _Agent(), today=lambda: "2026-11-18", **kw):
    """The bell's answers in turn, the last one repeated; a clock 10 s a step; one day unless today says otherwise."""
    import agent_session as A
    looks, clock = list(looks), iter(range(0, 100_000, 10))
    last = [set()]

    def look():
        if looks:
            last[0] = looks.pop(0)
        return set(last[0])
    return A, A.Session(rest, look, agent, fill=lambda t: t.replace("{agent}", "claude").replace("{login}", "claude"),
                        memory="a" * 32, clock=lambda: next(clock), sleep=lambda s: None, today=today,
                        idle=200, poll=10, heartbeat=60, recheck=0, **kw)


def test_one_session_briefed_once_then_handed_only_what_is_new_then_dropped_when_idle():
    _Agent.made = []
    rest = _Rest(["d1"])
    A, session = _session(rest, [{"m1"}, {"m1"}, {"m1", "m2"}, {"m1", "m2", "t5:9"}, set()])
    assert session.run("1 new message(s)") == 3
    agent = _Agent.made[0]
    assert len(_Agent.made) == 1 and agent.closed, "one session, closed when idle"
    assert "You are claude, signed in as claude." in agent.asked[0] and "a" * 32 in agent.asked[0], "the full brief, once"
    assert agent.asked[0].endswith("You were woken because: 1 new message(s)")
    assert agent.asked[1] == A.notice({"m2"}) and agent.asked[2] == A.notice({"t5:9"}), "afterwards only what is new"
    assert "1 message(s) and 0 review thread(s)" in agent.asked[1] and "0 message(s) and 1 review thread(s)" in agent.asked[2]
    assert rest.kept[0] == {"action": "keep"} and rest.kept[-1] == {"action": "drop"}
    assert rest.kept.count({"action": "keep"}) >= 2, "heartbeats while it lives"


def test_a_changed_document_or_a_new_day_briefs_it_again_and_a_broken_turn_starts_fresh():
    _Agent.made = []
    rest = _Rest(["d1", "d1", "d2", "d2"])
    turned = lambda: "2026-11-19" if _Agent.made and len(_Agent.made[0].asked) >= 3 else "2026-11-18"   # noqa: E731
    A, session = _session(rest, [set(), {"m1"}, {"m1", "m2"}, {"m1", "m2", "m3"}, {"m1", "m2", "m3", "m4"}], today=turned)
    session.run("first")
    asked = _Agent.made[0].asked
    assert "Your brief again" not in asked[1], "same documents, same day: only the notice"
    assert asked[2].startswith("Your documents changed. Your brief again, in full:")
    assert asked[3].startswith("A new day began (UTC). Your brief again, in full:"), "and again on a new day"
    assert "Your brief again" not in asked[4], "once"
    _Agent.made = []
    rest = _Rest(["d1"])
    A, session = _session(rest, [set(), {"m1"}, {"m1", "m2"}], agent=lambda: _Agent(fail_on=1) if not _Agent.made else _Agent())
    session.run("first")
    assert len(_Agent.made) == 2 and _Agent.made[0].closed, "a broken session is closed"
    assert "You are claude" in _Agent.made[1].asked[0], "the fresh one is briefed in full"
