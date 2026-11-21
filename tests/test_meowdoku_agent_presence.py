"""Agent status uses observed scheduler and checkout records, with no invented outbound activity."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku import agent_presence as presence  # noqa: E402


def test_scheduler_due_cooldown_and_pid_reuse(tmp_path, monkeypatch):
    now = datetime(2026, 11, 17, 21, 25, tzinfo=timezone.utc).timestamp()
    state = tmp_path / "state"
    state.mkdir()
    (state / "codex.json").write_text(json.dumps({"last_hourly": datetime.fromtimestamp(now - 3900, timezone.utc).isoformat(),
                                                    "bell_off_until": datetime.fromtimestamp(now + 600, timezone.utc).isoformat()}))
    os.utime(state / "codex.json", (now, now))
    assert presence._schedule(tmp_path, "codex", 20, now)["due"] is True
    assert presence._schedule(tmp_path, "codex", 20, now)["bell_off_until"] == now + 600
    os.utime(state / "codex.json", (now - 300, now - 300))
    assert presence._schedule(tmp_path, "codex", 20, now)["next_due"] is None
    running = tmp_path / "running"
    running.mkdir()
    (running / "run-agents-codex-12.json").write_text(json.dumps({"agent": "codex", "kind": "run", "pid": 12,
        "started": datetime.fromtimestamp(now - 100, timezone.utc).isoformat()}))
    monkeypatch.setattr(presence, "_process_started", lambda pid: now - 30)  # reused PID began after record
    assert presence._run_records(tmp_path, now)["codex"] == []
    monkeypatch.setattr(presence, "_process_started", lambda pid: now - 110)
    assert presence._run_records(tmp_path, now)["codex"][0]["kind"] == "run"
    (running / "console-agents-codex-13.json").write_text(json.dumps({"account": "agents-codex", "kind": "console", "pid": 13,
        "started": datetime.fromtimestamp(now - 100, timezone.utc).isoformat()}))
    assert {row["kind"] for row in presence._run_records(tmp_path, now)["codex"]} == {"run", "console"}


def test_non_object_records_do_not_hide_a_valid_run(tmp_path, monkeypatch):
    now = datetime(2026, 11, 17, 21, 25, tzinfo=timezone.utc).timestamp()
    running = tmp_path / "running"
    running.mkdir()
    for index, value in enumerate(([], None, "bad")):
        (running / f"invalid-{index}.json").write_text(json.dumps(value))
    (running / "run-agents-codex-12.json").write_text(json.dumps({"agent": "codex", "kind": "run", "pid": 12,
        "started": datetime.fromtimestamp(now - 100, timezone.utc).isoformat()}))
    monkeypatch.setattr(presence, "_process_started", lambda pid: now - 110)
    snapshot = presence.snapshot(tmp_path / "missing.db", tmp_path, now=now)
    codex = next(agent for agent in snapshot["agents"] if agent["username"] == "codex")
    assert codex["channels"]["local"]["state"] == "active"
    assert [run["kind"] for run in codex["runs"]] == ["run"]


def test_checkout_location_requires_recent_observation_and_outbound_unknown(tmp_path):
    now = datetime(2026, 11, 17, 21, 25, tzinfo=timezone.utc).timestamp()
    with presence._lock:
        presence._recent.clear()
    db = tmp_path / "accounts.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE checkout_calls (id INTEGER PRIMARY KEY, at REAL, username TEXT, repository TEXT, path TEXT, operation TEXT, ok INTEGER)")
        conn.executemany("INSERT INTO checkout_calls(at,username,repository,path,operation,ok) VALUES (?,?,?,?,?,?)", [
            (now - 60, "codex", "turing", "examples/meowdoku/server.py", "read", 1),
            (now - 10, "agents-codex", "turing", "examples/meowdoku/worldmap.py", "write", 1),
            (30, "claude", "turing", "../secret", "read", 1),
            (40, "gemini", "turing", "src/compiler/fortran_c_shell.py", "read", 0),
        ])
    (tmp_path / "running").mkdir()
    result = presence.snapshot(db, tmp_path, now=now)
    assert result["locations"] == [{"username": "codex", "repository": "turing", "path": "examples/meowdoku/worldmap.py", "at": now - 10,
                                     "age_seconds": 10}]
    assert presence.checkout_locations(db, now + presence.RECENT_SECONDS + 1) == []
    codex = next(agent for agent in result["agents"] if agent["username"] == "codex")
    assert codex["channels"]["mcp_out"]["state"] == "unknown"
    assert codex["channels"]["local"]["state"] == "unobserved"


def test_hosted_and_app_agents_have_truthful_unscheduled_badges(tmp_path):
    now = datetime(2026, 11, 17, 21, 25, tzinfo=timezone.utc).timestamp()
    result = presence.snapshot(tmp_path / "missing.db", tmp_path, now=now)
    by_name = {agent["username"]: agent for agent in result["agents"]}
    for name in ("llama", "gemma", "operator-claude", "operator-codex"):
        assert name in by_name
        assert by_name[name]["schedule"] is None
        assert by_name[name]["channels"]["local"]["state"] == "unknown"
        assert by_name[name]["channels"]["api_out"]["state"] == "unknown"


def test_outbound_requires_known_agent_and_real_channel():
    with presence._lock:
        presence._recent.clear()
    assert presence._observed("codex", 100)["api_out"]["state"] == "unknown"
    assert not presence.note_outbound("random-user", "api_out", now=100)
    assert not presence.note_outbound("codex", "api_in", now=100)
    assert presence.note_outbound("agents-codex", "api_out", now=100)
    presence.note_inbound("agents-codex", mcp=True, now=100)
    assert presence._observed("codex", 129)["api_out"] == {"state": "recent", "at": 100}
    assert presence._observed("codex", 129)["mcp_in"] == {"state": "recent", "at": 100}
    assert presence._observed("codex", 131)["api_out"]["state"] == "unobserved"
    assert presence._observed("codex", 100)["mcp_out"]["state"] == "unknown"


def test_inbound_is_recent_only(monkeypatch):
    with presence._lock:
        presence._recent.clear()
    presence.note_inbound("claude", mcp=True, now=100)
    assert presence._observed("claude", 129)["mcp_in"]["state"] == "recent"
    assert presence._observed("claude", 131)["mcp_in"]["state"] == "unobserved"
    assert presence._observed("claude", 129)["api_in"]["state"] == "unobserved"
