"""[TIMERS] Agent runs are never skipped silently; ticks every minute; the bell runs beside work (case 46d6ad2d)."""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import agent_workspace as W  # noqa: E402

DEPLOY = ROOT / "examples" / "meowdoku" / "deploy"


def _world(tmp_path):
    path = tmp_path / "accounts.db"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password_hash TEXT, display_name TEXT, created_at REAL)")
    raw.execute("CREATE TABLE temporary_accounts (user_id INTEGER PRIMARY KEY, expires_at REAL)")
    raw.execute("CREATE TABLE account_roles (user_id INTEGER, role TEXT)")
    raw.executemany("INSERT INTO users VALUES (?,?,'x',?,0)", [(1, "boss", "boss"), (5, "claude", "claude")])
    raw.commit(); raw.close()
    db = W.connect(path)
    doc = W.save_document(db, 1, {"kind": "guidance", "title": "Schedule", "body": "deficit = weight × 12 − done"})
    W.save_profile(db, {"title": "Claude", "individual": "claude", "document_ids": [doc["id"]]})
    db.close()
    return path


def test_a_character_outside_the_console_code_page_still_resolves(tmp_path):
    """2026-11-17: U+2212 in a document crashed the cp1252 print, and every claude and codex run was skipped."""
    path = _world(tmp_path)
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    done = subprocess.run([sys.executable, str(DEPLOY / "profile_prompt.py"), "--accounts-db", str(path), "--individual", "claude"],
                          capture_output=True, env=env, timeout=60)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert "weight × 12 − done" in done.stdout.decode("utf-8")


def _launcher():
    return (DEPLOY / "Start-AgentDay.ps1").read_text(encoding="utf-8")


def test_the_launcher_reads_python_as_utf8_and_logs_what_stops_a_run():
    text = _launcher()
    assert "$env:PYTHONIOENCODING = 'utf-8'" in text and "[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)" in text
    tick = text[text.index("# ---------------------------------------------------------------- one tick for one agent"):]
    assert 'Log "${Agent}: tick failed before a run:' in tick
    assert 'Log "${Agent}: $($run.reason) run not started:' in tick
    assert tick.index("Invoke-AgentRun $Agent") > tick.index("$lock.Dispose()"), "the run goes after the lock is let go"
    assert tick.index("Add-AgentRecord $AgentDays $run.kind") < tick.index("$lock.Dispose()"), "its record is written first"


def test_ticks_every_minute_in_parallel_and_the_bell_has_its_own_limits():
    text = _launcher()
    assert "-RepetitionInterval (New-TimeSpan -Minutes 1)" in text and "-MultipleInstances Parallel" in text
    assert "[int]$BellTimeoutMinutes = 4" in text and "[int]$MaxBells = 3" in text
    tick = text[text.index("# ---------------------------------------------------------------- one tick for one agent"):]
    assert "$_.kind -ne 'run'" in tick, "a bell run, a seat or a console of the account blocks the tick"
    assert "-not $working -and" in tick, "the hourly run waits while the account's work run goes; the bell does not"
    run = text[text.index("function Invoke-AgentRun"):text.index("# ---------------------------------------------------------------- the tasks")]
    assert "'bell.id'" in run and "($limit * 60)" in run, "a bell run has its own memory file and its own timeout"
