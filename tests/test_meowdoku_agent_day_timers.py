"""[TIMERS] Agent runs are never skipped silently; ticks every minute; the bell runs beside work (case 46d6ad2d)."""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

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


# ---- case 26f097a3ed5e4462b2acbe70017384ce: an agent CLI never reads the launcher's script

def test_every_run_line_gives_its_cli_an_empty_standard_input():
    text = _launcher()
    block = text[text.index("$Agents = [ordered]@{"):text.index("$Preamble = @'")]
    assert block.count("run = '$null | & ") == 2, "claude and codex"
    assert "`$null | & '$Python' '$Workspace\\turing\\examples\\meowdoku\\deploy\\gemini_agent.py'" in block


CHILD = "import sys; data = sys.stdin.read(); print('CHILD READ', len(data))"


def _split_pipe(guard: str) -> str:
    """pwsh runs a script given on stdin in two parts; its middle line starts a child that reads standard input."""
    python = sys.executable.replace("'", "''")
    first = f"$x = 1\n{guard}& '{python}' -c \"{CHILD}\"\nex"
    process = subprocess.Popen(["pwsh", "-NoProfile", "-NonInteractive", "-Command", "-"], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    process.stdin.write(first); process.stdin.flush()
    time.sleep(2)                                   # the child has started before the rest of the script arrives
    process.stdin.write("it 7\n"); process.stdin.close()
    out = process.stdout.read()
    process.wait(timeout=60)
    return out, process.returncode


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="needs PowerShell 7")
def test_an_empty_standard_input_keeps_the_script_whole():
    out, code = _split_pipe("")
    assert "CHILD READ 5" in out, "unguarded, the child takes the script's last line ('it 7\\n')"
    out, code = _split_pipe("$null | ")
    assert "CHILD READ 0" in out and code == 7, "guarded, the child reads nothing and the script's own exit runs"


def test_the_tick_starts_the_reducer_under_its_lock_and_a_reducer_never_blocks_a_run():
    """[GUIDANCE] case 46d6ad2dc4b14ce497e53d5fba151b8d, stage 4."""
    text = _launcher()
    tick = text[text.index("# ---------------------------------------------------------------- one tick for one agent"):]
    assert "$_.kind -ne 'reduce'" in tick, "the account's reducer is left out of what blocks a bell or a work run"
    start = tick.index("'guidance_reducer.py'")
    assert tick.index("--reduce-pending") < start < tick.index("$lock.Dispose()"), "read, then started, under the lock"
    assert tick.index("Add-AgentRecord $AgentDays 'reduce'") < tick.index("$lock.Dispose()"), "its record is written first"
    assert "$declared.reduce -and" in tick
    block = text[text.index("$Agents = [ordered]@{"):text.index("$Preamble = @'")]
    assert block.count("reduce = $true") == 2, "the CLIs guidance_reducer.py has an adapter for: claude and codex"
