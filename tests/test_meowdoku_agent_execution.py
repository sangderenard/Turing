"""The web queue never executes a command; the AgentDay owner task claims it."""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku import agent_execution  # noqa: E402


def test_only_server_owned_non_llama_accounts_join_execution(tmp_path):
    accounts = tmp_path / "accounts.db"
    registry = tmp_path / "admin_control.db"
    with sqlite3.connect(registry) as db:
        db.execute("CREATE TABLE external_agents(name TEXT, status TEXT)")
        db.executemany("INSERT INTO external_agents VALUES(?,?)", [("mistral", "installed"), ("waiting", "declared")])
    assert agent_execution.server_owned(accounts, "claude")
    assert agent_execution.server_owned(accounts, "codex")
    assert agent_execution.server_owned(accounts, "gemini")
    assert agent_execution.server_owned(accounts, "mistral")
    assert not agent_execution.server_owned(accounts, "llama")
    assert not agent_execution.server_owned(accounts, "waiting")
    assert not agent_execution.server_owned(accounts, "admin")


def test_jobs_are_owned_and_completed_after_a_single_claim(tmp_path):
    with agent_execution.connect(tmp_path / "accounts.db") as db:
        first = agent_execution.submit(db, 3, "claude", "exec", "Get-Location")
        second = agent_execution.submit(db, 4, "codex", "python", "print(1 + 1)")
        with pytest.raises(agent_execution.ExecutionError):
            agent_execution.read(db, 4, first["id"])
        assert agent_execution.claim(db, "claude") == {"id": first["id"], "kind": "exec", "source": "Get-Location"}
        assert agent_execution.claim(db, "claude") is None
        assert agent_execution.claim(db, "codex")["id"] == second["id"]
        agent_execution.complete(db, first["id"], {"output": "C:\\Users\\agents", "exit_code": 0})
        assert agent_execution.read(db, 3, first["id"])["status"] == "complete"
        with pytest.raises(agent_execution.ExecutionError):
            agent_execution.complete(db, first["id"], {"output": "again", "exit_code": 0})


def test_queue_limits_and_failed_process(tmp_path):
    with agent_execution.connect(tmp_path / "accounts.db") as db:
        with pytest.raises(agent_execution.ExecutionError):
            agent_execution.submit(db, 3, "claude", "python", "")
        with pytest.raises(agent_execution.ExecutionError):
            agent_execution.submit(db, 3, "claude", "shell", "whoami")
        for index in range(agent_execution.MAX_ACTIVE_PER_ACCOUNT):
            agent_execution.submit(db, 3, "claude", "python", f"print({index})")
        with pytest.raises(agent_execution.ExecutionError) as full:
            agent_execution.submit(db, 3, "claude", "python", "print(5)")
        assert full.value.status == 429
        job = agent_execution.claim(db, "claude")
        agent_execution.complete(db, job["id"], {"output": "failure", "exit_code": 1})
        assert agent_execution.read(db, 3, job["id"])["status"] == "failed"
