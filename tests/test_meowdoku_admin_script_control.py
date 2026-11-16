"""Focused authorization tests for the owner-run deploy broker."""
import json
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

from examples.meowdoku import admin_script_control as control
from meowdoku.accounts import _hash_password


@pytest.fixture
def setup(tmp_path, monkeypatch):
    accounts_db = tmp_path / "sql" / "accounts.db"
    accounts_db.parent.mkdir()
    with sqlite3.connect(accounts_db) as db:
        db.execute("CREATE TABLE users(id INTEGER PRIMARY KEY, username TEXT, password_hash TEXT)")
        db.execute("INSERT INTO users VALUES(1,'admin',?)", (_hash_password("admin-pass"),))
        db.execute("INSERT INTO users VALUES(2,'operator',?)", (_hash_password("operator-pass"),))
    roles = {1: "admin", 2: "operator"}
    monkeypatch.setattr(control, "_account_role", lambda path, user_id: roles.get(user_id))
    db = control.connect(tmp_path)
    yield db, accounts_db, roles
    db.close()


def test_grant_expires_before_worker_claim(setup):
    db, accounts_db, _ = setup
    preset = control.command(db, accounts_db, 1, True, {
        "action": "preset", "password": "admin-pass", "label": "Status",
        "script": "Status.ps1", "argv": [], "timeout": 30,
    })["preset_id"]
    control.command(db, accounts_db, 1, True, {
        "action": "grant", "password": "admin-pass", "preset_id": preset,
        "username": "operator", "hours": 1,
    })
    job = control.command(db, accounts_db, 2, False, {
        "action": "run", "password": "operator-pass", "preset_id": preset,
    })["job_id"]
    db.execute("UPDATE control_grants SET expires=?", (time.time() - 1,))
    db.commit()
    assert control.claim(db, accounts_db) is None
    assert db.execute("SELECT status FROM control_jobs WHERE id=?", (job,)).fetchone()[0] == "refused"


def test_revocation_and_role_loss_prevent_queued_execution(setup):
    db, accounts_db, roles = setup
    preset = control.command(db, accounts_db, 1, True, {
        "action": "preset", "password": "admin-pass", "label": "Status",
        "script": "Status.ps1", "argv": [], "timeout": 30,
    })["preset_id"]
    grant = control.command(db, accounts_db, 1, True, {
        "action": "grant", "password": "admin-pass", "preset_id": preset,
        "username": "operator", "hours": 24,
    })["grant_id"]
    job = control.command(db, accounts_db, 2, False, {
        "action": "run", "password": "operator-pass", "preset_id": preset,
    })["job_id"]
    control.command(db, accounts_db, 1, True, {
        "action": "revoke", "password": "admin-pass", "grant_id": grant,
    })
    db.commit()
    assert control.claim(db, accounts_db) is None
    assert db.execute("SELECT status FROM control_jobs WHERE id=?", (job,)).fetchone()[0] == "refused"
    db.execute("UPDATE control_grants SET revoked=NULL WHERE id=?", (grant,))
    db.execute("UPDATE control_jobs SET status='queued' WHERE id=?", (job,))
    db.commit()
    roles[2] = None
    assert control.claim(db, accounts_db) is None


def test_only_fixed_scripts_and_saved_arguments(setup):
    db, accounts_db, _ = setup
    for script, argv in [("../other.ps1", []), ("Status.ps1", ["-Password:steal"])]:
        with pytest.raises(control.ControlError):
            control.command(db, accounts_db, 1, True, {
                "action": "preset", "password": "admin-pass", "label": "Bad",
                "script": script, "argv": argv, "timeout": 30,
            })
    preset = control.command(db, accounts_db, 1, True, {
        "action": "preset", "password": "admin-pass", "label": "Status",
        "script": "Status.ps1", "argv": ["-Verbose"], "timeout": 30,
    })["preset_id"]
    control.command(db, accounts_db, 1, True, {
        "action": "run", "password": "admin-pass", "preset_id": preset,
    })
    db.commit()
    job = control.claim(db, accounts_db)
    assert job["script"] == "Status.ps1"
    with mock.patch.object(control.subprocess, "run") as execute:
        execute.return_value.returncode = 0
        control.run_one(db, accounts_db, db.execute("PRAGMA database_list").fetchone()[2].rsplit("sql", 1)[0], job)
        argv = execute.call_args.args[0]
        assert argv[:4] == ["pwsh", "-NoProfile", "-NonInteractive", "-File"]
        assert argv[-1:] == ["-Verbose"]
        assert argv[4].endswith("Status.ps1")
        assert execute.call_args.kwargs["shell"] is False


def test_csrf_binds_to_account_cookie(setup):
    db, _, _ = setup
    value = control.csrf_issue(db, "session-a")
    assert control.csrf_check(db, "session-a", value)
    assert not control.csrf_check(db, "session-b", value)
    assert not control.csrf_check(db, "session-a", "invalid")
