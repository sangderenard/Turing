"""Owner-run deploy script broker. Web requests queue named presets; the boot worker executes them.

No HTTP process runs PowerShell. A preset fixes a script name and an argv array; an
operator can only request a preset named in an unexpired admin grant. The worker
rechecks authority and expiry when it claims the job. Its output stays in an
owner-only local log, never in the API response.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import subprocess
import sys
import time
from http.cookies import SimpleCookie

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
DEPLOY = Path(__file__).resolve().parent / "deploy"
SCRIPTS = (
    "Add-CanvasContractPort.ps1", "Apache-Service.ps1", "Backup-State.ps1",
    "Close-Console.ps1", "Enable-CronTool.ps1", "Export-DevDatabase.ps1", "Migrate-LiaisonVault.ps1",
    "Migrate-State.ps1", "Open-Console.ps1", "Protect-DefaultBranch.ps1",
    "Provision-ServicePython.ps1",
    "Restart-Stack.ps1", "Restore-AgentBoxAcls.ps1", "Restore-State.ps1",
    "Rollback-State.ps1", "Run-HostedLlm.ps1", "Set-AgentBox.ps1",
    "Set-LiaisonCutover.ps1", "Set-ServiceFirewall.ps1", "Set-SiteCondition.ps1",
    "Setup-LiaisonService.ps1", "Setup-ServiceAccount.ps1", "Start-AgentDay.ps1",
    "Start-Apache.ps1", "Start-HostedLlm.ps1", "Start-Meowdoku.ps1",
    "Start-RoundTable.ps1", "Status.ps1", "Stop-Apache.ps1",
    "Stop-Meowdoku.ps1", "Update-LiveCheckouts.ps1",
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS control_csrf (
    session_hash TEXT PRIMARY KEY, token_hash TEXT NOT NULL, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS control_presets (
    id INTEGER PRIMARY KEY, label TEXT NOT NULL, script TEXT NOT NULL,
    argv TEXT NOT NULL, timeout INTEGER NOT NULL, created_by INTEGER NOT NULL,
    created REAL NOT NULL, disabled REAL);
CREATE TABLE IF NOT EXISTS control_grants (
    id INTEGER PRIMARY KEY, preset_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
    issued_by INTEGER NOT NULL, created REAL NOT NULL, expires REAL NOT NULL, revoked REAL);
CREATE TABLE IF NOT EXISTS control_jobs (
    id INTEGER PRIMARY KEY, preset_id INTEGER NOT NULL, requested_by INTEGER NOT NULL,
    requested REAL NOT NULL, status TEXT NOT NULL, started REAL, finished REAL,
    exit_code INTEGER, note TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS control_audit (
    id INTEGER PRIMARY KEY, at REAL NOT NULL, actor INTEGER, action TEXT NOT NULL,
    target TEXT NOT NULL, allowed INTEGER NOT NULL, note TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS control_jobs_queue ON control_jobs(status, requested);
CREATE INDEX IF NOT EXISTS control_grants_target ON control_grants(user_id, preset_id, expires);
"""
MAX_ARGS = 24
MAX_ARG_LENGTH = 500
MAX_TIMEOUT = 4 * 3600
MAX_GRANT = 30 * 86400
ACCOUNT_COOKIE = "meowdoku_account"


class ControlError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def connect(state_dir):
    path = Path(state_dir) / "sql" / "admin_control.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=10000")
    db.executescript(SCHEMA)
    return db


def audit(db, actor, action, target, allowed, note=""):
    db.execute("INSERT INTO control_audit(at,actor,action,target,allowed,note) VALUES(?,?,?,?,?,?)",
               (time.time(), actor, action, str(target)[:200], int(allowed), str(note)[:300]))


def account_token(handler):
    jar = SimpleCookie(handler.headers.get("Cookie", ""))
    item = jar.get(ACCOUNT_COOKIE)
    return item.value if item is not None else None


def csrf_issue(db, token):
    value = secrets.token_urlsafe(32)
    db.execute("INSERT OR REPLACE INTO control_csrf VALUES(?,?,?)",
               (hashlib.sha256(token.encode()).hexdigest(),
                hashlib.sha256(value.encode()).hexdigest(), time.time() + 900))
    return value


def csrf_check(db, token, value):
    if not isinstance(value, str) or len(value) > 256:
        return False
    key = hashlib.sha256(token.encode()).hexdigest()
    row = db.execute("SELECT token_hash,expires FROM control_csrf WHERE session_hash=?", (key,)).fetchone()
    return bool(row and row["expires"] > time.time() and
                hmac.compare_digest(row["token_hash"], hashlib.sha256(value.encode()).hexdigest()))


def verify_password(accounts_db, user_id, password):
    from meowdoku import accounts
    if not isinstance(password, str) or not 1 <= len(password) <= 1024:
        return False
    db = sqlite3.connect(accounts_db)
    try:
        row = db.execute("SELECT password_hash FROM users WHERE id=?", (user_id,)).fetchone()
    finally:
        db.close()
    return bool(row and accounts._verify_password(password, row[0]))


def _validate_argv(value):
    if not isinstance(value, list) or len(value) > MAX_ARGS or any(
        not isinstance(arg, str) or not arg or len(arg) > MAX_ARG_LENGTH or
        any(ord(ch) < 32 for ch in arg) for arg in value
    ):
        raise ControlError("argv must be a short array of printable strings")
    # These credentials are supplied by the worker. They must not be stored in a preset.
    if any(re.match(r"^-(?:Password|PasswordFile|AccountsDb|StateDir|User)(?:$|[:=])", arg, re.I) for arg in value):
        raise ControlError("credential and state arguments are controlled by the worker")
    return value


def _account_role(accounts_db, user_id):
    from meowdoku import authz
    db = sqlite3.connect(accounts_db)
    try:
        row = db.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
    finally:
        db.close()
    if not row:
        return None
    principal = authz.principal_from({"id": user_id, "username": row[0]}, "dns", "shader", accounts_db)
    return "admin" if principal.role == "admin" else (
        "operator" if "deploy_scripts" in authz.capabilities(principal, authz.Policy.beside(accounts_db).get()) else None)


def _granted(db, user_id, preset_id, now):
    return db.execute("SELECT 1 FROM control_grants WHERE user_id=? AND preset_id=? "
                      "AND revoked IS NULL AND expires>? ORDER BY expires DESC LIMIT 1",
                      (user_id, preset_id, now)).fetchone() is not None


def list_state(db, user_id, admin):
    presets = []
    for row in db.execute("SELECT id,label,script,argv,timeout FROM control_presets WHERE disabled IS NULL ORDER BY label"):
        if admin or _granted(db, user_id, row["id"], time.time()):
            item = {"id": row["id"], "label": row["label"], "script": row["script"],
                    "timeout": row["timeout"]}
            if admin:
                item["argv"] = json.loads(row["argv"])
            presets.append(item)
    grants = []
    # The accounts table lives in another database; expose IDs without copying it here.
    for row in db.execute("SELECT id,preset_id,user_id,created,expires,revoked FROM control_grants "
                          + ("" if admin else "WHERE user_id=? ") + "ORDER BY id DESC LIMIT 100",
                          () if admin else (user_id,)):
        grants.append(dict(row))
    jobs = [dict(row) for row in db.execute(
        "SELECT id,preset_id,requested_by,requested,status,started,finished,exit_code,note "
        "FROM control_jobs " + ("" if admin else "WHERE requested_by=? ") + "ORDER BY id DESC LIMIT 50",
        () if admin else (user_id,))]
    # [EXTERNAL AGENTS] declarations only (external_agents.py); the install is the owner's Install-ExternalAgent.ps1 -Name <name>.
    from meowdoku import external_agents
    return {"scripts": SCRIPTS if admin else (), "presets": presets, "grants": grants, "jobs": jobs,
            "agents": external_agents.agents(external_agents.ensure(db)) if admin else [],
            "providers": external_agents.PROVIDERS if admin else (),
            "agent_tools": external_agents.DAY_TOOLS if admin else ()}


def command(db, accounts_db, user_id, admin, body):
    if not isinstance(body, dict) or not isinstance(body.get("action"), str):
        raise ControlError("expected an action")
    action = body["action"]
    if not verify_password(accounts_db, user_id, body.get("password")):
        audit(db, user_id, action, "", False, "password verification failed")
        raise ControlError("password verification failed", 403)
    now = time.time()
    if action == "preset":
        if not admin:
            raise ControlError("admin account required", 403)
        script = body.get("script")
        label = body.get("label")
        if script not in SCRIPTS or not isinstance(label, str) or not 1 <= len(label) <= 100:
            raise ControlError("unknown script or invalid label")
        argv = _validate_argv(body.get("argv"))
        timeout = body.get("timeout", 600)
        if type(timeout) is not int or not 1 <= timeout <= MAX_TIMEOUT:
            raise ControlError("timeout must be 1 through 14400 seconds")
        cursor = db.execute("INSERT INTO control_presets(label,script,argv,timeout,created_by,created) VALUES(?,?,?,?,?,?)",
                            (label, script, json.dumps(argv), timeout, user_id, now))
        audit(db, user_id, action, cursor.lastrowid, True)
        return {"preset_id": cursor.lastrowid}
    if action == "disable":
        if not admin:
            raise ControlError("admin account required", 403)
        preset = int(body.get("preset_id", 0))
        db.execute("UPDATE control_presets SET disabled=? WHERE id=? AND disabled IS NULL", (now, preset))
        db.execute("UPDATE control_grants SET revoked=? WHERE preset_id=? AND revoked IS NULL", (now, preset))
        audit(db, user_id, action, preset, True)
        return {"disabled": preset}
    if action == "grant":
        if not admin:
            raise ControlError("admin account required", 403)
        preset = int(body.get("preset_id", 0))
        hours = body.get("hours")
        if type(hours) not in (int, float) or not 1 / 60 <= hours <= MAX_GRANT / 3600:
            raise ControlError("grant duration must be 1 minute through 30 days")
        if not db.execute("SELECT 1 FROM control_presets WHERE id=? AND disabled IS NULL", (preset,)).fetchone():
            raise ControlError("preset not found", 404)
        username = body.get("username")
        if not isinstance(username, str):
            raise ControlError("operator username required")
        users = sqlite3.connect(accounts_db)
        try:
            row = users.execute("SELECT id FROM users WHERE username=? COLLATE NOCASE", (username,)).fetchone()
        finally:
            users.close()
        if not row or _account_role(accounts_db, row[0]) != "operator":
            raise ControlError("operator account required", 403)
        cursor = db.execute("INSERT INTO control_grants(preset_id,user_id,issued_by,created,expires) VALUES(?,?,?,?,?)",
                            (preset, row[0], user_id, now, now + hours * 3600))
        audit(db, user_id, action, cursor.lastrowid, True)
        return {"grant_id": cursor.lastrowid, "expires": now + hours * 3600}
    if action == "revoke":
        if not admin:
            raise ControlError("admin account required", 403)
        grant = int(body.get("grant_id", 0))
        db.execute("UPDATE control_grants SET revoked=? WHERE id=? AND revoked IS NULL", (now, grant))
        audit(db, user_id, action, grant, True)
        return {"revoked": grant}
    if action == "run":
        preset = int(body.get("preset_id", 0))
        if db.execute("SELECT COUNT(*) FROM control_jobs WHERE status='queued'").fetchone()[0] >= 20:
            raise ControlError("run queue is full", 429)
        if db.execute("SELECT 1 FROM control_jobs WHERE status='queued' AND preset_id=? AND requested_by=?", (preset, user_id)).fetchone():
            raise ControlError("this action is already queued", 409)
        if not db.execute("SELECT 1 FROM control_presets WHERE id=? AND disabled IS NULL", (preset,)).fetchone():
            raise ControlError("preset not found", 404)
        if not admin and not _granted(db, user_id, preset, now):
            audit(db, user_id, action, preset, False, "no active grant")
            raise ControlError("no active grant for this preset", 403)
        cursor = db.execute("INSERT INTO control_jobs(preset_id,requested_by,requested,status) VALUES(?,?,?,'queued')",
                            (preset, user_id, now))
        audit(db, user_id, action, cursor.lastrowid, True)
        return {"job_id": cursor.lastrowid, "status": "queued"}
    if action in ("agent_declare", "agent_withdraw"):
        # Saves what is not secret. Nothing on the machine changes here: the owner runs Install-ExternalAgent.ps1 -Name <name>
        # elevated, over SSH, and that script alone makes the accounts and asks for the API key (case 989e507e).
        from meowdoku import external_agents
        if not admin:
            raise ControlError("admin account required", 403)
        try:
            if action == "agent_declare":
                agent = external_agents.declare(db, user_id, body.get("agent") or {}, accounts_db, now)
            else:
                agent = external_agents.mark(external_agents.ensure(db), str(body.get("name", "")), "withdrawn", "withdrawn on the page", now)
        except external_agents.ExternalAgentError as error:
            raise ControlError(str(error), error.status) from None
        audit(db, user_id, action, agent["name"], True)
        return {"agent": agent}
    raise ControlError("unknown action")


def route(handler, sessions, method):
    """Called only from the site's /api/operator-control route."""
    if not handler._is_secure() or handler.headers.get("Authorization"):
        return handler._json({"error": "secure browser session required"}, 403)
    if not handler._require("deploy_scripts"):
        return
    account = handler._account()
    token = account_token(handler)
    if not account or not token or account.get("temporary"):
        return handler._json({"error": "permanent account required"}, 403)
    user_id = account["id"]
    admin = _account_role(sessions.accounts_db, user_id) == "admin"
    db = connect(sessions.state.root)
    try:
        if method == "GET":
            value = list_state(db, user_id, admin)
            value["csrf"] = csrf_issue(db, token)
            value["admin"] = admin
            db.commit()
            return handler._json(value)
        expected = ("https://" if handler._is_secure() else "http://") + handler.headers.get("Host", "")
        if handler.headers.get("Origin") != expected or handler.headers.get("Sec-Fetch-Site") == "cross-site":
            return handler._json({"error": "same-origin request required"}, 403)
        if not csrf_check(db, token, handler.headers.get("X-Control-CSRF")):
            return handler._json({"error": "CSRF token expired; reload the page"}, 403)
        body = handler._body_json(limit=16384)
        if body is None:
            return
        try:
            db.execute("BEGIN IMMEDIATE")
            result = command(db, sessions.accounts_db, user_id, admin, body)
            db.commit()
            return handler._json(result, 202 if body["action"] == "run" else 200)
        except (ControlError, ValueError, TypeError) as error:
            db.rollback()
            audit(db, user_id, body.get("action", "invalid") if isinstance(body, dict) else "invalid", "", False, str(error))
            db.commit()
            return handler._json({"error": str(error)}, getattr(error, "status", 400))
    finally:
        db.close()


def claim(db, accounts_db):
    """One job at a time. A worker restart never replays a job marked running."""
    db.execute("BEGIN IMMEDIATE")
    if db.execute("SELECT 1 FROM control_jobs WHERE status='running' LIMIT 1").fetchone():
        db.commit()
        return None
    row = db.execute("SELECT j.id,j.preset_id,j.requested_by,p.script,p.argv,p.timeout "
                     "FROM control_jobs j JOIN control_presets p ON p.id=j.preset_id "
                     "WHERE j.status='queued' ORDER BY j.id LIMIT 1").fetchone()
    if not row:
        db.commit()
        return None
    now = time.time()
    role = _account_role(accounts_db, row["requested_by"])
    allowed = role == "admin" or (role == "operator" and _granted(db, row["requested_by"], row["preset_id"], now))
    preset_enabled = db.execute("SELECT disabled FROM control_presets WHERE id=?", (row["preset_id"],)).fetchone()
    if not allowed or preset_enabled is None or preset_enabled[0] is not None or row["script"] not in SCRIPTS or not (DEPLOY / row["script"]).is_file():
        db.execute("UPDATE control_jobs SET status='refused',finished=?,note=? WHERE id=?",
                   (now, "authority, grant or script unavailable", row["id"]))
        audit(db, row["requested_by"], "worker-refuse", row["id"], False)
        db.commit()
        return None
    db.execute("UPDATE control_jobs SET status='running',started=? WHERE id=?", (now, row["id"]))
    db.commit()
    return dict(row)


def run_one(db, accounts_db, state_dir, job):
    """Run the exact saved argv, without a shell or web-supplied script path."""
    script = DEPLOY / job["script"]
    log_dir = Path(state_dir) / "secrets" / "admin-control-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{job["id"]}.log'
    env = os.environ.copy()
    env["MEOWDOKU_OPERATOR"] = "admin"
    env["MEOWDOKU_STATE_DIR"] = str(state_dir)
    argv = ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(script), *json.loads(job["argv"])]
    result = None
    note = ""
    try:
        with open(log_path, "xb") as stream:
            result = subprocess.run(argv, cwd=DEPLOY, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=job["timeout"], check=False, shell=False).returncode
    except subprocess.TimeoutExpired:
        note = "timed out; inspect local process state before retrying"
    except (OSError, ValueError) as error:
        note = type(error).__name__
    status = "complete" if result == 0 else "failed"
    db.execute("UPDATE control_jobs SET status=?,finished=?,exit_code=?,note=? WHERE id=?",
               (status, time.time(), result, note, job["id"]))
    audit(db, job["requested_by"], "worker-finish", job["id"], result == 0, note)
    db.commit()


def worker(state_dir, accounts_db, poll=1.0):
    db = connect(state_dir)
    try:
        while True:
            job = claim(db, accounts_db)
            if job:
                run_one(db, accounts_db, state_dir, job)
            else:
                time.sleep(poll)
    finally:
        db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--state-dir", required=True)
    args = parser.parse_args(argv)
    if not args.worker:
        parser.error("--worker is required")
    state = Path(args.state_dir).resolve()
    worker(state, state / "sql" / "accounts.db")


if __name__ == "__main__":
    main()
