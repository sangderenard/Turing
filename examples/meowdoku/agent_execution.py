"""Account-owned command jobs executed by the existing AgentDay owner task.

HTTP only queues work and returns its result. Start-AgentDay.ps1 claims a job and
uses AgentAccounts.psm1 to launch it as that agent's isolated Windows account.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meowdoku import schema_version

BUILTINS = frozenset({"claude", "codex", "gemini"})
MAX_SOURCE_BYTES = 16 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
MAX_ACTIVE_PER_ACCOUNT = 4
MAX_ACTIVE_TOTAL = 64
MAX_SECONDS = 120
STALE_SECONDS = MAX_SECONDS + 120


class ExecutionError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def server_owned(accounts_db, username: str) -> bool:
    """Only the installed AgentDay identities; Llama is deliberately absent."""
    name = (username or "").casefold()
    if name in BUILTINS:
        return True
    if not name or name == "llama":
        return False
    registry = Path(accounts_db).with_name("admin_control.db")
    if not registry.is_file():
        return False
    try:
        with sqlite3.connect(f"file:{registry.as_posix()}?mode=ro", uri=True, timeout=5) as db:
            return db.execute("SELECT 1 FROM external_agents WHERE name=? AND status='installed'", (name,)).fetchone() is not None
    except sqlite3.Error:
        return False


def allowed(accounts_db, username: str) -> bool:
    if not server_owned(accounts_db, username):
        return False
    with sqlite3.connect(f"file:{Path(accounts_db).as_posix()}?mode=ro", uri=True, timeout=5) as db:
        try:
            row = db.execute("SELECT COALESCE(r.role,'member'), a.enabled FROM users u "
                             "LEFT JOIN account_roles r ON u.id=r.user_id "
                             "LEFT JOIN account_agent_execution a ON u.id=a.user_id "
                             "WHERE u.username=? COLLATE NOCASE", (username,)).fetchone()
        except sqlite3.OperationalError as error:
            if "no such table: account_agent_execution" not in str(error):
                raise
            row = db.execute("SELECT COALESCE(r.role,'member'), NULL FROM users u "
                             "LEFT JOIN account_roles r ON u.id=r.user_id WHERE u.username=? COLLATE NOCASE",
                             (username,)).fetchone()
    return row is not None and row[0] != "intern" and (row[1] is None or bool(row[1]))


def connect(accounts_db):
    db = sqlite3.connect(str(accounts_db), timeout=10, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=10000")
    prepare(db)
    return db


def prepare(db):
    schema_version.declare(db, "agent_execution", 1)
    db.execute("CREATE TABLE IF NOT EXISTS agent_execution_jobs ("
               "id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, username TEXT NOT NULL, kind TEXT NOT NULL, "
               "source TEXT NOT NULL, status TEXT NOT NULL, output TEXT NOT NULL DEFAULT '', exit_code INTEGER, "
               "timed_out INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL, started REAL, finished REAL)")
    db.execute("CREATE INDEX IF NOT EXISTS agent_execution_queue ON agent_execution_jobs(username,status,created)")


def submit(db, user_id: int, username: str, kind: str, source: str):
    if kind not in ("exec", "python"):
        raise ExecutionError("kind must be exec or python")
    if not isinstance(source, str) or not source.strip() or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ExecutionError("source must be nonempty UTF-8 text of at most 16 KiB")
    db.execute("BEGIN IMMEDIATE")
    try:
        active = "status IN ('queued','running')"
        if db.execute(f"SELECT COUNT(*) FROM agent_execution_jobs WHERE {active}").fetchone()[0] >= MAX_ACTIVE_TOTAL:
            raise ExecutionError("agent execution queue is full", 429)
        if db.execute(f"SELECT COUNT(*) FROM agent_execution_jobs WHERE user_id=? AND {active}", (user_id,)).fetchone()[0] >= MAX_ACTIVE_PER_ACCOUNT:
            raise ExecutionError("this agent already has four active jobs", 429)
        job_id = secrets.token_hex(16)
        db.execute("INSERT INTO agent_execution_jobs(id,user_id,username,kind,source,status,created) "
                   "VALUES(?,?,?,?,?,'queued',?)", (job_id, user_id, username.casefold(), kind, source, time.time()))
        db.commit()
        return {"id": job_id, "status": "queued"}
    except BaseException:
        db.rollback()
        raise


def read(db, user_id: int, job_id: str):
    row = db.execute("SELECT id,kind,status,output,exit_code,timed_out,created,started,finished "
                     "FROM agent_execution_jobs WHERE id=? AND user_id=?", (job_id, user_id)).fetchone()
    if row is None:
        raise ExecutionError("execution job not found", 404)
    return dict(row)


def claim(db, username: str):
    db.execute("BEGIN IMMEDIATE")
    try:
        now = time.time()
        db.execute("UPDATE agent_execution_jobs SET status='failed',output='worker ended before completion',finished=? "
                   "WHERE username=? AND status='running' AND started<?", (now, username.casefold(), now - STALE_SECONDS))
        running = db.execute("SELECT 1 FROM agent_execution_jobs WHERE username=? AND status='running' LIMIT 1",
                             (username.casefold(),)).fetchone()
        row = None if running else db.execute(
            "SELECT id,kind,source FROM agent_execution_jobs WHERE username=? AND status='queued' "
            "ORDER BY created LIMIT 1", (username.casefold(),)).fetchone()
        if row is not None:
            db.execute("UPDATE agent_execution_jobs SET status='running',started=? WHERE id=?", (now, row["id"]))
        db.commit()
        return dict(row) if row is not None else None
    except BaseException:
        db.rollback()
        raise


def complete(db, job_id: str, result: dict):
    output = str(result.get("output") or "")
    raw = output.encode("utf-8")
    if len(raw) > MAX_OUTPUT_BYTES:
        output = raw[:MAX_OUTPUT_BYTES].decode("utf-8", "ignore") + "\n[output truncated at 64 KiB]"
    exit_code = result.get("exit_code")
    if exit_code is not None and type(exit_code) is not int:
        raise ExecutionError("invalid exit code")
    timed_out = bool(result.get("timed_out"))
    db.execute("BEGIN IMMEDIATE")
    try:
        changed = db.execute("UPDATE agent_execution_jobs SET status=?,output=?,exit_code=?,timed_out=?,finished=? "
                             "WHERE id=? AND status='running'", ("failed" if timed_out or exit_code != 0 else "complete",
                                                           output, exit_code, int(timed_out), time.time(), job_id)).rowcount
        if not changed:
            raise ExecutionError("execution job is not running", 409)
        db.commit()
    except BaseException:
        db.rollback()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("claim").add_argument("--agent", required=True)
    sub.add_parser("complete").add_argument("--id", required=True)
    sub.add_parser("allowed").add_argument("--agent", required=True)
    args = parser.parse_args(argv)
    if args.action == "allowed":
        print(json.dumps({"allowed": allowed(args.db, args.agent)}))
        return
    if args.action == "claim" and not allowed(args.db, args.agent):
        print("null")
        return
    with connect(args.db) as db:
        if args.action == "claim":
            value = claim(db, args.agent)
        else:
            value = complete(db, args.id, json.load(sys.stdin))
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    main()
