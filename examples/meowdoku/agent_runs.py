"""Operator-submitted Llama profile runs, claimed in batches by the Llama account.

The HTTP service stores work and exact profile snapshots. The isolated hosted
runner owns inference; no model text is interpreted as a site command.
"""
from __future__ import annotations

import secrets
import sqlite3
import time

from meowdoku import agent_workspace as workspace, schema_version

MAX_PROMPT_BYTES = 16 * 1024
MAX_RESPONSE_BYTES = 16 * 1024
MAX_QUEUED = 256
LEASE_SECONDS = 15 * 60


def prepare(db):
    schema_version.declare(db, "agent_runs", 1)
    db.execute("CREATE TABLE IF NOT EXISTS agent_runs ("
               "id TEXT PRIMARY KEY, instance_id TEXT NOT NULL, profile_id TEXT NOT NULL, individual_id INTEGER NOT NULL, "
               "job_key TEXT, memory_id TEXT NOT NULL, prompt TEXT NOT NULL, preamble TEXT NOT NULL, "
               "status TEXT NOT NULL, response TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '', "
               "claim_token TEXT, created REAL NOT NULL, claimed REAL, finished REAL)")
    db.execute("CREATE INDEX IF NOT EXISTS agent_runs_queue ON agent_runs(individual_id,status,created)")


def submit(db, data):
    prepare(db)
    active = workspace.instance(db, data.get("instance_id"))
    profile_id = active["profile_id"]
    if active["individual"].casefold() != "llama":
        raise workspace.WorkspaceError("this batch runner accepts Llama profiles only")
    job_key = data.get("job_key") or None
    if job_key is not None and (not isinstance(job_key, str) or len(job_key) > 80):
        raise workspace.WorkspaceError("invalid job key")
    resolved = workspace.resolve_profile(db, profile_id, job_key)
    prompt = workspace._text(data.get("prompt"), "run prompt", MAX_PROMPT_BYTES).strip()
    if not prompt:
        raise workspace.WorkspaceError("run prompt is required")
    memory_id = active["memory_id"]
    workspace.memory_read(db, active["individual_id"], memory_id)
    db.execute("BEGIN IMMEDIATE")
    try:
        if db.execute("SELECT COUNT(*) FROM agent_runs WHERE status IN ('queued','running')").fetchone()[0] >= MAX_QUEUED:
            raise workspace.WorkspaceError("Llama run queue is full", 429)
        run_id = secrets.token_hex(16)
        db.execute("INSERT INTO agent_runs(id,instance_id,profile_id,individual_id,job_key,memory_id,prompt,preamble,status,created) "
                   "VALUES(?,?,?,?,?,?,?,?,'queued',?)",
                   (run_id, active["id"], profile_id, active["individual_id"], job_key, memory_id, prompt,
                    resolved["preamble"], time.time()))
        db.commit()
    except BaseException:
        db.rollback(); raise
    return read(db, run_id)


def list_runs(db):
    prepare(db)
    return {"runs": [dict(row) for row in db.execute(
        "SELECT id,instance_id,profile_id,job_key,memory_id,status,created,claimed,finished "
        "FROM agent_runs ORDER BY created DESC LIMIT 100")]}


def read(db, run_id):
    prepare(db)
    row = db.execute("SELECT id,instance_id,profile_id,job_key,memory_id,prompt,status,response,error,created,claimed,finished "
                     "FROM agent_runs WHERE id=?", (run_id,)).fetchone()
    if row is None:
        raise workspace.WorkspaceError("Llama run not found", 404)
    return dict(row)


def claim(db, individual_id, limit=16):
    prepare(db)
    if not isinstance(limit, int) or not 1 <= limit <= 32:
        raise workspace.WorkspaceError("batch limit must be 1 through 32")
    db.execute("BEGIN IMMEDIATE")
    try:
        now = time.time()
        rows = db.execute("SELECT id,instance_id,profile_id,job_key,memory_id,prompt,preamble FROM agent_runs "
                          "WHERE individual_id=? AND (status='queued' OR (status='running' AND claimed<?)) "
                          "ORDER BY created LIMIT ?", (individual_id, now - LEASE_SECONDS, limit)).fetchall()
        result = []
        for row in rows:
            token = secrets.token_hex(16)
            db.execute("UPDATE agent_runs SET status='running',claim_token=?,claimed=? WHERE id=?",
                       (token, now, row["id"]))
            try:
                memory = workspace.memory_read(db, individual_id, row["memory_id"])["body"]
            except workspace.WorkspaceError:
                db.execute("UPDATE agent_runs SET status='failed',error='memory file missing',"
                           "claim_token=NULL,finished=? WHERE id=?", (now, row["id"]))
                continue
            result.append({**dict(row), "claim_token": token, "memory": memory})
        db.commit()
    except BaseException:
        db.rollback(); raise
    return {"runs": result}


def complete(db, individual_id, data):
    prepare(db)
    response = workspace._text(data.get("response", ""), "run response", MAX_RESPONSE_BYTES)
    error = workspace._text(data.get("error", ""), "run error", 1000)
    if not response and not error:
        raise workspace.WorkspaceError("response or error is required")
    db.execute("BEGIN IMMEDIATE")
    try:
        row = db.execute("SELECT claim_token FROM agent_runs WHERE id=? AND individual_id=? AND status='running'",
                         (data.get("id"), individual_id)).fetchone()
        if row is None or not secrets.compare_digest(row[0], str(data.get("claim_token") or "")):
            raise workspace.WorkspaceError("run claim is no longer current", 409)
        db.execute("UPDATE agent_runs SET status=?,response=?,error=?,claim_token=NULL,finished=? WHERE id=?",
                   ("failed" if error else "done", response, error, time.time(), data["id"]))
        db.commit()
    except BaseException:
        db.rollback(); raise
    return read(db, data["id"])
