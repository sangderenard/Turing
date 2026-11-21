"""Durable, operator-owned runs for the agent training arena.

The site records specifications and receipts. An operator-owned PowerShell
coordinator claims a run and launches agents under their existing Windows
accounts; the web process never starts an agent or executes model output.
"""
from __future__ import annotations

import json
import re
import secrets
import sqlite3
import time

from meowdoku import schema_version

MAX_BODY = 128 * 1024
MAX_RESULT = 4 * 1024 * 1024
MAX_TEAMS = 64
AXES = ("understandability", "actionability", "completeness", "experience", "code_elegance")
AGENTS = ("claude", "codex")
ID = re.compile(r"[0-9a-f]{32}\Z")


class ArenaError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def prepare(db):
    schema_version.declare(db, "agent_training_arena", 1)
    db.execute("CREATE TABLE IF NOT EXISTS agent_training_arena ("
               "id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, source_kind TEXT NOT NULL, source_id TEXT NOT NULL, "
               "specification TEXT NOT NULL, "
               "status TEXT NOT NULL, result TEXT, error TEXT, claim_token TEXT, "
               "created REAL NOT NULL, claimed REAL, finished REAL)")
    db.execute("CREATE INDEX IF NOT EXISTS agent_training_arena_recent "
               "ON agent_training_arena(owner_id,created DESC)")
    db.execute("CREATE INDEX IF NOT EXISTS agent_training_arena_source "
               "ON agent_training_arena(source_kind,source_id,status,created DESC)")


def _text(value, label, limit=16000):
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > limit:
        raise ArenaError(f"{label} must be nonempty and at most {limit} bytes")
    return value.strip()


def _agent(value):
    if value not in AGENTS:
        raise ArenaError("agent must be claude or codex")
    return value


def validate(spec):
    if not isinstance(spec, dict) or set(spec) != {"source", "goal", "target", "teams", "judge", "training_mode", "runtime"}:
        raise ArenaError("source, goal, target, teams, judge, training_mode and runtime are required")
    source = spec["source"]
    if not isinstance(source, dict) or set(source) != {"kind", "id"} or source["kind"] not in ("ticket", "research"):
        raise ArenaError("source must identify a ticket or research case")
    _text(source["id"], "source ID", 128)
    _text(spec["goal"], "goal")
    target = spec["target"]
    if not isinstance(target, dict) or set(target) != {"agent", "stateful"} or type(target["stateful"]) is not bool:
        raise ArenaError("target needs agent and stateful")
    _agent(target["agent"])
    _agent(spec["judge"])
    if type(spec["training_mode"]) is not bool:
        raise ArenaError("training_mode must be a boolean")
    runtime = spec["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {"turn_seconds", "model", "parameters"}:
        raise ArenaError("runtime needs turn_seconds, model and parameters")
    if type(runtime["turn_seconds"]) is not int or not 30 <= runtime["turn_seconds"] <= 1800:
        raise ArenaError("turn_seconds must be 30 through 1800")
    if not isinstance(runtime["model"], str) or len(runtime["model"]) > 160:
        raise ArenaError("model must be a short string")
    if not isinstance(runtime["parameters"], dict) or len(json.dumps(runtime["parameters"])) > 4096:
        raise ArenaError("parameters must be a small JSON object")
    teams = spec["teams"]
    if not isinstance(teams, list) or not 2 <= len(teams) <= MAX_TEAMS:
        raise ArenaError("provide 2 through 64 teams")
    names = set()
    for team in teams:
        if not isinstance(team, dict) or set(team) != {"name", "agents", "stateful"}:
            raise ArenaError("each team needs name, agents and stateful")
        name = _text(team["name"], "team name", 120)
        if name in names:
            raise ArenaError("team names must be unique")
        names.add(name)
        if type(team["stateful"]) is not bool or not isinstance(team["agents"], list) or len(team["agents"]) not in (1, 2):
            raise ArenaError("each team needs one or two agents and a stateful choice")
        for agent in team["agents"]:
            _agent(agent)
    encoded = json.dumps(spec, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_BODY:
        raise ArenaError("arena specification is too large")
    return encoded


def submit(db, owner_id, spec):
    prepare(db)
    encoded = validate(spec)
    if db.execute("SELECT count(*) FROM agent_training_arena WHERE status IN ('queued','running')").fetchone()[0] >= 32:
        raise ArenaError("arena queue is full", 429)
    run_id = secrets.token_hex(16)
    db.execute("INSERT INTO agent_training_arena(id,owner_id,source_kind,source_id,specification,status,created) "
               "VALUES(?,?,?,?,?,'queued',?)",
               (run_id, owner_id, spec["source"]["kind"], spec["source"]["id"], encoded, time.time()))
    return read(db, owner_id, run_id)


def listing(db, owner_id):
    prepare(db)
    return {"runs": [dict(row) for row in db.execute(
        "SELECT id,status,created,claimed,finished,error FROM agent_training_arena "
        "WHERE owner_id=? ORDER BY created DESC LIMIT 100", (owner_id,))]}


def read(db, owner_id, run_id):
    prepare(db)
    if not isinstance(run_id, str) or not ID.fullmatch(run_id):
        raise ArenaError("arena run not found", 404)
    row = db.execute("SELECT id,specification,status,result,error,created,claimed,finished "
                     "FROM agent_training_arena WHERE id=? AND owner_id=?", (run_id, owner_id)).fetchone()
    if row is None:
        raise ArenaError("arena run not found", 404)
    result = dict(row)
    result["specification"] = json.loads(result["specification"])
    result["result"] = json.loads(result["result"]) if result["result"] else None
    return result


def by_source(db, kind, source_id):
    prepare(db)
    if kind not in ("ticket", "research") or not isinstance(source_id, str) or len(source_id) > 128:
        raise ArenaError("invalid source")
    rows = db.execute("SELECT id,specification,result,created,finished FROM agent_training_arena "
                      "WHERE source_kind=? AND source_id=? AND status='done' ORDER BY created DESC LIMIT 10",
                      (kind, source_id)).fetchall()
    return {"runs": [{"id": row["id"], "specification": json.loads(row["specification"]),
                      "result": json.loads(row["result"]), "created": row["created"], "finished": row["finished"]}
                     for row in rows]}


def claim(db, owner_id):
    prepare(db)
    db.execute("BEGIN IMMEDIATE")
    try:
        row = db.execute("SELECT id FROM agent_training_arena WHERE owner_id=? AND "
                         "(status='queued' OR (status='running' AND claimed<?)) ORDER BY created LIMIT 1",
                         (owner_id, time.time() - 4 * 3600)).fetchone()
        if row is None:
            db.commit()
            return {"run": None}
        token = secrets.token_hex(16)
        db.execute("UPDATE agent_training_arena SET status='running',claim_token=?,claimed=? WHERE id=?",
                   (token, time.time(), row[0]))
        db.commit()
        return {"run": {**read(db, owner_id, row[0]), "claim_token": token}}
    except BaseException:
        db.rollback()
        raise


def heartbeat(db, owner_id, body):
    prepare(db)
    if not isinstance(body, dict) or set(body) != {"id", "claim_token"}:
        raise ArenaError("id and claim_token required")
    row = db.execute("SELECT claim_token FROM agent_training_arena WHERE id=? AND owner_id=? AND status='running'",
                     (body["id"], owner_id)).fetchone()
    if row is None or not secrets.compare_digest(row[0], str(body["claim_token"])):
        raise ArenaError("arena claim is no longer current", 409)
    db.execute("UPDATE agent_training_arena SET claimed=? WHERE id=?", (time.time(), body["id"]))
    return {"id": body["id"], "running": True}


def complete(db, owner_id, body):
    prepare(db)
    if not isinstance(body, dict) or set(body) != {"id", "claim_token", "result", "error"}:
        raise ArenaError("id, claim_token, result and error are required")
    result, error = body["result"], body["error"]
    if not isinstance(error, str) or len(error) > 1000:
        raise ArenaError("invalid error")
    if not error:
        if not isinstance(result, dict) or not isinstance(result.get("teams"), list):
            raise ArenaError("result needs team receipts")
        expected = db.execute("SELECT specification FROM agent_training_arena WHERE id=? AND owner_id=?",
                              (body["id"], owner_id)).fetchone()
        if expected is None:
            raise ArenaError("arena run not found", 404)
        spec = json.loads(expected[0])
        if ([team.get("name") for team in result["teams"] if isinstance(team, dict)] !=
                [team["name"] for team in spec["teams"]] or
                result.get("source") != spec["source"] or result.get("goal") != spec["goal"] or
                result.get("runtime") != spec["runtime"] or result.get("training_mode") != spec["training_mode"]):
            raise ArenaError("result does not match the claimed task")
        for team in result["teams"]:
            if not isinstance(team, dict) or set(team.get("scores", {})) != set(AXES):
                raise ArenaError("each team needs all five scoring axes")
            if any(type(team["scores"][axis]) is not int or not 0 <= team["scores"][axis] <= 4 for axis in AXES):
                raise ArenaError("scores must be integers from 0 through 4")
            if not team.get("error"):
                pieces = team.get("pieces")
                if not isinstance(pieces, dict) or set(pieces) != {"prompt", "message", "response"}:
                    raise ArenaError("each successful team needs three scored pieces")
                for piece in pieces.values():
                    scores = piece.get("scores") if isinstance(piece, dict) else None
                    if not isinstance(scores, dict) or set(scores) != set(AXES) or any(
                            type(scores[axis]) is not int or not 0 <= scores[axis] <= 4 for axis in AXES):
                        raise ArenaError("each piece needs five integer scores from 0 through 4")
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_RESULT:
        raise ArenaError("result is too large")
    db.execute("BEGIN IMMEDIATE")
    try:
        row = db.execute("SELECT claim_token FROM agent_training_arena WHERE id=? AND owner_id=? AND status='running'",
                         (body["id"], owner_id)).fetchone()
        if row is None or not secrets.compare_digest(row[0], str(body["claim_token"])):
            raise ArenaError("arena claim is no longer current", 409)
        db.execute("UPDATE agent_training_arena SET status=?,result=?,error=?,claim_token=NULL,finished=? WHERE id=?",
                   ("failed" if error else "done", encoded, error, time.time(), body["id"]))
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return read(db, owner_id, body["id"])
