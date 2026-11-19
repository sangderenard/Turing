"""Operator controls for hosted agent conversation behavior.

A row overrides the server-wide value for one room or direct conversation. An absent
row inherits; an explicit "off" can disable a room even when global is "on".
This module owns only mode settings. Inference and scheduling run under the agent account.
"""
from __future__ import annotations

import re
import sqlite3
import time
from urllib.parse import parse_qs

from meowdoku import schema_version

ACCOUNT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
SCOPE = ("global", "room", "conversation")
MODE = ("on", "off", "inherit")
DDL = """CREATE TABLE IF NOT EXISTS hosted_agent_last_word (
    agent_id INTEGER NOT NULL REFERENCES users(id),
    scope TEXT NOT NULL CHECK(scope IN ('global','room','conversation')),
    scope_key TEXT NOT NULL,
    mode TEXT NOT NULL CHECK(mode IN ('on','off')),
    actor_id INTEGER NOT NULL REFERENCES users(id),
    updated_at REAL NOT NULL,
    PRIMARY KEY(agent_id,scope,scope_key)
)"""
AUDIT_DDL = """CREATE TABLE IF NOT EXISTS hosted_agent_mode_audit (
    id INTEGER PRIMARY KEY, agent_id INTEGER NOT NULL, actor_id INTEGER NOT NULL,
    scope TEXT NOT NULL, scope_key TEXT NOT NULL, mode TEXT NOT NULL,
    created_at REAL NOT NULL
)"""
CONTROL_DDL = """CREATE TABLE IF NOT EXISTS hosted_agent_controls (
    agent_id INTEGER PRIMARY KEY REFERENCES users(id),
    enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
    goal TEXT NOT NULL,
    actor_id INTEGER NOT NULL REFERENCES users(id),
    updated_at REAL NOT NULL
)"""
CONTROL_AUDIT_DDL = """CREATE TABLE IF NOT EXISTS hosted_agent_control_audit (
    id INTEGER PRIMARY KEY, agent_id INTEGER NOT NULL, actor_id INTEGER NOT NULL,
    enabled INTEGER NOT NULL, goal TEXT NOT NULL, created_at REAL NOT NULL
)"""


class ModeError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def prepare(db):
    schema_version.declare(db, "hosted_agent_modes", 2)
    db.execute(DDL)
    db.execute(AUDIT_DDL)
    db.execute(CONTROL_DDL)
    db.execute(CONTROL_AUDIT_DDL)


def controls(db, agent_id):
    prepare(db)
    row = db.execute("SELECT enabled,goal FROM hosted_agent_controls WHERE agent_id=?", (agent_id,)).fetchone()
    return {"enabled": bool(row[0]) if row else True, "goal": row[1] if row else ""}


def set_controls(db, *, agent_id, actor_id, actor_role, enabled, goal):
    if actor_role not in ("operator", "admin"):
        raise ModeError("operator or admin required", 403)
    if not isinstance(enabled, bool) or not isinstance(goal, str) or len(goal) > 4000:
        raise ModeError("enabled must be boolean and goal must be at most 4000 characters")
    goal = goal.strip()
    db.execute("INSERT INTO hosted_agent_controls(agent_id,enabled,goal,actor_id,updated_at) VALUES(?,?,?,?,?) "
               "ON CONFLICT(agent_id) DO UPDATE SET enabled=excluded.enabled,goal=excluded.goal,"
               "actor_id=excluded.actor_id,updated_at=excluded.updated_at",
               (agent_id, int(enabled), goal, actor_id, time.time()))
    db.execute("INSERT INTO hosted_agent_control_audit(agent_id,actor_id,enabled,goal,created_at) VALUES(?,?,?,?,?)",
               (agent_id, actor_id, int(enabled), goal, time.time()))
    return controls(db, agent_id)


def target(db, username):
    if not isinstance(username, str) or not ACCOUNT.fullmatch(username):
        raise ModeError("invalid agent account")
    row = db.execute("SELECT id, username FROM users WHERE username=? COLLATE NOCASE", (username,)).fetchone()
    if row is None:
        raise ModeError("agent account not found", 404)
    return row[0], row[1]


def validated_scope(db, scope, key, agent_id):
    if scope not in SCOPE or not isinstance(key, str) or len(key) > 64:
        raise ModeError("scope is global, room or conversation, with a bounded key")
    if scope == "global":
        if key:
            raise ModeError("global scope takes no key")
    elif not key or not ACCOUNT.fullmatch(key):
        raise ModeError("room or conversation needs a valid key")
    if scope == "room":
        from meowdoku import chat_rooms
        chat_rooms.ensure(db)
        if key not in chat_rooms.ROOMS and db.execute(
                "SELECT 1 FROM chat_room_defs WHERE room=?", (key,)).fetchone() is None:
            raise ModeError("room not found", 404)
    elif scope == "conversation":
        row = db.execute("SELECT id FROM users WHERE username=? COLLATE NOCASE", (key,)).fetchone()
        if row is None or row[0] == agent_id:
            raise ModeError("conversation peer not found", 404)
        key = db.execute("SELECT username FROM users WHERE id=?", (row[0],)).fetchone()[0]
    return key


def setting(db, agent_id, scope, key):
    prepare(db)
    key = validated_scope(db, scope, key, agent_id)
    global_row = db.execute("SELECT mode FROM hosted_agent_last_word WHERE agent_id=? AND scope='global' AND scope_key=''",
                            (agent_id,)).fetchone()
    row = db.execute("SELECT mode FROM hosted_agent_last_word WHERE agent_id=? AND scope=? AND scope_key=?",
                     (agent_id, scope, key)).fetchone()
    global_mode = global_row[0] if global_row else "off"
    return {"scope": scope, "key": key, "mode": row[0] if row else "inherit",
            "global_mode": global_mode,
            "effective": (row[0] if row else global_mode) == "on"}


def set_mode(db, *, agent_id, actor_id, actor_role, scope, key, mode):
    if actor_role not in ("operator", "admin"):
        raise ModeError("operator or admin required", 403)
    if mode not in MODE:
        raise ModeError("mode is on, off or inherit")
    prepare(db)
    key = validated_scope(db, scope, key, agent_id)
    if scope == "global" and mode == "inherit":
        raise ModeError("global mode is on or off")
    if mode == "inherit":
        db.execute("DELETE FROM hosted_agent_last_word WHERE agent_id=? AND scope=? AND scope_key=?",
                   (agent_id, scope, key))
    else:
        db.execute("INSERT INTO hosted_agent_last_word(agent_id,scope,scope_key,mode,actor_id,updated_at) "
                   "VALUES(?,?,?,?,?,?) ON CONFLICT(agent_id,scope,scope_key) DO UPDATE SET "
                   "mode=excluded.mode,actor_id=excluded.actor_id,updated_at=excluded.updated_at",
                   (agent_id, scope, key, mode, actor_id, time.time()))
    db.execute("INSERT INTO hosted_agent_mode_audit(agent_id,actor_id,scope,scope_key,mode,created_at) "
               "VALUES(?,?,?,?,?,?)", (agent_id, actor_id, scope, key, mode, time.time()))
    return setting(db, agent_id, scope, key)


def http_route(handler, path, method, *, accounts_db):
    principal = handler._principal()
    if principal.user_id is None:
        return handler._json({"error": "sign in first"}, 401)
    is_controls = path.endswith("/controls")
    username = path[len("/api/hosted-agents/"):-len("/controls" if is_controls else "/last-word")]
    db = sqlite3.connect(str(accounts_db), timeout=10, isolation_level=None)
    try:
        agent_id, canonical = target(db, username)
        if is_controls:
            if method == "GET":
                if principal.user_id != agent_id and principal.role not in ("operator", "admin"):
                    raise ModeError("only this agent or an operator may read its controls", 403)
                return handler._json({"agent": canonical, **controls(db, agent_id)})
            if principal.role not in ("operator", "admin"):
                raise ModeError("operator or admin required", 403)
            expected_origin = ("https://" if handler._is_secure() else "http://") + handler.headers.get("Host", "")
            if handler.headers.get("Origin") not in (None, expected_origin) or handler.headers.get("Sec-Fetch-Site") == "cross-site":
                raise ModeError("cross-origin control writes are not allowed", 403)
            body = handler._body_json()
            if body is None:
                return
            if not isinstance(body, dict) or set(body) != {"enabled", "goal"}:
                raise ModeError("expected enabled and goal")
            prepare(db)
            db.execute("BEGIN IMMEDIATE")
            answer = set_controls(db, agent_id=agent_id, actor_id=principal.user_id,
                                  actor_role=principal.role, **body)
            db.commit()
            return handler._json({"agent": canonical, **answer})
        if method == "GET":
            if principal.user_id != agent_id and principal.role not in ("operator", "admin"):
                raise ModeError("only this agent or an operator may read its mode", 403)
            query = parse_qs(handler.path.partition("?")[2], keep_blank_values=True)
            if set(query) - {"scope", "key"} or any(len(values) != 1 for values in query.values()):
                raise ModeError("expected one scope and key")
            scope = (query.get("scope") or ["global"])[0]
            key = (query.get("key") or [""])[0]
            return handler._json({"agent": canonical, **setting(db, agent_id, scope, key)})
        if method != "POST":
            raise ModeError("method not found", 404)
        if principal.role not in ("operator", "admin"):
            raise ModeError("operator or admin required", 403)
        expected_origin = ("https://" if handler._is_secure() else "http://") + handler.headers.get("Host", "")
        if handler.headers.get("Origin") not in (None, expected_origin) or handler.headers.get("Sec-Fetch-Site") == "cross-site":
            raise ModeError("cross-origin mode writes are not allowed", 403)
        body = handler._body_json()
        if body is None:
            return
        if not isinstance(body, dict) or set(body) != {"scope", "key", "mode"}:
            raise ModeError("expected scope, key and mode")
        prepare(db)
        validated_scope(db, body["scope"], body["key"], agent_id)
        db.execute("BEGIN IMMEDIATE")
        answer = set_mode(db, agent_id=agent_id, actor_id=principal.user_id, actor_role=principal.role, **body)
        db.commit()
        return handler._json({"agent": canonical, **answer})
    except ModeError as error:
        if db.in_transaction:
            db.rollback()
        return handler._json({"error": str(error)}, error.status)
    except sqlite3.Error:
        if db.in_transaction:
            db.rollback()
        return handler._json({"error": "mode storage unavailable"}, 503)
    finally:
        db.close()
