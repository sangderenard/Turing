"""Forced shutdown, restart and start of the service's instances, asked through SQL and done by the supervisor.

An admin's request is one row of ``instance_requests`` in the accounts database. The supervisor, which owns every
instance's process, reads the pending rows once a second and performs them: it does not ask the instance for
permission, so a hung server is still stopped. The row is also the record: who asked, when, why, what was done.

    request(db, instance="meowdoku-a", action="restart", user_id=..., username=..., reason=..., known=[...])
    pending(db) / finish(db, request_id, "done", detail) / recent(db)

``instance`` is one instance's name or ``*`` for all of them. Actions:

    restart    stop the process (ask, wait, then kill) and start it again
    shutdown   stop it and leave it stopped until a ``start`` request or the next start of the whole stack
    start      start an instance that is stopped or failed

Who may ask is the caller's business (the server checks the capability before calling ``request``). Nothing here
touches Apache: these are the instances the supervisor runs, not the proxy in front of them.
"""
from __future__ import annotations

import time

ACTIONS = ("restart", "shutdown", "start")
ALL = "*"
MAX_REASON = 300
DDL = ("CREATE TABLE IF NOT EXISTS instance_requests (id INTEGER PRIMARY KEY AUTOINCREMENT, instance TEXT NOT NULL, "
       "action TEXT NOT NULL CHECK(action IN ('restart','shutdown','start')), requested_by INTEGER NOT NULL, "
       "requested_name TEXT, reason TEXT, requested_at REAL NOT NULL, "
       "state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','done','refused')), handled_at REAL, detail TEXT)")
_COLUMNS = ("id", "instance", "action", "requested_by", "requested_name", "reason", "requested_at", "state", "handled_at", "detail")


class ControlError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def ensure(db) -> None:
    db.execute(DDL)


def _row(row) -> dict:
    return dict(zip(_COLUMNS, row))


def request(db, *, instance, action, user_id, username, reason=None, known=()) -> dict:
    """Record one request. Refused when the instance is unknown, the action is not one of ACTIONS, or the same
    request is already pending (asking twice does not restart twice)."""
    ensure(db)
    if action not in ACTIONS:
        raise ControlError(f"action must be one of {', '.join(ACTIONS)}")
    if not isinstance(instance, str) or (instance != ALL and instance not in set(known)):
        raise ControlError("no such instance", 404)
    if not isinstance(user_id, int) or isinstance(user_id, bool):
        raise ControlError("a signed-in account is required", 401)
    if reason is not None and (not isinstance(reason, str) or len(reason) > MAX_REASON):
        raise ControlError(f"reason must be text of at most {MAX_REASON} characters")
    db.execute("BEGIN IMMEDIATE")
    try:
        same = db.execute("SELECT id FROM instance_requests WHERE state = 'pending' AND instance = ? AND action = ?",
                          (instance, action)).fetchone()
        if same is not None:
            db.execute("ROLLBACK")
            raise ControlError("the same request is already pending", 409)
        cur = db.execute("INSERT INTO instance_requests (instance, action, requested_by, requested_name, reason, requested_at) "
                         "VALUES (?,?,?,?,?,?)", (instance, action, user_id, username, (reason or "").strip() or None, time.time()))
        db.execute("COMMIT")
    except ControlError:
        raise
    except BaseException:
        db.execute("ROLLBACK")
        raise
    return get(db, cur.lastrowid)


def get(db, request_id) -> dict:
    row = db.execute(f"SELECT {', '.join(_COLUMNS)} FROM instance_requests WHERE id = ?", (int(request_id),)).fetchone()
    if row is None:
        raise ControlError("no such request", 404)
    return _row(tuple(row))


def pending(db) -> list:
    ensure(db)
    return [_row(tuple(r)) for r in db.execute(
        f"SELECT {', '.join(_COLUMNS)} FROM instance_requests WHERE state = 'pending' ORDER BY id")]


def finish(db, request_id, state, detail) -> None:
    db.execute("UPDATE instance_requests SET state = ?, handled_at = ?, detail = ? WHERE id = ? AND state = 'pending'",
               (state, time.time(), str(detail)[:1000], int(request_id)))


def recent(db, limit=20) -> list:
    ensure(db)
    return [_row(tuple(r)) for r in db.execute(
        f"SELECT {', '.join(_COLUMNS)} FROM instance_requests ORDER BY id DESC LIMIT ?", (max(1, min(int(limit), 200)),))]
