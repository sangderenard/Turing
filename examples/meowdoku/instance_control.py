"""Forced shutdown, restart and start of the service's instances, asked through SQL and done by the supervisor.

An admin's request is one row of ``instance_requests`` in the accounts database. The supervisor, which owns every
instance's process, reads the rows once a second and performs the ones that are due: it does not ask the instance
for permission, so a hung server is still stopped. The row is also the record: who asked, when, why, what was done.

    request(db, instance="meowdoku-a", action="restart", user_id=..., username=..., reason=..., delay_s=60, known=[...])
    cancel(db, request_id, user_id=..., username=...)
    pending(db) / due(db) / finish(db, request_id, "done", detail) / recent(db)

``instance`` is one instance's name or ``*`` for all of them. Actions:

    restart    stop the process (ask, wait, then kill) and start it again
    shutdown   stop it and leave it stopped until a ``start`` request or the next start of the whole stack
    start      start an instance that is stopped or failed

``after_jobs`` (the policy, on by default): a restart or shutdown that is due still waits until no instance it
names is running a job a restart would interrupt. Jobs are the server-wide table ``service_jobs`` (service_jobs.py),
which every runner writes. The supervisor reads it as the restart account (instances.json ``restart_account``, Pale
Horse), writes why it waits into ``detail``, and when the last job ends performs the request and records that
account in ``authorized_by``. ``after_jobs=False`` is the old forced request. A start never waits. Anyone who may
make a request may cancel a waiting one.

``delay_s`` is a countdown: the request is performed that many seconds after it was made, and until then anyone who
may make a request may ``cancel`` it (the interrupt). The supervisor announces a counted-down request, and its
interruption, once each; ``announced`` records that (0 nothing, 1 the countdown, 2 the interruption).

Who may ask is the caller's business (the server checks the capability before calling ``request``). Nothing here
touches Apache: these are the instances the supervisor runs, not the proxy in front of them.
"""
from __future__ import annotations

import time

ACTIONS = ("restart", "shutdown", "start")
STATES = ("pending", "done", "refused", "cancelled")
ALL = "*"
MAX_REASON = 300
MAX_DELAY_S = 3600.0
DDL = ("CREATE TABLE IF NOT EXISTS instance_requests (id INTEGER PRIMARY KEY AUTOINCREMENT, instance TEXT NOT NULL, "
       "action TEXT NOT NULL CHECK(action IN ('restart','shutdown','start')), requested_by INTEGER NOT NULL, "
       "requested_name TEXT, reason TEXT, requested_at REAL NOT NULL, "
       "state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','done','refused','cancelled')), handled_at REAL, "
       "detail TEXT, due_at REAL NOT NULL DEFAULT 0, announced INTEGER NOT NULL DEFAULT 0, "
       "after_jobs INTEGER NOT NULL DEFAULT 1, authorized_by INTEGER)")
_COLUMNS = ("id", "instance", "action", "requested_by", "requested_name", "reason", "requested_at", "state", "handled_at",
            "detail", "due_at", "announced", "after_jobs", "authorized_by")


class ControlError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def ensure(db) -> None:
    """Create the table, or bring one made before the countdown existed up to this shape, keeping its rows."""
    row = db.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'instance_requests'").fetchone()
    if row is None:
        db.execute(DDL)
        return
    if "'cancelled'" in (row[0] or "") and "announced" in (row[0] or ""):
        if "after_jobs" not in (row[0] or ""):          # added in place: a running supervisor's older code reads on
            db.execute("ALTER TABLE instance_requests ADD COLUMN after_jobs INTEGER NOT NULL DEFAULT 1")
        if "authorized_by" not in (row[0] or ""):
            db.execute("ALTER TABLE instance_requests ADD COLUMN authorized_by INTEGER")
        return
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("ALTER TABLE instance_requests RENAME TO instance_requests_before_countdown")
        db.execute(DDL)
        db.execute("INSERT INTO instance_requests (id, instance, action, requested_by, requested_name, reason, requested_at, "
                   "state, handled_at, detail, due_at) SELECT id, instance, action, requested_by, requested_name, reason, "
                   "requested_at, state, handled_at, detail, requested_at FROM instance_requests_before_countdown")
        db.execute("DROP TABLE instance_requests_before_countdown")
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise


def _row(row) -> dict:
    item = dict(zip(_COLUMNS, row))
    item["after_jobs"] = bool(item["after_jobs"])
    return item


def _select(where: str) -> str:
    return f"SELECT {', '.join(_COLUMNS)} FROM instance_requests {where}"


def request(db, *, instance, action, user_id, username, reason=None, delay_s=0, known=(), after_jobs=True) -> dict:
    """Record one request. Refused when the instance is unknown, the action is not one of ACTIONS, the countdown is
    out of range, or the same request is already pending (asking twice does not restart twice)."""
    ensure(db)
    if action not in ACTIONS:
        raise ControlError(f"action must be one of {', '.join(ACTIONS)}")
    if not isinstance(instance, str) or (instance != ALL and instance not in set(known)):
        raise ControlError("no such instance", 404)
    if not isinstance(user_id, int) or isinstance(user_id, bool):
        raise ControlError("a signed-in account is required", 401)
    if reason is not None and (not isinstance(reason, str) or len(reason) > MAX_REASON):
        raise ControlError(f"reason must be text of at most {MAX_REASON} characters")
    if isinstance(delay_s, bool) or not isinstance(delay_s, (int, float)) or not 0 <= delay_s <= MAX_DELAY_S:
        raise ControlError(f"delay must be 0 to {MAX_DELAY_S:g} seconds")
    if not isinstance(after_jobs, bool):
        raise ControlError("after_jobs must be true or false")
    now = time.time()
    db.execute("BEGIN IMMEDIATE")
    try:
        same = db.execute("SELECT id FROM instance_requests WHERE state = 'pending' AND instance = ? AND action = ?",
                          (instance, action)).fetchone()
        if same is not None:
            db.execute("ROLLBACK")
            raise ControlError("the same request is already pending", 409)
        cur = db.execute("INSERT INTO instance_requests (instance, action, requested_by, requested_name, reason, requested_at, "
                         "due_at, after_jobs) VALUES (?,?,?,?,?,?,?,?)",
                         (instance, action, user_id, username, (reason or "").strip() or None, now, now + float(delay_s),
                          int(after_jobs)))
        db.execute("COMMIT")
    except ControlError:
        raise
    except BaseException:
        db.execute("ROLLBACK")
        raise
    return get(db, cur.lastrowid)


def cancel(db, request_id, *, user_id, username) -> dict:
    """The interrupt: a request that has not been performed yet is cancelled. Anything else is refused."""
    ensure(db)
    if isinstance(request_id, bool) or not isinstance(request_id, int):
        raise ControlError("id must be a request's number")
    if not isinstance(user_id, int) or isinstance(user_id, bool):
        raise ControlError("a signed-in account is required", 401)
    done = db.execute("UPDATE instance_requests SET state = 'cancelled', handled_at = ?, detail = ? "
                      "WHERE id = ? AND state = 'pending'", (time.time(), f"interrupted by {username}", request_id))
    if done.rowcount != 1:
        current = get(db, request_id)                 # raises 404 when there is no such request
        raise ControlError(f"request {request_id} is already {current['state']}", 409)
    return get(db, request_id)


def get(db, request_id) -> dict:
    row = db.execute(_select("WHERE id = ?"), (int(request_id),)).fetchone()
    if row is None:
        raise ControlError("no such request", 404)
    return _row(tuple(row))


def pending(db) -> list:
    ensure(db)
    return [_row(tuple(r)) for r in db.execute(_select("WHERE state = 'pending' ORDER BY id"))]


def due(db, now: float | None = None) -> list:
    """Pending requests whose countdown has run out."""
    now = time.time() if now is None else now
    return [item for item in pending(db) if item["due_at"] <= now]


def to_announce(db) -> list:
    """Requests the supervisor has not yet told anyone about: a countdown that began, or one that was interrupted."""
    ensure(db)
    return [_row(tuple(r)) for r in db.execute(_select(
        "WHERE (state = 'pending' AND announced = 0 AND due_at > requested_at) "
        "OR (state = 'cancelled' AND announced = 1) ORDER BY id"))]


def mark_announced(db, request_id, value: int) -> None:
    db.execute("UPDATE instance_requests SET announced = ? WHERE id = ?", (int(value), int(request_id)))


def waiting(db, request_id, detail) -> None:
    """A due request held for running jobs: say why in its row, which stays pending."""
    db.execute("UPDATE instance_requests SET detail = ? WHERE id = ? AND state = 'pending'", (str(detail)[:1000], int(request_id)))


def finish(db, request_id, state, detail, authorized_by=None) -> None:
    """Settle a request. ``authorized_by`` is the restart account's user id when it authorized a request that waited for
    jobs (the supervisor's ``restart_account``); None for a request performed without that check."""
    db.execute("UPDATE instance_requests SET state = ?, handled_at = ?, detail = ?, authorized_by = ? "
               "WHERE id = ? AND state = 'pending'", (state, time.time(), str(detail)[:1000], authorized_by, int(request_id)))


def recent(db, limit=20) -> list:
    ensure(db)
    return [_row(tuple(r)) for r in db.execute(_select("ORDER BY id DESC LIMIT ?"), (max(1, min(int(limit), 200)),))]
