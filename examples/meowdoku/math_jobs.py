"""Math lab job queue: one table in accounts.db and ONE worker thread (the source_divider database pattern).

    submit(db, state, owner_key, text, targets)   parse first (math_lab.parse_math; a MathSyntaxError is a 'rejected' row, no worker
                                                  time), then the per-owner and global limits, then a 'queued' row
    Worker(db_path, state).start()                one daemon thread: oldest queued job -> math_lab.run_job(text, <jobs_dir>/math/<id>,
                                                  timeout_s=120) -> record -> next.  Never two at once.

Limits: at most MAX_ACTIVE_PER_OWNER (3) queued+running jobs per owner, MAX_PER_HOUR (20) submissions per owner per hour (rejected
ones count: a parse costs the server something), MAX_QUEUE (32) queued+running in total.  On start the worker requeues 'running' rows
once (error column says so) and marks them 'error' the second time.  Between jobs it prunes jobs older than 24 h and keeps at most
MAX_JOBS (200) rows; a pruned job's directory goes with it.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "examples")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from meowdoku import math_lab                          # noqa: E402
from meowdoku import schema_version as _schema_version  # noqa: E402

MAX_ACTIVE_PER_OWNER, MAX_PER_HOUR, MAX_QUEUE = 3, 20, 32
MAX_JOBS, MAX_AGE_S, TIMEOUT_S = 200, 24 * 3600.0, 120
REQUEUED = "requeued after a server restart"

DDL = ("CREATE TABLE IF NOT EXISTS math_jobs (job_id TEXT PRIMARY KEY, owner_key TEXT NOT NULL, text TEXT NOT NULL, targets TEXT, "
       "state TEXT CHECK(state IN ('queued','running','complete','rejected','timeout','error')), created REAL, started REAL, "
       "finished REAL, queue_position INTEGER, error TEXT)",
       "CREATE INDEX IF NOT EXISTS math_jobs_owner ON math_jobs (owner_key, created)")
COLUMNS = ("job_id", "owner_key", "text", "targets", "state", "created", "started", "finished", "queue_position", "error")


class LimitError(Exception):
    def __init__(self, message: str, status: int = 429):
        super().__init__(message)
        self.message, self.status = message, status


def connect(db_path) -> sqlite3.Connection:
    """Own connection to accounts.db (autocommit).  Declares schema module ``math_jobs`` 1."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(db_path), timeout=10, isolation_level=None, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    try:
        _schema_version.declare(db, "math_jobs", 1)
    except _schema_version.SchemaVersionError:
        db.close()
        raise
    for ddl in DDL:
        db.execute(ddl)
    return db


def job_dir(state, job_id: str) -> Path:
    return Path(state.jobs_dir) / "math" / job_id


def _targets(targets) -> tuple:
    if targets is None:
        return math_lab.ALL_TARGETS
    if isinstance(targets, str):
        targets = targets.split(",")
    names = tuple(dict.fromkeys(str(t).strip() for t in targets if str(t).strip()))
    unknown = [t for t in names if t not in math_lab.ALL_TARGETS]
    if unknown or not names:
        raise ValueError(f"targets must be a non-empty subset of {math_lab.ALL_TARGETS}; got {unknown or 'none'}")
    return names


def position(db, row) -> int:
    """0 unless queued; else 1 + the queued/running jobs submitted before this one."""
    if row["state"] != "queued":
        return 0
    ahead = db.execute("SELECT COUNT(*) FROM math_jobs WHERE state IN ('queued','running') AND job_id != ? "
                       "AND (created < ? OR (created = ? AND job_id < ?))",
                       (row["job_id"], row["created"], row["created"], row["job_id"])).fetchone()[0]
    return int(ahead) + 1


def submit(db, state, owner_key: str, text: str, targets=None, now: float | None = None) -> dict:
    """-> {job_id, state, queue_position}, or for text outside the math language {job_id, state: 'rejected', error: {reason, line, col,
    construct}}.  Raises ValueError (bad targets) or LimitError (429 / 503)."""
    names = _targets(targets)
    now = time.time() if now is None else now
    try:
        math_lab.parse_math(text)
        problem = None
    except math_lab.MathSyntaxError as error:
        problem = {"reason": error.reason, "line": error.line, "col": error.col, "construct": error.construct}
    owner_key = str(owner_key)
    recent = db.execute("SELECT COUNT(*) FROM math_jobs WHERE owner_key = ? AND created > ?", (owner_key, now - 3600)).fetchone()[0]
    if recent >= MAX_PER_HOUR:
        raise LimitError(f"at most {MAX_PER_HOUR} submissions per hour")
    job_id = uuid.uuid4().hex[:16]
    if problem is not None:
        db.execute("INSERT INTO math_jobs (job_id, owner_key, text, targets, state, created, finished, queue_position, error) "
                   "VALUES (?,?,?,?,?,?,?,?,?)", (job_id, owner_key, str(text)[:math_lab.MAX_CHARS], ",".join(names), "rejected", now, now, 0,
                                                  json.dumps(problem)))
        return {"job_id": job_id, "state": "rejected", "error": problem}
    db.execute("BEGIN IMMEDIATE")
    try:
        active = db.execute("SELECT COUNT(*) FROM math_jobs WHERE owner_key = ? AND state IN ('queued','running')", (owner_key,)).fetchone()[0]
        total = db.execute("SELECT COUNT(*) FROM math_jobs WHERE state IN ('queued','running')").fetchone()[0]
        if active >= MAX_ACTIVE_PER_OWNER:
            raise LimitError(f"at most {MAX_ACTIVE_PER_OWNER} queued or running jobs per account")
        if total >= MAX_QUEUE:
            raise LimitError(f"the queue is full ({MAX_QUEUE} jobs); try again later", 503)
        db.execute("INSERT INTO math_jobs (job_id, owner_key, text, targets, state, created, queue_position) VALUES (?,?,?,?,?,?,?)",
                   (job_id, owner_key, text, ",".join(names), "queued", now, total + 1))
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    return {"job_id": job_id, "state": "queued", "queue_position": total + 1}


def _row(db, row) -> dict:
    out = {k: row[k] for k in COLUMNS if k not in ("owner_key", "text")}
    out["targets"] = (row["targets"] or "").split(",")
    out["queue_position"] = position(db, row)
    if row["state"] == "rejected" and row["error"]:
        try:
            out["error"] = json.loads(row["error"])
        except ValueError:
            pass
    return out


def list_jobs(db, owner_key: str) -> list[dict]:
    return [_row(db, r) for r in db.execute("SELECT * FROM math_jobs WHERE owner_key = ? ORDER BY created DESC LIMIT 50", (str(owner_key),))]


def get_job(db, state, job_id: str, owner_key: str) -> dict | None:
    """The job with its products.json / profile.server.json / profile.browser.json content, or None (not found OR not the owner's)."""
    row = db.execute("SELECT * FROM math_jobs WHERE job_id = ? AND owner_key = ?", (str(job_id), str(owner_key))).fetchone()
    if row is None:
        return None
    out = _row(db, row)
    out["text"] = row["text"]
    directory = job_dir(state, row["job_id"])
    for key, name in (("products", "products.json"), ("profile", "profile.server.json"), ("browser_profile", "profile.browser.json")):
        path = directory / name
        try:
            out[key] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        except (OSError, ValueError):
            out[key] = None
    return out


# ---- the one worker ----------------------------------------------------------------------------------------------------
def recover(db, now: float | None = None) -> None:
    """A 'running' row at startup was interrupted: requeue it once, then 'error'."""
    now = time.time() if now is None else now
    for row in db.execute("SELECT job_id, error FROM math_jobs WHERE state = 'running'").fetchall():
        if row["error"] == REQUEUED:
            db.execute("UPDATE math_jobs SET state='error', finished=?, error=? WHERE job_id=?",
                       (now, "interrupted by two server restarts", row["job_id"]))
        else:
            db.execute("UPDATE math_jobs SET state='queued', started=NULL, error=? WHERE job_id=?", (REQUEUED, row["job_id"]))


def prune(db, state, now: float | None = None) -> int:
    """Delete finished jobs older than 24 h and finished jobs beyond the newest MAX_JOBS; their directories go too."""
    now = time.time() if now is None else now
    doomed = {r[0] for r in db.execute("SELECT job_id FROM math_jobs WHERE state NOT IN ('queued','running') AND created < ?",
                                       (now - MAX_AGE_S,))}
    for job_id, final in [(r[0], r[1]) for r in db.execute("SELECT job_id, state FROM math_jobs ORDER BY created DESC")][MAX_JOBS:]:
        if final not in ("queued", "running"):
            doomed.add(job_id)
    for job_id in doomed:
        db.execute("DELETE FROM math_jobs WHERE job_id = ?", (job_id,))
        shutil.rmtree(job_dir(state, job_id), ignore_errors=True)
    return len(doomed)


def work_one(db, state, run=None, registry=None) -> bool:
    """Run the oldest queued job to its end.  False when nothing was queued.  ``registry`` (service_jobs.Registry) holds
    the job's row of the server-wide job table while it runs, so a restart waits for it."""
    run = run or math_lab.run_job
    row = db.execute("SELECT * FROM math_jobs WHERE state = 'queued' ORDER BY created, job_id LIMIT 1").fetchone()
    if row is None:
        return False
    db.execute("UPDATE math_jobs SET state='running', started=?, queue_position=0 WHERE job_id=?", (time.time(), row["job_id"]))
    if registry is not None:
        registry.begin("math_job", row["job_id"], owner=row["owner_key"], label="math lab job")
    try:
        result = run(row["text"], job_dir(state, row["job_id"]), timeout_s=TIMEOUT_S, targets=tuple(row["targets"].split(",")))
        final, error = result.get("state", "error"), result.get("error")
        if final not in ("complete", "rejected", "timeout", "error"):
            final = "error"
        if final == "timeout":
            error = f"timed out after {result.get('timeout_s', TIMEOUT_S)} s"
        error = json.dumps(error) if isinstance(error, dict) else error
    except Exception as exc:  # noqa: BLE001 - a job never takes the worker down
        final, error = "error", f"{type(exc).__name__}: {exc}"[:1200]
    finally:
        if registry is not None:
            registry.end("math_job", row["job_id"])
    db.execute("UPDATE math_jobs SET state=?, finished=?, error=? WHERE job_id=?", (final, time.time(), error, row["job_id"]))
    return True


class Worker:
    """ONE daemon thread.  ``wake()`` skips the poll wait; ``stop()`` ends it after the current job."""

    def __init__(self, db_path, state, poll_s: float = 0.5, run=None, registry=None):
        self.db_path, self.state, self.poll_s, self.run, self.registry = db_path, state, poll_s, run, registry
        self._stop, self._wake = threading.Event(), threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> "Worker":
        db = connect(self.db_path)
        try:
            recover(db)
        finally:
            db.close()
        self.thread = threading.Thread(target=self._loop, name="mathlab-worker", daemon=True)
        self.thread.start()
        return self

    def wake(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _loop(self) -> None:
        db = connect(self.db_path)
        try:
            while not self._stop.is_set():
                try:
                    did = work_one(db, self.state, self.run, self.registry)
                    prune(db, self.state)
                except Exception as exc:  # noqa: BLE001
                    print(f"[mathlab] worker error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                    did = False
                if not did:
                    self._wake.wait(self.poll_s)
                    self._wake.clear()
        finally:
            db.close()
