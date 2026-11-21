"""Public progress receipts for server-owned historical mock qualification runs.

Workers write to the paper arena SQLite file shared by Meowdoku instances.
The spectator API returns only bounded, numeric training status; no credentials,
weights, source paths, or per-account market activity enter these rows.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import re
import time
import uuid

from meowdoku import paper_arena


RUN_ID = re.compile(r"^[0-9a-f]{32}$")
PHASES = frozenset(("training", "validation", "compiling", "holdout", "autosaving", "complete"))
STATUSES = frozenset(("running", "stopped", "passed_historical", "unqualified", "failed"))
LIVE_SOURCES = frozenset(("waiting", "current", "stale", "missing", "unaligned", "invalid",
                          "insufficient", "unavailable"))


def _tables(db) -> None:
    db.execute("CREATE TABLE IF NOT EXISTS paper_qualification_runs ("
               "id TEXT PRIMARY KEY, market TEXT NOT NULL, model TEXT NOT NULL, "
               "worker TEXT NOT NULL, status TEXT NOT NULL, phase TEXT NOT NULL, "
               "completed_epochs INTEGER NOT NULL, planned_epochs INTEGER NOT NULL, "
               "created REAL NOT NULL, updated REAL NOT NULL, finished REAL, result TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS paper_qualification_progress ("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, phase TEXT NOT NULL, "
               "seed INTEGER, epoch INTEGER, loss REAL, validation_mean REAL, "
               "validation_passed INTEGER, created REAL NOT NULL)")
    db.execute("CREATE INDEX IF NOT EXISTS paper_qualification_progress_run "
               "ON paper_qualification_progress(run_id,id)")
    db.execute("CREATE TABLE IF NOT EXISTS paper_training_live_state ("
               "run_id TEXT PRIMARY KEY, source TEXT NOT NULL, observed_at TEXT, "
               "matured_frames INTEGER NOT NULL, historical_batch INTEGER NOT NULL, "
               "live_batch INTEGER NOT NULL, updated REAL NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS paper_qualification_cleared ("
               "run_id TEXT PRIMARY KEY, cleared_by TEXT NOT NULL, cleared_at REAL NOT NULL)")


def clear(db, run_id: str, *, actor_id: str) -> None:
    """Remove a finished or stalled receipt from the spectator view, retaining its audit rows."""
    _tables(db)
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id) or not actor_id:
        raise ValueError("invalid qualification run or actor")
    db.execute("BEGIN IMMEDIATE")
    try:
        row = db.execute("SELECT status,updated FROM paper_qualification_runs WHERE id=?",
                         (run_id,)).fetchone()
        if row is None:
            raise LookupError("qualification run not found")
        if row["status"] == "running" and time.time() - row["updated"] <= 90:
            raise RuntimeError("stop the active training before clearing its run")
        db.execute("INSERT OR IGNORE INTO paper_qualification_cleared VALUES(?,?,?)",
                   (run_id, actor_id, time.time()))
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise


def live_state(db, run_id: str, *, source: str, observed_at: str | None,
               matured_frames: int, historical_batch: int, live_batch: int) -> None:
    """Publish market freshness and the actual supervised batch composition."""
    _tables(db)
    if (not RUN_ID.fullmatch(run_id) or source not in LIVE_SOURCES
            or observed_at is not None and (not isinstance(observed_at, str) or len(observed_at) > 40)
            or any(type(value) is not int or value < 0 for value in
                   (matured_frames, historical_batch, live_batch))
            or live_batch and historical_batch != live_batch):
        raise ValueError("invalid live training state")
    db.execute("INSERT INTO paper_training_live_state VALUES(?,?,?,?,?,?,?) "
               "ON CONFLICT(run_id) DO UPDATE SET source=excluded.source, "
               "observed_at=excluded.observed_at, matured_frames=excluded.matured_frames, "
               "historical_batch=excluded.historical_batch, live_batch=excluded.live_batch, "
               "updated=excluded.updated",
               (run_id, source, observed_at, matured_frames, historical_batch, live_batch, time.time()))


def begin(db, *, market: str, model: str, worker: str, planned_epochs: int) -> str:
    _tables(db)
    if not (isinstance(market, str) and 1 <= len(market) <= 24
            and isinstance(model, str) and 1 <= len(model) <= 80
            and isinstance(worker, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", worker)
            and type(planned_epochs) is int and 0 <= planned_epochs <= 1000000):
        raise ValueError("invalid qualification run metadata")
    run_id = uuid.uuid4().hex
    now = time.time()
    db.execute("INSERT INTO paper_qualification_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
               (run_id, market, model, worker, "running", "training", 0,
                planned_epochs, now, now, None, "{}"))
    return run_id


def progress(db, run_id: str, *, phase: str, completed_epochs: int,
             seed: int | None = None, epoch: int | None = None,
             loss: float | None = None, validation_mean: float | None = None,
             validation_passed: bool | None = None) -> None:
    _tables(db)
    if not RUN_ID.fullmatch(run_id) or phase not in PHASES or type(completed_epochs) is not int:
        raise ValueError("invalid qualification progress")
    for value in (loss, validation_mean):
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("qualification metric must be finite")
    db.execute("BEGIN IMMEDIATE")
    try:
        run = db.execute("SELECT status,planned_epochs,completed_epochs FROM paper_qualification_runs WHERE id=?",
                         (run_id,)).fetchone()
        if (run is None or run["status"] != "running"
                or completed_epochs < run["completed_epochs"]
                or (run["planned_epochs"] and completed_epochs > run["planned_epochs"])):
            raise ValueError("qualification run is not active or progress moved backward")
        now = time.time()
        db.execute("UPDATE paper_qualification_runs SET phase=?,completed_epochs=?,updated=? WHERE id=?",
                   (phase, completed_epochs, now, run_id))
        db.execute("INSERT INTO paper_qualification_progress "
                   "(run_id,phase,seed,epoch,loss,validation_mean,validation_passed,created) "
                   "VALUES(?,?,?,?,?,?,?,?)",
                   (run_id, phase, seed, epoch, loss, validation_mean,
                    None if validation_passed is None else int(validation_passed), now))
        if run["planned_epochs"] == 0:
            db.execute("DELETE FROM paper_qualification_progress WHERE run_id=? AND id NOT IN "
                       "(SELECT id FROM paper_qualification_progress WHERE run_id=? ORDER BY id DESC LIMIT 1024)",
                       (run_id, run_id))
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise


def finish(db, run_id: str, *, status: str, result: dict) -> None:
    _tables(db)
    if not RUN_ID.fullmatch(run_id) or status not in STATUSES - {"running"} or not isinstance(result, dict):
        raise ValueError("invalid qualification result")
    public = {key: result[key] for key in ("dataset_sha256", "validation_passed",
              "holdout_passed", "holdout_mean_return_on_order_cap", "holdout_window_pnl_usd")
              if key in result}
    if len(json.dumps(public)) > 2048:
        raise ValueError("qualification result too large")
    now = time.time()
    changed = db.execute("UPDATE paper_qualification_runs SET status=?,phase='complete',"
                         "updated=?,finished=?,result=? WHERE id=? AND status='running'",
                         (status, now, now, json.dumps(public, separators=(",", ":")), run_id)).rowcount
    if changed != 1:
        raise ValueError("qualification run is not active")


def public_snapshot(db, *, limit: int = 12) -> dict:
    _tables(db)
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("invalid qualification page size")
    now = time.time()
    rows = db.execute("SELECT * FROM paper_qualification_runs WHERE id NOT IN "
                      "(SELECT run_id FROM paper_qualification_cleared) "
                      "ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
    runs = []
    for row in rows:
        stale = row["status"] == "running" and now - row["updated"] > 90
        recent = db.execute("SELECT phase,seed,epoch,loss,validation_mean,validation_passed,created "
                            "FROM paper_qualification_progress WHERE run_id=? ORDER BY id DESC LIMIT 24",
                            (row["id"],)).fetchall()
        live = db.execute("SELECT source,observed_at,matured_frames,historical_batch,live_batch,updated "
                          "FROM paper_training_live_state WHERE run_id=?", (row["id"],)).fetchone()
        runs.append({"id": row["id"], "market": row["market"], "model": row["model"],
                     "worker": row["worker"], "status": "stalled" if stale else row["status"],
                     "phase": row["phase"], "completed_epochs": row["completed_epochs"],
                     "planned_epochs": row["planned_epochs"], "created": row["created"],
                     "updated": row["updated"], "finished": row["finished"],
                     "result": json.loads(row["result"]),
                     "progress": [dict(event) for event in reversed(recent)],
                     "live": dict(live) if live is not None else None})
    return {"schema": "paper-qualification-spectator-v1", "runs": runs, "asof": now}


@contextmanager
def publishing(path, *, market: str, model: str, worker: str, planned_epochs: int):
    db = paper_arena.connect(path)
    try:
        run_id = begin(db, market=market, model=model, worker=worker, planned_epochs=planned_epochs)
        try:
            yield db, run_id
        except BaseException:
            finish(db, run_id, status="failed", result={})
            raise
    finally:
        db.close()
