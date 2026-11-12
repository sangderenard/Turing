"""Every running job of the service, in one SQL table: the work a restart or shutdown would interrupt.

A runner that starts work a restart would cut short writes one row of ``service_jobs`` in the accounts database when
the work starts (``begin``) and marks it ended when the work ends, however it ends (``end``). The table is
server-wide: any kind of job, from any instance, in one place, and anything that must not interrupt work reads it
(``running``) instead of asking each runner.

A row names the instance whose process runs the job and that process's pid. It counts only while that instance's
recorded pid in ``service_instances`` is the same, so a row left behind by a process that died or was restarted stops
counting by itself, with no heartbeat. A row whose job ended without ``end`` (a runner's fault) keeps counting while its
process lives: it is visible in the listing and in the reason a restart gives for waiting, and an admin can force.

    jobs = Registry(accounts_db, instance="meowdoku-a")      # one per process that runs jobs
    jobs.begin("marble_photo", digest, owner="account:3", label="exposure")
    jobs.end("marble_photo", digest)
    running(db, instances=("meowdoku-a",))                     # the live rows, for the restart account
    listing(db)                                                # running rows, live or not, for people

Registering never breaks a job: a database error is reported on stderr and the job goes on. A job that could not
register is then not waited for, which is the one way this table can be wrong, and the stderr line says so.

Runners registered: the stable server's GPU photo exposure (marble_photo.PhotoJobs) and the math lab worker
(math_jobs.Worker). A runner that is added later registers here; nothing infers jobs from process names.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time

DDL = ("CREATE TABLE IF NOT EXISTS service_jobs (kind TEXT NOT NULL, job_id TEXT NOT NULL, instance TEXT NOT NULL, "
       "pid INTEGER NOT NULL, owner TEXT, label TEXT, started_at REAL NOT NULL, ended_at REAL, "
       "state TEXT NOT NULL DEFAULT 'running' CHECK(state IN ('running','ended')), PRIMARY KEY(kind, job_id))")
_COLUMNS = ("kind", "job_id", "instance", "pid", "owner", "label", "started_at", "ended_at", "state")
MAX_TEXT = 200


def ensure(db) -> None:
    db.execute(DDL)


def _has_instances(db) -> bool:
    return db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'service_instances'").fetchone() is not None


def running(db, instances=None) -> list:
    """Running jobs whose process is still the one the supervisor recorded for their instance; ``instances`` narrows."""
    ensure(db)
    if not _has_instances(db):
        return []
    rows = db.execute(f"SELECT {', '.join('j.' + c for c in _COLUMNS)} FROM service_jobs j JOIN service_instances s "
                      "ON s.name = j.instance AND s.pid = j.pid WHERE j.state = 'running' ORDER BY j.started_at").fetchall()
    out = [dict(zip(_COLUMNS, tuple(r))) for r in rows]
    if instances is not None:
        wanted = set(instances)
        out = [row for row in out if row["instance"] in wanted]
    return out


def listing(db, limit: int = 100) -> list:
    """Every row still marked running, each saying whether it counts (``live``): what people read."""
    ensure(db)
    live = {(r["kind"], r["job_id"]) for r in running(db)}
    rows = db.execute(f"SELECT {', '.join(_COLUMNS)} FROM service_jobs WHERE state = 'running' ORDER BY started_at DESC "
                      "LIMIT ?", (max(1, min(int(limit), 500)),)).fetchall()
    return [{**dict(zip(_COLUMNS, tuple(r))), "live": (r[0], r[1]) in live} for r in rows]


class Registry:
    """One process's writer. ``instance`` is the supervisor's name for this process ('' outside the supervisor: its
    rows are kept but never match an instance, so they hold nothing)."""

    def __init__(self, db_path, instance: str | None, *, pid: int | None = None, clock=time.time):
        self.db_path, self.instance = str(db_path), str(instance or "")
        self.pid, self.clock = int(pid if pid is not None else os.getpid()), clock

    def _write(self, sql, args, what) -> bool:
        try:
            db = sqlite3.connect(self.db_path, timeout=5, isolation_level=None)
            try:
                ensure(db)
                db.execute(sql, args)
            finally:
                db.close()
            return True
        except sqlite3.Error as error:
            print(f"[jobs] could not {what}: {error}; a restart will not wait for it", file=sys.stderr, flush=True)
            return False

    def begin(self, kind, job_id, *, owner=None, label=None) -> bool:
        return self._write("INSERT OR REPLACE INTO service_jobs (kind, job_id, instance, pid, owner, label, started_at, "
                           "ended_at, state) VALUES (?,?,?,?,?,?,?,NULL,'running')",
                           (str(kind)[:MAX_TEXT], str(job_id)[:MAX_TEXT], self.instance, self.pid,
                            None if owner is None else str(owner)[:MAX_TEXT], None if label is None else str(label)[:MAX_TEXT],
                            self.clock()), f"register {kind} {job_id}")

    def end(self, kind, job_id) -> bool:
        return self._write("UPDATE service_jobs SET state = 'ended', ended_at = ? WHERE kind = ? AND job_id = ? AND pid = ? "
                           "AND state = 'running'", (self.clock(), str(kind)[:MAX_TEXT], str(job_id)[:MAX_TEXT], self.pid),
                           f"end {kind} {job_id}")


__all__ = ["DDL", "Registry", "ensure", "listing", "running"]
