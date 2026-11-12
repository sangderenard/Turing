"""The restart policy: a due restart or shutdown waits until no instance it names runs a job a restart would interrupt
(instance_control.after_jobs; the supervisor asks each server's lifecycle op ``busy``)."""
import importlib.util
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import instance_control as control  # noqa: E402
from meowdoku import lifecycle as lifecycle_mod  # noqa: E402
from meowdoku.marble_photo import PhotoJobs  # noqa: E402

_spec = importlib.util.spec_from_file_location("apache_supervisor_under_test", ROOT / "examples/meowdoku/deploy/apache_supervisor.py")
sup_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sup_mod)


def _db(path):
    db = sqlite3.connect(path, timeout=10, isolation_level=None)
    control.ensure(db)
    return db


def test_requests_wait_for_jobs_unless_told_not_to(tmp_path):
    db = _db(tmp_path / "a.db")
    waits = control.request(db, instance="web", action="restart", user_id=1, username="admin", known=["web"])
    forced = control.request(db, instance="web", action="shutdown", user_id=1, username="admin", known=["web"], after_jobs=False)
    assert (waits["after_jobs"], forced["after_jobs"]) == (True, False)
    with pytest.raises(control.ControlError):
        control.request(db, instance="web", action="start", user_id=1, username="admin", known=["web"], after_jobs="yes")
    control.waiting(db, waits["id"], "waiting for running jobs: web marble_photo abc")
    held = control.get(db, waits["id"])
    assert held["state"] == "pending" and held["detail"].startswith("waiting for running jobs")


def test_a_table_from_before_the_policy_gains_the_column_and_keeps_its_rows(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path, isolation_level=None)
    old.execute(control.DDL.replace(", after_jobs INTEGER NOT NULL DEFAULT 1", ""))
    old.execute("INSERT INTO instance_requests (instance, action, requested_by, requested_at) VALUES ('web','restart',1,1.0)")
    old.close()
    db = _db(path)
    assert [(r["instance"], r["after_jobs"]) for r in control.pending(db)] == [("web", True)]


class _Inst:
    def __init__(self, name, jobs):
        self.name, self.jobs, self.spec, self.done = name, jobs, {}, []

    def running_jobs(self):
        return self.jobs

    def force_stop(self):
        self.done.append("stop")
        return "stopped"

    def force_start(self):
        self.done.append("start")
        return "started"


def _supervisor(db_path, instances):
    logs = []
    fake = SimpleNamespace(db_path=str(db_path), instances=instances, log=logs.append, announce=lambda item: None)
    return fake, logs


def test_the_supervisor_holds_a_restart_while_a_job_runs_and_performs_it_after_the_last(tmp_path):
    path = tmp_path / "accounts.db"
    db = _db(path)
    web = _Inst("web", [{"kind": "marble_photo", "digest": "0123456789abcdef"}])
    worker = _Inst("worker", [])
    fake, logs = _supervisor(path, [web, worker])
    made = control.request(db, instance="*", action="restart", user_id=1, username="admin", known=["web", "worker"])
    sup_mod.Supervisor.handle_requests(fake)
    sup_mod.Supervisor.handle_requests(fake)            # still running: no second log line, nothing touched
    row = control.get(db, made["id"])
    assert row["state"] == "pending" and row["detail"] == "waiting for running jobs: web marble_photo 0123456789ab"
    assert web.done == worker.done == [] and len(logs) == 1
    web.jobs = []                                       # the last job ended
    sup_mod.Supervisor.handle_requests(fake)
    row = control.get(db, made["id"])
    assert row["state"] == "done" and web.done == ["stop", "start"] and worker.done == ["stop", "start"]


def test_a_forced_request_and_a_server_that_cannot_answer_do_not_wait(tmp_path):
    path = tmp_path / "accounts.db"
    db = _db(path)
    busy = _Inst("web", [{"kind": "marble_photo", "digest": "ff"}])
    fake, _ = _supervisor(path, [busy])
    forced = control.request(db, instance="web", action="restart", user_id=1, username="admin", known=["web"], after_jobs=False)
    sup_mod.Supervisor.handle_requests(fake)
    assert control.get(db, forced["id"])["state"] == "done"
    silent = _Inst("web", None)
    fake, _ = _supervisor(path, [silent])
    asked = control.request(db, instance="web", action="restart", user_id=1, username="admin", known=["web"])
    sup_mod.Supervisor.handle_requests(fake)
    row = control.get(db, asked["id"])
    assert row["state"] == "done" and "not asked for running jobs (no answer): web" in row["detail"]


def test_a_start_never_waits(tmp_path):
    path = tmp_path / "accounts.db"
    db = _db(path)
    busy = _Inst("web", [{"kind": "marble_photo", "digest": "ff"}])
    fake, _ = _supervisor(path, [busy])
    made = control.request(db, instance="web", action="start", user_id=1, username="admin", known=["web"])
    sup_mod.Supervisor.handle_requests(fake)
    assert control.get(db, made["id"])["state"] == "done" and busy.done == ["start"]


class _Process:
    def __init__(self, code):
        self.code = code

    def poll(self):
        return self.code


def test_the_server_reports_its_running_exposure_only_on_the_admin_op(tmp_path):
    photos = PhotoJobs(tmp_path / "photos", tmp_path)
    try:
        assert photos.running() == []
        photos.active_digest, photos.process = "a" * 64, _Process(None)
        assert photos.running() == [{"kind": "marble_photo", "digest": "a" * 64}]
        lc = lifecycle_mod.Lifecycle(SimpleNamespace(marble_photos=photos, tables={}), tmp_path / "state", 8799, lambda: None)
        assert lc.request_json({"op": "busy"}) == {"jobs": [{"kind": "marble_photo", "digest": "a" * 64}]}
        assert "jobs" not in lc.status()                # the public status every page reads says nothing of jobs
        photos.process = _Process(0)                    # the exposure finished
        assert lc.request_json({"op": "busy"}) == {"jobs": []}
    finally:
        photos._sweep_stop.set()
