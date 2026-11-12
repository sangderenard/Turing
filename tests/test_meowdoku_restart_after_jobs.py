"""The restart policy: a due restart or shutdown waits until the server-wide job table (service_jobs) shows no running
job on the instances it names; then the restart account (Pale Horse) authorizes it and is recorded on the row."""
import importlib.util
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import instance_control as control  # noqa: E402
from meowdoku import math_jobs  # noqa: E402
from meowdoku import service_jobs  # noqa: E402
from meowdoku.marble_photo import PhotoJobs  # noqa: E402

_spec = importlib.util.spec_from_file_location("apache_supervisor_under_test", ROOT / "examples/meowdoku/deploy/apache_supervisor.py")
sup_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sup_mod)


def _db(path):
    db = sqlite3.connect(path, timeout=10, isolation_level=None)
    control.ensure(db)
    service_jobs.ensure(db)
    db.execute("CREATE TABLE IF NOT EXISTS service_instances (name TEXT PRIMARY KEY, pid INTEGER)")
    return db


def _instance(db, name, pid):
    db.execute("INSERT OR REPLACE INTO service_instances (name, pid) VALUES (?, ?)", (name, pid))


def test_requests_wait_for_jobs_unless_told_not_to(tmp_path):
    db = _db(tmp_path / "a.db")
    waits = control.request(db, instance="web", action="restart", user_id=1, username="admin", known=["web"])
    forced = control.request(db, instance="web", action="shutdown", user_id=1, username="admin", known=["web"], after_jobs=False)
    assert (waits["after_jobs"], forced["after_jobs"]) == (True, False)
    with pytest.raises(control.ControlError):
        control.request(db, instance="web", action="start", user_id=1, username="admin", known=["web"], after_jobs="yes")
    control.waiting(db, waits["id"], "Pale Horse waits for running jobs: web marble_photo abc")
    held = control.get(db, waits["id"])
    assert held["state"] == "pending" and held["detail"].startswith("Pale Horse waits") and held["authorized_by"] is None


def test_a_table_from_before_the_policy_gains_its_columns_and_keeps_its_rows(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path, isolation_level=None)
    old.execute(control.DDL.replace(", after_jobs INTEGER NOT NULL DEFAULT 1, authorized_by INTEGER", ""))
    old.execute("INSERT INTO instance_requests (instance, action, requested_by, requested_at) VALUES ('web','restart',1,1.0)")
    old.close()
    db = _db(path)
    assert [(r["instance"], r["after_jobs"], r["authorized_by"]) for r in control.pending(db)] == [("web", True, None)]


def test_a_job_counts_only_while_its_process_is_the_instances_process(tmp_path):
    path = tmp_path / "a.db"
    db = _db(path)
    _instance(db, "web", 100)
    jobs = service_jobs.Registry(path, "web", pid=100)
    assert jobs.begin("marble_photo", "d1", label="exposure")
    assert [(j["kind"], j["job_id"]) for j in service_jobs.running(db, ["web"])] == [("marble_photo", "d1")]
    assert service_jobs.running(db, ["worker"]) == []
    _instance(db, "web", 200)                            # the server was restarted: its old process's job is gone
    assert service_jobs.running(db) == [] and service_jobs.listing(db)[0]["live"] is False
    _instance(db, "web", 100)
    jobs.end("marble_photo", "d1")
    assert service_jobs.running(db) == [] and service_jobs.listing(db) == []


def test_a_database_error_never_breaks_the_job(tmp_path, capsys):
    jobs = service_jobs.Registry(tmp_path / "missing" / "a.db", "web")
    assert jobs.begin("math_job", "j1") is False
    assert "a restart will not wait for it" in capsys.readouterr().err


class _Inst:
    def __init__(self, name):
        self.name, self.spec, self.done = name, {}, []

    def force_stop(self):
        self.done.append("stop")
        return "stopped"

    def force_start(self):
        self.done.append("start")
        return "started"


def _supervisor(db_path, instances, account_id=7):
    logs = []
    fake = SimpleNamespace(db_path=str(db_path), instances=instances, log=logs.append, announce=lambda item: None,
                           restart_account_id=account_id, restart_display="Pale Horse")
    return fake, logs


def test_pale_horse_holds_a_restart_while_a_job_runs_and_authorizes_it_after_the_last(tmp_path):
    path = tmp_path / "accounts.db"
    db = _db(path)
    _instance(db, "web", 100)
    _instance(db, "worker", 101)
    web, worker = _Inst("web"), _Inst("worker")
    fake, logs = _supervisor(path, [web, worker])
    photo = service_jobs.Registry(path, "web", pid=100)
    photo.begin("marble_photo", "0123456789abcdef")
    made = control.request(db, instance="*", action="restart", user_id=1, username="admin", known=["web", "worker"])
    sup_mod.Supervisor.handle_requests(fake)
    sup_mod.Supervisor.handle_requests(fake)            # still running: no second log line, nothing touched
    row = control.get(db, made["id"])
    assert row["state"] == "pending" and row["detail"] == "Pale Horse waits for running jobs: web marble_photo 0123456789ab"
    assert web.done == worker.done == [] and len(logs) == 1
    photo.end("marble_photo", "0123456789abcdef")      # the last job ended
    sup_mod.Supervisor.handle_requests(fake)
    row = control.get(db, made["id"])
    assert row["state"] == "done" and row["authorized_by"] == 7
    assert row["detail"].startswith("authorized by Pale Horse: no running jobs")
    assert web.done == ["stop", "start"] and worker.done == ["stop", "start"]


def test_a_job_on_another_instance_does_not_hold_a_restart_and_a_forced_one_never_waits(tmp_path):
    path = tmp_path / "accounts.db"
    db = _db(path)
    _instance(db, "web", 100)
    _instance(db, "dev", 102)
    service_jobs.Registry(path, "dev", pid=102).begin("math_job", "j1")
    web = _Inst("web")
    fake, _ = _supervisor(path, [web, _Inst("dev")])
    other = control.request(db, instance="web", action="restart", user_id=1, username="admin", known=["web", "dev"])
    sup_mod.Supervisor.handle_requests(fake)
    assert control.get(db, other["id"])["state"] == "done"
    forced = control.request(db, instance="dev", action="shutdown", user_id=1, username="admin", known=["web", "dev"],
                             after_jobs=False)
    sup_mod.Supervisor.handle_requests(fake)
    row = control.get(db, forced["id"])
    assert row["state"] == "done" and row["authorized_by"] is None


def test_a_start_never_waits(tmp_path):
    path = tmp_path / "accounts.db"
    db = _db(path)
    _instance(db, "web", 100)
    service_jobs.Registry(path, "web", pid=100).begin("marble_photo", "ff")
    web = _Inst("web")
    fake, _ = _supervisor(path, [web])
    made = control.request(db, instance="web", action="start", user_id=1, username="admin", known=["web"])
    sup_mod.Supervisor.handle_requests(fake)
    assert control.get(db, made["id"])["state"] == "done" and web.done == ["start"]


def test_an_exposure_holds_its_row_until_its_worker_process_exits(tmp_path):
    path = tmp_path / "accounts.db"
    db = _db(path)
    _instance(db, "web", os.getpid())
    photos = PhotoJobs(tmp_path / "photos", tmp_path)
    try:
        photos.registry = service_jobs.Registry(path, "web")
        worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.5)"])
        photos._watch("a" * 64, worker)
        assert [j["job_id"] for j in service_jobs.running(db)] == ["a" * 64]
        worker.wait()
        end = time.time() + 10
        while service_jobs.running(db) and time.time() < end:
            time.sleep(0.05)
        assert service_jobs.running(db) == []
    finally:
        photos._sweep_stop.set()


def test_a_math_job_holds_its_row_while_it_runs(tmp_path):
    path = tmp_path / "accounts.db"
    db = _db(path)
    _instance(db, "web", os.getpid())
    mdb = math_jobs.connect(path)
    mdb.execute("INSERT INTO math_jobs (job_id, owner_key, text, targets, state, created) VALUES ('j1','account:3','x','c','queued',1)")
    seen = []

    def run(text, directory, timeout_s, targets):
        seen.append([j["job_id"] for j in service_jobs.running(db)])
        return {"state": "complete"}
    assert math_jobs.work_one(mdb, SimpleNamespace(jobs_dir=tmp_path / "jobs"), run, service_jobs.Registry(path, "web")) is True
    assert seen == [["j1"]] and service_jobs.running(db) == []
