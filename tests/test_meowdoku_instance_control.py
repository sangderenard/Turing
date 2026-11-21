"""An admin's forced shutdown, restart and start: the SQL row, and a real supervisor performing it on a stand-in."""
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import instance_control as control  # noqa: E402

SUP = ROOT / "examples" / "meowdoku" / "deploy" / "apache_supervisor.py"


def db_at(path):
    db = sqlite3.connect(path, timeout=10, isolation_level=None)
    control.ensure(db)
    return db


def test_a_request_is_one_row_and_the_same_request_is_not_taken_twice(tmp_path):
    db = db_at(tmp_path / "a.db")
    made = control.request(db, instance="web", action="restart", user_id=3, username="ann", reason="hung", known=["web", "worker"])
    assert (made["state"], made["requested_by"], made["requested_name"], made["reason"]) == ("pending", 3, "ann", "hung")
    with pytest.raises(control.ControlError) as twice:
        control.request(db, instance="web", action="restart", user_id=4, username="bo", known=["web"])
    assert twice.value.status == 409
    other = control.request(db, instance="*", action="shutdown", user_id=3, username="ann", known=["web"])
    assert [r["id"] for r in control.pending(db)] == [made["id"], other["id"]]
    control.finish(db, made["id"], "done", "web: pid 1 stopped; web: started pid 2")
    assert [r["id"] for r in control.pending(db)] == [other["id"]]
    assert control.get(db, made["id"])["detail"].startswith("web: pid 1 stopped")
    assert control.request(db, instance="web", action="restart", user_id=3, username="ann", known=["web"])["state"] == "pending"
    assert [r["id"] for r in control.recent(db)][0] > other["id"]


@pytest.mark.parametrize("fields, status", [
    ({"instance": "nope", "action": "restart"}, 404),
    ({"instance": "web", "action": "explode"}, 400),
    ({"instance": "web", "action": "restart", "user_id": None}, 401),
    ({"instance": "web", "action": "restart", "reason": "x" * 400}, 400),
    ({"instance": 7, "action": "restart"}, 404),
])
def test_a_bad_request_is_refused_and_writes_nothing(tmp_path, fields, status):
    db = db_at(tmp_path / "a.db")
    given = {"instance": "web", "action": "restart", "user_id": 3, "username": "ann", "known": ["web"], **fields}
    with pytest.raises(control.ControlError) as refused:
        control.request(db, **given)
    assert refused.value.status == status and control.pending(db) == []


def _rows(db, sql):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql)]
    finally:
        conn.close()


def _wait(pred, timeout=45.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            got = pred()
        except sqlite3.Error:
            got = None
        if got:
            return got
        time.sleep(0.25)
    raise AssertionError("timed out waiting")


def _ask(db, **fields):
    conn = sqlite3.connect(db, timeout=10, isolation_level=None)
    try:
        return control.request(conn, user_id=1, username="admin", known=["standin", "deaf"], **fields)["id"]
    finally:
        conn.close()


def test_the_supervisor_restarts_shuts_down_and_starts_on_request_even_a_process_that_ignores_the_ask(tmp_path):
    polite = "import time\nprint('up', flush=True)\ntime.sleep(600)"
    deaf = ("import signal, time\n"
            "for name in ('SIGBREAK', 'SIGTERM', 'SIGINT'):\n"
            "    if hasattr(signal, name): signal.signal(getattr(signal, name), signal.SIG_IGN)\n"
            "print('up', flush=True)\n"
            "while True: time.sleep(1)")
    specs = [{"name": name, "kind": "command", "port": None, "route": "", "bind": "127.0.0.1", "account": f"svc-{name}",
              "args": [sys.executable, "-c", code], "leases": []} for name, code in (("standin", polite), ("deaf", deaf))]
    db, state, inst = tmp_path / "accounts.db", tmp_path / "state", tmp_path / "instances.json"
    inst.write_text(json.dumps(specs), encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(SUP), "--db", str(db), "--instances", str(inst), "--state-dir", str(state)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def instance(name):
        return next((r for r in _rows(db, "SELECT * FROM service_instances") if r["name"] == name), None)

    def handled(request_id):
        return next((r for r in _rows(db, "SELECT * FROM instance_requests") if r["id"] == request_id and r["state"] != "pending"), None)
    try:
        first = _wait(lambda: instance("standin") if (instance("standin") or {}).get("state") == "running" else None)
        _wait(lambda: (instance("deaf") or {}).get("state") == "running")
        # restart: a new process
        done = _wait(lambda: handled(_RESTART)) if (_RESTART := _ask(db, instance="standin", action="restart")) else None
        assert done["state"] == "done" and "stopped" in done["detail"] and "started pid" in done["detail"]
        again = _wait(lambda: instance("standin") if (instance("standin") or {}).get("state") == "running"
                      and instance("standin")["pid"] != first["pid"] else None)
        assert again["pid"] != first["pid"]
        # shutdown: it stays down; the supervisor does not bring it back
        request_id = _ask(db, instance="standin", action="shutdown")
        assert _wait(lambda: handled(request_id))["state"] == "done"
        time.sleep(3)
        assert instance("standin")["state"] == "stopped"
        # start
        request_id = _ask(db, instance="standin", action="start")
        assert "started pid" in _wait(lambda: handled(request_id))["detail"]
        _wait(lambda: (instance("standin") or {}).get("state") == "running")
        # forced: a process that ignores the request to stop is killed
        deaf_pid = instance("deaf")["pid"]
        request_id = _ask(db, instance="deaf", action="shutdown")
        forced = _wait(lambda: handled(request_id), timeout=60)
        assert forced["state"] == "done" and "was killed" in forced["detail"], forced
        assert instance("deaf")["state"] == "stopped" and instance("deaf")["pid"] in (None, deaf_pid)
        # everything at once
        request_id = _ask(db, instance="*", action="restart")
        everything = _wait(lambda: handled(request_id), timeout=60)
        assert "standin:" in everything["detail"] and "deaf:" in everything["detail"]
        _wait(lambda: (instance("deaf") or {}).get("state") == "running" and instance("deaf")["pid"] != deaf_pid)
        proc.stdin.close()
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.stderr.close()


def test_an_instance_off_at_start_waits_for_a_start_by_name_and_the_whole_stack_leaves_it_stopped(tmp_path):
    polite = "import time\nprint('up', flush=True)\ntime.sleep(600)"
    specs = [{"name": name, "kind": "command", "port": None, "route": "", "bind": "127.0.0.1", "account": f"svc-{name}",
              "args": [sys.executable, "-c", polite], "leases": [], **extra}
             for name, extra in (("standin", {}), ("deaf", {"autostart": False}))]
    db, state, inst = tmp_path / "accounts.db", tmp_path / "state", tmp_path / "instances.json"
    inst.write_text(json.dumps(specs), encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(SUP), "--db", str(db), "--instances", str(inst), "--state-dir", str(state)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def instance(name):
        return next((r for r in _rows(db, "SELECT * FROM service_instances") if r["name"] == name), None)

    def handled(request_id):
        return next((r for r in _rows(db, "SELECT * FROM instance_requests") if r["id"] == request_id and r["state"] != "pending"), None)
    try:
        _wait(lambda: (instance("standin") or {}).get("state") == "running")
        assert instance("deaf")["state"] == "stopped" and instance("deaf")["pid"] is None, "off at start"
        asked = _ask(db, instance="*", action="restart")
        everything = _wait(lambda: handled(asked), timeout=60)
        assert "deaf: left stopped (autostart: false)" in everything["detail"] and "standin:" in everything["detail"]
        assert instance("deaf")["state"] == "stopped"
        asked = _ask(db, instance="deaf", action="start")
        started = _wait(lambda: handled(asked))
        assert "started pid" in started["detail"]
        first = _wait(lambda: instance("deaf") if (instance("deaf") or {}).get("state") == "running" else None)
        asked = _ask(db, instance="*", action="restart")
        again = _wait(lambda: handled(asked), timeout=60)
        assert "deaf: started pid" in again["detail"], "running when the whole stack restarts: it comes back"
        _wait(lambda: (instance("deaf") or {}).get("state") == "running" and instance("deaf")["pid"] != first["pid"])
        proc.stdin.close()
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.stderr.close()
