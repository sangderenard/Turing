"""examples/meowdoku/deploy/apache_supervisor.py: the piped-CustomLog supervisor, over a temp accounts.db and a stand-in child."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
SUP = ROOT / "examples" / "meowdoku" / "deploy" / "apache_supervisor.py"

from meowdoku import lowering_library as lib  # noqa: E402


def _spec(child_code, **extra):
    return {"name": "standin", "kind": "command", "port": None, "route": "", "bind": "127.0.0.1",
            "account": "svc-standin", "args": [sys.executable, "-c", child_code], "leases": extra.get("leases", [])}


def _setup(tmp_path, spec):
    db, state, inst = tmp_path / "accounts.db", tmp_path / "state", tmp_path / "instances.json"
    inst.write_text(json.dumps([spec]), encoding="utf-8")
    return db, state, inst


def _cmd(db, state, inst, *more):
    return [sys.executable, str(SUP), "--db", str(db), "--instances", str(inst), "--state-dir", str(state), *more]


def _rows(db, sql, args=()):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def _wait(pred, timeout=40.0):
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


def _library(db):
    product = Path(db).parent / "tiny.c"
    product.write_text("int f(void){return 1;}\n", encoding="utf-8")
    sha = lib.store_blob(db, product)
    lib.register(db, lib.ProductRecord(core_id="core-x", family="fam", name="tiny", backend="c", entry="f", api_path="x.api",
                                       verdict="built", product_path=str(product), product_sha256=sha))
    return sha


def test_dry_run_prints_plan_and_changes_nothing(tmp_path):
    db, state, inst = _setup(tmp_path, _spec("pass"))
    out = subprocess.run(_cmd(db, state, inst, "--once", "--dry-run"), capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    plan = json.loads(out.stdout)
    assert plan["instances"][0]["name"] == "standin" and plan["instances"][0]["account"] == "svc-standin"
    assert plan["instances"][0]["command"][1] == "-c"
    assert not db.exists()


def test_shipped_instances_dry_run_lists_both():
    out = subprocess.run([sys.executable, str(SUP), "--once", "--dry-run"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    names = [i["name"] for i in json.loads(out.stdout)["instances"]]
    assert names == ["meowdoku-a", "nodus-messenger"]


def test_lifecycle_account_lease_stdin_eof(tmp_path):
    db, state, inst = _setup(tmp_path, _spec("print('hello', flush=True)\nimport time\ntime.sleep(120)",
                                             leases=[{"purpose": ["fam", "tiny"], "capabilities": ["c"], "prefer": []}]))
    sha = _library(db)
    proc = subprocess.Popen(_cmd(db, state, inst), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.stdin.write(b'127.0.0.1 - - [x] "GET / HTTP/1.1" 200 1\n')
        proc.stdin.flush()
        row = _wait(lambda: next((r for r in _rows(db, "SELECT * FROM service_instances") if r["state"] == "running"), None))
        assert row["pid"] and row["kind"] == "command" and row["supervisor_pid"] == proc.pid
        user = _rows(db, "SELECT u.id, r.role FROM users u JOIN account_roles r ON r.user_id = u.id WHERE u.username = 'svc-standin'")
        assert user and user[0]["role"] == "service" and row["account_user_id"] == user[0]["id"]
        assert _rows(db, "SELECT instance FROM account_instances WHERE user_id = ?", (user[0]["id"],))[0]["instance"] == "standin"
        leases = _rows(db, "SELECT * FROM instance_leases WHERE name = 'standin'")
        assert len(leases) == 1 and leases[0]["product_sha256"] == sha
        open_leases = _rows(db, "SELECT account_key, returned FROM lowering_leases")
        assert open_leases == [{"account_key": f"account:{user[0]['id']}", "returned": None}]
        logs = list((state / "logs").glob("standin-*.log"))
        assert logs and "hello" in logs[0].read_text(encoding="utf-8")
        assert _wait(lambda: b"GET /" in (state / "logs" / "apache-access.log").read_bytes() if (state / "logs" / "apache-access.log").exists() else None)
        proc.stdin.close()                                   # what Apache does at shutdown
        assert proc.wait(timeout=45) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close(); proc.stderr.close()
    final = _rows(db, "SELECT * FROM service_instances")[0]
    assert final["state"] == "stopped" and final["pid"] is None and final["stopped"]
    assert _rows(db, "SELECT returned FROM lowering_leases")[0]["returned"] is not None
    status = subprocess.run(_cmd(db, state, inst, "--status"), capture_output=True, text=True, timeout=60)
    assert json.loads(status.stdout)[0]["state"] == "stopped"


def test_restart_on_crash_once(tmp_path):
    marker = (tmp_path / "ran-once").as_posix()
    code = (f"import os,sys,time\nm={marker!r}\n"
            "if not os.path.exists(m):\n    open(m,'w').close(); sys.exit(3)\n"
            "print('second run', flush=True); time.sleep(120)")
    db, state, inst = _setup(tmp_path, _spec(code))
    proc = subprocess.Popen(_cmd(db, state, inst), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        row = _wait(lambda: next((r for r in _rows(db, "SELECT * FROM service_instances")
                                  if r["state"] == "running" and r["restarts"] >= 1), None), timeout=45)
        assert row["restarts"] == 1 and row["pid"]
        proc.stdin.close()
        assert proc.wait(timeout=45) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close(); proc.stderr.close()
    assert _rows(db, "SELECT state FROM service_instances")[0]["state"] == "stopped"


def test_instance_python_field_selects_interpreter(tmp_path):
    spec = {"name": "pyinst", "kind": "meowdoku", "python": sys.executable, "port": 8999, "route": "", "bind": "127.0.0.1",
            "account": "svc-pyinst", "args": [], "leases": []}
    db, state, inst = _setup(tmp_path, spec)
    out = subprocess.run(_cmd(db, state, inst, "--once", "--dry-run"), capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    command = json.loads(out.stdout)["instances"][0]["command"]
    assert command[0] == sys.executable
