"""examples/meowdoku/deploy/apache_supervisor.py: the piped-CustomLog supervisor, over a temp accounts.db and a stand-in child."""
from __future__ import annotations

import json
import os
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


def test_shipped_instances_dry_run_lists_all():
    out = subprocess.run([sys.executable, str(SUP), "--once", "--dry-run"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    names = [i["name"] for i in json.loads(out.stdout)["instances"]]
    assert names == ["meowdoku-a", "meowdoku-dev", "nodus-messenger", "nodus-ui-mirror", "nodus-training", "nodus-training-ui-mirror", "task-worker", "udp-router", "mail", "relay"]


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


def test_instance_env_reaches_only_its_child(tmp_path):
    seen = (tmp_path / "seen.json").as_posix()
    code = (f"import json,os,time\njson.dump({{k: os.environ.get(k) for k in ('NODUS_EXEC_ROWS', 'OTHER')}}, open({seen!r}, 'w'))\n"
            "time.sleep(120)")
    spec = dict(_spec(code), env={"NODUS_EXEC_ROWS": "C:/declared/rows.txt", "OTHER": 7})
    db, state, inst = _setup(tmp_path, spec)
    proc = subprocess.Popen(_cmd(db, state, inst), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        got = _wait(lambda: json.loads(Path(seen).read_text(encoding="utf-8")) if Path(seen).exists() else None)
        assert got == {"NODUS_EXEC_ROWS": "C:/declared/rows.txt", "OTHER": "7"}
        proc.stdin.close()
        assert proc.wait(timeout=45) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close(); proc.stderr.close()


def test_instance_python_field_selects_interpreter(tmp_path):
    spec = {"name": "pyinst", "kind": "meowdoku", "python": sys.executable, "port": 8999, "route": "", "bind": "127.0.0.1",
            "account": "svc-pyinst", "args": [], "leases": []}
    db, state, inst = _setup(tmp_path, spec)
    out = subprocess.run(_cmd(db, state, inst, "--once", "--dry-run"), capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    command = json.loads(out.stdout)["instances"][0]["command"]
    assert command[0] == sys.executable


_NODUS_STUB = """import json, os, sys, time
open(sys.argv[0] + '.argv', 'w').write(json.dumps(sys.argv[1:]))
tok = sys.argv[sys.argv.index('--console-token-file') + 1]
open(sys.argv[0] + '.tokseen', 'w').write(str(os.path.exists(tok)))
print(json.dumps({"canvas": sys.argv[sys.argv.index('--canvas') + 1], "contract_ports": [], "console_port": 47699, "pid": os.getpid()}), flush=True)
time.sleep(120)
"""


def test_nodus_console_token_port_and_cleanup(tmp_path):
    tool = tmp_path / "stub_canvas.py"
    tool.write_text(_NODUS_STUB, encoding="utf-8")
    canvas = tmp_path / "x.canvas"
    canvas.write_text("canvas", encoding="utf-8")
    spec = {"name": "nodus-stub", "kind": "nodus", "port": None, "route": "", "bind": "127.0.0.1", "account": "svc-nodus-stub",
            "args": [], "leases": [], "python": sys.executable,
            "nodus": {"tool": str(tool), "canvas": str(canvas), "acquire": [], "rate": 1, "console_port": 0}}
    db, state, inst = _setup(tmp_path, spec)
    token = state / "run" / "console" / "nodus-stub.console.secret"      # --state-dir is the state root
    plan = subprocess.run(_cmd(db, state, inst, "--once", "--dry-run"), capture_output=True, text=True, timeout=60)
    command = json.loads(plan.stdout)["instances"][0]["command"]
    assert command[command.index("--console-port") + 1] == "0" and command[command.index("--console-token-file") + 1] == str(token)
    proc = subprocess.Popen(_cmd(db, state, inst), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        row = _wait(lambda: next((r for r in _rows(db, "SELECT * FROM service_instances") if r["state"] == "running"), None))
        assert row["console_port"] == 47699
        assert token.exists() and len(token.read_text(encoding="utf-8")) >= 32
        argv = json.loads(Path(str(tool) + ".argv").read_text(encoding="utf-8"))
        assert "--console-port" in argv and argv[argv.index("--console-token-file") + 1] == str(token)
        assert Path(str(tool) + ".tokseen").read_text(encoding="utf-8") == "True"   # the file existed before the host started
        status = subprocess.run(_cmd(db, state, inst, "--status"), capture_output=True, text=True, timeout=60)
        assert json.loads(status.stdout)[0]["console_port"] == 47699
        proc.stdin.close()
        assert proc.wait(timeout=45) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close(); proc.stderr.close()
    assert not token.exists()
    assert _rows(db, "SELECT console_port, state FROM service_instances")[0] == {"console_port": None, "state": "stopped"}


def test_console_port_column_added_to_old_table(tmp_path):
    db = tmp_path / "accounts.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE service_instances (name TEXT PRIMARY KEY, kind TEXT NOT NULL, port INTEGER, route TEXT, "
                 "account_user_id INTEGER NOT NULL, pid INTEGER, state TEXT NOT NULL, started REAL, stopped REAL, "
                 "restarts INTEGER NOT NULL DEFAULT 0, supervisor_pid INTEGER)")
    conn.commit(); conn.close()
    for _ in range(2):                                    # idempotent
        out = subprocess.run([sys.executable, str(SUP), "--db", str(db), "--status"], capture_output=True, text=True, timeout=60)
        assert out.returncode == 0, out.stderr
    assert "console_port" in {r["name"] for r in _rows(db, "PRAGMA table_info(service_instances)")}


_NODUS_CONTRACT_STUB = """import json, os, sys, time
print(json.dumps({"canvas": sys.argv[sys.argv.index('--canvas') + 1],
                  "contract_ports": [{"prefix": "py.stub", "capacity": 4, "listen_port": 47777}], "pid": os.getpid()}), flush=True)
time.sleep(120)
"""

_ARGV_RECORDER = "import json, sys, time\nopen(sys.argv[1], 'w').write(json.dumps(sys.argv[2:]))\nprint('up', flush=True)\ntime.sleep(120)\n"


def test_federation_from_hands_contract_port_and_canvas_to_dependent(tmp_path):
    tool = tmp_path / "stub_contract_canvas.py"
    tool.write_text(_NODUS_CONTRACT_STUB, encoding="utf-8")
    canvas = tmp_path / "x.canvas"
    canvas.write_text("canvas", encoding="utf-8")
    recorder, record = tmp_path / "recorder.py", tmp_path / "argv.json"
    recorder.write_text(_ARGV_RECORDER, encoding="utf-8")
    nodus = {"name": "nodus-stub", "kind": "nodus", "port": None, "route": "", "bind": "127.0.0.1", "account": "svc-nodus-stub",
             "args": [], "leases": [], "python": sys.executable,
             "nodus": {"tool": str(tool), "canvas": str(canvas), "acquire": [], "rate": 1}}
    dependent = {"name": "meow-stub", "kind": "command", "port": None, "route": "", "bind": "127.0.0.1", "account": "svc-meow-stub",
                 "args": [sys.executable, str(recorder), str(record)], "leases": [], "federation_from": "nodus-stub",
                 "mail_prefix": "py.stub"}
    db, state, inst = tmp_path / "accounts.db", tmp_path / "state", tmp_path / "instances.json"
    inst.write_text(json.dumps([dependent, nodus]), encoding="utf-8")      # dependent listed first: the supervisor starts the source first
    proc = subprocess.Popen(_cmd(db, state, inst), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        _wait(lambda: len([r for r in _rows(db, "SELECT * FROM service_instances") if r["state"] == "running"]) == 2, timeout=60)
        argv = json.loads(record.read_text(encoding="utf-8"))
        assert argv[argv.index("--federation-port") + 1] == "47777"
        assert argv[argv.index("--federation-canvas") + 1] == str(canvas)
        assert argv[argv.index("--mail-port") + 1] == "47777"            # [MAIL] the declared mail table's port
        row = _rows(db, "SELECT contract_prefix, contract_port FROM service_instances WHERE name = 'nodus-stub'")[0]
        assert row == {"contract_prefix": "py.stub", "contract_port": 47777}
        status = subprocess.run(_cmd(db, state, inst, "--status"), capture_output=True, text=True, timeout=60)
        assert {r["name"]: r["contract_port"] for r in json.loads(status.stdout)}["nodus-stub"] == 47777
        proc.stdin.close()
        assert proc.wait(timeout=45) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close(); proc.stderr.close()
    assert _rows(db, "SELECT contract_port FROM service_instances WHERE name = 'nodus-stub'")[0]["contract_port"] is None


def _dev_pair(tmp_path, dev_enabled=True):
    def spec(name, route, site, **extra):
        return {"name": name, "kind": "meowdoku", "port": 59001 if site == "stable" else 59002, "route": route, "site": site, "bind": "127.0.0.1",
                "account": f"svc-{name}", "args": [], "leases": [], **extra}
    inst = tmp_path / "instances.json"
    inst.write_text(json.dumps({"state_dir": str(tmp_path / "root"), "instances": [
        spec("m-a", "a", "stable"), spec("m-dev", "b", "dev", state_dir=str(tmp_path / "devroot"), enabled=dev_enabled)]}), encoding="utf-8")
    return inst


def _plan(tmp_path, inst):
    out = subprocess.run([sys.executable, str(SUP), "--db", str(tmp_path / "a.db"), "--instances", str(inst), "--once", "--dry-run"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return {i["name"]: i for i in json.loads(out.stdout)["instances"]}


def test_per_instance_state_dir_reaches_only_that_child_and_routes_are_passed(tmp_path):
    plan = _plan(tmp_path, _dev_pair(tmp_path))
    a, dev = plan["m-a"]["command"], plan["m-dev"]["command"]
    assert a[a.index("--state-dir") + 1] == str(tmp_path / "root")
    assert dev[dev.index("--state-dir") + 1] == str(tmp_path / "devroot")
    for cmd in (a, dev):
        assert cmd[cmd.index("--routes") + 1] == "a,b"
        assert cmd[cmd.index("--stable-route") + 1] == "a" and cmd[cmd.index("--dev-route") + 1] == "b"
    assert not (tmp_path / "devroot").exists() and not (tmp_path / "a.db").exists()          # dry-run creates nothing


def test_disabled_instance_is_skipped_and_contributes_no_route(tmp_path):
    plan = _plan(tmp_path, _dev_pair(tmp_path, dev_enabled=False))
    assert plan["m-dev"]["skipped"] == "disabled (enabled: false)" and plan["m-dev"]["command"] is None
    a = plan["m-a"]["command"]
    assert a[a.index("--routes") + 1] == "a" and "--dev-route" not in a


def _alive(pid) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, timeout=30).stdout
        return str(pid) in out.split()
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_instances_end_when_the_supervisor_is_killed(tmp_path):
    """A supervisor ended abruptly (no shutdown, as a service stop does) takes its instances with it: they are in its
    kill-on-close job, so nothing is left holding their ports for the next supervisor."""
    spec = {"name": "sleeper", "kind": "command", "port": None, "route": "", "bind": "127.0.0.1", "account": "svc-sleeper",
            "args": [sys.executable, "-c", "import time\nprint('up', flush=True)\ntime.sleep(120)"], "leases": []}
    db, state, inst = _setup(tmp_path, spec)
    proc = subprocess.Popen(_cmd(db, state, inst), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        row = _wait(lambda: next((r for r in _rows(db, "SELECT * FROM service_instances") if r["state"] == "running" and r["pid"]), None))
        child = int(row["pid"])
        assert _alive(child)
        proc.kill()                                   # TerminateProcess: the supervisor's shutdown never runs
        proc.wait(timeout=30)
        if os.name == "nt":
            assert _wait(lambda: not _alive(child), timeout=20), "the instance outlived its supervisor"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close(); proc.stderr.close()
