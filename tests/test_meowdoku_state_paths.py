"""examples/meowdoku/state_paths.py and the --state-dir plumbing: the legacy layout equals today's literal paths, the new layout is one
root, and a server / supervisor given a root writes there (temp dirs only; no live state is touched)."""
from __future__ import annotations

import http.client
import json
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
SUP = ROOT / "examples" / "meowdoku" / "deploy" / "apache_supervisor.py"

from meowdoku import lifecycle as lifecycle_mod  # noqa: E402
from meowdoku import server as srv  # noqa: E402
from meowdoku.state_paths import StatePaths  # noqa: E402


def test_legacy_maps_to_todays_literal_paths(tmp_path):
    b = tmp_path / "build" / "meowdoku"
    p = StatePaths.legacy(tmp_path)
    assert p.is_legacy and p.root == b
    assert p.accounts_db == b / "accounts.db" and p.access_json == b / "access.json"
    assert p.sql_dir == p.content_dir == p.products_dir == p.secrets_dir == p.run_dir == b
    assert p.profiles_json == b / "profiles.json" and p.tls_dir == b / "tls"
    assert p.admin_token() == b / "admin_token.txt" and p.admin_token(8766) == b / "admin_token.txt"
    assert p.admin_token(8767) == b / "admin_token_8767.txt"
    assert p.operator_credentials_dir == b / "operator_credentials"
    assert p.logs_dir == b / "logs" and p.canvas_dir == b / "canvas" and p.log_ring == b / "log_ring.jsonl"
    assert p.job_library_db == b / "marble_photos" / "job-library.sqlite"
    assert (p.marble_objects_dir, p.marble_photos_dir, p.obj_assets_dir, p.shares_dir, p.tickets_dir) == (
        b / "marble_objects", b / "marble_photos", b / "obj_assets", b / "shares", b / "tickets")
    for port in (8766, 8767):       # the same three files as the plain-directory form
        assert lifecycle_mod.state_paths(p, port) == lifecycle_mod.state_paths(b, port)
    assert lifecycle_mod.STATE_DIR == lifecycle_mod.ROOT / "build" / "meowdoku"


def test_new_layout_paths(tmp_path):
    r = tmp_path / "state"
    p = StatePaths(r)
    assert not p.is_legacy
    assert (p.sql_dir, p.content_dir, p.products_dir, p.canvas_dir, p.logs_dir, p.secrets_dir, p.run_dir) == tuple(
        r / d for d in ("sql", "content", "products", "canvas", "logs", "secrets", "run"))
    assert p.accounts_db == r / "sql" / "accounts.db" and p.job_library_db == r / "sql" / "job-library.sqlite"
    assert p.profiles_json == r / "content" / "profiles.json"
    assert p.admin_token() == r / "secrets" / "admin_token.txt" and p.admin_token(9000) == r / "secrets" / "admin_token_9000.txt"
    assert p.operator_credentials_dir == r / "secrets" / "operator_credentials" and p.tls_dir == r / "secrets" / "tls"
    assert p.log_ring == r / "logs" / "log_ring.jsonl"
    assert lifecycle_mod.state_paths(p, 9000) == {"sessions": r / "run" / "sessions_9000", "token": r / "secrets" / "admin_token_9000.txt",
                                                  "launch": r / "run" / "launch_9000.json"}
    assert StatePaths.coerce(r).is_legacy and StatePaths.coerce(p) is p        # a plain directory keeps its old meaning


def test_ensure_creates_dirs_and_restricts_secrets(tmp_path, monkeypatch):
    sys.path.append(str(ROOT / "examples" / "meowdoku" / "deploy"))
    import operators
    seen = []
    monkeypatch.setattr(operators, "_restrict", lambda d: seen.append(Path(d)) or f"{d}: restricted")
    p = StatePaths(tmp_path / "state")
    report = p.ensure()
    for d in (p.sql_dir, p.content_dir, p.products_dir, p.canvas_dir, p.logs_dir, p.secrets_dir, p.run_dir, p.tls_dir, p.operator_credentials_dir, p.service_tokens_dir):
        assert d.is_dir(), d
    assert seen == [p.secrets_dir] and len(report) == 1
    legacy = StatePaths.legacy(tmp_path / "repo")
    assert legacy.ensure() == [] and legacy.root.is_dir() and not (legacy.root / "sql").exists()


def test_resolve_state():
    web = Path("/x/build/meowdoku/web")
    assert srv.resolve_state(None, web).is_legacy and srv.resolve_state(None, web).root == Path("/x/build/meowdoku")
    assert srv.resolve_state("/s", web).accounts_db == Path("/s/sql/accounts.db")


def test_server_with_state_dir_keeps_accounts_under_the_root(tmp_path):
    web = tmp_path / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    state = StatePaths(tmp_path / "root")
    state.ensure(restrict=False)
    sessions = srv.Sessions(web, None, state)
    sessions.lifecycle = SimpleNamespace(token="t" * 40, restore_for=lambda token: None)
    sessions.access.lookup = lambda url: None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=30)
        conn.request("GET", "/api/whoami")
        r = conn.getresponse()
        who = json.loads(r.read())
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert r.status == 200 and who["role"] == "guest"
    assert (tmp_path / "root" / "sql" / "accounts.db").exists()
    assert not (tmp_path / "build" / "meowdoku" / "accounts.db").exists()               # nothing beside web/


_NODUS_STUB = """import json, os, sys, time
print(json.dumps({"contract_ports": [], "console_port": 47699, "pid": os.getpid()}), flush=True)
time.sleep(120)
"""


def _wait(pred, timeout=40.0):
    end = time.time() + timeout
    while time.time() < end:
        got = pred()
        if got:
            return got
        time.sleep(0.25)
    raise AssertionError("timed out waiting")


def test_supervisor_state_dir_in_instances_json(tmp_path):
    tool = tmp_path / "stub_canvas.py"
    tool.write_text(_NODUS_STUB, encoding="utf-8")
    canvas = tmp_path / "x.canvas"
    canvas.write_text("canvas", encoding="utf-8")
    root = tmp_path / "root"
    spec = {"name": "nodus-stub", "kind": "nodus", "port": None, "route": "", "bind": "127.0.0.1", "account": "svc-nodus-stub",
            "args": [], "leases": [], "python": sys.executable,
            "nodus": {"tool": str(tool), "canvas": str(canvas), "acquire": [], "rate": 1, "console_port": 0}}
    inst = tmp_path / "instances.json"
    inst.write_text(json.dumps({"state_dir": str(root), "instances": [spec]}), encoding="utf-8")
    token = root / "secrets" / "service_tokens" / "nodus-stub.console.secret"
    cmd = [sys.executable, str(SUP), "--instances", str(inst)]                         # no --db, no --state-dir: both come from the file
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.stdin.write(b'127.0.0.1 - - [x] "GET / HTTP/1.1" 200 1\n')
        proc.stdin.flush()
        _wait(lambda: token.exists())
        assert len(token.read_text(encoding="utf-8")) >= 32
        access = root / "logs" / "apache-access.log"
        assert _wait(lambda: access.exists() and b"GET /" in access.read_bytes())
        assert list((root / "logs").glob("nodus-stub-*.log")) and (root / "logs" / "apache-supervisor.log").exists()
        assert (root / "run" / "supervisor.lock").exists() and (root / "sql" / "accounts.db").exists()
        status = subprocess.run([*cmd, "--status"], capture_output=True, text=True, timeout=60)         # --status reads the same db
        assert status.returncode == 0 and json.loads(status.stdout)[0]["name"] == "nodus-stub"
        proc.stdin.close()
        assert proc.wait(timeout=45) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close(); proc.stderr.close()
    assert not token.exists()


def test_supervisor_passes_state_dir_to_meowdoku_children(tmp_path):
    inst = tmp_path / "instances.json"
    spec = {"name": "m", "kind": "meowdoku", "python": sys.executable, "port": 8999, "route": "", "bind": "127.0.0.1", "account": "svc-m",
            "args": [], "leases": []}
    inst.write_text(json.dumps({"state_dir": str(tmp_path / "root"), "instances": [spec]}), encoding="utf-8")
    out = subprocess.run([sys.executable, str(SUP), "--instances", str(inst), "--once", "--dry-run"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    plan = json.loads(out.stdout)
    command = plan["instances"][0]["command"]
    assert command[command.index("--state-dir") + 1] == str(tmp_path / "root")
    assert plan["db"] == str(tmp_path / "root" / "sql" / "accounts.db")
    assert not (tmp_path / "root").exists()                                            # a dry run creates nothing
    # The shipped instances.json carries state_dir since the 2026-11-10 cutover, so the legacy check uses its own file.
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"instances": [spec]}), encoding="utf-8")
    legacy = subprocess.run([sys.executable, str(SUP), "--instances", str(bare), "--once", "--dry-run"], capture_output=True, text=True, timeout=60)
    assert "--state-dir" not in next(i for i in json.loads(legacy.stdout)["instances"] if i["kind"] == "meowdoku")["command"]
