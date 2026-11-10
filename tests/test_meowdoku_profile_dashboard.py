"""Read-only profile dashboard (examples/meowdoku/profile_dashboard.py, server.py [PROFILE]) over an ephemeral loopback port and temp roots."""
from __future__ import annotations

import http.client
import json
import sqlite3
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import authz  # noqa: E402
from meowdoku import profile_dashboard as pd  # noqa: E402
from meowdoku import server as srv  # noqa: E402

PASSWORD = "correct horse 12"
OUTSIDE = {"X-Forwarded-For": "203.0.113.9", "X-Forwarded-Proto": "https", "Host": "cats.example.org"}


def _call(port, path, method="GET", headers=None, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    hdrs = dict(headers or {})
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=data, headers=hdrs)
    r = conn.getresponse()
    out = r.read()
    status, cookies = r.status, r.msg.get_all("Set-Cookie") or []
    conn.close()
    try:
        return status, cookies, json.loads(out)
    except ValueError:
        return status, cookies, out.decode("utf-8", "replace")


@pytest.fixture()
def server(tmp_path):
    web = Path(tmp_path) / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    sessions = srv.Sessions(web, None)
    sessions.lifecycle = SimpleNamespace(token="t" * 40, restore_for=lambda token: None)
    sessions.access.lookup = lambda url: None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield SimpleNamespace(port=httpd.server_address[1], sessions=sessions, tmp=Path(tmp_path))
    httpd.shutdown()
    httpd.server_close()


def _cookie(server, name, role, headers=None):
    db = server.sessions.accounts_db
    server.sessions.accounts.ensure_user(name, PASSWORD)
    if role == "admin":
        authz.grant_admin(db, name, "operator:test")
    elif role != "member":
        authz.grant_role(db, name, role, "operator:test")
    status, cookies, _ = _call(server.port, "/api/account/login", "POST", headers, {"username": name, "password": PASSWORD})
    assert status == 200
    return {**(headers or {}), "Cookie": next(c.split(";")[0] for c in cookies if c.startswith(srv.ACCOUNT_COOKIE + "="))}


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], check=True, capture_output=True)


@pytest.fixture()
def seeded(server):
    repo = server.tmp / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "feature-x")
    (repo / "a.txt").write_text("a", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "one")
    (repo / "a.txt").write_text("changed", encoding="utf-8")          # dirty
    cfg = server.tmp / "instances.json"
    cfg.write_text(json.dumps({"instances": [{"name": "dev-test", "kind": "meowdoku", "enabled": True, "repo": str(repo),
                                              "state_dir": "C:/secret/place/state-dev"}]}), encoding="utf-8")
    db = sqlite3.connect(server.sessions.accounts_db)
    db.executescript("""CREATE TABLE IF NOT EXISTS service_instances (name TEXT PRIMARY KEY, kind TEXT NOT NULL, port INTEGER, route TEXT,
        account_user_id INTEGER NOT NULL, pid INTEGER, state TEXT NOT NULL, started REAL, stopped REAL, restarts INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS instance_leases (name TEXT NOT NULL, lease_id INTEGER NOT NULL, product_sha256 TEXT NOT NULL,
        generation INTEGER NOT NULL, PRIMARY KEY(name, lease_id));""")
    db.execute("INSERT INTO service_instances (name, kind, port, route, account_user_id, pid, state, started, restarts) VALUES ('dev-test','meowdoku',8768,'b',1,4242,'running',1.0,2)")
    db.execute("INSERT INTO instance_leases VALUES ('dev-test', 7, 'abcdef0123456789', 3)")
    db.commit()
    db.close()
    logs = Path(server.sessions.state.logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "dev-test-20260101.log").write_text("\n".join(["line %d" % i for i in range(300)] + ["login token=abc ok", "password: hunter2"]), encoding="utf-8")
    pd._git_cache.clear()
    return server


def test_member_sees_me_only(server):
    h = _cookie(server, "member1", "member")
    status, _, me = _call(server.port, "/api/profile/me", headers=h)
    assert status == 200 and me["username"] == "member1" and me["roles"] == ["member"] and "play" in me["capabilities"]
    for path in ("/api/profile/instances", "/api/profile/leases", "/api/profile/diagnostics?name=x"):
        assert _call(server.port, path, headers=h)[0] == 403


def test_anonymous_and_guest_refused(server):
    assert _call(server.port, "/api/profile/me")[0] == 403
    assert _call(server.port, "/api/profile/instances")[0] == 403
    assert _call(server.port, "/api/profile/instances", headers=OUTSIDE)[0] == 403
    assert _call(server.port, "/api/profile/me", "POST", body={})[0] in (403, 405)


@pytest.mark.parametrize("role", ["admin", "operator"])
def test_admin_and_operator_get_instances_with_git_state(seeded, role):
    h = _cookie(seeded, role + "1", role)
    rows = pd.instances(sqlite3.connect(seeded.sessions.accounts_db), seeded.tmp / "instances.json", seeded.tmp)
    assert rows[0]["git"]["branch"] == "feature-x" and rows[0]["git"]["dirty"] is True and len(rows[0]["git"]["commit"]) == 8
    assert rows[0]["state_dir"] == "state-dev" and rows[0]["pid"] == 4242 and rows[0]["restarts"] == 2
    status, _, body = _call(seeded.port, "/api/profile/instances", headers=h)
    assert status == 200 and body["instances"][0]["name"] == "dev-test"
    status, _, body = _call(seeded.port, "/api/profile/leases", headers=h)
    assert status == 200 and body["leases"][0]["owner"] == "dev-test" and body["leases"][0]["state"] == "running"


def test_git_failure_is_a_field(tmp_path):
    pd._git_cache.clear()
    state = pd.git_state(tmp_path / "nope")
    assert state["error"] and state["branch"] is None


def test_diagnostics_refusals_scrub_and_cap(seeded):
    h = _cookie(seeded, "admin2", "admin")
    assert _call(seeded.port, "/api/profile/diagnostics?name=nobody", headers=h)[0] == 404
    assert _call(seeded.port, "/api/profile/diagnostics?name=..%2F..%2Fetc%2Fpasswd", headers=h)[0] in (400, 404)
    assert _call(seeded.port, "/api/profile/diagnostics?name=..%5Cx", headers=h)[0] in (400, 404)
    status, _, body = _call(seeded.port, "/api/profile/diagnostics?name=dev-test&lines=9999", headers=h)
    assert status == 200 and len(body["lines"]) <= 200 and body["log"] == "dev-test-20260101.log"
    text = "\n".join(body["lines"])
    assert "abc" not in text and "hunter2" not in text and "token=[scrubbed]" in text
    status, _, body = _call(seeded.port, "/api/profile/diagnostics?name=dev-test&lines=5", headers=h)
    assert len(body["lines"]) == 5
