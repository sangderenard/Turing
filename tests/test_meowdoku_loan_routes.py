"""[LOAN][STABLE] The server routes: the dev menu's loan list, an owner's restart of their own loan, the HTTPS-only sign-off."""
from __future__ import annotations

import http.client
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import access as A  # noqa: E402
from meowdoku import authz, branch_servers, instance_control  # noqa: E402
from meowdoku import server as srv  # noqa: E402


def _server(tmp_path):
    web = Path(tmp_path) / "build" / "meowdoku" / "web"
    web.mkdir(parents=True, exist_ok=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    sessions = srv.Sessions(web, None)
    sessions.access = A.Access(None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    cookies, ids = {}, {}
    for name, group in (("ann", "dev"), ("bob", "dev"), ("max", None)):
        uid = sessions.accounts.ensure_user(name, "correcthorse1")
        if group:
            conn = authz._connect(sessions.accounts_db)
            conn.execute("INSERT INTO account_permission_grants(user_id,group_name,granted_by,created_at) VALUES(?,?,?,?)",
                         (uid, group, 0, 0.0))
            conn.commit(); conn.close()
        ids[name] = uid
        cookies[name] = {"Cookie": f"{srv.ACCOUNT_COOKIE}={sessions.accounts.create_session(uid)}"}
    return httpd, sessions, cookies, ids


def _call(httpd, method, path, headers, body=None):
    c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=10)
    data = None if body is None else json.dumps(body).encode()
    c.request(method, path, body=data, headers={**headers, **({"Content-Type": "application/json"} if data else {})})
    r = c.getresponse()
    return r.status, json.loads(r.read() or b"{}")


def test_loan_routes(tmp_path):
    httpd, sessions, cookies, ids = _server(tmp_path)
    try:
        import sqlite3
        db = sqlite3.connect(sessions.accounts_db, isolation_level=None)
        branch_servers.ensure(db)
        db.execute("INSERT INTO checkouts VALUES ('turing','ann/one',?,'ann','C:/nowhere','main',0)", (ids["ann"],))
        assert branch_servers.lend(db, repository="turing", branch="ann/one", user_id=ids["ann"])["instance"] == "loan-8769"
        status, body = _call(httpd, "GET", "/api/loans", cookies["ann"])
        assert status == 200 and [l["instance"] for l in body["loans"]] == ["loan-8769"] and "path" not in body["loans"][0]
        # (a loopback caller is the local origin, which gives every member dev_ui, so "max" cannot show a refusal here)
        status, body = _call(httpd, "POST", "/api/control/instances", cookies["bob"], {"instance": "loan-8769", "action": "restart"})
        assert status == 403 and "your own" in body["error"]
        status, body = _call(httpd, "POST", "/api/control/instances", cookies["ann"], {"instance": "loan-8769", "action": "shutdown"})
        assert status == 403, "an owner may restart or start, not shut down"
        status, body = _call(httpd, "POST", "/api/control/instances", cookies["ann"], {"instance": "meowdoku-a", "action": "restart"})
        assert status == 403, "an owner's right covers only their own loan"
        status, body = _call(httpd, "POST", "/api/control/instances", cookies["ann"], {"instance": "loan-8769", "action": "restart"})
        assert status == 202 and body["request"]["instance"] == "loan-8769"
        assert _call(httpd, "GET", "/api/control/instances", cookies["ann"])[0] == 403, "the instance list stays for managers"
        assert _call(httpd, "POST", "/api/stable/signoff", cookies["ann"], {"id": 1})[0] == 403
        assert [r["instance"] for r in instance_control.pending(db)] == ["loan-8769"]
        db.close()
    finally:
        httpd.shutdown(); httpd.server_close()
