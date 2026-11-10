"""[ROUTE] Stable server / Development server selection (examples/meowdoku/server.py, authz.py account_prefs): two handlers on
ephemeral ports with different --instance/--route and temp roots.  Outside visitors are simulated as a trusted-proxy peer."""
from __future__ import annotations

import http.client
import json
import sqlite3
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import authz  # noqa: E402
from meowdoku import server as srv  # noqa: E402

OUTSIDE = {"X-Forwarded-For": "203.0.113.9", "X-Forwarded-Proto": "https", "Host": "cats.example.org"}
PASSWORD = "correct horse 12"
ROUTE = "meowdoku_route"


def _call(port, path, method="GET", headers=None, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    hdrs = dict(OUTSIDE, **(headers or {}))
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=hdrs)
    r = conn.getresponse()
    out = r.read()
    cookies = r.msg.get_all("Set-Cookie") or []
    conn.close()
    try:
        return r.status, cookies, json.loads(out)
    except ValueError:
        return r.status, cookies, out.decode("utf-8", "replace")


def _route_cookies(cookies):
    return [c.split(";")[0] for c in cookies if c.startswith(ROUTE + "=")]


def _make(tmp_path, name, **attrs):
    web = Path(tmp_path) / name / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    (web / "login.html").write_text("<html>LOGIN</html>", encoding="utf-8")
    (web / "login.js").write_text("// login", encoding="utf-8")
    (web / "index.html").write_text("<html>GAME</html>", encoding="utf-8")
    sessions = srv.Sessions(web, None)
    sessions.lifecycle = SimpleNamespace(token="t" * 40, restore_for=lambda token: None)
    sessions.access.lookup = lambda url: None
    for key, value in attrs.items():
        setattr(sessions, key, value)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, SimpleNamespace(port=httpd.server_address[1], sessions=sessions)


def _accounts(site):
    for name in ("opcat", "memcat"):
        site.sessions.accounts.ensure_user(name, PASSWORD)
    authz.grant_role(site.sessions.accounts_db, "opcat", "operator", "operator:test")


def _uid(site, name):
    c = sqlite3.connect(site.sessions.accounts_db)
    try:
        return c.execute("SELECT id FROM users WHERE username = ?", (name,)).fetchone()[0]
    finally:
        c.close()


def _login(site, name, extra=None):
    status, cookies, _ = _call(site.port, "/api/account/login", "POST", extra, {"username": name, "password": PASSWORD})
    assert status == 200
    jar = next(c.split(";")[0] for c in cookies if c.startswith(srv.ACCOUNT_COOKIE + "=") and c.split(";")[0] != srv.ACCOUNT_COOKIE + "=")
    return jar, cookies


@pytest.fixture()
def pair(tmp_path):
    common = dict(routes=["a", "b"], stable_route="a", dev_route="b")
    h1, stable = _make(tmp_path, "stable", route="a", instance="meowdoku-a", **common)
    h2, dev = _make(tmp_path, "dev", route="b", instance="meowdoku-dev", require_cap="dev_ui", **common)
    _accounts(stable)
    _accounts(dev)
    yield SimpleNamespace(stable=stable, dev=dev)
    for h in (h1, h2):
        h.shutdown()
        h.server_close()


def test_member_cannot_switch_operator_can_and_cookie_is_set(pair):
    s = pair.stable
    mem, _ = _login(s, "memcat")
    assert _call(s.port, "/api/route", "POST", {"Cookie": mem}, {"route": "b"})[0] == 403
    op, _ = _login(s, "opcat")
    status, cookies, body = _call(s.port, "/api/route", "POST", {"Cookie": op}, {"route": "b"})
    assert status == 200 and _route_cookies(cookies) == [ROUTE + "=b"] and body["route"] == "b"
    assert _call(s.port, "/api/route", "POST", {"Cookie": op}, {"route": "zzz"})[0] == 400


def test_get_route_lists_routes_and_preference_is_stored(pair):
    s = pair.stable
    op, _ = _login(s, "opcat")
    _, _, body = _call(s.port, "/api/route", headers={"Cookie": op + "; " + ROUTE + "=b"})
    assert body["route"] == "a" and body["routes"] == ["a", "b"] and body["selected"] == "b"
    assert body["preference"] is None and body["dev_available"] is True
    _call(s.port, "/api/route", "POST", {"Cookie": op}, {"site": "dev"})
    assert _call(s.port, "/api/route", headers={"Cookie": op})[2]["preference"] == "dev"
    _call(s.port, "/api/route", "POST", {"Cookie": op}, {"route": "a"})          # a route name stores the matching preference too
    assert _call(s.port, "/api/route", headers={"Cookie": op})[2]["preference"] == "stable"
    assert authz.get_site_pref(s.sessions.accounts_db, _uid(s, "opcat")) == "stable"


def test_dev_instance_refuses_member_resets_cookie_allows_open_paths_and_operator(pair):
    d = pair.dev
    mem, _ = _login(d, "memcat", {"Cookie": ROUTE + "=b"})
    status, cookies, body = _call(d.port, "/api/mathlab/jobs", headers={"Cookie": mem + "; " + ROUTE + "=b"})
    assert status == 403 and body == {"error": "this instance needs dev_ui"} and _route_cookies(cookies) == [ROUTE + "=a"]
    assert _call(d.port, "/", headers={"Cookie": ROUTE + "=b"})[0] == 403          # the route cookie alone grants nothing
    for path in ("/login.html", "/login.js", "/api/whoami", "/api/account/me", "/api/route"):
        assert _call(d.port, path, headers={"Cookie": mem})[0] == 200, path
    assert _call(d.port, "/api/whoami", headers={"Cookie": mem})[2]["instance"] == "meowdoku-dev"
    op, _ = _login(d, "opcat")
    assert _call(d.port, "/", headers={"Cookie": op})[0] == 200


def test_login_sets_route_by_capability_and_preference(pair):
    s = pair.stable
    op, cookies = _login(s, "opcat")
    assert _route_cookies(cookies) == [ROUTE + "=a"]                                 # no stored preference yet
    _call(s.port, "/api/route", "POST", {"Cookie": op}, {"site": "dev"})
    assert _route_cookies(_login(s, "opcat")[1]) == [ROUTE + "=b"]
    assert _route_cookies(_login(s, "memcat", {"Cookie": ROUTE + "=b"})[1]) == [ROUTE + "=a"]   # member: stable route, clears a stale b
    authz.set_site_pref(s.sessions.accounts_db, _uid(s, "memcat"), "dev")               # never honoured without dev_ui
    assert _route_cookies(_login(s, "memcat")[1]) == [ROUTE + "=a"]


def test_logout_resets_route_to_stable(pair):
    s = pair.stable
    op, _ = _login(s, "opcat")
    status, cookies, _ = _call(s.port, "/api/account/logout", "POST", {"Cookie": op + "; " + ROUTE + "=b"}, {})
    assert status == 200 and _route_cookies(cookies) == [ROUTE + "=a"]


def test_dev_disabled_falls_back_to_stable_and_reports_unavailable(tmp_path):
    httpd, s = _make(tmp_path, "solo", route="a", instance="meowdoku-a", routes=["a"], stable_route="a", dev_route="")
    try:
        _accounts(s)
        op, _ = _login(s, "opcat")
        authz.set_site_pref(s.sessions.accounts_db, _uid(s, "opcat"), "dev")
        assert _route_cookies(_login(s, "opcat")[1]) == [ROUTE + "=a"]
        body = _call(s.port, "/api/route", headers={"Cookie": op})[2]
        assert body["dev_available"] is False and body["preference"] == "dev" and body["routes"] == ["a"]
        assert _call(s.port, "/api/route", "POST", {"Cookie": op}, {"site": "dev"})[0] == 409
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture()
def shared(tmp_path):
    """Different state roots, ONE accounts.db (the dev handler is wired as main() wires --accounts-db)."""
    common = dict(routes=["a", "b"], stable_route="a", dev_route="b")
    h1, stable = _make(tmp_path, "stable", route="a", instance="meowdoku-a", **common)
    h2, dev = _make(tmp_path, "dev", route="b", instance="meowdoku-dev", require_cap="dev_ui", **common)
    dev.sessions.accounts_db = stable.sessions.accounts_db
    dev.sessions.accounts = srv.accounts_mod.Accounts(dev.sessions.accounts_db)
    dev.sessions.authz_policy = authz.Policy.beside(dev.sessions.accounts_db)
    assert stable.sessions.state.root != dev.sessions.state.root
    _accounts(stable)
    yield SimpleNamespace(stable=stable, dev=dev)
    for h in (h1, h2):
        h.shutdown()
        h.server_close()


def test_shared_accounts_one_session_one_preference(shared):
    s, d = shared.stable, shared.dev
    op, _ = _login(s, "opcat")
    who = _call(d.port, "/api/whoami", headers={"Cookie": op})[2]
    assert who["signed_in"] is True and who["role"] == "operator" and who["instance"] == "meowdoku-dev"
    assert _call(d.port, "/", headers={"Cookie": op})[0] == 200                      # the stable session passes the dev gate
    _call(s.port, "/api/route", "POST", {"Cookie": op}, {"site": "dev"})
    assert _call(d.port, "/api/route", headers={"Cookie": op})[2]["preference"] == "dev"


def test_shared_accounts_member_is_refused_by_the_dev_gate(shared):
    s, d = shared.stable, shared.dev
    mem, _ = _login(s, "memcat")
    status, cookies, body = _call(d.port, "/api/mathlab/jobs", headers={"Cookie": mem + "; " + ROUTE + "=b"})
    assert status == 403 and body["error"] == "this instance needs dev_ui" and _route_cookies(cookies) == [ROUTE + "=a"]
    assert _call(d.port, "/api/whoami", headers={"Cookie": mem})[2]["role"] == "member"
