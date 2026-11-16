"""Public sign-in (examples/meowdoku/server.py [LOGIN], login_guard.py, authz.py CLI) over an ephemeral loopback port.

An outside visitor is simulated as the loopback peer (a trusted proxy) forwarding X-Forwarded-For: 203.0.113.9 -> origin dns -> tier shader.
"""
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
from meowdoku import login_guard  # noqa: E402
from meowdoku import messaging  # noqa: E402
from meowdoku import server as srv  # noqa: E402

OUTSIDE = {"X-Forwarded-For": "203.0.113.9", "X-Forwarded-Proto": "https", "Host": "cats.example.org"}
PASSWORD = "correct horse 12"


def _call(port, path, method="GET", headers=None, body=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    hdrs = dict(headers or {})
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        hdrs.setdefault("Content-Type", "application/json")
    conn.request(method, path, body=data, headers=hdrs)
    r = conn.getresponse()
    out = r.read()
    info = (r.status, r.getheader("Retry-After"), r.msg.get_all("Set-Cookie") or [])
    conn.close()
    try:
        return info + (json.loads(out),)
    except ValueError:
        return info + (out.decode("utf-8", "replace"),)


@pytest.fixture()
def server(tmp_path):
    web = Path(tmp_path) / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    (web / "shader.html").write_text("<html>SHADER-PAGE</html>", encoding="utf-8")
    (web / "index.html").write_text("<html>FEED-PAGE</html>", encoding="utf-8")
    (web / "play.html").write_text("<html>GAME-PAGE</html>", encoding="utf-8")
    (web / "login.html").write_text("<html>LOGIN-PAGE</html>", encoding="utf-8")
    (web / "login.js").write_text("// login", encoding="utf-8")
    sessions = srv.Sessions(web, None)
    sessions.lifecycle = SimpleNamespace(token="t" * 40, restore_for=lambda token: None)
    sessions.access.lookup = lambda url: None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield SimpleNamespace(port=httpd.server_address[1], sessions=sessions)
    httpd.shutdown()
    httpd.server_close()


def _account(server, name, role):
    db = server.sessions.accounts_db
    server.sessions.accounts.ensure_user(name, PASSWORD)
    if role != "member":
        authz.grant_role(db, name, role, "operator:test") if role != "admin" else authz.grant_admin(db, name, "operator:test")


def _login(server, name, password=PASSWORD):
    status, retry, cookies, body = _call(server.port, "/api/account/login", "POST", OUTSIDE, {"username": name, "password": password})
    cookie = next((c.split(";")[0] for c in cookies if c.startswith(srv.ACCOUNT_COOKIE + "=") and c.split(";")[0] != srv.ACCOUNT_COOKIE + "="), "")
    return status, retry, cookie, body, cookies


def test_read_only_preview_can_be_kept_with_a_unique_chosen_password(server):
    terms = {"alpha_terms_version": srv.accounts_mod.ALPHA_TERMS_VERSION,
             "agree_mit": True, "agree_observation": True, "agree_research": True}
    preview_headers = {**OUTSIDE, "X-Meowdoku-Preview": "1"}
    assert _call(server.port, "/api/account/preview", "POST", OUTSIDE, {})[0] == 400
    status, _, cookies, body = _call(server.port, "/api/account/preview", "POST", preview_headers, {})
    assert status == 201 and body["account"]["temporary"] is True
    assert "password" not in body["account"]
    cookie = next(c.split(";")[0] for c in cookies if c.startswith(srv.ACCOUNT_COOKIE + "="))
    assert any("Max-Age=604800" in c and "HttpOnly" in c for c in cookies)
    authorized = {**OUTSIDE, "Cookie": cookie}
    me = _call(server.port, "/api/account/me", headers=authorized)[3]
    assert me["role"] == "trial" and me["account"]["temporary"] is True
    who = _call(server.port, "/api/whoami", headers=authorized)[3]
    assert who["role"] == "trial" and set(who["capabilities"]) == {"view_site", "photo_read"}
    assert _call(server.port, "/api/reports", "POST", authorized, {"title": "blocked"})[0] == 403
    assert _call(server.port, "/api/account/keep", "POST", {**authorized, "X-Meowdoku-Preview": "1"},
                 {"username": "keeper", "password": PASSWORD, **terms, "agree_research": False})[0] == 400
    status, _, kept_cookies, kept = _call(server.port, "/api/account/keep", "POST",
                                          {**authorized, "X-Meowdoku-Preview": "1"},
                                          {"username": "keeper", "password": PASSWORD, **terms})
    assert status == 200 and kept["account"]["username"] == "keeper"
    new_cookie = next(c.split(";")[0] for c in kept_cookies if c.startswith(srv.ACCOUNT_COOKIE + "="))
    assert new_cookie != cookie
    assert _call(server.port, "/api/account/me", headers=authorized)[3]["account"] is None
    assert _call(server.port, "/api/account/me", headers={**OUTSIDE, "Cookie": new_cookie})[3]["role"] == "member"
    assert _login(server, "keeper", PASSWORD)[0] == 200


def test_preview_expires_after_seven_days(server):
    headers = {**OUTSIDE, "X-Meowdoku-Preview": "1"}
    status, _, cookies, body = _call(server.port, "/api/account/preview", "POST", headers, {})
    assert status == 201
    cookie = next(c.split(";")[0] for c in cookies if c.startswith(srv.ACCOUNT_COOKIE + "="))
    with sqlite3.connect(server.sessions.accounts_db) as db:
        db.execute("UPDATE temporary_accounts SET expires_at = 1 WHERE user_id = ?", (body["account"]["id"],))
    assert _call(server.port, "/api/account/me", headers={**OUTSIDE, "Cookie": cookie})[3]["account"] is None


def test_anonymous_outside_visitor_keeps_the_shader_lock(server):
    assert "FEED-PAGE" in _call(server.port, "/", headers=OUTSIDE)[3]
    assert "SHADER-PAGE" in _call(server.port, "/play.html", headers=OUTSIDE)[3]
    assert _call(server.port, "/api/mathlab/jobs", headers=OUTSIDE)[0] == 403
    who = _call(server.port, "/api/whoami", headers=OUTSIDE)[3]
    assert who["origin_tier"] == "shader" and who["tier"] == "shader" and who["signed_in"] is False and who["username"] is None
    assert _call(server.port, "/api/account/me", headers=OUTSIDE)[3]["account"] is None


def test_login_page_and_consented_alpha_signup_are_public(server):
    for name in messaging.INTRODUCTION_AGENTS:
        server.sessions.accounts.ensure_user(name, PASSWORD)
    page = (ROOT / "examples" / "meowdoku" / "web" / "register.html").read_text(encoding="utf-8")
    assert f'data-terms-version="{srv.accounts_mod.ALPHA_TERMS_VERSION}"' in page
    assert _call(server.port, "/login.html", headers=OUTSIDE)[3] == "<html>LOGIN-PAGE</html>"
    assert _call(server.port, "/login.js", headers=OUTSIDE)[3] == "// login"
    status, _, _, body = _call(server.port, "/api/account/signup", "POST", OUTSIDE, {"username": "newcat", "password": PASSWORD})
    assert status == 400 and "agree" in body["error"]
    terms = {"alpha_terms_version": srv.accounts_mod.ALPHA_TERMS_VERSION,
             "agree_mit": True, "agree_observation": True, "agree_research": True}
    assert _call(server.port, "/api/account/signup", "POST", OUTSIDE,
                 {"username": "newcat", "password": PASSWORD, **terms, "agree_research": False})[0] == 400
    assert _call(server.port, "/api/account/signup", "POST", OUTSIDE,
                 {"username": "newcat", "password": PASSWORD, **terms, "alpha_terms_version": "old"})[0] == 400
    status, _, cookies, body = _call(server.port, "/api/account/signup", "POST", OUTSIDE,
                                    {"username": "newcat", "password": PASSWORD, **terms})
    assert status == 201 and body["account"]["username"] == "newcat"
    assert any(c.startswith(srv.ACCOUNT_COOKIE + "=") for c in cookies)
    assert authz.role_of(server.sessions.accounts_db, body["account"]["id"]) == "member"
    with sqlite3.connect(server.sessions.accounts_db) as db:
        assert db.execute("SELECT terms_version FROM alpha_agreements WHERE user_id = ?",
                          (body["account"]["id"],)).fetchone() == (terms["alpha_terms_version"],)
        for name in messaging.INTRODUCTION_AGENTS:
            agent_id = db.execute("SELECT id FROM users WHERE username=?", (name,)).fetchone()[0]
            states = db.execute("SELECT user_id,peer_id,state FROM msg_contacts WHERE (user_id=? AND peer_id=?) "
                                "OR (user_id=? AND peer_id=?)", (agent_id, body["account"]["id"],
                                body["account"]["id"], agent_id)).fetchall()
            assert len(states) == 2 and all(row[2] == "accepted" for row in states)


def test_wrong_password_six_times_locks_with_retry_after(server, monkeypatch):
    monkeypatch.setattr(srv.time, "sleep", lambda s: None)
    _account(server, "cat_one", "member")
    for _ in range(5):
        status, _, cookie, body, _ = _login(server, "cat_one", "nope nope nope")
        assert status == 401 and body == {"error": "wrong username or password"} and cookie == ""
    status, retry, _, body, _ = _login(server, "cat_one", "nope nope nope")
    assert status == 429 and retry is not None and int(retry) >= 1 and body["retry_after"] == int(retry)
    assert _login(server, "cat_one")[0] == 429                                  # even the right password waits
    status, _, _, body, _ = _login(server, "ghost_user", "nope nope nope")      # unknown user: same failure body
    assert status == 401 and body == {"error": "wrong username or password"}


def test_member_login_lifts_the_tier_and_logout_restores_the_lock(server):
    _account(server, "cat_two", "member")
    status, _, cookie, body, cookies = _login(server, "cat_two")
    assert status == 200 and body["account"]["username"] == "cat_two" and cookie
    assert any(c.startswith(srv.ACCOUNT_COOKIE) and "; Secure" in c and "HttpOnly" in c and "SameSite=Lax" in c for c in cookies)
    who = _call(server.port, "/api/whoami", headers={**OUTSIDE, "Cookie": cookie})[3]
    assert who["signed_in"] is True and who["username"] == "cat_two" and who["origin_tier"] == "shader"
    assert who["tier"] == "play" and who["origin"] == "dns" and "compile_submit" in who["capabilities"]
    assert "FEED-PAGE" in _call(server.port, "/", headers={**OUTSIDE, "Cookie": cookie})[3]
    assert "GAME-PAGE" in _call(server.port, "/play.html", headers={**OUTSIDE, "Cookie": cookie})[3]
    assert _call(server.port, "/api/account/logout", "POST", {**OUTSIDE, "Cookie": cookie}, {})[0] == 200
    assert "FEED-PAGE" in _call(server.port, "/", headers={**OUTSIDE, "Cookie": cookie})[3]
    assert "SHADER-PAGE" in _call(server.port, "/play.html", headers={**OUTSIDE, "Cookie": cookie})[3]
    assert _call(server.port, "/api/mathlab/jobs", headers={**OUTSIDE, "Cookie": cookie})[0] == 403


def test_admin_from_outside_gets_owner_tools_but_not_lifecycle_admin(server):
    _account(server, "boss_cat", "admin")
    cookie = _login(server, "boss_cat")[2]
    who = _call(server.port, "/api/whoami", headers={**OUTSIDE, "Cookie": cookie})[3]
    assert who["tier"] == "edit" and who["role"] == "admin" and who["origin_tier"] == "shader"
    assert "owner_tools" in who["capabilities"] and "lifecycle_admin" not in who["capabilities"]


def test_operator_lift_is_dev(server):
    _account(server, "op_cat", "operator")
    cookie = _login(server, "op_cat")[2]
    who = _call(server.port, "/api/whoami", headers={**OUTSIDE, "Cookie": cookie})[3]
    assert who["tier"] == "dev" and "lifecycle_admin" not in who["capabilities"]


def test_form_content_type_is_refused(server):
    _account(server, "cat_three", "member")
    raw = ("username=cat_three&password=" + PASSWORD.replace(" ", "+")).encode()
    status, _, cookies, _ = _call(server.port, "/api/account/login", "POST", {**OUTSIDE, "Content-Type": "application/x-www-form-urlencoded"}, raw=raw)
    assert status == 415 and not any(c.startswith(srv.ACCOUNT_COOKIE + "=") and not c.startswith(srv.ACCOUNT_COOKIE + "=;") for c in cookies)


def test_guard_limits_and_pruning(tmp_path):
    db = login_guard.connect(tmp_path / "accounts.db")
    now = 1_000_000.0
    assert login_guard.check(db, "1.1.1.1", "a", now) == 0
    for i in range(4):
        login_guard.record(db, "1.1.1.1", "a", False, now + i)
    assert login_guard.check(db, "1.1.1.1", "a", now + 5) == 0
    login_guard.record(db, "1.1.1.1", "a", False, now + 5)
    assert 1 <= login_guard.check(db, "1.1.1.1", "a", now + 6) <= 600
    assert login_guard.check(db, "1.1.1.1", "a", now + 700) == 0                  # aged out of the 10 min window
    assert login_guard.check(db, "1.1.1.1", "b", now + 6) == 0                    # a different username is not paired
    for i in range(15):
        login_guard.record(db, "2.2.2.2", f"u{i}", False, now + i)               # 20 failures from one client in total
    for i in range(5):
        login_guard.record(db, "2.2.2.2", f"v{i}", False, now + 20 + i)
    assert login_guard.check(db, "2.2.2.2", "fresh", now + 30) > 0
    for i in range(50):
        login_guard.record(db, f"9.9.9.{i}", "victim", False, now + i)
    assert login_guard.check(db, "8.8.8.8", "victim", now + 60) > 0
    login_guard.record(db, "3.3.3.3", "z", True, now + 3 * 86400)
    assert db.execute("SELECT COUNT(*) FROM login_attempts WHERE ts < ?", (now + 2 * 86400,)).fetchone()[0] == 0


def test_cli_set_password_and_create_user(tmp_path, monkeypatch, capsys):
    db = str(tmp_path / "accounts.db")
    answers = iter([PASSWORD, PASSWORD, "short", "short"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
    assert authz.main(["create-user", "owner_cat", "--role", "admin", "--db", db, "--prompt", "--granted-by", "operator:test"]) == 0
    assert capsys.readouterr().out == "OK owner_cat\n"
    assert authz.role_of(db, authz._accounts_module().Accounts(Path(db)).login("owner_cat", PASSWORD)["id"]) == "admin"
    assert authz.main(["set-password", "owner_cat", "--db", db, "--prompt"]) == 1          # too short: refused
    assert authz.main(["set-password", "owner_cat", "--db", db]) == 1                      # no --prompt / --password-file: refused
    assert PASSWORD not in capsys.readouterr().out
