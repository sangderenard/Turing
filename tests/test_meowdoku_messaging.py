"""User-to-user messaging (examples/meowdoku/messaging.py, server.py [MESSAGES], deploy/agent_messenger.py) over an ephemeral loopback port."""
from __future__ import annotations

import http.client
import json
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples"), str(ROOT / "examples" / "meowdoku" / "deploy")]

from meowdoku import authz  # noqa: E402
from meowdoku import messaging  # noqa: E402
from meowdoku import server as srv  # noqa: E402
import agent_messenger  # noqa: E402

PASSWORD = "correct horse 12"


def _call(port, path, method="GET", cookie="", body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=40)
    headers = {"Cookie": cookie} if cookie else {}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=headers)
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    try:
        return r.status, json.loads(raw)
    except ValueError:
        return r.status, raw.decode("utf-8", "replace")


@pytest.fixture()
def server(tmp_path):
    web = Path(tmp_path) / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}), encoding="utf-8")
    (web / "index.html").write_text("<html>GAME</html>", encoding="utf-8")
    sessions = srv.Sessions(web, None)
    sessions.lifecycle = SimpleNamespace(token="t" * 40, restore_for=lambda token: None)
    sessions.access.lookup = lambda url: None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield SimpleNamespace(port=httpd.server_address[1], sessions=sessions, tmp=Path(tmp_path))
    httpd.shutdown()
    httpd.server_close()


def _user(server, name, display=None, admin=False):
    server.sessions.accounts.ensure_user(name, PASSWORD)
    if display:
        server.sessions.accounts._conn.execute("UPDATE users SET display_name = ? WHERE username = ?", (display, name))
        server.sessions.accounts._conn.commit()
    if admin:
        authz.grant_admin(server.sessions.accounts_db, name, "operator:test")
    return _login(server, name)


def _login(server, name):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    conn.request("POST", "/api/account/login", body=json.dumps({"username": name, "password": PASSWORD}), headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    r.read()
    cookie = next(c.split(";")[0] for c in r.msg.get_all("Set-Cookie") if c.startswith(srv.ACCOUNT_COOKIE + "="))
    conn.close()
    return cookie


def _friends(server, a, ca, b, cb):
    assert _call(server.port, "/api/messages/contacts", "POST", ca, {"username": b, "action": "request"})[0] == 200
    assert _call(server.port, "/api/messages/contacts", "POST", cb, {"username": a, "action": "accept"})[0] == 200


def _send(server, cookie, to, text):
    return _call(server.port, "/api/messages/send", "POST", cookie, {"to": to, "body": text})


def test_guest_is_refused_naming_message(server):
    for path in ("/api/messages/contacts", "/api/messages/since?after=0"):
        status, body = _call(server.port, path)
        assert status == 403 and "message" in body["error"]
    assert _call(server.port, "/api/messages/send", "POST", "", {"to": "x", "body": "y"})[0] == 403


def test_request_accept_flow_and_no_send_before_mutual(server):
    ca, cb = _user(server, "ann"), _user(server, "ben")
    _call(server.port, "/api/messages/contacts", "POST", ca, {"username": "ben", "action": "request"})
    status, body = _send(server, ca, "ben", "hi")
    assert status == 403 and body == {"error": "you cannot message this user"}
    assert _send(server, cb, "ann", "hi")[0] == 403                      # not accepted on his side either
    mine = _call(server.port, "/api/messages/contacts", cookie=cb)[1]
    assert [r["username"] for r in mine["requests"]] == ["ann"] and mine["contacts"] == []
    assert [o["username"] for o in _call(server.port, "/api/messages/contacts", cookie=ca)[1]["outgoing"]] == ["ben"]
    _call(server.port, "/api/messages/contacts", "POST", cb, {"username": "ann", "action": "accept"})
    assert _send(server, ca, "ben", "hi")[0] == 200
    assert [c["username"] for c in _call(server.port, "/api/messages/contacts", cookie=ca)[1]["contacts"]] == ["ben"]


def test_unknown_username_same_response_shape_as_valid_request(server):
    ca = _user(server, "ann")
    _user(server, "ben")
    real = _call(server.port, "/api/messages/contacts", "POST", ca, {"username": "ben", "action": "request"})
    unknown = _call(server.port, "/api/messages/contacts", "POST", ca, {"username": "nobody", "action": "request"})
    myself = _call(server.port, "/api/messages/contacts", "POST", ca, {"username": "ann", "action": "request"})
    assert real == unknown == myself
    assert _send(server, ca, "nobody", "x") == _send(server, ca, "ann", "x")


def test_round_trip_thread_since_read_and_third_party_sees_nothing(server):
    ca, cb, cc = _user(server, "ann", "Ann A"), _user(server, "ben", "Ben B"), _user(server, "cyd")
    _friends(server, "ann", ca, "ben", cb)
    ids = [_send(server, ca, "ben", "one")[1]["id"], _send(server, cb, "ann", "  two <b>x</b>  ")[1]["id"], _send(server, ca, "ben", "three")[1]["id"]]
    thread = _call(server.port, "/api/messages/thread?peer=ann", cookie=cb)[1]
    assert [m["body"] for m in thread["messages"]] == ["one", "two <b>x</b>", "three"] and thread["more"] is False
    older = _call(server.port, f"/api/messages/thread?peer=ann&before={ids[2]}", cookie=cb)[1]
    assert [m["id"] for m in older["messages"]] == ids[:2]
    since = _call(server.port, f"/api/messages/since?after={ids[0]}", cookie=ca)[1]
    assert [m["id"] for m in since["messages"]] == ids[1:] and since["latest"] == ids[2]
    assert _call(server.port, "/api/messages/since?after=-1", cookie=ca)[1] == {"messages": [], "latest": ids[2]}
    unread = {c["username"]: c["unread"] for c in _call(server.port, "/api/messages/contacts", cookie=cb)[1]["contacts"]}
    assert unread == {"ann": 2}
    assert _call(server.port, "/api/messages/read", "POST", cb, {"peer": "ann", "up_to": ids[2]})[1]["marked"] == 2
    assert _call(server.port, "/api/messages/contacts", cookie=cb)[1]["contacts"][0]["unread"] == 0
    # a third account sees none of it
    assert _call(server.port, "/api/messages/thread?peer=ann", cookie=cc)[1]["messages"] == []
    assert _call(server.port, "/api/messages/thread?peer=ben", cookie=cc)[1]["messages"] == []
    assert _call(server.port, "/api/messages/since?after=0", cookie=cc)[1]["messages"] == []
    other = _call(server.port, "/api/messages/contacts", cookie=cc)[1]
    assert other == {"contacts": [], "requests": [], "outgoing": [], "blocked": []}


def test_exact_key_sets_a_peer_sees(server):
    ca, cb = _user(server, "ann", "Ann A"), _user(server, "ben", "Ben B")
    _friends(server, "ann", ca, "ben", cb)
    _send(server, ca, "ben", "hello")
    contact = _call(server.port, "/api/messages/contacts", cookie=cb)[1]["contacts"][0]
    assert set(contact) == {"username", "display_name", "unread", "last"} and contact["display_name"] == "Ann A"
    message = _call(server.port, "/api/messages/thread?peer=ann", cookie=cb)[1]["messages"][0]
    assert set(message) == {"id", "from", "to", "created", "body", "removed"}
    polled = _call(server.port, "/api/messages/since?after=0", cookie=ca)[1]["messages"][0]       # the sender sees no read state either
    assert set(polled) == set(message)
    _call(server.port, "/api/messages/contacts", "POST", cb, {"username": "ann", "action": "request"})
    cc = _user(server, "cyd", "Cyd")
    _call(server.port, "/api/messages/contacts", "POST", cc, {"username": "ben", "action": "request"})
    request = _call(server.port, "/api/messages/contacts", cookie=cb)[1]["requests"][0]
    assert set(request) == {"username", "display_name", "created"}


def test_block_stops_delivery(server):
    ca, cb = _user(server, "ann"), _user(server, "ben")
    _friends(server, "ann", ca, "ben", cb)
    assert _send(server, ca, "ben", "before")[0] == 200
    assert _call(server.port, "/api/messages/contacts", "POST", cb, {"username": "ann", "action": "block"})[0] == 200
    assert _send(server, ca, "ben", "after")[0] == 403 and _send(server, cb, "ann", "after")[0] == 403
    assert [m["body"] for m in _call(server.port, "/api/messages/thread?peer=ben", cookie=ca)[1]["messages"]] == ["before"]
    assert _call(server.port, "/api/messages/contacts", cookie=cb)[1]["blocked"] == [{"username": "ann"}]
    _call(server.port, "/api/messages/contacts", "POST", ca, {"username": "ben", "action": "request"})     # a blocker is not re-opened by a request
    assert _send(server, ca, "ben", "again")[0] == 403


def test_body_and_rate_limits(server, monkeypatch):
    ca, cb = _user(server, "ann"), _user(server, "ben")
    _friends(server, "ann", ca, "ben", cb)
    monkeypatch.setattr(messaging, "MAX_BODY", 10)
    assert _send(server, ca, "ben", "   ")[0] == 400 and _send(server, ca, "ben", "x" * 11)[0] == 400
    assert _send(server, ca, "ben", "x" * 10)[0] == 200
    monkeypatch.setattr(messaging, "MAX_PER_MINUTE", 3)
    assert _send(server, ca, "ben", "b")[0] == 200 and _send(server, ca, "ben", "c")[0] == 200
    assert _send(server, ca, "ben", "d")[0] == 429
    assert _send(server, cb, "ann", "free")[0] == 200                  # per sender
    monkeypatch.setattr(messaging, "MAX_PER_MINUTE", 100)
    monkeypatch.setattr(messaging, "MAX_PER_DAY", 3)
    assert _send(server, ca, "ben", "e")[0] == 429
    monkeypatch.setattr(messaging, "MAX_PENDING", 1)
    _user(server, "cyd"), _user(server, "dia")
    assert _call(server.port, "/api/messages/contacts", "POST", cb, {"username": "cyd", "action": "request"})[0] == 200
    assert _call(server.port, "/api/messages/contacts", "POST", cb, {"username": "dia", "action": "request"})[0] == 429


def test_json_only_and_bad_action(server):
    ca = _user(server, "ann")
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    conn.request("POST", "/api/messages/send", body=b"to=x&body=y", headers={"Cookie": ca, "Content-Type": "application/x-www-form-urlencoded"})
    r = conn.getresponse()
    r.read()
    conn.close()
    assert r.status == 415
    assert _call(server.port, "/api/messages/contacts", "POST", ca, {"username": "x", "action": "nope"})[0] == 400


def test_long_poll_returns_when_a_message_arrives(server):
    ca, cb = _user(server, "ann"), _user(server, "ben")
    _friends(server, "ann", ca, "ben", cb)
    threading.Timer(0.5, lambda: _send(server, ca, "ben", "late")).start()
    start = time.monotonic()
    status, body = _call(server.port, "/api/messages/since?after=0&wait=10", cookie=cb)
    assert status == 200 and [m["body"] for m in body["messages"]] == ["late"] and time.monotonic() - start < 6
    start = time.monotonic()
    assert _call(server.port, f"/api/messages/since?after={body['latest']}&wait=1", cookie=cb)[1]["messages"] == []
    assert 0.8 < time.monotonic() - start < 5


def test_waiter_caps(server, monkeypatch):
    hub = messaging.Hub()
    monkeypatch.setattr(messaging, "MAX_WAITERS_PER_ACCOUNT", 2)
    hub.enter(1), hub.enter(1)
    with pytest.raises(messaging.MessagingError) as caught:
        hub.enter(1)
    assert caught.value.status == 429
    monkeypatch.setattr(messaging, "MAX_WAITERS", 3)
    hub.enter(2)
    with pytest.raises(messaging.MessagingError):
        hub.enter(3)


def test_admin_endpoints_refuse_non_admin_and_audit(server):
    ca, cb = _user(server, "ann"), _user(server, "ben")
    boss = _user(server, "boss", admin=True)
    _friends(server, "ann", ca, "ben", cb)
    mid = _send(server, ca, "ben", "secret words")[1]["id"]
    for path, method, body in (("/api/messages/admin/thread?a=ann&b=ben", "GET", None), ("/api/messages/admin/remove", "POST", {"id": mid}),
                               ("/api/messages/admin/suspend", "POST", {"username": "ann", "suspended": True})):
        status, reply = _call(server.port, path, method, ca, body)
        assert status == 403 and "account_manage" in reply["error"]
    seen = _call(server.port, "/api/messages/admin/thread?a=ann&b=ben", cookie=boss)[1]["messages"]
    assert [m["body"] for m in seen] == ["secret words"]
    assert _call(server.port, "/api/messages/admin/remove", "POST", boss, {"id": mid})[0] == 200
    for cookie, peer in ((ca, "ben"), (cb, "ann")):                       # removal is visible as removed to both sides
        message = _call(server.port, f"/api/messages/thread?peer={peer}", cookie=cookie)[1]["messages"][0]
        assert message["body"] == "" and message["removed"] is True
    assert _call(server.port, "/api/messages/admin/thread?a=ann&b=ben", cookie=boss)[1]["messages"][0]["body"] == "secret words"
    db = messaging.connect(server.sessions.accounts_db)
    assert [r["action"] for r in db.execute("SELECT action FROM msg_audit ORDER BY id")] == ["read_thread", "remove_message", "read_thread"]
    db.close()


def test_suspended_sender_is_refused(server):
    ca, cb = _user(server, "ann"), _user(server, "ben")
    boss = _user(server, "boss", admin=True)
    _friends(server, "ann", ca, "ben", cb)
    assert _call(server.port, "/api/messages/admin/suspend", "POST", boss, {"username": "ann", "suspended": True})[1]["suspended"] is True
    assert _send(server, ca, "ben", "hi")[0] == 403
    assert _call(server.port, "/api/messages/contacts", "POST", ca, {"username": "boss", "action": "request"})[0] == 403
    assert _send(server, cb, "ann", "still can receive")[0] == 200
    _call(server.port, "/api/messages/admin/suspend", "POST", boss, {"username": "ann", "suspended": False})
    assert _send(server, ca, "ben", "back")[0] == 200


def test_agent_client_drives_the_flow(server, capsys, tmp_path):
    _user(server, "claude"), _user(server, "codex")
    secret = tmp_path / "pw.secret"
    secret.write_text(PASSWORD + "\n", encoding="utf-8")
    base = f"http://127.0.0.1:{server.port}"

    def run(user, *argv):
        code = agent_messenger.main(["--base", base, "--user", user, "--password-file", str(secret), *argv])
        out = capsys.readouterr()
        assert PASSWORD not in out.out + out.err
        return code, out.out.strip()

    assert run("claude", "add", "codex")[0] == 0
    assert json.loads(run("codex", "inbox")[1])["requests"] == ["claude"]
    assert run("codex", "accept", "claude")[0] == 0
    assert run("claude", "send", "codex", "hello codex")[0] == 0
    assert json.loads(run("codex", "inbox")[1])["unread"] == {"claude": 1}
    code, out = run("codex", "read", "claude")
    assert code == 0 and json.loads(out)["body"] == "hello codex"
    assert json.loads(run("codex", "inbox")[1])["unread"] == {}
    assert run("codex", "wait", "--timeout", "1")[0] == 3
    threading.Timer(0.5, lambda: _send(server, _login(server, "claude"), "codex", "wake up")).start()
    code, out = run("codex", "wait", "--timeout", "10")
    line = json.loads(out)
    assert code == 0 and line["body"] == "wake up" and line["from"] == "claude" and set(line) == {"id", "from", "to", "created", "body"}
    assert run("codex", "contacts")[0] == 0
    assert agent_messenger.main(["--base", base, "--user", "claude", "--password-file", str(tmp_path / "missing"), "contacts"]) == 1
