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
    assert set(message) == {"id", "from", "to", "created", "body", "removed", "subject", "type"}
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


def test_agent_client_drives_the_flow(server, capsys, tmp_path, monkeypatch):
    _user(server, "claude"), _user(server, "codex")
    secret = tmp_path / "pw.secret"
    secret.write_text(PASSWORD + "\n", encoding="utf-8")
    base = "https://messenger.test"
    monkeypatch.setattr(srv.proxy_trust_mod, "TRUSTED_PROXIES", {"127.0.0.1"})

    class FixtureClient(agent_messenger.Client):
        # CLI flow against the real handler via its trusted TLS-termination path.
        # Actual network TLS verification remains the standard urllib SSL layer.
        def __init__(self, base, user, password, context=None):
            self.user = user
            self.token = _login(server, user).split("=", 1)[1]

        def call(self, path, body=None, timeout=40):
            status, result = _rest_call(server, path, self.token, secure=True, body=body)
            if status >= 400:
                raise agent_messenger.ClientError(result.get("error", "request failed"))
            return result

    monkeypatch.setattr(agent_messenger, "Client", FixtureClient)

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
    assert code == 0 and line["body"] == "wake up" and line["from"] == "claude" and set(line) == {"id", "from", "to", "created", "body", "subject", "type"}
    assert run("codex", "contacts")[0] == 0
    # pending / ack / typed + subject send / wait --from-cursor
    assert run("claude", "send", "codex", "need approval", "--type", "approval_request", "--subject", "job:j-7")[0] == 0
    code, out = run("codex", "pending")
    lines = [json.loads(x) for x in out.splitlines()]
    assert code == 0 and [m["body"] for m in lines] == ["hello codex", "wake up", "need approval"]
    assert lines[-1]["type"] == "approval_request" and lines[-1]["subject"] == {"kind": "job", "ref": "j-7"}
    assert run("codex", "ack", str(lines[1]["id"]))[0] == 0
    assert [json.loads(x)["body"] for x in run("codex", "pending")[1].splitlines()] == ["need approval"]
    code, out = run("codex", "wait", "--from-cursor", "--timeout", "5")
    assert code == 0 and json.loads(out)["body"] == "need approval"
    assert run("claude", "send", "codex", "x", "--subject", "nocolon")[0] == 1
    assert agent_messenger.main(["--base", base, "--user", "claude", "--password-file", str(tmp_path / "missing"), "contacts"]) == 1


def test_migration_v1_to_v2_keeps_rows(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    old = sqlite3.connect(str(path))
    old.execute("CREATE TABLE msg_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, sender_id INTEGER NOT NULL, recipient_id INTEGER NOT NULL, "
                "body TEXT NOT NULL, created REAL NOT NULL, read_at REAL, removed_at REAL, removed_by INTEGER)")
    old.execute("INSERT INTO msg_messages (sender_id, recipient_id, body, created) VALUES (1, 2, 'legacy', 5.0)")
    old.execute("CREATE TABLE schema_version (module TEXT PRIMARY KEY, version INTEGER NOT NULL, updated_at REAL NOT NULL, writer TEXT NOT NULL)")
    old.execute("INSERT INTO schema_version VALUES ('messaging', 1, 0, 'test')")
    old.commit()
    old.close()
    db = messaging.connect(path)
    row = db.execute("SELECT body, subject_kind, subject_ref, msg_type FROM msg_messages").fetchone()
    assert tuple(row) == ("legacy", None, None, "note")
    assert db.execute("SELECT version FROM schema_version WHERE module = 'messaging'").fetchone()[0] == 2
    db.close()
    messaging.connect(path).close()                                      # a second open is a no-op


def test_subject_and_type_validation_and_filters(server):
    ca, cb = _user(server, "ann"), _user(server, "ben")
    _friends(server, "ann", ca, "ben", cb)

    def send(**extra):
        return _call(server.port, "/api/messages/send", "POST", ca, {"to": "ben", "body": "m", **extra})

    for bad in ({"type": "shout"}, {"type": 5}, {"subject": {"kind": "bogus", "ref": "x"}}, {"subject": {"kind": "job", "ref": ""}},
                {"subject": {"kind": "job", "ref": "x" * 201}}, {"subject": {"kind": "job"}}, {"subject": "job:1"}):
        assert send(**bad)[0] == 400, bad
    plain = send()[1]["id"]
    tagged = send(type="question", subject={"kind": "ticket", "ref": "T-1"})[1]["id"]
    other = send(subject={"kind": "lease", "ref": "L-9"})[1]["id"]
    got = {m["id"]: m for m in _call(server.port, "/api/messages/thread?peer=ann", cookie=cb)[1]["messages"]}
    assert got[plain]["subject"] is None and got[plain]["type"] == "note"
    assert got[tagged]["subject"] == {"kind": "ticket", "ref": "T-1"} and got[tagged]["type"] == "question"
    only = _call(server.port, "/api/messages/thread?peer=ann&subject_kind=ticket&subject_ref=T-1", cookie=cb)[1]["messages"]
    assert [m["id"] for m in only] == [tagged]
    assert [m["id"] for m in _call(server.port, "/api/messages/since?after=0&subject_kind=lease", cookie=cb)[1]["messages"]] == [other]
    assert _call(server.port, "/api/messages/since?after=0&subject_kind=bogus", cookie=cb)[0] == 400


def test_inbox_ack_round_trip_monotonic_and_hidden_from_sender(server):
    ca, cb = _user(server, "ann"), _user(server, "ben")
    _friends(server, "ann", ca, "ben", cb)
    ids = [_send(server, ca, "ben", f"m{i}")[1]["id"] for i in range(3)]
    mine = _send(server, cb, "ann", "from ben")[1]["id"]
    box = _call(server.port, "/api/messages/inbox", cookie=cb)[1]
    assert [m["id"] for m in box["messages"]] == ids and box["cursor"] == 0       # only messages TO ben
    assert _call(server.port, "/api/messages/ack", "POST", cb, {"up_to": ids[1]})[1]["cursor"] == ids[1]
    box = _call(server.port, "/api/messages/inbox", cookie=cb)[1]
    assert [m["id"] for m in box["messages"]] == [ids[2]] and box["cursor"] == ids[1]
    assert _call(server.port, "/api/messages/ack", "POST", cb, {"up_to": 1})[1]["cursor"] == ids[1]         # never backwards
    assert _call(server.port, "/api/messages/ack", "POST", cb, {"up_to": mine + 1000})[1]["cursor"] == ids[2]  # never beyond the latest to me
    assert _call(server.port, "/api/messages/inbox", cookie=cb)[1] == {"messages": [], "cursor": ids[2]}
    assert _call(server.port, "/api/messages/ack", "POST", cb, {"up_to": "x"})[0] == 400
    assert _call(server.port, "/api/messages/ack", "POST", cb, {"up_to": -1})[0] == 400
    # the sender sees nothing of the recipient's cursor on any endpoint
    seen = [_call(server.port, p, cookie=ca)[1] for p in (f"/api/messages/thread?peer=ben", "/api/messages/since?after=0", "/api/messages/contacts")]
    seen.append(_call(server.port, "/api/messages/inbox", cookie=ca)[1])
    text = json.dumps(seen)
    assert "cursor" not in text.replace('"cursor": 0', "") and "acked" not in text
    assert _call(server.port, "/api/messages/inbox", cookie=ca)[1]["cursor"] == 0
    assert [m["id"] for m in _call(server.port, "/api/messages/inbox", cookie=ca)[1]["messages"]] == [mine]



def _rest_call(server, path, token, *, secure=False, cookie="", body=None):
    # Exercise the existing trusted TLS-termination path, not a real TLS socket.
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=40)
    headers = {"Authorization": "Bearer " + token, "Cookie": cookie}
    if secure:
        headers.update({"X-Forwarded-For": "192.0.2.44", "X-Forwarded-Proto": "https"})
    if body is not None:
        headers["Content-Type"] = "application/json"
    conn.request("POST" if body is not None else "GET", path,
                 body=json.dumps(body) if body is not None else None, headers=headers)
    response = conn.getresponse()
    result = response.status, json.loads(response.read())
    conn.close()
    return result


def test_rest_bearer_reuses_sessions_and_requires_secure_transport(server, monkeypatch):
    monkeypatch.setattr(srv.proxy_trust_mod, "TRUSTED_PROXIES", {"127.0.0.1"})
    ca, cb = _user(server, "ann"), _user(server, "ben")
    _friends(server, "ann", ca, "ben", cb)
    ta, tb = ca.split("=", 1)[1], cb.split("=", 1)[1]
    path = "/api/messages/inbox"
    assert _rest_call(server, path, ta, cookie=ca)[0] == 403
    assert _rest_call(server, path, "invalid", secure=True, cookie=ca)[0] == 403
    status, sent = _rest_call(server, "/api/messages/send", ta, secure=True,
                              cookie=cb, body={"to": "ben", "body": "REST token test"})
    assert status == 200
    box = _rest_call(server, path, tb, secure=True)[1]
    assert [m["from"] for m in box["messages"]] == ["ann"]
    assert _rest_call(server, "/api/messages/read", tb, secure=True,
                      body={"peer": "ann", "up_to": sent["id"]})[1]["marked"] == 1
    assert len(_rest_call(server, path, tb, secure=True)[1]["messages"]) == 1
    assert _rest_call(server, "/api/messages/ack", tb, secure=True,
                      body={"up_to": sent["id"]})[1]["cursor"] == sent["id"]
    assert _rest_call(server, path, tb, secure=True)[1]["messages"] == []
    assert _rest_call(server, "/api/messages/admin/thread?a=ann&b=ben", ta, secure=True)[0] == 403
    # The header does not become a site-wide login mechanism.
    assert _rest_call(server, "/api/account/me", ta, secure=True)[1]["account"] is None
    server.sessions.accounts.logout(ta)
    assert _rest_call(server, path, ta, secure=True)[0] == 403


def test_rest_bearer_ignores_untrusted_forwarding_and_expired_tokens(server, monkeypatch):
    ca = _user(server, "ann")
    token = ca.split("=", 1)[1]
    monkeypatch.setattr(srv.proxy_trust_mod, "TRUSTED_PROXIES", set())
    assert _rest_call(server, "/api/messages/inbox", token, secure=True)[0] == 403
    monkeypatch.setattr(srv.proxy_trust_mod, "TRUSTED_PROXIES", {"127.0.0.1"})
    server.sessions.accounts._conn.execute("UPDATE sessions SET expires_at=0 WHERE token=?", (token,))
    server.sessions.accounts._conn.commit()
    assert _rest_call(server, "/api/messages/inbox", token, secure=True)[0] == 403


def test_conversation_handle_http_contract_and_private_storage(server, monkeypatch, capsys):
    monkeypatch.setattr(srv.proxy_trust_mod, "TRUSTED_PROXIES", {"127.0.0.1"})
    ca, cb, cc = _user(server, "ann"), _user(server, "ben"), _user(server, "cyd")
    _friends(server, "ann", ca, "ben", cb)
    ta, tb, tc = (c.split("=", 1)[1] for c in (ca, cb, cc))
    path = "/api/messages/conversations"
    assert _call(server.port, path)[0] == 403
    assert _call(server.port, path, cookie=ca)[0] == 403
    status, result = _rest_call(server, path, ta, secure=True)
    assert status == 200 and len(result["contacts"]) == 1
    item = result["contacts"][0]
    handle = item["conversation_handle"]
    assert item["username"] == "ben" and len(handle) >= 40
    payload = {"conversation_handle": handle, "body": "hello", "idempotency_key": "unique-request-0001"}
    assert _rest_call(server, "/api/messages/send", tc, secure=True, body=payload)[0] == 403
    assert _rest_call(server, "/api/messages/send", ta, body=payload)[0] == 403
    assert _rest_call(server, "/api/messages/send", ta, secure=True, body={**payload, "to": "ben"})[0] == 400
    first = _rest_call(server, "/api/messages/send", ta, secure=True, body=payload)
    assert first[0] == 200
    assert _rest_call(server, "/api/messages/send", ta, secure=True, body=payload) == first
    assert _rest_call(server, "/api/messages/send", ta, secure=True, body={**payload, "body": "changed"})[0] == 409
    fresh = _rest_call(server, path, ta, secure=True)[1]["contacts"][0]["conversation_handle"]
    assert _rest_call(server, "/api/messages/send", ta, secure=True, body={**payload, "conversation_handle": fresh}) == first
    assert len(_rest_call(server, "/api/messages/inbox", tb, secure=True)[1]["messages"]) == 1
    db = messaging.connect(server.sessions.accounts_db)
    try:
        assert handle not in json.dumps([tuple(r) for r in db.execute("SELECT * FROM msg_handles")])
        assert db.execute("SELECT COUNT(*) FROM msg_messages").fetchone()[0] == 1
        # Revocation wins even for an otherwise idempotent replay.
        messaging.block(db, server.sessions.accounts.login("ben", PASSWORD)["id"], "ann")
        assert _rest_call(server, "/api/messages/send", ta, secure=True, body=payload)[0] == 403
    finally:
        db.close()
    captured = capsys.readouterr()
    assert handle not in captured.out + captured.err


def test_handles_expire_revoke_and_race_without_duplicates(server):
    ca, cb = _user(server, "ann"), _user(server, "ben")
    _friends(server, "ann", ca, "ben", cb)
    db = messaging.connect(server.sessions.accounts_db)
    me = server.sessions.accounts.login("ann", PASSWORD)["id"]
    peer = server.sessions.accounts.login("ben", PASSWORD)["id"]
    try:
        handle = messaging.conversations(db, me, now=100)["contacts"][0]["conversation_handle"]
        with pytest.raises(messaging.MessagingError) as error:
            messaging.send(db, me, None, "old", now=160, conversation_handle=handle, idempotency_key="request-expired-001")
        assert error.value.status == 403
        handle = messaging.conversations(db, me)["contacts"][0]["conversation_handle"]
        barrier = threading.Barrier(2)
        results, errors = [], []
        def concurrent_send():
            connection = messaging.connect(server.sessions.accounts_db)
            try:
                barrier.wait(timeout=5)
                results.append(messaging.send(connection, me, None, "once", conversation_handle=handle,
                                              idempotency_key="concurrent-request-001"))
            except Exception as exc:
                errors.append(exc)
            finally:
                connection.close()
        workers = [threading.Thread(target=concurrent_send) for _ in range(2)]
        for worker in workers: worker.start()
        for worker in workers: worker.join(timeout=10)
        assert not errors and len(results) == 2 and results[0] == results[1]
        assert db.execute("SELECT COUNT(*) FROM msg_messages").fetchone()[0] == 1
        messaging.remove(db, me, "ben")
        messaging.request_contact(db, me, "ben")
        messaging.respond_contact(db, peer, "ann", True)
        with pytest.raises(messaging.MessagingError):
            messaging.send(db, me, None, "old relation", conversation_handle=handle, idempotency_key="old-relation-0001")
        fresh = messaging.conversations(db, me)["contacts"][0]["conversation_handle"]
        messaging.admin_suspend(db, me, "ann", True)
        with pytest.raises(messaging.MessagingError):
            messaging.send(db, me, None, "suspended", conversation_handle=fresh, idempotency_key="suspended-send-001")
    finally:
        db.close()

def test_handle_expiry_after_lock_and_same_timestamp_regrant(server, monkeypatch):
    ca, cb = _user(server, "ann"), _user(server, "ben")
    _friends(server, "ann", ca, "ben", cb)
    me = server.sessions.accounts.login("ann", PASSWORD)["id"]
    peer = server.sessions.accounts.login("ben", PASSWORD)["id"]
    db = messaging.connect(server.sessions.accounts_db)
    try:
        db.execute("UPDATE msg_contacts SET updated=1")
        handle = messaging.conversations(db, me, now=100)["contacts"][0]["conversation_handle"]
        messaging.remove(db, me, "ben")
        messaging.request_contact(db, me, "ben", now=1)
        messaging.respond_contact(db, peer, "ann", True, now=1)
        with pytest.raises(messaging.MessagingError):
            messaging.send(db, me, None, "revoked", now=101, conversation_handle=handle, idempotency_key="regrant-request-001")
        handle = messaging.conversations(db, me, now=100)["contacts"][0]["conversation_handle"]
        clock = [100]
        monkeypatch.setattr(messaging, "time", SimpleNamespace(time=lambda: clock[0]))
        class LockDelay:
            def execute(self, sql, *args):
                result = db.execute(sql, *args)
                if sql == "BEGIN IMMEDIATE":
                    clock[0] = 160  # deterministic simulation of time spent waiting for the lock
                return result
        with pytest.raises(messaging.MessagingError) as error:
            messaging.send(LockDelay(), me, None, "expired while waiting", conversation_handle=handle,
                           idempotency_key="lock-delay-request-001")
        assert error.value.status == 403
        assert db.execute("SELECT COUNT(*) FROM msg_messages").fetchone()[0] == 0
    finally:
        db.close()
