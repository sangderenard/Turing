"""Invitations (invitations.py): a one-time code in a link registers someone; only its hash is kept; who invited whom."""
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku import invitations as inv  # noqa: E402
from meowdoku import login_guard  # noqa: E402

ADMIN, OPAL, INTERN = 1, 2, 6


@pytest.fixture
def db(tmp_path):
    conn = sqlite3.connect(tmp_path / "accounts.db", isolation_level=None)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL)")
    conn.executemany("INSERT INTO users VALUES (?, ?)", [(ADMIN, "admin"), (OPAL, "opal"), (INTERN, "kiosk")])
    conn.execute("CREATE TABLE account_roles (user_id INTEGER PRIMARY KEY, role TEXT NOT NULL)")
    conn.executemany("INSERT INTO account_roles VALUES (?, ?)", [(ADMIN, "admin"), (INTERN, "intern")])
    yield conn
    conn.close()


class Taken(Exception):
    status = 409


def signup_into(db):
    def signup(username, password):
        if len(password) < 8:
            raise Taken("password must be at least 8 characters")
        if db.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            raise Taken("that username is taken")
        return {"id": db.execute("INSERT INTO users (username) VALUES (?)", (username,)).lastrowid, "username": username}
    return signup


def test_an_invitation_registers_one_person_and_keeps_only_a_hash(db):
    made, code = inv.make(db, user_id=ADMIN, username="admin", note="  Sam   from work ", email="sam@example.org", now=100.0)
    assert len(code) >= 43 and made["state"] == "waiting" and made["note"] == "Sam from work"
    assert made["expires"] == 100.0 + inv.DEFAULT_DAYS * 86400
    stored = db.execute("SELECT * FROM site_invitations").fetchall()
    assert code not in json.dumps(stored) and inv.digest(code) in json.dumps(stored), "the code is not kept, its hash is"
    assert "token_hash" not in made, "the hash never leaves the server either"
    assert inv.check(db, code, now=200.0) == {"inviter": "admin", "chain": ["admin"], "expires": made["expires"]}
    signup = signup_into(db)
    with pytest.raises(Taken):
        inv.redeem(db, code, username="opal", password="correcthorse1", signup=signup, now=300.0)
    assert inv.mine(db, user_id=ADMIN, now=300.0)[0]["state"] == "waiting", "a refused sign-up leaves the invitation waiting"
    done = inv.redeem(db, code, username="sam", password="correcthorse1", signup=signup, now=300.0)
    assert done == {"account": {"id": done["account"]["id"], "username": "sam"}, "invited_by": "admin"}
    record = inv.mine(db, user_id=ADMIN, now=300.0)[0]
    assert (record["state"], record["used_by"], record["used_at"]) == ("used", "sam", 300.0)
    for later in (lambda: inv.check(db, code, now=301.0),
                  lambda: inv.redeem(db, code, username="sam2", password="correcthorse1", signup=signup, now=301.0)):
        with pytest.raises(inv.InvitationError) as used:
            later()
        assert used.value.status == 404 and "not valid" in used.value.message, "one use"


def test_the_chain_of_inviters_and_who_may_invite(db):
    signup = signup_into(db)
    _, first = inv.make(db, user_id=ADMIN, username="admin", note="Sam", now=1.0)
    sam = inv.redeem(db, first, username="sam", password="correcthorse1", signup=signup, now=2.0)["account"]
    _, second = inv.make(db, user_id=sam["id"], username="sam", note="Lee", now=3.0)
    assert inv.check(db, second, now=4.0)["chain"] == ["sam", "admin"], "who invited the inviter, nearest first"
    with pytest.raises(inv.InvitationError) as guest:
        inv.make(db, user_id=INTERN, username="kiosk", note="x")
    assert guest.value.status == 403
    for note, email, days in (("", "", 14), ("x" * 81, "", 14), ("ok", "not an address", 14), ("ok", "", 0),
                              ("ok", "", inv.MAX_DAYS + 1), ("ok", "", True), ("ok", "", "14")):
        with pytest.raises(inv.InvitationError):
            inv.make(db, user_id=OPAL, username="opal", note=note, email=email, days=days)
    for n in range(inv.MAX_WAITING):
        inv.make(db, user_id=OPAL, username="opal", note=f"n{n}", now=10.0)
    with pytest.raises(inv.InvitationError) as full:
        inv.make(db, user_id=OPAL, username="opal", note="one more", now=10.0)
    assert full.value.status == 429


def test_expiry_and_taking_back(db):
    made, code = inv.make(db, user_id=OPAL, username="opal", note="Kim", days=1, now=0.0)
    assert inv.mine(db, user_id=OPAL, now=86401.0)[0]["state"] == "expired"
    with pytest.raises(inv.InvitationError):
        inv.check(db, code, now=86401.0)
    other, other_code = inv.make(db, user_id=OPAL, username="opal", note="Lou", now=0.0)
    with pytest.raises(inv.InvitationError) as stranger:
        inv.revoke(db, other["id"], user_id=INTERN, username="kiosk", now=1.0)
    assert stranger.value.status == 403
    assert inv.revoke(db, other["id"], user_id=ADMIN, username="admin", now=1.0)["state"] == "taken back", "an admin may"
    with pytest.raises(inv.InvitationError):
        inv.check(db, other_code, now=2.0)
    with pytest.raises(inv.InvitationError) as again:
        inv.revoke(db, other["id"], user_id=OPAL, username="opal", now=2.0)
    assert again.value.status == 409
    for wrong in ("nope", "0" * 12):
        with pytest.raises(inv.InvitationError) as missing:
            inv.revoke(db, wrong, user_id=OPAL, username="opal")
        assert missing.value.status == 404


class Handler:
    def __init__(self, path, user=None, username=None, body=None):
        self.path, self._body, self.sent = path, body, None
        self._who = SimpleNamespace(user_id=user, username=username)
        self._client_ip, self.client_address = "203.0.113.9", ("203.0.113.9", 0)
        self.headers = {"Content-Type": "application/json"}

    def _principal(self):
        return self._who

    def _body_json(self, limit):
        return self._body

    def _json(self, document, status=200):
        self.sent = (int(status), json.loads(json.dumps(document)))


def test_invitations_over_rest_with_the_login_guard(db, tmp_path):
    signed_in = []
    accounts_db = tmp_path / "accounts.db"

    def ask(path, method="GET", **kw):
        handler = Handler(path, **kw)
        inv.http_route(handler, path, method, accounts_db=accounts_db, signup=signup_into(db), sign_in=signed_in.append,
                       account_error=Taken, guard=login_guard)
        return handler.sent

    assert ask("/api/invitations")[0] == 401
    assert ask("/api/invitations", "POST", body={"note": "Sam"})[0] == 401
    status, made = ask("/api/invitations", "POST", user=OPAL, username="opal", body={"note": "Sam", "days": 7})
    assert status == 201 and made["path"] == "/invited.html#code=" + made["code"], "the code rides in the fragment"
    assert [i["note"] for i in ask("/api/invitations", user=OPAL, username="opal")[1]["invitations"]] == ["Sam"]
    assert made["code"] not in json.dumps(ask("/api/invitations", user=OPAL, username="opal")[1]), "listed without its code"
    assert ask("/api/invitations/check", "POST", body={"code": made["code"]})[1]["inviter"] == "opal", "no account needed"
    assert ask("/api/invitations/redeem", "POST", body={"code": made["code"], "username": "opal", "password": "correcthorse1"})[0] == 409
    status, done = ask("/api/invitations/redeem", "POST", body={"code": made["code"], "username": "sam", "password": "correcthorse1"})
    assert status == 201 and done["invited_by"] == "opal" and signed_in == [done["account"]], "registered and signed in"
    for n in range(login_guard.PAIR_LIMIT):
        assert ask("/api/invitations/check", "POST", body={"code": f"wrong-code-{n:020d}"})[0] == 404
    status, slowed = ask("/api/invitations/check", "POST", body={"code": made["code"]})
    assert status == 429 and slowed["retry_after"] >= 1, "wrong codes are slowed by the login guard"


def test_the_public_site_registers_by_invitation_only(tmp_path):
    """On the shader tier (the public site: sign-up closed) an invitation still registers and signs in, over the server."""
    import http.client
    import threading
    from http.server import ThreadingHTTPServer
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
    from meowdoku import server as srv
    from test_meowdoku_access import _web
    web = _web(tmp_path)
    sessions = srv.Sessions(web, None)
    sessions.lifecycle = None
    (web.parent / "access.json").write_text(json.dumps({"policy": {"lan": "shader"}, "public_ip": []}), encoding="utf-8")
    sessions.access._stamp = object()
    owner = sessions.accounts.ensure_user("ownercat", "correcthorse1")
    db = sqlite3.connect(sessions.accounts_db, isolation_level=None)
    _, code = inv.make(db, user_id=owner, username="ownercat", note="Sam")
    db.close()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def post(path, body):
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=30)
        conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        answer = json.loads(r.read() or b"{}")
        cookies = [v.split(";")[0].split("=")[0] for k, v in r.getheaders() if k.lower() == "set-cookie"]
        conn.close()
        return r.status, answer, cookies

    try:
        assert post("/api/account/signup", {"username": "sam", "password": "correcthorse1"})[0] == 403, "sign-up is closed here"
        status, checked, _ = post("/api/invitations/check", {"code": code})
        assert status == 200 and checked["inviter"] == "ownercat"
        status, done, cookies = post("/api/invitations/redeem", {"code": code, "username": "sam", "password": "correcthorse1"})
        assert status == 201 and done["invited_by"] == "ownercat" and srv.ACCOUNT_COOKIE in cookies, "registered and signed in"
        assert post("/api/invitations", {"note": "x"})[0] in (401, 403), "making one still needs an account"
    finally:
        httpd.shutdown()
        httpd.server_close()
