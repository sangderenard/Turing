"""The chat server (chat_rooms.py): declared rooms open to their roles, order, tail and after, audited removal, REST."""
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku import chat_rooms  # noqa: E402


@pytest.fixture
def db(tmp_path):
    conn = sqlite3.connect(tmp_path / "accounts.db", isolation_level=None)
    conn.execute("CREATE TABLE account_roles (user_id INTEGER PRIMARY KEY, role TEXT NOT NULL)")
    conn.executemany("INSERT INTO account_roles VALUES (?, ?)", [(1, "admin"), (2, "operator"), (3, "dev")])
    yield conn
    conn.close()


def test_the_reviewers_room_is_for_operators_and_admins_only(db):
    assert [r["room"] for r in chat_rooms.rooms_for(db, 2)] == ["reviewers"]
    assert chat_rooms.rooms_for(db, 3) == [] and chat_rooms.rooms_for(db, 9) == [], "a dev or a member does not see it"
    for n in range(25):
        chat_rooms.post(db, "reviewers", user_id=1 + n % 2, username="admin" if n % 2 == 0 else "opal", body=f"m{n}", now=float(n))
    said = chat_rooms.messages(db, "reviewers", user_id=2, tail=20)
    assert len(said) == 20 and said[0]["body"] == "m5" and said[-1]["body"] == "m24"
    assert [m["body"] for m in chat_rooms.messages(db, "reviewers", user_id=2, after=said[-2]["id"])] == ["m24"]
    for user in (3, 9):
        with pytest.raises(chat_rooms.ChatError, match="admin and operator"):
            chat_rooms.post(db, "reviewers", user_id=user, username="x", body="hi")
        with pytest.raises(chat_rooms.ChatError):
            chat_rooms.messages(db, "reviewers", user_id=user)
    with pytest.raises(chat_rooms.ChatError, match="no such room"):
        chat_rooms.post(db, "lobby", user_id=1, username="admin", body="hi")
    gone = chat_rooms.remove(db, said[0]["id"], admin="admin")
    assert gone["removed"] and gone["body"] == "" and db.execute("SELECT count(*) FROM chat_audit").fetchone()[0] == 1


class Handler:
    def __init__(self, path, user=2, username="opal", capabilities=("dev_ui",), body=None):
        self.path, self._body, self.sent = path, body, None
        self._who = SimpleNamespace(user_id=user, username=username)
        self._capabilities = set(capabilities)
        self.headers = {"Content-Type": "application/json"}

    def _principal(self):
        return self._who

    def _caps(self):
        return self._capabilities

    def _body_json(self, limit):
        return self._body

    def _json(self, document, status=200):
        self.sent = (int(status), json.loads(json.dumps(document)))


def test_the_chat_over_rest(db, tmp_path):
    def ask(url, method="GET", **kw):
        handler = Handler(url, **kw)
        chat_rooms.http_route(handler, url.split("?")[0], method, accounts_db=tmp_path / "accounts.db")
        return handler.sent

    assert ask("/api/chat/rooms")[1]["rooms"][0]["room"] == "reviewers"
    assert ask("/api/chat/rooms", user=3)[1]["rooms"] == []
    assert ask("/api/chat/rooms/reviewers", "POST", body={"body": "hello"})[0] == 200
    assert ask("/api/chat/rooms/reviewers", "POST", user=3, body={"body": "hello"})[0] == 403
    status, answer = ask("/api/chat/rooms/reviewers?tail=20")
    assert status == 200 and answer["tail"] == 20 and [m["body"] for m in answer["messages"]] == ["hello"]
    assert ask("/api/chat/rooms/reviewers?tail=zero")[0] == 400
    assert ask("/api/chat/rooms", user=None)[0] == 401


DEV = frozenset({"dev_ui"})


@pytest.fixture
def people(db):
    db.executemany("INSERT INTO account_roles VALUES (?, ?)", [(4, "dev"), (5, "member")])
    return {"admin": 1, "opal": 2, "devi": 3, "dora": 4, "mem": 5}


def _public(db, room, owner):
    return chat_rooms.share(db, room, user_id=owner, body={"visibility": "public", "group_id": None,
                                                            "expected_revision": 0, "sign_off": True})


def test_dev_level_makes_a_room_that_starts_personal_on_the_dev_site(db, people):
    with pytest.raises(chat_rooms.ChatError, match="dev level"):
        chat_rooms.create(db, user_id=5, username="mem", caps=frozenset(), room="mine", title="Mine")
    made = chat_rooms.create(db, user_id=3, username="devi", caps=DEV, room="shader-lab", title="Shader lab", about="demos")
    assert made["stage"] == "dev" and made["write"] == "members" and made["sharing"]["visibility"] == "personal"
    with pytest.raises(chat_rooms.ChatError, match="exists"):
        chat_rooms.create(db, user_id=3, username="devi", caps=DEV, room="shader-lab", title="again")
    for bad in ("reviewers", "No", "x"):
        with pytest.raises(chat_rooms.ChatError):
            chat_rooms.create(db, user_id=3, username="devi", caps=DEV, room=bad, title="t")
    assert not chat_rooms.may_enter(db, "shader-lab", user_id=4, caps=DEV), "personal: the owner alone"
    assert chat_rooms.may_enter(db, "shader-lab", user_id=1), "an admin always may"
    _public(db, "shader-lab", 3)
    assert chat_rooms.may_enter(db, "shader-lab", user_id=4, caps=DEV), "public on the dev site: dev permission"
    assert not chat_rooms.may_enter(db, "shader-lab", user_id=5), "a member without dev permission does not see a dev room"
    chat_rooms.configure(db, "shader-lab", user_id=3, caps=DEV, changes={"stage": "release"})
    assert chat_rooms.may_enter(db, "shader-lab", user_id=5), "released and public: anyone signed in"
    assert [r["room"] for r in chat_rooms.rooms_for(db, 5)] == ["shader-lab"]
    chat_rooms.configure(db, "shader-lab", user_id=3, caps=DEV, changes={"write": "owner"})
    with pytest.raises(chat_rooms.ChatError, match="owner writes"):
        chat_rooms.post(db, "shader-lab", user_id=5, username="mem", body="hi")
    assert chat_rooms.post(db, "shader-lab", user_id=3, username="devi", body="welcome", caps=DEV)["body"] == "welcome"
    assert chat_rooms.messages(db, "shader-lab", user_id=5)[0]["body"] == "welcome", "members still read"
    with pytest.raises(chat_rooms.ChatError, match="owner changes"):
        chat_rooms.configure(db, "shader-lab", user_id=4, caps=DEV, changes={"title": "mine now"})


def test_a_rooms_set_is_bounded_and_bridges_are_only_listings(db, people):
    chat_rooms.create(db, user_id=3, username="devi", caps=DEV, room="render-bay", title="Render bay")
    set_ = chat_rooms.configure(db, "render-bay", user_id=3, caps=DEV, changes={
        "objects": [{"kind": "shader", "ref": "page-1"}, {"kind": "mesh", "ref": "m-9"}], "bridges": ["photo_status", "photo_status", "nodus_graph"]})
    assert set_["objects"][1] == {"kind": "mesh", "ref": "m-9"} and set_["bridges"] == ["photo_status", "nodus_graph"]
    for bad in ({"objects": [{"kind": "file", "ref": "x"}]}, {"objects": [{"kind": "mesh", "ref": "x"}] * 9},
                {"bridges": ["x"] * 9}, {"bridges": ["no spaces"]}, {"color": "red"}):
        with pytest.raises(chat_rooms.ChatError):
            chat_rooms.configure(db, "render-bay", user_id=3, caps=DEV, changes=bad)
    listed = chat_rooms.tools(db, "render-bay", user_id=3, caps=DEV)
    assert listed["bridges"] == ["photo_status", "nodus_graph"] and listed["forwarded"] == [] and "own capabilities" in listed["say"]


def test_a_member_forwards_tools_of_their_own_connection_and_others_call_them_under_it(db, people):
    chat_rooms.create(db, user_id=3, username="devi", caps=DEV, room="music", title="Music")
    _public(db, "music", 3)
    chat_rooms.configure(db, "music", user_id=3, caps=DEV, changes={"stage": "release"})
    with pytest.raises(chat_rooms.ChatError, match="does not offer"):
        chat_rooms.forward(db, "music", user_id=3, username="devi", caps=DEV, provider="suno", tool_names=["generate", "delete"],
                           offered=["generate", "status"])
    grant = chat_rooms.forward(db, "music", user_id=3, username="devi", caps=DEV, provider="suno",
                               tool_names=["generate"], hours=2, offered=["generate", "status"], now=1000.0)
    assert chat_rooms.tools(db, "music", user_id=5, now=1001.0)["forwarded"][0]["tools"] == ["generate"]
    calls = []

    def call(owner, provider, name, arguments):
        calls.append((owner, provider, name, arguments))
        return {"song": "made"}

    answer = chat_rooms.forwarded_call(db, "music", grant["grant"], user_id=5, username="mem", name="generate",
                                       arguments={"prompt": "cats"}, call=call, now=1002.0)
    assert answer == {"song": "made"} and calls == [(3, "suno", "generate", {"prompt": "cats"})], "under the lender's connection"
    with pytest.raises(chat_rooms.ChatError, match="lends only"):
        chat_rooms.forwarded_call(db, "music", grant["grant"], user_id=5, username="mem", name="status", arguments={}, call=call,
                                  now=1003.0)
    for n in range(chat_rooms.CALLS_PER_HOUR - 1):
        chat_rooms.forwarded_call(db, "music", grant["grant"], user_id=5, username="mem", name="generate", arguments={},
                                  call=call, now=1004.0 + n)
    with pytest.raises(chat_rooms.ChatError) as limited:
        chat_rooms.forwarded_call(db, "music", grant["grant"], user_id=5, username="mem", name="generate", arguments={},
                                  call=call, now=1100.0)
    assert limited.value.status == 429
    assert db.execute("SELECT count(*), sum(ok) FROM chat_grant_calls WHERE caller = 'mem'").fetchone() == (30, 30)
    with pytest.raises(chat_rooms.ChatError, match="expired"):
        chat_rooms.forwarded_call(db, "music", grant["grant"], user_id=4, username="dora", caps=DEV, name="generate",
                                  arguments={}, call=call, now=1000.0 + 2 * 3600.0 + 1)
    with pytest.raises(chat_rooms.ChatError, match="lender"):
        chat_rooms.revoke_grant(db, "music", grant["grant"], user_id=5)
    chat_rooms.revoke_grant(db, "music", grant["grant"], user_id=3)
    with pytest.raises(chat_rooms.ChatError, match="revoked"):
        chat_rooms.forwarded_call(db, "music", grant["grant"], user_id=4, username="dora", caps=DEV, name="generate",
                                  arguments={}, call=call, now=1200.0)

    def failing(owner, provider, name, arguments):
        raise RuntimeError("provider down")
    again = chat_rooms.forward(db, "music", user_id=3, username="devi", caps=DEV, provider="suno", tool_names=["generate"],
                               offered=["generate"], now=2000.0)
    with pytest.raises(RuntimeError):
        chat_rooms.forwarded_call(db, "music", again["grant"], user_id=4, username="dora", caps=DEV, name="generate",
                                  arguments={}, call=failing, now=2001.0)
    assert db.execute("SELECT ok FROM chat_grant_calls WHERE caller = 'dora'").fetchall() == [(0,)], "a failed call is audited too"


def test_owned_rooms_over_rest_with_a_connection_service(db, people, tmp_path):
    class Connections:
        def __init__(self):
            self.calls = []

        def tools(self, owner, provider):
            return [{"name": "generate"}] if owner == 3 else []

        def call(self, owner, provider, name, arguments):
            self.calls.append((owner, name))
            return {"ok": True}

    service = Connections()

    def ask(url, method="GET", connections=service, **kw):
        handler = Handler(url, **kw)
        chat_rooms.http_route(handler, url.split("?")[0], method, accounts_db=tmp_path / "accounts.db", connections=connections)
        return handler.sent

    status, made = ask("/api/chat/rooms", "POST", user=3, username="devi", body={"room": "jam", "title": "Jam"})
    assert status == 200 and made["settings"]["owner_name"] == "devi"
    assert ask("/api/chat/rooms", "POST", user=5, username="mem", capabilities=(), body={"room": "x-y-z", "title": "t"})[0] == 403
    assert ask("/api/chat/rooms/jam/access", "POST", user=3, username="devi", body={
        "visibility": "public", "group_id": None, "expected_revision": 0, "sign_off": True})[1]["sharing"]["visibility"] == "public"
    assert ask("/api/chat/rooms/jam/settings", "POST", user=3, username="devi", body={"stage": "release", "bridges": ["photo_status"]})[0] == 200
    assert ask("/api/chat/rooms/jam/settings", user=5, username="mem", capabilities=())[1]["settings"]["bridges"] == ["photo_status"]
    status, lent = ask("/api/chat/rooms/jam/forward", "POST", user=3, username="devi", body={"provider": "suno", "tools": ["generate"]})
    assert status == 200 and lent["tools"] == ["generate"]
    assert ask("/api/chat/rooms/jam/forward", "POST", user=5, username="mem", capabilities=(),
               body={"provider": "suno", "tools": ["generate"]})[0] == 409, "mem has no connection to lend"
    status, called = ask(f"/api/chat/rooms/jam/forward/{lent['grant']}/call", "POST", user=5, username="mem", capabilities=(),
                         body={"name": "generate", "arguments": {}})
    assert status == 200 and called == {"result": {"ok": True}} and service.calls == [(3, "generate")]
    assert ask("/api/chat/rooms/jam/tools", user=5, username="mem", capabilities=())[1]["forwarded"][0]["owner"] == "devi"
    assert ask(f"/api/chat/rooms/jam/forward/{lent['grant']}/revoke", "POST", user=3, username="devi", body={})[1]["revoked"]
    assert ask("/api/chat/rooms/jam/forward", "POST", connections=None, user=3, username="devi",
               body={"provider": "suno", "tools": ["generate"]})[0] == 503
