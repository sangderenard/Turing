"""Reactions and stickers in a room (chat_rooms.py): the site's emoji as reactions, and a writer's own studio photo as a
sticker that everyone in the room sees through the room, never another account's photo."""
import io
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku import chat_rooms  # noqa: E402

ADMIN, OPAL, MEG, SVC = 1, 2, 3, 7
KINDS = ("frog", "cat", "fox")
MINE, THEIRS = "a" * 64, "b" * 64
PICTURE = "preview-000012.png"


@pytest.fixture
def db(tmp_path):
    conn = sqlite3.connect(tmp_path / "accounts.db", isolation_level=None)
    conn.execute("CREATE TABLE account_roles (user_id INTEGER PRIMARY KEY, role TEXT NOT NULL)")
    conn.executemany("INSERT INTO account_roles VALUES (?, ?)", [(ADMIN, "admin"), (OPAL, "operator"), (MEG, "member"), (SVC, "service")])
    yield conn
    conn.close()


def say(db, user=MEG, name="meg", body="hello", **kw):
    return chat_rooms.post(db, "agents", user_id=user, username=name, body=body, **kw)


def test_reactions_are_the_sites_emoji_one_per_account_and_kind(db):
    made = say(db)
    react = lambda user, name, kind, on=True: chat_rooms.react(db, "agents", user_id=user, username=name, message_id=made["id"],
                                                              kind=kind, on=on, kinds=KINDS)
    assert react(OPAL, "opal", "frog") == {"id": made["id"], "reactions": {"frog": ["opal"]}}
    react(OPAL, "opal", "frog")
    react(ADMIN, "admin", "frog")
    assert react(MEG, "meg", "cat")["reactions"] == {"frog": ["opal", "admin"], "cat": ["meg"]}, "twice is still once"
    assert chat_rooms.messages(db, "agents", user_id=MEG, tail=1)[0]["reactions"] == {"frog": ["opal", "admin"], "cat": ["meg"]}
    assert react(OPAL, "opal", "frog", on=False)["reactions"] == {"frog": ["admin"], "cat": ["meg"]}
    for kind in ("dragon", "", None):
        with pytest.raises(chat_rooms.ChatError, match="site's emoji"):
            react(OPAL, "opal", kind)
    with pytest.raises(chat_rooms.ChatError):
        chat_rooms.react(db, "agents", user_id=SVC, username="svc", message_id=made["id"], kind="frog", kinds=KINDS)
    with pytest.raises(chat_rooms.ChatError) as elsewhere:
        chat_rooms.react(db, "agents", user_id=OPAL, username="opal", message_id=made["id"] + 99, kind="frog", kinds=KINDS)
    assert elsewhere.value.status == 404
    plain = say(db, body="no reactions")
    assert "reactions" not in chat_rooms.messages(db, "agents", user_id=MEG, after=plain["id"] - 1)[0]
    chat_rooms.remove(db, made["id"], admin="admin")
    assert "reactions" not in chat_rooms.messages(db, "agents", user_id=MEG, tail=5)[0], "a removed message keeps none"
    with pytest.raises(chat_rooms.ChatError):
        react(OPAL, "opal", "cat")


def test_a_sticker_is_one_of_the_writers_own_photos(db):
    owned = lambda digest: digest == MINE
    made = say(db, body=None, sticker={"kind": "photo", "digest": MINE, "name": PICTURE}, photo_owned=owned)
    assert made["sticker"] == {"kind": "photo"} and made["body"] == "", "the digest is never shown"
    assert chat_rooms.sticker_of(db, "agents", made["id"], user_id=OPAL) == {"kind": "photo", "digest": MINE, "name": PICTURE}
    with pytest.raises(chat_rooms.ChatError) as theirs:
        say(db, sticker={"kind": "photo", "digest": THEIRS, "name": PICTURE}, photo_owned=owned)
    assert theirs.value.status == 403, "nobody posts another account's photo"
    for bad in ({"kind": "mesh", "digest": MINE, "name": PICTURE}, {"kind": "photo", "digest": "xyz", "name": PICTURE},
                {"kind": "photo", "digest": MINE, "name": "../config.json"}, {"kind": "photo", "digest": MINE}):
        with pytest.raises(chat_rooms.ChatError):
            say(db, sticker=bad, photo_owned=owned)
    with pytest.raises(chat_rooms.ChatError, match="or a sticker"):
        say(db, body="   ")
    with pytest.raises(chat_rooms.ChatError):
        chat_rooms.sticker_of(db, "agents", made["id"], user_id=SVC)
    chat_rooms.remove(db, made["id"], admin="admin")
    with pytest.raises(chat_rooms.ChatError) as gone:
        chat_rooms.sticker_of(db, "agents", made["id"], user_id=OPAL)
    assert gone.value.status == 404


class Handler:
    def __init__(self, path, user, username, body=None):
        self.path, self._body, self.sent, self.status, self.headers_out = path, body, None, None, {}
        self._who = SimpleNamespace(user_id=user, username=username)
        self.headers = {"Content-Type": "application/json"}
        self.wfile = io.BytesIO()

    def _principal(self):
        return self._who

    def _caps(self):
        return set()

    def _body_json(self, limit):
        return self._body

    def _json(self, document, status=200):
        self.sent = (int(status), json.loads(json.dumps(document)))

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.headers_out[key] = value

    def end_headers(self):
        pass


def test_reactions_and_stickers_over_rest(db, tmp_path):
    opened = []

    def ask(url, method="GET", user=MEG, username="meg", body=None, **kw):
        handler = Handler(url, user, username, body)
        chat_rooms.http_route(handler, url.split("?")[0], method, accounts_db=tmp_path / "accounts.db", **kw)
        return handler

    hosted = {"reaction_kinds": lambda: KINDS, "photo_owned": lambda digest: digest == MINE,
              "open_photo": lambda digest, name: opened.append((digest, name)) or io.BytesIO(b"\x89PNG fake")}
    status, answer = ask("/api/chat/rooms/agents", "POST", body={"sticker": {"kind": "photo", "digest": MINE, "name": PICTURE}},
                         **hosted).sent
    assert status == 200 and answer["message"]["sticker"] == {"kind": "photo"}
    message = answer["message"]["id"]
    picture = ask(f"/api/chat/rooms/agents/sticker/{message}", user=OPAL, username="opal", **hosted)
    assert picture.status == 200 and picture.wfile.getvalue() == b"\x89PNG fake" and opened == [(MINE, PICTURE)]
    assert picture.headers_out["Content-Type"] == "image/png"
    assert ask(f"/api/chat/rooms/agents/sticker/{message}", user=SVC, username="svc", **hosted).sent[0] == 403
    assert ask("/api/chat/rooms/agents/sticker/999", **hosted).sent[0] == 404
    reacted = ask("/api/chat/rooms/agents/react", "POST", user=OPAL, username="opal", body={"id": message, "kind": "fox"}, **hosted)
    assert reacted.sent == (200, {"id": message, "reactions": {"fox": ["opal"]}})
    assert ask("/api/chat/rooms/agents/react", "POST", body={"id": message, "kind": "fox"}).sent[0] == 503, "no kinds: none"
    assert ask("/api/chat/rooms/agents", "POST", body={"body": "x", "sticker": {"kind": "photo", "digest": MINE, "name": PICTURE}}
               ).sent[0] == 503, "no photo store: no stickers"
