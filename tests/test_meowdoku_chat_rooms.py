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
