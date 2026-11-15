"""@mentions (mentions.py): a name in a chat post or a comment tells that account through the host's notify, only when it
may read the place, never the writer, never twice."""
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku import chat_rooms, comments, mentions  # noqa: E402

ADMIN, OPERATOR, DEV, GEMINI = 1, 2, 3, 4


@pytest.fixture
def db(tmp_path):
    conn = sqlite3.connect(tmp_path / "accounts.db", isolation_level=None)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL)")
    conn.executemany("INSERT INTO users VALUES (?, ?)", [(ADMIN, "admin"), (OPERATOR, "opal"), (DEV, "devon"), (GEMINI, "gemini")])
    conn.execute("CREATE TABLE account_roles (user_id INTEGER PRIMARY KEY, role TEXT NOT NULL)")
    conn.executemany("INSERT INTO account_roles VALUES (?, ?)", [(ADMIN, "admin"), (OPERATOR, "operator"), (DEV, "dev"),
                                                                  (GEMINI, "operator")])
    yield conn
    conn.close()


def test_names_are_read_as_mentions_only_where_written_as_one():
    assert mentions.names_in("@gemini, look at this. @Opal: and mail me at a@b.com; @@x; @gemini again; end @devon.") == \
        ["gemini", "Opal", "devon"], "an address, a doubled @ and a repeat are not mentions; a sentence's '.' is not the name's"
    assert mentions.names_in("@operator-claude please") == ["operator-claude"]
    assert len(mentions.names_in(" ".join(f"@u{n}" for n in range(30)))) == mentions.MAX_MENTIONS
    assert mentions.names_in(None) == [] and mentions.names_in("no names here") == []


def test_only_real_readers_other_than_the_writer_are_told(db):
    sent = []
    told = mentions.tell(db, writer_id=OPERATOR, writer="opal", body="@Gemini @opal @nobody @devon hello",
                         where="the room x", answer="chat_post room=x",
                         may_read=lambda account: account["username"] != "devon",
                         notify=lambda names, text, kind, ref: sent.append((names, text, kind, ref)), kind="room", ref="x")
    assert told == ["gemini"], "the writer, an unknown name and an account that may not read the place are left out"
    assert sent == [(["gemini"], '@opal mentioned you in the room x: "@Gemini @opal @nobody @devon hello" Answer there: chat_post room=x.',
                     "room", "x")]
    assert mentions.tell(db, writer_id=OPERATOR, writer="opal", body="@gemini", where="w", answer="a", may_read=lambda a: True,
                         notify=lambda *args: sent.append(args), kind="room", ref="x", already=["Gemini"]) == [], "told already"
    assert mentions.tell(db, writer_id=OPERATOR, writer="opal", body="@gemini", where="w", answer="a", may_read=lambda a: True,
                         notify=None, kind="room", ref="x") == []
    long = mentions.notice("opal", "w", "a", "x" * 1000)
    assert len(long) < 400 and "…" in long


class Handler:
    def __init__(self, path, user, username, body):
        self.path, self._body, self.sent = path, body, None
        self._who = SimpleNamespace(user_id=user, username=username)
        self.headers = {"Content-Type": "application/json"}

    def _principal(self):
        return self._who

    def _caps(self):
        return {"dev_ui"}

    def _body_json(self, limit):
        return self._body

    def _json(self, document, status=200):
        self.sent = (int(status), json.loads(json.dumps(document)))


def test_a_mention_in_a_room_tells_only_who_may_enter_it(db, tmp_path):
    sent = []

    def post(room, body, user=OPERATOR, username="opal"):
        handler = Handler(f"/api/chat/rooms/{room}", user, username, {"body": body})
        chat_rooms.http_route(handler, handler.path, "POST", accounts_db=tmp_path / "accounts.db",
                              notify=lambda names, text, kind, ref: sent.append((names, kind, ref)),
                              caps_for=lambda account: frozenset())
        return handler.sent

    status, answer = post("reviewers", "@gemini @devon please look")
    assert status == 200 and answer["mentioned"] == ["gemini"], "devon (dev) may not enter the operators' and admins' room"
    assert sent == [(["gemini"], "room", "reviewers")]
    status, answer = post("agents", "@devon @gemini hello")
    assert answer["mentioned"] == ["devon", "gemini"] and sent[-1] == (["devon", "gemini"], "room", "agents")
    assert post("agents", "no one named")[1]["mentioned"] == []


def test_a_mention_in_a_thread_tells_a_reader_who_was_not_already_told(db, tmp_path):
    sent = []
    kind = comments.Kind(may_read=lambda db, who, ref: True if who["username"] != "devon" else "not for devon",
                         may_write=lambda db, who, ref: True, tell=lambda db, ref, comment: ["admin"],
                         label=lambda db, ref: f"note {ref}")
    handler = Handler("/api/comments", OPERATOR, "opal", {"kind": "note", "ref": "n1", "body": "@admin @gemini @devon see this"})
    comments.http_route(handler, "/api/comments", "POST", accounts_db=tmp_path / "accounts.db", kinds={"note": kind},
                        notify=lambda names, text, k, ref: sent.append((names, text, k, ref)), caps_for=lambda a: frozenset())
    status, answer = handler.sent
    assert status == 200 and answer["told"] == ["admin"] and answer["mentioned"] == ["gemini"], \
        "admin is told as a participant already, devon may not read it"
    assert [s[0] for s in sent] == [["admin"], ["gemini"]] and sent[1][2:] == ("note", "n1")
    assert sent[1][1].startswith("@opal mentioned you in the thread of note n1:") and "comments_post kind=note ref=n1" in sent[1][1]
