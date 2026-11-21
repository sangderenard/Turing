"""Comments under a post (comments.py): threads by (kind, ref), the owning kind's say on who reads and writes, the tail and
after reads, an admin's audited removal, and the REST route."""
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku import comments  # noqa: E402


@pytest.fixture
def db(tmp_path):
    conn = sqlite3.connect(tmp_path / "accounts.db", isolation_level=None)
    yield conn
    conn.close()


def test_a_thread_keeps_order_tail_and_after_and_replies_stay_in_their_thread(db):
    first = comments.post(db, "note", "p1", user_id=1, username="ann", body="  hello  ", now=1.0)
    assert first["body"] == "hello" and first["reply_to"] is None and not first["removed"]
    for n in range(30):
        comments.post(db, "note", "p1", user_id=2, username="bo", body=f"m{n}", now=2.0 + n)
    comments.post(db, "note", "p2", user_id=2, username="bo", body="elsewhere", now=50.0)
    assert [c["body"] for c in comments.thread(db, "note", "p1", tail=3)] == ["m27", "m28", "m29"]
    assert len(comments.thread(db, "note", "p1", tail=comments.tail_of("all"))) == 31
    later = comments.thread(db, "note", "p1", after=first["id"])
    assert len(later) == 30 and later[0]["body"] == "m0"
    reply = comments.post(db, "note", "p1", user_id=1, username="ann", body="re", reply_to=first["id"], now=60.0)
    assert reply["reply_to"] == first["id"]
    with pytest.raises(comments.CommentError, match="same thread"):
        comments.post(db, "note", "p2", user_id=1, username="ann", body="re", reply_to=first["id"])
    assert comments.writers(db, "note", "p1") == {"ann", "bo"}
    assert [ref for ref, _ in comments.active(db, "note")] == ["p1", "p2"], "the most recently written first"
    for kind, ref in (("No Caps", "x"), ("note", "")):
        with pytest.raises(comments.CommentError):
            comments.post(db, kind, ref, user_id=1, username="ann", body="x")
    with pytest.raises(comments.CommentError):
        comments.post(db, "note", "p1", user_id=1, username="ann", body=" ")
    with pytest.raises(comments.CommentError):
        comments.tail_of("0")


def test_an_admin_removes_a_comment_and_the_act_is_audited(db):
    made = comments.post(db, "note", "p1", user_id=1, username="ann", body="oops")
    gone = comments.remove(db, made["id"], admin="root")
    assert gone["removed"] and gone["body"] == ""
    assert comments.thread(db, "note", "p1")[0]["body"] == ""
    assert db.execute("SELECT admin, action, comment_id FROM cm_audit").fetchall() == [("root", "remove", made["id"])]
    with pytest.raises(comments.CommentError):
        comments.remove(db, made["id"], admin="root")


class Handler:
    def __init__(self, path, user=1, username="ann", capabilities=("dev_ui",), body=None):
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


def test_the_route_asks_the_kind_and_tells_its_people(tmp_path):
    path = tmp_path / "accounts.db"
    kind = comments.Kind(may_read=lambda db, who, ref: True,
                         may_write=lambda db, who, ref: True if who["username"] != "eve" else "eve may not",
                         tell=lambda db, ref, made: {"ann", "bo", "cy"}, label=lambda db, ref: f"post {ref}")
    told = []

    def ask(url, method="GET", **kw):
        handler = Handler(url, **kw)
        comments.http_route(handler, url.split("?")[0], method, accounts_db=path, kinds={"note": kind},
                            notify=lambda names, text, k, ref: told.append((names, text, k, ref)))
        return handler.sent

    status, answer = ask("/api/comments", "POST", body={"kind": "note", "ref": "p1", "body": "hi all"})
    assert status == 200 and answer["told"] == ["bo", "cy"]
    assert told == [(["bo", "cy"], "ann wrote in the thread of post p1: hi all", "note", "p1")]
    assert ask("/api/comments", "POST", username="eve", body={"kind": "note", "ref": "p1", "body": "x"}) == (403, {"error": "eve may not"})
    assert ask("/api/comments", "POST", body={"kind": "other", "ref": "p1", "body": "x"})[0] == 404
    status, answer = ask("/api/comments?kind=note&ref=p1&tail=5")
    assert status == 200 and [c["body"] for c in answer["comments"]] == ["hi all"]
    assert ask("/api/comments?kind=note&ref=p1&after=x")[0] == 400
    assert ask("/api/comments", user=None)[0] == 401
    first = answer["comments"][0]["id"]
    assert ask("/api/comments/remove", "POST", body={"id": first})[0] == 403
    assert ask("/api/comments/remove", "POST", capabilities=("account_manage",), body={"id": first})[0] == 200


def test_a_pages_thread_follows_the_pages_own_reading_rule(db):
    from meowdoku import authored_pages
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)")
    db.execute("CREATE TABLE authored_pages (id TEXT PRIMARY KEY, owner_id INTEGER, stage TEXT, title TEXT)")
    db.executemany("INSERT INTO users VALUES (?, ?)", [(1, "ann"), (2, "bo")])
    db.executemany("INSERT INTO authored_pages VALUES (?, ?, ?, ?)", [("pub", 1, "release", "Shader demo"), ("wip", 1, "dev", "Draft")])
    kind = authored_pages.comment_kind()
    bo, ann = {"user_id": 2, "username": "bo", "caps": frozenset()}, {"user_id": 1, "username": "ann", "caps": frozenset()}
    assert kind.may_write(db, bo, "pub") is True and kind.may_read(db, bo, "pub") is True, "a released page: every signed-in account"
    assert isinstance(kind.may_read(db, bo, "wip"), str) and kind.may_write(db, ann, "wip") is True, "a dev page: its author alone"
    with pytest.raises(comments.CommentError):
        kind.may_read(db, bo, "gone")
    comments.post(db, "page", "pub", user_id=2, username="bo", body="lovely")
    assert kind.tell(db, "pub", None) == {"ann", "bo"} and kind.label(db, "pub") == "the page Shader demo"
