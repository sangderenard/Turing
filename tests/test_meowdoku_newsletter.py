"""The newsletter (newsletter.py): anyone reads what is published, an operator posts directly, anyone else's article and
later edits wait for an operator, and an article's key edits it and names its original poster."""
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku import comments, newsletter  # noqa: E402

ADMIN, OPAL, MEG, NED = 1, 2, 3, 4


@pytest.fixture
def db(tmp_path):
    conn = sqlite3.connect(tmp_path / "accounts.db", isolation_level=None)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL)")
    conn.executemany("INSERT INTO users VALUES (?, ?)", [(ADMIN, "admin"), (OPAL, "opal"), (MEG, "meg"), (NED, "ned")])
    conn.execute("CREATE TABLE account_roles (user_id INTEGER PRIMARY KEY, role TEXT NOT NULL)")
    conn.executemany("INSERT INTO account_roles VALUES (?, ?)", [(ADMIN, "admin"), (OPAL, "operator")])
    yield conn
    conn.close()


def meg(**kw):
    return {"user_id": MEG, "username": "meg", "operator": False, **kw}


def test_a_members_article_waits_for_an_operator_and_then_anyone_reads_it(db):
    made, key = newsletter.submit(db, **meg(), title="  Notes   from the lab ", body="# Hello\n\nfirst words")
    assert made["status"] == "pending" and made["title"] == "Notes from the lab" and len(key) >= 30
    assert made["key_hash"] != key and key not in json.dumps(made), "only a hash of the key is kept"
    assert newsletter.listing(db)["published"] == []
    with pytest.raises(newsletter.NewsletterError) as hidden:
        newsletter.read(db, made["id"])
    assert hidden.value.status == 404, "a waiting article is not shown to be there"
    assert newsletter.read(db, made["id"], user_id=MEG)["body"] == "# Hello\n\nfirst words"
    assert [a["id"] for a in newsletter.listing(db, user_id=OPAL, operator=True)["waiting"]] == [made["id"]]
    done = newsletter.decide(db, made["id"], username="opal", decision="published", note="welcome")
    assert done["status"] == "published" and done["decided_by"] == "opal" and done["published_at"]
    assert [a["title"] for a in newsletter.listing(db)["published"]] == ["Notes from the lab"]
    assert newsletter.read(db, made["id"])["body"].startswith("# Hello"), "published: anyone reads it"
    with pytest.raises(newsletter.NewsletterError, match="not waiting"):
        newsletter.decide(db, made["id"], username="opal", decision="declined")


def test_an_operator_publishes_at_once(db):
    made, _ = newsletter.submit(db, user_id=OPAL, username="opal", operator=True, title="Server notes", body="all well")
    assert made["status"] == "published" and newsletter.listing(db)["published"][0]["id"] == made["id"]


def test_an_edit_waits_for_review_while_the_published_text_stays(db):
    made, key = newsletter.submit(db, **meg(), title="T", body="old")
    newsletter.decide(db, made["id"], username="opal", decision="published")
    with pytest.raises(newsletter.NewsletterError) as wrong:
        newsletter.edit(db, made["id"], **meg(), key="not-the-key", title="T", body="new")
    assert wrong.value.status == 403
    version = newsletter.edit(db, made["id"], user_id=NED, username="ned", operator=False, key=key, title="T2", body="new")
    assert version["status"] == "pending" and version["revision_of"] == made["id"], "the key, not the account, is the authority"
    assert newsletter.read(db, made["id"])["body"] == "old", "the published text stays until the version is accepted"
    assert newsletter.read(db, made["id"], user_id=MEG)["waiting_version"] == version["id"]
    again = newsletter.edit(db, made["id"], **meg(), key=key, title="T3", body="newer")
    assert newsletter._get(db, version["id"])["status"] == "superseded", "one waiting version at a time"
    assert newsletter.verify(db, again["id"], key=key)["original_poster"] and not newsletter.verify(db, made["id"], key="x")["original_poster"]
    article = newsletter.decide(db, again["id"], username="opal", decision="published")
    assert article["id"] == made["id"] and (article["title"], article["body"]) == ("T3", "newer") and article["status"] == "published"
    operator_edit = newsletter.edit(db, made["id"], user_id=OPAL, username="opal", operator=True, key=key, title="T4", body="fixed")
    assert operator_edit["id"] == made["id"] and newsletter.read(db, made["id"])["body"] == "fixed", "an operator's edit applies at once"


def test_withdrawing_and_the_limits(db):
    made, key = newsletter.submit(db, **meg(), title="T", body="b")
    with pytest.raises(newsletter.NewsletterError):
        newsletter.withdraw(db, made["id"], key="wrong")
    assert newsletter.withdraw(db, made["id"], key=key)["status"] == "withdrawn"
    with pytest.raises(newsletter.NewsletterError, match="withdrawn"):
        newsletter.edit(db, made["id"], **meg(), key=key, title="T", body="b2")
    for n in range(newsletter.MAX_PENDING):
        newsletter.submit(db, **meg(), title=f"t{n}", body="b")
    with pytest.raises(newsletter.NewsletterError) as full:
        newsletter.submit(db, **meg(), title="one more", body="b")
    assert full.value.status == 429
    for title, body in (("", "b"), ("t", ""), ("x" * 201, "b"), ("t", "x" * (newsletter.MAX_BODY + 1))):
        with pytest.raises(newsletter.NewsletterError):
            newsletter.submit(db, user_id=NED, username="ned", operator=False, title=title, body=body)
    with pytest.raises(newsletter.NewsletterError) as missing:
        newsletter.read(db, "nope")
    assert missing.value.status == 404
    assert newsletter.operators(db) == ["admin", "opal"]


class Handler:
    def __init__(self, path, user=None, username=None, groups=(), body=None):
        self.path, self._body, self.sent = path, body, None
        self._who = SimpleNamespace(user_id=user, username=username, groups=frozenset(groups))
        self.headers = {"Content-Type": "application/json"}

    def _principal(self):
        return self._who

    def _body_json(self, limit):
        return self._body

    def _json(self, document, status=200):
        self.sent = (int(status), json.loads(json.dumps(document)))


def test_the_newsletter_over_rest(db, tmp_path):
    told = []

    def ask(path, method="GET", **kw):
        handler = Handler(path, **kw)
        newsletter.http_route(handler, path, method, accounts_db=tmp_path / "accounts.db",
                              notify=lambda names, text, kind, ref: told.append((names, kind)))
        return handler.sent

    assert ask("/api/newsletter") == (200, {"published": []}), "a guest reads the published list"
    assert ask("/api/newsletter", "POST", body={"title": "t", "body": "b"})[0] == 401, "writing needs an account"
    status, made = ask("/api/newsletter", "POST", user=MEG, username="meg", groups=("member",), body={"title": "t", "body": "b"})
    assert status == 200 and made["article"]["status"] == "pending" and made["key"] and "key_hash" not in made["article"]
    assert told == [(["admin", "opal"], "article")], "the operators are told of a submission"
    article = made["article"]["id"]
    assert ask(f"/api/newsletter/{article}")[0] == 404
    assert ask(f"/api/newsletter/{article}/decide", "POST", user=MEG, username="meg", groups=("member",),
               body={"decision": "published"})[0] == 403
    assert ask(f"/api/newsletter/{article}/decide", "POST", user=OPAL, username="opal", groups=("member", "dev", "operator"),
               body={"decision": "published"})[1]["article"]["status"] == "published"
    assert ask(f"/api/newsletter/{article}")[1]["article"]["body"] == "b"
    assert ask(f"/api/newsletter/{article}/verify", "POST", body={"key": made["key"]})[1]["original_poster"] is True
    listed = ask("/api/newsletter", user=OPAL, username="opal", groups=("member", "dev", "operator"))[1]
    assert [a["id"] for a in listed["published"]] == [article] and listed["waiting"] == []


def test_an_articles_thread_is_open_once_it_is_published(db):
    kind = newsletter.comment_kind()
    made, _ = newsletter.submit(db, **meg(), title="T", body="b")
    stranger = {"user_id": NED, "username": "ned", "caps": frozenset()}
    assert kind.may_read(db, stranger, made["id"]) is not True
    assert kind.may_read(db, {"user_id": MEG, "username": "meg", "caps": frozenset()}, made["id"]) is True
    newsletter.decide(db, made["id"], username="opal", decision="published")
    assert kind.may_read(db, stranger, made["id"]) is True
    comments.post(db, "article", made["id"], user_id=NED, username="ned", body="nice")
    assert kind.tell(db, made["id"], None) == ["meg", "ned", "opal"]
