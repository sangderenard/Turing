"""The server's newsletter: articles anyone may read; an operator or above posts directly, anyone else signed in submits
for review, and an operator publishes or declines it. Each article's author is given a key once, when it is submitted:
the key edits it later and identifies its original poster. An edit by anyone below operator goes back to review; the
published text stays as it was until the new version is accepted.

The articles live in the accounts database (server state, never a repository), table ``nl_articles`` (schema module
``newsletter`` 1, this module its one writer). A row is an article or a pending revision of one (``revision_of``). Only a
hash of a key is kept. Text is stored as given (Markdown) and only ever returned as JSON strings; the page renders it
without HTML. Discussion of an article is the comments engine's kind ``article``; an @mention there tells who it names.

    GET  /api/newsletter                  published articles (anyone); with an account also its own, and for an
                                          operator the ones waiting for review
    GET  /api/newsletter/<id>             one article: published to anyone; pending to its author and operators
    POST /api/newsletter                  {title, body}            submit (operator and above: published at once); {key}
    POST /api/newsletter/<id>/edit        {key, title, body}       a new version (operator and above: applied at once)
    POST /api/newsletter/<id>/decide      {decision: published|declined, note?}   an operator, on a pending row
    POST /api/newsletter/<id>/verify      {key}                    is this the original poster's key
    POST /api/newsletter/<id>/withdraw    {key}                    the author takes an article or a pending edit back
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
import uuid

from meowdoku import schema_version

SCHEMA_VERSION = 1
MAX_TITLE = 200
MAX_BODY = 100_000
MAX_PENDING = 10                    # waiting rows per author
MAX_LIST = 200
ID = re.compile(r"[0-9a-f]{32}\Z")
STATUSES = ("pending", "published", "declined", "withdrawn", "superseded")
DDL = (
    "CREATE TABLE IF NOT EXISTS nl_articles (id TEXT PRIMARY KEY, revision_of TEXT, author_id INTEGER NOT NULL, "
    "author_name TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL, status TEXT NOT NULL, key_hash TEXT, "
    "created REAL NOT NULL, updated REAL NOT NULL, published_at REAL, decided_by TEXT, decided_at REAL, note TEXT)",
    "CREATE INDEX IF NOT EXISTS nl_articles_by_status ON nl_articles(status, published_at DESC)",
    "CREATE INDEX IF NOT EXISTS nl_articles_by_author ON nl_articles(author_id, updated DESC)",
)
_COLUMNS = ("id", "revision_of", "author_id", "author_name", "title", "body", "status", "key_hash", "created", "updated",
            "published_at", "decided_by", "decided_at", "note")


class NewsletterError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def ensure(db) -> None:
    schema_version.declare(db, "newsletter", SCHEMA_VERSION)
    for statement in DDL:
        db.execute(statement)


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _get(db, article_id) -> dict:
    if not isinstance(article_id, str) or not ID.match(article_id):
        raise NewsletterError("no such article", 404)
    row = db.execute(f"SELECT {', '.join(_COLUMNS)} FROM nl_articles WHERE id = ?", (article_id,)).fetchone()
    if row is None:
        raise NewsletterError("no such article", 404)
    return dict(zip(_COLUMNS, row))


def _public(row, *, body=True) -> dict:
    out = {k: row[k] for k in ("id", "revision_of", "author_name", "title", "status", "created", "updated", "published_at",
                                "decided_by", "decided_at", "note")}
    if body:
        out["body"] = row["body"]
    return out


def _text(title, body) -> tuple:
    if not isinstance(title, str) or not title.strip() or len(title) > MAX_TITLE:
        raise NewsletterError(f"a title is text of 1 to {MAX_TITLE} characters")
    if not isinstance(body, str) or not body.strip() or len(body) > MAX_BODY:
        raise NewsletterError(f"an article is text of 1 to {MAX_BODY} characters")
    return " ".join(title.split()), body.strip()


def _key_matches(db, article, key) -> bool:
    original = _get(db, article["revision_of"]) if article["revision_of"] else article
    return isinstance(key, str) and bool(original["key_hash"]) and hmac.compare_digest(original["key_hash"], _hash(key))


def _waiting(db, author_id) -> int:
    return db.execute("SELECT count(*) FROM nl_articles WHERE author_id = ? AND status = 'pending'", (author_id,)).fetchone()[0]


def submit(db, *, user_id, username, operator, title, body, now=None) -> tuple:
    """(the article, its key). The key is returned this once and never stored, only its hash."""
    ensure(db)
    title, body = _text(title, body)
    if not operator and _waiting(db, user_id) >= MAX_PENDING:
        raise NewsletterError(f"you have {MAX_PENDING} articles waiting for review already", 429)
    now = time.time() if now is None else now
    key, article_id = secrets.token_urlsafe(24), uuid.uuid4().hex
    status = "published" if operator else "pending"
    db.execute(f"INSERT INTO nl_articles ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})",
               (article_id, None, int(user_id), str(username), title, body, status, _hash(key), now, now,
                now if operator else None, str(username) if operator else None, now if operator else None, None))
    return _get(db, article_id), key


def edit(db, article_id, *, user_id, username, operator, key, title, body, now=None) -> dict:
    """A new version of an article, by its key. An operator's edit replaces the text at once; anyone else's becomes a
    pending revision (replacing an earlier pending one) while the article stays as it was."""
    ensure(db)
    article = _get(db, article_id)
    if article["revision_of"]:
        raise NewsletterError("edit the article itself, not one of its pending versions", 409)
    if not _key_matches(db, article, key):
        raise NewsletterError("that is not this article's key", 403)
    if article["status"] in ("declined", "withdrawn", "superseded"):
        raise NewsletterError(f"this article is {article['status']}; submit it again as a new article", 409)
    title, body = _text(title, body)
    now = time.time() if now is None else now
    if operator or article["status"] == "pending":        # still waiting anyway: the waiting text is simply replaced
        db.execute("UPDATE nl_articles SET title = ?, body = ?, updated = ? WHERE id = ?", (title, body, now, article_id))
        return _get(db, article_id)
    db.execute("UPDATE nl_articles SET status = 'superseded', updated = ? WHERE revision_of = ? AND status = 'pending'",
               (now, article_id))
    revision = uuid.uuid4().hex
    db.execute(f"INSERT INTO nl_articles ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})",
               (revision, article_id, int(user_id), str(username), title, body, "pending", None, now, now, None, None, None,
                None))
    return _get(db, revision)


def decide(db, row_id, *, username, decision, note="", now=None) -> dict:
    """An operator publishes or declines a pending article or pending revision. Publishing a revision puts its text in
    the article; the article keeps its id, author and key."""
    ensure(db)
    if decision not in ("published", "declined"):
        raise NewsletterError("decision must be published or declined")
    if not isinstance(note, str) or len(note) > 2000:
        raise NewsletterError("a note is text of at most 2000 characters")
    row = _get(db, row_id)
    if row["status"] != "pending":
        raise NewsletterError(f"this is {row['status']}, not waiting for review", 409)
    now = time.time() if now is None else now
    db.execute("UPDATE nl_articles SET status = ?, decided_by = ?, decided_at = ?, note = ?, updated = ?, "
               "published_at = CASE WHEN ? = 'published' AND revision_of IS NULL THEN ? ELSE published_at END WHERE id = ?",
               (decision, username, now, note.strip() or None, now, decision, now, row_id))
    if decision == "published" and row["revision_of"]:
        db.execute("UPDATE nl_articles SET title = ?, body = ?, updated = ? WHERE id = ?",
                   (row["title"], row["body"], now, row["revision_of"]))
        return _get(db, row["revision_of"])
    return _get(db, row_id)


def withdraw(db, row_id, *, key, now=None) -> dict:
    ensure(db)
    row = _get(db, row_id)
    if not _key_matches(db, row, key):
        raise NewsletterError("that is not this article's key", 403)
    if row["status"] not in ("pending", "published"):
        raise NewsletterError(f"this is {row['status']} already", 409)
    now = time.time() if now is None else now
    db.execute("UPDATE nl_articles SET status = 'withdrawn', updated = ? WHERE id = ?", (now, row_id))
    if not row["revision_of"]:                             # an article taken back takes its waiting versions with it
        db.execute("UPDATE nl_articles SET status = 'withdrawn', updated = ? WHERE revision_of = ? AND status = 'pending'",
                   (now, row_id))
    return _get(db, row_id)


def verify(db, row_id, *, key) -> dict:
    ensure(db)
    row = _get(db, row_id)
    return {"id": row_id, "original_poster": _key_matches(db, row, key)}


def listing(db, *, user_id=None, operator=False, limit=MAX_LIST) -> dict:
    ensure(db)
    pick = f"SELECT {', '.join(_COLUMNS)} FROM nl_articles "
    published = [dict(zip(_COLUMNS, r)) for r in db.execute(
        pick + "WHERE status = 'published' AND revision_of IS NULL ORDER BY published_at DESC LIMIT ?", (limit,))]
    out = {"published": [_public(r, body=False) for r in published]}
    if user_id is not None:
        out["mine"] = [_public(dict(zip(_COLUMNS, r)), body=False) for r in db.execute(
            pick + "WHERE author_id = ? ORDER BY updated DESC LIMIT ?", (int(user_id), limit))]
    if operator:
        out["waiting"] = [_public(dict(zip(_COLUMNS, r)), body=False) for r in db.execute(
            pick + "WHERE status = 'pending' ORDER BY created LIMIT ?", (limit,))]
    return out


def read(db, row_id, *, user_id=None, operator=False) -> dict:
    ensure(db)
    row = _get(db, row_id)
    visible = (row["status"] == "published" and not row["revision_of"]) or operator or \
        (user_id is not None and row["author_id"] == int(user_id))
    if not visible:
        raise NewsletterError("no such article", 404)        # a waiting article is not shown to be there
    out = _public(row)
    if not row["revision_of"]:
        out["waiting_version"] = next((r[0] for r in db.execute(
            "SELECT id FROM nl_articles WHERE revision_of = ? AND status = 'pending'", (row_id,))), None) \
            if operator or (user_id is not None and row["author_id"] == int(user_id)) else None
    return out


def operators(db) -> list:
    """The accounts told of a new submission: every operator and admin, by their authority (permission_groups)."""
    from meowdoku import permission_groups
    permission_groups.prepare(db)
    names = []
    for uid, name in db.execute("SELECT id, username FROM users"):
        if "operator" in permission_groups.authority(db, uid)["groups"]:
            names.append(name)
    return names


def comment_kind():
    """The ``article`` kind of the comments engine: a published article's thread is for every signed-in account, a
    waiting one's for its author and the operators; the author and the operators who reviewed it are told."""
    from meowdoku import comments

    def row_of(db, ref):
        try:
            return _get(db, ref)
        except NewsletterError as error:
            raise comments.CommentError(error.message, error.status) from None

    def may(db, who, ref):
        row = row_of(db, ref)
        if row["status"] == "published" and not row["revision_of"]:
            return True
        return True if (who["user_id"] == row["author_id"] or "account_manage" in who["caps"] or
                        "dev_ui" in who["caps"]) else "a waiting article's thread is for its author and the reviewers"

    def tell(db, ref, comment):
        row = row_of(db, ref)
        return sorted({row["author_name"], *([row["decided_by"]] if row["decided_by"] else []),
                       *comments.writers(db, "article", ref)})

    return comments.Kind(may_read=may, may_write=may, tell=tell, label=lambda db, ref: f"the article {row_of(db, ref)['title']!r}")


def http_route(self, path, method, *, accounts_db, notify=None):
    """The newsletter over REST. Reading published articles needs no account; everything else does, and the account
    always comes from the session. ``notify(usernames, text, kind, ref)`` tells the operators of a new submission."""
    import sqlite3
    principal = self._principal()
    user_id = principal.user_id
    operator = user_id is not None and "operator" in set(getattr(principal, "groups", ()) or ())
    db = sqlite3.connect(accounts_db, timeout=10, isolation_level=None)
    try:
        parts = path[len("/api/newsletter"):].strip("/").split("/") if path != "/api/newsletter" else []
        article_id, action = (parts + [None, None])[:2] if parts else (None, None)
        if method == "GET":
            if article_id is None:
                return self._json(listing(db, user_id=user_id, operator=operator))
            if action is None:
                return self._json({"article": read(db, article_id, user_id=user_id, operator=operator)})
            return self._json({"error": "not found"}, 404)
        if method != "POST":
            return self._json({"error": "not found"}, 404)
        self.close_connection = True
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return self._json({"error": "JSON only"}, 415)
        body = self._body_json(MAX_BODY * 4 + 4096)
        if body is None:
            return
        if not isinstance(body, dict):
            return self._json({"error": "a JSON object is required"}, 400)
        if action == "verify":
            return self._json(verify(db, article_id, key=body.get("key")))
        if user_id is None:
            return self._json({"error": "sign in to write in the newsletter"}, 401)
        who = {"user_id": user_id, "username": principal.username, "operator": operator}
        if article_id is None:
            made, key = submit(db, **who, title=body.get("title"), body=body.get("body"))
            if made["status"] == "pending" and notify is not None:
                others = [n for n in operators(db) if n != principal.username]
                if others:
                    notify(others, f"{principal.username} submitted the article {made['title']!r} to the newsletter; it waits "
                                   f"for review on /newsletter.html", "article", made["id"])
            return self._json({"article": _public(made), "key": key,
                               "note": "keep this key: it edits the article and shows you are its original poster; "
                                       "it is not shown again"})
        if action == "edit":
            made = edit(db, article_id, **who, key=body.get("key"), title=body.get("title"), body=body.get("body"))
            return self._json({"article": _public(made)})
        if action == "withdraw":
            return self._json({"article": _public(withdraw(db, article_id, key=body.get("key")))})
        if action == "decide":
            if not operator:
                return self._json({"error": "an operator or an admin reviews articles"}, 403)
            made = decide(db, article_id, username=principal.username, decision=body.get("decision"),
                          note=body.get("note") or "")
            return self._json({"article": _public(made)})
        return self._json({"error": "not found"}, 404)
    except NewsletterError as error:
        return self._json({"error": error.message}, error.status)
    finally:
        db.close()
