"""Comments under a post: one engine for every kind of post that carries a thread.

A POST is ``(kind, ref)``: a change under review is ``("review", <review id>)``; a report, an authored page or a ticket can be
another kind later. This module keeps the threads and nothing about what a post is. The module that owns a kind registers a
``Kind`` with the host: it says who may read the thread, who may write in it, and whom to tell of a new comment. A kind that is
not registered has no thread.

Tables in accounts.db (schema module ``comments`` 1; this module is their one writer):
    cm_comments  (id, kind, ref, user_id, username, body, reply_to, created, removed_at, removed_by)
    cm_audit     (at, admin, action, comment_id)

Text is stored stripped and only ever returned as a JSON string; nothing here produces HTML. An admin (account_manage) may
soft-delete a comment: readers then see body "" and removed true, and the act is a ``cm_audit`` row. Who is asking always comes
from the authenticated session, never from a request.

    GET  /api/comments?kind=K&ref=R&tail=N     the thread's last N comments, oldest first (N: a count or all; 20 by default)
    GET  /api/comments?kind=K&ref=R&after=ID   the comments after one already read, for polling
    POST /api/comments            {kind, ref, body, reply_to?}   one comment; the kind's participants are told
    POST /api/comments/remove     {id}                           an admin soft-deletes one
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable

from meowdoku import schema_version

SCHEMA_VERSION = 1
MAX_BODY = 4000
MAX_TAIL = 1000
KIND = re.compile(r"[a-z][a-z0-9_]{1,31}\Z")
DDL = (
    "CREATE TABLE IF NOT EXISTS cm_comments (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, ref TEXT NOT NULL, "
    "user_id INTEGER NOT NULL, username TEXT NOT NULL, body TEXT NOT NULL, reply_to INTEGER, created REAL NOT NULL, "
    "removed_at REAL, removed_by TEXT)",
    "CREATE INDEX IF NOT EXISTS cm_comments_by_post ON cm_comments(kind, ref, id)",
    "CREATE TABLE IF NOT EXISTS cm_audit (at REAL NOT NULL, admin TEXT NOT NULL, action TEXT NOT NULL, comment_id INTEGER NOT NULL)",
)


class CommentError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


@dataclass(frozen=True)
class Kind:
    """What the owner of a kind of post decides. Each callable gets the open database and the caller
    (``who`` = {"user_id", "username", "caps"}) and the post's ref.

    ``may_read(db, who, ref)`` / ``may_write(db, who, ref)``: True, or a reason string (refused with 403); raise CommentError
    404 when there is no such post. ``tell(db, ref, comment)``: the usernames to tell of a new comment (the writer is left
    out by the engine). ``label(db, ref)``: a short name of the post for the notice."""
    may_read: Callable
    may_write: Callable
    tell: Callable
    label: Callable


def ensure(db) -> None:
    schema_version.declare(db, "comments", SCHEMA_VERSION)
    for statement in DDL:
        db.execute(statement)


def tail_of(value) -> int:
    """A history length: a positive count up to MAX_TAIL, or "all" (MAX_TAIL). 20 when not given."""
    if value in (None, ""):
        return 20
    if value == "all":
        return MAX_TAIL
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise CommentError("tail must be a count or all") from None
    if not 1 <= count <= MAX_TAIL:
        raise CommentError(f"tail must be 1 to {MAX_TAIL}, or all")
    return count


def _post(kind, ref) -> tuple:
    if not isinstance(kind, str) or not KIND.match(kind):
        raise CommentError("kind must be a lowercase name")
    if not isinstance(ref, str) or not ref or len(ref) > 200:
        raise CommentError("ref must be 1 to 200 characters")
    return kind, ref


def _row(r) -> dict:
    removed = r[8] is not None
    return {"id": r[0], "kind": r[1], "ref": r[2], "username": r[3], "body": "" if removed else r[4], "reply_to": r[5],
            "at": r[6], "removed": removed}


_SELECT = "SELECT id, kind, ref, username, body, reply_to, created, user_id, removed_at FROM cm_comments"


def post(db, kind, ref, *, user_id, username, body, reply_to=None, now=None) -> dict:
    """One comment. The caller has been allowed by the kind already (``http_route`` asks it)."""
    kind, ref = _post(kind, ref)
    if not isinstance(body, str) or not body.strip() or len(body) > MAX_BODY:
        raise CommentError(f"a comment must be text of 1 to {MAX_BODY} characters")
    ensure(db)
    if reply_to is not None:
        if isinstance(reply_to, bool) or not isinstance(reply_to, int) or db.execute(
                "SELECT 1 FROM cm_comments WHERE id = ? AND kind = ? AND ref = ?", (reply_to, kind, ref)).fetchone() is None:
            raise CommentError("reply_to must be a comment of the same thread")
    now = time.time() if now is None else now
    cursor = db.execute("INSERT INTO cm_comments (kind, ref, user_id, username, body, reply_to, created) VALUES (?,?,?,?,?,?,?)",
                        (kind, ref, int(user_id), str(username), body.strip(), reply_to, now))
    return _row(db.execute(_SELECT + " WHERE id = ?", (cursor.lastrowid,)).fetchone())


def thread(db, kind, ref, *, tail=20, after=None) -> list:
    """The thread oldest first: its last ``tail`` comments, or every comment after ``after`` (at most MAX_TAIL)."""
    kind, ref = _post(kind, ref)
    ensure(db)
    if after is not None:
        rows = db.execute(_SELECT + " WHERE kind = ? AND ref = ? AND id > ? ORDER BY id LIMIT ?",
                          (kind, ref, int(after), MAX_TAIL)).fetchall()
        return [_row(r) for r in rows]
    rows = db.execute(_SELECT + " WHERE kind = ? AND ref = ? ORDER BY id DESC LIMIT ?", (kind, ref, int(tail))).fetchall()
    return [_row(r) for r in reversed(rows)]


def writers(db, kind, ref) -> set:
    """Everyone who has written in the thread."""
    ensure(db)
    return {r[0] for r in db.execute("SELECT DISTINCT username FROM cm_comments WHERE kind = ? AND ref = ?", (kind, ref))}


def active(db, kind, *, limit=50) -> list:
    """The refs of this kind with a thread, the most recently written first: [(ref, last comment time)]."""
    ensure(db)
    return [(r[0], r[1]) for r in db.execute(
        "SELECT ref, MAX(created) FROM cm_comments WHERE kind = ? GROUP BY ref ORDER BY MAX(created) DESC LIMIT ?",
        (kind, int(limit)))]


def remove(db, comment_id, *, admin, now=None) -> dict:
    """An admin soft-deletes one comment; the act is audited."""
    ensure(db)
    now = time.time() if now is None else now
    done = db.execute("UPDATE cm_comments SET removed_at = ?, removed_by = ? WHERE id = ? AND removed_at IS NULL",
                      (now, str(admin), int(comment_id)))
    if done.rowcount != 1:
        raise CommentError("no such comment, or it is already removed", 404)
    db.execute("INSERT INTO cm_audit (at, admin, action, comment_id) VALUES (?, ?, 'remove', ?)", (now, str(admin), int(comment_id)))
    return _row(db.execute(_SELECT + " WHERE id = ?", (int(comment_id),)).fetchone())


def _allowed(answer) -> None:
    if answer is not True:
        raise CommentError(answer if isinstance(answer, str) and answer else "this thread is not open to you", 403)


def http_route(self, path, method, *, accounts_db, kinds: dict, notify=None):
    """The threads over REST. ``kinds`` maps a kind to its ``Kind``; ``notify(usernames, text, kind, ref)`` tells people of a
    new comment (None: they find it on the page)."""
    import sqlite3
    from urllib.parse import parse_qs
    principal = self._principal()
    if principal.user_id is None:
        return self._json({"error": "sign in first"}, 401)
    who = {"user_id": principal.user_id, "username": principal.username, "caps": frozenset(self._caps())}
    db = sqlite3.connect(accounts_db, timeout=10, isolation_level=None)
    try:
        if method == "GET" and path == "/api/comments":
            query = parse_qs(self.path.partition("?")[2])
            kind, ref = _post((query.get("kind") or [""])[0], (query.get("ref") or [""])[0])
            if kind not in kinds:
                return self._json({"error": "this kind of post has no thread"}, 404)
            _allowed(kinds[kind].may_read(db, who, ref))
            after = (query.get("after") or [None])[0]
            tail = tail_of((query.get("tail") or [None])[0])
            if after is not None:
                try:
                    after = int(after)
                except ValueError:
                    return self._json({"error": "after must be a comment id"}, 400)
            return self._json({"kind": kind, "ref": ref, "tail": tail, "comments": thread(db, kind, ref, tail=tail, after=after)})
        if method != "POST" or path not in ("/api/comments", "/api/comments/remove"):
            return self._json({"error": "not found"}, 404)
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return self._json({"error": "JSON only"}, 415)
        body = self._body_json(16384)
        if body is None:
            return
        if not isinstance(body, dict):
            return self._json({"error": "a JSON object is required"}, 400)
        if path == "/api/comments/remove":
            if "account_manage" not in who["caps"]:
                return self._json({"error": "an admin removes a comment"}, 403)
            return self._json({"comment": remove(db, body.get("id"), admin=principal.username)})
        kind, ref = _post(body.get("kind"), body.get("ref"))
        if kind not in kinds:
            return self._json({"error": "this kind of post has no thread"}, 404)
        _allowed(kinds[kind].may_write(db, who, ref))
        made = post(db, kind, ref, user_id=principal.user_id, username=principal.username, body=body.get("body"),
                    reply_to=body.get("reply_to"))
        tell = sorted(set(kinds[kind].tell(db, ref, made)) - {principal.username})
        if notify is not None and tell:
            notify(tell, f"{principal.username} wrote in the thread of {kinds[kind].label(db, ref)}: {made['body'][:300]}",
                   kind, ref)
        print(f"[comments] {principal.username} wrote in {kind} {ref[:12]} ({len(tell)} told)", flush=True)
        return self._json({"comment": made, "told": tell})
    except CommentError as error:
        return self._json({"error": error.message}, error.status)
    finally:
        db.close()
