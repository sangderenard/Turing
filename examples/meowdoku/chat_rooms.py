"""The chat server: persistent rooms, each open to the roles it declares; messages in order.

Rooms are declared here, not made by users. A room names the ``account_roles`` roles that may enter it; an account with
another role (or none: a member) does not see it at all. The first room is ``reviewers``, the operators and admins beside the
change reviews (branch_review.py). A change's author with less privilege talks in that change's thread (comments.py), not here.

Tables in accounts.db (schema module ``chat_rooms`` 1; this module is their one writer):
    chat_messages  (id, room, user_id, username, body, created, removed_at, removed_by)
    chat_audit     (at, admin, action, message_id)

Text is stored stripped and only ever returned as a JSON string. An admin may soft-delete a message (audited). Who is asking
comes from the authenticated session.

    GET  /api/chat/rooms                        the rooms this account may enter
    GET  /api/chat/rooms/<room>?tail=N          its last N messages, oldest first (a count or all; 20 by default)
    GET  /api/chat/rooms/<room>?after=ID        the messages after one already read, for polling
    POST /api/chat/rooms/<room>    {body}       one message
    POST /api/chat/remove          {id}         an admin soft-deletes one
"""
from __future__ import annotations

import time

from meowdoku import schema_version

SCHEMA_VERSION = 1
MAX_BODY = 4000
MAX_TAIL = 1000
ROOMS = {
    "reviewers": {"title": "Reviewers' chat", "about": "The operators and admins, beside the change reviews.",
                  "roles": ("admin", "operator")},
}
DDL = (
    "CREATE TABLE IF NOT EXISTS chat_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, room TEXT NOT NULL, user_id INTEGER NOT NULL, "
    "username TEXT NOT NULL, body TEXT NOT NULL, created REAL NOT NULL, removed_at REAL, removed_by TEXT)",
    "CREATE INDEX IF NOT EXISTS chat_messages_by_room ON chat_messages(room, id)",
    "CREATE TABLE IF NOT EXISTS chat_audit (at REAL NOT NULL, admin TEXT NOT NULL, action TEXT NOT NULL, message_id INTEGER NOT NULL)",
)


class ChatError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def ensure(db) -> None:
    schema_version.declare(db, "chat_rooms", SCHEMA_VERSION)
    for statement in DDL:
        db.execute(statement)


def role_of(db, user_id) -> str:
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'account_roles'").fetchone():
        return "member"
    row = db.execute("SELECT role FROM account_roles WHERE user_id = ?", (int(user_id),)).fetchone()
    return row[0] if row else "member"


def rooms_for(db, user_id) -> list:
    role = role_of(db, user_id)
    return [{"room": name, "title": room["title"], "about": room["about"]} for name, room in ROOMS.items() if role in room["roles"]]


def _enter(db, room, user_id) -> None:
    if room not in ROOMS:
        raise ChatError("no such room", 404)
    if role_of(db, user_id) not in ROOMS[room]["roles"]:
        raise ChatError(f"the room {room} is for {' and '.join(ROOMS[room]['roles'])} accounts", 403)


def _row(r) -> dict:
    removed = r[5] is not None
    return {"id": r[0], "room": r[1], "username": r[2], "body": "" if removed else r[3], "at": r[4], "removed": removed}


_SELECT = "SELECT id, room, username, body, created, removed_at FROM chat_messages"


def tail_of(value) -> int:
    if value in (None, ""):
        return 20
    if value == "all":
        return MAX_TAIL
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise ChatError("tail must be a count or all") from None
    if not 1 <= count <= MAX_TAIL:
        raise ChatError(f"tail must be 1 to {MAX_TAIL}, or all")
    return count


def post(db, room, *, user_id, username, body, now=None) -> dict:
    ensure(db)
    _enter(db, room, user_id)
    if not isinstance(body, str) or not body.strip() or len(body) > MAX_BODY:
        raise ChatError(f"a message must be text of 1 to {MAX_BODY} characters")
    now = time.time() if now is None else now
    cursor = db.execute("INSERT INTO chat_messages (room, user_id, username, body, created) VALUES (?, ?, ?, ?, ?)",
                        (room, int(user_id), str(username), body.strip(), now))
    return _row(db.execute(_SELECT + " WHERE id = ?", (cursor.lastrowid,)).fetchone())


def messages(db, room, *, user_id, tail=20, after=None) -> list:
    """The room oldest first: its last ``tail`` messages, or every message after ``after`` (at most MAX_TAIL)."""
    ensure(db)
    _enter(db, room, user_id)
    if after is not None:
        rows = db.execute(_SELECT + " WHERE room = ? AND id > ? ORDER BY id LIMIT ?", (room, int(after), MAX_TAIL)).fetchall()
        return [_row(r) for r in rows]
    rows = db.execute(_SELECT + " WHERE room = ? ORDER BY id DESC LIMIT ?", (room, int(tail))).fetchall()
    return [_row(r) for r in reversed(rows)]


def remove(db, message_id, *, admin, now=None) -> dict:
    ensure(db)
    now = time.time() if now is None else now
    done = db.execute("UPDATE chat_messages SET removed_at = ?, removed_by = ? WHERE id = ? AND removed_at IS NULL",
                      (now, str(admin), int(message_id)))
    if done.rowcount != 1:
        raise ChatError("no such message, or it is already removed", 404)
    db.execute("INSERT INTO chat_audit (at, admin, action, message_id) VALUES (?, ?, 'remove', ?)", (now, str(admin), int(message_id)))
    return _row(db.execute(_SELECT + " WHERE id = ?", (int(message_id),)).fetchone())


def http_route(self, path, method, *, accounts_db):
    import sqlite3
    from urllib.parse import parse_qs
    principal = self._principal()
    if principal.user_id is None:
        return self._json({"error": "sign in first"}, 401)
    db = sqlite3.connect(accounts_db, timeout=10, isolation_level=None)
    try:
        if method == "GET" and path == "/api/chat/rooms":
            ensure(db)
            return self._json({"rooms": rooms_for(db, principal.user_id)})
        if path.startswith("/api/chat/rooms/"):
            room = path[len("/api/chat/rooms/"):]
            if method == "GET":
                query = parse_qs(self.path.partition("?")[2])
                tail = tail_of((query.get("tail") or [None])[0])
                after = (query.get("after") or [None])[0]
                if after is not None:
                    try:
                        after = int(after)
                    except ValueError:
                        return self._json({"error": "after must be a message id"}, 400)
                return self._json({"room": room, "tail": tail,
                                   "messages": messages(db, room, user_id=principal.user_id, tail=tail, after=after)})
        if method != "POST":
            return self._json({"error": "not found"}, 404)
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return self._json({"error": "JSON only"}, 415)
        body = self._body_json(16384)
        if body is None:
            return
        if not isinstance(body, dict):
            return self._json({"error": "a JSON object is required"}, 400)
        if path == "/api/chat/remove":
            if "account_manage" not in frozenset(self._caps()):
                return self._json({"error": "an admin removes a message"}, 403)
            return self._json({"message": remove(db, body.get("id"), admin=principal.username)})
        if path.startswith("/api/chat/rooms/"):
            made = post(db, path[len("/api/chat/rooms/"):], user_id=principal.user_id, username=principal.username,
                        body=body.get("body"))
            return self._json({"message": made})
        return self._json({"error": "not found"}, 404)
    except ChatError as error:
        return self._json({"error": error.message}, error.status)
    finally:
        db.close()
