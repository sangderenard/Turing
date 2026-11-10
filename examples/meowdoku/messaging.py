"""Meowdoku user-to-user messaging: contacts and messages in accounts.db (server.py [MESSAGES], web/messenger-widget.js).

Tables (schema module ``messaging`` 1; this module is their one writer):
    msg_contacts  (user_id, peer_id, state requested|accepted|blocked, created, updated)   DIRECTED rows, one per side
    msg_messages  (id, sender_id, recipient_id, body, created, read_at)

A conversation exists only when BOTH directions are ``accepted``.  ``request_contact(me, X)`` sets MY row for X to accepted and X's row
for me to requested (pending X's decision); ``respond_contact(me, X, True)`` sets my row accepted.  A row ``blocked`` on either side
ends the conversation.  Every function takes ``me`` = the account id from the SESSION, never an id from a request, and answers only
about ``me``'s own contacts and ``me``'s own messages.

Username probing: ``request_contact`` returns the same result for an unknown name, yourself, a blocker and a real request; ``send``
refuses every non-conversation (unknown, self, not accepted, blocked) with the same 403.  Text is stored as the stripped string and
only ever returned as a JSON string; nothing here produces HTML.

SERVER-MEDIATED AND OPERATOR-READABLE.  Every message passes through and is stored by the server; there is no peer-to-peer channel and
no address of one user is ever handed to another.  A user's responses expose about another user ONLY username and display_name (no id,
presence, last-seen, typing, or read receipt: ``read_at`` only feeds the recipient's own unread count).  An admin (capability
account_manage) may read any thread, soft-delete a message (removed_at/removed_by: participants then see body "" and removed true) and
suspend an account (msg_suspended: it cannot send or request contacts); each admin action is a ``msg_audit`` row.

Long poll: ``HUB`` is a per-process Condition ``send`` notifies; ``long_poll`` waits on it (at most MAX_WAITERS in total,
MAX_WAITERS_PER_ACCOUNT each).
"""
from __future__ import annotations

import sqlite3
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parents[1]), str(HERE.parents[1] / "examples")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from meowdoku import schema_version as _schema_version  # noqa: E402

MAX_BODY = 4000
MAX_PER_MINUTE, MAX_PER_DAY = 30, 2000
MAX_CONTACTS, MAX_PENDING = 200, 20
MAX_WAITERS, MAX_WAITERS_PER_ACCOUNT = 64, 2
MAX_WAIT_S = 25.0

DDL = ("CREATE TABLE IF NOT EXISTS msg_contacts (user_id INTEGER NOT NULL, peer_id INTEGER NOT NULL, "
       "state TEXT NOT NULL CHECK(state IN ('requested','accepted','blocked')), created REAL NOT NULL, updated REAL NOT NULL, "
       "PRIMARY KEY(user_id, peer_id))",
       "CREATE TABLE IF NOT EXISTS msg_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, sender_id INTEGER NOT NULL, "
       "recipient_id INTEGER NOT NULL, body TEXT NOT NULL, created REAL NOT NULL, read_at REAL, removed_at REAL, removed_by INTEGER)",
       "CREATE TABLE IF NOT EXISTS msg_suspended (user_id INTEGER PRIMARY KEY, by_id INTEGER NOT NULL, created REAL NOT NULL)",
       "CREATE TABLE IF NOT EXISTS msg_audit (id INTEGER PRIMARY KEY AUTOINCREMENT, actor_id INTEGER NOT NULL, action TEXT NOT NULL, "
       "target TEXT NOT NULL, created REAL NOT NULL)",
       "CREATE INDEX IF NOT EXISTS msg_messages_recipient ON msg_messages (recipient_id, id)",
       "CREATE INDEX IF NOT EXISTS msg_messages_pair ON msg_messages (sender_id, recipient_id, id)")


class MessagingError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def connect(db_path) -> sqlite3.Connection:
    """Own connection to accounts.db (autocommit).  Declares schema module ``messaging`` 1."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(db_path), timeout=10, isolation_level=None, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    try:
        _schema_version.declare(db, "messaging", 1)
    except _schema_version.SchemaVersionError:
        db.close()
        raise
    for ddl in DDL:
        db.execute(ddl)
    return db


class Hub:
    """Wake-ups for long polls: a sequence number under a Condition, plus the waiter census."""

    def __init__(self):
        self.cond = threading.Condition()
        self.seq = 0
        self._waiters: dict = {}

    def notify(self) -> None:
        with self.cond:
            self.seq += 1
            self.cond.notify_all()

    def wait_change(self, seen: int, timeout: float) -> None:
        with self.cond:
            if self.seq == seen and timeout > 0:
                self.cond.wait(timeout)

    def enter(self, account_id: int) -> None:
        with self.cond:
            if sum(self._waiters.values()) >= MAX_WAITERS or self._waiters.get(account_id, 0) >= MAX_WAITERS_PER_ACCOUNT:
                raise MessagingError("too many open message polls", 429)
            self._waiters[account_id] = self._waiters.get(account_id, 0) + 1

    def leave(self, account_id: int) -> None:
        with self.cond:
            left = self._waiters.get(account_id, 0) - 1
            if left > 0:
                self._waiters[account_id] = left
            else:
                self._waiters.pop(account_id, None)


HUB = Hub()


def _peer(db, username):
    """(id, canonical username, display_name) or None.  Case-insensitive like the account lookups."""
    if not isinstance(username, str) or not username.strip() or len(username) > 200:
        return None
    row = db.execute("SELECT id, username, display_name FROM users WHERE username = ? COLLATE NOCASE", (username.strip(),)).fetchone()
    return None if row is None else (row["id"], row["username"], row["display_name"])


def _state(db, user_id, peer_id):
    row = db.execute("SELECT state FROM msg_contacts WHERE user_id = ? AND peer_id = ?", (user_id, peer_id)).fetchone()
    return row["state"] if row else None


def _put(db, user_id, peer_id, state, now):
    db.execute("INSERT INTO msg_contacts (user_id, peer_id, state, created, updated) VALUES (?,?,?,?,?) "
               "ON CONFLICT(user_id, peer_id) DO UPDATE SET state = excluded.state, updated = excluded.updated",
               (user_id, peer_id, state, now, now))


def _mutual(db, a, b) -> bool:
    return _state(db, a, b) == "accepted" and _state(db, b, a) == "accepted"


# ---- contacts -------------------------------------------------------------------------------------------------------------------
def request_contact(db, me: int, username, now: float | None = None) -> dict:
    """Ask ``username`` to be a contact.  The result is the same for an unknown name, yourself, a blocker and success."""
    now = time.time() if now is None else now
    generic = {"ok": True, "result": "requested"}
    if _suspended(db, me):
        raise MessagingError("you cannot message this user", 403)
    held = db.execute("SELECT COUNT(*) FROM msg_contacts WHERE user_id = ? AND state = 'accepted'", (me,)).fetchone()[0]
    pending = db.execute("SELECT COUNT(*) FROM msg_contacts a JOIN msg_contacts b ON b.user_id = a.peer_id AND b.peer_id = a.user_id "
                         "WHERE a.user_id = ? AND a.state = 'accepted' AND b.state = 'requested'", (me,)).fetchone()[0]
    if held >= MAX_CONTACTS:
        raise MessagingError(f"at most {MAX_CONTACTS} contacts", 429)
    if pending >= MAX_PENDING:
        raise MessagingError(f"at most {MAX_PENDING} pending requests", 429)
    peer = _peer(db, username)
    if peer is None or peer[0] == me:
        return generic
    pid = peer[0]
    db.execute("BEGIN IMMEDIATE")
    try:
        _put(db, me, pid, "accepted", now)
        if _state(db, pid, me) is None:
            _put(db, pid, me, "requested", now)          # blocked / accepted / requested on their side stay as they are
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return generic


def respond_contact(db, me: int, peer_username, accept: bool, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    peer = _peer(db, peer_username)
    if peer is None or _state(db, me, peer[0]) != "requested":
        raise MessagingError("no such request", 404)
    pid = peer[0]
    db.execute("BEGIN IMMEDIATE")
    try:
        if accept:
            _put(db, me, pid, "accepted", now)
        else:
            db.execute("DELETE FROM msg_contacts WHERE user_id = ? AND peer_id = ?", (me, pid))
            db.execute("DELETE FROM msg_contacts WHERE user_id = ? AND peer_id = ? AND state = 'accepted'", (pid, me))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return {"ok": True, "result": "accepted" if accept else "declined"}


def block(db, me: int, peer_username, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    peer = _peer(db, peer_username)
    if peer is not None and peer[0] != me:
        _put(db, me, peer[0], "blocked", now)
    return {"ok": True, "result": "blocked"}


def remove(db, me: int, peer_username) -> dict:
    """Drop the contact (and a request) in both directions, except a block the OTHER side holds against me."""
    peer = _peer(db, peer_username)
    if peer is not None and peer[0] != me:
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("DELETE FROM msg_contacts WHERE user_id = ? AND peer_id = ?", (me, peer[0]))
            db.execute("DELETE FROM msg_contacts WHERE user_id = ? AND peer_id = ? AND state != 'blocked'", (peer[0], me))
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
    return {"ok": True, "result": "removed"}


def contacts(db, me: int) -> dict:
    """{contacts: [{username, display_name, unread, last}], requests: [...], outgoing: [...], blocked: [...]}.
    Only ``me``'s own rows: contacts are mutual peers; no one else's list is ever visible."""
    mine = db.execute("SELECT c.peer_id, c.state, c.created, u.username, u.display_name FROM msg_contacts c "
                      "JOIN users u ON u.id = c.peer_id WHERE c.user_id = ? ORDER BY u.username COLLATE NOCASE", (me,)).fetchall()
    out = {"contacts": [], "requests": [], "outgoing": [], "blocked": []}
    for row in mine:
        item = {"username": row["username"], "display_name": row["display_name"]}
        theirs = _state(db, row["peer_id"], me)
        if row["state"] == "blocked":
            out["blocked"].append({"username": row["username"]})
        elif row["state"] == "requested":
            out["requests"].append({**item, "created": row["created"]})
        elif theirs == "accepted":
            unread = db.execute("SELECT COUNT(*) FROM msg_messages WHERE recipient_id = ? AND sender_id = ? AND read_at IS NULL",
                                (me, row["peer_id"])).fetchone()[0]
            last = db.execute("SELECT MAX(created) FROM msg_messages WHERE (sender_id = ? AND recipient_id = ?) "
                              "OR (sender_id = ? AND recipient_id = ?)", (me, row["peer_id"], row["peer_id"], me)).fetchone()[0]
            out["contacts"].append({**item, "unread": unread, "last": last})
        elif theirs == "requested":
            out["outgoing"].append({"username": row["username"]})
    return out


# ---- messages -------------------------------------------------------------------------------------------------------------------
def _clean(body) -> str:
    if not isinstance(body, str):
        raise MessagingError("body must be text", 400)
    text = body.strip()
    if not text or len(text) > MAX_BODY or "\x00" in text:
        raise MessagingError(f"body must be 1 to {MAX_BODY} characters", 400)
    return text


def _row(db_row, names, admin: bool = False) -> dict:
    """The one message shape a user sees: exactly id, from, to, created, body, removed (no read state, no ids of accounts)."""
    gone = db_row["removed_at"] is not None
    return {"id": db_row["id"], "from": names[db_row["sender_id"]], "to": names[db_row["recipient_id"]],
            "created": db_row["created"], "body": db_row["body"] if (admin or not gone) else "", "removed": gone}


def _suspended(db, user_id) -> bool:
    return db.execute("SELECT 1 FROM msg_suspended WHERE user_id = ?", (user_id,)).fetchone() is not None


def _names(db, rows) -> dict:
    ids = {r["sender_id"] for r in rows} | {r["recipient_id"] for r in rows}
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    return {r["id"]: r["username"] for r in db.execute(f"SELECT id, username FROM users WHERE id IN ({marks})", tuple(ids))}


def send(db, me: int, peer_username, body, now: float | None = None, hub: Hub | None = None) -> dict:
    now = time.time() if now is None else now
    text = _clean(body)
    peer = _peer(db, peer_username)
    if _suspended(db, me) or peer is None or peer[0] == me or not _mutual(db, me, peer[0]):
        raise MessagingError("you cannot message this user", 403)
    db.execute("BEGIN IMMEDIATE")
    try:
        if not _mutual(db, me, peer[0]):
            raise MessagingError("you cannot message this user", 403)
        minute = db.execute("SELECT COUNT(*) FROM msg_messages WHERE sender_id = ? AND created > ?", (me, now - 60)).fetchone()[0]
        day = db.execute("SELECT COUNT(*) FROM msg_messages WHERE sender_id = ? AND created > ?", (me, now - 86400)).fetchone()[0]
        if minute >= MAX_PER_MINUTE:
            raise MessagingError(f"at most {MAX_PER_MINUTE} messages per minute", 429)
        if day >= MAX_PER_DAY:
            raise MessagingError(f"at most {MAX_PER_DAY} messages per day", 429)
        mid = db.execute("INSERT INTO msg_messages (sender_id, recipient_id, body, created) VALUES (?,?,?,?)",
                         (me, peer[0], text, now)).lastrowid
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    (hub or HUB).notify()
    return {"ok": True, "id": mid, "created": now}


def thread(db, me: int, peer_username, before_id=None, limit: int = 50) -> dict:
    """Messages between ``me`` and the peer, oldest first: the newest ``limit`` below ``before_id``.  An unknown peer is an empty thread."""
    limit = max(1, min(int(limit), 200))
    peer = _peer(db, peer_username)
    if peer is None:
        return {"messages": [], "more": False}
    args = [me, peer[0], peer[0], me]
    sql = "SELECT * FROM msg_messages WHERE ((sender_id = ? AND recipient_id = ?) OR (sender_id = ? AND recipient_id = ?))"
    if before_id is not None:
        sql += " AND id < ?"
        args.append(int(before_id))
    rows = db.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit + 1)).fetchall()
    more = len(rows) > limit
    rows = list(reversed(rows[:limit]))
    names = _names(db, rows)
    return {"messages": [_row(r, names) for r in rows], "more": more}


def since(db, me: int, after_id, limit: int = 200) -> list:
    """Every message to or from ``me`` with id > after_id, oldest first: the poll primitive."""
    limit = max(1, min(int(limit), 500))
    rows = db.execute("SELECT * FROM msg_messages WHERE id > ? AND (sender_id = ? OR recipient_id = ?) ORDER BY id LIMIT ?",
                      (int(after_id or 0), me, me, limit)).fetchall()
    names = _names(db, rows)
    return [_row(r, names) for r in rows]


def latest_id(db, me: int) -> int:
    """The highest id of a message to or from ``me`` (0 when none): where a fresh poller starts."""
    return db.execute("SELECT COALESCE(MAX(id), 0) FROM msg_messages WHERE sender_id = ? OR recipient_id = ?", (me, me)).fetchone()[0]


def long_poll(db, me: int, after_id, wait: float, hub: Hub | None = None, limit: int = 200) -> list:
    hub = hub or HUB
    if int(after_id or 0) < 0:                  # after=-1: no messages, just ``latest_id`` (the caller reports it)
        return []
    wait = max(0.0, min(float(wait), MAX_WAIT_S))
    if wait <= 0:
        return since(db, me, after_id, limit)
    hub.enter(me)
    deadline = time.monotonic() + wait
    try:
        while True:
            seen = hub.seq                      # read BEFORE the query: a send in between shows as a changed seq
            rows = since(db, me, after_id, limit)
            if rows:
                return rows
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return []
            hub.wait_change(seen, min(remaining, 2.0))   # re-query every 2 s: another server process on the same database cannot notify this hub
    finally:
        hub.leave(me)


def mark_read(db, me: int, peer_username, up_to_id) -> dict:
    peer = _peer(db, peer_username)
    if peer is None:
        return {"ok": True, "marked": 0}
    cur = db.execute("UPDATE msg_messages SET read_at = ? WHERE recipient_id = ? AND sender_id = ? AND id <= ? AND read_at IS NULL",
                     (time.time(), me, peer[0], int(up_to_id)))
    return {"ok": True, "marked": cur.rowcount}


# ---- operator oversight (server.py gates these on account_manage) ----------------------------------------------------------------
def _audit(db, actor: int, action: str, target: str, now: float) -> None:
    db.execute("INSERT INTO msg_audit (actor_id, action, target, created) VALUES (?,?,?,?)", (actor, action, target, now))


def admin_thread(db, actor: int, a_name, b_name, before_id=None, limit: int = 50, now: float | None = None) -> dict:
    """Any two accounts' messages, original bodies included (each removed one flagged).  Audited."""
    now = time.time() if now is None else now
    a, b = _peer(db, a_name), _peer(db, b_name)
    if a is None or b is None:
        raise MessagingError("no such account", 404)
    limit = max(1, min(int(limit), 200))
    sql = "SELECT * FROM msg_messages WHERE ((sender_id = ? AND recipient_id = ?) OR (sender_id = ? AND recipient_id = ?))"
    args = [a[0], b[0], b[0], a[0]]
    if before_id is not None:
        sql += " AND id < ?"
        args.append(int(before_id))
    rows = db.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit + 1)).fetchall()
    more = len(rows) > limit
    rows = list(reversed(rows[:limit]))
    _audit(db, actor, "read_thread", f"{a[1]}|{b[1]}", now)
    names = _names(db, rows)
    return {"messages": [_row(r, names, admin=True) for r in rows], "more": more}


def admin_remove(db, actor: int, message_id, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    cur = db.execute("UPDATE msg_messages SET removed_at = ?, removed_by = ? WHERE id = ? AND removed_at IS NULL", (now, actor, int(message_id)))
    if cur.rowcount == 0 and db.execute("SELECT 1 FROM msg_messages WHERE id = ?", (int(message_id),)).fetchone() is None:
        raise MessagingError("no such message", 404)
    _audit(db, actor, "remove_message", str(int(message_id)), now)
    return {"ok": True, "id": int(message_id), "removed": True}


def admin_suspend(db, actor: int, username, suspended: bool, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    peer = _peer(db, username)
    if peer is None:
        raise MessagingError("no such account", 404)
    if suspended:
        db.execute("INSERT OR REPLACE INTO msg_suspended (user_id, by_id, created) VALUES (?,?,?)", (peer[0], actor, now))
    else:
        db.execute("DELETE FROM msg_suspended WHERE user_id = ?", (peer[0],))
    _audit(db, actor, "suspend" if suspended else "unsuspend", peer[1], now)
    return {"ok": True, "username": peer[1], "suspended": bool(suspended)}
