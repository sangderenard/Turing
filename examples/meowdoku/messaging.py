"""Meowdoku user-to-user messaging: contacts and messages in accounts.db (server.py [MESSAGES], web/messenger-widget.js).

Tables (schema module ``messaging`` 2; this module is their one writer):
    msg_contacts  (user_id, peer_id, state requested|accepted|blocked, created, updated)   DIRECTED rows, one per side
    msg_messages  (id, sender_id, recipient_id, body, created, read_at, ..., subject_kind, subject_ref, msg_type)   v2 adds the last three
    msg_cursor    (user_id, acked_id, updated)   the RECIPIENT's own ack cursor; never returned to any other account

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

import hashlib
import json
import secrets
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
SUBJECT_KINDS = ("lease", "job", "ticket", "file", "topic")
MSG_TYPES = ("note", "status", "question", "handoff", "approval_request", "answer")
MAX_SUBJECT_REF = 200
HANDLE_TTL = 60
HANDLE_ACTION = "send"

DDL = ("CREATE TABLE IF NOT EXISTS msg_contacts (user_id INTEGER NOT NULL, peer_id INTEGER NOT NULL, "
       "state TEXT NOT NULL CHECK(state IN ('requested','accepted','blocked')), created REAL NOT NULL, updated REAL NOT NULL, "
       "PRIMARY KEY(user_id, peer_id))",
       "CREATE TABLE IF NOT EXISTS msg_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, sender_id INTEGER NOT NULL, "
       "recipient_id INTEGER NOT NULL, body TEXT NOT NULL, created REAL NOT NULL, read_at REAL, removed_at REAL, removed_by INTEGER, "
       "subject_kind TEXT, subject_ref TEXT, msg_type TEXT NOT NULL DEFAULT 'note')",
       "CREATE TABLE IF NOT EXISTS msg_cursor (user_id INTEGER PRIMARY KEY, acked_id INTEGER NOT NULL, updated REAL NOT NULL)",
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
    """Own connection to accounts.db (autocommit).  Declares schema module ``messaging`` 2 (v1 migrates in place by ADD COLUMN)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(db_path), timeout=10, isolation_level=None, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    try:
        _schema_version.declare(db, "messaging", 2)
    except _schema_version.SchemaVersionError:
        db.close()
        raise
    for ddl in DDL:
        db.execute(ddl)
    have = {r["name"] for r in db.execute("PRAGMA table_info(msg_messages)")}
    for column, decl in (("subject_kind", "TEXT"), ("subject_ref", "TEXT"), ("msg_type", "TEXT NOT NULL DEFAULT 'note'")):
        if column not in have:
            try:
                db.execute(f"ALTER TABLE msg_messages ADD COLUMN {column} {decl}")
            except sqlite3.OperationalError as error:       # another process on the same database added it first
                if "duplicate column" not in str(error).lower():
                    raise
    _schema_version.declare(db, "messaging_rest", 1)
    db.execute("CREATE TABLE IF NOT EXISTS msg_handles (digest TEXT PRIMARY KEY, sender_id INTEGER NOT NULL, "
               "peer_id INTEGER NOT NULL, action TEXT NOT NULL, expires REAL NOT NULL, "
               "sender_updated REAL NOT NULL, peer_updated REAL NOT NULL)")
    db.execute("CREATE INDEX IF NOT EXISTS msg_handles_pair ON msg_handles(sender_id, peer_id, expires)")
    db.execute("CREATE TABLE IF NOT EXISTS msg_send_keys (sender_id INTEGER NOT NULL, peer_id INTEGER NOT NULL, "
               "request_key TEXT NOT NULL, fingerprint TEXT NOT NULL, message_id INTEGER NOT NULL, "
               "PRIMARY KEY(sender_id, peer_id, request_key))")
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


def _revoke_relationship_handles(db, a, b):
    db.execute("DELETE FROM msg_handles WHERE (sender_id=? AND peer_id=?) OR (sender_id=? AND peer_id=?)", (a, b, b, a))


def _put(db, user_id, peer_id, state, now):
    # Invalidate capabilities with the contact write, even if wall-clock timestamps repeat.
    _revoke_relationship_handles(db, user_id, peer_id)
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
            _revoke_relationship_handles(db, me, pid)
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
        db.execute("BEGIN IMMEDIATE")
        try:
            _put(db, me, peer[0], "blocked", now)
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
    return {"ok": True, "result": "blocked"}


def remove(db, me: int, peer_username) -> dict:
    """Drop the contact (and a request) in both directions, except a block the OTHER side holds against me."""
    peer = _peer(db, peer_username)
    if peer is not None and peer[0] != me:
        db.execute("BEGIN IMMEDIATE")
        try:
            _revoke_relationship_handles(db, me, peer[0])
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


def _relationship_stamp(db, me, peer):
    rows = db.execute("SELECT user_id, state, updated FROM msg_contacts WHERE "
                      "(user_id=? AND peer_id=?) OR (user_id=? AND peer_id=?)", (me, peer, peer, me)).fetchall()
    stamps = {r["user_id"]: r["updated"] for r in rows if r["state"] == "accepted"}
    return (stamps[me], stamps[peer]) if me in stamps and peer in stamps else None


def conversations(db, me: int, now=None) -> dict:
    """Issue sender-bound, short-lived send handles from existing relationships.

    Only digests are retained. Names label the returned picker; callers do not
    send a recipient identifier to obtain a handle or to submit its message.
    """
    db.execute("BEGIN IMMEDIATE")
    try:
        now = time.time() if now is None else now
        if _suspended(db, me):
            raise MessagingError("messaging unavailable", 403)
        out = contacts(db, me)
        db.execute("DELETE FROM msg_handles WHERE expires <= ?", (now,))
        for contact in out["contacts"]:
            peer = _peer(db, contact["username"])[0]
            stamp = _relationship_stamp(db, me, peer)
            token = secrets.token_urlsafe(32)
            expiry = now + HANDLE_TTL
            # Bound outstanding credentials per relationship while allowing tabs
            # and concurrent clients to retain separate handles.
            db.execute("DELETE FROM msg_handles WHERE digest IN (SELECT digest FROM msg_handles "
                       "WHERE sender_id=? AND peer_id=? ORDER BY expires DESC LIMIT -1 OFFSET 9)", (me, peer))
            db.execute("INSERT INTO msg_handles VALUES (?,?,?,?,?,?,?)",
                       (hashlib.sha256(token.encode()).hexdigest(), me, peer, HANDLE_ACTION, expiry, *stamp))
            contact.update(conversation_handle=token, expires_at=expiry)
        db.execute("COMMIT")
        return out
    except Exception:
        db.execute("ROLLBACK")
        raise


def _handle_peer(db, me, token, now):
    if not isinstance(token, str) or not 32 <= len(token) <= 128:
        raise MessagingError("conversation handle unavailable; refresh conversations", 403)
    row = db.execute("SELECT * FROM msg_handles WHERE digest=? AND sender_id=? AND action=? AND expires>?",
                     (hashlib.sha256(token.encode()).hexdigest(), me, HANDLE_ACTION, now)).fetchone()
    if row is None or _relationship_stamp(db, me, row["peer_id"]) != (row["sender_updated"], row["peer_updated"]):
        raise MessagingError("conversation handle unavailable; refresh conversations", 403)
    return row["peer_id"]


# ---- messages -------------------------------------------------------------------------------------------------------------------
def _clean(body) -> str:
    if not isinstance(body, str):
        raise MessagingError("body must be text", 400)
    text = body.strip()
    if not text or len(text) > MAX_BODY or "\x00" in text:
        raise MessagingError(f"body must be 1 to {MAX_BODY} characters", 400)
    return text


def _row(db_row, names, admin: bool = False) -> dict:
    """The one message shape a user sees: exactly id, from, to, created, body, removed, subject, type (no read/ack state, no account ids)."""
    gone = db_row["removed_at"] is not None
    return {"id": db_row["id"], "from": names[db_row["sender_id"]], "to": names[db_row["recipient_id"]],
            "created": db_row["created"], "body": db_row["body"] if (admin or not gone) else "", "removed": gone,
            "subject": None if db_row["subject_kind"] is None else {"kind": db_row["subject_kind"], "ref": db_row["subject_ref"]},
            "type": db_row["msg_type"]}


def _subject(subject):
    """(None, None), or (kind, ref) from {"kind": one of SUBJECT_KINDS, "ref": 1..MAX_SUBJECT_REF chars}."""
    if subject is None:
        return None, None
    if not isinstance(subject, dict) or set(subject) != {"kind", "ref"}:
        raise MessagingError("subject must be {kind, ref}", 400)
    kind, ref = subject["kind"], subject["ref"]
    if kind not in SUBJECT_KINDS or not isinstance(ref, str) or not ref.strip() or len(ref) > MAX_SUBJECT_REF or "\x00" in ref:
        raise MessagingError(f"subject kind must be one of {', '.join(SUBJECT_KINDS)} and ref 1 to {MAX_SUBJECT_REF} characters", 400)
    return kind, ref


def _type(msg_type) -> str:
    if msg_type is None:
        return "note"
    if msg_type not in MSG_TYPES:
        raise MessagingError(f"type must be one of {', '.join(MSG_TYPES)}", 400)
    return msg_type


def _subject_filter(kind, ref):
    """(sql fragment, args) for the optional subject_kind / subject_ref filters of thread and since."""
    sql, args = "", []
    if kind is not None:
        if kind not in SUBJECT_KINDS:
            raise MessagingError(f"subject_kind must be one of {', '.join(SUBJECT_KINDS)}", 400)
        sql += " AND subject_kind = ?"
        args.append(kind)
    if ref is not None:
        sql += " AND subject_ref = ?"
        args.append(str(ref))
    return sql, args


def _suspended(db, user_id) -> bool:
    return db.execute("SELECT 1 FROM msg_suspended WHERE user_id = ?", (user_id,)).fetchone() is not None


def _names(db, rows) -> dict:
    ids = {r["sender_id"] for r in rows} | {r["recipient_id"] for r in rows}
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    return {r["id"]: r["username"] for r in db.execute(f"SELECT id, username FROM users WHERE id IN ({marks})", tuple(ids))}


def send(db, me: int, peer_username, body, now: float | None = None, hub: Hub | None = None, subject=None, msg_type=None,
         conversation_handle=None, idempotency_key=None) -> dict:
    text = _clean(body)
    skind, sref = _subject(subject)
    mtype = _type(msg_type)
    using_handle = conversation_handle is not None
    if using_handle and (peer_username is not None or not isinstance(idempotency_key, str)
                         or not 16 <= len(idempotency_key) <= 128):
        raise MessagingError("handle sends require an idempotency_key (16-128 characters) and no recipient name", 400)
    fingerprint = hashlib.sha256(json.dumps([text, skind, sref, mtype], ensure_ascii=False,
                                           separators=(",", ":")).encode()).hexdigest()
    db.execute("BEGIN IMMEDIATE")
    try:
        now = time.time() if now is None else now
        if using_handle:
            peer_id = _handle_peer(db, me, conversation_handle, now)
        else:
            peer = _peer(db, peer_username)
            peer_id = peer[0] if peer is not None else None
        if _suspended(db, me) or peer_id is None or peer_id == me or not _mutual(db, me, peer_id):
            raise MessagingError("you cannot message this user", 403)
        if using_handle:
            prior = db.execute("SELECT k.fingerprint, m.id, m.created FROM msg_send_keys k "
                               "JOIN msg_messages m ON m.id=k.message_id WHERE k.sender_id=? AND k.peer_id=? AND k.request_key=?",
                               (me, peer_id, idempotency_key)).fetchone()
            if prior is not None:
                if prior["fingerprint"] != fingerprint:
                    raise MessagingError("idempotency key already used for different content", 409)
                db.execute("COMMIT")
                return {"ok": True, "id": prior["id"], "created": prior["created"]}
        minute = db.execute("SELECT COUNT(*) FROM msg_messages WHERE sender_id = ? AND created > ?", (me, now - 60)).fetchone()[0]
        day = db.execute("SELECT COUNT(*) FROM msg_messages WHERE sender_id = ? AND created > ?", (me, now - 86400)).fetchone()[0]
        if minute >= MAX_PER_MINUTE:
            raise MessagingError(f"at most {MAX_PER_MINUTE} messages per minute", 429)
        if day >= MAX_PER_DAY:
            raise MessagingError(f"at most {MAX_PER_DAY} messages per day", 429)
        mid = db.execute("INSERT INTO msg_messages (sender_id, recipient_id, body, created, subject_kind, subject_ref, msg_type) "
                         "VALUES (?,?,?,?,?,?,?)", (me, peer_id, text, now, skind, sref, mtype)).lastrowid
        if using_handle:
            db.execute("INSERT INTO msg_send_keys VALUES (?,?,?,?,?)", (me, peer_id, idempotency_key, fingerprint, mid))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    (hub or HUB).notify()
    return {"ok": True, "id": mid, "created": now}


def thread(db, me: int, peer_username, before_id=None, limit: int = 50, subject_kind=None, subject_ref=None) -> dict:
    """Messages between ``me`` and the peer, oldest first: the newest ``limit`` below ``before_id``.  An unknown peer is an empty thread."""
    limit = max(1, min(int(limit), 200))
    peer = _peer(db, peer_username)
    if peer is None:
        return {"messages": [], "more": False}
    args = [me, peer[0], peer[0], me]
    sql = "SELECT * FROM msg_messages WHERE ((sender_id = ? AND recipient_id = ?) OR (sender_id = ? AND recipient_id = ?))"
    extra, extra_args = _subject_filter(subject_kind, subject_ref)
    sql += extra
    args += extra_args
    if before_id is not None:
        sql += " AND id < ?"
        args.append(int(before_id))
    rows = db.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit + 1)).fetchall()
    more = len(rows) > limit
    rows = list(reversed(rows[:limit]))
    names = _names(db, rows)
    return {"messages": [_row(r, names) for r in rows], "more": more}


def since(db, me: int, after_id, limit: int = 200, subject_kind=None, subject_ref=None) -> list:
    """Every message to or from ``me`` with id > after_id, oldest first: the poll primitive."""
    limit = max(1, min(int(limit), 500))
    extra, extra_args = _subject_filter(subject_kind, subject_ref)
    rows = db.execute("SELECT * FROM msg_messages WHERE id > ? AND (sender_id = ? OR recipient_id = ?)" + extra + " ORDER BY id LIMIT ?",
                      (int(after_id or 0), me, me, *extra_args, limit)).fetchall()
    names = _names(db, rows)
    return [_row(r, names) for r in rows]


def latest_id(db, me: int) -> int:
    """The highest id of a message to or from ``me`` (0 when none): where a fresh poller starts."""
    return db.execute("SELECT COALESCE(MAX(id), 0) FROM msg_messages WHERE sender_id = ? OR recipient_id = ?", (me, me)).fetchone()[0]


def long_poll(db, me: int, after_id, wait: float, hub: Hub | None = None, limit: int = 200, subject_kind=None, subject_ref=None) -> list:
    hub = hub or HUB
    if int(after_id or 0) < 0:                  # after=-1: no messages, just ``latest_id`` (the caller reports it)
        return []
    wait = max(0.0, min(float(wait), MAX_WAIT_S))
    if wait <= 0:
        return since(db, me, after_id, limit, subject_kind, subject_ref)
    hub.enter(me)
    deadline = time.monotonic() + wait
    try:
        while True:
            seen = hub.seq                      # read BEFORE the query: a send in between shows as a changed seq
            rows = since(db, me, after_id, limit, subject_kind, subject_ref)
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


# ---- server-kept cursor (the recipient's own state; no function here returns it to anyone else) ----------------------------------
def _cursor(db, me) -> int:
    row = db.execute("SELECT acked_id FROM msg_cursor WHERE user_id = ?", (me,)).fetchone()
    return row["acked_id"] if row else 0


def inbox(db, me: int, limit: int = 200) -> dict:
    """Messages TO ``me`` with id > my acked cursor, oldest first (at most 200), plus the cursor."""
    limit = max(1, min(int(limit), 200))
    cursor = _cursor(db, me)
    rows = db.execute("SELECT * FROM msg_messages WHERE recipient_id = ? AND id > ? ORDER BY id LIMIT ?", (me, cursor, limit)).fetchall()
    names = _names(db, rows)
    return {"messages": [_row(r, names) for r in rows], "cursor": cursor}


def ack(db, me: int, up_to, now: float | None = None) -> dict:
    """Advance my cursor to ``up_to`` (monotonic; never beyond the latest id addressed to me).  Returns only my own cursor."""
    now = time.time() if now is None else now
    if isinstance(up_to, bool) or not isinstance(up_to, int) or up_to < 0:
        raise MessagingError("up_to must be a non-negative integer", 400)
    db.execute("BEGIN IMMEDIATE")
    try:
        top = db.execute("SELECT COALESCE(MAX(id), 0) FROM msg_messages WHERE recipient_id = ?", (me,)).fetchone()[0]
        cursor = max(_cursor(db, me), min(up_to, top))
        db.execute("INSERT INTO msg_cursor (user_id, acked_id, updated) VALUES (?,?,?) "
                   "ON CONFLICT(user_id) DO UPDATE SET acked_id = excluded.acked_id, updated = excluded.updated", (me, cursor, now))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return {"ok": True, "cursor": cursor}


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
