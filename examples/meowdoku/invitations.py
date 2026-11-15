"""Invitations: a signed-in account invites someone with a link that carries a one-time code, and the link registers them.

Registration on this server is by invitation (open sign-up is closed on the public site). Any signed-in account except a
guest agent (intern) or a service account may invite. An invitation is:
    code       32 random bytes (secrets.token_urlsafe: 43 characters), shown to the inviter ONCE, in the link and its QR
               code. Only its SHA-256 is stored, so the database never holds a code that works.
    link       https://<this site>/invited.html#code=<code>. The code is in the fragment, which a browser never sends to a
               server: it stays out of access logs, the proxy and Referer headers. The page reads it, takes it out of the
               address bar, and sends it in a POST body.
    note       who it is for, so the inviter can tell their invitations apart (1 to 80 characters); an optional email
               for the inviter's own record (nothing is sent from here: the invite page opens the inviter's mail app)
    lifetime   DEFAULT_DAYS (1 to MAX_DAYS) and one use. The inviter (or an admin) may revoke it while it waits.
Redeeming creates an ordinary member account through accounts.py signup (its username and password rules) and signs it
in; the row records who used it, so who invited whom is kept, and the invited page shows the chain of inviters. A wrong
code counts against the login guard (login_guard.py, under the name LOOKUP_NAME, which no username can be), so codes
cannot be guessed at speed; an invitation that is used, revoked, expired or unknown answers the same.

Table in accounts.db (schema module ``invitations`` 1; this module is its one writer):
    site_invitations (id, token_hash, inviter_id, inviter, note, email, created, expires, used_at, used_by_id, used_by,
                      revoked_at, revoked_by)

    GET  /api/invitations                    your invitations and their state (never a code)
    POST /api/invitations                    {note, email?, days?}   make one; the answer carries its code and link path once
    POST /api/invitations/<id>/revoke        the inviter or an admin takes a waiting invitation back
    POST /api/invitations/check              {code}   no account needed: who invited you, until when
    POST /api/invitations/redeem             {code, username, password}   no account needed: register and sign in
"""
from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import time

from meowdoku import schema_version

SCHEMA_VERSION = 1
TOKEN_BYTES = 32
DEFAULT_DAYS = 14
MAX_DAYS = 90
MAX_WAITING = 20                      # waiting invitations per inviter
CHAIN_DEPTH = 6
LOOKUP_NAME = "~invitation"           # the login guard's name for code lookups; "~" is not allowed in a username
NOT_VALID = "this invitation is not valid: it may have been used, taken back or expired. Ask the person who invited you for a new one."
NO_INVITING = ("intern", "service")
EMAIL = re.compile(r"[^@\s]{1,100}@[^@\s]{1,100}\Z")
ID = re.compile(r"[0-9a-f]{12}\Z")
REGISTERING = "(registering)"
DDL = (
    "CREATE TABLE IF NOT EXISTS site_invitations (id TEXT PRIMARY KEY, token_hash TEXT UNIQUE NOT NULL, inviter_id INTEGER NOT NULL, "
    "inviter TEXT NOT NULL, note TEXT NOT NULL, email TEXT NOT NULL DEFAULT '', created REAL NOT NULL, expires REAL NOT NULL, "
    "used_at REAL, used_by_id INTEGER, used_by TEXT, revoked_at REAL, revoked_by TEXT)",
    "CREATE INDEX IF NOT EXISTS site_invitations_by_inviter ON site_invitations(inviter_id, created)",
)
FIELDS = ("id", "inviter_id", "inviter", "note", "email", "created", "expires", "used_at", "used_by_id", "used_by",
          "revoked_at", "revoked_by")


class InvitationError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def ensure(db) -> None:
    schema_version.declare(db, "invitations", SCHEMA_VERSION)
    for statement in DDL:
        db.execute(statement)


def digest(code: str) -> str:
    return hashlib.sha256(str(code).encode("utf-8")).hexdigest()


def role_of(db, user_id) -> str:
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'account_roles'").fetchone():
        return "member"
    row = db.execute("SELECT role FROM account_roles WHERE user_id = ?", (int(user_id),)).fetchone()
    return row[0] if row else "member"


def state_of(record: dict, now: float) -> str:
    if record["revoked_at"] is not None:
        return "taken back"
    if record["used_at"] is not None:
        return "registering" if record["used_by"] == REGISTERING else "used"
    return "expired" if record["expires"] <= now else "waiting"


def _record(row, now) -> dict:
    record = dict(zip(FIELDS, row))
    record["state"] = state_of(record, now)
    return record


_SELECT = "SELECT " + ", ".join(FIELDS) + " FROM site_invitations"


def make(db, *, user_id, username, note, email="", days=DEFAULT_DAYS, now=None) -> tuple[dict, str]:
    """A new invitation and its code (returned once; only its hash is kept)."""
    ensure(db)
    now = time.time() if now is None else now
    if role_of(db, user_id) in NO_INVITING:
        raise InvitationError("a guest agent or a service account does not invite", 403)
    if not isinstance(note, str) or not 1 <= len(note.strip()) <= 80:
        raise InvitationError("say who it is for: 1 to 80 characters")
    email = (email or "").strip() if isinstance(email, str) or email is None else None
    if email is None or (email and not EMAIL.match(email)):
        raise InvitationError("the email, if given, is one address")
    if not isinstance(days, int) or isinstance(days, bool) or not 1 <= days <= MAX_DAYS:
        raise InvitationError(f"days is a whole number from 1 to {MAX_DAYS}")
    waiting = db.execute("SELECT count(*) FROM site_invitations WHERE inviter_id = ? AND used_at IS NULL AND revoked_at IS NULL "
                         "AND expires > ?", (int(user_id), now)).fetchone()[0]
    if waiting >= MAX_WAITING:
        raise InvitationError(f"at most {MAX_WAITING} invitations wait at once; take one back or let one be used first", 429)
    code, ident = secrets.token_urlsafe(TOKEN_BYTES), secrets.token_hex(6)
    db.execute("INSERT INTO site_invitations (id, token_hash, inviter_id, inviter, note, email, created, expires) "
               "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
               (ident, digest(code), int(user_id), str(username), " ".join(note.split()), email, now, now + days * 86400.0))
    return _record(db.execute(_SELECT + " WHERE id = ?", (ident,)).fetchone(), now), code


def mine(db, *, user_id, now=None) -> list:
    ensure(db)
    now = time.time() if now is None else now
    return [_record(r, now) for r in db.execute(_SELECT + " WHERE inviter_id = ? ORDER BY created DESC LIMIT 200", (int(user_id),))]


def revoke(db, ident, *, user_id, username, now=None) -> dict:
    ensure(db)
    now = time.time() if now is None else now
    row = db.execute(_SELECT + " WHERE id = ?", (str(ident),)).fetchone() if ID.match(str(ident)) else None
    if row is None:
        raise InvitationError("no such invitation", 404)
    record = _record(row, now)
    if record["inviter_id"] != user_id and role_of(db, user_id) != "admin":
        raise InvitationError("only its inviter, or an admin, takes an invitation back", 403)
    if record["state"] != "waiting":
        raise InvitationError(f"this invitation is {record['state']}, not waiting", 409)
    db.execute("UPDATE site_invitations SET revoked_at = ?, revoked_by = ? WHERE id = ? AND used_at IS NULL", (now, str(username), record["id"]))
    return _record(db.execute(_SELECT + " WHERE id = ?", (record["id"],)).fetchone(), now)


def chain(db, inviter: str) -> list:
    """Who invited the inviter, and who invited them, nearest first (as far as CHAIN_DEPTH)."""
    out, name = [str(inviter)], str(inviter)
    for _ in range(CHAIN_DEPTH):
        row = db.execute("SELECT inviter FROM site_invitations WHERE used_by = ? AND used_at IS NOT NULL", (name,)).fetchone()
        if row is None or row[0] in out:
            break
        out.append(row[0])
        name = row[0]
    return out


def _waiting(db, code, now):
    if not isinstance(code, str) or not 20 <= len(code) <= 100:
        return None
    row = db.execute(_SELECT + " WHERE token_hash = ?", (digest(code.strip()),)).fetchone()
    record = _record(row, now) if row else None
    return record if record and record["state"] == "waiting" else None


def check(db, code, *, now=None) -> dict:
    ensure(db)
    now = time.time() if now is None else now
    record = _waiting(db, code, now)
    if record is None:
        raise InvitationError(NOT_VALID, 404)
    return {"inviter": record["inviter"], "chain": chain(db, record["inviter"]), "expires": record["expires"]}


def redeem(db, code, *, username, password, signup, now=None) -> dict:
    """Register with a waiting invitation. ``signup(username, password)`` is accounts.py's; its refusal (a taken name, a short
    password) leaves the invitation waiting. The invitation is claimed first, in one conditional update, so two people with
    the same link cannot both register with it."""
    ensure(db)
    now = time.time() if now is None else now
    record = _waiting(db, code, now)
    if record is None:
        raise InvitationError(NOT_VALID, 404)
    claimed = db.execute("UPDATE site_invitations SET used_at = ?, used_by = ? WHERE id = ? AND used_at IS NULL AND revoked_at IS NULL "
                         "AND expires > ?", (now, REGISTERING, record["id"], now)).rowcount
    if claimed != 1:
        raise InvitationError(NOT_VALID, 404)
    try:
        user = signup(username, password)
    except BaseException:
        db.execute("UPDATE site_invitations SET used_at = NULL, used_by = NULL WHERE id = ? AND used_by = ?", (record["id"], REGISTERING))
        raise
    db.execute("UPDATE site_invitations SET used_by_id = ?, used_by = ? WHERE id = ?", (int(user["id"]), str(user["username"]), record["id"]))
    return {"account": {"id": user["id"], "username": user["username"]}, "invited_by": record["inviter"]}


def http_route(self, path, method, *, accounts_db, signup, sign_in, account_error=(), guard=None):
    """``signup(username, password)`` and ``account_error`` are accounts.py's (its refusals carry .status); ``sign_in(user)``
    sets the new account's session cookie on this answer. ``guard`` is login_guard (check/record/connect). A new invitation's
    answer carries its code and the link's path; the page puts its own origin in front (behind the proxy the Host header
    is not the site's public name)."""
    principal = self._principal()
    db = sqlite3.connect(accounts_db, timeout=10, isolation_level=None)
    try:
        ensure(db)
        rest = path[len("/api/invitations"):].strip("/").split("/") if path != "/api/invitations" else []
        if method == "GET":
            if rest:
                return self._json({"error": "not found"}, 404)
            if principal.user_id is None:
                return self._json({"error": "sign in to see your invitations"}, 401)
            return self._json({"invitations": mine(db, user_id=principal.user_id)})
        if method != "POST":
            return self._json({"error": "not found"}, 404)
        body = self._body_json(4096)
        if body is None:
            return
        if rest in (["check"], ["redeem"]):
            client = getattr(self, "_client_ip", None) or self.client_address[0]
            gate = guard.connect(accounts_db) if guard else None
            try:
                wait = guard.check(gate, client, LOOKUP_NAME) if gate else 0
                if wait:
                    return self._json({"error": "too many tries; wait and try again", "retry_after": int(wait)}, 429)
                try:
                    if rest == ["check"]:
                        answer = check(db, body.get("code"))
                    else:
                        answer = redeem(db, body.get("code"), username=str(body.get("username", "")),
                                        password=str(body.get("password", "")), signup=signup)
                except InvitationError as error:
                    if gate and error.status == 404:
                        guard.record(gate, client, LOOKUP_NAME, False)
                    raise
            finally:
                if gate:
                    gate.close()
            if rest == ["redeem"]:
                sign_in(answer["account"])
                return self._json(answer, 201)
            return self._json(answer)
        if principal.user_id is None:
            return self._json({"error": "sign in to invite someone"}, 401)
        if not rest:
            made, code = make(db, user_id=principal.user_id, username=principal.username, note=body.get("note"),
                              email=body.get("email", ""), days=body.get("days", DEFAULT_DAYS))
            return self._json({"invitation": made, "code": code, "path": f"/invited.html#code={code}"}, 201)
        if len(rest) == 2 and rest[1] == "revoke":
            return self._json({"invitation": revoke(db, rest[0], user_id=principal.user_id, username=principal.username)})
        return self._json({"error": "not found"}, 404)
    except InvitationError as error:
        return self._json({"error": error.message}, error.status)
    except account_error as error:
        return self._json({"error": str(error)}, getattr(error, "status", 400))
    finally:
        db.close()
