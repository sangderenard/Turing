"""Sharing one's secured connection with chosen operator and admin accounts.

The owner of a connection (remote_connections.py: an MCP provider called under its owner) ticks, on the Connected
services page, which operator or admin accounts may use it. That tick is the whole grant: nothing else makes an account
a sharer. A shared account lists the provider's tools and calls them under the OWNER's connection. The credential never
leaves the owner's vault, a shared account never connects or disconnects it, and every call that reaches the provider
is audited (who, which tool, the outcome) and limited per caller (CALLS_PER_HOUR), as forwarding tools to a room is
(chat_rooms.py). A share holds only while the account is an operator or admin; it is checked at every use.

    GET  /api/connections/<provider>/shares                 the owner: the operator and admin accounts, ticked or not
    POST /api/connections/<provider>/shares  {account, shared}   the owner ticks or unticks one
    GET  /api/connections/<provider>/shared                 the connections shared with you
    GET  /api/connections/<provider>/shared/tools?owner=A   A's provider tools, if A shares with you
    POST /api/connections/<provider>/shared/call  {owner, name, arguments}   call one under A's connection
"""
from __future__ import annotations

import json
import sqlite3
import time
from urllib.parse import parse_qs, urlsplit

from meowdoku import schema_version
from meowdoku.chat_rooms import role_of

SCHEMA_VERSION = 1
ROLES = ("operator", "admin")
CALLS_PER_HOUR = 120
METHODS = {"shares": ("GET", "POST"), "shared": ("GET",), "shared/tools": ("GET",), "shared/call": ("POST",)}
ACTIONS = tuple(METHODS)


class ShareError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def ensure(db) -> None:
    schema_version.declare(db, "connection_shares", SCHEMA_VERSION)
    db.execute("CREATE TABLE IF NOT EXISTS connection_shares (owner_id INTEGER NOT NULL, provider TEXT NOT NULL, "
               "account_id INTEGER NOT NULL, granted REAL NOT NULL, PRIMARY KEY (owner_id, provider, account_id))")
    db.execute("CREATE TABLE IF NOT EXISTS connection_share_calls (at REAL NOT NULL, owner_id INTEGER NOT NULL, "
               "provider TEXT NOT NULL, caller_id INTEGER NOT NULL, caller TEXT NOT NULL, tool TEXT NOT NULL, "
               "ok INTEGER NOT NULL)")


def candidates(db, owner_id, provider) -> list:
    """The operator and admin accounts other than the owner, each with whether the owner shares with it."""
    ensure(db)
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'account_roles'").fetchone():
        return []
    shared = {r[0] for r in db.execute("SELECT account_id FROM connection_shares WHERE owner_id = ? AND provider = ?",
                                       (int(owner_id), provider))}
    rows = db.execute("SELECT u.id, u.username, r.role FROM users u JOIN account_roles r ON r.user_id = u.id "
                      "WHERE r.role IN ('operator', 'admin') AND u.id != ? ORDER BY u.username COLLATE NOCASE",
                      (int(owner_id),))
    return [{"username": r[1], "role": r[2], "shared": r[0] in shared} for r in rows]


def set_share(db, owner_id, provider, account, shared, now=None) -> list:
    ensure(db)
    if not isinstance(account, str) or not account or not isinstance(shared, bool):
        raise ShareError("give an account's name and whether it is shared (true or false)")
    row = db.execute("SELECT id FROM users WHERE username = ? COLLATE NOCASE", (account,)).fetchone()
    if row is None:
        raise ShareError("no such account", 404)
    if row[0] == owner_id:
        raise ShareError("your own connection is already yours")
    if shared:
        if role_of(db, row[0]) not in ROLES:
            raise ShareError("a connection is shared only with operator and admin accounts", 403)
        db.execute("INSERT OR IGNORE INTO connection_shares (owner_id, provider, account_id, granted) VALUES (?,?,?,?)",
                   (int(owner_id), provider, row[0], time.time() if now is None else now))
    else:
        db.execute("DELETE FROM connection_shares WHERE owner_id = ? AND provider = ? AND account_id = ?",
                   (int(owner_id), provider, row[0]))
    db.commit()
    return candidates(db, owner_id, provider)


def shared_with(db, account_id, provider) -> list:
    """The owners who share their connection to provider with this account, while it is an operator or admin."""
    ensure(db)
    if role_of(db, account_id) not in ROLES:
        return []
    rows = db.execute("SELECT u.id, u.username FROM connection_shares s JOIN users u ON u.id = s.owner_id "
                      "WHERE s.account_id = ? AND s.provider = ? ORDER BY u.username COLLATE NOCASE",
                      (int(account_id), provider))
    return [{"owner_id": r[0], "owner": r[1]} for r in rows]


def owner_of(db, account_id, provider, owner) -> int:
    """The id of the named owner, if it shares its connection to provider with this account."""
    for share in shared_with(db, account_id, provider):
        if isinstance(owner, str) and share["owner"].casefold() == owner.casefold():
            return share["owner_id"]
    raise ShareError(f"{owner!s} does not share a {provider} connection with you", 403)


def shared_call(db, *, owner_id, provider, caller_id, caller, name, arguments, call, now=None):
    """One tool under the owner's connection. The hourly limit is counted and the slot taken in one write transaction
    BEFORE the provider is asked; the outcome is written to that row afterwards (as chat_rooms.forwarded_call)."""
    if not isinstance(name, str) or not name:
        raise ShareError("name the tool")
    if not isinstance(arguments, dict):
        raise ShareError("arguments must be an object")
    now = time.time() if now is None else now
    if db.in_transaction:
        db.commit()
    db.execute("BEGIN IMMEDIATE")
    try:
        recent = db.execute("SELECT count(*) FROM connection_share_calls WHERE owner_id = ? AND provider = ? "
                            "AND caller_id = ? AND at > ?", (int(owner_id), provider, int(caller_id), now - 3600.0)
                            ).fetchone()[0]
        if recent >= CALLS_PER_HOUR:
            raise ShareError(f"at most {CALLS_PER_HOUR} calls an hour on a shared connection", 429)
        slot = db.execute("INSERT INTO connection_share_calls (at, owner_id, provider, caller_id, caller, tool, ok) "
                          "VALUES (?,?,?,?,?,?,0)", (now, int(owner_id), provider, int(caller_id), str(caller), name)
                          ).lastrowid
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    answer = call(owner_id, provider, name, arguments)     # a failure leaves the slot taken and its ok 0: audited, counted
    db.execute("UPDATE connection_share_calls SET ok = 1 WHERE rowid = ?", (slot,))
    db.commit()
    return answer


def http_route(handler, sessions, provider, action, method, user_id, service_for):
    """The share routes, after remote_connections.route checked HTTPS and the sign-in. ``service_for(sessions)`` is the
    host's connection service."""
    principal = handler._principal()
    if method not in METHODS[action]:
        return handler._json({"error": "method not allowed"}, 405)
    body = {}
    if method == "POST":
        expected = "https://" + handler.headers.get("Host", "")
        # The owner's page sends the same-origin header; an agent's REST client sends no Origin at all.
        if handler.headers.get("Origin") not in (None, expected) or handler.headers.get("Sec-Fetch-Site") == "cross-site":
            return handler._json({"error": "same-origin request required"}, 403)
        if action == "shares" and handler.headers.get("X-Meowdoku-Secret") != "1":
            return handler._json({"error": "same-origin connection request required"}, 403)
        if not (handler.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return handler._json({"error": "JSON only"}, 415)
        body = handler._body_json(1 << 20)
        if body is None:
            return
        if not isinstance(body, dict):
            return handler._json({"error": "invalid request"}, 400)
    db = sqlite3.connect(str(sessions.accounts_db), timeout=30)
    try:
        if action == "shares":
            if method == "GET":
                return handler._json({"provider": provider, "accounts": candidates(db, user_id, provider)})
            if set(body) != {"account", "shared"}:
                return handler._json({"error": "give account and shared"}, 400)
            accounts = set_share(db, user_id, provider, body["account"], body["shared"])
            return handler._json({"provider": provider, "accounts": accounts})
        if action == "shared":
            return handler._json({"provider": provider,
                                  "owners": [s["owner"] for s in shared_with(db, user_id, provider)]})
        if action == "shared/tools":
            owner = parse_qs(urlsplit(handler.path).query).get("owner", [""])[0]
            owner_id = owner_of(db, user_id, provider, owner)
            return handler._json({"owner": owner, "tools": service_for(sessions).tools(owner_id, provider)})
        if set(body) != {"owner", "name", "arguments"}:
            return handler._json({"error": "give owner, name and arguments"}, 400)
        owner_id = owner_of(db, user_id, provider, body["owner"])
        answer = shared_call(db, owner_id=owner_id, provider=provider, caller_id=user_id,
                             caller=getattr(principal, "username", None) or str(user_id), name=body["name"],
                             arguments=body["arguments"], call=service_for(sessions).call)
        return handler._json(answer)
    except ShareError as error:
        return handler._json({"error": str(error)}, error.status)
    except (PermissionError, ValueError, KeyError):
        return handler._json({"error": "connection request refused"}, 400)
    except sqlite3.Error:
        return handler._json({"error": "connection shares unavailable"}, 503)
    except Exception:
        return handler._json({"error": "provider operation unavailable"}, 502)
    finally:
        db.close()
