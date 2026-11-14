"""The chat server: persistent rooms and their messages. A room is a site object.

SYSTEM ROOMS are declared here (``ROOMS``) and open to the roles they name: ``reviewers`` is the operators and admins beside
the change reviews (branch_review.py). A change's author with less privilege talks in that change's thread (comments.py).

OWNED ROOMS are made by an account at dev level or above (dev_ui) and belong to it, as an authored page does:
    stage        dev (the development site; entering needs dev permission) or release (public facing)
    sharing      account_content.py, kind "chat_room": personal (the owner), group, or public (the owner's sign-off);
                 a room is entered by whoever passes BOTH its stage and its sharing (an admin always may)
    write        "members" (everyone who may enter) or "owner"
    objects      at most MAX_OBJECTS site objects the room offers ({kind, ref}: a page, a render, a mesh, a shader demo...)
    bridges      at most MAX_BRIDGES tool names (REST endpoints, which are MCP tools) the room points its bots at. A bridge is
                 a listing, never a grant: each caller's own capabilities still decide every call.
A bot in a room has the room's set and whatever its own account brings.

FORWARDING. An account that holds a secured connection (remote_connections.py: an MCP provider, called under its owner) may
lend a named subset of that connection's tools to a room it can write in, until an expiry. Every member who may write in the
room may then call those tools; the call runs under the LENDER's connection and the credential never leaves its vault. The
lender revokes at any time; each call is audited (who, which tool, the outcome) and limited per caller (CALLS_PER_HOUR). A
bot forwards what its own account holds in the same way.

Tables in accounts.db (schema module ``chat_rooms`` 2; this module is their one writer):
    chat_messages      (id, room, user_id, username, body, created, removed_at, removed_by)
    chat_audit         (at, admin, action, message_id)
    chat_room_defs     (room, title, about, owner_id, owner_name, stage, write, objects, bridges, created, updated)
    chat_room_grants   (id, room, owner_id, owner_name, provider, tools, created, expires, revoked_at)
    chat_grant_calls   (at, grant_id, caller_id, caller, tool, ok)

Text is stored stripped and only ever returned as a JSON string. Who is asking always comes from the authenticated session.

    GET  /api/chat/rooms                                 the rooms this account may enter
    POST /api/chat/rooms             {room, title, about?, stage?, write?}   make a room (dev level and up)
    GET  /api/chat/rooms/<room>?tail=N|after=ID          its messages, oldest first (a count or all; 20 by default)
    POST /api/chat/rooms/<room>      {body}              one message
    GET  /api/chat/rooms/<room>/settings                 its settings, set and sharing (who may enter)
    POST /api/chat/rooms/<room>/settings  {title?, about?, stage?, write?, objects?, bridges?}   the owner changes them
    POST /api/chat/rooms/<room>/access    {visibility, group_id, expected_revision, sign_off}   the owner's sharing
    GET  /api/chat/rooms/<room>/tools                    the room's set for a bot: objects, bridges, forwarded tools
    POST /api/chat/rooms/<room>/forward   {provider, tools, hours?}   lend tools of your own connection to the room
    POST /api/chat/rooms/<room>/forward/<grant>/revoke   the lender takes them back
    POST /api/chat/rooms/<room>/forward/<grant>/call  {name, arguments}   call one, under the lender's connection
    POST /api/chat/remove            {id}                an admin soft-deletes a message
"""
from __future__ import annotations

import json
import re
import time
import uuid
from contextlib import contextmanager

from meowdoku import schema_version

SCHEMA_VERSION = 2
MAX_BODY = 4000
MAX_TAIL = 1000
MAX_OBJECTS = 8
MAX_BRIDGES = 8
MAX_FORWARDED_TOOLS = 16
MAX_GRANT_HOURS = 720
CALLS_PER_HOUR = 30
ROOM = re.compile(r"[a-z0-9][a-z0-9-]{2,39}\Z")
TOOL = re.compile(r"[a-zA-Z][a-zA-Z0-9_.-]{0,63}\Z")
OBJECT_KINDS = ("page", "render", "mesh", "shader", "media", "room")
STAGES = ("dev", "release")
WRITE = ("members", "owner")
ROOMS = {
    "reviewers": {"title": "Reviewers' chat", "about": "The operators and admins, beside the change reviews.",
                  "roles": ("admin", "operator")},
}
DDL = (
    "CREATE TABLE IF NOT EXISTS chat_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, room TEXT NOT NULL, user_id INTEGER NOT NULL, "
    "username TEXT NOT NULL, body TEXT NOT NULL, created REAL NOT NULL, removed_at REAL, removed_by TEXT)",
    "CREATE INDEX IF NOT EXISTS chat_messages_by_room ON chat_messages(room, id)",
    "CREATE TABLE IF NOT EXISTS chat_audit (at REAL NOT NULL, admin TEXT NOT NULL, action TEXT NOT NULL, message_id INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS chat_room_defs (room TEXT PRIMARY KEY, title TEXT NOT NULL, about TEXT NOT NULL, "
    "owner_id INTEGER NOT NULL, owner_name TEXT NOT NULL, stage TEXT NOT NULL CHECK(stage IN ('dev','release')), "
    "write TEXT NOT NULL CHECK(write IN ('members','owner')), objects TEXT NOT NULL, bridges TEXT NOT NULL, "
    "created REAL NOT NULL, updated REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS chat_room_grants (id TEXT PRIMARY KEY, room TEXT NOT NULL, owner_id INTEGER NOT NULL, "
    "owner_name TEXT NOT NULL, provider TEXT NOT NULL, tools TEXT NOT NULL, created REAL NOT NULL, expires REAL NOT NULL, "
    "revoked_at REAL)",
    "CREATE TABLE IF NOT EXISTS chat_grant_calls (at REAL NOT NULL, grant_id TEXT NOT NULL, caller_id INTEGER NOT NULL, "
    "caller TEXT NOT NULL, tool TEXT NOT NULL, ok INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS chat_grant_calls_by_caller ON chat_grant_calls(grant_id, caller_id, at)",
)


class ChatError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def ensure(db) -> None:
    schema_version.declare(db, "chat_rooms", SCHEMA_VERSION)
    for statement in DDL:
        db.execute(statement)


@contextmanager
def _sharing(db):
    """account_content, which reads rows by name: its tables made (prepare may open a transaction, committed here) and
    the connection's rows named while it is used."""
    import sqlite3
    from meowdoku import account_content
    before = db.row_factory
    db.row_factory = sqlite3.Row
    try:
        account_content.prepare(db)
        if db.in_transaction:
            db.commit()
        yield account_content
        if db.in_transaction:
            db.commit()
    finally:
        db.row_factory = before


def role_of(db, user_id) -> str:
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'account_roles'").fetchone():
        return "member"
    row = db.execute("SELECT role FROM account_roles WHERE user_id = ?", (int(user_id),)).fetchone()
    return row[0] if row else "member"


def _owned(db, room):
    row = db.execute("SELECT room, title, about, owner_id, owner_name, stage, write, objects, bridges, created, updated "
                     "FROM chat_room_defs WHERE room = ?", (room,)).fetchone()
    if row is None:
        return None
    keys = ("room", "title", "about", "owner_id", "owner_name", "stage", "write", "objects", "bridges", "created", "updated")
    record = dict(zip(keys, row))
    record["objects"], record["bridges"] = json.loads(record["objects"]), json.loads(record["bridges"])
    return record


def may_enter(db, room, *, user_id, caps=frozenset()) -> bool:
    if room in ROOMS:
        return role_of(db, user_id) in ROOMS[room]["roles"]
    owned = _owned(db, room)
    if owned is None:
        return False
    if role_of(db, user_id) == "admin" or owned["owner_id"] == user_id:
        return True
    if owned["stage"] == "dev" and "dev_ui" not in caps:
        return False
    with _sharing(db) as account_content:
        return account_content.can_read(db, "chat_room", room, owned["owner_id"], user_id)


def may_write(db, room, *, user_id, caps=frozenset()) -> bool:
    if not may_enter(db, room, user_id=user_id, caps=caps):
        return False
    owned = _owned(db, room)
    return owned is None or owned["write"] == "members" or owned["owner_id"] == user_id


def _enter(db, room, user_id, caps, *, write=False) -> None:
    ensure(db)
    if room not in ROOMS and _owned(db, room) is None:
        raise ChatError("no such room", 404)
    if not may_enter(db, room, user_id=user_id, caps=caps):
        if room in ROOMS:
            raise ChatError(f"the room {room} is for {' and '.join(ROOMS[room]['roles'])} accounts", 403)
        raise ChatError("this room is not open to you", 403)
    if write and not may_write(db, room, user_id=user_id, caps=caps):
        raise ChatError("only the room's owner writes in this room", 403)


def rooms_for(db, user_id, caps=frozenset()) -> list:
    ensure(db)
    out = [{"room": name, "title": room["title"], "about": room["about"], "system": True}
           for name, room in ROOMS.items() if role_of(db, user_id) in room["roles"]]
    for (name,) in db.execute("SELECT room FROM chat_room_defs ORDER BY updated DESC"):
        if may_enter(db, name, user_id=user_id, caps=caps):
            owned = _owned(db, name)
            out.append({"room": name, "title": owned["title"], "about": owned["about"], "system": False,
                        "owner": owned["owner_name"], "stage": owned["stage"]})
    return out


def _text(value, what, limit, *, empty=False) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ChatError(f"{what} must be text of {0 if empty else 1} to {limit} characters")
    return value.strip()


def _objects(value) -> list:
    if not isinstance(value, list) or len(value) > MAX_OBJECTS:
        raise ChatError(f"objects must be a list of at most {MAX_OBJECTS}")
    out = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"kind", "ref"} or item["kind"] not in OBJECT_KINDS \
                or not isinstance(item["ref"], str) or not 1 <= len(item["ref"]) <= 200:
            raise ChatError(f"each object is {{kind, ref}} with kind one of {', '.join(OBJECT_KINDS)}")
        out.append({"kind": item["kind"], "ref": item["ref"]})
    return out


def _bridges(value) -> list:
    if not isinstance(value, list) or len(value) > MAX_BRIDGES or not all(isinstance(t, str) and TOOL.match(t) for t in value):
        raise ChatError(f"bridges must be a list of at most {MAX_BRIDGES} tool names")
    return list(dict.fromkeys(value))


def create(db, *, user_id, username, caps, room, title, about="", stage="dev", write="members", now=None) -> dict:
    """A room of one's own: dev level and up. It starts on the development site and personal (the owner alone)."""
    ensure(db)
    if "dev_ui" not in caps:
        raise ChatError("making a room needs dev level or above", 403)
    if not isinstance(room, str) or not ROOM.match(room) or room in ROOMS:
        raise ChatError("a room's name is 3 to 40 lowercase letters, digits and dashes, and not a system room's")
    if stage not in STAGES or write not in WRITE:
        raise ChatError("stage is dev or release; write is members or owner")
    now = time.time() if now is None else now
    try:
        db.execute("INSERT INTO chat_room_defs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (room, _text(title, "a title", 120), _text(about, "about", 500, empty=True), int(user_id), str(username),
                    stage, write, "[]", "[]", now, now))
    except Exception as error:
        if "UNIQUE" in str(error):
            raise ChatError("a room of that name exists", 409) from None
        raise
    return settings(db, room, user_id=user_id, caps=caps)


def _owner_or_admin(db, room, user_id) -> dict:
    ensure(db)
    owned = _owned(db, room)
    if owned is None:
        raise ChatError("no such room, or a system room (its settings are in code)", 404)
    if owned["owner_id"] != user_id and role_of(db, user_id) != "admin":
        raise ChatError("only the room's owner changes it", 403)
    return owned


def settings(db, room, *, user_id, caps=frozenset()) -> dict:
    _enter(db, room, user_id, caps)
    if room in ROOMS:
        return {"room": room, "system": True, "title": ROOMS[room]["title"], "about": ROOMS[room]["about"],
                "roles": list(ROOMS[room]["roles"])}
    owned = _owned(db, room)
    with _sharing(db) as account_content:
        sharing = account_content.policy(db, "chat_room", room)
    return {**owned, "system": False, "sharing": sharing}


def configure(db, room, *, user_id, caps, changes: dict, now=None) -> dict:
    owned = _owner_or_admin(db, room, user_id)
    allowed = {"title", "about", "stage", "write", "objects", "bridges"}
    if not isinstance(changes, dict) or not changes or set(changes) - allowed:
        raise ChatError(f"settings are {', '.join(sorted(allowed))}")
    new = dict(owned)
    if "title" in changes:
        new["title"] = _text(changes["title"], "a title", 120)
    if "about" in changes:
        new["about"] = _text(changes["about"], "about", 500, empty=True)
    if "stage" in changes:
        if changes["stage"] not in STAGES:
            raise ChatError("stage is dev or release")
        new["stage"] = changes["stage"]
    if "write" in changes:
        if changes["write"] not in WRITE:
            raise ChatError("write is members or owner")
        new["write"] = changes["write"]
    if "objects" in changes:
        new["objects"] = _objects(changes["objects"])
    if "bridges" in changes:
        new["bridges"] = _bridges(changes["bridges"])
    db.execute("UPDATE chat_room_defs SET title=?, about=?, stage=?, write=?, objects=?, bridges=?, updated=? WHERE room=?",
               (new["title"], new["about"], new["stage"], new["write"], json.dumps(new["objects"]), json.dumps(new["bridges"]),
                time.time() if now is None else now, room))
    return settings(db, room, user_id=user_id, caps=caps)


def share(db, room, *, user_id, body) -> dict:
    """The owner's sharing of the room: account_content.set_access (personal, group, or public with sign-off)."""
    owned = _owner_or_admin(db, room, user_id)
    with _sharing(db) as account_content:
        try:
            account_content.set_access(db, "chat_room", room, owned["owner_id"], user_id, body)
        except account_content.AccessError as error:
            raise ChatError(str(error), getattr(error, "status", 400)) from None
        return account_content.policy(db, "chat_room", room)


def _grants(db, room, now) -> list:
    rows = db.execute("SELECT id, owner_name, provider, tools, created, expires FROM chat_room_grants "
                      "WHERE room = ? AND revoked_at IS NULL AND expires > ? ORDER BY created", (room, now)).fetchall()
    return [{"grant": r[0], "owner": r[1], "provider": r[2], "tools": json.loads(r[3]), "created": r[4], "expires": r[5]}
            for r in rows]


def tools(db, room, *, user_id, caps=frozenset(), now=None) -> dict:
    """The room's set for a bot in it: its objects, its bridges (listings) and the tools members have forwarded."""
    _enter(db, room, user_id, caps)
    now = time.time() if now is None else now
    owned = _owned(db, room) or {"objects": [], "bridges": []}
    return {"room": room, "objects": owned["objects"], "bridges": owned["bridges"], "forwarded": _grants(db, room, now),
            "say": "Bridges are tools your own account calls with its own capabilities. Forwarded tools are called with "
                   "chat_forwarded_call under their lender's connection, by members who may write here."}


def forward(db, room, *, user_id, username, provider, tool_names, caps=frozenset(), hours=24, offered=(), now=None) -> dict:
    """Lend a subset of your own connection's tools to a room you may write in. ``offered`` are the tool names your
    connection to ``provider`` has (the host asks the connection service); a tool it does not offer cannot be lent."""
    _enter(db, room, user_id, caps, write=True)
    if not isinstance(provider, str) or not TOOL.match(provider):
        raise ChatError("provider must be a connection's provider name")
    if not isinstance(tool_names, list) or not tool_names or len(tool_names) > MAX_FORWARDED_TOOLS \
            or not all(isinstance(t, str) and TOOL.match(t) for t in tool_names):
        raise ChatError(f"tools must be a list of 1 to {MAX_FORWARDED_TOOLS} tool names")
    missing = sorted(set(tool_names) - set(offered))
    if missing:
        raise ChatError(f"your {provider} connection does not offer: {', '.join(missing)}", 409)
    if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not 0 < hours <= MAX_GRANT_HOURS:
        raise ChatError(f"hours must be more than 0 and at most {MAX_GRANT_HOURS}")
    now = time.time() if now is None else now
    grant = uuid.uuid4().hex
    db.execute("INSERT INTO chat_room_grants (id, room, owner_id, owner_name, provider, tools, created, expires) "
               "VALUES (?,?,?,?,?,?,?,?)", (grant, room, int(user_id), str(username), provider,
                                           json.dumps(list(dict.fromkeys(tool_names))), now, now + hours * 3600.0))
    return {"grant": grant, "room": room, "provider": provider, "tools": list(dict.fromkeys(tool_names)),
            "expires": now + hours * 3600.0}


def revoke_grant(db, room, grant, *, user_id, now=None) -> dict:
    ensure(db)
    row = db.execute("SELECT owner_id FROM chat_room_grants WHERE id = ? AND room = ?", (str(grant), room)).fetchone()
    if row is None:
        raise ChatError("no such forwarded grant in this room", 404)
    if row[0] != user_id and role_of(db, user_id) != "admin":
        raise ChatError("only its lender (or an admin) takes a forwarded grant back", 403)
    db.execute("UPDATE chat_room_grants SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
               (time.time() if now is None else now, str(grant)))
    return {"grant": grant, "revoked": True}


def forwarded_call(db, room, grant, *, user_id, username, name, arguments, call, caps=frozenset(), now=None):
    """One forwarded tool, under its lender's connection. ``call(owner_id, provider, name, arguments)`` is the host's
    connection service. Answers what the tool answered; every attempt that reaches the provider is audited."""
    _enter(db, room, user_id, caps, write=True)
    now = time.time() if now is None else now
    row = db.execute("SELECT owner_id, provider, tools FROM chat_room_grants WHERE id = ? AND room = ? AND revoked_at IS NULL "
                     "AND expires > ?", (str(grant), room, now)).fetchone()
    if row is None:
        raise ChatError("no such forwarded grant in this room, or it was revoked or has expired", 404)
    owner_id, provider, lent = row[0], row[1], json.loads(row[2])
    if not isinstance(name, str) or name not in lent:
        raise ChatError(f"this grant lends only: {', '.join(lent)}", 403)
    if not isinstance(arguments, dict):
        raise ChatError("arguments must be an object")
    recent = db.execute("SELECT count(*) FROM chat_grant_calls WHERE grant_id = ? AND caller_id = ? AND at > ?",
                        (str(grant), int(user_id), now - 3600.0)).fetchone()[0]
    if recent >= CALLS_PER_HOUR:
        raise ChatError(f"at most {CALLS_PER_HOUR} forwarded calls an hour per member and grant", 429)
    ok = False
    try:
        answer = call(owner_id, provider, name, arguments)
        ok = True
        return answer
    finally:
        db.execute("INSERT INTO chat_grant_calls (at, grant_id, caller_id, caller, tool, ok) VALUES (?,?,?,?,?,?)",
                   (now, str(grant), int(user_id), str(username), name, int(ok)))


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


def post(db, room, *, user_id, username, body, caps=frozenset(), now=None) -> dict:
    _enter(db, room, user_id, caps, write=True)
    if not isinstance(body, str) or not body.strip() or len(body) > MAX_BODY:
        raise ChatError(f"a message must be text of 1 to {MAX_BODY} characters")
    now = time.time() if now is None else now
    cursor = db.execute("INSERT INTO chat_messages (room, user_id, username, body, created) VALUES (?, ?, ?, ?, ?)",
                        (room, int(user_id), str(username), body.strip(), now))
    return _row(db.execute(_SELECT + " WHERE id = ?", (cursor.lastrowid,)).fetchone())


def messages(db, room, *, user_id, caps=frozenset(), tail=20, after=None) -> list:
    """The room oldest first: its last ``tail`` messages, or every message after ``after`` (at most MAX_TAIL)."""
    _enter(db, room, user_id, caps)
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


def http_route(self, path, method, *, accounts_db, connections=None):
    """``connections`` is the host's connection service (remote_connections.service_for), or None when there is none:
    forwarding then answers 503."""
    import sqlite3
    from urllib.parse import parse_qs
    principal = self._principal()
    if principal.user_id is None:
        return self._json({"error": "sign in first"}, 401)
    me = {"user_id": principal.user_id, "caps": frozenset(self._caps())}
    db = sqlite3.connect(accounts_db, timeout=10, isolation_level=None)
    try:
        parts = path[len("/api/chat/rooms/"):].split("/") if path.startswith("/api/chat/rooms/") else []
        room = parts[0] if parts else None
        rest = parts[1:]
        if method == "GET":
            if path == "/api/chat/rooms":
                return self._json({"rooms": rooms_for(db, **me)})
            if room and not rest:
                query = parse_qs(self.path.partition("?")[2])
                tail = tail_of((query.get("tail") or [None])[0])
                after = (query.get("after") or [None])[0]
                if after is not None:
                    try:
                        after = int(after)
                    except ValueError:
                        return self._json({"error": "after must be a message id"}, 400)
                return self._json({"room": room, "tail": tail, "messages": messages(db, room, **me, tail=tail, after=after)})
            if room and rest == ["settings"]:
                return self._json({"settings": settings(db, room, **me)})
            if room and rest == ["tools"]:
                return self._json(tools(db, room, **me))
            return self._json({"error": "not found"}, 404)
        if method != "POST":
            return self._json({"error": "not found"}, 404)
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return self._json({"error": "JSON only"}, 415)
        body = self._body_json(1 << 20)
        if body is None:
            return
        if not isinstance(body, dict):
            return self._json({"error": "a JSON object is required"}, 400)
        if path == "/api/chat/remove":
            if "account_manage" not in me["caps"]:
                return self._json({"error": "an admin removes a message"}, 403)
            return self._json({"message": remove(db, body.get("id"), admin=principal.username)})
        if path == "/api/chat/rooms":
            fields = {k: body[k] for k in ("room", "title", "about", "stage", "write") if k in body}
            return self._json({"settings": create(db, **me, username=principal.username, **fields)})
        if room and not rest:
            return self._json({"message": post(db, room, **me, username=principal.username, body=body.get("body"))})
        if room and rest == ["settings"]:
            return self._json({"settings": configure(db, room, **me, changes=body)})
        if room and rest == ["access"]:
            return self._json({"sharing": share(db, room, user_id=principal.user_id, body=body)})
        if room and rest[:1] == ["forward"]:
            if connections is None:
                return self._json({"error": "this server holds no connection service, so nothing can be forwarded"}, 503)
            if rest == ["forward"]:
                provider = body.get("provider")
                try:
                    offered = [t.get("name") for t in connections.tools(principal.user_id, provider) if isinstance(t, dict)]
                except Exception:
                    offered = []                  # no connection of yours to that provider: nothing to lend
                return self._json(forward(db, room, **me, username=principal.username, provider=provider,
                                          tool_names=body.get("tools"), hours=body.get("hours", 24), offered=offered))
            if len(rest) == 3 and rest[2] == "revoke":
                return self._json(revoke_grant(db, room, rest[1], user_id=principal.user_id))
            if len(rest) == 3 and rest[2] == "call":
                try:
                    answer = forwarded_call(db, room, rest[1], **me, username=principal.username, name=body.get("name"),
                                            arguments=body.get("arguments"), call=connections.call)
                except ChatError:
                    raise
                except Exception:
                    return self._json({"error": "the lender's provider could not answer"}, 502)
                return self._json({"result": answer})
        return self._json({"error": "not found"}, 404)
    except ChatError as error:
        return self._json({"error": error.message}, error.status)
    finally:
        db.close()
