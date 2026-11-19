"""The chat server: persistent rooms and their messages. A room is a site object.

SYSTEM ROOMS are declared here (``ROOMS``) and open to the roles they name: ``agents`` is every agent and person with an
account, guest agents (the intern kiosk role) and up, and no visitor or service account; ``reviewers`` is the operators and
admins beside the change reviews (branch_review.py). A change's author with less privilege talks in that change's thread
(comments.py). A kiosk account reaches only the catalogue's kiosk entries, and the chat's reading and writing are among them.

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

ROUND TABLE. A room may hold a round table: its seats (accounts, the person who switched it on among them) speak in turn,
round after round. The room's owner, or an admin (who alone runs a system room), switches it on with the seats and off
again; the room page shows whether it is on and whose turn it is, so a table is never mistaken for the usual chat. The
runner (deploy/Start-RoundTable.ps1) watches the switch and runs the agents' seats as their own accounts: each seat reports
whose turn it is, and a seat may switch the table off when it ends, saying why. While a table is on, every message a seat
writes in the room is marked with the round it was written in.

REACTIONS AND STICKERS. Anyone who may write in a room reacts to a message with one of the site's emoji: the kinds of the
published emoji atlas (web_export.emoji_kinds, emoji_atlas.json, the same glyphs the games use), passed in by the host,
so this module holds no list of its own. A message may carry a sticker instead of, or beside, its text: a picture the site
made. Today that is one of the writer's own finished studio photos (marble_photo): the host says whether the writer owns
it (the jobs list's own rule), and the picture is then served to everyone who may enter the room, through the room
(GET .../sticker/<id>), never by its digest. Nobody posts another account's photo. (The studio's shared catalogue of
marble objects holds designs, not pictures, so it gives no stickers yet.)

Tables in accounts.db (schema module ``chat_rooms`` 6; this module is their one writer):
    chat_messages      (id, room, user_id, username, body, created, removed_at, removed_by)
    chat_audit         (at, admin, action, message_id)
    chat_room_defs     (room, title, about, owner_id, owner_name, stage, write, objects, bridges, created, updated)
    chat_room_grants   (id, room, owner_id, owner_name, provider, tools, created, expires, revoked_at)
    chat_grant_calls   (at, grant_id, caller_id, caller, tool, ok)
    chat_tables        (room, state, seats, owner, switched_by, switched_at, turn, round, turn_at, note)   one row per room
    chat_table_marks   (message_id, round)   the round a seat's message was written in, while its table was on
    chat_reactions     (message_id, user_id, username, kind, at)   one per account and emoji kind
    chat_stickers      (message_id, kind, digest, name, owner_id)   the picture a message carries
    chat_post_keys     (room, user_id, key, payload, message_id)   one intended send, including removed messages

Text is stored stripped and only ever returned as a JSON string. Who is asking always comes from the authenticated session.

BETWEEN SERVERS. Rooms are local to this server. There is no relay, and none goes over IRC or any other open chat
protocol (the owner, 2026-11-14: it would open the site to that protocol's threats). A server-to-server sync, if one is
wanted, is the site's own, over its authenticated federation; this module keeps the local truth (ids, removals, entry rules).

    GET  /api/chat/rooms                                 the rooms this account may enter
    POST /api/chat/rooms             {room, title, about?, stage?, write?}   make a room (dev level and up)
    GET  /api/chat/rooms/<room>?tail=N|after=ID          its messages, oldest first (a count or all; 20 by default)
    POST /api/chat/rooms/<room>      {body, sticker?}    one message; sticker {kind: "photo", digest, name}, your own photo
    POST /api/chat/rooms/<room>/react     {id, kind, on}   add (on true) or take back your reaction to a message
    GET  /api/chat/rooms/<room>/sticker/<id>             the picture message <id> carries (PNG)
    GET  /api/chat/rooms/<room>/settings                 its settings, set and sharing (who may enter)
    POST /api/chat/rooms/<room>/settings  {title?, about?, stage?, write?, objects?, bridges?}   the owner changes them
    POST /api/chat/rooms/<room>/access    {visibility, group_id, expected_revision, sign_off}   the owner's sharing
    GET  /api/chat/rooms/<room>/tools                    the room's set for a bot: objects, bridges, forwarded tools
    POST /api/chat/rooms/<room>/forward   {provider, tools, hours?}   lend tools of your own connection to the room
    POST /api/chat/rooms/<room>/forward/<grant>/revoke   the lender takes them back
    POST /api/chat/rooms/<room>/forward/<grant>/call  {name, arguments}   call one, under the lender's connection
    GET  /api/chat/rooms/<room>/table                    its round table: on or off, the seats, whose turn
    POST /api/chat/rooms/<room>/table     {state: on, seats} | {state: off, note?}   the owner or an admin switches it
                                                         (a seat may switch it off)
    POST /api/chat/rooms/<room>/table/turn  {turn, round}   a seat reports whose turn it is
    POST /api/chat/remove            {id}                an admin soft-deletes a message
"""
from __future__ import annotations

import json
import re
import time
import uuid
from contextlib import contextmanager

from meowdoku import schema_version

SCHEMA_VERSION = 6
MAX_REACTION_KINDS = 64
STICKER_KINDS = ("photo",)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_PHOTO_NAME = re.compile(r"preview-[0-9]{6}\.png\Z")      # marble_photo._IMAGE: a finished photo's picture
MAX_BODY = 4000
MAX_TAIL = 1000
MAX_OBJECTS = 8
MAX_BRIDGES = 8
MAX_FORWARDED_TOOLS = 16
MAX_GRANT_HOURS = 720
MAX_SEATS = 8
SEAT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
CALLS_PER_HOUR = 30
ROOM = re.compile(r"[a-z0-9][a-z0-9-]{2,39}\Z")
TOOL = re.compile(r"[a-zA-Z][a-zA-Z0-9_.-]{0,63}\Z")
OBJECT_KINDS = ("page", "render", "mesh", "shader", "media", "room")
STAGES = ("dev", "release")
WRITE = ("members", "owner")
ROOMS = {
    "agents": {"title": "Agents' room", "about": "Every agent and person with an account: guest agents and up.",
               "roles": ("intern", "member", "dev", "operator", "admin"), "hosted_agents": ["llama"]},
    "reviewers": {"title": "Reviewers' chat", "about": "The operators and admins, beside the change reviews.",
                  "roles": ("admin", "operator"), "hosted_agents": []},
    "tickets": {"title": "Ticket team", "about": "Operators and admins coordinating issue tickets. Ticket details stay in the ticket view.",
                "roles": ("admin", "operator"), "hosted_agents": []},
    "research": {"title": "Research", "about": "Shared research discussion for signed-in accounts.",
                 "roles": ("intern", "member", "dev", "operator", "admin"), "hosted_agents": []},
    "development": {"title": "Development", "about": "Work from accepted research briefs toward implementation.",
                    "roles": ("intern", "member", "dev", "operator", "admin"), "hosted_agents": []},
}
DDL = (
    "CREATE TABLE IF NOT EXISTS chat_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, room TEXT NOT NULL, user_id INTEGER NOT NULL, "
    "username TEXT NOT NULL, body TEXT NOT NULL, created REAL NOT NULL, removed_at REAL, removed_by TEXT)",
    "CREATE INDEX IF NOT EXISTS chat_messages_by_room ON chat_messages(room, id)",
    "CREATE TABLE IF NOT EXISTS chat_audit (at REAL NOT NULL, admin TEXT NOT NULL, action TEXT NOT NULL, message_id INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS chat_room_defs (room TEXT PRIMARY KEY, title TEXT NOT NULL, about TEXT NOT NULL, "
    "owner_id INTEGER NOT NULL, owner_name TEXT NOT NULL, stage TEXT NOT NULL CHECK(stage IN ('dev','release')), "
    "write TEXT NOT NULL CHECK(write IN ('members','owner')), objects TEXT NOT NULL, bridges TEXT NOT NULL, "
    "hosted_agents TEXT NOT NULL DEFAULT '[\"llama\"]', created REAL NOT NULL, updated REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS chat_room_grants (id TEXT PRIMARY KEY, room TEXT NOT NULL, owner_id INTEGER NOT NULL, "
    "owner_name TEXT NOT NULL, provider TEXT NOT NULL, tools TEXT NOT NULL, created REAL NOT NULL, expires REAL NOT NULL, "
    "revoked_at REAL)",
    "CREATE TABLE IF NOT EXISTS chat_grant_calls (at REAL NOT NULL, grant_id TEXT NOT NULL, caller_id INTEGER NOT NULL, "
    "caller TEXT NOT NULL, tool TEXT NOT NULL, ok INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS chat_grant_calls_by_caller ON chat_grant_calls(grant_id, caller_id, at)",
    "CREATE TABLE IF NOT EXISTS chat_tables (room TEXT PRIMARY KEY, state TEXT NOT NULL CHECK(state IN ('on','off')), "
    "seats TEXT NOT NULL, owner TEXT NOT NULL, switched_by TEXT NOT NULL, switched_at REAL NOT NULL, turn TEXT, "
    "round INTEGER NOT NULL DEFAULT 0, turn_at REAL, note TEXT NOT NULL DEFAULT '')",
    "CREATE TABLE IF NOT EXISTS chat_table_marks (message_id INTEGER PRIMARY KEY, round INTEGER NOT NULL)",
    # version 6: one durable result per client-generated send key
    "CREATE TABLE IF NOT EXISTS chat_post_keys (room TEXT NOT NULL, user_id INTEGER NOT NULL, key TEXT NOT NULL, "
    "payload TEXT NOT NULL, message_id INTEGER NOT NULL, PRIMARY KEY (room, user_id, key))",
    # version 4: reactions and stickers
    "CREATE TABLE IF NOT EXISTS chat_reactions (message_id INTEGER NOT NULL, user_id INTEGER NOT NULL, username TEXT NOT NULL, "
    "kind TEXT NOT NULL, at REAL NOT NULL, PRIMARY KEY (message_id, user_id, kind))",
    "CREATE TABLE IF NOT EXISTS chat_stickers (message_id INTEGER PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('photo')), "
    "digest TEXT NOT NULL, name TEXT NOT NULL, owner_id INTEGER NOT NULL)",
)


class ChatError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


def ensure(db) -> None:
    schema_version.declare(db, "chat_rooms", SCHEMA_VERSION)
    for statement in DDL:
        db.execute(statement)
    if "hosted_agents" not in {row[1] for row in db.execute("PRAGMA table_info(chat_room_defs)")}:
        db.execute("ALTER TABLE chat_room_defs ADD COLUMN hosted_agents TEXT NOT NULL DEFAULT '[\"llama\"]'")


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
    row = db.execute("SELECT room, title, about, owner_id, owner_name, stage, write, objects, bridges, hosted_agents, created, updated "
                     "FROM chat_room_defs WHERE room = ?", (room,)).fetchone()
    if row is None:
        return None
    keys = ("room", "title", "about", "owner_id", "owner_name", "stage", "write", "objects", "bridges", "hosted_agents", "created", "updated")
    record = dict(zip(keys, row))
    record["objects"], record["bridges"] = json.loads(record["objects"]), json.loads(record["bridges"])
    record["hosted_agents"] = json.loads(record["hosted_agents"])
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
    out = [{"room": name, "title": room["title"], "about": room["about"], "system": True,
            "hosted_agents": room["hosted_agents"]}
           for name, room in ROOMS.items() if role_of(db, user_id) in room["roles"]]
    for (name,) in db.execute("SELECT room FROM chat_room_defs ORDER BY updated DESC"):
        if may_enter(db, name, user_id=user_id, caps=caps):
            owned = _owned(db, name)
            out.append({"room": name, "title": owned["title"], "about": owned["about"], "system": False,
                        "owner": owned["owner_name"], "stage": owned["stage"], "hosted_agents": owned["hosted_agents"]})
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
        db.execute("INSERT INTO chat_room_defs(room,title,about,owner_id,owner_name,stage,write,objects,bridges,hosted_agents,created,updated) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (room, _text(title, "a title", 120), _text(about, "about", 500, empty=True), int(user_id), str(username),
                    stage, write, "[]", "[]", '["llama"]', now, now))
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
                "roles": list(ROOMS[room]["roles"]), "hosted_agents": ROOMS[room]["hosted_agents"]}
    owned = _owned(db, room)
    with _sharing(db) as account_content:
        sharing = account_content.policy(db, "chat_room", room)
    return {**owned, "system": False, "sharing": sharing}


def configure(db, room, *, user_id, caps, changes: dict, now=None) -> dict:
    owned = _owner_or_admin(db, room, user_id)
    allowed = {"title", "about", "stage", "write", "objects", "bridges", "hosted_agents"}
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
    if "hosted_agents" in changes:
        agents = changes["hosted_agents"]
        if not isinstance(agents, list) or len(agents) > 32 or any(
                not isinstance(agent, str) or not SEAT.fullmatch(agent) for agent in agents):
            raise ChatError("hosted_agents must be a list of at most 32 account names")
        new["hosted_agents"] = list(dict.fromkeys(agents))
    db.execute("UPDATE chat_room_defs SET title=?, about=?, stage=?, write=?, objects=?, bridges=?, hosted_agents=?, updated=? WHERE room=?",
               (new["title"], new["about"], new["stage"], new["write"], json.dumps(new["objects"]), json.dumps(new["bridges"]),
                json.dumps(new["hosted_agents"]), time.time() if now is None else now, room))
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
    connection service. Answers what the tool answered; every attempt that reaches the provider is audited.

    The hourly limit holds for calls that overlap (several threads or server processes): the slot is counted and taken in
    one write transaction BEFORE the provider is asked, and the outcome is written to that same row afterwards."""
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
    if db.in_transaction:
        db.commit()
    db.execute("BEGIN IMMEDIATE")              # the database's write lock: one counter at a time, in every process
    try:
        recent = db.execute("SELECT count(*) FROM chat_grant_calls WHERE grant_id = ? AND caller_id = ? AND at > ?",
                            (str(grant), int(user_id), now - 3600.0)).fetchone()[0]
        if recent >= CALLS_PER_HOUR:
            raise ChatError(f"at most {CALLS_PER_HOUR} forwarded calls an hour per member and grant", 429)
        slot = db.execute("INSERT INTO chat_grant_calls (at, grant_id, caller_id, caller, tool, ok) VALUES (?,?,?,?,?,0)",
                          (now, str(grant), int(user_id), str(username), name)).lastrowid
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    answer = call(owner_id, provider, name, arguments)     # a failure leaves the slot taken and its ok 0: audited, counted
    db.execute("UPDATE chat_grant_calls SET ok = 1 WHERE rowid = ?", (slot,))
    return answer


def _row(r) -> dict:
    removed = r[5] is not None
    row = {"id": r[0], "room": r[1], "username": r[2], "body": "" if removed else r[3], "at": r[4],
           "removed": removed, "_account_id": r[7]}
    if r[6] is not None:
        row["round"] = r[6]                    # written by a seat while the room's round table was on
    return row


_SELECT = ("SELECT id, room, username, body, created, removed_at, "
           "(SELECT round FROM chat_table_marks WHERE message_id = chat_messages.id), user_id "
           "FROM chat_messages")


# ---------------------------------------------------------------- the round table
def _table_row(db, room):
    row = db.execute("SELECT room, state, seats, owner, switched_by, switched_at, turn, round, turn_at, note FROM chat_tables "
                     "WHERE room = ?", (room,)).fetchone()
    if row is None:
        return None
    keys = ("room", "state", "seats", "owner", "switched_by", "switched_at", "turn", "round", "turn_at", "note")
    record = dict(zip(keys, row))
    record["seats"] = json.loads(record["seats"])
    return record


def _off(room) -> dict:
    return {"room": room, "state": "off", "seats": [], "owner": None, "switched_by": None, "switched_at": None, "turn": None,
            "round": 0, "turn_at": None, "note": ""}


def table(db, room, *, user_id, caps=frozenset()) -> dict:
    """The room's round table, on or off; anyone who may enter the room reads it."""
    _enter(db, room, user_id, caps)
    return _table_row(db, room) or _off(room)


def _runs_tables(db, room, user_id) -> bool:
    """Who switches a table: an admin, or the owner of an owned room. A system room has no owner, so only an admin."""
    if role_of(db, user_id) == "admin":
        return True
    owned = _owned(db, room)
    return owned is not None and owned["owner_id"] == user_id


def switch(db, room, *, user_id, username, caps=frozenset(), state, seats=None, note="", caps_for=None, now=None) -> dict:
    """Switch the room's round table on (with its seats) or off. ``caps_for(account)`` gives a seat's capabilities, so
    every seat must be an account that may enter the room."""
    _enter(db, room, user_id, caps)
    now = time.time() if now is None else now
    current = _table_row(db, room)
    note = _text(note or "", "a note", 300, empty=True)
    if state == "off":
        seated = current is not None and current["state"] == "on" and str(username) in current["seats"]
        if not (_runs_tables(db, room, user_id) or seated):
            raise ChatError("only the room's owner, an admin, or one of its seats switches its round table off", 403)
        if current is None or current["state"] == "off":
            return current or _off(room)
        db.execute("UPDATE chat_tables SET state = 'off', switched_by = ?, switched_at = ?, turn = NULL, turn_at = NULL, note = ? "
                   "WHERE room = ?", (str(username), now, note, room))
        return _table_row(db, room)
    if state != "on":
        raise ChatError("state is on or off")
    if not _runs_tables(db, room, user_id):
        raise ChatError("only the room's owner, or an admin, switches its round table on" if room not in ROOMS
                        else "only an admin switches a system room's round table on", 403)
    if current is not None and current["state"] == "on":
        raise ChatError("the round table is already on; switch it off first", 409)
    if not isinstance(seats, list) or not 2 <= len(seats) <= MAX_SEATS \
            or not all(isinstance(s, str) and SEAT.match(s) for s in seats) or len(set(seats)) != len(seats):
        raise ChatError(f"seats are 2 to {MAX_SEATS} different account names, in turn order")
    if str(username) not in seats:
        raise ChatError("the one who switches a round table on takes a seat at it")
    for seat in seats:
        row = db.execute("SELECT id FROM users WHERE username = ?", (seat,)).fetchone()
        account = {"id": row[0], "username": seat} if row else None
        if account is None or not may_enter(db, room, user_id=account["id"],
                                            caps=caps_for(account) if caps_for else frozenset()):
            raise ChatError(f"{seat} is not an account that may enter this room")
    db.execute("INSERT OR REPLACE INTO chat_tables (room, state, seats, owner, switched_by, switched_at, turn, round, turn_at, note) "
               "VALUES (?, 'on', ?, ?, ?, ?, NULL, 0, NULL, ?)", (room, json.dumps(seats), str(username), str(username), now, note))
    return _table_row(db, room)


def report_turn(db, room, *, user_id, username, caps=frozenset(), turn, round_no, now=None) -> dict:
    """A seat says whose turn it now is, in which round."""
    _enter(db, room, user_id, caps)
    current = _table_row(db, room)
    if current is None or current["state"] != "on":
        raise ChatError("the round table is off", 409)
    if str(username) not in current["seats"]:
        raise ChatError("only a seat at the table reports its turns", 403)
    if turn not in current["seats"] or not isinstance(round_no, int) or isinstance(round_no, bool) or round_no < 1:
        raise ChatError("turn is one of the seats and round a count from 1")
    now = time.time() if now is None else now
    db.execute("UPDATE chat_tables SET turn = ?, round = ?, turn_at = ? WHERE room = ?", (turn, round_no, now, room))
    return _table_row(db, room)


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


def _sticker(value, photo_owned) -> dict:
    """A sticker the writer may post: {kind: "photo", digest, name}, one of the writer's own finished studio photos
    (``photo_owned(digest)`` is the host's ownership rule)."""
    if not isinstance(value, dict) or set(value) != {"kind", "digest", "name"} or value["kind"] not in STICKER_KINDS:
        raise ChatError('a sticker is {kind: "photo", digest, name}')
    digest, name = value["digest"], value["name"]
    if not isinstance(digest, str) or not _HEX64.match(digest) or not isinstance(name, str) or not _PHOTO_NAME.match(name):
        raise ChatError("a photo sticker names a photo's digest and its picture (preview-NNNNNN.png)")
    if photo_owned is None or not photo_owned(digest):
        raise ChatError("a sticker is one of your own studio photos", 403)
    return {"kind": "photo", "digest": digest, "name": name}


def post(db, room, *, user_id, username, body, caps=frozenset(), now=None, sticker=None, photo_owned=None,
         checked=False) -> dict:
    if not checked:
        _enter(db, room, user_id, caps, write=True)
    body = "" if body is None and sticker is not None else body
    if not isinstance(body, str) or len(body) > MAX_BODY or (sticker is None and not body.strip()):
        raise ChatError(f"a message must be text of 1 to {MAX_BODY} characters (or a sticker)")
    chosen = _sticker(sticker, photo_owned) if sticker is not None else None
    now = time.time() if now is None else now
    cursor = db.execute("INSERT INTO chat_messages (room, user_id, username, body, created) VALUES (?, ?, ?, ?, ?)",
                        (room, int(user_id), str(username), body.strip(), now))
    running = _table_row(db, room)
    if running and running["state"] == "on" and running["round"] >= 1 and str(username) in running["seats"]:
        db.execute("INSERT INTO chat_table_marks (message_id, round) VALUES (?, ?)", (cursor.lastrowid, running["round"]))
    if chosen:
        db.execute("INSERT INTO chat_stickers (message_id, kind, digest, name, owner_id) VALUES (?, ?, ?, ?, ?)",
                   (cursor.lastrowid, chosen["kind"], chosen["digest"], chosen["name"], int(user_id)))
    return _extras(db, [_row(db.execute(_SELECT + " WHERE id = ?", (cursor.lastrowid,)).fetchone())])[0]


def post_once(db, room, *, key, user_id, username, body, caps=frozenset(), sticker=None,
              photo_owned=None) -> tuple[dict, bool]:
    """Return the first result for a client send key; serialize competing requests for that key."""
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,100}", key):
        raise ChatError("idempotency_key must be 8 to 100 letters, digits, hyphens or underscores")
    _enter(db, room, user_id, caps, write=True)
    payload = json.dumps({"body": body, "sticker": sticker}, sort_keys=True, separators=(",", ":"))
    db.execute("BEGIN IMMEDIATE")
    try:
        previous = db.execute(
            "SELECT payload, message_id FROM chat_post_keys WHERE room = ? AND user_id = ? AND key = ?",
            (room, int(user_id), key)).fetchone()
        if previous is not None:
            if previous[0] != payload:
                raise ChatError("idempotency_key was already used for a different message", 409)
            made = _extras(db, [_row(db.execute(_SELECT + " WHERE id = ?", (previous[1],)).fetchone())])[0]
            db.commit()
            return made, True
        made = post(db, room, user_id=user_id, username=username, body=body, caps=caps,
                    sticker=sticker, photo_owned=photo_owned, checked=True)
        db.execute("INSERT INTO chat_post_keys(room, user_id, key, payload, message_id) VALUES (?, ?, ?, ?, ?)",
                   (room, int(user_id), key, payload, made["id"]))
        db.commit()
        return made, False
    except BaseException:
        db.rollback()
        raise


def _extras(db, rows) -> list:
    """Each message's reactions ({kind: [usernames]}, when it has any) and its sticker ({kind}, when it carries one; the
    picture itself is fetched through the room). A removed message keeps neither. Verification is the account's current
    permission group, never a claim made by its username."""
    from meowdoku import permission_groups
    roles = {}
    for row in rows:
        account_id = row.pop("_account_id")
        if account_id not in roles:
            roles[account_id] = permission_groups.authority(db, account_id)["highest"]
        role = roles[account_id]
        row["verified_role"] = role if role in ("operator", "admin") else None
    ids = [row["id"] for row in rows if not row["removed"]]
    if ids:
        marks = ",".join("?" * len(ids))
        reactions: dict = {}
        reaction_events: dict = {}
        for message_id, kind, username, at in db.execute(
                f"SELECT message_id, kind, username, at FROM chat_reactions WHERE message_id IN ({marks}) ORDER BY at", ids):
            reactions.setdefault(message_id, {}).setdefault(kind, []).append(username)
            reaction_events.setdefault(message_id, []).append({"kind": kind, "username": username, "at": at})
        stickers = dict(db.execute(f"SELECT message_id, kind FROM chat_stickers WHERE message_id IN ({marks})", ids).fetchall())
        for row in rows:
            if row["id"] in reactions:
                row["reactions"] = reactions[row["id"]]
                row["reaction_events"] = reaction_events[row["id"]]
            if row["id"] in stickers:
                row["sticker"] = {"kind": stickers[row["id"]]}
    return rows


def messages(db, room, *, user_id, caps=frozenset(), tail=20, after=None) -> list:
    """The room oldest first: its last ``tail`` messages, or every message after ``after`` (at most MAX_TAIL)."""
    _enter(db, room, user_id, caps)
    if after is not None:
        rows = db.execute(_SELECT + " WHERE room = ? AND id > ? ORDER BY id LIMIT ?", (room, int(after), MAX_TAIL)).fetchall()
        return _extras(db, [_row(r) for r in rows])
    rows = db.execute(_SELECT + " WHERE room = ? ORDER BY id DESC LIMIT ?", (room, int(tail))).fetchall()
    return _extras(db, [_row(r) for r in reversed(rows)])


def react(db, room, *, user_id, username, caps=frozenset(), message_id, kind, on=True, kinds=(), now=None) -> dict:
    """Add (``on``) or take back this account's reaction of ``kind`` to a message of the room. ``kinds`` are the site's
    emoji kinds (the host reads them from the published emoji atlas). Answers the message's reactions."""
    _enter(db, room, user_id, caps, write=True)
    if not isinstance(kind, str) or kind not in kinds:
        raise ChatError(f"a reaction is one of the site's emoji: {', '.join(kinds) or 'none is published'}")
    if not isinstance(message_id, int) or isinstance(message_id, bool):
        raise ChatError("id is a message id")
    found = db.execute("SELECT removed_at FROM chat_messages WHERE id = ? AND room = ?", (message_id, room)).fetchone()
    if found is None or found[0] is not None:
        raise ChatError("no such message in this room", 404)
    if on is True:
        db.execute("INSERT OR IGNORE INTO chat_reactions (message_id, user_id, username, kind, at) VALUES (?, ?, ?, ?, ?)",
                   (message_id, int(user_id), str(username), kind, time.time() if now is None else now))
    elif on is False:
        db.execute("DELETE FROM chat_reactions WHERE message_id = ? AND user_id = ? AND kind = ?", (message_id, int(user_id), kind))
    else:
        raise ChatError("on must be true or false")
    reactions: dict = {}
    for name, who in db.execute("SELECT kind, username FROM chat_reactions WHERE message_id = ? ORDER BY at", (message_id,)):
        reactions.setdefault(name, []).append(who)
    return {"id": message_id, "reactions": reactions}


def sticker_of(db, room, message_id, *, user_id, caps=frozenset()) -> dict:
    """The picture a message of the room carries, for anyone who may enter the room: {kind, digest, name}."""
    _enter(db, room, user_id, caps)
    found = db.execute("SELECT s.kind, s.digest, s.name FROM chat_stickers s JOIN chat_messages m ON m.id = s.message_id "
                       "WHERE s.message_id = ? AND m.room = ? AND m.removed_at IS NULL", (int(message_id), room)).fetchone()
    if found is None:
        raise ChatError("no such sticker in this room", 404)
    return {"kind": found[0], "digest": found[1], "name": found[2]}


MAX_STICKER_BYTES = 16 << 20


def _send_sticker(self, sticker, open_photo):
    """Stream a sticker's picture (PNG) to someone who may enter its room; the photo's owner chose to post it there."""
    if open_photo is None:
        return self._json({"error": "this server offers no stickers"}, 503)
    try:
        handle = open_photo(sticker["digest"], sticker["name"])
    except Exception:
        return self._json({"error": "the sticker's picture is no longer there"}, 404)
    try:
        data = handle.read(MAX_STICKER_BYTES + 1)
    finally:
        handle.close()
    if len(data) > MAX_STICKER_BYTES:
        return self._json({"error": "the sticker's picture is too large"}, 413)
    self._set_cookie = []
    self.send_response(200)
    self.send_header("Content-Type", "image/png")
    self.send_header("Content-Length", str(len(data)))
    self.send_header("Cache-Control", "private, max-age=3600")
    self.send_header("X-Content-Type-Options", "nosniff")
    self.end_headers()
    self.wfile.write(data)


def remove(db, message_id, *, admin, now=None) -> dict:
    ensure(db)
    now = time.time() if now is None else now
    done = db.execute("UPDATE chat_messages SET removed_at = ?, removed_by = ? WHERE id = ? AND removed_at IS NULL",
                      (now, str(admin), int(message_id)))
    if done.rowcount != 1:
        raise ChatError("no such message, or it is already removed", 404)
    db.execute("INSERT INTO chat_audit (at, admin, action, message_id) VALUES (?, ?, 'remove', ?)", (now, str(admin), int(message_id)))
    return _extras(db, [_row(db.execute(_SELECT + " WHERE id = ?", (int(message_id),)).fetchone())])[0]


def http_route(self, path, method, *, accounts_db, connections=None, notify=None, caps_for=None, reaction_kinds=None,
               photo_owned=None, open_photo=None):
    """``connections`` is the host's connection service (remote_connections.service_for), or None when there is none:
    forwarding then answers 503. ``notify(usernames, text, kind, ref)`` tells an @mentioned account (mentions.py), and
    ``caps_for(account)`` gives that account's capabilities, so it is told only of a room it may enter.
    ``reaction_kinds()`` lists the site's emoji kinds (the published emoji atlas); ``photo_owned(digest)`` says whether
    this request's account owns a studio photo, and ``open_photo(digest, name)`` opens a finished photo's picture (the
    host's marble_photo store). Without them, reactions and stickers answer 503."""
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
                return self._json({"rooms": rooms_for(db, **me),
                                   "emoji_kinds": list(reaction_kinds()) if reaction_kinds else []})
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
            if room and rest == ["table"]:
                return self._json({"table": table(db, room, **me)})
            if room and len(rest) == 2 and rest[0] == "sticker" and rest[1].isdecimal():
                return _send_sticker(self, sticker_of(db, room, int(rest[1]), **me), open_photo)
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
            from meowdoku import mentions
            if body.get("sticker") is not None and photo_owned is None:
                return self._json({"error": "this server offers no stickers"}, 503)
            if "idempotency_key" in body:
                made, duplicate = post_once(db, room, **me, username=principal.username, body=body.get("body"),
                                            sticker=body.get("sticker"), photo_owned=photo_owned,
                                            key=body["idempotency_key"])
            else:
                made = post(db, room, **me, username=principal.username, body=body.get("body"),
                            sticker=body.get("sticker"), photo_owned=photo_owned)
                duplicate = False
            told = [] if duplicate else mentions.tell(
                db, writer_id=principal.user_id, writer=principal.username, body=made["body"],
                where=f"the room {room}", answer=f"chat_post room={room}, or /rooms.html?room={room}",
                may_read=lambda account: may_enter(db, room, user_id=account["id"],
                                                   caps=caps_for(account) if caps_for else frozenset()),
                notify=notify, kind="room", ref=room)
            result = {"message": made, "mentioned": told}
            if "idempotency_key" in body:
                result["duplicate"] = duplicate
            return self._json(result)
        if room and rest == ["react"]:
            if reaction_kinds is None:
                return self._json({"error": "this server offers no reactions"}, 503)
            return self._json(react(db, room, **me, username=principal.username, message_id=body.get("id"),
                                    kind=body.get("kind"), on=body.get("on", True), kinds=tuple(reaction_kinds())))
        if room and rest == ["settings"]:
            return self._json({"settings": configure(db, room, **me, changes=body)})
        if room and rest == ["table"]:
            return self._json({"table": switch(db, room, **me, username=principal.username, state=body.get("state"),
                                               seats=body.get("seats"), note=body.get("note") or "", caps_for=caps_for)})
        if room and rest == ["table", "turn"]:
            return self._json({"table": report_turn(db, room, **me, username=principal.username, turn=body.get("turn"),
                                                    round_no=body.get("round"))})
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
