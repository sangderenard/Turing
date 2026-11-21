"""account_admin: the admin's view of accounts and of the SpeakToMe tools their groups give them."""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from meowdoku import account_admin, authz, permission_groups  # noqa: E402
from meowdoku.rest_api import catalog  # noqa: E402  the shared SpeakToMe catalog, as server.py loads it

LIFT = {"member": "play", "dev": "dev", "service": "play", "operator": "dev", "intern": "play", "admin": "edit"}


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "accounts.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, "
                 "display_name TEXT, created_at REAL NOT NULL)")
    conn.executemany("INSERT INTO users VALUES (?, ?, 'x', ?, ?)",
                     [(1, "boss", "Boss", 100.0), (2, "helper", "Helper", 200.0), (3, "newcomer", "Newcomer", 300.0)])
    conn.commit()
    conn.close()
    authz.grant_admin(path, "boss", "operator:test")
    conn = authz._connect(path)
    permission_groups.prepare(conn)
    permission_groups.set_grant(conn, 1, 2, "operator", True)
    conn.commit()
    conn.close()
    return path


def tools(db, username, tier="edit"):
    return account_admin.tools_for(db, username, origin="lan", origin_tier=tier, policy=authz.DEFAULT_POLICY,
                                   catalog=catalog, lift=LIFT)


def test_people_lists_every_account_with_its_authority(db):
    listed = {p["username"]: p for p in account_admin.people(db)}
    assert set(listed) == {"boss", "helper", "newcomer"}
    assert listed["boss"]["authority"]["highest"] == "admin" and listed["boss"]["role_row"] == "admin"
    assert listed["helper"]["authority"]["highest"] == "operator" and listed["helper"]["role_row"] is None, "a grant, not a role row"
    assert listed["newcomer"]["authority"]["highest"] == "member"


def test_tools_follow_the_groups(db):
    boss, helper, newcomer = tools(db, "boss"), tools(db, "helper"), tools(db, "newcomer")
    names = lambda t: {x["name"] for x in t["tools"]}
    assert "messages_admin_remove" in names(boss), "an account_manage tool is an admin's"
    assert "messages_admin_remove" not in names(helper) | names(newcomer)
    missing = {x["name"]: x for x in helper["not_offered"]}
    assert missing["messages_admin_remove"]["needs"] == "account_manage", "a tool not offered says what it needs"
    assert "control_instances" in names(helper), "any one of its capabilities offers it (lifecycle_admin, on the LAN)"
    assert names(newcomer) <= names(helper) <= names(boss), "a higher group never offers fewer tools"
    assert all(x["name"] for x in boss["tools"]) and boss["origin"] == "lan" and boss["role"] == "admin"
    granted = set(boss["capabilities"])
    assert "account_manage" in granted and "account_manage" not in set(newcomer["capabilities"])


def test_a_shader_request_is_lifted_by_the_accounts_role_as_the_server_does(db):
    assert tools(db, "boss", tier="shader")["tier"] == "edit"
    assert tools(db, "newcomer", tier="shader")["tier"] == "play"


def test_the_route_answers_an_admin_and_refuses_what_it_cannot_answer(db):
    class Handler:
        _origin, _origin_tier = "lan", "edit"

        def __init__(self, path):
            self.path, self.sent = path, None

        def _json(self, payload, status=200):
            self.sent = (int(status), payload)

    def get(path):
        handler = Handler(path)
        account_admin.http_route(handler, path.split("?")[0], accounts_db=db, policy=authz.DEFAULT_POLICY, catalog=catalog, lift=LIFT)
        return handler.sent

    status, body = get("/api/account/admin/people")
    assert status == 200 and len(body["people"]) == 3
    assert get("/api/account/admin/tools")[0] == 400
    assert get("/api/account/admin/tools?username=nobody")[0] == 404
    status, body = get("/api/account/admin/tools?username=helper")
    assert status == 200 and body["username"] == "helper" and json.dumps(body)
