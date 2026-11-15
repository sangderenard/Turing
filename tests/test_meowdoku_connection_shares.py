"""Sharing a secured connection with ticked operator and admin accounts (connection_shares.py), through the host route
(remote_connections.route) with a fake connection service: no provider is called."""
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku import connection_shares as shares, remote_connections  # noqa: E402

OWNER, CLAUDE, CODEX, MEMBER = 3, 17, 18, 19


class Handler:
    def __init__(self, user_id, username, path, body=None, headers=None):
        self.path, self.body, self.output = path, body, None
        self.headers = {"Host": "example.test", "Content-Type": "application/json", **(headers or {})}
        self.user = types.SimpleNamespace(user_id=user_id, username=username)
    def _is_secure(self): return True
    def _principal(self): return self.user
    def _body_json(self, limit): return self.body
    def _json(self, value, status=200): self.output = (status, value)


class Service:
    def __init__(self): self.calls = []
    def tools(self, owner, provider): return [{"name": "library-list", "owner": owner}]
    def call(self, owner, provider, name, arguments):
        self.calls.append((owner, provider, name, arguments))
        return {"content": [{"type": "text", "text": name}], "owner": owner}


class ShareTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.db_path = Path(self.folder.name) / "accounts.db"
        db = sqlite3.connect(self.db_path)
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT NOT NULL)")
        db.execute("CREATE TABLE account_roles (user_id INTEGER PRIMARY KEY, role TEXT NOT NULL)")
        for uid, name, role in ((OWNER, "admin", "admin"), (CLAUDE, "operator-claude", "operator"),
                                (CODEX, "operator-codex", "operator"), (MEMBER, "dot", "member")):
            db.execute("INSERT INTO users VALUES (?, ?)", (uid, name))
            db.execute("INSERT INTO account_roles VALUES (?, ?)", (uid, role))
        db.commit()
        db.close()
        self.sessions = types.SimpleNamespace(accounts_db=self.db_path)
        self.service = Service()
        self.original = remote_connections.service_for
        remote_connections.service_for = lambda sessions: self.service
    def tearDown(self):
        remote_connections.service_for = self.original
        self.folder.cleanup()
    def route(self, user_id, username, action, method="GET", body=None, query="", headers=None):
        h = Handler(user_id, username, "/api/connections/suno/" + action + query, body, headers)
        remote_connections.route(h, self.sessions, "/api/connections/suno/" + action, method)
        return h.output
    def tick(self, account, shared=True):
        return self.route(OWNER, "admin", "shares", "POST", {"account": account, "shared": shared},
                          headers={"Origin": "https://example.test", "X-Meowdoku-Secret": "1"})

    def test_the_owner_sees_operators_and_admins_but_not_members_or_itself(self):
        status, answer = self.route(OWNER, "admin", "shares")
        self.assertEqual(status, 200)
        self.assertEqual([(a["username"], a["shared"]) for a in answer["accounts"]],
                         [("operator-claude", False), ("operator-codex", False)])
        self.assertEqual(self.tick("operator-claude")[1]["accounts"][0]["shared"], True)

    def test_only_operators_and_admins_can_be_ticked(self):
        self.assertEqual(self.tick("dot")[0], 403)
        self.assertEqual(self.tick("nobody")[0], 404)
        self.assertEqual(self.tick("admin")[0], 400)

    def test_ticking_needs_the_owners_page(self):
        status, _ = self.route(OWNER, "admin", "shares", "POST", {"account": "operator-claude", "shared": True})
        self.assertEqual(status, 403)                # no X-Meowdoku-Secret: not the Connected services page

    def test_a_ticked_account_calls_under_the_owners_connection_and_is_audited(self):
        self.tick("operator-claude")
        self.assertEqual(self.route(CLAUDE, "operator-claude", "shared")[1]["owners"], ["admin"])
        status, tools = self.route(CLAUDE, "operator-claude", "shared/tools", query="?owner=admin")
        self.assertEqual((status, tools["tools"][0]["owner"]), (200, OWNER))
        status, answer = self.route(CLAUDE, "operator-claude", "shared/call", "POST",
                                    {"owner": "admin", "name": "library-list", "arguments": {"limit": 5}})
        self.assertEqual((status, answer["owner"]), (200, OWNER))
        self.assertEqual(self.service.calls, [(OWNER, "suno", "library-list", {"limit": 5})])
        db = sqlite3.connect(self.db_path)
        audited = db.execute("SELECT owner_id, caller, tool, ok FROM connection_share_calls").fetchall()
        db.close()
        self.assertEqual(audited, [(OWNER, "operator-claude", "library-list", 1)])

    def test_an_unticked_account_is_refused_before_the_provider(self):
        self.tick("operator-claude")
        status, _ = self.route(CODEX, "operator-codex", "shared/call", "POST",
                               {"owner": "admin", "name": "library-list", "arguments": {}})
        self.assertEqual(status, 403)
        self.tick("operator-claude", False)
        status, _ = self.route(CLAUDE, "operator-claude", "shared/tools", query="?owner=admin")
        self.assertEqual(status, 403)
        self.assertEqual(self.service.calls, [])

    def test_a_share_lapses_when_the_account_is_no_longer_an_operator(self):
        self.tick("operator-claude")
        db = sqlite3.connect(self.db_path)
        db.execute("UPDATE account_roles SET role = 'member' WHERE user_id = ?", (CLAUDE,))
        db.commit()
        db.close()
        status, _ = self.route(CLAUDE, "operator-claude", "shared/call", "POST",
                               {"owner": "admin", "name": "library-list", "arguments": {}})
        self.assertEqual(status, 403)
        self.assertEqual(self.service.calls, [])

    def test_a_foreign_origin_is_refused_and_an_agent_without_one_is_not(self):
        self.tick("operator-claude")
        call = {"owner": "admin", "name": "library-list", "arguments": {}}
        self.assertEqual(self.route(CLAUDE, "operator-claude", "shared/call", "POST", call,
                                    headers={"Origin": "https://evil.test"})[0], 403)
        self.assertEqual(self.route(CLAUDE, "operator-claude", "shared/call", "POST", call)[0], 200)

    def test_calls_are_limited_per_hour(self):
        self.tick("operator-claude")
        db = sqlite3.connect(self.db_path)
        db.executemany("INSERT INTO connection_share_calls VALUES (strftime('%s','now'), ?, 'suno', ?, ?, 'x', 1)",
                       [(OWNER, CLAUDE, "operator-claude")] * shares.CALLS_PER_HOUR)
        db.commit()
        db.close()
        status, _ = self.route(CLAUDE, "operator-claude", "shared/call", "POST",
                               {"owner": "admin", "name": "library-list", "arguments": {}})
        self.assertEqual(status, 429)
        self.assertEqual(self.service.calls, [])

    def test_wrong_methods(self):
        self.assertEqual(self.route(CLAUDE, "operator-claude", "shared/call")[0], 405)
        self.assertEqual(self.route(CLAUDE, "operator-claude", "shared", "POST", {})[0], 405)


if __name__ == "__main__":
    unittest.main()
