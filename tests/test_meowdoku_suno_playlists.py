"""Suno playlists onto the jukebox (suno_playlists.py) with a fake connection service, a fake jukebox store and a stubbed
import: no provider and no download is reached."""
from contextlib import contextmanager
import json
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku import connection_shares, suno_playlists as mod  # noqa: E402
from meowdoku.nodus_radio_control import ControlError  # noqa: E402

OWNER, CLAUDE, DOT = 3, 17, 19
PLAYLIST = "11111111-2222-3333-4444-555555555555"
MINE = ["aaaaaaaa-0000-0000-0000-00000000000%d" % i for i in range(3)]
THEIRS = "bbbbbbbb-0000-0000-0000-000000000000"


def text(value):
    return {"content": [{"type": "text", "text": json.dumps(value)}]}


class Service:
    def __init__(self, connected=(OWNER,)):
        self.connected, self.calls = set(connected), []
    def status(self, owner, provider): return {"configured": owner in self.connected}
    def call(self, owner, provider, name, arguments):
        self.calls.append((owner, name, arguments))
        if name == "account-me":
            return text({"handle": "owner-handle"})
        if name == "playlists-list":
            return text({"playlists": [{"id": PLAYLIST, "name": "Road", "clip_count": 4, "visibility": "private"}],
                         "next_cursor": None})
        if name == "playlists-list-clips":
            items = [{"id": i, "title": "song %d" % n, "status": "complete", "handle": "owner-handle"} for n, i in enumerate(MINE)]
            items.insert(1, {"id": THEIRS, "title": "theirs", "status": "complete", "handle": "someone-else"})
            return text({"items": items})
        raise AssertionError(name)


class Store:
    def __init__(self):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.execute("CREATE TABLE radio_requests (id INTEGER PRIMARY KEY, user_id INTEGER, song_id TEXT, created REAL)")
    @contextmanager
    def connect(self):
        yield self.db
    def request(self, user, key):
        self.db.execute("INSERT INTO radio_requests (user_id, song_id, created) VALUES (?, ?, 0)", (user, key))
    def requested(self, user):
        return [r[0] for r in self.db.execute("SELECT song_id FROM radio_requests WHERE user_id = ? ORDER BY id", (user,))]


def stub_import(outcomes):
    def fake(state, db, owner, song_id, storage, confirm, service):
        assert storage == "mp3"
        status = outcomes.get(song_id, "ready")
        if status == "requires_credit" and confirm:
            status = "ready"
        return {"status": status, "song_id": song_id, "credits_remaining": 7}
    return fake


class Handler:
    def __init__(self, user_id, username, path, body=None, headers=None):
        self.path, self.body, self.output = path, body, None
        self.headers = {"Host": "example.test", "Content-Type": "application/json", **(headers or {})}
        self.user = types.SimpleNamespace(user_id=user_id, username=username)
    def _classify(self): pass
    def _is_secure(self): return True
    def _principal(self): return self.user
    def _body_json(self, limit): return self.body
    def _json(self, value, status=200): self.output = (status, value)


class PlaylistTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.db_path = Path(self.folder.name) / "accounts.db"
        db = sqlite3.connect(self.db_path)
        db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT NOT NULL)")
        db.execute("CREATE TABLE account_roles (user_id INTEGER PRIMARY KEY, role TEXT NOT NULL)")
        for uid, name, role in ((OWNER, "admin", "admin"), (CLAUDE, "operator-claude", "operator"), (DOT, "dot", "member")):
            db.execute("INSERT INTO users VALUES (?, ?)", (uid, name))
            db.execute("INSERT INTO account_roles VALUES (?, ?)", (uid, role))
        db.commit()
        connection_shares.set_share(db, OWNER, "suno", "operator-claude", True)
        db.close()
        self.service, self.store = Service(), Store()
        self.sessions = types.SimpleNamespace(accounts_db=self.db_path, state=types.SimpleNamespace())
        mod._jobs.clear()
    def tearDown(self):
        self.folder.cleanup()
    def conn(self, user_id=OWNER, name="admin"):
        return mod.Connection(self.service, self.db_path, user_id, name)
    def job(self, conn, user_id, confirm=False, outcomes=None):
        job = {"state": "running", "done": 0, "total": None, "needs_credit": 0, "credits_remaining": None, "songs": [],
               "ended": None, "error": None}
        with patch.object(mod, "import_song", stub_import(outcomes or {})):
            mod.run(self.sessions.state, self.db_path, self.store, conn, user_id, PLAYLIST, confirm, job)
        return job

    def test_the_owner_uses_its_own_connection_and_a_ticked_operator_the_shared_one(self):
        own, shared = self.conn(), self.conn(CLAUDE, "operator-claude")
        self.assertEqual((own.owner_id, own.shared), (OWNER, False))
        self.assertEqual((shared.owner_id, shared.owner, shared.shared), (OWNER, "admin", True))
        mod.playlists(shared)
        db = sqlite3.connect(self.db_path)
        audited = db.execute("SELECT caller, tool, ok FROM connection_share_calls").fetchall()
        db.close()
        self.assertEqual(audited, [("operator-claude", "playlists-list", 1)])
        with self.assertRaises(ControlError):
            self.conn(DOT, "dot")                         # neither connected nor shared with

    def test_listing_marks_other_creators_songs(self):
        self.assertEqual(mod.playlists(self.conn())["playlists"][0]["songs"], 4)
        songs = mod.songs(self.conn(), PLAYLIST)
        self.assertEqual([s["importable"] for s in songs], [True, False, True, True])
        with self.assertRaises(ControlError):
            mod.songs(self.conn(), "../etc")

    def test_a_run_requests_the_owners_songs_in_order_and_reports_the_rest(self):
        job = self.job(self.conn(), OWNER, outcomes={MINE[1]: "requires_credit"})
        self.assertEqual(job["state"], "done")
        self.assertEqual(self.store.requested(OWNER), [f"suno:{OWNER}:{MINE[0]}", f"suno:{OWNER}:{MINE[2]}"])
        self.assertEqual([s["outcome"].split(":")[0] for s in job["songs"]],
                         ["requested", "skipped", "needs a Suno download credit", "requested"])
        self.assertEqual((job["needs_credit"], job["credits_remaining"]), (1, 7))
        again = self.job(self.conn(), OWNER, confirm=True, outcomes={MINE[1]: "requires_credit"})
        self.assertEqual(self.store.requested(OWNER), [f"suno:{OWNER}:{s}" for s in (MINE[0], MINE[2], MINE[1])])
        self.assertEqual([s["outcome"] for s in again["songs"]][0], "already on your request list")

    def test_a_shared_run_files_the_songs_under_the_owner_and_requests_them_for_the_operator(self):
        self.job(self.conn(CLAUDE, "operator-claude"), CLAUDE)
        self.assertEqual(self.store.requested(CLAUDE), [f"suno:{OWNER}:{s}" for s in MINE])

    def test_route_lists_and_guards_the_queue(self):
        def route(user_id, name, path, method="GET", body=None, headers=None):
            h = Handler(user_id, name, path, body, headers)
            mod.route(h, self.sessions, path, method, self.store, lambda sessions: self.service)
            return h.output
        status, answer = route(CLAUDE, "operator-claude", "/api/radio/suno/playlists")
        self.assertEqual((status, answer["owner"], answer["shared"]), (200, "admin", True))
        self.assertEqual(route(DOT, "dot", "/api/radio/suno/playlists")[0], 403)
        self.assertEqual(route(None, None, "/api/radio/suno/playlists")[0], 401)
        queue = "/api/radio/suno/playlists/" + PLAYLIST + "/queue"
        self.assertEqual(route(OWNER, "admin", queue, "POST", {"confirm_credit": False})[0], 403)   # no radio header
        self.assertEqual(route(OWNER, "admin", queue, "GET")[0], 405)
        self.assertEqual(route(OWNER, "admin", "/api/radio/suno/playlists/x/y/z")[0], 404)
        self.assertEqual(route(OWNER, "admin", "/api/radio/suno/playlist-job")[1], {"job": None})


if __name__ == "__main__":
    unittest.main()
