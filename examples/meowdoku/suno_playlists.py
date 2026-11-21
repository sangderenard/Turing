"""Suno playlists on the server jukebox: list a Suno connection's playlists, and put one on the jukebox.

The connection is the signed-in account's own, or else one an owner shares with it (connection_shares.py; those calls are
audited there). Putting a playlist on the jukebox imports each song in it as MP3 through the existing import
(suno_radio_assets.import_song: the file belongs to the connection owner's account and joins the station catalogue) and
then requests it, in the playlist's order, for the account that asked. Suno's own rules decide each download:
audio-download serves only the owner's own songs, so another creator's song is skipped; a song that needs a download
credit is skipped unless the account confirmed spending credits. A song already on that account's request list is not
requested twice, so a second run (say, with credits confirmed) adds only what the first could not.

One job at a time per account. It runs in a thread of the web server and its progress is read back; a restart ends it
(what was imported and requested so far stays).

    GET  /api/radio/suno/playlists[?cursor=C]          the connection's playlists, a page at a time
    GET  /api/radio/suno/playlists/<id>                its songs, in order, each marked whether it can be imported
    POST /api/radio/suno/playlists/<id>/queue  {confirm_credit}   import and request them: starts the job
    GET  /api/radio/suno/playlist-job                 the account's job: progress and each song's outcome
"""
from __future__ import annotations

from contextlib import closing
import re
import sqlite3
import threading
import time
from urllib.parse import parse_qs, urlsplit

from meowdoku import connection_shares
from meowdoku.nodus_radio_control import ControlError
from meowdoku.suno_radio_assets import _payload, import_song

PROVIDER = "suno"
MAX_SONGS = 100              # the jukebox keeps at most 100 requests per account
MAX_PAGES = 20
_ID = re.compile(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\Z")
_jobs: dict = {}
_lock = threading.Lock()


class Connection:
    """The Suno connection an account uses, with the connection service's call(owner, provider, name, arguments)."""

    def __init__(self, service, db_path, user_id, username):
        self.service, self.db_path, self.user_id, self.username = service, str(db_path), user_id, username
        if (service.status(user_id, PROVIDER) or {}).get("configured"):
            self.owner_id, self.owner, self.shared = user_id, username, False
            return
        with closing(sqlite3.connect(self.db_path, timeout=30)) as db:
            owners = connection_shares.shared_with(db, user_id, PROVIDER)
        if not owners:
            raise ControlError("No Suno connection: connect one on Connected services, or ask an owner to share theirs "
                               "with you", 403)
        self.owner_id, self.owner, self.shared = owners[0]["owner_id"], owners[0]["owner"], True

    def call(self, owner, provider, name, arguments):
        if owner != self.owner_id or provider != PROVIDER:
            raise PermissionError("not this account's Suno connection")
        if not self.shared:
            return self.service.call(owner, provider, name, arguments)
        with closing(sqlite3.connect(self.db_path, timeout=30)) as db:
            return connection_shares.shared_call(db, owner_id=owner, provider=provider, caller_id=self.user_id,
                                                 caller=self.username, name=name, arguments=arguments,
                                                 call=self.service.call)

    def tool(self, name, arguments):
        return _payload(self.call(self.owner_id, PROVIDER, name, arguments))


def playlists(conn, cursor=None) -> dict:
    arguments = {"limit": 50}
    if cursor:
        arguments["cursor"] = str(cursor)
    answer = conn.tool("playlists-list", arguments)
    rows = [{"id": p["id"], "name": p.get("name") or p["id"], "songs": p.get("clip_count"),
             "visibility": p.get("visibility")}
            for p in answer.get("playlists", []) if isinstance(p, dict) and isinstance(p.get("id"), str)]
    return {"owner": conn.owner, "shared": conn.shared, "playlists": rows, "next_cursor": answer.get("next_cursor")}


def songs(conn, playlist_id) -> list:
    """The playlist's songs in order (at most MAX_SONGS), each with whether this connection may import it."""
    if not isinstance(playlist_id, str) or not _ID.fullmatch(playlist_id):
        raise ControlError("Choose a playlist from the list", 400)
    handle = conn.tool("account-me", {}).get("handle")
    out, cursor = [], None
    for _ in range(MAX_PAGES):
        arguments = {"id": playlist_id, "limit": 50}
        if cursor:
            arguments["cursor"] = cursor
        answer = conn.tool("playlists-list-clips", arguments)
        for item in answer.get("items", []):
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not _ID.fullmatch(item["id"]):
                continue
            own, done = item.get("handle") == handle, item.get("status") == "complete"
            out.append({"id": item["id"], "title": item.get("title") or item["id"], "status": item.get("status"),
                        "importable": own and done,
                        "why": None if own and done else ("another creator's song: Suno downloads only the owner's own"
                                                         if not own else "not finished on Suno")})
        cursor = answer.get("next_cursor")
        if not cursor or len(out) >= MAX_SONGS:
            break
    return out[:MAX_SONGS]


def _requested(store, user_id, key) -> bool:
    with store.connect() as db:
        return db.execute("SELECT 1 FROM radio_requests WHERE user_id = ? AND song_id = ?", (user_id, key)).fetchone() \
            is not None


def run(state, accounts_db, store, conn, user_id, playlist_id, confirm_credit, job) -> None:
    """Import and request each song of the playlist, writing every outcome into ``job`` as it goes."""
    try:
        listing = songs(conn, playlist_id)
        job["total"] = len(listing)
        for song in listing:
            entry = {"id": song["id"], "title": song["title"]}
            if not song["importable"]:
                entry["outcome"] = "skipped: " + song["why"]
            else:
                try:
                    result = import_song(state, accounts_db, conn.owner_id, song["id"], "mp3", confirm_credit, conn)
                    if result["status"] == "ready":
                        key = f"suno:{conn.owner_id}:{song['id']}"
                        if _requested(store, user_id, key):
                            entry["outcome"] = "already on your request list"
                        else:
                            store.request(user_id, key)
                            entry["outcome"] = "requested"
                    elif result["status"] == "requires_credit":
                        entry["outcome"] = "needs a Suno download credit"
                        job["needs_credit"] += 1
                        job["credits_remaining"] = result.get("credits_remaining")
                    else:
                        entry["outcome"] = "Suno is still preparing it; run the playlist again shortly"
                except ControlError as error:
                    entry["outcome"] = "failed: " + str(error)
                except (OSError, sqlite3.Error, ValueError, KeyError):
                    entry["outcome"] = "failed: the import could not finish"
            job["songs"].append(entry)
            job["done"] += 1
        job["state"] = "done"
    except ControlError as error:
        job.update(state="failed", error=str(error))
    except Exception:
        job.update(state="failed", error="the playlist could not be read from Suno")
    finally:
        job["ended"] = time.time()


def start(state, accounts_db, store, conn, user_id, playlist_id, confirm_credit) -> dict:
    if type(confirm_credit) is not bool:
        raise ControlError("confirm_credit must be true or false", 400)
    if not isinstance(playlist_id, str) or not _ID.fullmatch(playlist_id):
        raise ControlError("Choose a playlist from the list", 400)
    with _lock:
        current = _jobs.get(user_id)
        if current is not None and current["state"] == "running":
            raise ControlError("A playlist is already being put on the jukebox for you", 409)
        job = {"state": "running", "playlist": playlist_id, "owner": conn.owner, "confirm_credit": confirm_credit,
               "started": time.time(), "ended": None, "total": None, "done": 0, "needs_credit": 0,
               "credits_remaining": None, "songs": [], "error": None}
        _jobs[user_id] = job
    threading.Thread(target=run, args=(state, accounts_db, store, conn, user_id, playlist_id, confirm_credit, job),
                     daemon=True, name="suno-playlist").start()
    return snapshot(user_id)


def snapshot(user_id) -> dict:
    with _lock:
        job = _jobs.get(user_id)
        return {"job": None if job is None else {**job, "songs": list(job["songs"])}}


def route(handler, sessions, path, method, store, service_for):
    """``store`` is the server's jukebox (nodus_radio_jukebox.Jukebox); ``service_for(sessions)`` its connection service."""
    handler._classify()
    principal = handler._principal()
    user_id = principal.user_id
    if user_id is None:
        return handler._json({"error": "Sign in to use Suno playlists"}, 401)
    try:
        if path == "/api/radio/suno/playlist-job" and method == "GET":
            return handler._json(snapshot(user_id))
        rest = path.removeprefix("/api/radio/suno/playlists").strip("/")
        parts = rest.split("/") if rest else []
        if not path.startswith("/api/radio/suno/playlists") or len(parts) > 2 or (len(parts) == 2 and parts[1] != "queue"):
            return handler._json({"error": "Unknown playlist action"}, 404)
        if method != ("POST" if len(parts) == 2 else "GET"):
            return handler._json({"error": "method not allowed"}, 405)
        body = None
        if method == "POST":
            expected = ("https://" if handler._is_secure() else "http://") + handler.headers.get("Host", "")
            if (handler.headers.get("Origin") != expected or handler.headers.get("Sec-Fetch-Site") == "cross-site"
                    or handler.headers.get("X-Meowdoku-Radio") != "1"):
                raise ControlError("Same-origin Radio action required", 403)
            if handler.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                raise ControlError("JSON only", 415)
            body = handler._body_json(2048)
            if body is None:
                return
            if not isinstance(body, dict) or set(body) != {"confirm_credit"}:
                raise ControlError("Send confirm_credit (true or false)", 400)
        username = getattr(principal, "username", None) or str(user_id)
        conn = Connection(service_for(sessions), sessions.accounts_db, user_id, username)
        if not parts:
            cursor = parse_qs(urlsplit(handler.path).query).get("cursor", [None])[0]
            return handler._json(playlists(conn, cursor))
        if len(parts) == 1:
            return handler._json({"playlist": parts[0], "owner": conn.owner, "songs": songs(conn, parts[0])})
        return handler._json(start(sessions.state, sessions.accounts_db, store, conn, user_id, parts[0],
                                   body["confirm_credit"]))
    except ControlError as error:
        return handler._json({"error": str(error)}, error.status)
    except connection_shares.ShareError as error:
        return handler._json({"error": str(error)}, error.status)
    except PermissionError:
        return handler._json({"error": "That Suno connection is not yours to use"}, 403)
    except sqlite3.Error:
        return handler._json({"error": "Playlist storage unavailable"}, 503)
    except Exception:
        return handler._json({"error": "Suno could not answer"}, 502)
