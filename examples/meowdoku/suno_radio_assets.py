"""Account-owned Suno audio admitted to the existing Nodus Radio jukebox.

Only the signed-in account's MCP connection can obtain a download URL. A song
is offered to the shared station only by the explicit import action. Durable
media stays in that account's user asset area; playback WAVs are temporary.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import subprocess
from urllib.parse import urlsplit

from meowdoku import schema_version
from meowdoku.nodus_radio_control import ControlError

_SONG_ID = re.compile(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\Z")
MAX_MEDIA_BYTES = 256 * 1024 * 1024
MAX_OWNER_BYTES = 2 * 1024 * 1024 * 1024


def _payload(result):
    """Suno MCP returns JSON in a text content block; preserve error fields."""
    if not isinstance(result, dict) or result.get("isError") is True:
        raise ControlError("Suno refused the media request", 502)
    content = result.get("content")
    if not isinstance(content, list):
        raise ControlError("Suno returned no media record", 502)
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            try:
                value = json.loads(part["text"])
            except (KeyError, TypeError, ValueError):
                continue
            if isinstance(value, dict):
                return value
    raise ControlError("Suno returned no media record", 502)


def _asset_path(state, owner, song_id, storage):
    suffix = "mp3" if storage == "mp3" else "flac"
    return state.users_dir / "account" / str(owner) / "assets" / "suno_radio" / song_id / ("audio." + suffix)


@contextmanager
def _connect(db_path):
    db = sqlite3.connect(str(db_path), timeout=10)
    try:
        db.row_factory = sqlite3.Row
        schema_version.declare(db, "suno_radio_assets", 1)
        db.execute("CREATE TABLE IF NOT EXISTS suno_radio_assets ("
                   "owner_id INTEGER NOT NULL, song_id TEXT NOT NULL, title TEXT NOT NULL, "
                   "storage TEXT NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, "
                   "state TEXT NOT NULL, created REAL NOT NULL DEFAULT (unixepoch()), "
                   "PRIMARY KEY(owner_id,song_id))")
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def owned(state, db_path, owner):
    with _connect(db_path) as db:
        return [dict(row) for row in db.execute(
            "SELECT song_id,title,storage,bytes,sha256,created FROM suno_radio_assets "
            "WHERE owner_id=? AND state='ready' ORDER BY created DESC", (owner,))]


def station_catalogue(state, db_path):
    """Resolve paths from the SQL owner and UUID; never accept an HTTP path."""
    with _connect(db_path) as db:
        rows = db.execute("SELECT owner_id,song_id,title,storage,bytes FROM suno_radio_assets "
                          "WHERE state='ready'").fetchall()
    songs = {}
    for row in rows:
        path = _asset_path(state, row["owner_id"], row["song_id"], row["storage"])
        if path.is_file() and path.stat().st_size == row["bytes"]:
            key = f"suno:{row['owner_id']}:{row['song_id']}"
            songs[key] = {"id": key, "title": row["title"], "path": str(path),
                          "storage": row["storage"]}
    return songs


def _download_url(value):
    url = value.get("url")
    if not isinstance(url, str) or len(url) > 4096:
        raise ControlError("Suno returned no downloadable audio URL", 502)
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if (parsed.scheme != "https" or parsed.username or parsed.password or
            parsed.port not in (None, 443) or not parsed.path or
            not (host == "suno.com" or host.endswith(".suno.com") or
                 host == "suno.ai" or host.endswith(".suno.ai"))):
        raise ControlError("Suno media destination is outside the verified provider", 502)
    return url


async def _fetch_to_file(url, target):
    import httpx
    from speaktome_api.remote_mcp import PublicHTTPTransport
    digest = hashlib.sha256()
    size = 0
    async with httpx.AsyncClient(transport=PublicHTTPTransport(), follow_redirects=False,
                                 timeout=httpx.Timeout(120, connect=15)) as client:
        async with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise ControlError("Suno media transfer was refused", 502)
            if int(response.headers.get("content-length", "0")) > MAX_MEDIA_BYTES:
                raise ControlError("Suno media exceeds the server asset limit", 413)
            with open(target, "xb") as stream:
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_MEDIA_BYTES:
                        raise ControlError("Suno media exceeds the server asset limit", 413)
                    stream.write(chunk)
                    digest.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
    if size < 1024:
        raise ControlError("Suno media was empty", 502)
    return size, digest.hexdigest()


def run_codec(state, source, target, mode):
    """The host-declared Nodus exec span row performs every media conversion."""
    config = json.loads(state.radio_json.read_text(encoding="utf-8"))
    native = config["jukebox"]["native"]
    codec = native["codec"]
    executable, ffmpeg = Path(codec["executable"]), Path(codec["ffmpeg_executable"])
    roots = codec["core_roots"]
    dll_dirs = codec.get("dll_dirs", [])
    if (not executable.is_absolute() or not executable.is_file() or
            not ffmpeg.is_absolute() or not ffmpeg.is_file() or
            not isinstance(roots, list) or not roots or
            any(not isinstance(root, str) or not Path(root).is_absolute() or ";" in root for root in roots) or
            not isinstance(dll_dirs, list) or
            any(not isinstance(folder, str) or not Path(folder).is_absolute() or
                not Path(folder).is_dir() for folder in dll_dirs)):
        raise ControlError("Nodus audio codec is not configured", 503)
    args = {
        "mp3_decode": ["-f", "mp3", "-i", "pipe:0", "-vn", "-t", "900", "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"],
        "flac_decode": ["-f", "flac", "-i", "pipe:0", "-vn", "-t", "900", "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"],
        "flac_encode": ["-f", "wav", "-i", "pipe:0", "-vn", "-c:a", "flac", "-compression_level", "12", "-f", "flac", "pipe:1"],
    }.get(mode)
    if args is None:
        raise ValueError("invalid Nodus audio codec mode")
    source, target = Path(source), Path(target)
    if (source.parent != target.parent or not source.is_file() or target.exists() or
            state.users_dir.resolve() not in source.resolve().parents):
        raise ControlError("Audio codec paths are not account-owned", 400)
    row_id = "exec.radio." + mode
    rows = source.parent / ("." + secrets.token_hex(8) + ".rows")
    lines = [f"row {row_id}", "name Radio audio codec", f"program {ffmpeg}",
             f"directory {source.parent}", "most 1"]
    lines.extend(f"argument {arg}" for arg in ["-hide_banner", "-loglevel", "error", "-nostdin", *args[:-1],
                                                   "-fs", str(MAX_MEDIA_BYTES), args[-1]])
    rows.write_text("\n".join(lines) + "\n", encoding="utf-8")
    env = dict(os.environ, NODUS_EXEC_CORE_ROOTS=";".join(roots),
               NODUS_EXEC_WORK_ROOTS=str(state.users_dir))
    if dll_dirs:
        env["PATH"] = os.pathsep.join(dll_dirs) + os.pathsep + env.get("PATH", "")
    try:
        result = subprocess.run([str(executable), str(rows), row_id, str(source), str(target)],
                                env=env, capture_output=True, timeout=360,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode or not target.is_file():
            target.unlink(missing_ok=True)
            raise ControlError("Nodus audio codec failed", 502)
    finally:
        rows.unlink(missing_ok=True)


def import_song(state, db_path, owner, song_id, storage, confirm_credit, service, *, fetch=_fetch_to_file):
    if (type(owner) is not int or owner < 1 or not isinstance(song_id, str) or
            not _SONG_ID.fullmatch(song_id) or storage not in ("mp3", "flac") or
            type(confirm_credit) is not bool):
        raise ControlError("Choose one Suno song and a supported storage format", 400)
    with _connect(db_path) as db:
        existing = db.execute("SELECT state FROM suno_radio_assets WHERE owner_id=? AND song_id=?",
                              (owner, song_id)).fetchone()
        if existing:
            if existing["state"] == "ready":
                return {"status": "ready", "song_id": song_id, "already_imported": True}
            raise ControlError("This song is already being imported", 409)
        db.execute("INSERT INTO suno_radio_assets(owner_id,song_id,title,storage,bytes,sha256,state) "
                   "VALUES(?,?,?, ?,0,'','pending')", (owner, song_id, song_id, storage))
        db.commit()
    folder = _asset_path(state, owner, song_id, storage).parent
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / ("." + secrets.token_hex(8) + (".mp3" if storage == "mp3" else ".wav"))
    final = _asset_path(state, owner, song_id, storage)
    encoded = folder / ("." + secrets.token_hex(8) + ".flac")
    try:
        detail = _payload(service.call(owner, "suno", "audio-retrieve", {"id": song_id}))
        if detail.get("id") != song_id or detail.get("status") != "complete":
            raise ControlError("Suno song is not a completed clip", 409)
        title = detail.get("title")
        if not isinstance(title, str) or not 1 <= len(title) <= 200:
            title = song_id
        response = _payload(service.call(owner, "suno", "audio-download",
                                         {"id": song_id, "format": "mp3" if storage == "mp3" else "wav",
                                          "confirm": confirm_credit}))
        if response.get("status") == "requires_credit":
            return {"status": "requires_credit", "song_id": song_id,
                    "credits_remaining": response.get("credits_remaining")}
        if response.get("status") == "processing":
            return {"status": "processing", "song_id": song_id}
        url = _download_url(response)
        size, digest = asyncio.run(fetch(url, raw))
        with open(raw, "rb") as stream:
            signature = stream.read(12)
        if storage == "mp3" and not (signature.startswith(b"ID3") or signature[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
            raise ControlError("Suno media is not MP3", 502)
        if storage == "flac":
            if signature[:4] != b"RIFF" or signature[8:12] != b"WAVE":
                raise ControlError("Suno media is not WAV", 502)
            run_codec(state, raw, encoded, "flac_encode")
            raw.unlink()
            raw = encoded
            size = raw.stat().st_size
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        with _connect(db_path) as db:
            total = db.execute("SELECT COALESCE(SUM(bytes),0) FROM suno_radio_assets "
                               "WHERE owner_id=? AND state='ready'", (owner,)).fetchone()[0]
            if total + size > MAX_OWNER_BYTES:
                raise ControlError("Radio asset storage quota reached", 413)
            os.replace(raw, final)
            db.execute("UPDATE suno_radio_assets SET title=?,bytes=?,sha256=?,state='ready' "
                       "WHERE owner_id=? AND song_id=? AND state='pending'",
                       (title, size, digest, owner, song_id))
            db.commit()
        return {"status": "ready", "song_id": song_id, "title": title,
                "storage": storage, "bytes": size}
    finally:
        raw.unlink(missing_ok=True)
        encoded.unlink(missing_ok=True)
        with _connect(db_path) as db:
            db.execute("DELETE FROM suno_radio_assets WHERE owner_id=? AND song_id=? AND state='pending'",
                       (owner, song_id))
            db.commit()


def route(handler, sessions, path, method):
    handler._classify()
    owner = handler._principal().user_id
    if owner is None:
        return handler._json({"error": "Sign in to import Suno audio"}, 401)
    try:
        if path == "/api/radio/suno/assets" and method == "GET":
            return handler._json({"assets": owned(sessions.state, sessions.accounts_db, owner)})
        if path != "/api/radio/suno/import" or method != "POST":
            return handler._json({"error": "Unknown Radio asset action"}, 404)
        expected = ("https://" if handler._is_secure() else "http://") + handler.headers.get("Host", "")
        if (handler.headers.get("Origin") != expected or
                handler.headers.get("Sec-Fetch-Site") == "cross-site" or
                handler.headers.get("X-Meowdoku-Radio") != "1"):
            raise ControlError("Same-origin Radio action required", 403)
        if handler.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise ControlError("JSON only", 415)
        body = handler._body_json(2048)
        if body is None:
            return
        if not isinstance(body, dict) or set(body) != {"song_id", "storage", "confirm_credit"}:
            raise ControlError("Invalid Suno import request", 400)
        from meowdoku.remote_connections import service_for
        result = import_song(sessions.state, sessions.accounts_db, owner, body["song_id"],
                             body["storage"], body["confirm_credit"], service_for(sessions))
        return handler._json(result)
    except ControlError as error:
        return handler._json({"error": str(error)}, error.status)
    except (OSError, sqlite3.Error, ValueError, KeyError, subprocess.TimeoutExpired):
        return handler._json({"error": "Radio asset import unavailable"}, 503)
