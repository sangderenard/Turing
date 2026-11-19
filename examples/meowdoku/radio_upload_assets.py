"""Direct, account-owned FLAC admission to the shared Nodus Radio catalogue.

The FLAC original is the library object. Radio uploads have no private-file or
Suno per-owner quota; the bounded request and available library disk are the
storage limits. Playback creates only a temporary WAV through the existing
Nodus codec row.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import secrets
import sqlite3

from meowdoku import schema_version
from meowdoku.nodus_radio_control import ControlError
from meowdoku.suno_radio_assets import MAX_MEDIA_BYTES

MAX_DURATION_SECONDS = 900
_READ_CHUNK = 1024 * 1024


@contextmanager
def _connect(db_path):
    db = sqlite3.connect(str(db_path), timeout=10)
    try:
        db.row_factory = sqlite3.Row
        schema_version.declare(db, "radio_upload_assets", 1)
        db.execute("CREATE TABLE IF NOT EXISTS radio_upload_assets ("
                   "id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, title TEXT NOT NULL, "
                   "bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, "
                   "created REAL NOT NULL DEFAULT (unixepoch()))")
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def _asset_path(state, owner, asset_id):
    return state.users_dir / "account" / str(owner) / "assets" / "radio_upload" / asset_id / "audio.flac"


def _check_flac(path):
    """Check the required FLAC marker and first STREAMINFO block before admission."""
    with open(path, "rb") as stream:
        header = stream.read(42)
    if (len(header) < 42 or header[:4] != b"fLaC" or
            header[4] & 0x7f != 0 or int.from_bytes(header[5:8], "big") != 34):
        raise ControlError("Upload must be a FLAC with STREAMINFO", 415)
    packed = int.from_bytes(header[18:26], "big")
    sample_rate = packed >> 44
    channels = ((packed >> 41) & 7) + 1
    total_samples = packed & ((1 << 36) - 1)
    if (not 8000 <= sample_rate <= 384000 or not 1 <= channels <= 8 or
            not 0 < total_samples <= MAX_DURATION_SECONDS * sample_rate):
        raise ControlError("FLAC duration or stream format is unsupported", 415)


def upload(state, db_path, owner, title, stream, length):
    if type(owner) is not int or owner < 1:
        raise ControlError("Sign in to upload Radio audio", 401)
    if not isinstance(title, str) or not 1 <= len(title.strip()) <= 200 or any(ord(c) < 32 for c in title):
        raise ControlError("Give the track a title of 1 to 200 characters", 400)
    if type(length) is not int or not 42 <= length <= MAX_MEDIA_BYTES:
        raise ControlError("FLAC exceeds the Radio media limit", 413)
    asset_id = secrets.token_hex(16)
    final = _asset_path(state, owner, asset_id)
    final.parent.mkdir(parents=True, exist_ok=False)
    stage = final.parent / ("." + secrets.token_hex(8) + ".flac")
    published = False
    digest = hashlib.sha256()
    try:
        remaining = length
        with open(stage, "xb") as output:
            while remaining:
                chunk = stream.read(min(_READ_CHUNK, remaining))
                if not chunk:
                    raise ControlError("Upload ended before Content-Length", 400)
                output.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            output.flush()
            os.fsync(output.fileno())
        _check_flac(stage)
        from meowdoku.suno_radio_assets import run_codec
        verified = final.parent / ("." + secrets.token_hex(8) + ".wav")
        try:
            run_codec(state, stage, verified, "flac_decode")
            with open(verified, "rb") as wav:
                if wav.read(4) != b"RIFF":
                    raise ControlError("FLAC could not be decoded", 415)
        finally:
            verified.unlink(missing_ok=True)
        with _connect(db_path) as db:
            os.replace(stage, final)
            published = True
            db.execute("INSERT INTO radio_upload_assets(id,owner_id,title,bytes,sha256) VALUES(?,?,?,?,?)",
                       (asset_id, owner, title.strip(), length, digest.hexdigest()))
        return {"status": "ready", "id": f"upload:{owner}:{asset_id}", "title": title.strip(),
                "bytes": length, "sha256": digest.hexdigest()}
    except BaseException:
        if published:
            final.unlink(missing_ok=True)
        raise
    finally:
        stage.unlink(missing_ok=True)
        if not final.exists():
            final.parent.rmdir()


def owned(db_path, owner):
    with _connect(db_path) as db:
        return [dict(row) for row in db.execute(
            "SELECT id,title,bytes,sha256,created FROM radio_upload_assets "
            "WHERE owner_id=? ORDER BY created DESC", (owner,))]


def station_catalogue(state, db_path):
    with _connect(db_path) as db:
        rows = db.execute("SELECT id,owner_id,title,bytes FROM radio_upload_assets").fetchall()
    songs = {}
    for row in rows:
        path = _asset_path(state, row["owner_id"], row["id"])
        if path.is_file() and path.stat().st_size == row["bytes"]:
            key = f"upload:{row['owner_id']}:{row['id']}"
            songs[key] = {"id": key, "title": row["title"], "path": str(path), "storage": "flac"}
    return songs


def route(handler, sessions, method):
    handler._classify()
    owner = handler._principal().user_id
    if owner is None:
        return handler._json({"error": "Sign in to upload Radio audio"}, 401)
    try:
        if method == "GET":
            return handler._json({"assets": owned(sessions.accounts_db, owner)})
        expected = ("https://" if handler._is_secure() else "http://") + handler.headers.get("Host", "")
        if (handler.headers.get("Origin") != expected or
                handler.headers.get("Sec-Fetch-Site") == "cross-site" or
                handler.headers.get("X-Meowdoku-Radio") != "1"):
            raise ControlError("Same-origin Radio action required", 403)
        if handler.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "audio/flac":
            raise ControlError("FLAC only", 415)
        if handler.headers.get("Transfer-Encoding") or len(handler.headers.get_all("Content-Length", [])) != 1:
            raise ControlError("One Content-Length is required", 400)
        try:
            length = int(handler.headers["Content-Length"])
        except (TypeError, ValueError):
            raise ControlError("Invalid Content-Length", 400) from None
        from urllib.parse import parse_qs
        titles = parse_qs(handler.path.partition("?")[2]).get("title", [])
        if len(titles) != 1:
            raise ControlError("One track title is required", 400)
        return handler._json(upload(sessions.state, sessions.accounts_db, owner, titles[0], handler.rfile, length))
    except ControlError as error:
        return handler._json({"error": str(error)}, error.status)
    except (OSError, sqlite3.Error, schema_version.SchemaVersionError):
        return handler._json({"error": "Radio upload unavailable"}, 503)
