"""Direct, account-owned FLAC admission to the shared Nodus Radio catalogue.

The FLAC original is the library object. Radio uploads have no private-file or
Suno per-owner quota; each request is bounded and the shared FLAC library
has a 10 GiB ceiling. Playback creates only a temporary WAV through the
existing Nodus codec row.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import subprocess

from meowdoku import schema_version
from meowdoku.nodus_radio_control import ControlError
from meowdoku.suno_radio_assets import MAX_MEDIA_BYTES

MAX_DURATION_SECONDS = 900
LIBRARY_CAP_BYTES = 10 * 1024 * 1024 * 1024
ANALYSIS_CAP_BYTES = 1024 * 1024 * 1024
_READ_CHUNK = 1024 * 1024


@contextmanager
def _connect(db_path):
    db = sqlite3.connect(str(db_path), timeout=10)
    try:
        db.row_factory = sqlite3.Row
        db.execute("CREATE TABLE IF NOT EXISTS radio_upload_assets ("
                   "id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, title TEXT NOT NULL, "
                   "bytes INTEGER NOT NULL, analysis_bytes INTEGER NOT NULL DEFAULT 0, sha256 TEXT NOT NULL, "
                   "created REAL NOT NULL DEFAULT (unixepoch()))")
        if "analysis_bytes" not in {row[1] for row in db.execute("PRAGMA table_info(radio_upload_assets)")}:
            db.execute("ALTER TABLE radio_upload_assets ADD COLUMN analysis_bytes INTEGER NOT NULL DEFAULT 0")
        schema_version.declare(db, "radio_upload_assets", 2)
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def _asset_path(state, owner, asset_id):
    return state.users_dir / "account" / str(owner) / "assets" / "radio_upload" / asset_id / "audio.flac"


def _analysis_path(state, owner, asset_id):
    return _asset_path(state, owner, asset_id).with_name("levels.nrad")


def analysis_used(db):
    total = db.execute("SELECT COALESCE(SUM(analysis_bytes), 0) FROM radio_upload_assets").fetchone()[0]
    if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='suno_radio_assets'").fetchone():
        columns = {row[1] for row in db.execute("PRAGMA table_info(suno_radio_assets)")}
        if "analysis_bytes" in columns:
            total += db.execute("SELECT COALESCE(SUM(analysis_bytes), 0) FROM suno_radio_assets "
                                "WHERE state='ready'").fetchone()[0]
    return total


def _analyze(state, source, output):
    """Run the configured Nodus Radio rows to EOF and verify their NRAD file."""
    try:
        native = json.loads(state.radio_json.read_text(encoding="utf-8"))["jukebox"]["native"]
        codec = native["codec"]
        executable = Path(native["executable"])
        ffmpeg = Path(codec["ffmpeg_executable"])
        scope = Path(native["scope_dll"])
        fft = Path(native["fft_dll"])
        roots = codec["core_roots"]
        dll_dirs = codec.get("dll_dirs", [])
        if (any(not p.is_absolute() or not p.is_file() for p in (executable, ffmpeg, scope, fft)) or
                not isinstance(roots, list) or not roots or
                any(not isinstance(root, str) or not Path(root).is_absolute() or ";" in root for root in roots) or
                not isinstance(dll_dirs, list) or
                any(not isinstance(folder, str) or not Path(folder).is_absolute() or
                    not Path(folder).is_dir() for folder in dll_dirs)):
            raise ValueError("invalid native Radio analysis configuration")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ControlError("Nodus Radio analysis is not configured", 503) from error
    env = dict(os.environ, NODUS_RADIO_DECODER=str(ffmpeg),
               NODUS_EXEC_CORE_ROOTS=";".join(roots), NODUS_EXEC_WORK_ROOTS=str(state.users_dir),
               SDL_AUDIODRIVER="dummy")
    if dll_dirs:
        env["PATH"] = os.pathsep.join(dll_dirs) + os.pathsep + env.get("PATH", "")
    args = [str(executable), str(scope), str(fft), "127.0.0.1", "1", "0", "5", "1024",
            str(source), "0", "0", "0", "0", "log", str(output)]
    try:
        result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=360,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ControlError("Nodus Radio analysis failed", 502) from error
    match = re.search(r"^analysis_frames=(\d+) analysis_packets=(\d+) analysis_dropped=(\d+)$",
                      result.stdout, re.MULTILINE)
    if result.returncode or match is None or not output.is_file():
        raise ControlError("Nodus Radio analysis failed", 502)
    from meowdoku.nodus_radio import decode_packet
    from meowdoku.nodus_radio import HEADER
    packet_bytes = HEADER.size + 5 * 4
    size = output.stat().st_size
    if not 0 < size <= 16 * 1024 * 1024 or size % packet_bytes:
        raise ControlError("Nodus Radio analysis was incomplete", 502)
    count = 0
    last = 0
    with output.open("rb") as stream:
        while packet := stream.read(packet_bytes):
            try:
                frame = decode_packet(packet)
            except ValueError as error:
                raise ControlError("Nodus Radio analysis was invalid", 502) from error
            current = int(frame["frame"])
            if frame["sample_rate"] != 48000 or len(frame["bands"]) != 5 or current != last + 256:
                raise ControlError("Nodus Radio analysis was discontinuous", 502)
            last = current
            count += 1
    if count != int(match[2]) or int(match[3]) or last > int(match[1]) or int(match[1]) - last >= 256:
        raise ControlError("Nodus Radio analysis was incomplete", 502)
    return size


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
    if type(length) is not int or length < 42:
        raise ControlError("FLAC is too short", 400)
    if length > MAX_MEDIA_BYTES:
        raise ControlError("FLAC exceeds the Radio media limit", 413)
    asset_id = secrets.token_hex(16)
    final = _asset_path(state, owner, asset_id)
    levels = _analysis_path(state, owner, asset_id)
    final.parent.mkdir(parents=True, exist_ok=False)
    stage = final.parent / ("." + secrets.token_hex(8) + ".flac")
    analysis_stage = final.parent / ("." + secrets.token_hex(8) + ".nrad")
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
            analysis_bytes = _analyze(state, verified, analysis_stage)
        finally:
            verified.unlink(missing_ok=True)
        with _connect(db_path) as db:
            db.execute("BEGIN IMMEDIATE")  # serialize admission against the shared byte ceiling
            used = db.execute("SELECT COALESCE(SUM(bytes), 0) FROM radio_upload_assets").fetchone()[0]
            if used + length > LIBRARY_CAP_BYTES:
                raise ControlError("Radio library is full (10 GiB shared limit)", 507)
            if analysis_used(db) + analysis_bytes > ANALYSIS_CAP_BYTES:
                raise ControlError("Radio analysis storage is full (1 GiB shared limit)", 507)
            os.replace(stage, final)
            published = True
            os.replace(analysis_stage, levels)
            db.execute("INSERT INTO radio_upload_assets(id,owner_id,title,bytes,analysis_bytes,sha256) VALUES(?,?,?,?,?,?)",
                       (asset_id, owner, title.strip(), length, analysis_bytes, digest.hexdigest()))
        return {"status": "ready", "id": f"upload:{owner}:{asset_id}", "title": title.strip(),
                "bytes": length, "sha256": digest.hexdigest()}
    except BaseException:
        if published:
            final.unlink(missing_ok=True)
            levels.unlink(missing_ok=True)
        raise
    finally:
        stage.unlink(missing_ok=True)
        analysis_stage.unlink(missing_ok=True)
        if not final.exists():
            try:
                final.parent.rmdir()
            except OSError:
                pass


def owned(db_path, owner):
    with _connect(db_path) as db:
        return [dict(row) for row in db.execute(
            "SELECT id,title,bytes,sha256,created FROM radio_upload_assets "
            "WHERE owner_id=? ORDER BY created DESC", (owner,))]


def station_catalogue(state, db_path):
    with _connect(db_path) as db:
        rows = db.execute("SELECT id,owner_id,title,bytes,analysis_bytes FROM radio_upload_assets").fetchall()
    songs = {}
    for row in rows:
        path = _asset_path(state, row["owner_id"], row["id"])
        if path.is_file() and path.stat().st_size == row["bytes"]:
            key = f"upload:{row['owner_id']}:{row['id']}"
            songs[key] = {"id": key, "title": row["title"], "path": str(path), "storage": "flac",
                          "analysis": row["analysis_bytes"] > 0 and _analysis_path(state, row["owner_id"], row["id"]).is_file()}
    return songs


def read_levels(state, db_path, song_id):
    """Return the immutable, bounded analysis of an admitted shared upload."""
    parts = song_id.split(":") if isinstance(song_id, str) else []
    if len(parts) != 3 or not parts[1].isdigit() or len(parts[1]) > 20 or int(parts[1]) < 1:
        raise ControlError("Radio levels not found", 404)
    owner, asset_id = int(parts[1]), parts[2]
    if parts[0] == "upload" and re.fullmatch(r"[0-9a-f]{32}", asset_id):
        with _connect(db_path) as db:
            row = db.execute("SELECT analysis_bytes FROM radio_upload_assets WHERE id=? AND owner_id=?",
                             (asset_id, owner)).fetchone()
        path = _analysis_path(state, owner, asset_id)
    elif parts[0] == "suno":
        from meowdoku import suno_radio_assets
        if not suno_radio_assets._SONG_ID.fullmatch(asset_id):
            raise ControlError("Radio levels not found", 404)
        with suno_radio_assets._connect(db_path) as db:
            row = db.execute("SELECT storage,analysis_bytes FROM suno_radio_assets "
                             "WHERE song_id=? AND owner_id=? AND state='ready'", (asset_id, owner)).fetchone()
        path = suno_radio_assets._analysis_path(state, owner, asset_id, row[0]) if row else None
        row = (row[1],) if row else None
    else:
        raise ControlError("Radio levels not found", 404)
    if row is None or not 0 < row[0] <= 16 * 1024 * 1024:
        raise ControlError("Radio levels not found", 404)
    try:
        if path.stat().st_size != row[0]:
            raise ControlError("Radio levels unavailable", 503)
        return path.read_bytes()
    except FileNotFoundError as error:
        raise ControlError("Radio levels unavailable", 503) from error


def ensure_analysis(state, db_path, song, decoded):
    """Backfill one legacy catalogue entry from the WAV already decoded for playback."""
    key = song.get("id", "")
    parts = key.split(":")
    if len(parts) != 3 or not parts[1].isdigit():
        return False  # operator-declared tracks have no account asset to attach levels to
    owner, asset_id = int(parts[1]), parts[2]
    if parts[0] == "upload" and re.fullmatch(r"[0-9a-f]{32}", asset_id):
        with _connect(db_path) as db:
            row = db.execute("SELECT analysis_bytes FROM radio_upload_assets WHERE id=? AND owner_id=?",
                             (asset_id, owner)).fetchone()
        destination = _analysis_path(state, owner, asset_id)
        update = "UPDATE radio_upload_assets SET analysis_bytes=? WHERE id=? AND owner_id=? AND analysis_bytes=0"
        values = (asset_id, owner)
    elif parts[0] == "suno":
        from meowdoku import suno_radio_assets
        if not suno_radio_assets._SONG_ID.fullmatch(asset_id):
            return False
        with suno_radio_assets._connect(db_path) as db:
            row = db.execute("SELECT analysis_bytes FROM suno_radio_assets "
                             "WHERE song_id=? AND owner_id=? AND state='ready'", (asset_id, owner)).fetchone()
        destination = suno_radio_assets._analysis_path(state, owner, asset_id, song["storage"])
        update = ("UPDATE suno_radio_assets SET analysis_bytes=? WHERE song_id=? AND owner_id=? "
                  "AND state='ready' AND analysis_bytes=0")
        values = (asset_id, owner)
        with _connect(db_path):
            pass
    else:
        return False
    if row is None or row[0] > 0:
        return False
    if destination.exists():
        raise ControlError("Unindexed Radio levels require operator inspection", 503)
    stage = destination.with_name("." + secrets.token_hex(8) + ".nrad")
    published = False
    try:
        size = _analyze(state, decoded, stage)
        with _connect(db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            if analysis_used(db) + size > ANALYSIS_CAP_BYTES:
                raise ControlError("Radio analysis storage is full (1 GiB shared limit)", 507)
            os.replace(stage, destination)
            published = True
            changed = db.execute(update, (size, *values)).rowcount
            if changed != 1:
                raise ControlError("Radio levels changed during backfill", 409)
        return True
    except BaseException:
        if published:
            destination.unlink(missing_ok=True)
        raise
    finally:
        stage.unlink(missing_ok=True)


def route(handler, sessions, method):
    handler._classify()
    owner = handler._principal().user_id
    if owner is None:
        if method != "GET":
            handler.close_connection = True  # the refused request body is unread
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
        if method != "GET":
            handler.close_connection = True  # an upload may leave its body unread
        return handler._json({"error": str(error)}, error.status)
    except (OSError, sqlite3.Error, schema_version.SchemaVersionError):
        if method != "GET":
            handler.close_connection = True
        return handler._json({"error": "Radio upload unavailable"}, 503)
