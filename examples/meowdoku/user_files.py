"""Account-owned encrypted files under StatePaths.users_dir.

The existing messaging conversation is the only sharing authority. A recipient
may download a file only while a mutual contact relationship and a non-removed
message from its owner naming that file both exist. The server stores only
client-encrypted bytes, encrypted names, and recipient-wrapped keys.
"""
from __future__ import annotations

import hashlib
import base64
import json
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path

from meowdoku import schema_version

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_BLOB_BYTES = MAX_FILE_BYTES + 16
MAX_OWNER_BYTES = 64 * 1024 * 1024
MAX_OWNER_FILES = 128


class FileError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def connect(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path), timeout=10, isolation_level=None)
    db.row_factory = sqlite3.Row
    schema_version.declare(db, "user_files", 2)
    db.execute("CREATE TABLE IF NOT EXISTS user_files ("
               "id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, name TEXT NOT NULL, "
               "bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, created REAL NOT NULL, "
               "recipient_id INTEGER, envelope TEXT)")
    columns = {row[1] for row in db.execute("PRAGMA table_info(user_files)")}
    if "recipient_id" not in columns:
        db.execute("ALTER TABLE user_files ADD COLUMN recipient_id INTEGER")
    if "envelope" not in columns:
        db.execute("ALTER TABLE user_files ADD COLUMN envelope TEXT")
    db.execute("CREATE INDEX IF NOT EXISTS user_files_owner ON user_files(owner_id, created)")
    db.execute("CREATE TABLE IF NOT EXISTS user_file_keys ("
               "user_id INTEGER PRIMARY KEY, public_spki TEXT NOT NULL, encrypted_private TEXT NOT NULL, "
               "salt TEXT NOT NULL, iv TEXT NOT NULL, iterations INTEGER NOT NULL, fingerprint TEXT NOT NULL)")
    return db


def _bytes(value: str, minimum: int, maximum: int) -> bytes:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise FileError("invalid encrypted file metadata")
    try:
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except ValueError:
        raise FileError("invalid encrypted file metadata") from None
    if not minimum <= len(raw) <= maximum:
        raise FileError("invalid encrypted file metadata")
    return raw


def key_bundle(db: sqlite3.Connection, user_id: int) -> dict:
    row = db.execute("SELECT public_spki, encrypted_private, salt, iv, iterations, fingerprint "
                     "FROM user_file_keys WHERE user_id=?", (user_id,)).fetchone()
    return {"configured": bool(row), **(dict(row) if row else {})}


def register_key(db: sqlite3.Connection, user_id: int, bundle: dict) -> dict:
    if (not isinstance(bundle, dict) or set(bundle) != {"public_spki", "encrypted_private", "salt", "iv", "iterations"}
            or bundle.get("iterations") != 600000):
        raise FileError("invalid file key bundle")
    public = _bytes(bundle.get("public_spki"), 300, 800)
    _bytes(bundle.get("encrypted_private"), 1000, 4000)
    _bytes(bundle.get("salt"), 16, 16)
    _bytes(bundle.get("iv"), 12, 12)
    fingerprint = hashlib.sha256(public).hexdigest()
    try:
        db.execute("INSERT INTO user_file_keys VALUES (?,?,?,?,?,?,?)",
                   (user_id, bundle["public_spki"], bundle["encrypted_private"], bundle["salt"],
                    bundle["iv"], 600000, fingerprint))
    except sqlite3.IntegrityError:
        raise FileError("file key already exists; changing it would strand encrypted files", 409) from None
    return {"configured": True, "fingerprint": fingerprint}


def peer_key(db: sqlite3.Connection, me: int, username: str) -> dict:
    row = db.execute("SELECT u.id, u.username, k.public_spki, k.fingerprint FROM users u "
                     "JOIN user_file_keys k ON k.user_id=u.id "
                     "JOIN msg_contacts a ON a.user_id=? AND a.peer_id=u.id AND a.state='accepted' "
                     "JOIN msg_contacts b ON b.user_id=u.id AND b.peer_id=? AND b.state='accepted' "
                     "WHERE u.username=? COLLATE NOCASE", (me, me, username)).fetchone()
    if row is None:
        raise FileError("contact has not set up file encryption", 404)
    return dict(row)


def _envelope(db: sqlite3.Connection, owner_id: int, envelope: dict) -> tuple[str, int | None]:
    fields = {"v", "iv", "name_iv", "name_cipher", "owner_wrap", "owner_fp",
              "recipient", "recipient_wrap", "recipient_fp"}
    if (not isinstance(envelope, dict) or set(envelope) != fields or
            envelope.get("v") != 1 or len(json.dumps(envelope)) > 3000):
        raise FileError("invalid encrypted file metadata")
    for name, lo, hi in (("iv", 12, 12), ("name_iv", 12, 12), ("name_cipher", 17, 512),
                         ("owner_wrap", 384, 384)):
        _bytes(envelope.get(name), lo, hi)
    owner_key = key_bundle(db, owner_id)
    if not owner_key["configured"] or envelope.get("owner_fp") != owner_key["fingerprint"]:
        raise FileError("owner file key changed or is unavailable", 409)
    recipient = envelope.get("recipient")
    if recipient is None:
        if envelope.get("recipient_wrap") is not None or envelope.get("recipient_fp") is not None:
            raise FileError("invalid encrypted file metadata")
        recipient_id = None
    else:
        if not isinstance(recipient, str) or len(recipient) > 200:
            raise FileError("invalid recipient")
        peer = peer_key(db, owner_id, recipient)
        _bytes(envelope.get("recipient_wrap"), 384, 384)
        if envelope.get("recipient_fp") != peer["fingerprint"]:
            raise FileError("recipient file key changed; refresh before sending", 409)
        recipient_id = peer["id"]
    return json.dumps(envelope, separators=(",", ":")), recipient_id


def list_owned(db: sqlite3.Connection, owner_id: int) -> list[dict]:
    return [dict(row) for row in db.execute(
        "SELECT id, name, bytes, sha256, created, envelope FROM user_files "
        "WHERE owner_id=? AND envelope IS NOT NULL ORDER BY created DESC LIMIT ?",
        (owner_id, MAX_OWNER_FILES))]


def put(db: sqlite3.Connection, root: Path, owner_id: int, envelope: dict, data: bytes) -> dict:
    meta, recipient_id = _envelope(db, owner_id, envelope)
    if not isinstance(data, bytes) or not 17 <= len(data) <= MAX_BLOB_BYTES:
        raise FileError("encrypted file must be at most 8 MiB", 413)
    file_id = secrets.token_hex(16)
    digest = hashlib.sha256(data).hexdigest()
    folder = root / str(owner_id)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / file_id
    created_on_disk = False
    db.execute("BEGIN IMMEDIATE")
    try:
        count, total = db.execute("SELECT COUNT(*), COALESCE(SUM(bytes),0) FROM user_files WHERE owner_id=?",
                                  (owner_id,)).fetchone()
        if count >= MAX_OWNER_FILES or total + len(data) > MAX_OWNER_BYTES:
            raise FileError("file storage quota reached", 429)
        with open(target, "xb") as stream:
            created_on_disk = True
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        created = time.time()
        db.execute("INSERT INTO user_files VALUES (?,?,?,?,?,?,?,?)",
                   (file_id, owner_id, "Encrypted file", len(data), digest, created, recipient_id, meta))
        db.execute("COMMIT")
        return {"id": file_id, "name": "Encrypted file", "bytes": len(data), "sha256": digest,
                "created": created, "envelope": meta}
    except Exception:
        db.execute("ROLLBACK")
        if created_on_disk:
            target.unlink(missing_ok=True)
        raise


def readable(db: sqlite3.Connection, root: Path, actor_id: int, file_id: str) -> tuple[dict, bytes]:
    if not isinstance(file_id, str) or len(file_id) != 32 or any(c not in "0123456789abcdef" for c in file_id):
        raise FileError("file not found", 404)
    row = db.execute("SELECT * FROM user_files WHERE id=?", (file_id,)).fetchone()
    if row is None:
        raise FileError("file not found", 404)
    owner = row["owner_id"]
    if row["envelope"] is None:
        raise FileError("file not found", 404)
    if owner != actor_id:
        if row["recipient_id"] != actor_id:
            raise FileError("file not found", 404)
        allowed = db.execute(
            "SELECT 1 FROM msg_messages m JOIN msg_contacts a ON a.user_id=m.sender_id AND a.peer_id=m.recipient_id "
            "JOIN msg_contacts b ON b.user_id=m.recipient_id AND b.peer_id=m.sender_id "
            "WHERE m.sender_id=? AND m.recipient_id=? AND m.subject_kind='file' AND m.subject_ref=? "
            "AND m.removed_at IS NULL AND a.state='accepted' AND b.state='accepted' LIMIT 1",
            (owner, actor_id, file_id)).fetchone()
        if allowed is None:
            raise FileError("file not found", 404)
    try:
        data = (root / str(owner) / file_id).read_bytes()
    except OSError:
        raise FileError("file unavailable", 503) from None
    if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise FileError("file unavailable", 503)
    return dict(row), data
