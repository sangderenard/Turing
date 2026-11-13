"""Account-owned, bounded files under StatePaths.users_dir.

The existing messaging conversation is the only sharing authority. A recipient
may download a file only while a mutual contact relationship and a non-removed
message from its owner naming that file both exist. File bytes are never served
as active web content.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import time
from pathlib import Path

from meowdoku import schema_version

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_OWNER_BYTES = 64 * 1024 * 1024
MAX_OWNER_FILES = 128


class FileError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def connect(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path), timeout=10, isolation_level=None)
    db.row_factory = sqlite3.Row
    schema_version.declare(db, "user_files", 1)
    db.execute("CREATE TABLE IF NOT EXISTS user_files ("
               "id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL, name TEXT NOT NULL, "
               "bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, created REAL NOT NULL)")
    db.execute("CREATE INDEX IF NOT EXISTS user_files_owner ON user_files(owner_id, created)")
    return db


def clean_name(name: str) -> str:
    if (not isinstance(name, str) or not 1 <= len(name) <= 120 or
            name in (".", "..") or any(c in name for c in "/\\:\x00") or
            any(ord(c) < 32 or ord(c) == 127 for c in name) or
            name.rstrip(" .") != name or
            name.split(".", 1)[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                                              *(f"LPT{i}" for i in range(1, 10))}):
        raise FileError("invalid filename")
    return name


def list_owned(db: sqlite3.Connection, owner_id: int) -> list[dict]:
    return [dict(row) for row in db.execute(
        "SELECT id, name, bytes, sha256, created FROM user_files WHERE owner_id=? ORDER BY created DESC LIMIT ?",
        (owner_id, MAX_OWNER_FILES))]


def put(db: sqlite3.Connection, root: Path, owner_id: int, name: str, data: bytes) -> dict:
    name = clean_name(name)
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_FILE_BYTES:
        raise FileError("file must be between 1 byte and 8 MiB", 413)
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
        db.execute("INSERT INTO user_files VALUES (?,?,?,?,?,?)",
                   (file_id, owner_id, name, len(data), digest, created))
        db.execute("COMMIT")
        return {"id": file_id, "name": name, "bytes": len(data), "sha256": digest, "created": created}
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
    if owner != actor_id:
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
