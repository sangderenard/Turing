"""User accounts: real password-based login, server-side, SQLite-backed.

Separate from ``profiles.py``'s passwordless display-name (an arcade "who's
playing" label with no security, bound to a long-lived cookie). An account
here is a real login -- username + scrypt-hashed password -- that owns
Studio marble designs and marble photos, and that can carry forward the
browser's existing passwordless profile (its game stats/inventory) on first
signup or login, so nothing already played is lost by creating an account.

One SQLite file (``accounts.db``) under the given directory. Three tables:

``users``        -- the login identity: username, salted password hash.
``sessions``      -- bearer tokens for the account cookie (a different
                     cookie from the game-table session and the passwordless
                     player cookie -- three independent identities that
                     happen to share a browser).
``owned_assets``  -- one row per claimed digest/profile name. First claim
                     wins (``INSERT`` with a composite primary key, caught
                     on conflict), so content saved before this system
                     existed is simply unclaimed, not broken -- whoever
                     reaches it first through a logged-in session can claim
                     it, and nothing already stored needs to move or change
                     shape.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

try:
    from meowdoku import schema_version as _schema_version
except ImportError:      # loaded by path or run as a script: take the sibling file
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("meowdoku_schema_version", Path(__file__).with_name("schema_version.py"))
    _schema_version = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_schema_version)

USERNAME = re.compile(r"^[A-Za-z0-9_-]{3,32}$")
MIN_PASSWORD = 8
SESSION_SECONDS = 30 * 24 * 3600
# scrypt cost parameters: the stdlib default-ish "interactive" tuning (~tens
# of milliseconds per hash on ordinary hardware) -- real KDF stretching, not
# a bare sha256, with no third-party dependency.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P, _SCRYPT_DKLEN = 2 ** 14, 8, 1, 32


class AccountError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                             n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN)
    return f"{salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, _ = stored.split("$", 1)
    except ValueError:
        return False
    return secrets.compare_digest(_hash_password(password, bytes.fromhex(salt_hex)), stored)


class Accounts:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS owned_assets (
                asset_type TEXT NOT NULL,
                asset_key TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (asset_type, asset_key)
            );
        """)
        _schema_version.declare(self._conn, "accounts", 1)
        self._conn.commit()

    # ---- identity -----------------------------------------------------------
    @staticmethod
    def _clean_username(username: str) -> str:
        username = str(username).strip()
        if not USERNAME.match(username):
            raise AccountError("username must be 3-32 letters, digits, _ or -")
        return username

    def signup(self, username: str, password: str) -> dict:
        username = self._clean_username(username)
        if len(str(password)) < MIN_PASSWORD:
            raise AccountError(f"password must be at least {MIN_PASSWORD} characters")
        password_hash = _hash_password(str(password))
        with self.lock:
            existing = self._conn.execute(
                "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE", (username,)).fetchone()
            if existing:
                raise AccountError("that username is taken", 409)
            cursor = self._conn.execute(
                "INSERT INTO users (username, password_hash, display_name, created_at) VALUES (?, ?, ?, ?)",
                (username, password_hash, username, time.time()))
            self._conn.commit()
            return {"id": cursor.lastrowid, "username": username}

    def ensure_user(self, username: str, password: str, display_name: str | None = None) -> int:
        """Create the account if absent (same hashing as signup) and return its id; an existing account is left unchanged."""
        username = self._clean_username(username)
        with self.lock:
            row = self._conn.execute("SELECT id FROM users WHERE username = ? COLLATE NOCASE", (username,)).fetchone()
        if row:
            return row[0]
        user = self.signup(username, password)
        if display_name:
            with self.lock:
                self._conn.execute("UPDATE users SET display_name = ? WHERE id = ?", (display_name, user["id"]))
                self._conn.commit()
        return user["id"]

    def set_password(self, username: str, password: str) -> int:
        if len(str(password)) < MIN_PASSWORD:
            raise AccountError(f"password must be at least {MIN_PASSWORD} characters")
        password_hash = _hash_password(str(password))
        with self.lock:
            row = self._conn.execute("SELECT id FROM users WHERE username = ? COLLATE NOCASE", (str(username),)).fetchone()
            if row is None:
                raise AccountError("no such account", 404)
            self._conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, row[0]))
            self._conn.commit()
            return row[0]

    def login(self, username: str, password: str) -> dict:
        username = self._clean_username(username)
        with self.lock:
            row = self._conn.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ? COLLATE NOCASE",
                (username,)).fetchone()
        if row is None or not _verify_password(str(password), row[2]):
            raise AccountError("wrong username or password", 401)
        return {"id": row[0], "username": row[1]}

    def create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self.lock:
            self._conn.execute(
                "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token, user_id, now, now + SESSION_SECONDS))
            self._conn.commit()
        return token

    def user_for_token(self, token: str | None) -> dict | None:
        if not token:
            return None
        now = time.time()
        with self.lock:
            row = self._conn.execute(
                "SELECT sessions.user_id, users.username, users.display_name, sessions.expires_at "
                "FROM sessions JOIN users ON users.id = sessions.user_id WHERE sessions.token = ?",
                (token,)).fetchone()
            if row is None:
                return None
            if row[3] < now:
                self._conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
                self._conn.commit()
                return None
        return {"id": row[0], "username": row[1], "display_name": row[2]}

    def logout(self, token: str | None) -> None:
        if not token:
            return
        with self.lock:
            self._conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            self._conn.commit()

    # ---- ownership ------------------------------------------------------------
    def claim(self, user_id: int, asset_type: str, asset_key: str) -> bool:
        """First claim wins; returns True only if this call set the owner."""
        with self.lock:
            try:
                self._conn.execute(
                    "INSERT INTO owned_assets (asset_type, asset_key, user_id, created_at) VALUES (?, ?, ?, ?)",
                    (asset_type, asset_key, user_id, time.time()))
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                # A duplicate claim still begins a write transaction. Release
                # it before another connection (login guard, operator audit)
                # needs the shared accounts database.
                self._conn.rollback()
                return False

    def owner_of(self, asset_type: str, asset_key: str) -> int | None:
        with self.lock:
            row = self._conn.execute(
                "SELECT user_id FROM owned_assets WHERE asset_type = ? AND asset_key = ?",
                (asset_type, asset_key)).fetchone()
            return row[0] if row else None

    def assets_for_user(self, user_id: int, asset_type: str) -> list[str]:
        with self.lock:
            rows = self._conn.execute(
                "SELECT asset_key FROM owned_assets WHERE asset_type = ? AND user_id = ? ORDER BY created_at DESC",
                (asset_type, user_id)).fetchall()
            return [row[0] for row in rows]
