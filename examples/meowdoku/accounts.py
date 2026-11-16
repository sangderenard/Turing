"""User accounts: real password-based login, server-side, SQLite-backed.

Separate from ``profiles.py``'s passwordless display-name (an arcade "who's
playing" label with no security, bound to a long-lived cookie). An account
here is a real login -- username + scrypt-hashed password -- that owns
Studio marble designs and marble photos, and that can carry forward the
browser's existing passwordless profile (its game stats/inventory) on first
signup or login, so nothing already played is lost by creating an account.

One SQLite file (``accounts.db``) under the given directory. Five tables:

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
``alpha_agreements`` -- the terms version and acceptance time for an account
                       created through the alpha request form.
``temporary_accounts`` -- the seven-day expiry for a read-only preview; absence means the account is kept.
``preview_recovery`` -- hashed return cookie/code and frozen state for temporary previews.
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
PREVIEW_SECONDS = 7 * 24 * 3600
ALPHA_TERMS_VERSION = "alpha-2026-11-15"
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
        self._last_freeze = 0.0
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
            CREATE TABLE IF NOT EXISTS alpha_agreements (
                user_id INTEGER PRIMARY KEY,
                terms_version TEXT NOT NULL,
                accepted_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS temporary_accounts (
                user_id INTEGER PRIMARY KEY,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS preview_recovery (
                user_id INTEGER PRIMARY KEY,
                cookie_hash TEXT UNIQUE,
                code_hash TEXT UNIQUE,
                created_at REAL NOT NULL,
                frozen_at REAL
            );
        """)
        _schema_version.declare(self._conn, "accounts", 4)
        self._conn.commit()

    # ---- identity -----------------------------------------------------------
    @staticmethod
    def _clean_username(username: str) -> str:
        username = str(username).strip()
        if not USERNAME.match(username):
            raise AccountError("username must be 3-32 letters, digits, _ or -")
        return username

    def signup(self, username: str, password: str, *, alpha_terms_version: str | None = None) -> dict:
        username = self._clean_username(username)
        if alpha_terms_version is not None and alpha_terms_version != ALPHA_TERMS_VERSION:
            raise AccountError("the alpha terms have changed; reload registration and agree to the current terms")
        if len(str(password)) < MIN_PASSWORD:
            raise AccountError(f"password must be at least {MIN_PASSWORD} characters")
        password_hash = _hash_password(str(password))
        with self.lock:
            existing = self._conn.execute(
                "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE", (username,)).fetchone()
            if existing:
                raise AccountError("that username is taken", 409)
            try:
                cursor = self._conn.execute(
                    "INSERT INTO users (username, password_hash, display_name, created_at) VALUES (?, ?, ?, ?)",
                    (username, password_hash, username, time.time()))
                if alpha_terms_version is not None:
                    self._conn.execute("INSERT INTO alpha_agreements (user_id, terms_version, accepted_at) VALUES (?, ?, ?)",
                                       (cursor.lastrowid, alpha_terms_version, time.time()))
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return {"id": cursor.lastrowid, "username": username}

    def create_preview(self) -> dict:
        """Create a read-only account with a random password known only to the server."""
        now = time.time()
        expires = now + PREVIEW_SECONDS
        recovery_cookie = secrets.token_urlsafe(32)
        recovery_code = "look-" + secrets.token_urlsafe(24)
        with self.lock:
            while True:
                username = "look_" + secrets.token_hex(6)
                if not self._conn.execute(
                    "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE", (username,)).fetchone():
                    break
            try:
                cursor = self._conn.execute(
                    "INSERT INTO users (username, password_hash, display_name, created_at) VALUES (?, ?, ?, ?)",
                    (username, _hash_password(secrets.token_urlsafe(32)), username, now))
                self._conn.execute("INSERT INTO temporary_accounts (user_id, expires_at) VALUES (?, ?)",
                                   (cursor.lastrowid, expires))
                self._conn.execute("INSERT INTO preview_recovery(user_id,cookie_hash,code_hash,created_at) VALUES (?,?,?,?)",
                                   (cursor.lastrowid, hashlib.sha256(recovery_cookie.encode()).hexdigest(),
                                    hashlib.sha256(recovery_code.encode()).hexdigest(), now))
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return {"id": cursor.lastrowid, "username": username, "temporary": True, "expires_at": expires,
                "recovery_cookie": recovery_cookie, "recovery_code": recovery_code}

    def _freeze_expired_locked(self, now: float) -> int:
        """Move expired previews out of active sessions, retaining their recovery record."""
        expired = [row[0] for row in self._conn.execute(
            "SELECT t.user_id FROM temporary_accounts t LEFT JOIN preview_recovery r ON r.user_id=t.user_id "
            "WHERE t.expires_at <= ? AND (r.frozen_at IS NULL OR r.user_id IS NULL)", (now,))]
        if not expired:
            return 0
        try:
            for user_id in expired:
                self._conn.execute(
                    "INSERT OR IGNORE INTO preview_recovery(user_id,created_at) VALUES (?,?)", (user_id, now))
            for user_id in expired:
                self._conn.execute("UPDATE preview_recovery SET frozen_at=? WHERE user_id=?", (now, user_id))
                self._conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
            if self._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='oauth_mcp_tokens'").fetchone():
                for user_id in expired:
                    self._conn.execute("DELETE FROM oauth_mcp_tokens WHERE user_id=?", (user_id,))
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        return len(expired)

    def freeze_expired(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self.lock:
            self._last_freeze = now
            return self._freeze_expired_locked(now)

    def look_queue(self) -> list[dict]:
        """Every temporary preview, active or frozen, for the admin's separate queue."""
        self.freeze_expired()
        with self.lock:
            rows = self._conn.execute(
                "SELECT u.id,u.username,u.created_at,t.expires_at,r.frozen_at "
                "FROM temporary_accounts t JOIN users u ON u.id=t.user_id "
                "LEFT JOIN preview_recovery r ON r.user_id=u.id "
                "ORDER BY u.created_at DESC").fetchall()
        return [{"id": row[0], "username": row[1], "created_at": row[2],
                 "expires_at": row[3], "state": "frozen" if row[4] is not None else "active",
                 "frozen_at": row[4]} for row in rows]

    def issue_legacy_recovery(self, user_id: int) -> dict | None:
        """Give a pre-existing, still-live preview one return credential pair."""
        now = time.time()
        with self.lock:
            row = self._conn.execute("SELECT expires_at FROM temporary_accounts WHERE user_id=?", (user_id,)).fetchone()
            if row is None or row[0] <= now:
                return None
            if self._conn.execute("SELECT 1 FROM preview_recovery WHERE user_id=?", (user_id,)).fetchone():
                return None
            cookie = secrets.token_urlsafe(32)
            code = "look-" + secrets.token_urlsafe(24)
            self._conn.execute("INSERT INTO preview_recovery(user_id,cookie_hash,code_hash,created_at) VALUES (?,?,?,?)",
                               (user_id, hashlib.sha256(cookie.encode()).hexdigest(),
                                hashlib.sha256(code.encode()).hexdigest(), now))
            self._conn.commit()
            return {"recovery_cookie": cookie, "recovery_code": code}

    def recover_preview(self, cookie: str | None, code: str | None) -> dict:
        """A recovery cookie is preferred; a saved high-entropy code is the fallback."""
        if (not isinstance(cookie, str) or not 20 <= len(cookie) <= 256):
            cookie = None
        if (not isinstance(code, str) or not 20 <= len(code) <= 256):
            code = None
        if not cookie and not code:
            raise AccountError("recovery cookie or code required", 400)
        cookie_hash = hashlib.sha256(cookie.encode()).hexdigest() if cookie else None
        code_hash = hashlib.sha256(code.encode()).hexdigest() if code else None
        now = time.time()
        with self.lock:
            row = self._conn.execute(
                "SELECT u.id,u.username,t.expires_at FROM preview_recovery r "
                "JOIN temporary_accounts t ON t.user_id=r.user_id "
                "JOIN users u ON u.id=r.user_id "
                "WHERE r.cookie_hash=? OR r.code_hash=?",
                (cookie_hash, code_hash)).fetchone()
            if row is None:
                raise AccountError("recovery credential not found", 401)
            fresh_cookie = secrets.token_urlsafe(32)
            fresh_code = "look-" + secrets.token_urlsafe(24)
            expires = row[2] if row[2] > now else now + PREVIEW_SECONDS
            try:
                self._conn.execute("UPDATE temporary_accounts SET expires_at=? WHERE user_id=?", (expires, row[0]))
                self._conn.execute(
                    "UPDATE preview_recovery SET cookie_hash=?,code_hash=?,frozen_at=NULL WHERE user_id=?",
                    (hashlib.sha256(fresh_cookie.encode()).hexdigest(),
                     hashlib.sha256(fresh_code.encode()).hexdigest(), row[0]))
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return {"id": row[0], "username": row[1], "temporary": True, "expires_at": expires,
                "recovery_cookie": fresh_cookie, "recovery_code": fresh_code}

    def wipe_preview(self, user_id: int, expected_username: str) -> None:
        """Remove a temporary identity and private preview state; public posts/audit stay."""
        with self.lock:
            row = self._conn.execute(
                "SELECT u.username FROM users u JOIN temporary_accounts t ON t.user_id=u.id "
                "JOIN preview_recovery r ON r.user_id=u.id WHERE u.id=? AND r.frozen_at IS NOT NULL",
                (user_id,)).fetchone()
            if row is None:
                raise AccountError("frozen preview not found", 404)
            if row[0] != expected_username:
                raise AccountError("preview name does not match", 409)
            private_tables = (
                "oauth_mcp_codes", "oauth_mcp_tokens", "guest_post_queue",
                "guest_test_evidence", "guest_qualification", "guest_actor_kind",
                "orientation_state", "account_prefs", "account_instances",
                "account_roles", "account_permission_grants",
                "account_capability_modifiers", "account_nodus_grants",
                "account_content_invitations", "account_content_members", "account_git_identity",
                "msg_contacts", "msg_cursor", "msg_image_consent", "msg_suspended",
                "relay_addresses", "user_file_keys", "research_affirmations",
                "research_ballots", "research_eligible", "research_surveys",
                "radio_requests", "radio_turns", "radio_votes",
                "owned_assets", "sessions", "preview_recovery", "temporary_accounts",
            )
            try:
                present = {item[0] for item in self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                for table in private_tables:
                    if table in present:
                        self._conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
                self._conn.execute("DELETE FROM users WHERE id=?", (user_id,))
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def keep_preview(self, user_id: int, username: str, password: str, *, alpha_terms_version: str) -> dict:
        """Convert a live preview to a normal account with a chosen login."""
        username = self._clean_username(username)
        if alpha_terms_version != ALPHA_TERMS_VERSION:
            raise AccountError("the alpha terms have changed; reload registration and agree to the current terms")
        if len(str(password)) < MIN_PASSWORD:
            raise AccountError(f"password must be at least {MIN_PASSWORD} characters")
        password_hash = _hash_password(str(password))
        with self.lock:
            row = self._conn.execute(
                "SELECT expires_at FROM temporary_accounts WHERE user_id = ?", (user_id,)).fetchone()
            if row is None or row[0] <= time.time():
                raise AccountError("this preview has expired; start a new preview", 410)
            existing = self._conn.execute(
                "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE AND id != ?", (username, user_id)).fetchone()
            if existing:
                raise AccountError("that username is taken", 409)
            try:
                self._conn.execute(
                    "UPDATE users SET username = ?, display_name = ?, password_hash = ? WHERE id = ?",
                    (username, username, password_hash, user_id))
                self._conn.execute(
                    "INSERT INTO alpha_agreements (user_id, terms_version, accepted_at) VALUES (?, ?, ?)",
                    (user_id, alpha_terms_version, time.time()))
                self._conn.execute("DELETE FROM temporary_accounts WHERE user_id = ?", (user_id,))
                self._conn.execute("DELETE FROM preview_recovery WHERE user_id = ?", (user_id,))
                self._conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
                self._conn.commit()
            except sqlite3.IntegrityError as error:
                self._conn.rollback()
                raise AccountError("that username is taken", 409) from error
            except BaseException:
                self._conn.rollback()
                raise
        return {"id": user_id, "username": username}

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
                "SELECT users.id, users.username, users.password_hash, temporary_accounts.user_id "
                "FROM users LEFT JOIN temporary_accounts ON temporary_accounts.user_id = users.id "
                "WHERE users.username = ? COLLATE NOCASE",
                (username,)).fetchone()
        if row is None or row[3] is not None or not _verify_password(str(password), row[2]):
            raise AccountError("wrong username or password", 401)
        return {"id": row[0], "username": row[1]}

    def create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self.lock:
            preview = self._conn.execute(
                "SELECT expires_at FROM temporary_accounts WHERE user_id = ?", (user_id,)).fetchone()
            if preview is not None and preview[0] <= now:
                raise AccountError("this preview has expired; start a new preview", 410)
            expires = min(now + SESSION_SECONDS, preview[0]) if preview else now + SESSION_SECONDS
            self._conn.execute(
                "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token, user_id, now, expires))
            self._conn.commit()
        return token

    def user_for_token(self, token: str | None) -> dict | None:
        if not token:
            return None
        now = time.time()
        if now - self._last_freeze >= 60:
            self.freeze_expired(now)
        with self.lock:
            row = self._conn.execute(
                "SELECT sessions.user_id, users.username, users.display_name, sessions.expires_at, temporary_accounts.expires_at "
                "FROM sessions JOIN users ON users.id = sessions.user_id "
                "LEFT JOIN temporary_accounts ON temporary_accounts.user_id = users.id WHERE sessions.token = ?",
                (token,)).fetchone()
            if row is None:
                return None
            if row[3] <= now or (row[4] is not None and row[4] <= now):
                self._conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
                self._conn.commit()
                return None
        return {"id": row[0], "username": row[1], "display_name": row[2],
                "temporary": row[4] is not None, "expires_at": row[4]}

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
