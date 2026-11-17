"""[PRIVATE] Private values are hashed where they are captured; the raw value stays in a record on the server.

A private value is an IP address, an email address or a user agent.  Before one enters anything that is displayed (the log
ring behind ``GET /api/log``, a ticket, a printed line, a page), it is replaced by a TOKEN: ``ip~3f9a1c0b7d2e``, the kind
and the first 12 hex digits of HMAC-SHA256 under this state's key.  The same value gives the same token on one server, so
a developer can still see "the same visitor again" without learning who it is; an unkeyed hash would not do, since every
IPv4 address can be hashed in minutes.

The raw value is written once, beside its token, into ``private_record`` (``<state>/sql/private_record.db``), sealed
with the state's retained-records cipher (secure_records.py, ``<state>/secrets/records.key``) when one is given.  Only the
``private_read`` capability reads it back (``resolve``): authz.py ties that capability to the operator and admin roles in
code, so no origin tier, policy file or dev role confers it, on the live server or the dev server.  Each state directory has
its own key (``<state>/secrets/private_hash.key``), so a dev server's tokens say nothing about the live server's.

``configure`` is called once by the server's main().  Until then (tests, tools) the module hashes under a random key held in
memory and records nothing: a value is never passed through raw because the record is not set up.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

KINDS = ("ip", "email", "ua")
TOKEN_RE = re.compile(r"\b(?:ip|email|ua)~[0-9a-f]{12}\b")
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_RE = re.compile(rf"(?<![\w.]){_OCTET}(?:\.{_OCTET}){{3}}(?!\.?\w)")
# A full eight-group address, or any address written with "::".  A clock reading (12:30:45) has neither.
_H = r"[0-9A-Fa-f]{1,4}"
IPV6_RE = re.compile(rf"(?<![\w:])(?:(?:{_H}:){{7}}{_H}|(?:{_H}(?::{_H})*)?::(?:{_H}(?::{_H})*)?)"
                     rf"(?:%\w+)?(?:\.{_OCTET}(?:\.{_OCTET}){{3}})?(?![\w:])")
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")

_DDL = """
CREATE TABLE IF NOT EXISTS private_record (
    token TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    value TEXT NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    seen INTEGER NOT NULL DEFAULT 1
);
"""


class Record:
    """Tokens under one key, and (when ``db_path`` is given) the server-side record of what each token stands for."""

    def __init__(self, key: bytes, db_path: Path | None = None, cipher=None):
        if len(key) < 32:
            raise ValueError("the private hash key is too short")
        self.key, self.db_path, self.lock = key, (Path(db_path) if db_path else None), threading.Lock()
        self.cipher = cipher                    # a secure_records.RecordCipher: the stored value is sealed at rest
        self._known: set[str] = set()
        if self.db_path is not None:
            with self._connect() as conn:
                conn.executescript(_DDL)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self.db_path), timeout=10)

    def token(self, kind: str, value) -> str:
        if kind not in KINDS:
            raise ValueError(f"unknown private kind {kind!r}")
        value = str(value)
        if not value or value == "-":
            return "-"
        digest = hmac.new(self.key, f"{kind}\0{value}".encode("utf-8", "replace"), hashlib.sha256).hexdigest()[:12]
        token = f"{kind}~{digest}"
        self._note(token, kind, value)
        return token

    def _note(self, token: str, kind: str, value: str) -> None:
        if self.db_path is None:
            return
        now = time.time()
        with self.lock:
            fresh = token not in self._known
            self._known.add(token)
            try:
                with self._connect() as conn:
                    if fresh:
                        conn.execute("INSERT INTO private_record(token, kind, value, first_seen, last_seen) VALUES (?,?,?,?,?) "
                                     "ON CONFLICT(token) DO UPDATE SET last_seen = excluded.last_seen, seen = seen + 1",
                                     (token, kind, self._seal(value), now, now))
                    else:
                        conn.execute("UPDATE private_record SET last_seen = ?, seen = seen + 1 WHERE token = ?", (now, token))
            except sqlite3.Error:
                self._known.discard(token)      # the token is still returned: a record failure never lets a raw value through

    def _seal(self, value: str) -> str:
        if self.cipher is None:
            return value
        return self.cipher.seal(value.encode("utf-8")).decode("ascii").strip()

    def _open(self, stored: str) -> str:
        if self.cipher is None or not stored.startswith("enc:v1:"):
            return stored
        return self.cipher.open(stored.encode("ascii")).decode("utf-8")

    def redact(self, text) -> str:
        """``text`` with every IP and email address replaced by its token (tokens already present are left alone)."""
        text = str(text)
        text = EMAIL_RE.sub(lambda m: self.token("email", m.group(0)), text)
        text = IPV6_RE.sub(lambda m: self.token("ip", m.group(0)) if m.group(0).count(":") >= 2 else m.group(0), text)
        return IPV4_RE.sub(lambda m: self.token("ip", m.group(0)), text)

    def redact_json(self, value):
        """A JSON-shaped value with ``redact`` applied to every string, keys included."""
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, dict):
            return {self.redact(k) if isinstance(k, str) else k: self.redact_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact_json(v) for v in value]
        return value

    def resolve(self, tokens) -> dict:
        """{token: {kind, value, first_seen, last_seen, seen}} for the tokens this record holds.  The caller holds private_read."""
        wanted = sorted({t for t in tokens if isinstance(t, str) and TOKEN_RE.fullmatch(t)})
        if not wanted or self.db_path is None:
            return {}
        with self._connect() as conn:
            rows = conn.execute(f"SELECT token, kind, value, first_seen, last_seen, seen FROM private_record "
                                f"WHERE token IN ({','.join('?' * len(wanted))})", wanted).fetchall()
        return {r[0]: {"kind": r[1], "value": self._open(r[2]), "first_seen": r[3], "last_seen": r[4], "seen": r[5]} for r in rows}


def load_key(path: Path) -> bytes:
    """The state's hash key, made once (never replaced: a new key would break every token already shown)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "wb") as out:
            out.write(secrets.token_hex(32).encode("ascii"))
            out.flush()
            os.fsync(out.fileno())
    key = path.read_bytes().strip()
    if len(key) < 32:
        raise ValueError(f"{path} is not a private hash key")
    return key


RECORD = Record(secrets.token_bytes(32))


def configure(key_path: Path, db_path: Path, cipher=None) -> Record:
    global RECORD
    RECORD = Record(load_key(key_path), db_path, cipher)
    return RECORD


def token(kind: str, value) -> str:
    return RECORD.token(kind, value)


def redact(text) -> str:
    return RECORD.redact(text)


def redact_json(value):
    return RECORD.redact_json(value)
