"""Brute-force protection for POST /api/account/login, backed by accounts.db (table ``login_attempts``).

check() refuses (returning retry_after seconds) when, within the windows below, there are too many recent FAILURES for the
(client, username) pair, for the client alone, or for the username alone.  record() stores each outcome and prunes rows older
than 24 h.  The client is the proxy-trust client address the server classified, never a raw header.
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meowdoku import schema_version as _schema_version  # noqa: E402

PAIR_LIMIT, PAIR_WINDOW = 5, 600.0
CLIENT_LIMIT, CLIENT_WINDOW = 20, 600.0
USER_LIMIT, USER_WINDOW = 50, 3600.0
KEEP_SECONDS = 24 * 3600.0

DDL = ("CREATE TABLE IF NOT EXISTS login_attempts (ts REAL NOT NULL, client TEXT NOT NULL, username TEXT NOT NULL, ok INTEGER NOT NULL)",
       "CREATE INDEX IF NOT EXISTS login_attempts_ts ON login_attempts (ts)")


def connect(db_path) -> sqlite3.Connection:
    """Own connection to accounts.db (autocommit).  Declares schema module ``login_guard`` 1."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(db_path), timeout=10, isolation_level=None, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    try:
        _schema_version.declare(db, "login_guard", 1)
    except _schema_version.SchemaVersionError:
        db.close()
        raise
    for ddl in DDL:
        db.execute(ddl)
    return db


def _norm(username) -> str:
    return str(username).strip().lower()


def _retry(db, where: str, args: tuple, limit: int, window: float, now: float) -> float:
    rows = [r[0] for r in db.execute(
        f"SELECT ts FROM login_attempts WHERE ok = 0 AND ts > ? AND {where} ORDER BY ts DESC LIMIT ?",
        (now - window, *args, limit))]
    if len(rows) < limit:
        return 0.0
    return max(0.0, rows[-1] + window - now)          # the failure that must age out before the count drops under the limit


def check(db, client_ip: str, username: str, now: float | None = None) -> float:
    """0 when the attempt may proceed, otherwise the whole seconds (>= 1) to wait."""
    now = time.time() if now is None else now
    user, client = _norm(username), str(client_ip)
    wait = max(_retry(db, "client = ? AND username = ?", (client, user), PAIR_LIMIT, PAIR_WINDOW, now),
               _retry(db, "client = ?", (client,), CLIENT_LIMIT, CLIENT_WINDOW, now),
               _retry(db, "username = ?", (user,), USER_LIMIT, USER_WINDOW, now))
    return float(int(wait) + 1) if wait > 0 else 0.0


def record(db, client_ip: str, username: str, ok: bool, now: float | None = None) -> None:
    now = time.time() if now is None else now
    db.execute("DELETE FROM login_attempts WHERE ts < ?", (now - KEEP_SECONDS,))
    db.execute("INSERT INTO login_attempts (ts, client, username, ok) VALUES (?, ?, ?, ?)",
               (now, str(client_ip), _norm(username)[:64], int(bool(ok))))
