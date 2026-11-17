"""[REDACT] The dev export holds nothing a developer could not read on live, and every log value is aliased."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import private_record, redactor  # noqa: E402

LIVE_KEY = b"L" * 32


@pytest.fixture
def live(tmp_path):
    state = tmp_path / "live"
    (state / "sql").mkdir(parents=True)
    (state / "secrets").mkdir()
    db = sqlite3.connect(state / "sql" / "accounts.db")
    live_token = private_record.Record(LIVE_KEY).token("ip", "198.51.100.77")
    db.executescript(f"""
        CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password_hash TEXT NOT NULL, display_name TEXT, created_at REAL);
        CREATE TABLE account_roles (user_id INTEGER, role TEXT, granted_by INTEGER, created_at REAL);
        CREATE TABLE account_permission_grants (user_id INTEGER, group_name TEXT, granted_by INTEGER, created_at REAL);
        CREATE TABLE sessions (token TEXT, user_id INTEGER, created_at REAL, expires_at REAL);
        CREATE TABLE login_attempts (ts REAL, client TEXT, username TEXT, ok INTEGER);
        CREATE TABLE msg_messages (id INTEGER PRIMARY KEY, sender_id INTEGER, recipient_id INTEGER, body TEXT, created REAL,
            read_at REAL, removed_at REAL, removed_by INTEGER, subject_kind TEXT, subject_ref TEXT, msg_type TEXT);
        CREATE TABLE chat_messages (id INTEGER PRIMARY KEY, room TEXT, user_id INTEGER, username TEXT, body TEXT, created REAL,
            removed_at REAL, removed_by INTEGER);
        CREATE TABLE nl_articles (id TEXT, revision_of TEXT, author_id INTEGER, author_name TEXT, title TEXT, body TEXT, status TEXT,
            key_hash TEXT, created REAL, updated REAL, published_at REAL, decided_by INTEGER, decided_at REAL, note TEXT);
        INSERT INTO users VALUES (1, 'boss', 'aa$bb', 'boss', 0), (2, 'dev', 'cc$dd', 'dev', 0), (3, 'member', 'ee$ff', 'member', 0);
        INSERT INTO account_roles VALUES (1, 'admin', 0, 0);
        INSERT INTO account_permission_grants VALUES (2, 'dev', 1, 0);
        INSERT INTO sessions VALUES ('bearer-secret-0001', 3, 0, 9e9);
        INSERT INTO login_attempts VALUES (1, '203.0.113.7', 'member', 1), (2, '{live_token}', 'boss', 1);
        INSERT INTO msg_messages (id, sender_id, recipient_id, body, created) VALUES (1, 3, 1, 'my secret plan, mail me at me@home.example', 0);
        INSERT INTO chat_messages VALUES (1, 'agents', 1, 'boss', 'seen from 192.0.2.44 and {live_token}', 0, NULL, NULL);
        INSERT INTO nl_articles VALUES ('a', NULL, 1, 'boss', 'Out', 'published text', 'published', 'k1', 0, 0, 0, NULL, NULL, NULL),
                                       ('b', NULL, 3, 'member', 'Draft', 'unpublished draft', 'pending', 'k2', 0, 0, NULL, NULL, NULL, NULL);
    """)
    db.commit(); db.close()
    return state, live_token


def _export(tmp_path, state):
    out = tmp_path / "dev" / "sql" / "accounts.db"
    redactor.export(state / "sql" / "accounts.db", out, key_path=tmp_path / "dev" / "secrets" / "private_hash.key",
                    log=lambda line: None)
    return out, sqlite3.connect(out)


def test_a_table_without_a_rule_stops_the_export(tmp_path, live):
    state, _ = live
    db = sqlite3.connect(state / "sql" / "accounts.db")
    db.execute("CREATE TABLE brand_new_feature (secret TEXT)"); db.commit(); db.close()
    with pytest.raises(redactor.RedactError, match="brand_new_feature"):
        _export(tmp_path, state)
    assert not (tmp_path / "dev" / "sql" / "accounts.db").exists(), "nothing is left behind"


def test_the_export_aliases_logs_drops_secrets_and_withholds_private_content(tmp_path, live):
    state, live_token = live
    before = (state / "sql" / "accounts.db").read_bytes()
    out, db = _export(tmp_path, state)
    blob = out.read_bytes()
    for raw in (b"203.0.113.7", b"192.0.2.44", live_token.encode(), b"bearer-secret-0001", b"my secret plan",
                b"me@home.example", b"unpublished draft", b"k1", b"k2", b"ee$ff"):
        assert raw not in blob, raw                       # secure_delete + VACUUM: not even in a freed page
    assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    clients = [r[0] for r in db.execute("SELECT client FROM login_attempts ORDER BY ts")]
    dev = private_record.Record(private_record.load_key(tmp_path / "dev" / "secrets" / "private_hash.key"))
    assert clients == [dev.token("ip", "203.0.113.7"), dev.token("ip", live_token)], "aliased with the dev key, live token re-aliased"
    body = db.execute("SELECT body FROM chat_messages").fetchone()[0]
    assert body == f"seen from {dev.token('ip', '192.0.2.44')} and {dev.token('ip', live_token)}", "room text kept, aliased"
    assert db.execute("SELECT body FROM msg_messages").fetchone()[0] == redactor.WITHHELD, "a direct message is withheld"
    assert dict(db.execute("SELECT id, body FROM nl_articles")) == {"a": "published text", "b": redactor.WITHHELD}
    hashes = dict(db.execute("SELECT username, password_hash FROM users"))
    assert hashes == {"boss": "aa$bb", "dev": "cc$dd", "member": redactor.NEVER_VERIFIES}, "only dev-side accounts keep a hash"
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert (state / "sql" / "accounts.db").read_bytes() == before, "the source is never written"
    db.close()


def test_the_export_refuses_the_live_key_and_its_own_source(tmp_path, live):
    state, _ = live
    with pytest.raises(redactor.RedactError, match="dev state's own"):
        redactor.export(state / "sql" / "accounts.db", tmp_path / "out.db", key_path=state / "secrets" / "private_hash.key")
    with pytest.raises(redactor.RedactError, match="replace its source"):
        redactor.export(state / "sql" / "accounts.db", state / "sql" / "accounts.db", key_path=tmp_path / "k")


def test_every_rule_is_a_known_kind():
    kinds = {"ip", "email", "ua", "text", "json", "withhold", "never", "null", "unpublished", "hash"}
    for table, rule in redactor.RULES.items():
        assert rule in (redactor.KEEP, redactor.DROP) or (isinstance(rule, dict) and set(rule.values()) <= kinds), table
