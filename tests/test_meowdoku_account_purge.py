"""[PURGE] An admin purges an ordinary permanent account: private rows and sign-in go, public posts stay under a sealed name."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import accounts as A  # noqa: E402


def _store(tmp_path):
    store = A.Accounts(tmp_path / "accounts.db")
    db = store._conn
    db.execute("CREATE TABLE IF NOT EXISTS account_roles (user_id INTEGER PRIMARY KEY, role TEXT, granted_by TEXT, granted_at REAL)")
    db.execute("CREATE TABLE IF NOT EXISTS account_permission_grants (user_id INTEGER, group_name TEXT, granted_by INTEGER, created_at REAL)")
    db.execute("CREATE TABLE IF NOT EXISTS msg_contacts (user_id INTEGER, other_id INTEGER)")
    db.execute("CREATE TABLE IF NOT EXISTS posts (id INTEGER PRIMARY KEY, owner_id INTEGER NOT NULL REFERENCES users(id), body TEXT)")
    db.commit()
    return store


def test_private_rows_and_sign_in_go_and_posts_stay_with_a_valid_owner(tmp_path):
    store = _store(tmp_path)
    user = store.signup("barnacleboy", "a long enough password", alpha_terms_version=A.ALPHA_TERMS_VERSION)
    other = store.signup("someone", "another long password")
    store.create_session(user["id"])
    db = store._conn
    db.execute("INSERT INTO msg_contacts VALUES (?,?)", (user["id"], other["id"]))
    db.execute("INSERT INTO posts(owner_id, body) VALUES (?, 'kept')", (user["id"],))
    db.commit()
    store.purge_account(user["id"], "barnacleboy")
    assert db.execute("SELECT username, display_name FROM users WHERE id=?", (user["id"],)).fetchone() == \
        (f"purged-{user['id']}", "(purged account)")
    for table in ("sessions", "msg_contacts", "alpha_agreements"):
        assert db.execute(f"SELECT COUNT(*) FROM {table} WHERE user_id=?", (user["id"],)).fetchone()[0] == 0, table
    assert db.execute("SELECT body FROM posts WHERE owner_id=?", (user["id"],)).fetchone() == ("kept",)
    assert db.execute("PRAGMA foreign_key_check").fetchall() == [], "no post is left without an owner"
    with pytest.raises(A.AccountError):
        store.login("barnacleboy", "a long enough password")
    with pytest.raises(A.AccountError):
        store.login(f"purged-{user['id']}", "a long enough password")
    assert store.signup("barnacleboy", "a new long password")["username"] == "barnacleboy", "the name is free again"


@pytest.mark.parametrize("table,row", [("account_roles", (None, "member", "x", 0)), ("account_permission_grants", (None, "dev", 1, 0))])
def test_an_account_with_a_role_or_a_group_is_refused(tmp_path, table, row):
    store = _store(tmp_path)
    user = store.signup("agent", "a long enough password")
    store._conn.execute(f"INSERT INTO {table} VALUES ({','.join('?' * len(row))})", (user["id"],) + row[1:])
    store._conn.commit()
    with pytest.raises(A.AccountError) as refused:
        store.purge_account(user["id"], "agent")
    assert refused.value.status == 409


def test_the_name_must_match_and_a_preview_is_wiped_not_purged(tmp_path):
    store = _store(tmp_path)
    user = store.signup("person", "a long enough password")
    with pytest.raises(A.AccountError) as refused:
        store.purge_account(user["id"], "someone-else")
    assert refused.value.status == 409
    preview = store.create_preview()
    with pytest.raises(A.AccountError, match="wiped, not purged"):
        store.purge_account(preview["id"], preview["username"])


def test_the_route_is_admin_only_same_site_and_rechecks_the_password():
    text = (ROOT / "examples" / "meowdoku" / "server.py").read_text(encoding="utf-8")
    route = text[text.index('if path0 == "/api/account/admin/purge":'):text.index('if path0 == "/api/account/admin/look-wipe":')]
    for needed in ('self._require("account_manage")', '!= "admin"', '"X-Meowdoku-Account-Purge"', "login_guard_mod.check",
                   "sessions.accounts.login(actor", "sessions.accounts.purge_account(", "logring_mod.RING.add"):
        assert needed in route, needed
