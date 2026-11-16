"""Preview freezing, return credentials, explicit wipe, and active People separation."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku.accounts import AccountError, Accounts
from meowdoku.account_admin import people


def test_freeze_recover_and_wipe_are_distinct(tmp_path):
    accounts = Accounts(tmp_path / "accounts.db")
    member = accounts.signup("member", "distinct-test-password")
    preview = accounts.create_preview()
    token = accounts.create_session(preview["id"])
    assert accounts.user_for_token(token)["id"] == preview["id"]
    assert [p["username"] for p in people(accounts.path)] == [member["username"]]
    assert accounts.look_queue()[0]["state"] == "active"

    with accounts.lock:
        accounts._conn.execute("CREATE TABLE chat_messages (user_id INTEGER, text TEXT)")
        accounts._conn.execute("INSERT INTO chat_messages VALUES (?,?)", (preview["id"], "approved post"))
        accounts._conn.execute("UPDATE temporary_accounts SET expires_at=? WHERE user_id=?",
                               (time.time() - 1, preview["id"]))
        accounts._conn.commit()
    assert accounts.freeze_expired() == 1
    assert accounts.freeze_expired() == 0
    assert accounts.user_for_token(token) is None
    assert accounts.look_queue()[0]["state"] == "frozen"
    with pytest.raises(AccountError):
        accounts.wipe_preview(preview["id"], "wrong-name")

    returned = accounts.recover_preview(preview["recovery_cookie"], None)
    assert returned["id"] == preview["id"]
    assert accounts.look_queue()[0]["state"] == "active"
    with pytest.raises(AccountError):
        accounts.recover_preview(preview["recovery_cookie"], None)
    with accounts.lock:
        accounts._conn.execute("UPDATE temporary_accounts SET expires_at=? WHERE user_id=?",
                               (time.time() - 1, preview["id"]))
        accounts._conn.commit()
    accounts.freeze_expired()
    returned_by_code = accounts.recover_preview(None, returned["recovery_code"])
    assert returned_by_code["id"] == preview["id"]
    with accounts.lock:
        accounts._conn.execute("UPDATE temporary_accounts SET expires_at=? WHERE user_id=?",
                               (time.time() - 1, preview["id"]))
        accounts._conn.commit()
    accounts.freeze_expired()
    accounts.wipe_preview(preview["id"], preview["username"])
    assert accounts.look_queue() == []
    assert accounts._conn.execute("SELECT text FROM chat_messages WHERE user_id=?", (preview["id"],)).fetchone()[0] == "approved post"
    assert [p["username"] for p in people(accounts.path)] == [member["username"]]


def test_legacy_preview_gets_recovery_once_and_keep_removes_it(tmp_path):
    accounts = Accounts(tmp_path / "accounts.db")
    preview = accounts.create_preview()
    with accounts.lock:
        accounts._conn.execute("DELETE FROM preview_recovery WHERE user_id=?", (preview["id"],))
        accounts._conn.commit()
    credentials = accounts.issue_legacy_recovery(preview["id"])
    assert credentials and accounts.issue_legacy_recovery(preview["id"]) is None
    assert accounts.recover_preview(None, credentials["recovery_code"])["id"] == preview["id"]
    accounts.keep_preview(preview["id"], "kept_member", "distinct-test-password",
                          alpha_terms_version="alpha-2026-11-15")
    assert accounts.look_queue() == []
    with pytest.raises(AccountError):
        accounts.recover_preview(credentials["recovery_cookie"], None)
