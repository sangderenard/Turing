"""Repeated asset ownership must not hold the shared accounts DB writer lock."""
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku.accounts import Accounts


def test_duplicate_claim_rolls_back_and_allows_other_writers(tmp_path):
    path = tmp_path / "accounts.db"
    accounts = Accounts(path)
    try:
        assert accounts.claim(7, "photo", "same-digest") is True
        assert accounts.claim(7, "photo", "same-digest") is False
        assert not accounts._conn.in_transaction
        assert accounts.owner_of("photo", "same-digest") == 7
        assert accounts.claim(8, "photo", "same-digest") is False
        assert not accounts._conn.in_transaction
        assert accounts.owner_of("photo", "same-digest") == 7
        with sqlite3.connect(path, timeout=0.1) as other:
            other.execute("CREATE TABLE other_writer (value INTEGER)")
            other.execute("INSERT INTO other_writer VALUES (1)")
        assert accounts.claim(7, "photo", "another-digest") is True
    finally:
        accounts._conn.close()
