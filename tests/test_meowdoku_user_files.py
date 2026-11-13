"""Owner and conversation boundaries for account-owned uploads."""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import alpaca_credentials, messaging, user_files  # noqa: E402


class UserFilesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "accounts.db"
        self.db = messaging.connect(self.path)
        self.addCleanup(self.db.close)
        self.db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE, display_name TEXT)")
        self.db.executemany("INSERT INTO users VALUES (?,?,?)", [(1, "alice", "Alice"), (2, "bob", "Bob"),
                                                                  (3, "mallory", "Mallory")])
        user_files.connect(self.path).close()
        self.files_root = self.root / "users"

    def test_private_then_message_to_mutual_contact_then_revoked(self):
        item = user_files.put(self.db, self.files_root, 1, "alpaca-paper.env", b"secret example")
        self.assertEqual(user_files.readable(self.db, self.files_root, 1, item["id"])[1], b"secret example")
        with self.assertRaises(user_files.FileError):
            user_files.readable(self.db, self.files_root, 2, item["id"])
        messaging.request_contact(self.db, 1, "bob")
        messaging.respond_contact(self.db, 2, "alice", True)
        with self.assertRaises(user_files.FileError):
            user_files.readable(self.db, self.files_root, 2, item["id"])
        messaging.send(self.db, 1, "bob", "Shared file", subject={"kind": "file", "ref": item["id"]})
        self.assertEqual(user_files.readable(self.db, self.files_root, 2, item["id"])[1], b"secret example")
        with self.assertRaises(user_files.FileError):
            user_files.readable(self.db, self.files_root, 3, item["id"])
        messaging.remove(self.db, 2, "alice")
        with self.assertRaises(user_files.FileError):
            user_files.readable(self.db, self.files_root, 2, item["id"])

    def test_bad_name_and_quota_do_not_publish(self):
        for name in ("../x", "C:secret", "CON.txt", "x\n.txt"):
            with self.assertRaises(user_files.FileError):
                user_files.put(self.db, self.files_root, 1, name, b"x")
        with self.assertRaises(user_files.FileError):
            user_files.put(self.db, self.files_root, 1, "big.bin", b"x" * (user_files.MAX_FILE_BYTES + 1))
        self.assertEqual(user_files.list_owned(self.db, 1), [])

    def test_paper_credentials_stay_outside_downloadable_files(self):
        secrets_root = self.root / "secrets"
        self.assertFalse(alpaca_credentials.configured(secrets_root, 1))
        alpaca_credentials.save_paper(secrets_root, 1, "paper-key-id", "paper-secret-key")
        self.assertTrue(alpaca_credentials.configured(secrets_root, 1))
        self.assertFalse(alpaca_credentials.configured(secrets_root, 2))
        self.assertEqual(alpaca_credentials.load_paper(secrets_root, 1), ("paper-key-id", "paper-secret-key"))
        self.assertEqual(user_files.list_owned(self.db, 1), [])
        with self.assertRaises(alpaca_credentials.CredentialError):
            alpaca_credentials.save_paper(secrets_root, 1, "bad\nkey-id", "paper-secret-key")


if __name__ == "__main__":
    unittest.main()
