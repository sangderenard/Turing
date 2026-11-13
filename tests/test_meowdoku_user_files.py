"""Owner and conversation boundaries for account-owned uploads."""
from __future__ import annotations

import sqlite3
import base64
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

from meowdoku import alpaca_credentials, messaging, user_files  # noqa: E402


class UserFilesTest(unittest.TestCase):
    @staticmethod
    def b64(data):
        return base64.urlsafe_b64encode(data).decode().rstrip("=")

    def bundle(self, byte):
        return {"public_spki": self.b64(bytes([byte]) * 400),
                "encrypted_private": self.b64(bytes([byte]) * 1800),
                "salt": self.b64(bytes([byte]) * 16), "iv": self.b64(bytes([byte]) * 12),
                "iterations": 600000}

    def envelope(self, recipient=None):
        owner = user_files.key_bundle(self.db, 1)
        peer = user_files.key_bundle(self.db, 2) if recipient else None
        return {"v": 1, "iv": self.b64(b"a" * 12), "name_iv": self.b64(b"b" * 12),
                "name_cipher": self.b64(b"encrypted-name-and-tag"),
                "owner_wrap": self.b64(b"o" * 384), "owner_fp": owner["fingerprint"],
                "recipient": recipient, "recipient_wrap": self.b64(b"p" * 384) if peer else None,
                "recipient_fp": peer["fingerprint"] if peer else None}

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
        user_files.register_key(self.db, 1, self.bundle(1))
        user_files.register_key(self.db, 2, self.bundle(2))
        self.files_root = self.root / "users"

    def test_private_then_message_to_mutual_contact_then_revoked(self):
        cipher = b"ciphertext-and-gcm-tag"
        private = user_files.put(self.db, self.files_root, 1, self.envelope(), cipher)
        self.assertEqual(user_files.readable(self.db, self.files_root, 1, private["id"])[1], cipher)
        with self.assertRaises(user_files.FileError):
            user_files.readable(self.db, self.files_root, 2, private["id"])
        messaging.request_contact(self.db, 1, "bob")
        messaging.respond_contact(self.db, 2, "alice", True)
        item = user_files.put(self.db, self.files_root, 1, self.envelope("bob"), cipher)
        with self.assertRaises(user_files.FileError):
            user_files.readable(self.db, self.files_root, 2, item["id"])
        messaging.send(self.db, 1, "bob", "Shared encrypted file", subject={"kind": "file", "ref": item["id"]})
        self.assertEqual(user_files.readable(self.db, self.files_root, 2, item["id"])[1], cipher)
        with self.assertRaises(user_files.FileError):
            user_files.readable(self.db, self.files_root, 3, item["id"])
        messaging.remove(self.db, 2, "alice")
        with self.assertRaises(user_files.FileError):
            user_files.readable(self.db, self.files_root, 2, item["id"])

    def test_invalid_envelope_and_quota_do_not_publish(self):
        with self.assertRaises(user_files.FileError):
            user_files.put(self.db, self.files_root, 1, {}, b"ciphertext-and-gcm-tag")
        with self.assertRaises(user_files.FileError):
            user_files.put(self.db, self.files_root, 1, self.envelope(), b"plaintext")
        with self.assertRaises(user_files.FileError):
            user_files.put(self.db, self.files_root, 1, self.envelope(), b"x" * (user_files.MAX_BLOB_BYTES + 1))
        self.assertEqual(user_files.list_owned(self.db, 1), [])

    def test_key_registration_is_immutable_and_peer_requires_contact(self):
        with self.assertRaises(user_files.FileError):
            user_files.register_key(self.db, 1, self.bundle(3))
        with self.assertRaises(user_files.FileError):
            user_files.peer_key(self.db, 1, "bob")
        messaging.request_contact(self.db, 1, "bob")
        messaging.respond_contact(self.db, 2, "alice", True)
        self.assertEqual(user_files.peer_key(self.db, 1, "bob")["fingerprint"],
                         user_files.key_bundle(self.db, 2)["fingerprint"])

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
