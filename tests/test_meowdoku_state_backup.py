import json
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

MK = Path(__file__).resolve().parents[1] / "examples" / "meowdoku"
sys.path.insert(0, str(MK))
import state_backup as sb  # noqa: E402


def make_root(base: Path) -> Path:
    root = base / "state"
    for d in ("sql", "content", "products", "jobs", "canvas", "secrets", "logs", "run"):
        (root / d).mkdir(parents=True)
    db = sqlite3.connect(root / "sql" / "accounts.db")
    db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    db.executemany("INSERT INTO t (v) VALUES (?)", [("x",)] * 50)
    db.commit()
    db.close()
    (root / "content" / "profiles.json").write_text("{}")
    (root / "canvas" / "c.bin").write_bytes(b"abc")
    (root / "secrets" / "admin_token.txt").write_text("SECRET")
    (root / "logs" / "a.log").write_text("log")
    (root / "run" / "p.pid").write_text("1")
    return root


class StateBackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mkbk"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = make_root(self.tmp)
        self.out = self.tmp / "backups"

    def test_snapshot_live_consistent_and_excludes_secrets(self):
        stop = threading.Event()

        def writer():
            db = sqlite3.connect(self.root / "sql" / "accounts.db", timeout=10)
            while not stop.is_set():
                db.execute("INSERT INTO t (v) VALUES ('w')")
                db.commit()
            db.close()

        th = threading.Thread(target=writer)
        th.start()
        try:
            time.sleep(0.05)
            m = sb.snapshot(self.root, self.out)
        finally:
            stop.set()
            th.join()
        snap = Path(m["snapshot"])
        self.assertTrue(sb.verify(snap)["ok"])
        rels = {f["rel"] for f in m["files"]}
        self.assertIn("sql/accounts.db", rels)
        self.assertIn("content/profiles.json", rels)
        self.assertFalse(any(r.startswith(("secrets/", "logs/", "run/")) for r in rels))
        self.assertFalse((snap / "secrets").exists())
        db = next(f for f in m["files"] if f["kind"] == "sqlite")
        self.assertEqual(db["integrity"], "ok")
        self.assertGreaterEqual(db["rows"]["t"], 50)

    def test_refuses_inside_state_dir_and_git_tree(self):
        with self.assertRaises(sb.BackupError):
            sb.snapshot(self.root, self.root / "backups")
        repo = self.tmp / "repo"
        repo.mkdir()
        import subprocess
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        with self.assertRaises(sb.BackupError):
            sb.snapshot(self.root, repo / "bk")
        self.assertFalse((repo / "bk").exists())

    def test_retention_keeps_n_and_spares_foreign_dirs(self):
        self.out.mkdir()
        for name in ("20200101T000000Z", "20200102T000000Z", "20200103T000000Z"):
            (self.out / name).mkdir()
            (self.out / name / "manifest.json").write_text(json.dumps({"files": [], "skipped": [], "include_secrets": False}))
        (self.out / "20190101T000000Z").mkdir()          # matching name, no manifest: kept
        (self.out / "keepme").mkdir()
        (self.out / "keepme" / "manifest.json").write_text("{}")  # manifest, wrong name: kept
        sb.snapshot(self.root, self.out, keep=2)
        names = sorted(p.name for p in self.out.iterdir())
        self.assertIn("20190101T000000Z", names)
        self.assertIn("keepme", names)
        self.assertNotIn("20200101T000000Z", names)
        self.assertNotIn("20200102T000000Z", names)
        self.assertIn("20200103T000000Z", names)
        self.assertEqual(len(sb._snapshots(self.out)), 2)

    def test_restore_into_empty_verifies_and_refuses_occupied(self):
        snap = Path(sb.snapshot(self.root, self.out)["snapshot"])
        target = self.tmp / "restored"
        target.mkdir()
        r = sb.restore(snap, target, into_empty=True)
        self.assertTrue(r["ok"])
        self.assertEqual(sqlite3.connect(target / "sql" / "accounts.db").execute("PRAGMA integrity_check").fetchone()[0], "ok")
        with self.assertRaises(sb.BackupError):
            sb.restore(snap, target)                       # occupied now
        with self.assertRaises(sb.BackupError):
            sb.restore(snap, self.root)                    # live root
        self.assertEqual(sb.restore(snap, self.tmp / "other", dry_run=True)["dry_run"], True)
        self.assertFalse((self.tmp / "other").exists())

    def test_tampered_file_fails_verify(self):
        snap = Path(sb.snapshot(self.root, self.out)["snapshot"])
        (snap / "content" / "profiles.json").write_text("{\"x\":1}")
        v = sb.verify(snap)
        self.assertFalse(v["ok"])
        with self.assertRaises(sb.BackupError):
            sb.restore(snap, self.tmp / "t2")


if __name__ == "__main__":
    unittest.main()
