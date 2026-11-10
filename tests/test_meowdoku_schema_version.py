import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "examples", ROOT / "examples" / "meowdoku", ROOT / "examples" / "meowdoku" / "deploy"):
    sys.path.insert(0, str(p))

from meowdoku import accounts, authz, job_library, lowering_library, schema_version  # noqa: E402
from meowdoku.deploy import apache_supervisor as sup  # noqa: E402


class SchemaVersionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "accounts.db"

    def rows(self):
        c = sqlite3.connect(self.db)
        try:
            return schema_version.report(c)
        finally:
            c.close()

    def test_all_modules_declare(self):
        with sup.connect(self.db):
            pass
        with lowering_library.connect(self.db):
            pass
        with job_library.connect(Path(self.tmp.name)):
            pass
        rows = self.rows()
        self.assertEqual({r[0] for r in rows}, {"accounts", "authz", "lowering_library", "supervisor"})
        # job_library keeps its own sqlite file
        c = sqlite3.connect(Path(self.tmp.name) / "job-library.sqlite")
        try:
            self.assertEqual([r[:2] for r in schema_version.report(c)], [("job_library", 1)])
        finally:
            c.close()
        self.assertTrue(all(r[1] == (2 if r[0] == "authz" else 1) and r[3] for r in rows))

    def test_newer_database_refused_and_nothing_written(self):
        c = sqlite3.connect(self.db)
        schema_version.declare(c, "authz", 99)
        c.close()
        with self.assertRaises(schema_version.SchemaVersionError) as cm:
            authz._connect(self.db)
        self.assertEqual((cm.exception.module, cm.exception.stored, cm.exception.version), ("authz", 99, 2))
        c = sqlite3.connect(self.db)
        try:
            self.assertEqual([r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")], ["schema_version"])
            self.assertEqual([r[:2] for r in schema_version.report(c)], [("authz", 99)])
        finally:
            c.close()

    def test_bump_path(self):
        c = sqlite3.connect(self.db)
        schema_version.declare(c, "authz", 0, writer="old")
        schema_version.declare(c, "authz", 1, writer="new")
        row = [r for r in schema_version.report(c) if r[0] == "authz"][0]
        c.close()
        self.assertEqual((row[1], row[3]), (1, "new"))


if __name__ == "__main__":
    unittest.main()
