"""Site identity, storage and lifetime tests. Never imports/runs a compiler."""
import json
from pathlib import Path
import sqlite3
import stat
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'examples')]
from meowdoku import compile_artifacts as ca
from meowdoku import workspace_accounting as wa
from meowdoku import divider_nodus_bridge as br
from meowdoku import source_divider as sd
from meowdoku.state_paths import StatePaths
from meowdoku.site_compiler import SiteCompiler


class CompileArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='site-compile-accounting-')
        self.addCleanup(self.temp.cleanup)
        self.state = StatePaths(Path(self.temp.name) / 'state')
        self.db = sd.connect(self.state.accounts_db)
        self.addCleanup(self.db.close)
        self.db.execute('CREATE TABLE users(id INTEGER PRIMARY KEY)')
        self.db.execute('INSERT INTO users VALUES(1),(2)')
        self.root = self.state.content_dir / 'workspaces'
        self.root.mkdir(parents=True)
        self.store = wa.Store(self.state.accounts_db, self.root, capacity_bytes=1_000_000,
                              user_quota_bytes=500_000, free_floor_bytes=0)
        self.artifacts = ca.Artifacts(self.state, self.store, reservation_bytes=10_000, review_after_s=60)
        now = time.time()
        self.db.execute('INSERT INTO divider_jobs VALUES(?,?,?,?,?,?,?)', ('job', 'account:1', '', '', now, 'running', 'c'))
        self.db.execute("INSERT INTO divider_attempts(job_id,piece_id,backend,state,lease_owner,lease_generation,lease_deadline) "
                        "VALUES('job',0,'c','leased','worker',1,?)", (now + 60,))
        self.claim = dict(job_id='job', piece_id=0, backend='c', worker_id='worker', generation=1)

    def admit(self):
        return self.artifacts.admit(self.claim, 42, 'a' * 64)

    def test_owner_comes_from_site_job_not_claim(self):
        self.claim['owner_id'] = 2
        context = self.admit()
        self.assertEqual(context['owner_id'], 1)
        self.assertEqual(self.store.list_owned(2), [])
        self.assertEqual(self.store.get(1, context['workspace_id'])['reserved_bytes'], 10_000)

    def test_idempotent_binding_reserves_once(self):
        self.assertEqual(self.admit(), self.admit())
        self.assertEqual(len(self.store.list_owned(1)), 1)

    def test_stale_or_foreign_worker_refused_before_reservation(self):
        for field, value in [('generation', 2), ('worker_id', 'other'), ('job_id', 'missing')]:
            claim = {**self.claim, field: value}
            with self.assertRaises(wa.Refused):
                self.artifacts.admit(claim, 42, 'a' * 64)
        self.assertEqual(self.store.list_owned(1), [])

    def test_browser_and_missing_account_owners_refused(self):
        for owner in ('browser:abc', 'account:999', 'account:01', '../../x'):
            self.db.execute('UPDATE divider_jobs SET owner_key=?', (owner,))
            with self.assertRaises(wa.Refused):
                self.admit()

    def test_quota_refusal_has_no_orphan_run_or_lease(self):
        self.store.reserve(1, 500_000, base_revision='a' * 64)
        with self.assertRaises(wa.Refused):
            self.admit()
        self.assertEqual(self.db.execute('SELECT count(*) FROM compile_artifact_runs').fetchone()[0], 0)
        self.assertEqual(len(self.store.list_owned(1)), 1)

    def test_failed_mapping_insert_rolls_back_reservation(self):
        self.db.execute("CREATE TRIGGER break_run BEFORE INSERT ON compile_artifact_runs BEGIN SELECT RAISE(ABORT, 'failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.admit()
        self.assertEqual(self.store.list_owned(1), [])

    def test_all_host_files_measured_and_mutable_usage_not_sealed(self):
        context = self.admit()
        out = self.root / '42'
        out.mkdir()
        (out / 'big.pkl').write_bytes(b'a' * 12_000)
        (self.root / '42.request.json').write_bytes(b'abc')
        (self.root / '42.divider.json').write_bytes(b'defg')
        (self.root / '42.extraction-contract.yaml').write_bytes(b'hi')
        self.assertTrue(self.artifacts.reconcile(42))
        row = self.store.get(1, context['workspace_id'])
        self.assertEqual(row['measured_bytes'], 12_009)
        self.assertIsNone(row['revision'])
        view = ca.job_storage(self.db, 'job', 'account:1')[0]
        self.assertEqual(view['charged_bytes'], 12_009)  # overrun is truthful
        self.assertEqual(ca.job_storage(self.db, 'job', 'account:2'), [])

    def test_scan_failure_preserves_last_charge_and_records_problem(self):
        self.admit()
        (self.root / '42.request.json').write_bytes(b'abc')
        self.artifacts.reconcile(42)
        with patch.object(ca, '_regular_size', side_effect=PermissionError('busy')):
            self.assertFalse(self.artifacts.reconcile(42))
        row = ca.job_storage(self.db, 'job', 'account:1')[0]
        self.assertEqual(row['measured_bytes'], 3)
        self.assertEqual(row['measurement_error'], 'measurement requires attention')
        self.assertIn('PermissionError', self.db.execute('SELECT measurement_error FROM compile_artifact_runs').fetchone()[0])

    def test_release_request_stays_charged_and_measurement_invalidates_old_receipt(self):
        context = self.admit()
        generation = self.store.begin_release(1, context['workspace_id'])
        (self.root / '42.request.json').write_bytes(b'x' * 12_000)
        self.artifacts.monitor(force=True)
        row = self.store.get(1, context['workspace_id'])
        self.assertEqual(row['state'], 'releasing')
        self.assertEqual(row['measured_bytes'], 12_000)
        with self.assertRaises(wa.Refused):
            self.store.confirm_removed(1, context['workspace_id'], expected_generation=generation)

    def test_elapsed_review_or_unfinished_attempt_never_deletes_or_releases(self):
        context = self.admit()
        file = self.root / '42.request.json'
        file.write_bytes(b'keep')
        self.db.execute('UPDATE compile_artifact_runs SET review_after=0')
        self.artifacts.status(42, 'unfinished')
        self.artifacts.monitor()
        row = ca.job_storage(self.db, 'job', 'account:1')[0]
        self.assertTrue(file.exists() and row['review_due'])
        self.assertEqual(row['charged_bytes'], 10_000)
        self.assertIsNone(row['finished_at'])
        self.assertEqual(self.store.get(1, context['workspace_id'])['state'], 'active')

    def test_repository_root_rejected_before_creation(self):
        repo = Path(self.temp.name) / 'repo'
        repo.mkdir()
        (repo / '.git').write_text('gitdir: elsewhere')
        target = repo / 'generated'
        with self.assertRaises(wa.Refused):
            ca.external_directory(target)
        self.assertFalse(target.exists())

    def test_measurement_refuses_hardlinks_and_reparse_points(self):
        for links, attributes in ((2, 0), (1, 0x400)):
            info = SimpleNamespace(st_mode=stat.S_IFREG, st_nlink=links, st_file_attributes=attributes, st_size=100)
            with patch.object(Path, 'lstat', return_value=info), self.assertRaises(wa.Refused):
                ca._regular_size(self.root / '42.request.json')

    def test_binding_to_other_database_or_jobs_root_refused(self):
        with self.assertRaises(wa.Refused):
            self.artifacts.bind(self.db, self.root / 'elsewhere')
        db = sqlite3.connect(':memory:')
        try:
            with self.assertRaises(wa.Refused):
                self.artifacts.bind(db, self.root)
        finally:
            db.close()

    def test_newer_schema_refuses_before_run_or_workspace_write(self):
        self.db.execute("UPDATE schema_version SET version=999 WHERE module='compile_artifacts'")
        with self.assertRaises(ca.schema_version.SchemaVersionError):
            self.admit()
        self.assertEqual(self.store.list_owned(1), [])

    def test_unconfigured_dispatch_refused_before_claim(self):
        self.db.execute("UPDATE divider_attempts SET state='queued'")
        route = br.NodusCompileRoute('127.0.0.1', 1, 7, self.root)
        with self.assertRaises(ValueError):
            br.run_one(self.db, route, lease_owner='x')
        self.assertEqual(self.db.execute('SELECT state FROM divider_attempts').fetchone()[0], 'queued')

    def test_site_config_reuses_accounts_ledger_and_wake_hook(self):
        accounts = SimpleNamespace(_conn=self.db, lock=threading.Lock())
        config = dict(jobs_dir=str(self.root), service_id=7, port=1234, backend='llvm', capacity_bytes=1_000_000,
                      user_quota_bytes=500_000, free_floor_bytes=0, reservation_bytes=10_000, review_after_s=60)
        worker = SiteCompiler(self.state, accounts, config)
        self.assertEqual(worker.backends, ('llvm',))
        self.assertEqual(worker.route.artifacts.store.db_path, self.state.accounts_db.resolve())
        worker('job')
        self.assertTrue(worker._wake.is_set())
        self.assertIsNone(worker.thread)  # configuration alone does not run a compiler


if __name__ == '__main__':
    unittest.main()
