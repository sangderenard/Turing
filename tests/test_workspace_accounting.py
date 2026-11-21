"""Small offline SQL tests; no server, compiler, account credentials or network."""
import concurrent.futures
import base64
import os
import json
from contextlib import contextmanager
import importlib.util
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'examples/meowdoku/workspace_accounting.py'
spec = importlib.util.spec_from_file_location('workspace_accounting', MODULE)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
A, B, C = 'a' * 64, 'b' * 64, 'c' * 64


@contextmanager
def db_connect(path):
    db = sqlite3.connect(path)
    try:
        with db:
            yield db
    finally:
        db.close()


class WorkspaceAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='workspace-ledger-test-')
        self.root = Path(self.temp.name)
        self.db = self.root / 'accounts.db'
        with db_connect(self.db) as db:
            db.executescript("CREATE TABLE users(id INTEGER PRIMARY KEY); INSERT INTO users VALUES(1),(2),(3); CREATE TABLE account_roles(user_id INTEGER PRIMARY KEY,role TEXT); INSERT INTO account_roles VALUES(3,'admin');")
        self.disk = patch.object(w.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(10000, 0, 10000))
        self.disk.start()
        self.addCleanup(self.disk.stop)
        self.addCleanup(self.temp.cleanup)
        self.store = self.new_store()

    def new_store(self):
        return w.Store(self.db, self.root, capacity_bytes=1000, user_quota_bytes=600, free_floor_bytes=100)

    def active(self, owner=1, size=400):
        key = self.store.reserve(owner, size, base_revision=A)
        self.store.record_usage(owner, key, measured_bytes=200, revision=B, expected_generation=0)
        return key

    def ticket(self):
        key = self.active()
        ticket = self.store.submit_review(1, key, snapshot_digest=B, snapshot_bytes=100, expected_generation=1)
        return key, ticket

    def test_owner_isolation_all_workspace_operations(self):
        key = self.active()
        for action in [lambda: self.store.get(2, key),
                       lambda: self.store.record_usage(2, key, measured_bytes=0, revision=C, expected_generation=1),
                       lambda: self.store.submit_review(2, key, snapshot_digest=B, snapshot_bytes=1, expected_generation=1),
                       lambda: self.store.begin_release(2, key),
                       lambda: self.store.confirm_removed(2, key, expected_generation=1)]:
            with self.assertRaises(w.NotFound): action()
        self.assertEqual(self.store.list_owned(2), [])
        self.assertEqual(self.store.get(1, key)['state'], 'active')

    def test_published_bundle_prevents_workspace_release(self):
        key = self.active()
        with db_connect(self.db) as db:
            db.execute('CREATE TABLE authored_pages(workspace_id TEXT)')
            db.execute('INSERT INTO authored_pages VALUES(?)', (key,))
        with self.assertRaisesRegex(w.Refused, 'published pages prevent workspace release'):
            self.store.begin_release(1, key)

    def test_ticket_owner_and_admin_queue(self):
        key, ticket = self.ticket()
        with self.assertRaises(w.NotFound): self.store.get_ticket(2, ticket)
        with self.assertRaises(w.Refused): self.store.review_queue(2)
        with self.assertRaises(w.Refused): self.store.decide(2, ticket, snapshot_digest=B, decision='approved')
        self.assertEqual(len(self.store.review_queue(3)), 1)
        self.store.decide(3, ticket, snapshot_digest=B, decision='approved')
        self.assertEqual(self.store.get_ticket(1, ticket)['status'], 'approved')
        self.assertEqual(self.store.get(1, key)['revision'], B)  # approval never applies content

    def test_missing_admin_role_table_fails_closed(self):
        with db_connect(self.db) as db: db.execute('DROP TABLE account_roles')
        with self.assertRaises(w.Refused): self.store.review_queue(3)

    def test_admin_revocation_checked_at_decision(self):
        _, ticket = self.ticket()
        self.store.review_queue(3)
        with db_connect(self.db) as db: db.execute('DELETE FROM account_roles WHERE user_id=3')
        with self.assertRaises(w.Refused): self.store.decide(3, ticket, snapshot_digest=B, decision='approved')

    def test_unknown_identity_and_bool_refused(self):
        for actor in [None, True, -1, 99, '1']:
            with self.assertRaises(w.Refused): self.store.reserve(actor, 20, base_revision=A)

    def test_invalid_sizes_and_revisions(self):
        for size in [-1, True, 1.1, '2', 2 ** 63, 0]:
            with self.assertRaises(ValueError): self.store.reserve(1, size, base_revision=A)
        with self.assertRaises(ValueError): self.store.reserve(1, 1, base_revision='../secrets')

    def test_per_user_and_volume_caps(self):
        self.store.reserve(1, 600, base_revision=A)
        with self.assertRaises(w.Refused): self.store.reserve(1, 1, base_revision=A)
        self.store.reserve(2, 400, base_revision=A)
        with self.assertRaises(w.Refused): self.store.reserve(2, 1, base_revision=A)

    def test_outstanding_growth_not_double_charged(self):
        self.active(size=400)  # 200 measured, 200 still reserved
        with patch.object(w.shutil, 'disk_usage', return_value=shutil._ntuple_diskusage(1000, 500, 500)):
            self.store.reserve(2, 200, base_revision=A)  # 200 + 200 future + 100 floor
            with self.assertRaises(w.Refused): self.store.reserve(2, 1, base_revision=A)

    def test_overrun_truthful_and_admission_refuses(self):
        key = self.active()
        self.store.record_usage(1, key, measured_bytes=700, revision=C, expected_generation=1)
        self.assertEqual(self.store.get(1, key)['measured_bytes'], 700)
        with self.assertRaises(w.Refused): self.store.reserve(1, 1, base_revision=A)

    def test_concurrent_reservations_no_overcommit(self):
        def reserve():
            try: return self.new_store().reserve(1, 400, base_revision=A)
            except w.Refused: return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: reserve(), range(2)))
        self.assertEqual(sum(x is not None for x in results), 1)

    def test_stale_revision_and_generation_refused(self):
        key = self.active()
        with self.assertRaises(w.Refused): self.store.record_usage(1, key, measured_bytes=2, revision=C, expected_generation=0)
        with self.assertRaises(w.Refused): self.store.submit_review(1, key, snapshot_digest=C, snapshot_bytes=1, expected_generation=1)
        with self.assertRaises(w.Refused): self.store.submit_review(1, key, snapshot_digest=B, snapshot_bytes=1, expected_generation=0)

    def test_review_snapshot_immutable_idempotent_and_retained(self):
        key, ticket = self.ticket()
        self.assertEqual(self.store.submit_review(1, key, snapshot_digest=B, snapshot_bytes=100, expected_generation=1), ticket)
        with self.assertRaises(w.Refused): self.store.submit_review(1, key, snapshot_digest=B, snapshot_bytes=101, expected_generation=1)
        with self.assertRaises(w.Refused): self.store.record_usage(1, key, measured_bytes=99, revision=C, expected_generation=1)
        self.store.record_usage(1, key, measured_bytes=200, revision=C, expected_generation=1)
        self.assertEqual(self.store.get_ticket(1, ticket)['snapshot_digest'], B)
        with self.assertRaises(w.Refused): self.store.begin_release(1, key)

    def test_unaccounted_snapshot_and_stale_decision(self):
        key, ticket = self.ticket()
        with self.assertRaises(w.Refused): self.store.decide(3, ticket, snapshot_digest=C, decision='approved')
        self.store.decide(3, ticket, snapshot_digest=B, decision='rejected')
        with self.assertRaises(w.Refused): self.store.decide(3, ticket, snapshot_digest=B, decision='approved')
        self.store.record_usage(1, key, measured_bytes=200, revision=C, expected_generation=1)
        with self.assertRaises(w.Refused): self.store.submit_review(1, key, snapshot_digest=C, snapshot_bytes=101, expected_generation=2)

    def test_release_keeps_reservation_until_cleanup_receipt(self):
        key = self.active(size=600)
        self.assertEqual(self.store.begin_release(1, key), 2)
        self.assertEqual(self.store.begin_release(1, key), 2)
        with self.assertRaises(w.Refused): self.store.reserve(1, 1, base_revision=A)
        with self.assertRaises(w.Refused): self.store.confirm_removed(1, key, expected_generation=1)
        self.store.confirm_removed(1, key, expected_generation=2)
        self.store.reserve(1, 600, base_revision=A)

    def test_policy_or_root_change_cannot_split_budget(self):
        other = self.root / 'other'
        other.mkdir()
        with self.assertRaises(w.Refused): w.Store(self.db, other, capacity_bytes=1000, user_quota_bytes=600, free_floor_bytes=100)
        with self.assertRaises(w.Refused): w.Store(self.db, self.root, capacity_bytes=2000, user_quota_bytes=600, free_floor_bytes=100)


class LifecycleRepairTests(unittest.TestCase):
    setUp = WorkspaceAccountingTests.setUp
    new_store = WorkspaceAccountingTests.new_store
    active = WorkspaceAccountingTests.active
    ticket = WorkspaceAccountingTests.ticket

    def test_shared_schema_registered_and_newer_refused_before_writes(self):
        with db_connect(self.db) as db:
            self.assertEqual(db.execute("SELECT version FROM schema_version WHERE module='workspace_accounting'").fetchone()[0],w.SCHEMA_VERSION)
            db.execute("UPDATE schema_version SET version=999 WHERE module='workspace_accounting'")
            db.execute('DROP TABLE workspace_size_profiles')
        with self.assertRaises(w._schema_version.SchemaVersionError): self.new_store()
        with self.assertRaises(w._schema_version.SchemaVersionError): self.store.reserve(1,1,base_revision=A)
        with db_connect(self.db) as db:
            self.assertIsNone(db.execute("SELECT 1 FROM sqlite_master WHERE name='workspace_size_profiles'").fetchone())
            self.assertEqual(db.execute('SELECT COUNT(*) FROM workspace_leases').fetchone()[0],0)

    def test_schema_declare_commit_does_not_split_snapshot_transaction(self):
        key = self.store.reserve(1,600,base_revision=A)
        with db_connect(self.db) as db:
            db.execute("DELETE FROM schema_version WHERE module='workspace_accounting'")
            db.execute("CREATE TRIGGER fail_ticket BEFORE INSERT ON workspace_review_tickets BEGIN SELECT RAISE(ABORT,'test rollback'); END")
        with self.assertRaises(sqlite3.IntegrityError): self.store.seal_and_submit(1,key,{'x':b'1'},expected_generation=0)
        self.assertEqual(self.store.get(1,key)['measured_bytes'],0)
        with db_connect(self.db) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM workspace_review_snapshots').fetchone()[0],0)

    def test_sql_snapshot_retirement_releases_after_review(self):
        key = self.store.reserve(1,600,base_revision=A)
        ticket = self.store.seal_and_submit(1,key,{'x':b'1'},expected_generation=0)
        record = self.store.get_ticket(1,ticket)
        digest = record['snapshot_digest']
        with self.assertRaises(w.Refused): self.store.retire_snapshot(3,ticket,snapshot_digest=digest)
        self.store.decide(3,ticket,snapshot_digest=digest,decision='rejected')
        with self.assertRaises(w.Refused): self.store.retire_snapshot(2,ticket,snapshot_digest=digest)
        with self.assertRaises(w.Refused): self.store.retire_snapshot(3,ticket,snapshot_digest=A)
        with self.assertRaises(w.Refused): self.store.confirm_external_snapshot_removed(3,ticket,snapshot_digest=digest,expected_generation=1,measured_bytes=0)
        self.assertEqual(self.store.retire_snapshot(3,ticket,snapshot_digest=digest),record['snapshot_bytes'])
        self.assertEqual(self.store.retire_snapshot(3,ticket,snapshot_digest=digest),record['snapshot_bytes'])
        self.assertEqual(self.store.get(1,key)['measured_bytes'],0)
        self.assertEqual(self.store.get_ticket(1,ticket)['status'],'rejected')
        with self.assertRaises(w.NotFound): self.store.read_snapshot(1,ticket)
        with self.assertRaises(w.Refused): self.store.seal_and_submit(1,key,{'x':b'1'},expected_generation=2)
        generation = self.store.begin_release(1,key)
        self.store.confirm_removed(1,key,expected_generation=generation)
        self.store.reserve(1,600,base_revision=A)

    def test_external_snapshot_requires_explicit_verified_removal_receipt(self):
        key,ticket = self.ticket()
        self.store.decide(3,ticket,snapshot_digest=B,decision='approved')
        with self.assertRaises(w.Refused): self.store.begin_release(1,key)
        with self.assertRaises(w.Refused): self.store.retire_snapshot(3,ticket,snapshot_digest=B)
        self.assertEqual(self.store.confirm_external_snapshot_removed(3,ticket,snapshot_digest=B,expected_generation=1,measured_bytes=100),100)
        self.assertEqual(self.store.get(1,key)['measured_bytes'],100)
        self.assertEqual(self.store.get_ticket(1,ticket)['status'],'approved')
        generation = self.store.begin_release(1,key)
        self.store.confirm_removed(1,key,expected_generation=generation)
        self.assertEqual(self.store.get(1,key)['state'],'released')

    def test_external_retirement_reconciliation_cannot_double_subtract(self):
        key,ticket = self.ticket()  # 200 measured, 100 retained
        self.store.decide(3,ticket,snapshot_digest=B,decision='rejected')
        # A reconciler observes external deletion before the retirement receipt.
        self.store.record_usage(1,key,measured_bytes=100,revision=C,expected_generation=1)
        with self.assertRaises(w.Refused):
            self.store.confirm_external_snapshot_removed(3,ticket,snapshot_digest=B,expected_generation=1,measured_bytes=100)
        self.assertEqual(self.store.get(1,key)['measured_bytes'],100)
        self.store.confirm_external_snapshot_removed(3,ticket,snapshot_digest=B,expected_generation=2,measured_bytes=100)
        self.assertEqual(self.store.get(1,key)['measured_bytes'],100)

    def test_retirement_rolls_back_payload_deletion_if_audit_insert_fails(self):
        key = self.store.reserve(1,600,base_revision=A)
        ticket = self.store.seal_and_submit(1,key,{'x':b'1'},expected_generation=0)
        digest = self.store.get_ticket(1,ticket)['snapshot_digest']
        self.store.decide(3,ticket,snapshot_digest=digest,decision='approved')
        before = self.store.read_snapshot(1,ticket)
        with db_connect(self.db) as db:
            db.execute("CREATE TRIGGER fail_retire BEFORE INSERT ON workspace_snapshot_retirements BEGIN SELECT RAISE(ABORT,'test rollback'); END")
        with self.assertRaises(sqlite3.IntegrityError): self.store.retire_snapshot(3,ticket,snapshot_digest=digest)
        self.assertEqual(self.store.read_snapshot(1,ticket),before)
        self.assertEqual(self.store.get(1,key)['measured_bytes'],len(before))

    def test_size_profile_keeps_unknown_distinct_from_zero(self):
        self.store.record_size_profile(1,A,checkout_bytes=100)
        self.assertIsNone(self.store.get_size_profile(1,A)['build_peak_bytes'])
        with self.assertRaises(w.NotFound): self.store.get_size_profile(2,A)
        self.store.record_size_profile(1,A,checkout_bytes=100,build_peak_bytes=0)
        self.assertEqual(self.store.get_size_profile(1,A)['build_peak_bytes'],0)
        self.assertEqual(self.store.list_owned(1),[])
        with self.assertRaises(ValueError): self.store.reserve(1,None,base_revision=A)


class SnapshotTests(unittest.TestCase):
    setUp = WorkspaceAccountingTests.setUp
    new_store = WorkspaceAccountingTests.new_store
    def test_explicit_bytes_atomic_immutable_and_charged(self):
        key = self.store.reserve(1,600,base_revision=A)
        ticket = self.store.seal_and_submit(1,key,{'src/main.py':b'print(1)'},expected_generation=0)
        payload = self.store.read_snapshot(1,ticket)
        body = json.loads(payload)
        self.assertEqual(base64.b64decode(body['files'][0]['base64']),b'print(1)')
        self.assertEqual(body['owner_id'],1)
        self.assertEqual(self.store.reviewer_snapshot(3,ticket),payload)
        with self.assertRaises(w.Refused): self.store.reviewer_snapshot(2,ticket)
        with db_connect(self.db) as db: db.execute("UPDATE account_roles SET role='member' WHERE user_id=3")
        with self.assertRaises(w.Refused): self.store.reviewer_snapshot(3,ticket)
        self.assertEqual(self.store.get(1,key)['measured_bytes'],len(payload))
        self.assertEqual(self.store.get_ticket(1,ticket)['snapshot_bytes'],len(payload))
        self.assertEqual(self.store.seal_and_submit(1,key,{'src/main.py':b'print(1)'},expected_generation=0),ticket)
        with self.assertRaises(w.NotFound): self.store.read_snapshot(2,ticket)
        with self.assertRaises(w.NotFound): self.store.seal_and_submit(2,key,{'x':b'x'},expected_generation=1)
        with self.assertRaises(w.Refused): self.store.begin_release(1,key)

    def test_snapshot_quota_failure_is_atomic(self):
        key = self.store.reserve(1,100,base_revision=A)
        before = self.store.get(1,key)
        with self.assertRaises(w.Refused):
            self.store.seal_and_submit(1,key,{'big':b'x'*500},expected_generation=0)
        self.assertEqual(self.store.get(1,key),before)
        self.assertEqual(self.store.review_queue(3),[])
        with db_connect(self.db) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM workspace_review_snapshots').fetchone()[0],0)

    def test_snapshot_path_rejection_before_materialization(self):
        key = self.store.reserve(1,600,base_revision=A)
        for name in ['../other','/absolute','C:/secret','a\\b','x:stream','a/../b','.git/config','NUL','COM1.txt','a.','a//b','a/./b']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store.seal_and_submit(1,key,{name:b'x'},expected_generation=0)
        for files in [{'A':b'a','a':b'b'},{'a':b'a','a/b':b'b'}]:
            with self.assertRaises(ValueError): self.store.seal_and_submit(1,key,files,expected_generation=0)
        self.assertEqual(self.store.get(1,key)['generation'],0)

    def test_snapshot_database_volume_mismatch_refused(self):
        original = Path.stat
        def stat(path, *args, **kwargs):
            result = original(path,*args,**kwargs)
            if path == self.db:
                fields = list(result)
                fields[2] += 1  # st_dev; simulate another filesystem without IO
                return os.stat_result(fields)
            return result
        with patch.object(Path,'stat',stat):
            with self.assertRaises(w.Refused): self.new_store()

    def test_snapshot_bounds(self):
        key = self.store.reserve(1,600,base_revision=A)
        for files in [{}, {'x':bytearray(b'a')}, {'x':b'x'*(w.MAX_SNAPSHOT_BYTES+1)}, {str(i):b'' for i in range(129)}]:
            with self.assertRaises(ValueError): self.store.seal_and_submit(1,key,files,expected_generation=0)
        self.assertEqual(self.store.get(1,key)['measured_bytes'],0)

    def test_snapshot_identity_binds_owner_workspace_and_base(self):
        first = w._snapshot_payload(1,'one',A,{'b':b'2','a':b'1'})
        self.assertEqual(first,w._snapshot_payload(1,'one',A,{'a':b'1','b':b'2'}))
        for args in [(2,'one',A),(1,'two',A),(1,'one',B)]:
            self.assertNotEqual(first[0],w._snapshot_payload(*args,{'a':b'1','b':b'2'})[0])

    def test_corrupt_snapshot_fails_closed(self):
        key = self.store.reserve(1,600,base_revision=A)
        ticket = self.store.seal_and_submit(1,key,{'x':b'1'},expected_generation=0)
        with db_connect(self.db) as db: db.execute("UPDATE workspace_review_snapshots SET payload=X'00'")
        with self.assertRaises(w.Refused): self.store.read_snapshot(1,ticket)

    def test_stale_snapshot_and_disk_pressure_fail_before_write(self):
        key = self.store.reserve(1,600,base_revision=A)
        with self.assertRaises(w.Refused): self.store.seal_and_submit(1,key,{'x':b'1'},expected_generation=1)
        with patch.object(w.shutil,'disk_usage',return_value=shutil._ntuple_diskusage(1000,999,1)):
            with self.assertRaises(w.Refused): self.store.seal_and_submit(1,key,{'x':b'1'},expected_generation=0)
        self.assertEqual(self.store.get(1,key)['measured_bytes'],0)


class SessionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='workspace-session-test-')
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        def load(name):
            spec = importlib.util.spec_from_file_location(name, MODULE.parent / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        self.accounts_mod = load('accounts')
        self.paths = load('state_paths').StatePaths(self.root)
        self.paths.content_dir.mkdir()
        (self.paths.content_dir / 'workspaces').mkdir()
        self.accounts = self.accounts_mod.Accounts(self.paths.accounts_db)
        self.addCleanup(self.accounts._conn.close)
        # Synthetic fixture rows only, no password issuance/login or live DB.
        with db_connect(self.accounts.path) as db:
            db.execute("INSERT INTO users(id,username,password_hash,created_at) VALUES(1,'fixture_alice','not-a-password',0),(2,'fixture_bob','not-a-password',0),(3,'fixture_admin','not-a-password',0)")
            db.execute('CREATE TABLE account_roles(user_id INTEGER PRIMARY KEY,role TEXT)')
            db.execute("INSERT INTO account_roles VALUES(3,'admin')")
            for token, user in [('fixture-a',1),('fixture-b',2),('fixture-admin',3)]:
                db.execute('INSERT INTO sessions VALUES(?,?,0,?)', (token,user,9999999999))
        self.store, self.access = w.for_state(self.paths, self.accounts, capacity_bytes=1000,
                                            user_quota_bytes=600, free_floor_bytes=0)

    def test_real_accounts_session_isolation(self):
        key = self.access.reserve('fixture-a', 100, base_revision=A)
        self.assertEqual(self.access.get('fixture-a', key)['owner_id'], 1)
        with self.assertRaises(w.NotFound): self.access.get('fixture-b', key)
        self.assertEqual(self.access.list_owned('fixture-b'), [])
        with self.assertRaises(w.Refused): self.access.review_queue('fixture-b')

    def test_logout_and_expiry_rechecked(self):
        key = self.access.reserve('fixture-a', 100, base_revision=A)
        self.accounts.logout('fixture-a')
        with self.assertRaises(w.Refused): self.access.get('fixture-a', key)
        with db_connect(self.accounts.path) as db:
            db.execute("UPDATE sessions SET expires_at=0 WHERE token='fixture-b'")
        with self.assertRaises(w.Refused): self.access.list_owned('fixture-b')

    def test_no_fallback_for_anonymous_or_malformed_session(self):
        for token in [None, '', {}, 1, 'missing', 'x' * 4097]:
            with self.assertRaises(w.Refused): self.access.list_owned(token)

    def test_no_owner_dispatch_for_worker_receipts(self):
        for name in ['record_usage', 'confirm_removed', 'submit_review', 'seal_and_submit', 'read_snapshot', 'retire_snapshot', 'confirm_external_snapshot_removed', 'record_size_profile']:
            self.assertFalse(hasattr(self.access, name))
        with self.assertRaises(TypeError):
            self.access.reserve('fixture-a', 1, base_revision=A, actor_id=2)

    def test_accounts_database_mismatch_refused(self):
        other = self.accounts_mod.Accounts(self.root / 'other.db')
        try:
            with self.assertRaises(w.Refused): w.SessionAccess(other, self.store)
            other.path = self.accounts.path  # same label cannot disguise another open DB
            with self.assertRaises(w.Refused): w.SessionAccess(other, self.store)
            with self.assertRaises(w.Refused):
                w.for_state(self.paths, other, capacity_bytes=1000, user_quota_bytes=600, free_floor_bytes=0)
        finally:
            other._conn.close()

    def test_admin_session_cannot_spoof_or_keep_revoked_role(self):
        key = self.access.reserve('fixture-a', 100, base_revision=A)
        self.store.record_usage(1,key,measured_bytes=50,revision=B,expected_generation=0)
        ticket = self.store.submit_review(1,key,snapshot_digest=B,snapshot_bytes=40,expected_generation=1)
        self.assertEqual(len(self.access.review_queue('fixture-admin')),1)
        with db_connect(self.accounts.path) as db:
            db.execute("UPDATE account_roles SET role='member' WHERE user_id=3")
        with self.assertRaises(w.Refused):
            self.access.decide('fixture-admin',ticket,snapshot_digest=B,decision='approved')
        self.assertEqual(self.access.get_ticket('fixture-a',ticket)['status'],'open')

    def test_state_adapter_never_creates_or_restricts_directories(self):
        # Empty workspace root is provisioned by state owner, not this adapter.
        (self.paths.content_dir / 'workspaces').rmdir()
        with self.assertRaises(FileNotFoundError):
            w.for_state(self.paths,self.accounts,capacity_bytes=1000,user_quota_bytes=600,free_floor_bytes=0)
        self.assertFalse((self.paths.content_dir / 'workspaces').exists())
        self.assertFalse(self.paths.secrets_dir.exists())


if __name__ == '__main__':
    unittest.main()
