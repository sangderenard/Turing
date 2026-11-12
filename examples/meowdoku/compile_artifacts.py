"""Nodus compile files charged to the site's existing account workspace ledger.

The page's authenticated divider_jobs.owner_key is the owner authority, not a
request body, filesystem username, Nodus participant id or an operator account.
This module owns only compile_artifact_runs, referencing workspace_leases. It
does not delete files, free an expired reservation, launch a compiler, or claim
that an accounting reservation is a sandbox / hard disk quota.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
import re
import stat
import time

from meowdoku import schema_version
from meowdoku import workspace_accounting as workspace

SCHEMA_VERSION = 1
DDL = """CREATE TABLE IF NOT EXISTS compile_artifact_runs (
    volume_id TEXT NOT NULL, contract_id INTEGER NOT NULL,
    workspace_id TEXT NOT NULL UNIQUE REFERENCES workspace_leases,
    job_id TEXT NOT NULL, piece_id INTEGER NOT NULL, backend TEXT NOT NULL,
    lease_generation INTEGER NOT NULL, lease_owner TEXT NOT NULL,
    created_at REAL NOT NULL, review_after REAL NOT NULL,
    status TEXT NOT NULL, finished_at REAL, measured_at REAL,
    measurement_error TEXT,
    PRIMARY KEY(volume_id,contract_id),
    UNIQUE(volume_id,job_id,piece_id,backend,lease_generation))"""


def external_directory(path):
    """Refuse repository storage and redirected components, including junctions."""
    raw = Path(path).absolute()
    resolved = raw.resolve()
    if raw != resolved:
        raise workspace.Refused('compile storage must not use a redirected path')
    for parent in (raw, *raw.parents):
        if (parent / '.git').exists():
            raise workspace.Refused('compile storage must be outside repository worktrees')
        if parent.exists():
            info = parent.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise workspace.Refused('compile storage must not use links or reparse points')
    return resolved


def _regular_size(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise workspace.Refused('compile measurement encountered a link or reparse point')
    if stat.S_ISDIR(info.st_mode):
        with os.scandir(path) as entries:
            return sum(_regular_size(Path(entry.path)) for entry in entries)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise workspace.Refused('compile measurement requires private regular files')
    return info.st_size


class Artifacts:
    """Trusted site worker adapter over workspace_accounting.Store.

    Capacity, per-account quota, free-space floor, per-attempt reservation and
    review interval are explicit operator configuration. No invented defaults.
    Use the same Store/Accounts database and provisioned content/workspaces root.
    A review deadline is a reminder to review retention, NEVER deletion consent.
    """
    def __init__(self, state, store, *, reservation_bytes, review_after_s, workspace_root=None):
        self.store = store
        self.root = external_directory(store.root)
        expected = Path(workspace_root) if workspace_root is not None else Path(state.content_dir) / 'workspaces'
        if self.root != external_directory(expected) or store.db_path != Path(state.accounts_db).resolve():
            raise workspace.Refused('compile artifacts must use this state root and workspace ledger')
        self.reservation_bytes = workspace._bytes(reservation_bytes)
        if not self.reservation_bytes:
            raise ValueError('compile reservation must be positive')
        if isinstance(review_after_s, bool) or not isinstance(review_after_s, (float, int)) \
                or not math.isfinite(review_after_s) or review_after_s <= 0:
            raise ValueError('compile retention review interval must be positive and finite')
        self.review_after_s = float(review_after_s)
        self._next_monitor = 0.0
        with self._transaction():
            pass

    @contextmanager
    def _transaction(self):
        with self.store._transaction() as db:
            schema_version.declare(db, 'compile_artifacts', SCHEMA_VERSION)
            if not db.in_transaction:
                db.execute('BEGIN IMMEDIATE')
                schema_version.declare(db, 'compile_artifacts', SCHEMA_VERSION)
            db.execute(DDL)
            yield db

    def bind(self, db, jobs_dir):
        main = next((r[2] for r in db.execute('PRAGMA database_list') if r[1] == 'main'), None)
        if not main or Path(main).resolve() != self.store.db_path or external_directory(jobs_dir) != self.root:
            raise workspace.Refused('compile route and account ledger binding mismatch')

    def _run(self, db, contract_id):
        row = db.execute('SELECT a.*,w.owner_id,w.state AS workspace_state FROM compile_artifact_runs a '
                         'JOIN workspace_leases w USING(workspace_id) WHERE a.volume_id=? AND a.contract_id=?',
                         (self.store.volume_id, contract_id)).fetchone()
        if row is None:
            raise workspace.NotFound('compile run not found')
        return row

    def admit(self, claim, contract_id, source_digest):
        """Atomic account reservation and binding before any request is written."""
        if type(contract_id) is not int or not 0 < contract_id < 2**63:
            raise ValueError('invalid compile contract id')
        with self._transaction() as db:
            row = db.execute('SELECT j.owner_key FROM divider_jobs j JOIN divider_attempts a USING(job_id) '
                             "WHERE a.job_id=? AND a.piece_id=? AND a.backend=? AND a.state='leased' "
                             'AND a.lease_owner=? AND a.lease_generation=? AND a.lease_deadline>?',
                             (claim['job_id'], claim['piece_id'], claim['backend'], claim['worker_id'],
                              claim['generation'], time.time())).fetchone()
            match = re.fullmatch(r'account:([1-9][0-9]*)', row['owner_key']) if row is not None else None
            if match is None:
                raise workspace.Refused('site compile requires a current account-owned divider lease')
            owner = int(match[1])
            self.store._user(db, owner)
            previous = db.execute('SELECT * FROM compile_artifact_runs WHERE volume_id=? AND contract_id=?',
                                  (self.store.volume_id, contract_id)).fetchone()
            if previous is not None:
                if any(previous[k] != v for k, v in (('job_id', claim['job_id']), ('piece_id', claim['piece_id']),
                       ('backend', claim['backend']), ('lease_generation', claim['generation']), ('lease_owner', claim['worker_id']))):
                    raise workspace.Refused('compile contract already belongs to another attempt')
                held = self.store._owned(db, owner, previous['workspace_id'])
                if held['state'] not in ('reserved', 'active'):
                    raise workspace.Refused('compile workspace is being released')
                key = previous['workspace_id']
            else:
                key = self.store._reserve(db, owner, self.reservation_bytes, base_revision=source_digest)
                now = time.time()
                db.execute('INSERT INTO compile_artifact_runs '
                           '(volume_id,contract_id,workspace_id,job_id,piece_id,backend,lease_generation,lease_owner,created_at,review_after,status) '
                           'VALUES (?,?,?,?,?,?,?,?,?,?,?)', (self.store.volume_id, contract_id, key, claim['job_id'],
                           claim['piece_id'], claim['backend'], claim['generation'], claim['worker_id'], now,
                           now + self.review_after_s, 'prepared'))
            return {'owner_id': owner, 'workspace_id': key, 'job_id': claim['job_id'],
                    'job_directory': str(self.root / str(contract_id))}

    def reconcile(self, contract_id):
        """Sample all exact host-convention files, including failed/unknown runs.

        Mutating files are an observation, not an immutable snapshot. An unsafe
        or partial scan never replaces a previous byte count with zero.
        """
        external_directory(self.root)
        try:
            paths = [self.root / str(contract_id)]
            # The contract namespace includes interrupted atomic-write temps.
            paths += list(self.root.glob(f'{contract_id}.*'))
            amount = sum(_regular_size(path) for path in paths if path.exists() or path.is_symlink())
        except (OSError, workspace.WorkspaceError) as error:
            with self._transaction() as db:
                self._run(db, contract_id)
                db.execute('UPDATE compile_artifact_runs SET measurement_error=? WHERE volume_id=? AND contract_id=?',
                           (f'{type(error).__name__}: {error}'[:500], self.store.volume_id, contract_id))
            return False
        with self._transaction() as db:
            row = self._run(db, contract_id)
            self.store._record_live_usage(db, row['owner_id'], row['workspace_id'], amount)
            db.execute('UPDATE compile_artifact_runs SET measured_at=?,measurement_error=NULL WHERE volume_id=? AND contract_id=?',
                       (time.time(), self.store.volume_id, contract_id))
        return True

    def status(self, contract_id, status, *, finished=False):
        if status not in ('prepared', 'submitted', 'unfinished', 'complete', 'refused', 'error', 'lost-lease'):
            raise ValueError('unknown compile lifecycle status')
        with self._transaction() as db:
            self._run(db, contract_id)
            db.execute('UPDATE compile_artifact_runs SET status=?,finished_at=? WHERE volume_id=? AND contract_id=?',
                       (status, time.time() if finished else None, self.store.volume_id, contract_id))
        self.reconcile(contract_id)

    def monitor(self, *, force=False):
        """Refresh retained files on service work, including interrupted runs."""
        if not force and time.monotonic() < self._next_monitor:
            return
        self._next_monitor = time.monotonic() + 60.0
        with self._transaction() as db:
            ids = [r[0] for r in db.execute('SELECT a.contract_id FROM compile_artifact_runs a '
                   'JOIN workspace_leases w USING(workspace_id) WHERE a.volume_id=? '
                   "AND w.state IN ('reserved','active','releasing')", (self.store.volume_id,))]
        for contract_id in ids:
            self.reconcile(contract_id)


def job_storage(db, job_id, owner_key):
    """Read-only owner-bound view for the existing page; no schema creation.

    Call after the existing divider authorization. Admin visibility does not
    transfer ownership. Host paths and compiler diagnostics are not exposed.
    """
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='compile_artifact_runs'").fetchone():
        return []
    rows = db.execute('SELECT a.contract_id,a.workspace_id,a.status,a.created_at,a.review_after,a.finished_at,a.measured_at,'
                      'a.measurement_error,w.reserved_bytes,w.measured_bytes,w.state AS retention_state '
                      'FROM compile_artifact_runs a JOIN workspace_leases w USING(workspace_id) '
                      "WHERE a.job_id=? AND ('account:' || w.owner_id)=? ORDER BY a.created_at", (job_id, owner_key)).fetchall()
    return [{**dict(row), 'contract_id': str(row['contract_id']),
             'measurement_error': 'measurement requires attention' if row['measurement_error'] else None,
             'review_due': row['review_after'] <= time.time(),
             'charged_bytes': max(row['reserved_bytes'], row['measured_bytes'])} for row in rows]
