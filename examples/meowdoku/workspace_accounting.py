"""Offline workspace reservations and immutable local review receipts.

This is a service-side SQL foundation, NOT a filesystem sandbox or HTTP API.
Callers must authenticate actor_id through Accounts, never accept it from a body.
The trusted storage worker alone reports usage/revisions and confirms removal.
Bounded explicit review bytes may be retained in SQL; no host-path reads, secret
issuance, Git operations or executable submissions are performed here.
Approval is a review decision, never an application/merge receipt.
"""
from __future__ import annotations

import re
import base64
import hashlib
import json
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

try:
    from meowdoku import schema_version as _schema_version
except ImportError:  # Same sibling-loader convention as authz.py.
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location('meowdoku_schema_version', Path(__file__).with_name('schema_version.py'))
    _schema_version = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_schema_version)

SCHEMA_VERSION = 1

MAX_INT = (1 << 63) - 1
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class WorkspaceError(Exception):
    pass


class NotFound(WorkspaceError):
    """Identical response for missing objects and another user's objects."""


class Refused(WorkspaceError):
    pass


def _bytes(value):
    if type(value) is not int or not 0 <= value <= MAX_INT:
        raise ValueError("byte counts must be nonnegative signed 64-bit integers")
    return value


def _digest(value):
    if not isinstance(value, str) or not DIGEST.fullmatch(value):
        raise ValueError("expected a SHA-256 revision")
    return value


MAX_SNAPSHOT_BYTES = 1024 * 1024
MAX_SNAPSHOT_FILES = 128


def _snapshot_payload(owner_id, workspace_id, base_revision, files):
    """Canonical bounded bytes, never filesystem paths to open or extract."""
    if type(files) is not dict or not 1 <= len(files) <= MAX_SNAPSHOT_FILES:
        raise ValueError('snapshot requires 1..128 explicit files')
    rows, names, total = [], set(), 0
    for name, content in files.items():
        if not isinstance(name, str) or not name.isascii() or not 1 <= len(name) <= 240:
            raise ValueError('snapshot paths must be bounded portable ASCII')
        parts = name.split('/')
        for part in parts:
            stem = part.split('.')[0].upper()
            if (not part or part in ('.', '..') or part.endswith((' ', '.'))
                    or any(ord(c) < 32 or c in '\\:<>"|?*' for c in part)
                    or stem in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}
                    or part.casefold() == '.git'):
                raise ValueError('unsafe or repository-metadata snapshot path')
        key = name.casefold()
        if key in names:
            raise ValueError('case-colliding snapshot paths')
        names.add(key)
        if type(content) is not bytes:
            raise ValueError('snapshot content must be immutable bytes')
        total += len(content)
        if total > MAX_SNAPSHOT_BYTES:
            raise ValueError('snapshot exceeds 1 MiB input limit')
        rows.append({'path':name, 'sha256':hashlib.sha256(content).hexdigest(),
                     'bytes':len(content), 'base64':base64.b64encode(content).decode('ascii')})
    for name in names:
        parts = name.split('/')
        if any('/'.join(parts[:i]) in names for i in range(1,len(parts))):
            raise ValueError('file/directory path collision')
    value = {'schema':'workspace-review-snapshot-v1','owner_id':owner_id,
             'workspace_id':workspace_id,'base_revision':base_revision,
             'files':sorted(rows,key=lambda r:r['path'])}
    payload = json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii')
    return hashlib.sha256(payload).hexdigest(), payload


DDL = (
    """CREATE TABLE IF NOT EXISTS workspace_volumes (
        volume_id TEXT PRIMARY KEY, root TEXT NOT NULL UNIQUE,
        capacity_bytes INTEGER NOT NULL, user_quota_bytes INTEGER NOT NULL,
        free_floor_bytes INTEGER NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS workspace_leases (
        workspace_id TEXT PRIMARY KEY, volume_id TEXT NOT NULL REFERENCES workspace_volumes,
        owner_id INTEGER NOT NULL REFERENCES users(id), base_revision TEXT NOT NULL,
        revision TEXT, reserved_bytes INTEGER NOT NULL CHECK(reserved_bytes >= 0),
        measured_bytes INTEGER NOT NULL DEFAULT 0 CHECK(measured_bytes >= 0),
        generation INTEGER NOT NULL DEFAULT 0,
        state TEXT NOT NULL CHECK(state IN ('reserved','active','releasing','released')),
        created_at REAL NOT NULL, updated_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS workspace_review_snapshots (
        workspace_id TEXT NOT NULL REFERENCES workspace_leases, snapshot_digest TEXT NOT NULL,
        byte_count INTEGER NOT NULL CHECK(byte_count >= 0), payload BLOB NOT NULL,
        PRIMARY KEY(workspace_id,snapshot_digest))""",
    """CREATE TABLE IF NOT EXISTS workspace_review_tickets (
        ticket_id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspace_leases,
        owner_id INTEGER NOT NULL REFERENCES users(id), base_revision TEXT NOT NULL,
        snapshot_digest TEXT NOT NULL, snapshot_bytes INTEGER NOT NULL CHECK(snapshot_bytes >= 0),
        status TEXT NOT NULL CHECK(status IN ('open','approved','rejected')),
        created_at REAL NOT NULL, reviewed_at REAL, reviewer_id INTEGER REFERENCES users(id),
        note TEXT NOT NULL DEFAULT '', UNIQUE(workspace_id, snapshot_digest))""",
    """CREATE TABLE IF NOT EXISTS workspace_size_profiles (
        volume_id TEXT NOT NULL REFERENCES workspace_volumes, owner_id INTEGER NOT NULL REFERENCES users(id),
        profile_digest TEXT NOT NULL, checkout_bytes INTEGER NOT NULL CHECK(checkout_bytes >= 0),
        build_peak_bytes INTEGER CHECK(build_peak_bytes >= 0), measured_at REAL NOT NULL,
        PRIMARY KEY(volume_id,owner_id,profile_digest))""",
    """CREATE TABLE IF NOT EXISTS workspace_snapshot_retirements (
        ticket_id TEXT PRIMARY KEY REFERENCES workspace_review_tickets,
        snapshot_digest TEXT NOT NULL, retired_bytes INTEGER NOT NULL,
        retired_at REAL NOT NULL, retired_by INTEGER NOT NULL REFERENCES users(id),
        storage_kind TEXT NOT NULL CHECK(storage_kind IN ('sql','external')))""",
)


class Store:
    def __init__(self, db_path, workspace_root, *, capacity_bytes, user_quota_bytes,
                 free_floor_bytes):
        self.db_path = Path(db_path).resolve(strict=True)
        self.root = Path(workspace_root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("workspace root must already exist")
        # One accounting authority per filesystem in this accounts database.
        self.volume_id = str(self.root.stat().st_dev)
        if str(self.db_path.stat().st_dev) != self.volume_id:
            raise Refused('snapshot database and workspace root must share a filesystem')
        self.policy = tuple(_bytes(x) for x in
                            (capacity_bytes, user_quota_bytes, free_floor_bytes))
        with self._transaction() as db:
            row = db.execute("SELECT * FROM workspace_volumes WHERE volume_id=?", (self.volume_id,)).fetchone()
            if row is None:
                db.execute("INSERT INTO workspace_volumes VALUES (?,?,?,?,?)",
                           (self.volume_id, str(self.root), *self.policy))
            elif (row['root'], row['capacity_bytes'], row['user_quota_bytes'], row['free_floor_bytes']) != (str(self.root), *self.policy):
                raise Refused("volume already has a different accounting policy/root")

    @contextmanager
    def _transaction(self):
        # No journal-mode/security changes and no creation of a missing live database.
        db = sqlite3.connect(self.db_path.as_uri() + '?mode=rw', uri=True, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('BEGIN IMMEDIATE')
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='users'").fetchone():
                raise Refused('existing Accounts database required')
            # Shared declare can commit when registering/upgrading. No business
            # write precedes it. Reacquire and recheck before DDL or admission.
            _schema_version.declare(db,'workspace_accounting',SCHEMA_VERSION)
            if not db.in_transaction:
                db.execute('BEGIN IMMEDIATE')
                _schema_version.declare(db,'workspace_accounting',SCHEMA_VERSION)
            for statement in DDL:
                db.execute(statement)
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _user(db, actor_id):
        if type(actor_id) is not int or actor_id <= 0 or not db.execute('SELECT 1 FROM users WHERE id=?', (actor_id,)).fetchone():
            raise Refused('authenticated account required')

    def _owned(self, db, actor_id, workspace_id):
        self._user(db, actor_id)
        row = db.execute('SELECT * FROM workspace_leases WHERE workspace_id=? AND owner_id=? AND volume_id=?',
                         (workspace_id, actor_id, self.volume_id)).fetchone()
        if row is None:
            raise NotFound('workspace not found')
        return row

    def _admit(self, db, actor_id, extra):
        rows = db.execute("SELECT owner_id,reserved_bytes,measured_bytes FROM workspace_leases WHERE volume_id=? AND state!='released'",
                          (self.volume_id,)).fetchall()
        total = sum(max(r['reserved_bytes'], r['measured_bytes']) for r in rows)
        owned = sum(max(r['reserved_bytes'], r['measured_bytes']) for r in rows if r['owner_id'] == actor_id)
        future = sum(max(0, r['reserved_bytes'] - r['measured_bytes']) for r in rows)
        capacity, quota, floor = self.policy
        if total + extra > capacity or owned + extra > quota:
            raise Refused('workspace budget exhausted')
        if future + extra + floor > shutil.disk_usage(self.root).free:
            raise Refused('filesystem headroom exhausted')

    def record_size_profile(self, actor_id, profile_digest, *, checkout_bytes, build_peak_bytes=None):
        """Trusted profiler: None is explicitly unknown, never a zero estimate.

        This is measurement evidence, not permission to run/build. Reservations
        still require an explicit known positive budget; profiles do not create
        leases or turn an unknown peak into an admission estimate.
        """
        _digest(profile_digest)
        _bytes(checkout_bytes)
        if build_peak_bytes is not None:
            _bytes(build_peak_bytes)
        with self._transaction() as db:
            self._user(db,actor_id)
            db.execute('INSERT INTO workspace_size_profiles VALUES(?,?,?,?,?,?) ON CONFLICT(volume_id,owner_id,profile_digest) DO UPDATE SET checkout_bytes=excluded.checkout_bytes,build_peak_bytes=excluded.build_peak_bytes,measured_at=excluded.measured_at',
                       (self.volume_id,actor_id,profile_digest,checkout_bytes,build_peak_bytes,time.time()))

    def get_size_profile(self, actor_id, profile_digest):
        _digest(profile_digest)
        with self._transaction() as db:
            self._user(db,actor_id)
            row = db.execute('SELECT * FROM workspace_size_profiles WHERE volume_id=? AND owner_id=? AND profile_digest=?',(self.volume_id,actor_id,profile_digest)).fetchone()
            if row is None:
                raise NotFound('size profile not found')
            return dict(row)

    def reserve(self, actor_id, byte_count, *, base_revision):
        with self._transaction() as db:
            return self._reserve(db, actor_id, byte_count, base_revision=base_revision)

    def _reserve(self, db, actor_id, byte_count, *, base_revision):
        """Reserve inside a trusted service's existing transaction (same ledger)."""
        byte_count = _bytes(byte_count)
        if byte_count == 0:
            raise ValueError('reservation must be positive')
        _digest(base_revision)
        self._user(db, actor_id)
        self._admit(db, actor_id, byte_count)
        key, now = uuid.uuid4().hex, time.time()
        db.execute("INSERT INTO workspace_leases (workspace_id,volume_id,owner_id,base_revision,reserved_bytes,state,created_at,updated_at) VALUES (?,?,?,?,?,'reserved',?,?)",
                   (key, self.volume_id, actor_id, base_revision, byte_count, now, now))
        return key

    def _record_live_usage(self, db, actor_id, workspace_id, measured_bytes):
        """Trusted service transaction: mutable usage is NOT a sealed revision.

        Used by a running compile's monitor. Review submission still requires
        record_usage's separately sealed revision and generation contract.
        """
        _bytes(measured_bytes)
        row = self._owned(db, actor_id, workspace_id)
        if row['state'] not in ('reserved', 'active', 'releasing'):
            raise Refused('workspace is not writable')
        if measured_bytes < self._retained_bytes(db, workspace_id):
            raise Refused('retained review snapshots must remain accounted')
        db.execute("UPDATE workspace_leases SET measured_bytes=?,revision=NULL,generation=generation+1,state=CASE WHEN state='reserved' THEN 'active' ELSE state END,updated_at=? WHERE workspace_id=?",
                   (measured_bytes, time.time(), workspace_id))

    def get(self, actor_id, workspace_id):
        with self._transaction() as db:
            return dict(self._owned(db, actor_id, workspace_id))

    def list_owned(self, actor_id):
        with self._transaction() as db:
            self._user(db, actor_id)
            return [dict(r) for r in db.execute('SELECT * FROM workspace_leases WHERE owner_id=? AND volume_id=? ORDER BY created_at,workspace_id', (actor_id, self.volume_id))]

    def record_usage(self, actor_id, workspace_id, *, measured_bytes, revision, expected_generation):
        """Trusted reconciler only: usage includes retained review snapshots.

        Record overruns truthfully; never erase disk usage to make a quota pass.
        revision is a sealed, immutable snapshot prepared by the storage worker,
        not a client-provided digest of still-mutable files.
        """
        _bytes(measured_bytes)
        _digest(revision)
        with self._transaction() as db:
            row = self._owned(db, actor_id, workspace_id)
            if row['state'] not in ('reserved', 'active') or row['generation'] != expected_generation:
                raise Refused('workspace changed or is not writable')
            retained = self._retained_bytes(db,workspace_id)
            if measured_bytes < retained:
                raise Refused('retained review snapshots must remain accounted')
            db.execute("UPDATE workspace_leases SET measured_bytes=?,revision=?,generation=generation+1,state='active',updated_at=? WHERE workspace_id=?",
                       (measured_bytes, revision, time.time(), workspace_id))
            return row['generation'] + 1

    def submit_review(self, actor_id, workspace_id, *, snapshot_digest, snapshot_bytes, expected_generation):
        """Queue a trusted sealed snapshot already included in measured_bytes.

        The storage worker must keep its bytes until a future explicit retention
        workflow disposes them. No arbitrary filesystem path can be submitted.
        """
        _digest(snapshot_digest)
        _bytes(snapshot_bytes)
        with self._transaction() as db:
            row = self._owned(db, actor_id, workspace_id)
            if row['state'] != 'active' or row['generation'] != expected_generation or row['revision'] != snapshot_digest:
                raise Refused('submission does not match sealed workspace revision')
            old = db.execute('SELECT * FROM workspace_review_tickets WHERE workspace_id=? AND snapshot_digest=?', (workspace_id, snapshot_digest)).fetchone()
            if old:
                if db.execute('SELECT 1 FROM workspace_snapshot_retirements WHERE ticket_id=?',(old['ticket_id'],)).fetchone():
                    raise Refused('snapshot was retired; submit a new revision')
                if old['snapshot_bytes'] != snapshot_bytes:
                    raise Refused('retry changed snapshot size')
                return old['ticket_id']
            retained = self._retained_bytes(db,workspace_id)
            if retained + snapshot_bytes > row['measured_bytes']:
                raise Refused('snapshot bytes not accounted')
            key = uuid.uuid4().hex
            db.execute("INSERT INTO workspace_review_tickets (ticket_id,workspace_id,owner_id,base_revision,snapshot_digest,snapshot_bytes,status,created_at) VALUES (?,?,?,?,?,?,'open',?)",
                       (key, workspace_id, actor_id, row['base_revision'], snapshot_digest, snapshot_bytes, time.time()))
            return key

    def seal_and_submit(self, actor_id, workspace_id, files, *, expected_generation):
        """Trusted adapter: atomically retain explicit bytes and their review ticket.

        This stores a bounded review bundle, not a runnable system archive. Never
        pass server files or another owner's content; no host paths are read here.
        Stored encoded bytes are charged, not just the smaller source byte total.
        """
        with self._transaction() as db:
            row = self._owned(db, actor_id, workspace_id)
            digest, payload = _snapshot_payload(actor_id,workspace_id,row['base_revision'],files)
            old = db.execute('SELECT ticket_id FROM workspace_review_tickets WHERE workspace_id=? AND snapshot_digest=?',
                             (workspace_id,digest)).fetchone()
            if old:
                retained = db.execute('SELECT payload FROM workspace_review_snapshots WHERE workspace_id=? AND snapshot_digest=?', (workspace_id,digest)).fetchone()
                if retained is None or bytes(retained['payload']) != payload:
                    raise Refused('ticket does not retain the exact snapshot bytes')
                return old['ticket_id']
            if row['state'] not in ('reserved','active') or row['generation'] != expected_generation:
                raise Refused('workspace changed or is not writable')
            measured = _bytes(row['measured_bytes'] + len(payload))
            extra = max(0, measured - max(row['reserved_bytes'],row['measured_bytes']))
            self._admit(db,actor_id,extra)
            key, now = uuid.uuid4().hex,time.time()
            db.execute('INSERT INTO workspace_review_snapshots VALUES(?,?,?,?)',
                       (workspace_id,digest,len(payload),payload))
            db.execute("INSERT INTO workspace_review_tickets (ticket_id,workspace_id,owner_id,base_revision,snapshot_digest,snapshot_bytes,status,created_at) VALUES (?,?,?,?,?,?,'open',?)",
                       (key,workspace_id,actor_id,row['base_revision'],digest,len(payload),now))
            db.execute("UPDATE workspace_leases SET measured_bytes=?,revision=?,generation=generation+1,state='active',updated_at=? WHERE workspace_id=?",
                       (measured,digest,now,workspace_id))
            return key

    def read_snapshot(self, actor_id, ticket_id):
        """Owner-only retained bytes; no filename lookup or host-path exposure."""
        with self._transaction() as db:
            self._user(db,actor_id)
            row = db.execute('SELECT s.payload,s.snapshot_digest FROM workspace_review_tickets t JOIN workspace_leases w USING(workspace_id) JOIN workspace_review_snapshots s ON s.workspace_id=t.workspace_id AND s.snapshot_digest=t.snapshot_digest WHERE t.ticket_id=? AND t.owner_id=? AND w.volume_id=?',
                             (ticket_id,actor_id,self.volume_id)).fetchone()
            if row is None:
                raise NotFound('snapshot not found')
            payload = bytes(row['payload'])
            if hashlib.sha256(payload).hexdigest() != row['snapshot_digest']:
                raise Refused('stored snapshot integrity mismatch')
            return payload

    def reviewer_snapshot(self, actor_id, ticket_id):
        """An administrator may inspect the exact submitted bundle before deciding."""
        with self._transaction() as db:
            self._admin(db,actor_id)
            row = db.execute('SELECT s.payload,s.snapshot_digest FROM workspace_review_tickets t JOIN workspace_leases w USING(workspace_id) JOIN workspace_review_snapshots s ON s.workspace_id=t.workspace_id AND s.snapshot_digest=t.snapshot_digest WHERE t.ticket_id=? AND w.volume_id=?',
                             (ticket_id,self.volume_id)).fetchone()
            if row is None:
                raise NotFound('snapshot not found')
            payload = bytes(row['payload'])
            if hashlib.sha256(payload).hexdigest() != row['snapshot_digest']:
                raise Refused('stored snapshot integrity mismatch')
            return payload

    @staticmethod
    def _admin(db, actor_id):
        Store._user(db, actor_id)
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='account_roles'").fetchone():
            raise Refused('administrator required')
        if not db.execute("SELECT 1 FROM account_roles WHERE user_id=? AND role='admin'", (actor_id,)).fetchone():
            raise Refused('administrator required')

    def review_queue(self, actor_id):
        with self._transaction() as db:
            self._admin(db, actor_id)
            return [dict(r) for r in db.execute("SELECT t.* FROM workspace_review_tickets t JOIN workspace_leases w USING(workspace_id) WHERE w.volume_id=? AND t.status='open' ORDER BY t.created_at,t.ticket_id", (self.volume_id,))]

    def get_ticket(self, actor_id, ticket_id):
        with self._transaction() as db:
            self._user(db, actor_id)
            row = db.execute('SELECT t.* FROM workspace_review_tickets t JOIN workspace_leases w USING(workspace_id) WHERE t.ticket_id=? AND t.owner_id=? AND w.volume_id=?', (ticket_id, actor_id, self.volume_id)).fetchone()
            if row is None:
                raise NotFound('ticket not found')
            return dict(row)

    def decide(self, actor_id, ticket_id, *, snapshot_digest, decision, note=''):
        _digest(snapshot_digest)
        if decision not in ('approved', 'rejected') or not isinstance(note, str) or len(note) > 4000:
            raise ValueError('invalid review decision')
        with self._transaction() as db:
            self._admin(db, actor_id)
            row = db.execute('SELECT t.* FROM workspace_review_tickets t JOIN workspace_leases w USING(workspace_id) WHERE t.ticket_id=? AND w.volume_id=?', (ticket_id, self.volume_id)).fetchone()
            if row is None:
                raise NotFound('ticket not found')
            if row['status'] != 'open' or row['snapshot_digest'] != snapshot_digest:
                raise Refused('review is stale or already decided')
            db.execute('UPDATE workspace_review_tickets SET status=?,reviewed_at=?,reviewer_id=?,note=? WHERE ticket_id=?',
                       (decision, time.time(), actor_id, note, ticket_id))

    @staticmethod
    def _retained_bytes(db, workspace_id):
        return db.execute('SELECT COALESCE(SUM(t.snapshot_bytes),0) FROM workspace_review_tickets t LEFT JOIN workspace_snapshot_retirements r USING(ticket_id) WHERE t.workspace_id=? AND r.ticket_id IS NULL',(workspace_id,)).fetchone()[0]

    def retire_snapshot(self, actor_id, ticket_id, *, snapshot_digest):
        """Admin-controlled retention action for closed SQL-backed reviews.

        Deletes retained payload, never the ticket/decision audit. Not exposed
        through SessionAccess. Real use requires explicit retention authorization.
        SQLite may reuse pages without shrinking the database; capacity admission
        still samples actual free space and never claims those bytes were freed.
        """
        return self._retire_snapshot(actor_id,ticket_id,snapshot_digest,external=False)

    def confirm_external_snapshot_removed(self, actor_id, ticket_id, *, snapshot_digest, expected_generation, measured_bytes):
        """Trusted cleanup receipt, after external snapshot removal is verified.

        Admin identity AND trusted worker provenance are required at integration;
        no ordinary owner/request dispatch exposes this method. Report the absolute
        measured total AFTER removal, never a byte delta, with the generation
        sampled before the measurement. Stale measurements refuse; resample
        rather than subtract again. No files deleted.
        """
        _bytes(measured_bytes)
        return self._retire_snapshot(actor_id,ticket_id,snapshot_digest,external=True,
                                     expected_generation=expected_generation,measured_after=measured_bytes)

    def _retire_snapshot(self, actor_id, ticket_id, digest, *, external, expected_generation=None, measured_after=None):
        _digest(digest)
        with self._transaction() as db:
            self._admin(db,actor_id)
            row = db.execute('SELECT t.*,w.measured_bytes,w.generation FROM workspace_review_tickets t JOIN workspace_leases w USING(workspace_id) WHERE t.ticket_id=? AND w.volume_id=?',(ticket_id,self.volume_id)).fetchone()
            if row is None:
                raise NotFound('ticket not found')
            if row['snapshot_digest'] != digest or row['status'] == 'open':
                raise Refused('review must be closed and match the retained snapshot')
            kind = 'external' if external else 'sql'
            old = db.execute('SELECT * FROM workspace_snapshot_retirements WHERE ticket_id=?',(ticket_id,)).fetchone()
            if old:
                if old['snapshot_digest'] != digest or old['storage_kind'] != kind:
                    raise Refused('retirement receipt mismatch')
                return old['retired_bytes']
            if external and row['generation'] != expected_generation:
                raise Refused('external cleanup measurement is stale')
            payload = db.execute('SELECT byte_count FROM workspace_review_snapshots WHERE workspace_id=? AND snapshot_digest=?',(row['workspace_id'],digest)).fetchone()
            if external == (payload is not None):
                raise Refused('retirement receipt does not match snapshot storage')
            amount = row['snapshot_bytes']
            if row['measured_bytes'] < amount or (payload is not None and payload['byte_count'] != amount):
                raise Refused('snapshot accounting mismatch')
            remaining = self._retained_bytes(db,row['workspace_id']) - amount
            after = measured_after if external else row['measured_bytes'] - amount
            if after < remaining:
                raise Refused('other retained snapshots must remain accounted')
            if not external:
                db.execute('DELETE FROM workspace_review_snapshots WHERE workspace_id=? AND snapshot_digest=?',(row['workspace_id'],digest))
            db.execute('INSERT INTO workspace_snapshot_retirements VALUES(?,?,?,?,?,?)',(ticket_id,digest,amount,time.time(),actor_id,kind))
            db.execute('UPDATE workspace_leases SET measured_bytes=?,revision=CASE WHEN revision=? THEN NULL ELSE revision END,generation=generation+1,updated_at=? WHERE workspace_id=?',(after,digest,time.time(),row['workspace_id']))
            return amount

    def begin_release(self, actor_id, workspace_id):
        with self._transaction() as db:
            row = self._owned(db, actor_id, workspace_id)
            if row['state'] in ('releasing', 'released'):
                return row['generation']
            if db.execute('SELECT 1 FROM workspace_review_tickets t LEFT JOIN workspace_snapshot_retirements r USING(ticket_id) WHERE t.workspace_id=? AND r.ticket_id IS NULL', (workspace_id,)).fetchone():
                raise Refused('retained review snapshots prevent release')
            db.execute("UPDATE workspace_leases SET state='releasing',generation=generation+1,updated_at=? WHERE workspace_id=?", (time.time(), workspace_id))
            return row['generation'] + 1

    def confirm_removed(self, actor_id, workspace_id, *, expected_generation):
        """Trusted cleanup worker only, after absence and no writers are verified.

        This does not delete files. Until this receipt, the full charge remains.
        """
        with self._transaction() as db:
            row = self._owned(db, actor_id, workspace_id)
            if row['state'] != 'releasing' or row['generation'] != expected_generation:
                raise Refused('removal receipt is stale')
            db.execute("UPDATE workspace_leases SET state='released',measured_bytes=0,reserved_bytes=0,generation=generation+1,updated_at=? WHERE workspace_id=?", (time.time(), workspace_id))

def _accounts_database(accounts):
    # Bind the already-open incumbent connection, not just a mutable path label.
    with accounts.lock:
        rows = accounts._conn.execute('PRAGMA database_list').fetchall()
    main = next((row[2] for row in rows if row[1] == 'main'), None)
    if not main:
        raise Refused('file-backed Accounts database required')
    return Path(main).resolve(strict=True)


class SessionAccess:
    """Narrow request-facing operations using the incumbent Accounts authority.

    Construct once with the server's Accounts and Store, then pass its cookie
    token per operation. No client user_id, role, path, measurement, snapshot
    evidence or cleanup confirmation is accepted. HTTP still owns method/CSRF,
    rate/size limits and error serialization; this class is not a route installer.
    """
    def __init__(self, accounts, store):
        if _accounts_database(accounts) != store.db_path:
            raise Refused('Accounts and workspace ledger must use the same database')
        self._accounts = accounts
        self._store = store

    def _actor(self, token):
        if not isinstance(token, str) or not token or len(token) > 4096:
            raise Refused('authenticated session required')
        account = self._accounts.user_for_token(token)
        if account is None:
            raise Refused('authenticated session required')
        return account['id']

    def list_owned(self, token):
        return self._store.list_owned(self._actor(token))

    def get(self, token, workspace_id):
        return self._store.get(self._actor(token), workspace_id)

    def reserve(self, token, byte_count, *, base_revision):
        return self._store.reserve(self._actor(token), byte_count, base_revision=base_revision)

    def begin_release(self, token, workspace_id):
        return self._store.begin_release(self._actor(token), workspace_id)

    def get_ticket(self, token, ticket_id):
        return self._store.get_ticket(self._actor(token), ticket_id)

    def review_queue(self, token):
        return self._store.review_queue(self._actor(token))

    def decide(self, token, ticket_id, *, snapshot_digest, decision, note=''):
        return self._store.decide(self._actor(token), ticket_id, snapshot_digest=snapshot_digest,
                                  decision=decision, note=note)


def for_state(state_paths, accounts, *, capacity_bytes, user_quota_bytes, free_floor_bytes, workspace_root=None):
    """Bind to the existing StatePaths without invoking ensure() or any ACL work.

    The state-root owner must provision content/workspaces first. This function
    never falls back to a checkout, creates directories, or opens another user's
    database. The shared schema-version gate runs before business operations.
    """
    db_path = Path(state_paths.accounts_db).resolve(strict=True)
    if _accounts_database(accounts) != db_path:
        raise Refused('Accounts does not belong to this state root')
    root = Path(workspace_root) if workspace_root is not None else Path(state_paths.content_dir) / 'workspaces'
    # Refuse a root symlink/junction: the state owner's provisioned directory is
    # the boundary. This does not replace sandboxing of future mutable contents.
    if root.is_symlink() or (root.exists() and root.resolve() != root.absolute()):
        raise Refused('workspace root must not redirect outside provisioned state')
    store = Store(db_path, root, capacity_bytes=capacity_bytes,
                  user_quota_bytes=user_quota_bytes, free_floor_bytes=free_floor_bytes)
    return store, SessionAccess(accounts, store)
