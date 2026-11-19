"""Account-owned incident tickets, separate from workspace snapshot reviews.

The existing Accounts session is the only caller identity. Administrators and operators are checked against account_roles in each transaction.
Assigned operators may read and update their tickets; only administrators assign the team.
No filesystem receipts, merges, role changes, or automatic issue resolution.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid

from meowdoku import schema_version

SCHEMA_VERSION = 2
MAX_ASSIGNEES = 10
MAX_BODY = 100_000
MAX_DIAGNOSTICS = 80_000
ID = re.compile(r'[0-9a-f]{32}\Z')
STATUSES = ('open', 'investigating', 'resolved')


class IssueError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


DDL = (
    """CREATE TABLE IF NOT EXISTS issue_tickets (
        id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL REFERENCES users(id),
        client_key TEXT NOT NULL, request_digest TEXT NOT NULL,
        title TEXT NOT NULL, description TEXT NOT NULL, diagnostics TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('open','investigating','resolved')),
        version INTEGER NOT NULL DEFAULT 1, created_at REAL NOT NULL, updated_at REAL NOT NULL,
        server_instance TEXT NOT NULL, server_log_seq INTEGER NOT NULL,
        UNIQUE(owner_id,client_key))""",
    """CREATE TABLE IF NOT EXISTS issue_ticket_events (
        id INTEGER PRIMARY KEY, ticket_id TEXT NOT NULL REFERENCES issue_tickets(id),
        actor_id INTEGER NOT NULL REFERENCES users(id), created_at REAL NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('open','investigating','resolved')),
        note TEXT NOT NULL, version INTEGER NOT NULL, UNIQUE(ticket_id,version))""",
    'CREATE INDEX IF NOT EXISTS issue_tickets_owner_created ON issue_tickets(owner_id,created_at)',
    'CREATE INDEX IF NOT EXISTS issue_ticket_events_actor_created ON issue_ticket_events(actor_id,created_at)',
    """CREATE TABLE IF NOT EXISTS issue_ticket_assignees (
        ticket_id TEXT NOT NULL REFERENCES issue_tickets(id), user_id INTEGER NOT NULL REFERENCES users(id),
        assigned_by INTEGER NOT NULL REFERENCES users(id), created_at REAL NOT NULL,
        PRIMARY KEY(ticket_id,user_id))""",
    'CREATE INDEX IF NOT EXISTS issue_ticket_assignees_user ON issue_ticket_assignees(user_id,ticket_id)',
)


def text(value, name, maximum, *, empty=False):
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        raise IssueError(f'{name} must be {1 if not empty else 0}..{maximum} characters')
    return value.strip()


class Service:
    def __init__(self, accounts):
        self.accounts = accounts
        # Follow the open incumbent connection, not a client path or mutable label.
        with accounts.lock:
            rows = accounts._conn.execute('PRAGMA database_list').fetchall()
        filename = next((row[2] for row in rows if row[1] == 'main'), None)
        if not filename:
            raise IssueError('file-backed Accounts database required', 503)
        self.path = Path(filename).resolve(strict=True)

    @contextmanager
    def transaction(self, token):
        if not isinstance(token, str) or not token or len(token) > 4096:
            raise IssueError('sign in to use issue tickets', 401)
        account = self.accounts.user_for_token(token)
        if account is None:
            raise IssueError('sign in to use issue tickets', 401)
        db = sqlite3.connect(self.path.as_uri() + '?mode=rw', uri=True, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('BEGIN IMMEDIATE')
            # Gate before our DDL. declare may commit registration; reacquire.
            schema_version.declare(db, 'issue_tickets', SCHEMA_VERSION)
            if not db.in_transaction:
                db.execute('BEGIN IMMEDIATE')
                schema_version.declare(db, 'issue_tickets', SCHEMA_VERSION)
            for statement in DDL:
                db.execute(statement)
            current = db.execute('SELECT user_id FROM sessions WHERE token=? AND expires_at>=?', (token,time.time())).fetchone()
            if current is None or current['user_id'] != account['id']:
                raise IssueError('sign in to use issue tickets', 401)
            admin = False
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='account_roles'").fetchone():
                admin = db.execute("SELECT 1 FROM account_roles WHERE user_id=? AND role='admin'", (account['id'],)).fetchone() is not None
            yield db, account['id'], admin
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def is_operator(db, actor):
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='account_roles'").fetchone():
            return False
        return db.execute("SELECT 1 FROM account_roles WHERE user_id=? AND role='operator'", (actor,)).fetchone() is not None

    @staticmethod
    def visible(db, actor, admin, ticket_id):
        if not isinstance(ticket_id,str) or not ID.fullmatch(ticket_id):
            raise IssueError('ticket not found',404)
        row=db.execute('SELECT * FROM issue_tickets WHERE id=?',(ticket_id,)).fetchone()
        if row is None:
            raise IssueError('ticket not found',404)
        if row['owner_id'] != actor and not admin:
            assigned = Service.is_operator(db, actor) and db.execute(
                'SELECT 1 FROM issue_ticket_assignees WHERE ticket_id=? AND user_id=?',
                (ticket_id, actor)).fetchone()
            if not assigned:
                raise IssueError('ticket not found',404)
        return row

    @staticmethod
    def assignees(db, ticket_id):
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='account_roles'").fetchone():
            return []
        return [dict(row) for row in db.execute(
            'SELECT users.id,users.username FROM issue_ticket_assignees '
            'JOIN users ON users.id=issue_ticket_assignees.user_id '
            'JOIN account_roles ON account_roles.user_id=users.id AND account_roles.role=\'operator\' '
            'WHERE issue_ticket_assignees.ticket_id=? ORDER BY users.username COLLATE NOCASE',
            (ticket_id,)).fetchall()]

    def create(self, token, payload, *, server_instance='', server_log_seq=0):
        if not isinstance(payload,dict) or set(payload) - {'title','description','diagnostics','client_key'}:
            raise IssueError('unexpected issue fields')
        title=text(payload.get('title'),'title',200)
        description=text(payload.get('description'),'description',12000)
        key=text(payload.get('client_key'),'client_key',64)
        if not re.fullmatch(r'[A-Za-z0-9_-]{16,64}',key):
            raise IssueError('invalid client_key')
        diagnostics=payload.get('diagnostics',{})
        if not isinstance(diagnostics,dict):
            raise IssueError('diagnostics must be an object')
        try:
            encoded=json.dumps(diagnostics,ensure_ascii=True,allow_nan=False,sort_keys=True,separators=(',',':'))
        except (ValueError,TypeError,RecursionError):
            raise IssueError('diagnostics must be bounded JSON') from None
        if len(encoded.encode()) > MAX_DIAGNOSTICS:
            raise IssueError('diagnostics exceed 80000 bytes',413)
        digest=hashlib.sha256(json.dumps([title,description,encoded],ensure_ascii=True).encode()).hexdigest()
        with self.transaction(token) as (db,actor,admin):
            old=db.execute('SELECT id,request_digest FROM issue_tickets WHERE owner_id=? AND client_key=?',(actor,key)).fetchone()
            if old:
                if old['request_digest'] != digest:
                    raise IssueError('submission key already used for different content',409)
                return {'id':old['id'],'created':False}
            now=time.time()
            if db.execute('SELECT COUNT(*) FROM issue_tickets WHERE owner_id=? AND created_at>?',(actor,now-600)).fetchone()[0] >= 10:
                raise IssueError('too many tickets; wait before submitting another',429)
            ticket_id=uuid.uuid4().hex
            db.execute('INSERT INTO issue_tickets VALUES(?,?,?,?,?,?,?,\'open\',1,?,?,?,?)',
                       (ticket_id,actor,key,digest,title,description,encoded,now,now,str(server_instance)[:200],max(0,int(server_log_seq))))
            db.execute('INSERT INTO issue_ticket_events(ticket_id,actor_id,created_at,status,note,version) VALUES(?,?,?,\'open\',\'Ticket created\',1)',(ticket_id,actor,now))
            return {'id':ticket_id,'created':True}

    def list(self, token, *, scope='mine', status='all', offset=0):
        if scope not in ('mine','assigned','all') or status not in (*STATUSES,'all','active') or type(offset) is not int or not 0 <= offset <= 1_000_000:
            raise IssueError('invalid list filter')
        with self.transaction(token) as (db,actor,admin):
            operator = self.is_operator(db,actor)
            if scope == 'all' and not admin:
                raise IssueError('administrator required',403)
            if scope == 'assigned' and not (operator or admin):
                raise IssueError('operator required',403)
            visibility = {
                'mine': 'owner_id=?',
                'assigned': 'EXISTS (SELECT 1 FROM issue_ticket_assignees WHERE ticket_id=issue_tickets.id AND user_id=?)',
                'all': '1=1',
            }[scope]
            params = (actor,) if scope != 'all' else ()
            rows=db.execute(
                'SELECT id,owner_id,title,status,version,created_at,updated_at FROM issue_tickets '
                'WHERE '+visibility+' AND (?=\'all\' OR (?=\'active\' AND status IN (\'open\',\'investigating\')) OR status=?) ORDER BY updated_at DESC,id DESC LIMIT 51 OFFSET ?',
                (*params,status,status,status,offset)).fetchall()
            tickets=[dict(r) for r in rows[:50]]
            for ticket in tickets:
                ticket['assignees']=self.assignees(db,ticket['id'])
            return {'tickets':tickets,'next_offset':offset+50 if len(rows)>50 else None,
                    'can_review_all':admin,'can_view_assigned':operator or admin}

    def get(self, token, ticket_id, *, before_version=None):
        if before_version is not None and (type(before_version) is not int or not 1 <= before_version <= (1 << 63)-1):
            raise IssueError('invalid history cursor')
        with self.transaction(token) as (db,actor,admin):
            row=dict(self.visible(db,actor,admin,ticket_id))
            row.pop('request_digest');row.pop('client_key')
            row['diagnostics']=json.loads(row['diagnostics'])
            row['assignees']=self.assignees(db,ticket_id)
            events=db.execute('SELECT actor_id,created_at,status,note,version FROM issue_ticket_events WHERE ticket_id=? AND (? IS NULL OR version<?) ORDER BY version DESC LIMIT 21',(ticket_id,before_version,before_version)).fetchall()
            row['events']=[dict(r) for r in reversed(events[:20])]
            row['next_before_version']=events[19]['version'] if len(events)>20 else None
            return {'ticket':row,'can_review_all':admin,'can_assign':admin,'max_assignees':MAX_ASSIGNEES}

    def update(self, token, ticket_id, payload):
        if not isinstance(payload,dict) or set(payload) != {'status','note','version'}:
            raise IssueError('expected status, note and version')
        status=payload['status'];version=payload['version']
        note=text(payload['note'],'note',4000)
        if status not in STATUSES or type(version) is not int or version < 1:
            raise IssueError('invalid status or version')
        with self.transaction(token) as (db,actor,admin):
            row=self.visible(db,actor,admin,ticket_id)
            if version != row['version']:
                raise IssueError('ticket changed; refresh before updating',409)
            now=time.time()
            from meowdoku import rate_limits
            if rate_limits.enabled() and db.execute('SELECT COUNT(*) FROM issue_ticket_events WHERE actor_id=? AND created_at>?',(actor,now-60)).fetchone()[0]>=30:
                raise IssueError('too many updates; wait before trying again',429)
            db.execute('UPDATE issue_tickets SET status=?,version=version+1,updated_at=? WHERE id=?',(status,now,ticket_id))
            db.execute('INSERT INTO issue_ticket_events(ticket_id,actor_id,created_at,status,note,version) VALUES(?,?,?,?,?,?)',(ticket_id,actor,now,status,note,version+1))
            return {'id':ticket_id,'status':status,'version':version+1}

    def operators(self, token):
        with self.transaction(token) as (db,actor,admin):
            if not admin:
                raise IssueError('administrator required',403)
            return {'operators':[dict(row) for row in db.execute(
                "SELECT users.id,users.username FROM users JOIN account_roles ON account_roles.user_id=users.id "
                "WHERE account_roles.role='operator' ORDER BY users.username COLLATE NOCASE").fetchall()],
                'max_assignees':MAX_ASSIGNEES}

    def assign(self, token, ticket_id, payload):
        if not isinstance(payload,dict) or set(payload) != {'operator_ids','version'}:
            raise IssueError('expected operator_ids and version')
        ids=payload['operator_ids'];version=payload['version']
        if not isinstance(ids,list) or len(ids)>MAX_ASSIGNEES or any(type(i) is not int or i<1 for i in ids) or len(set(ids))!=len(ids):
            raise IssueError(f'select at most {MAX_ASSIGNEES} distinct operators')
        if type(version) is not int or version<1:
            raise IssueError('invalid version')
        with self.transaction(token) as (db,actor,admin):
            if not admin:
                raise IssueError('administrator required',403)
            row=self.visible(db,actor,admin,ticket_id)
            if version!=row['version']:
                raise IssueError('ticket changed; refresh before assigning',409)
            selected=[]
            if ids:
                marks=','.join('?' for _ in ids)
                selected=db.execute(
                    "SELECT users.id,users.username FROM users JOIN account_roles ON account_roles.user_id=users.id "
                    f"WHERE account_roles.role='operator' AND users.id IN ({marks}) ORDER BY users.username COLLATE NOCASE",
                    tuple(ids)).fetchall()
                if len(selected)!=len(ids):
                    raise IssueError('select active operator accounts only')
            old={item['user_id'] for item in db.execute('SELECT user_id FROM issue_ticket_assignees WHERE ticket_id=?',(ticket_id,)).fetchall()}
            if old==set(ids):
                return {'id':ticket_id,'version':version,'assignees':self.assignees(db,ticket_id)}
            now=time.time()
            from meowdoku import rate_limits
            if rate_limits.enabled() and db.execute('SELECT COUNT(*) FROM issue_ticket_events WHERE actor_id=? AND created_at>?',(actor,now-60)).fetchone()[0]>=30:
                raise IssueError('too many updates; wait before trying again',429)
            db.execute('DELETE FROM issue_ticket_assignees WHERE ticket_id=?',(ticket_id,))
            db.executemany('INSERT INTO issue_ticket_assignees VALUES(?,?,?,?)',
                           [(ticket_id,user_id,actor,now) for user_id in ids])
            db.execute('UPDATE issue_tickets SET version=version+1,updated_at=? WHERE id=?',(now,ticket_id))
            names=', '.join('@'+item['username'] for item in selected) if selected else '(none)'
            db.execute('INSERT INTO issue_ticket_events(ticket_id,actor_id,created_at,status,note,version) VALUES(?,?,?,?,?,?)',
                       (ticket_id,actor,now,row['status'],'Ticket team: '+names,version+1))
            return {'id':ticket_id,'version':version+1,'assignees':self.assignees(db,ticket_id)}
