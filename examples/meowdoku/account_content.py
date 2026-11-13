"""Account groups and reusable read-sharing policy, independent of content storage.

Content services retain their own authoritative owner and data. They call these
policy functions in the same SQL transaction as their content read/write. No
request may supply the owner, actor, content kind, or filesystem path. Public
means anonymous Internet readers; it is an explicit owner sign-off, not a role.
"""
from contextlib import contextmanager
from pathlib import Path
import re
import sqlite3
import time
import uuid

from meowdoku import schema_version

SCHEMA_VERSION = 1
ID = re.compile(r'[0-9a-f]{32}\Z')
SCOPES = ('personal', 'group', 'public')


class AccessError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


DDL = (
    """CREATE TABLE IF NOT EXISTS account_content_groups (
        id TEXT PRIMARY KEY, owner_id INTEGER NOT NULL REFERENCES users(id),
        name TEXT NOT NULL, request_id TEXT NOT NULL, created_at REAL NOT NULL,
        revision INTEGER NOT NULL DEFAULT 0, UNIQUE(owner_id,request_id))""",
    """CREATE TABLE IF NOT EXISTS account_content_members (
        group_id TEXT NOT NULL REFERENCES account_content_groups(id),
        user_id INTEGER NOT NULL REFERENCES users(id),
        role TEXT NOT NULL CHECK(role IN ('admin','member')),
        granted_by INTEGER NOT NULL REFERENCES users(id), updated_at REAL NOT NULL,
        PRIMARY KEY(group_id,user_id))""",
    """CREATE TABLE IF NOT EXISTS account_content_invitations (
        id TEXT NOT NULL UNIQUE, group_id TEXT NOT NULL REFERENCES account_content_groups(id),
        user_id INTEGER NOT NULL REFERENCES users(id), invited_by INTEGER NOT NULL REFERENCES users(id),
        request_message TEXT NOT NULL, created_at REAL NOT NULL,
        PRIMARY KEY(group_id,user_id))""",
    """CREATE TABLE IF NOT EXISTS account_content_access (
        content_kind TEXT NOT NULL, content_key TEXT NOT NULL,
        visibility TEXT NOT NULL CHECK(visibility IN ('personal','group','public')),
        group_id TEXT REFERENCES account_content_groups(id), revision INTEGER NOT NULL,
        signed_off_by INTEGER REFERENCES users(id), signed_off_at REAL,
        updated_by INTEGER NOT NULL REFERENCES users(id), updated_at REAL NOT NULL,
        CHECK((visibility='group')=(group_id IS NOT NULL)),
        CHECK((visibility='public')=(signed_off_by IS NOT NULL)),
        CHECK((signed_off_by IS NULL)=(signed_off_at IS NULL)),
        PRIMARY KEY(content_kind,content_key))""",
    """CREATE TABLE IF NOT EXISTS account_content_audit (
        id INTEGER PRIMARY KEY, actor_id INTEGER NOT NULL REFERENCES users(id),
        event TEXT NOT NULL, target TEXT NOT NULL, detail TEXT NOT NULL, created_at REAL NOT NULL)""",
    'CREATE INDEX IF NOT EXISTS account_content_members_user ON account_content_members(user_id,group_id)',
)


def prepare(db):
    # Shared schema_version.declare can commit. Call before all business writes,
    # then reacquire and recheck so revocations and content changes stay atomic.
    schema_version.declare(db, 'account_content', SCHEMA_VERSION)
    if not db.in_transaction:
        db.execute('BEGIN IMMEDIATE')
        schema_version.declare(db, 'account_content', SCHEMA_VERSION)
    for sql in DDL:
        db.execute(sql)


def administrator(db, actor):
    if actor is None or not db.execute("SELECT 1 FROM sqlite_master WHERE name='account_roles' AND type='table'").fetchone():
        return False
    return db.execute("SELECT 1 FROM account_roles WHERE user_id=? AND role='admin'", (actor,)).fetchone() is not None


def group_role(db, group_id, actor):
    if actor is None:
        return None
    group = db.execute('SELECT owner_id FROM account_content_groups WHERE id=?', (group_id,)).fetchone()
    if group is None:
        return None
    if group['owner_id'] == actor:
        return 'owner'
    member = db.execute('SELECT role FROM account_content_members WHERE group_id=? AND user_id=?',
                        (group_id, actor)).fetchone()
    return None if member is None else member['role']


def policy(db, kind, key):
    row = db.execute('SELECT visibility,group_id,revision,signed_off_by,signed_off_at FROM account_content_access '
                     'WHERE content_kind=? AND content_key=?', (kind, key)).fetchone()
    return dict(row) if row else dict(visibility='personal', group_id=None, revision=0,
                                     signed_off_by=None, signed_off_at=None)


def can_read(db, kind, key, owner, actor, *, allow_admin=False):
    if actor is not None and (actor == owner or (allow_admin and administrator(db, actor))):
        return True
    access = policy(db, kind, key)
    return access['visibility'] == 'public' or (access['visibility'] == 'group'
           and group_role(db, access['group_id'], actor) is not None)


def set_access(db, kind, key, owner, actor, body):
    """Called only after the resource service resolves its existing owner."""
    if type(actor) is not int or actor <= 0 or actor != owner:
        raise AccessError('content not found', 404)
    if not isinstance(body, dict) or set(body) != {'visibility','group_id','expected_revision','sign_off'}:
        raise AccessError('expected visibility, group_id, expected_revision and sign_off')
    visibility, group, expected = body['visibility'], body['group_id'], body['expected_revision']
    if (visibility not in SCOPES or type(expected) is not int or expected < 0
            or type(body['sign_off']) is not bool):
        raise AccessError('invalid sharing policy')
    if visibility == 'group':
        if not isinstance(group, str) or not ID.fullmatch(group) or group_role(db, group, actor) is None:
            raise AccessError('group not found', 404)
    elif group is not None:
        raise AccessError('group_id must be null outside group visibility')
    if body['sign_off'] != (visibility == 'public'):
        raise AccessError('public requires sign_off=true; other scopes require false')
    old = policy(db, kind, key)
    if old['revision'] != expected:
        raise AccessError('sharing revision changed; read before retrying', 409)
    now = time.time()
    signed_by, signed_at = (actor, now) if visibility == 'public' else (None, None)
    db.execute('INSERT INTO account_content_access VALUES(?,?,?,?,?,?,?,?,?) '
               'ON CONFLICT(content_kind,content_key) DO UPDATE SET visibility=excluded.visibility,'
               'group_id=excluded.group_id,revision=excluded.revision,signed_off_by=excluded.signed_off_by,'
               'signed_off_at=excluded.signed_off_at,updated_by=excluded.updated_by,updated_at=excluded.updated_at',
               (kind,key,visibility,group,expected+1,signed_by,signed_at,actor,now))
    db.execute('INSERT INTO account_content_audit(actor_id,event,target,detail,created_at) VALUES(?,?,?,?,?)',
               (actor,'sharing',kind+':'+key,visibility + (':' + group if group else ''),now))
    return policy(db, kind, key)


class Groups:
    def __init__(self, accounts):
        self.accounts = accounts
        with accounts.lock:
            rows = accounts._conn.execute('PRAGMA database_list').fetchall()
        filename = next((r[2] for r in rows if r[1] == 'main'), None)
        if not filename:
            raise AccessError('file-backed Accounts database required',503)
        self.path = Path(filename).resolve(strict=True)

    @contextmanager
    def transaction(self, token):
        if not isinstance(token, str) or not token or len(token) > 4096:
            raise AccessError('sign in to use account groups',401)
        account = self.accounts.user_for_token(token)
        if account is None:
            raise AccessError('sign in to use account groups',401)
        db = sqlite3.connect(self.path.as_uri()+'?mode=rw',uri=True,timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('BEGIN IMMEDIATE')
            prepare(db)
            current=db.execute('SELECT user_id FROM sessions WHERE token=? AND expires_at>=?', (token,time.time())).fetchone()
            if current is None or current['user_id'] != account['id']:
                raise AccessError('sign in to use account groups',401)
            yield db, account['id']
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def listing(self, token):
        with self.transaction(token) as (db, actor):
            rows=db.execute('SELECT g.id,g.name,g.owner_id,g.created_at,g.revision FROM account_content_groups g '
                            'WHERE g.owner_id=? OR EXISTS(SELECT 1 FROM account_content_members m WHERE m.group_id=g.id AND m.user_id=?) '
                            'ORDER BY g.created_at DESC,g.id LIMIT 101',(actor,actor)).fetchall()
            if len(rows)>100:
                raise AccessError('too many memberships to list',409)
            return {'groups':[dict(r,role=group_role(db,r['id'],actor)) for r in rows]}

    def create(self, token, body):
        if not isinstance(body,dict) or set(body) != {'name','request_id'}:
            raise AccessError('expected name and request_id')
        name,key=body['name'],body['request_id']
        if (not isinstance(name,str) or not name.strip() or len(name)>100 or
                not isinstance(key,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',key)):
            raise AccessError('invalid group name or request_id')
        with self.transaction(token) as (db,actor):
            old=db.execute('SELECT id,name FROM account_content_groups WHERE owner_id=? AND request_id=?',(actor,key)).fetchone()
            if old:
                if old['name'] != name:
                    raise AccessError('request_id reused with different group',409)
                return {'id':old['id'],'created':False}
            if db.execute('SELECT count(*) FROM account_content_groups WHERE owner_id=?',(actor,)).fetchone()[0]>=32:
                raise AccessError('group limit reached',429)
            identity=uuid.uuid4().hex; now=time.time()
            db.execute('INSERT INTO account_content_groups VALUES(?,?,?,?,?,0)',(identity,actor,name,key,now))
            db.execute('INSERT INTO account_content_audit(actor_id,event,target,detail,created_at) VALUES(?,?,?,?,?)',
                       (actor,'group-create',identity,name,now))
            return {'id':identity,'created':True}

    def examine(self,token,group_id):
        with self.transaction(token) as (db,actor):
            role=group_role(db,group_id,actor)
            if role is None:
                raise AccessError('group not found',404)
            group=dict(db.execute('SELECT id,name,owner_id,created_at,revision FROM account_content_groups WHERE id=?',(group_id,)).fetchone())
            group['role']=role
            # Membership is visible only to the group itself, never public enumeration.
            owner=db.execute('SELECT id AS user_id,username FROM users WHERE id=?',(group['owner_id'],)).fetchone()
            group['members']=[dict(owner,role='owner')]+[dict(r) for r in db.execute('SELECT m.user_id,u.username,m.role FROM account_content_members m '
                            'JOIN users u ON u.id=m.user_id WHERE m.group_id=? ORDER BY u.username',(group_id,))]
            return {'group':group}

    def member(self,token,group_id,body):
        if not isinstance(body,dict) or set(body) != {'username','role','expected_revision'}:
            raise AccessError('expected username, role and expected_revision')
        username, role, expected = body['username'],body['role'],body['expected_revision']
        if (not isinstance(username,str) or not 1<=len(username)<=64 or role not in ('admin','member',None)
                or type(expected) is not int or expected<0):
            raise AccessError('invalid group membership')
        with self.transaction(token) as (db,actor):
            actor_role=group_role(db,group_id,actor)
            if actor_role is None:
                raise AccessError('group not found',404)
            if actor_role not in ('owner','admin'):
                raise AccessError('group administrator required',403)
            group=db.execute('SELECT * FROM account_content_groups WHERE id=?',(group_id,)).fetchone()
            if group['revision'] != expected:
                raise AccessError('group revision changed; read before retrying',409)
            target=db.execute('SELECT u.id FROM users u JOIN account_content_members m ON m.user_id=u.id '
                              'WHERE m.group_id=? AND u.username=? COLLATE NOCASE',(group_id,username)).fetchone()
            if target is None:
                raise AccessError('member not found',404)
            target=target['id']
            old_role=group_role(db,group_id,target)
            if target==group['owner_id']:
                raise AccessError('group owner cannot be changed by membership operations',403)
            if actor_role=='admin' and (role=='admin' or old_role=='admin'):
                raise AccessError('only group owner may change administrators',403)
            if role is not None and old_role is None:
                raise AccessError('invite and recipient acceptance required before membership',403)
            now=time.time()
            if old_role=='admin' and role!='admin':
                db.execute('DELETE FROM account_content_invitations WHERE group_id=? AND invited_by=?',(group_id,target))
            if role is None:
                db.execute('DELETE FROM account_content_members WHERE group_id=? AND user_id=?',(group_id,target))
            else:
                db.execute('INSERT INTO account_content_members VALUES(?,?,?,?,?) '
                           'ON CONFLICT(group_id,user_id) DO UPDATE SET role=excluded.role,granted_by=excluded.granted_by,updated_at=excluded.updated_at',
                           (group_id,target,role,actor,now))
            db.execute('UPDATE account_content_groups SET revision=revision+1 WHERE id=?',(group_id,))
            db.execute('INSERT INTO account_content_audit(actor_id,event,target,detail,created_at) VALUES(?,?,?,?,?)',
                       (actor,'group-member',group_id,str(target)+':'+str(role),now))
            return {'id':group_id,'revision':expected+1,'user_id':target,'role':role}


    def invite(self,token,group_id,body):
        if not isinstance(body,dict) or set(body) != {'username','request_message'}:
            raise AccessError('expected username and request_message')
        username,message=body['username'],body['request_message']
        if (not isinstance(username,str) or not 1<=len(username)<=64 or
                not isinstance(message,str) or not message.strip() or len(message)>2000):
            raise AccessError('nonempty bounded account and request message required')
        with self.transaction(token) as (db,actor):
            role=group_role(db,group_id,actor)
            if role is None:
                raise AccessError('group not found',404)
            if role not in ('owner','admin'):
                raise AccessError('group administrator required',403)
            target=db.execute('SELECT id FROM users WHERE username=? COLLATE NOCASE',(username,)).fetchone()
            # Opaque acknowledgment prevents invitations becoming an account directory.
            # Membership is not friendship and never changes pair-consent state.
            generic={'requested':True}
            if target is None or target['id']==actor:
                return generic
            target=target['id']
            if group_role(db,group_id,target) is not None:
                return generic
            old=db.execute('SELECT request_message FROM account_content_invitations WHERE group_id=? AND user_id=?',(group_id,target)).fetchone()
            if old and old['request_message']==message:
                return generic
            if not old and (db.execute('SELECT count(*) FROM account_content_invitations WHERE invited_by=?',(actor,)).fetchone()[0]>=20
                    or db.execute('SELECT count(*) FROM account_content_invitations WHERE user_id=?',(target,)).fetchone()[0]>=20):
                return generic
            now=time.time()
            db.execute('INSERT INTO account_content_invitations VALUES(?,?,?,?,?,?) '
                       'ON CONFLICT(group_id,user_id) DO UPDATE SET id=excluded.id,invited_by=excluded.invited_by,'
                       'request_message=excluded.request_message,created_at=excluded.created_at',
                       (uuid.uuid4().hex,group_id,target,actor,message,now))
            db.execute('INSERT INTO account_content_audit(actor_id,event,target,detail,created_at) VALUES(?,?,?,?,?)',
                       (actor,'group-invite',group_id,str(target),now))
            return generic

    def invitations(self,token):
        with self.transaction(token) as (db,actor):
            rows=db.execute('SELECT i.id,i.group_id,g.name,i.invited_by,u.username AS invited_by_username,i.request_message,i.created_at '
                            'FROM account_content_invitations i JOIN account_content_groups g ON g.id=i.group_id '
                            'JOIN users u ON u.id=i.invited_by WHERE i.user_id=? ORDER BY i.created_at DESC',(actor,)).fetchall()
            return {'invitations':[{k:r[k] for k in ('id','group_id','name','request_message','created_at')}
                    for r in rows if group_role(db,r['group_id'],r['invited_by']) in ('owner','admin')]}

    def respond(self,token,invitation_id,body):
        if not isinstance(body,dict) or set(body)!={'accept'} or type(body['accept']) is not bool:
            raise AccessError('expected boolean accept')
        with self.transaction(token) as (db,actor):
            row=db.execute('SELECT * FROM account_content_invitations WHERE id=? AND user_id=?',(invitation_id,actor)).fetchone()
            if row is None:
                raise AccessError('invitation not found',404)
            group_id=row['group_id']
            if body['accept']:
                if group_role(db,group_id,row['invited_by']) not in ('owner','admin'):
                    raise AccessError('invitation no longer available',409)
                if db.execute('SELECT count(*) FROM account_content_members WHERE group_id=?',(group_id,)).fetchone()[0]>=100:
                    raise AccessError('group member limit reached',429)
                if db.execute('SELECT count(*) FROM account_content_members WHERE user_id=?',(actor,)).fetchone()[0]>=64:
                    raise AccessError('account membership limit reached',429)
                db.execute('INSERT INTO account_content_members VALUES(?,?,?,?,?) ON CONFLICT DO NOTHING',
                           (group_id,actor,'member',row['invited_by'],time.time()))
                db.execute('UPDATE account_content_groups SET revision=revision+1 WHERE id=?',(group_id,))
            db.execute('DELETE FROM account_content_invitations WHERE id=?',(invitation_id,))
            db.execute('INSERT INTO account_content_audit(actor_id,event,target,detail,created_at) VALUES(?,?,?,?,?)',
                       (actor,'group-response',group_id,'accepted' if body['accept'] else 'declined',time.time()))
            return {'id':group_id,'accepted':body['accept']}

    def leave(self,token,group_id,body):
        if body != {}:
            raise AccessError('leave accepts an empty object')
        with self.transaction(token) as (db,actor):
            role=group_role(db,group_id,actor)
            if role is None:
                raise AccessError('group not found',404)
            if role=='owner':
                raise AccessError('owner cannot leave the owned group',403)
            db.execute('DELETE FROM account_content_invitations WHERE group_id=? AND invited_by=?',(group_id,actor))
            db.execute('DELETE FROM account_content_members WHERE group_id=? AND user_id=?',(group_id,actor))
            db.execute('DELETE FROM account_content_invitations WHERE group_id=? AND user_id=?',(group_id,actor))
            db.execute('UPDATE account_content_groups SET revision=revision+1 WHERE id=?',(group_id,))
            db.execute('INSERT INTO account_content_audit(actor_id,event,target,detail,created_at) VALUES(?,?,?,?,?)',
                       (actor,'group-leave',group_id,'left',time.time()))
            return {'left':True}
