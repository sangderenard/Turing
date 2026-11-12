"""Disposable Accounts/HTTP tests. Never accesses a live service or credential."""
from pathlib import Path
import http.client
import json
import sqlite3
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'examples')]
from meowdoku import accounts, issue_tickets as issues, schema_version


def users(account):
    with account.lock:
        db=account._conn
        db.execute("INSERT INTO users(id,username,password_hash,created_at) VALUES(1,'alice','fixture',0),(2,'bob','fixture',0),(3,'admin','fixture',0),(4,'operator','fixture',0)")
        db.execute('CREATE TABLE IF NOT EXISTS account_roles(user_id INTEGER PRIMARY KEY,role TEXT,granted_by TEXT,created_at REAL)')
        db.execute("INSERT INTO account_roles VALUES(3,'admin','fixture',0),(4,'operator','fixture',0)")
        for i in range(1,5): db.execute('INSERT INTO sessions VALUES(?,?,0,9999999999)',('fixture-token-'+str(i),i))
        db.commit()


def payload(key='abcdefghijklmnop', **changes):
    return dict(title='Incident',description='Observed failure, not yet resolved.',diagnostics={'url':'/marble_studio.html'},client_key=key,**changes)


class IssueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='issue-tests-');self.addCleanup(self.temp.cleanup)
        self.accounts=accounts.Accounts(Path(self.temp.name)/'accounts.db');self.addCleanup(self.accounts._conn.close)
        users(self.accounts);self.service=issues.Service(self.accounts)

    def create(self): return self.service.create('fixture-token-1',payload())['id']

    def test_owner_admin_and_operator_isolation(self):
        ticket=self.create()
        self.assertEqual(self.service.get('fixture-token-1',ticket)['ticket']['owner_id'],1)
        self.assertEqual(self.service.list('fixture-token-2')['tickets'],[])
        for token in ['fixture-token-2','fixture-token-4']:
            with self.assertRaises(issues.IssueError) as error:self.service.get(token,ticket)
            self.assertEqual(error.exception.status,404)
            with self.assertRaises(issues.IssueError):self.service.list(token,scope='all')
        self.assertEqual(self.service.list('fixture-token-3',scope='all')['tickets'][0]['id'],ticket)
        self.assertEqual(self.service.get('fixture-token-3',ticket)['ticket']['id'],ticket)

    def test_idempotency_and_changed_content_conflict(self):
        ticket=self.create()
        self.assertEqual(self.service.create('fixture-token-1',payload()),{'id':ticket,'created':False})
        changed=payload();changed['description']='different'
        with self.assertRaises(issues.IssueError) as error:self.service.create('fixture-token-1',changed)
        self.assertEqual(error.exception.status,409)

    def test_resolution_reopen_and_history(self):
        ticket=self.create()
        self.service.update('fixture-token-1',ticket,{'status':'resolved','note':'Verified recovery','version':1})
        self.service.update('fixture-token-3',ticket,{'status':'open','note':'Regression observed','version':2})
        row=self.service.get('fixture-token-1',ticket)['ticket']
        self.assertEqual(row['status'],'open');self.assertEqual(row['version'],3)
        self.assertEqual([e['actor_id'] for e in row['events']],[1,1,3])
        self.assertEqual(row['description'],'Observed failure, not yet resolved.')

    def test_stale_version_and_owner_spoof_refused(self):
        ticket=self.create()
        self.service.update('fixture-token-1',ticket,{'status':'investigating','note':'Checking','version':1})
        with self.assertRaises(issues.IssueError) as error:self.service.update('fixture-token-1',ticket,{'status':'resolved','note':'stale','version':1})
        self.assertEqual(error.exception.status,409)
        bad=payload('differentkey00001');bad['owner_id']=2
        with self.assertRaises(issues.IssueError):self.service.create('fixture-token-1',bad)

    def test_admin_revocation_logout_and_expiry(self):
        ticket=self.create()
        with self.accounts.lock:
            self.accounts._conn.execute("UPDATE account_roles SET role='operator' WHERE user_id=3");self.accounts._conn.commit()
        with self.assertRaises(issues.IssueError):self.service.get('fixture-token-3',ticket)
        self.accounts.logout('fixture-token-1')
        with self.assertRaises(issues.IssueError) as error:self.service.get('fixture-token-1',ticket)
        self.assertEqual(error.exception.status,401)
        with self.accounts.lock:
            self.accounts._conn.execute("UPDATE sessions SET expires_at=0 WHERE user_id=2");self.accounts._conn.commit()
        with self.assertRaises(issues.IssueError):self.service.list('fixture-token-2')

    def test_atomic_audit_failure(self):
        ticket=self.create()
        with self.accounts.lock:
            self.accounts._conn.execute("CREATE TRIGGER fail_issue_audit BEFORE INSERT ON issue_ticket_events BEGIN SELECT RAISE(ABORT,'fixture'); END");self.accounts._conn.commit()
        with self.assertRaises(sqlite3.IntegrityError):self.service.update('fixture-token-1',ticket,{'status':'resolved','note':'attempt','version':1})
        self.assertEqual(self.service.get('fixture-token-1',ticket)['ticket']['version'],1)

    def test_schema_newer_refuses_without_issue_ddl(self):
        with self.accounts.lock:schema_version.declare(self.accounts._conn,'issue_tickets',2)
        with self.assertRaises(schema_version.SchemaVersionError):self.create()
        with self.accounts.lock:self.assertIsNone(self.accounts._conn.execute("SELECT name FROM sqlite_master WHERE name='issue_tickets'").fetchone())

    def test_invalid_diagnostics_and_text(self):
        for change in [{'title':''},{'title':'x'*201},{'diagnostics':[]},{'diagnostics':{'x':float('nan')}},{'diagnostics':{'x':'a'*80001}},{'client_key':'bad'}]:
            body=payload();body.update(change)
            with self.subTest(change=list(change)),self.assertRaises(issues.IssueError):self.service.create('fixture-token-1',body)

    def test_create_rate_limit_and_unchanged_retry(self):
        for i in range(10):self.service.create('fixture-token-1',payload(f'fixture-key-{i:08d}'))
        with self.assertRaises(issues.IssueError) as error:self.service.create('fixture-token-1',payload('fixture-key-99999999'))
        self.assertEqual(error.exception.status,429)
        self.assertFalse(self.service.create('fixture-token-1',payload('fixture-key-00000000'))['created'])

    def test_history_pages_are_bounded_and_complete(self):
        ticket=self.create()
        for version in range(1,25):
            self.service.update('fixture-token-1',ticket,{'status':'investigating','note':'note','version':version})
        recent=self.service.get('fixture-token-1',ticket)['ticket']
        self.assertEqual(len(recent['events']),20)
        self.assertEqual(recent['next_before_version'],6)
        older=self.service.get('fixture-token-1',ticket,before_version=6)['ticket']
        self.assertEqual([e['version'] for e in older['events']],list(range(1,6)))
        self.assertIsNone(older['next_before_version'])

    def test_concurrent_updates_have_one_winner(self):
        ticket=self.create();results=[];barrier=threading.Barrier(2)
        def update():
            barrier.wait()
            try:
                self.service.update('fixture-token-1',ticket,{'status':'resolved','note':'fixture','version':1});results.append(200)
            except issues.IssueError as error:results.append(error.status)
        threads=[threading.Thread(target=update) for _ in range(2)]
        for thread in threads:thread.start()
        for thread in threads:thread.join(15)
        self.assertEqual(sorted(results),[200,409])
        self.assertEqual(len(self.service.get('fixture-token-1',ticket)['ticket']['events']),2)


class IssueHTTPTests(unittest.TestCase):
    def setUp(self):
        from meowdoku import server
        self.server_mod=server
        self.temp=tempfile.TemporaryDirectory(prefix='issue-http-');self.addCleanup(self.temp.cleanup)
        web=Path(self.temp.name)/'web';web.mkdir();(web/'boards.json').write_text('{"boards":[],"levels":{}}')
        self.sessions=server.Sessions(web,None)
        self.addCleanup(self.sessions.accounts._conn.close)
        self.sessions.lifecycle=SimpleNamespace(token='fixture-only',restore_for=lambda _:None)
        self.sessions.access.lookup=lambda _:None
        users(self.sessions.accounts)
        self.httpd=ThreadingHTTPServer(('127.0.0.1',0),server.make_handler(self.sessions))
        threading.Thread(target=self.httpd.serve_forever,daemon=True).start()
        self.addCleanup(self.httpd.server_close);self.addCleanup(self.httpd.shutdown)

    def call(self,path='/api/issues',method='GET',token=1,body=None,headers=None):
        h={'Cookie':f'{self.server_mod.ACCOUNT_COOKIE}=fixture-token-{token}'} if token else {}
        if body is not None:h.update({'Content-Type':'application/json','X-Meowdoku-Issue':'1'})
        h.update(headers or {})
        conn=http.client.HTTPConnection('127.0.0.1',self.httpd.server_port,timeout=10)
        conn.request(method,path,body=None if body is None else json.dumps(body),headers=h)
        response=conn.getresponse();data=response.read();status=response.status;conn.close()
        return status,json.loads(data)

    def test_http_roundtrip_and_anonymous_lan_denied(self):
        self.assertEqual(self.call(token=None)[0],401)
        status,result=self.call(method='POST',body=payload());self.assertEqual(status,201)
        ticket=result['id']
        self.assertEqual(self.call('/api/issues/'+ticket,token=2)[0],404)
        self.assertEqual(self.call('/api/issues/'+ticket,token=3)[0],200)
        self.assertEqual(self.call('/api/issues/'+ticket+'/status','POST',1,{'status':'resolved','note':'Fixture verified','version':1})[0],200)
        self.assertEqual(self.call('/api/issues/'+ticket)[1]['ticket']['status'],'resolved')
        self.assertEqual(self.call(method='POST',body=payload())[0],200)

    def test_csrf_types_injection_and_size(self):
        for headers,expected in [({'X-Meowdoku-Issue':''},403),({'Origin':'https://attacker.invalid'},403),({'Sec-Fetch-Site':'cross-site'},403),({'Content-Type':'text/plain'},415)]:
            with self.subTest(headers=headers):self.assertEqual(self.call(method='POST',body=payload(),headers=headers)[0],expected)
        body=payload();body['description']='x'*100001
        self.assertEqual(self.call(method='POST',body=body)[0],413)
        body=payload();body['owner_id']=2
        self.assertEqual(self.call(method='POST',body=body)[0],400)
        self.assertEqual(self.call('/api/issues?scope=all',token=4)[0],403)
        self.assertEqual(self.call('/api/issues?scope=mine&scope=all')[0],400)

    def test_rejected_post_closes_and_cannot_be_reused(self):
        conn=http.client.HTTPConnection('127.0.0.1',self.httpd.server_port,timeout=10)
        conn.request('POST','/api/issues',body=json.dumps(payload()),headers={'Content-Type':'application/json','Cookie':f'{self.server_mod.ACCOUNT_COOKIE}=fixture-token-1'})
        response=conn.getresponse();response.read()
        self.assertEqual(response.status,403)
        self.assertTrue(response.will_close)
        conn.request('GET','/api/issues',headers={'Cookie':f'{self.server_mod.ACCOUNT_COOKIE}=fixture-token-1'})
        response=conn.getresponse();self.assertEqual(response.status,200);response.read();conn.close()


if __name__=='__main__':unittest.main()
