"""Real Accounts and loopback HTTP, exclusively disposable fixture state."""
from pathlib import Path
import concurrent.futures
import http.client
import json
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'examples')]
from meowdoku import accounts, experience_reports as reports, schema_version


def fixture_users(account):
    with account.lock:
        db = account._conn
        for i, name in enumerate(('alice', 'bob', 'operator'), 1):
            db.execute('INSERT INTO users VALUES(?,?,?,NULL,0)', (i, name, 'fixture-not-a-password'))
            db.execute('INSERT INTO sessions VALUES(?,?,0,9999999999)', ('fixture-' + str(i), i))
        db.commit()


def payload(request_id='first-report', category='DOC'):
    return {'request_id': request_id, 'category': category, 'title': 'Measured result',
            'body': '## Observed Behaviour\nThe bounded fixture passed.\n',
            'prompt_history': ['  Exact original prompt.\nSecond line.  '],
            'next_steps': ['Check the next declared boundary.']}


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='experience-reports-')
        self.addCleanup(self.temp.cleanup)
        self.accounts = accounts.Accounts(Path(self.temp.name) / 'accounts.db')
        self.addCleanup(self.accounts._conn.close)
        fixture_users(self.accounts)
        self.service = reports.Service(self.accounts)

    def test_categories_prompt_history_and_linked_steps(self):
        for category in reports.CATEGORIES:
            body = payload(category, category)
            receipt = self.service.make('fixture-1', body)
            report = self.service.examine('fixture-1', receipt['id'])['report']
            self.assertEqual(report['category'], category)
            self.assertEqual(report['body'], body['body'])
            self.assertEqual(report['prompt_history'], body['prompt_history'])
            self.assertEqual(report['next_steps'], [{'ordinal': 0, 'text': body['next_steps'][0]}])
            self.assertEqual(report['owner_id'], 1)

    def test_owner_only_and_unknown_are_same(self):
        rid = self.service.make('fixture-1', payload())['id']
        for token in ('fixture-2', 'fixture-3'):
            self.assertEqual(self.service.listing(token)['reports'], [])
            for target in (rid, 'f' * 32):
                with self.assertRaises(reports.ReportError) as caught:
                    self.service.examine(token, target)
                self.assertEqual((str(caught.exception), caught.exception.status), ('report not found', 404))
        self.assertFalse(self.service.map('fixture-2')['map']['world']['objects'])

    def test_replay_and_changed_content(self):
        first = self.service.make('fixture-1', payload())
        replay = self.service.make('fixture-1', payload())
        self.assertEqual(first['id'], replay['id'])
        self.assertFalse(replay['created'])
        changed = payload(); changed['next_steps'] = ['Different']
        with self.assertRaises(reports.ReportError) as caught:
            self.service.make('fixture-1', changed)
        self.assertEqual(caught.exception.status, 409)
        self.assertNotEqual(first['id'], self.service.make('fixture-2', payload())['id'])

    def test_concurrent_identical_requests_create_once(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.service.make('fixture-1', payload()), range(4)))
        self.assertEqual(len({r['id'] for r in results}), 1)
        self.assertEqual(sum(r['created'] for r in results), 1)

    def test_followup_failure_rolls_back_report(self):
        self.service.listing('fixture-1')
        with self.accounts.lock:
            self.accounts._conn.execute("CREATE TRIGGER fail_step BEFORE INSERT ON experience_report_followups BEGIN SELECT RAISE(ABORT,'fixture'); END")
            self.accounts._conn.commit()
        with self.assertRaises(Exception):
            self.service.make('fixture-1', payload())
        self.assertEqual(self.service.listing('fixture-1')['reports'], [])

    def test_logout_and_expired_session(self):
        self.accounts.logout('fixture-1')
        for token in ('fixture-1', None, 'missing'):
            with self.assertRaises(reports.ReportError) as caught:
                self.service.make(token, payload())
            self.assertEqual(caught.exception.status, 401)
        with self.accounts.lock:
            self.accounts._conn.execute('UPDATE sessions SET expires_at=0 WHERE user_id=2')
            self.accounts._conn.commit()
        with self.assertRaises(reports.ReportError):
            self.service.listing('fixture-2')

    def test_newer_schema_refuses_before_tables(self):
        with self.accounts.lock:
            schema_version.declare(self.accounts._conn, 'experience_reports', 99)
        with self.assertRaises(schema_version.SchemaVersionError):
            self.service.make('fixture-1', payload())
        self.assertIsNone(self.accounts._conn.execute("SELECT name FROM sqlite_master WHERE name='experience_reports'").fetchone())

    def test_bad_fields_bounds_and_no_automatic_data(self):
        cases = []
        for key, value in [('owner_id', 2), ('path', 'C:/host/file'), ('category', 'ISSUE'),
                           ('prompt_history', []), ('prompt_history', ['x', 4]), ('next_steps', 'run'),
                           ('body', ' '), ('request_id', '../x')]:
            item = payload(); item[key] = value; cases.append(item)
        item = payload(); item['body'] = '\u20ac' * 30000; cases.append(item)
        for item in cases:
            with self.subTest(item=str(item)[:100]), self.assertRaises(reports.ReportError):
                self.service.make('fixture-1', item)
        self.assertEqual(self.service.listing('fixture-1')['reports'], [])

    def test_rate_limit_replay_still_succeeds_and_map_projection(self):
        for i in range(10):
            self.service.make('fixture-1', payload('report-' + str(i)))
        with self.assertRaises(reports.ReportError) as caught:
            self.service.make('fixture-1', payload('eleventh'))
        self.assertEqual(caught.exception.status, 429)
        self.assertFalse(self.service.make('fixture-1', payload('report-0'))['created'])
        from meowdoku.living_data_query import GraphView
        mapped = self.service.map('fixture-1')['map']
        self.assertEqual(len(mapped['world']['objects']), 10)
        self.assertNotIn('body', mapped['world']['objects'][0])
        self.assertEqual(GraphView(mapped, 'layout').query(reports.ROOT, radius=1)['total'], 4)

    def test_pagination_and_category_filter(self):
        for i in range(51):
            with patch('meowdoku.experience_reports.time.time', return_value=(i + 1) * 601):
                self.service.make('fixture-1', payload('page-' + str(i), 'LOG' if i == 0 else 'DOC'))
        first = self.service.map('fixture-1')
        self.assertEqual(first['next_offset'], 50)
        second = self.service.map('fixture-1', offset=50)
        self.assertIsNone(second['next_offset'])
        self.assertEqual(len(second['map']['world']['objects']), 1)
        self.assertEqual(len(self.service.listing('fixture-1', category='LOG')['reports']), 1)


class ReportHTTPTests(unittest.TestCase):
    def setUp(self):
        from meowdoku import server
        self.server_mod = server
        self.temp = tempfile.TemporaryDirectory(prefix='experience-report-http-'); self.addCleanup(self.temp.cleanup)
        web = Path(self.temp.name) / 'web'; web.mkdir(); (web / 'boards.json').write_text('{"boards":[],"levels":{}}')
        self.sessions = server.Sessions(web, None)
        self.addCleanup(self.sessions.accounts._conn.close)
        self.sessions.lifecycle = SimpleNamespace(token='fixture-only', restore_for=lambda _: None)
        self.sessions.access.lookup = lambda _: None
        fixture_users(self.sessions.accounts)
        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), server.make_handler(self.sessions))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close); self.addCleanup(self.httpd.shutdown)

    def call(self, path='/api/reports', method='GET', token=1, body=None, headers=None):
        h = {'Cookie': f'{self.server_mod.ACCOUNT_COOKIE}=fixture-{token}'} if token else {}
        if body is not None: h['Content-Type'] = 'application/json'
        h.update(headers or {})
        conn = http.client.HTTPConnection('127.0.0.1', self.httpd.server_port, timeout=10)
        conn.request(method, path, body=None if body is None else json.dumps(body), headers=h)
        response = conn.getresponse(); data = response.read(); status = response.status
        self.assertEqual(response.getheader('Cache-Control'), 'no-store')
        conn.close()
        return status, json.loads(data)

    def test_real_roundtrip_owner_and_graph(self):
        self.assertEqual(self.call(token=None)[0], 401)
        status, result = self.call(method='POST', body=payload()); self.assertEqual(status, 201)
        rid = result['id']
        self.assertEqual(self.call('/api/reports/' + rid)[1]['report']['prompt_history'], payload()['prompt_history'])
        self.assertEqual(self.call('/api/reports/' + rid, token=2)[0], 404)
        self.assertEqual(self.call('/api/reports/map')[1]['map']['world']['objects'][0]['report_id'], rid)
        self.assertEqual(self.call(method='POST', body=payload())[0], 200)

    def test_auth_origin_bounds_and_query_denials(self):
        for headers, status in [({'Origin': 'https://attacker.invalid'}, 403),
                                ({'Sec-Fetch-Site': 'cross-site'}, 403), ({'Content-Type': 'text/plain'}, 415)]:
            self.assertEqual(self.call(method='POST', body=payload(), headers=headers)[0], status)
        bad = payload(); bad['owner_id'] = 2
        self.assertEqual(self.call(method='POST', body=bad)[0], 400)
        self.assertEqual(self.call('/api/reports?scope=all')[0], 400)
        self.assertEqual(self.call('/api/reports?offset=0&offset=1')[0], 400)
        huge = payload(); huge['body'] = 'x' * 140000
        self.assertEqual(self.call(method='POST', body=huge)[0], 413)
        self.assertEqual(self.call('/api/reports?offset=0', method='POST', body=payload())[0], 400)

    def test_viewer_and_missing_dev_capability_cannot_initialize_reports(self):
        for method in ('GET', 'POST'):
            self.assertEqual(self.call(method=method, body=payload() if method == 'POST' else None,
                headers={self.server_mod.share_mod.VIEWER_HEADER: 'fixture-viewer'})[0], 403)
        with patch.object(self.sessions.access, 'decide', return_value=('dns', 'play')):
            self.assertEqual(self.call()[0], 403)
            self.assertEqual(self.call(method='POST', body=payload())[0], 403)
        self.assertIsNone(self.sessions.accounts._conn.execute(
            "SELECT name FROM sqlite_master WHERE name='experience_reports'").fetchone())

    def test_malformed_framing_closes_without_schema_writes(self):
        for lengths, transfer, raw in ((['2', '2'], None, b'{}'), (['2'], 'chunked', b'{}'),
                                      ([], None, b''), (['1'], None, b'{')):
            with self.subTest(lengths=lengths, transfer=transfer):
                conn = http.client.HTTPConnection('127.0.0.1', self.httpd.server_port, timeout=10)
                conn.putrequest('POST', '/api/reports')
                conn.putheader('Cookie', f'{self.server_mod.ACCOUNT_COOKIE}=fixture-1')
                conn.putheader('Content-Type', 'application/json')
                for length in lengths: conn.putheader('Content-Length', length)
                if transfer: conn.putheader('Transfer-Encoding', transfer)
                conn.endheaders(raw)
                response = conn.getresponse(); response.read()
                self.assertEqual(response.status, 400)
                self.assertTrue(response.will_close)
                conn.close()
        self.assertIsNone(self.sessions.accounts._conn.execute(
            "SELECT name FROM sqlite_master WHERE name='experience_reports'").fetchone())

    def test_mcp_make_enter_and_examine_actual_http(self):
        # Inject only transport. The production MCP operations still call the real
        # HTTP handler and Accounts service, with no test-only direct SQL path.
        import importlib
        sys.path.insert(0, str(ROOT.parent / 'speaktome'))
        try:
            mcp = importlib.import_module('speaktome_mcp.server')
        finally:
            sys.path.pop(0)
        outer = self
        class Rest:
            def call(self, path, body=None, **kwargs):
                status, data = outer.call(path, 'POST' if body is not None else 'GET', body=body)
                if status >= 400:
                    raise mcp.RestError(data.get('error', ''), status)
                return data
        tools = mcp.Tools(Rest())
        request = payload(); key = request.pop('request_id')
        receipt = tools.make({'request_id': key, 'report': request})
        entered = tools.enter({'map': 'reports'})
        self.assertEqual(entered['at']['identity'], reports.ROOT)
        report = tools.examine({'at': 'report:' + receipt['id']})['untrusted']['report']
        self.assertEqual(report['body'], request['body'])
        self.assertTrue(tools.leave({})['closed'])
        self.assertEqual(tools.examine({'report': receipt['id']})['untrusted']['report']['id'], receipt['id'])

    def test_production_mcp_transport_preserves_bounded_unicode(self):
        import importlib
        import urllib.request
        sys.path.insert(0, str(ROOT.parent / 'speaktome'))
        try:
            transport = importlib.import_module('speaktome_mcp.rest')
        finally:
            sys.path.pop(0)
        opener = urllib.request.build_opener()
        opener.addheaders = [('Cookie', f'{self.server_mod.ACCOUNT_COOKIE}=fixture-1')]
        rest = transport.Rest(f'http://127.0.0.1:{self.httpd.server_port}', 'alice', 'unused', opener=opener)
        rest.signed_in = True
        body = payload(); body['body'] = '\u00e9' * 30000
        receipt = rest.call('/api/reports', body)
        self.assertEqual(rest.call('/api/reports/' + receipt['id'])['report']['body'], body['body'])


if __name__ == '__main__':
    unittest.main()
