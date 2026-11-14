import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from meowdoku import remote_connections as mod

class Handler:
    def __init__(self, owner=7, body=None):
        self.path = '/api/connections/suno/status'
        self.headers = {'Host':'example.test', 'Origin':'https://example.test',
                        'X-Meowdoku-Secret':'1', 'Content-Type':'application/json'}
        self.owner, self.body, self.output = owner, body or {}, None
        self.response_headers = {}
    def _is_secure(self): return True
    def _principal(self): return types.SimpleNamespace(user_id=self.owner)
    def _body_json(self, limit): return self.body
    def _json(self, value, status=200): self.output = (status, value)
    def send_response(self, status): self.code = status
    def send_header(self, key, value): self.response_headers[key] = value
    def end_headers(self): pass

class Service:
    def status(self, owner, provider): return {'owner':owner,'status':'authorized'}
    def call(self, owner, provider, name, args): return {'content':[{'type':'text','text':name}], 'owner':owner}
    def callback(self, owner, provider, code, state):
        if (owner, code, state) != (7,'one-use','state'): raise PermissionError()

class HostTests(unittest.TestCase):
    def run_route(self, h, action, method='GET'):
        with patch.object(mod, 'service_for', return_value=Service()):
            mod.route(h, object(), '/api/connections/suno/'+action, method)
    def test_owner_and_guest(self):
        h=Handler(); self.run_route(h,'status'); self.assertEqual(h.output[1]['owner'],7)
        h=Handler(None); self.run_route(h,'status'); self.assertEqual(h.output[0],401)
    def test_csrf_and_owner_injection(self):
        h=Handler(body={'name':'tool','arguments':{}}); h.headers['Origin']='https://evil.test'
        self.run_route(h,'call','POST'); self.assertEqual(h.output[0],403)
        h=Handler(body={'owner':9}); self.run_route(h,'connect','POST'); self.assertEqual(h.output[0],400)
    def test_call_preserves_result(self):
        h=Handler(body={'name':'real_tool','arguments':{'nested':[1,2]}})
        self.run_route(h,'call','POST'); self.assertEqual(h.output[1]['content'][0]['text'],'real_tool')
    def test_callback_cleans_location_and_checks_owner(self):
        h=Handler(); h.path='/api/connections/suno/callback?code=one-use&state=state'
        self.run_route(h,'callback'); self.assertEqual(h.code,303)
        self.assertEqual(h.response_headers['Location'],'/connected-services.html')
        self.assertEqual(h.response_headers['Referrer-Policy'],'no-referrer')
        h.owner=8; self.run_route(h,'callback'); self.assertEqual(h.output[0],400)
    def test_unknown_provider_and_method(self):
        h=Handler(); self.run_route(h,'status','POST'); self.assertEqual(h.output[0],405)
        mod.route(h,object(),'/api/connections/evil/status','GET'); self.assertEqual(h.output[0],404)
    def test_vault_isolated_atomic_clear(self):
        with tempfile.TemporaryDirectory() as d:
            r,w,c=mod.vault_callbacks(d,7,'suno'); w({'tokens':{'access_token':'test-only'}})
            self.assertEqual(r()['tokens']['access_token'],'test-only')
            self.assertIsNone(mod.vault_callbacks(d,8,'suno')[0]())
            self.assertEqual(list(Path(d).rglob('*.tmp')),[]); c(); self.assertIsNone(r())
            with self.assertRaises(PermissionError): mod.vault_callbacks(d,'../','suno')
    def test_host_wiring_and_callback_log_redaction(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/'examples/meowdoku/server.py').read_text()
        self.assertIn('return route(self, sessions, path, "GET")',source)
        self.assertIn('return route(self, sessions, path0, "POST")',source)
        self.assertIn('OAuth codes/state',source)
        conf=(root/'examples/meowdoku/deploy/apache/meowdoku-proxy.conf').read_text()
        self.assertIn('meowdoku_safe',conf)
        self.assertNotIn('" combined',conf)

if __name__=='__main__': unittest.main()
