"""Real host factory + official SDK integration; network replaced only at transport."""
import importlib.util
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
from test_meowdoku_remote_connections import Handler
from meowdoku import liaison, remote_connections as host
try:
    import httpx
    from speaktome_api import remote_mcp
except ImportError:
    remote_mcp = None

@unittest.skipIf(remote_mcp is None, 'optional remote MCP SDK not installed')
class SDKHostTests(unittest.TestCase):
    def test_interim_factory_remains_active_without_cutover_marker(self):
        with tempfile.TemporaryDirectory() as root:
            state=SimpleNamespace(run_dir=Path(root)/'run',secrets_dir=Path(root)/'secrets')
            state.run_dir.mkdir()
            (state.run_dir/'mcp_connections.json').write_text(json.dumps({'suno':{
                'callback_url':'https://example.test/api/connections/suno/callback',
                'allowed_origins':['https://api.suno.com']}}))
            sessions=SimpleNamespace(state=state,accounts_db=Path(root)/'sql'/'accounts.db')
            service=host.service_for(sessions)
            try:
                self.assertIsInstance(service,remote_mcp.ConnectionService)
            finally:
                service.close()

    def test_actual_factory_vault_callback_and_tool_forward(self):
        requests=[]
        base='https://api.suno.com'
        callback='https://example.test/api/connections/suno/callback'
        schema={'type':'object','$defs':{'Track':{'type':'string'}},'properties':{'track':{'$ref':'#/$defs/Track'}}}
        def respond(r):
            requests.append(r)
            path=r.url.path
            if path.startswith('/.well-known/oauth-protected-resource'):
                return httpx.Response(200,json={'resource':base+'/mcp','authorization_servers':[base]})
            if path.startswith('/.well-known/'):
                return httpx.Response(200,json={'issuer':base,'authorization_endpoint':base+'/authorize',
                    'token_endpoint':base+'/token','registration_endpoint':base+'/register',
                    'response_types_supported':['code'],'code_challenge_methods_supported':['S256']})
            if path=='/register': return httpx.Response(201,json={'client_id':'fixture-client','redirect_uris':[callback], 'token_endpoint_auth_method':'none'})
            if path=='/token':
                form=parse_qs(r.content.decode()); self.assertTrue(form['code_verifier'])
                return httpx.Response(200,json={'access_token':'fixture-token','token_type':'Bearer','expires_in':3600})
            if r.headers.get('Authorization')!='Bearer fixture-token':
                return httpx.Response(401,headers={'WWW-Authenticate':'Bearer resource_metadata="'+base+'/.well-known/oauth-protected-resource/mcp"'})
            if r.method=='DELETE': return httpx.Response(200)
            if r.method=='GET': return httpx.Response(405)
            body=json.loads(r.content)
            if 'id' not in body: return httpx.Response(202)
            method=body['method']
            if method=='initialize': result={'protocolVersion':'2025-11-25','serverInfo':{'name':'fixture','version':'1'},'capabilities':{'tools':{}}}
            elif method=='tools/list': result={'tools':[{'name':'fixture_tool','inputSchema':schema}]}
            elif method=='tools/call': result={'content':[{'type':'text','text':'fixture-result'}]}
            else: raise AssertionError(method)
            return httpx.Response(200,json={'jsonrpc':'2.0','id':body['id'],'result':result})
        with tempfile.TemporaryDirectory() as root:
            base_dir=Path(root)
            config=base_dir/'mcp_connections.json'
            config.write_text(json.dumps({'suno':{'callback_url':callback,'allowed_origins':[base]}}))
            key_file=base_dir/'liaison-ipc.key'; key_file.write_bytes(b'x'*32)
            def call(action,method='GET',body=None,owner=7,query=''):
                h=Handler(owner,body); h.path='/api/connections/suno/'+action+query
                host.route(h,object(),h.path.partition('?')[0],method); return h
            with patch.object(remote_mcp,'PublicHTTPTransport',lambda:httpx.MockTransport(respond)):
                service=liaison.connection_service(config,base_dir/'vault')
                server=liaison.LiaisonServer(key_file,service,port=0)
                thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
                client=liaison.LiaisonClient(key_file,port=server.server_address[1])
                try:
                    with patch.object(host,'service_for',return_value=client):
                        first=call('connect','POST').output
                        self.assertEqual(first[0],200)
                        self.assertEqual(first[1]['status'],'authorization_required')
                        state_token=parse_qs(urlsplit(first[1]['authorization_url']).query)['state'][0]
                        query='?code=fixture-code&state='+state_token
                        self.assertEqual(call('callback',owner=8,query=query).output[0],400)
                        self.assertEqual(call('callback',query=query).code,303)
                        for _ in range(100):
                            status=call('status').output[1]
                            if status['status']=='connected': break
                            time.sleep(.01)
                        self.assertEqual(status,{'status':'connected','configured':True})
                        self.assertEqual(call('tools').output[1]['tools'][0]['inputSchema'],schema)
                        result=call('call','POST',{'name':'fixture_tool','arguments':{'track':'x'}})
                        self.assertEqual(result.output[1]['content'][0]['text'],'fixture-result')
                        self.assertFalse(call('status',owner=8).output[1]['configured'])
                        call('disconnect','POST')
                        self.assertFalse(call('status').output[1]['configured'])
                finally:
                    server.shutdown(); server.server_close(); service.close()
        self.assertTrue(requests)

if __name__=='__main__': unittest.main()
