import json
from pathlib import Path
import tempfile
import threading
import unittest
import sys
from email.message import Message
from types import SimpleNamespace
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from meowdoku.nodus_radio_jukebox import Jukebox, http_route, ControlError

class JukeboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);self.config=root/'radio.json'
        self.config.write_text(json.dumps({'jukebox':{'songs':[{'id':s,'title':s.upper(),'path':'private.wav'} for s in ('a','b','c')]}}))
        self.store=Jukebox(root/'accounts.db',self.config)
    def play(self):
        p=self.store.claim();self.assertIsNotNone(p);self.assertTrue(self.store.finish(p['token']));return p
    def test_round_robin_preserves_per_user_fifo_and_empty_skip(self):
        for user,song in [(1,'a'),(1,'b'),(1,'c'),(2,'c'),(2,'a'),(3,'b')]:self.store.request(user,song)
        self.assertIsNone(self.store.claim());self.store.pause(9,False)
        played=[self.play() for _ in range(6)]
        self.assertEqual([p['user_id'] for p in played],[1,2,3,1,2,1])
        self.assertEqual([p['song_id'] for p in played if p['user_id']==1],['a','b','c'])
        self.assertIsNone(self.store.claim())
    def test_rejoin_and_new_listener_join_tail(self):
        self.store.request(1,'a');self.store.request(1,'b');self.store.request(2,'a');self.store.pause(9,False)
        self.play();self.store.request(3,'a')
        self.assertEqual([self.play()['user_id'] for _ in range(3)],[2,1,3])
    def test_claim_is_atomic_and_token_fences_stale_end(self):
        self.store.request(1,'a');self.store.request(2,'b');self.store.pause(9,False)
        # Initialize schema before parallel callers (normal first request path).
        claimed=[];errors=[]
        def work():
            try:claimed.append(self.store.claim())
            except Exception as e:errors.append(e)
        threads=[threading.Thread(target=work) for _ in range(8)]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertEqual(errors,[]);self.assertEqual(sum(p is not None for p in claimed),1)
        p=next(p for p in claimed if p)
        self.assertFalse(self.store.finish('wrong'))
        self.assertTrue(self.store.finish(p['token']));new=self.store.claim()
        self.assertFalse(self.store.finish(p['token']));self.assertEqual(self.store.status(1)['current']['id'],new['id'])
    def test_pause_current_finishes_without_next(self):
        self.store.request(1,'a');self.store.request(2,'b');self.store.pause(9,False)
        p=self.store.claim();self.store.pause(10,True)
        self.assertEqual(self.store.status(1)['current']['id'],p['id'])
        self.store.finish(p['token']);self.assertIsNone(self.store.claim())
    def test_persistence_and_private_paths_not_exposed(self):
        self.store.request(1,'a');self.store.pause(9,False);p=self.store.claim()
        new=Jukebox(self.store.db_path,self.config)
        self.assertEqual(new.status(1)['current']['id'],p['id']);self.assertIsNone(new.claim())
        self.assertNotIn('private.wav',json.dumps(new.status(2)))
    def test_own_remove_admin_remove(self):
        r=self.store.request(1,'a')
        with self.assertRaises(ControlError):self.store.remove(2,r)
        self.store.remove(2,r,admin=True);self.assertEqual(self.store.status(1)['requests'],[])
    def test_zero_vote_is_real_and_update_not_duplicate(self):
        self.store.request(1,'a');self.store.pause(9,False);p=self.store.claim()
        self.store.rate(1,p['id'],0);self.store.rate(2,p['id'],4)
        chart=self.store.status(1)['charts'][0]
        self.assertEqual((chart['votes'],chart['points'],chart['average']),(2,4,2))
        self.assertEqual(self.store.status(1)['my_vote'],0)
        self.store.rate(1,p['id'],2)
        self.assertEqual(self.store.status(1)['charts'][0]['votes'],2)
        for bad in (True,-1,6,2.5,'5'):
            with self.assertRaises(ControlError):self.store.rate(1,p['id'],bad)
        self.store.finish(p['token'])
        with self.assertRaises(ControlError):self.store.rate(1,p['id'],5)
    def test_removed_catalogue_song_not_started(self):
        self.store.request(1,'a');self.config.write_text('{"jukebox":{"songs":[]}}');self.store.pause(9,False)
        self.assertIsNone(self.store.claim())
    def handler(self,user=1,admin=False,body=None):
        h=Mock();h._principal.return_value=SimpleNamespace(user_id=user);h._caps.return_value=['account_manage'] if admin else []
        h._is_secure.return_value=True;h._body_json.return_value=body
        h.headers=Message()
        for k,v in {'Host':'test','Origin':'https://test','Content-Type':'application/json','Content-Length':'20','X-Meowdoku-Radio':'1'}.items():h.headers[k]=v
        return h
    def test_http_member_requests_but_admin_controls(self):
        runner=Mock();h=self.handler(body={'action':'request','song_id':'a'})
        http_route(h,'POST',self.store,runner);self.assertEqual(len(self.store.status(1)['requests']),1)
        for action in ('pause','resume','skip'):
            h=self.handler(body={'action':action});http_route(h,'POST',self.store,runner)
            self.assertEqual(h._json.call_args.args[1],403)
            h=self.handler(user=99,admin=True,body={'action':action});http_route(h,'POST',self.store,runner)
            getattr(runner,action).assert_called_once_with(99)
    def test_bad_action_payload(self):
        for body in ({'action':[]},{'action':'request','song_id':{}},{'action':'remove','request_id':True},{'action':'rate','play_id':1,'rating':False}):
            h=self.handler(body=body);http_route(h,'POST',self.store)
            self.assertEqual(h._json.call_args.args[1],400)

    def test_inert_get_and_csrf(self):
        runner=Mock();h=self.handler();http_route(h,'GET',self.store,runner);runner.resume.assert_not_called()
        h=self.handler(body={'action':'request','song_id':'a'});h.headers.replace_header('Origin','https://evil')
        http_route(h,'POST',self.store,runner);self.assertEqual(h._json.call_args.args[1],403)
        h=self.handler(user=None);http_route(h,'GET',self.store,runner);self.assertEqual(h._json.call_args.args[1],401)

if __name__=='__main__':unittest.main()
