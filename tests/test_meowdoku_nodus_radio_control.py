import json, pathlib, sys, tempfile, threading, unittest
from email.message import Message
from types import SimpleNamespace
from unittest.mock import Mock
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'examples'))
from meowdoku.nodus_radio_control import Manager, ControlError, http_route

class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root=pathlib.Path(self.temp.name); exe=root/'sink.exe'; exe.touch()
        self.path=root/'radio.json'
        self.config={'enabled':True,'source':'file','playback':dict(enabled=True,router_port=49500,listen_port=49501,peer_port=49502,peer_host='127.0.0.1',ssrc=42,device_id=0,executable=str(exe))}
        self.path.write_text(json.dumps(self.config))
        self.jobs=Mock();self.jobs.begin.return_value=True
        self.sub=Mock();self.sub.ready=threading.Event();self.sub.ready.set();self.sub.error=None
        self.sink=Mock();self.sink.consume.side_effect=lambda sub,event:event.wait(2)
        self.sf=Mock(return_value=self.sub);self.pf=Mock(return_value=self.sink)
        self.manager=Manager(self.path,self.jobs,subscription=self.sf,playback=self.pf)
        self.addCleanup(self.manager.close)
    def test_status_is_inert_and_start_is_owned(self):
        self.assertTrue(self.manager.status(7)['configured']);self.sf.assert_not_called();self.pf.assert_not_called()
        self.manager.start(7)
        with self.assertRaises(ControlError):self.manager.start(8)
        with self.assertRaises(ControlError):self.manager.stop(8)
        self.assertTrue(self.manager.status(7)['can_stop'])
        self.manager.stop(7);self.manager.close()
        self.sub.close.assert_called_once();self.sink.stop.assert_called_once()
        self.jobs.end.assert_called_once();self.assertEqual(self.manager.status(7)['status'],'stopped')
        self.assertEqual(self.jobs.begin.call_args.kwargs['owner'],'account:7')
    def test_registration_failure_has_no_side_effect(self):
        self.jobs.begin.return_value=False
        with self.assertRaises(ControlError):self.manager.start(7)
        self.sf.assert_not_called();self.pf.assert_not_called()
    def test_disabled_invalid_and_remote_configs_fail_closed(self):
        for cfg in ([],{'playback':[]},{'enabled':False},dict(self.config,playback=dict(self.config['playback'],peer_host='example.org'))):
            self.path.write_text(json.dumps(cfg))
            with self.assertRaises(ControlError):self.manager.start(7)
        self.sf.assert_not_called()
    def test_cancel_during_device_start(self):
        self.sink.start.side_effect=lambda:self.manager.cancel.set()
        self.manager.start(7);self.manager.worker.join(2)
        self.sink.consume.assert_not_called();self.sub.close.assert_called_once()
    def test_sink_failure_releases_subscription_and_job(self):
        self.sink.start.side_effect=RuntimeError('device unavailable')
        self.manager.start(7);self.manager.worker.join(2)
        self.sub.close.assert_called_once();self.jobs.end.assert_called_once()
        self.assertEqual(self.manager.status(7)['status'],'failed')
    def test_cleanup_failure_not_reported_success(self):
        self.sink.consume.side_effect=None;self.sink.stop.side_effect=RuntimeError('stop failed')
        self.manager.start(7);self.manager.worker.join(2)
        self.sub.close.assert_called_once();self.jobs.end.assert_called_once()
        self.assertEqual(self.manager.status(7)['status'],'failed')
    def handler(self, user=7, caps=('account_manage',), body=None):
        h=Mock();h._principal.return_value=SimpleNamespace(user_id=user);h._caps.return_value=caps
        h._is_secure.return_value=True;h._body_json.return_value=body or {'action':'start'}
        h.headers=Message()
        for key,value in {'Host':'example.org','Origin':'https://example.org','Content-Type':'application/json','Content-Length':'18','X-Meowdoku-Radio':'1'}.items():h.headers[key]=value
        return h
    def test_auth_csrf_and_payload_gates(self):
        for h,expected in ((self.handler(user=None),401),(self.handler(caps=()),403)):
            m=Mock();http_route(h,'POST',m);self.assertEqual(h._json.call_args.args[1],expected);m.start.assert_not_called()
        for key,value,expected in [('Origin','https://evil.org',403),('X-Meowdoku-Radio','0',403),('Content-Type','text/plain',415)]:
            h=self.handler();h.headers.replace_header(key,value);m=Mock();http_route(h,'POST',m)
            self.assertEqual(h._json.call_args.args[1],expected);m.start.assert_not_called()
        for body in ([],['start'],{'action':'start','device_id':1},{'action':'launch'}):
            h=self.handler();h._body_json.return_value=body;m=Mock();http_route(h,'POST',m)
            self.assertEqual(h._json.call_args.args[1],400);m.start.assert_not_called()
    def test_valid_actions_and_inert_get(self):
        m=Mock();h=self.handler();http_route(h,'GET',m);m.status.assert_called_once_with(7);m.start.assert_not_called()
        http_route(h,'POST',m);m.start.assert_called_once_with(7);self.assertEqual(h._json.call_args.args[1],202)
        h._body_json.return_value={'action':'stop'};http_route(h,'POST',m);m.stop.assert_called_once_with(7)

if __name__=='__main__':unittest.main()
