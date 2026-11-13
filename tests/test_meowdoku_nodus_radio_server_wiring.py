"""Exercise actual server handler wiring without importing its render dependencies."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))

class ServerRadioWiringTests(unittest.TestCase):
    def run_case(self, native, first):
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'radio.json'
            config.write_text(json.dumps({'jukebox':{'native':{}}} if native else {}))
            sessions=SimpleNamespace(radio_lock=threading.Lock(),radio_control=None,radio_jukebox=None,radio_jukebox_runner=None,accounts_db=Path(tmp)/'accounts.db',state=SimpleNamespace(radio_json=config),jobs=Mock())
            tree=ast.parse((Path(__file__).resolve().parents[1]/'examples/meowdoku/server.py').read_text())
            names={'_radio_playback','_radio_jukebox_services','_radio_jukebox'}
            methods=[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name in names]
            scope={'sessions':sessions,'json':json}
            exec(compile(ast.Module(body=methods,type_ignores=[]),'server-radio-methods','exec'),scope)
            Handler=type('Handler',(),{name:scope[name] for name in names});handler=Handler()
            with patch('meowdoku.nodus_radio_control.http_route') as pa_route,patch('meowdoku.nodus_radio_jukebox.http_route') as queue_route:
                getattr(handler,first)('GET')
                handler._radio_playback('GET');handler._radio_jukebox('GET')
                self.assertIsNone(sessions.radio_control.worker)
                self.assertIsNone(sessions.radio_jukebox_runner.worker)
                self.assertFalse(sessions.jobs.begin.called)
                self.assertEqual(sessions.radio_control.source_provider,sessions.radio_jukebox_runner if native else None)
                self.assertEqual(queue_route.call_args.args[2],sessions.radio_jukebox)
                self.assertEqual(pa_route.call_args.args[2],sessions.radio_control)
                original=sessions.radio_jukebox_runner
                handler._radio_jukebox('GET');handler._radio_playback('GET')
                self.assertIs(sessions.radio_jukebox_runner,original)
    def test_jukebox_pa_first_is_inert_and_shared(self):self.run_case(True,'_radio_playback')
    def test_jukebox_queue_first_is_inert_and_shared(self):self.run_case(True,'_radio_jukebox')
    def test_fixed_station_keeps_existing_subscription(self):self.run_case(False,'_radio_playback')

if __name__=='__main__':unittest.main()
