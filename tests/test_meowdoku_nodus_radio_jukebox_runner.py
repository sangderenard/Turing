import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from meowdoku.nodus_radio_jukebox import Jukebox
from meowdoku.nodus_radio_jukebox_runner import Runner

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        p=Path(self.tmp.name);self.file=p/'asset';self.file.write_bytes(b'x')
        self.config=p/'radio.json'
        native={k:str(self.file) for k in ('executable','scope_dll','fft_dll')}
        native.update(router_port=10001,pcm_port=10002,spectrum_port=10003)
        self.config.write_text(json.dumps({'enabled':True,'source':'file','jukebox':{'native':native,'songs':[{'id':'a','title':'A','path':str(self.file)}]}}))
        self.store=Jukebox(p/'db',self.config)
        self.jobs=Mock();self.jobs.begin.return_value=True
    def wait(self,predicate):
        end=time.monotonic()+3
        while not predicate():
            self.assertLess(time.monotonic(),end);time.sleep(.01)
    def finished(self):
        with self.store.connect() as db:
            row=db.execute('SELECT status FROM radio_plays').fetchone()
            return row is not None and row[0] != 'playing'
    def test_inert_until_resume_and_success_finishes_claim(self):
        track=Mock();track.return_value.spectrum_peer=None
        runner=Runner(self.store,self.config,self.jobs,track=track);self.addCleanup(runner.close)
        self.store.request(1,'a');self.assertFalse(track.called)
        runner.resume(9);self.wait(self.finished)
        self.assertEqual(track.call_count,1);track.return_value.close.assert_called_once()
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT status FROM radio_plays').fetchone()[0],'completed')
    def test_failed_start_does_not_burn_rest_of_queue(self):
        track=Mock();track.return_value.run.side_effect=RuntimeError('bad source')
        runner=Runner(self.store,self.config,self.jobs,track=track);self.addCleanup(runner.close)
        self.store.request(1,'a');self.store.request(2,'a');runner.resume(9)
        self.wait(self.finished)
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT status FROM radio_plays').fetchone()[0],'failed')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM radio_requests').fetchone()[0],1)
    def test_skip_cancels_owned_source(self):
        entered=threading.Event()
        class Track:
            spectrum_peer=None
            def __init__(self,*a):pass
            def run(self,cancel):entered.set();cancel.wait(2)
            def close(self):pass
        runner=Runner(self.store,self.config,self.jobs,track=Track);self.addCleanup(runner.close)
        self.store.request(1,'a');runner.resume(9);self.assertTrue(entered.wait(2));runner.skip(9)
        self.wait(self.finished)
        with self.store.connect() as db:self.assertEqual(db.execute('SELECT status FROM radio_plays').fetchone()[0],'skipped')
    def test_pcm_tap_is_bounded_and_close_releases_pending(self):
        from meowdoku.nodus_radio_jukebox_runner import PcmTap
        tap=PcmTap()
        for _ in range(65):tap.put(b'1234')
        self.assertEqual(tap.blocks.qsize(),64)
        with self.assertRaisesRegex(RuntimeError,'fell behind'):tap.take()
        tap.close();self.assertEqual(tap.blocks.qsize(),0)

    def test_previous_current_refuses_implicit_replay(self):
        self.store.request(1,'a');self.store.pause(9,False);self.store.claim()
        runner=Runner(self.store,self.config,self.jobs,track=Mock())
        with self.assertRaisesRegex(Exception,'interrupted recovery'):runner.resume(9)
        self.assertFalse(self.jobs.begin.called)

if __name__=='__main__':unittest.main()
