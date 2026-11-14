import json
import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from meowdoku import nodus_radio_hls
from meowdoku.nodus_radio_jukebox_runner import NativeTrack, Runner
from meowdoku.state_paths import StatePaths


class RadioHlsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.state = StatePaths(self.root)
        self.config = self.state.radio_json
        self.config.parent.mkdir()
        self.binary = self.root / 'ffmpeg.exe'
        self.binary.write_bytes(b'fixture')
        self.hls = {'enabled': True, 'ffmpeg_executable': str(self.binary),
                    'core_roots': [str(self.root / 'core')]}
        self.config.write_text(json.dumps({'enabled': True, 'source': 'file',
            'jukebox': {'native': {'hls': self.hls}}}), encoding='utf-8')

    def test_declared_row_and_bounded_file_reads(self):
        environment = nodus_radio_hls.prepare(self.state, self.hls)
        declaration = Path(environment['NODUS_EXEC_ROWS']).read_text(encoding='utf-8')
        self.assertIn('row exec.radio.hls\n', declaration)
        self.assertIn('mode stream\n', declaration)
        self.assertIn('argument -probesize\nargument 32\n', declaration)
        self.assertIn('argument -muxdelay\nargument 0\nargument -muxpreload\nargument 0\n', declaration)
        self.assertIn('argument pipe:0\n', declaration)
        directory = nodus_radio_hls.output_dir(self.state)
        (directory / 'radio.m3u8').write_bytes(b'#EXTM3U\n')
        (directory / 'radio_003.ts').write_bytes(b'packet')
        self.assertEqual(nodus_radio_hls.read(self.state, 'radio.m3u8'),
                         (b'#EXTM3U\n', 'application/vnd.apple.mpegurl'))
        self.assertEqual(nodus_radio_hls.read(self.state, 'radio_003.ts'),
                         (b'packet', 'video/mp2t'))
        for name in ('../nodus_radio.json', 'radio_003.ts/other', 'radio_abc.ts'):
            with self.assertRaises(FileNotFoundError): nodus_radio_hls.read(self.state, name)
        nodus_radio_hls.clear_output(self.state)
        self.assertFalse((directory / 'radio.m3u8').exists())
        self.assertFalse((directory / 'radio_003.ts').exists())
        self.assertTrue(Path(environment['NODUS_EXEC_ROWS']).exists())

    def test_disabled_and_invalid_operator_configuration(self):
        self.hls['enabled'] = False
        self.config.write_text(json.dumps({'enabled': True, 'source': 'file',
            'jukebox': {'native': {'hls': self.hls}}}), encoding='utf-8')
        with self.assertRaises(FileNotFoundError): nodus_radio_hls.read(self.state, 'radio.m3u8')
        with self.assertRaises(ValueError): nodus_radio_hls.prepare(self.state, {
            'ffmpeg_executable': str(self.binary), 'core_roots': ['relative']})

    def test_runner_prepares_only_on_explicit_configuration(self):
        asset = self.root / 'asset.wav'
        asset.write_bytes(b'fixture')
        native = {key: str(asset) for key in ('executable', 'scope_dll', 'fft_dll')}
        native.update(router_port=10001, pcm_port=10002, spectrum_port=10003, hls=self.hls)
        self.config.write_text(json.dumps({'enabled': True, 'source': 'file',
            'jukebox': {'native': native, 'songs': []}}), encoding='utf-8')
        runner = Runner(None, self.config, None, state=self.state)
        self.assertFalse((self.config.parent / 'nodus_radio_hls_rows.txt').exists())
        prepared = runner._config()
        self.assertIn('NODUS_EXEC_ROWS', prepared['hls_env'])
        self.assertIs(prepared['hls_state'], self.state)
        calls = []
        class Process:
            stdout = iter(['radio_pcm_source_port=10004 rtp_payload=96 encoding=L16 rate=48000 channels=2 ssrc=9\n',
                           'radio_udp_source_port=10005\n', 'radio_waiting_for_start=1\n'])
            stdin = io.StringIO()
            def poll(self): return None
        process = Process()
        def popen(argv, **kwargs):
            calls.append((argv, kwargs))
            return process
        class Subscription:
            error = None
            ready = threading.Event()
            ready.set()
            def __init__(self, **kwargs): self.kwargs = kwargs
        track = NativeTrack(prepared, {'path': str(asset)}, subscription=Subscription, popen=popen)
        track.start(threading.Event())
        self.assertEqual(calls[0][0][-2:], ['1', '1'])
        self.assertEqual(calls[0][1]['env']['NODUS_EXEC_ROWS'], prepared['hls_env']['NODUS_EXEC_ROWS'])
        self.assertEqual(process.stdin.getvalue(), 'START\n')


if __name__ == '__main__': unittest.main()
