"""Silent real native PA ownership test. Argument: nodus_pcm_playback.exe.
SDL dummy is explicitly selected for every child; never uses speakers.
"""
import os
from pathlib import Path
import re
import struct
import sys
import threading
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from meowdoku.nodus_radio_pcm import JitterBuffer, PcmPacket
from meowdoku.nodus_radio_playback import Playback


class Source:
    error = None
    def __init__(self):
        self.buffer = JitterBuffer(ssrc=42)
        for i in range(20):
            self.buffer.put(PcmPacket(i, i*256, 42, struct.pack('<512h', *([1234, -2345]*256))))
        self.buffer.put(PcmPacket(20, 5120, 42, struct.pack('<14h', *range(14))))
        self.buffer.finish_timestamp(5127)
    def take_due(self): return self.buffer.take_due()


def main():
    with patch.dict(os.environ, SDL_AUDIODRIVER='dummy'):
        sink = Playback(sys.argv[1])
        assert sink.process is None, 'construction must not open audio'
        try:
            sink.start()
            try:
                sink.write(b'x')
                raise AssertionError('misaligned sample accepted')
            except ValueError: pass
            source = Source()
            sink.consume(source, threading.Event())
            done = next(line for line in sink.events if line.startswith('pcm_sink_done '))
            assert 'frames=5127 ' in done and 'drained=1 result=0' in done, done
            match = re.search(r'max_queue_frames=(\d+)', done)
            assert int(match[1]) <= 2304, done
            assert source.buffer.lost == 0
            print('PASS explicit start, native canvas FIFO, SDL resampler/drain, short final block:', done)
        finally: sink.stop()
        # A separate explicitly owned process cancels without affecting others.
        cancel = Playback(sys.argv[1])
        cancel.start()
        cancel.stop()
        assert cancel.process.poll() is not None
        print('PASS idle user cancellation releases owned process; no auto start on import/construct')
        eager = Playback(sys.argv[1])
        try:
            eager.start()
            block = struct.pack('<512h', *([1234, -2345]*256))
            for _ in range(375): eager.write(block)
            eager.finish()
            done = next(line for line in eager.events if line.startswith('pcm_sink_done '))
            assert 'frames=96000 ' in done and 'drained=1 result=0' in done, done
            assert int(re.search(r'max_queue_frames=(\d+)', done)[1]) <= 2304, done
            print('PASS eager producer backpressure, no lost blocks:', done)
        finally: eager.stop()
        invalid = Playback(sys.argv[1], device_id=123456789)
        try:
            try:
                invalid.start()
                raise AssertionError('invalid device fell back to another device')
            except RuntimeError:
                assert invalid.process.poll() is not None
                print('PASS invalid-device refusal; no fallback device and no orphan process')
        finally: invalid.stop()


if __name__ == '__main__': main()
