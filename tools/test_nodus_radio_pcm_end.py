"""Silent native integration. Args: radio.exe scope.dll fft.dll udp_router.exe.

Creates disposable two-second PCM fixture with seven-frame final block, selects
SDL dummy, and runs only isolated loopback processes. No service/PA output.
"""
from pathlib import Path
import math
import os
import re
import socket
import struct
import subprocess
import sys
import tempfile
import time
import wave
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from meowdoku.nodus_radio_pcm import Subscription
from meowdoku.nodus_radio_playback import Playback


def port(kind):
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def main():
    radio, scope, fft, router = (str(Path(p).resolve()) for p in sys.argv[1:5])
    sink = Playback(sys.argv[5]) if len(sys.argv) > 5 else None
    control, destination = port(socket.SOCK_STREAM), port(socket.SOCK_DGRAM)
    processes, sub = [], None
    with tempfile.TemporaryDirectory(prefix='radio-end-proof-') as td:
        fixture = Path(td) / 'short-final.wav'
        frames = 48000 * 2 + 7
        with wave.open(str(fixture), 'wb') as wav:
            wav.setparams((2, 2, 48000, frames, 'NONE', 'not compressed'))
            wav.writeframes(b''.join(struct.pack('<hh', *([int(8000 * math.sin(i * .1))] * 2))
                                     for i in range(frames)))
        log = Path(td) / 'native.log'
        with log.open('w') as output:
            try:
                if sink:
                    with patch.dict(os.environ, SDL_AUDIODRIVER='dummy'): sink.start()
                processes.append(subprocess.Popen([router, '127.0.0.1', str(control)],
                    cwd=td, stdout=output, stderr=output))
                deadline = time.perf_counter() + 5
                while True:
                    try:
                        with socket.create_connection(('127.0.0.1', control), .1): pass
                        break
                    except OSError:
                        assert time.perf_counter() < deadline, 'router not ready'
                        time.sleep(.02)
                producer = subprocess.Popen([radio, scope, fft, '127.0.0.1',
                    str(port(socket.SOCK_DGRAM)), '10', '5', '1024', str(fixture), '0', str(destination)],
                    cwd=td, stdout=output, stderr=output, env=dict(os.environ, SDL_AUDIODRIVER='dummy'))
                processes.append(producer)
                match = None
                while time.perf_counter() < deadline:
                    match = re.search(r'radio_pcm_source_port=(\d+).*ssrc=(\d+)', log.read_text(errors='replace'))
                    if match: break
                    time.sleep(.001)
                assert match, 'producer not ready'
                sub = Subscription(router_port=control, listen_port=destination,
                    peer=('127.0.0.1', int(match[1])), ssrc=int(match[2]))
                assert sub.ready.wait(2), sub.error
                deadline = time.perf_counter() + 8
                blocks = nonzero = 0
                last_size = None
                while not sub.buffer.ended:
                    assert time.perf_counter() < deadline, (sub.buffer.status(), sub.error)
                    value = sub.take_due()
                    if value is not None:
                        blocks += 1; nonzero += any(value); last_size = len(value)
                        if sink: sink.write(value)
                    time.sleep(.0005)
                producer.wait(timeout=3)
                assert producer.returncode == 0
                state = sub.buffer.status()
                assert blocks == state['received'] and blocks > 300, state
                assert last_size == 7 * 4, last_size
                assert nonzero == blocks and state['lost'] == state['late'] == state['resyncs'] == 0, state
                assert sub.invalid == sub.foreign == 0 and sub.error is None
                assert sub.take_due() is None
                if sink:
                    sink.finish()
                    print('PASS owned native PA sink:', '\n'.join(sink.events))
                print('PASS native canvas / RTP+RTCP / router / adaptive readback:', state)
                print('Final block bytes:', last_size, '; no post-EOF silence; SDL dummy only')
            finally:
                if sink: sink.stop()
                if sub: sub.close()
                for process in reversed(processes):
                    if process.poll() is None: process.terminate(); process.wait(timeout=3)
                print('\n'.join(line for line in log.read_text(errors='replace').splitlines()
                    if line.startswith(('radio_pcm_', 'pcm_', 'shutdown_'))))


if __name__ == '__main__': main()
