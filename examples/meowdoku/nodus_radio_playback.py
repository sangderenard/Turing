"""Explicit, process-owned PA sink for an already-authorized PCM subscription.

No listener, account bypass or automatic device opening. The service must
authorize the owner/device/subscription before calling start(). One child owns
one Nodus canvas and SDL device; stop never touches another user's process.
"""
from collections import deque
from pathlib import Path
import os
import subprocess
import threading
import time


class Playback:
    def __init__(self, executable, *, device_id=0):
        self.executable = str(Path(executable).resolve())
        if not isinstance(device_id, int) or not 0 <= device_id <= 0xffffffff:
            raise ValueError('invalid SDL playback device')
        self.device_id = device_id
        self.process = self.reader = None
        self.ready = threading.Event()
        self.events = deque(maxlen=32)
        self.finished = False

    def start(self):
        """Explicit user-controlled transition: this opens the audio device."""
        if self.process is not None:
            raise RuntimeError('playback already started; create a new owned sink')
        if not Path(self.executable).is_file():
            raise FileNotFoundError(self.executable)
        self.process = subprocess.Popen([self.executable, '--device', str(self.device_id)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.reader = threading.Thread(target=self._read_status, name='owned-pcm-output', daemon=True)
        self.reader.start()
        deadline = time.perf_counter() + 4
        while not self.ready.wait(.01) and self.process.poll() is None and time.perf_counter() < deadline:
            pass
        if not self.ready.is_set():
            self.stop()
            raise RuntimeError('native PCM output did not become ready: ' + '\n'.join(self.events))

    def _read_status(self):
        try:
            for raw in self.process.stdout:
                line = raw.decode('utf-8', errors='replace').rstrip()[:1024]
                if line.startswith('pcm_sink_'):
                    self.events.append(line)
                    if line.startswith('pcm_sink_ready '): self.ready.set()
        finally:
            self.process.stdout.close()

    def write(self, pcm):
        if not self.ready.is_set() or self.finished or self.process.poll() is not None:
            raise RuntimeError('PCM sink is not accepting data')
        if not isinstance(pcm, bytes) or not 4 <= len(pcm) <= 1024 or len(pcm) % 4:
            raise ValueError('expected 1..256 stereo signed16 little-endian frames')
        self.process.stdin.write(pcm)
        self.process.stdin.flush()

    def finish(self):
        """Close only this stream, drain its bounded queues, then release device."""
        if self.process is None: raise RuntimeError('playback was not started')
        if self.finished: return
        self.finished = True
        try:
            self.process.stdin.close()
            result = self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.stop()
            raise RuntimeError('PCM sink did not drain within its bounded timeout')
        except OSError:
            self.stop()
            raise
        self.reader.join(timeout=1)
        if result:
            raise RuntimeError('PCM sink failed: ' + '\n'.join(self.events))

    def stop(self):
        """User cancellation, not graceful EOF. Own child only; no global kill."""
        self.finished = True
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=3)
            if self.process.stdin and not self.process.stdin.closed:
                try: self.process.stdin.close()
                except OSError: pass
            if self.reader: self.reader.join(timeout=1)

    def consume(self, subscription, stop_event):
        """Run on the service's owned worker, not a request/UI thread.

        Requires an explicitly started sink. Caller retains subscription/port
        ownership and closes it. Cancellation stops output; natural RTCP EOF
        drains it. No silence timeout is promoted to a successful completion.
        """
        if not self.ready.is_set() or self.finished:
            raise RuntimeError('start playback explicitly before consuming')
        try:
            while not stop_event.is_set():
                if subscription.error: raise RuntimeError(subscription.error)
                block = subscription.take_due()
                if block is not None: self.write(block)
                status = subscription.buffer.status()
                if status['ended']:
                    self.finish()
                    return
                deadline = status['next_read_at']
                stop_event.wait(.001 if deadline is None else
                                max(.0001, min(.005, deadline-time.perf_counter())))
        finally:
            if not self.finished: self.stop()
