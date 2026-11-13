"""Explicit jukebox lifecycle around the existing native Radio canvas rows.

Construction/GET never launches a source. Operator catalogue paths only; the
producer is silent (SDL dummy), while its real NRAD and PCM outputs are retained.
"""
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
from meowdoku.nodus_radio_control import ControlError
from meowdoku.nodus_radio_pcm import Subscription


class NativeTrack:
    def __init__(self, config, song, *, subscription=Subscription, popen=subprocess.Popen):
        self.sub = None
        self.process = None
        self.error = None
        self.lines = queue.Queue(maxsize=64)
        self.reader = None
        self.spectrum_peer = None
        self.on_pcm = None
        self.config, self.song = config, song
        self.subscription, self.popen = subscription, popen

    def start(self, cancel):
        c = self.config
        argv = [c['executable'], c['scope_dll'], c['fft_dll'], '127.0.0.1',
                str(c['spectrum_port']), '0', '5', '1024', self.song['path'],
                '0', str(c['pcm_port']), '1']
        self.process = self.popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
            env=dict(os.environ, SDL_AUDIODRIVER='dummy'),
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        def read():
            for line in self.process.stdout:
                # Consume diagnostics continuously; never let a full pipe stall native EOF.
                if line.startswith(('radio_pcm_source_port=', 'radio_waiting_for_start=', 'radio_udp_source_port=')):
                    try: self.lines.put_nowait(line)
                    except queue.Full: self.error = 'Native source readiness overflow'
        self.reader = threading.Thread(target=read, daemon=True, name='radio-source-output')
        self.reader.start()
        deadline = time.perf_counter() + 10
        pin = None
        ready = False
        while not (pin and ready):
            if cancel.is_set(): return
            if self.process.poll() is not None or time.perf_counter() >= deadline:
                raise RuntimeError('Native source did not announce a gated PCM stream')
            try: line = self.lines.get(timeout=.05)
            except queue.Empty: continue
            match = re.fullmatch(r'radio_pcm_source_port=(\d+) rtp_payload=96 encoding=L16 rate=48000 channels=2 ssrc=(\d+)\s*', line)
            if match: pin = int(match[1]), int(match[2])
            if line.startswith('radio_udp_source_port='):
                self.spectrum_peer = '127.0.0.1:' + str(int(line.strip().split('=')[1]))
            if line.strip() == 'radio_waiting_for_start=1': ready = True
        self.sub = self.subscription(router_port=c['router_port'], listen_port=c['pcm_port'],
            peer=('127.0.0.1', pin[0]), ssrc=pin[1])
        if not self.sub.ready.wait(3) or self.sub.error:
            raise RuntimeError('PCM receiver was not ready; source remains held')
        if not cancel.is_set():
            self.process.stdin.write('START\n'); self.process.stdin.flush()

    def run(self, cancel):
        self.start(cancel)
        exit_at = None
        while not cancel.is_set():
            if self.sub.error: raise RuntimeError(self.sub.error)
            block = self.sub.take_due()
            if block is not None and self.on_pcm is not None: self.on_pcm(block)
            ended = self.sub.buffer.status()['ended']
            code = self.process.poll()
            if code is not None:
                if code != 0: raise RuntimeError('Native source failed')
                if ended: return
                if exit_at is None: exit_at = time.perf_counter()
                if time.perf_counter() - exit_at > 3:
                    raise RuntimeError('Native source exited without verified PCM EOF')
            time.sleep(.001)

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=5)
            if self.process.stdin: self.process.stdin.close()
            if self.reader: self.reader.join(timeout=2)
            if self.process.stdout: self.process.stdout.close()
        if self.sub is not None: self.sub.close()


class PcmTap:
    """Bounded fanout of already paced PCM, not a second jitter/transport engine."""
    def __init__(self):
        self.blocks = queue.Queue(maxsize=64)
        self.error = None
        self.closed = False

    def put(self, block):
        if self.closed or self.error: return
        try: self.blocks.put_nowait(block)
        except queue.Full: self.error = 'PA reader fell behind the bounded PCM feed'

    def take(self, timeout=.05):
        if self.error: raise RuntimeError(self.error)
        if self.closed: raise RuntimeError('Jukebox PCM feed closed')
        try: return self.blocks.get(timeout=timeout)
        except queue.Empty: return None

    def close(self):
        self.closed = True
        while True:
            try: self.blocks.get_nowait()
            except queue.Empty: break


class Runner:
    def __init__(self, store, config_path, jobs, *, track=NativeTrack):
        self.store, self.config_path, self.jobs, self.track = store, Path(config_path), jobs, track
        self.lock = threading.RLock()
        self.worker = None
        self.stop = threading.Event()
        self.cancel = threading.Event()
        self.outcome = 'interrupted'
        self.error = None
        self.current_track = None
        self.current_song = None
        self.taps = set()

    def subscribe_pcm(self):
        # This does not start the source or a device. The authorized PA manager
        # owns the returned tap and releases it when its explicit request ends.
        with self.lock:
            tap = PcmTap(); self.taps.add(tap); return tap

    def _publish_pcm(self, block):
        with self.lock:
            self.taps = {tap for tap in self.taps if not tap.closed}
            for tap in self.taps: tap.put(block)

    def status(self):
        with self.lock:
            return {"running": self.worker is not None and self.worker.is_alive(),
                    "error": "Native jukebox stopped; inspect service diagnostics" if self.error else None}

    def station(self):
        with self.lock:
            if self.current_track is None or self.current_track.spectrum_peer is None: return None
            return {"peer": self.current_track.spectrum_peer, "title": self.current_song["title"]}

    def _config(self):
        try:
            root = json.loads(self.config_path.read_text(encoding='utf-8'))
            c = root['jukebox']['native']
            if root.get('enabled') is not True or root.get('source') != 'file': raise ValueError()
            for key in ('executable', 'scope_dll', 'fft_dll'):
                if not Path(c[key]).is_absolute() or not Path(c[key]).is_file(): raise ValueError()
            for key in ('router_port', 'pcm_port', 'spectrum_port'):
                if type(c[key]) is not int or not 0 < c[key] < 65536: raise ValueError()
            if c['pcm_port'] == c['spectrum_port']: raise ValueError()
            return c
        except (OSError, ValueError, KeyError, TypeError):
            raise ControlError('Native jukebox source is not configured', 503)

    def resume(self, actor):
        config = self._config()
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                if self.stop.is_set(): raise ControlError('Jukebox is stopping; try again shortly', 409)
                self.store.pause(actor, False)
                return
            # A previous service may have died. Do not silently replay or erase its claim.
            if self.store.status(actor, admin=True)['current'] is not None:
                raise ControlError('Previous playback requires interrupted recovery before resuming', 409)
            self.store.pause(actor, False)
            self.stop = threading.Event()
            self.worker = threading.Thread(target=self._run, args=(config,), daemon=True, name='radio-jukebox')
            self.worker.start()

    def pause(self, actor):
        # Pause the rotation after the current song. Skip is the explicit interrupt.
        self.store.pause(actor, True)

    def skip(self, actor):
        with self.lock:
            if self.worker is None or not self.worker.is_alive():
                raise ControlError('There is no owned playback to skip', 409)
            if self.current_track is None:
                raise ControlError('There is no current source to skip', 409)
            self.store.audit(actor, 'skip')
            self.outcome = 'skipped'
            self.cancel.set()

    def close(self):
        with self.lock:
            self.stop.set(); self.outcome = 'interrupted'; self.cancel.set()
        if self.worker: self.worker.join(timeout=8)
        with self.lock:
            for tap in self.taps: tap.close()
            self.taps.clear()

    def _run(self, config):
        while not self.stop.is_set():
            owned = None; claim = None; registered = False
            try:
                with self.lock:
                    if self.stop.is_set(): break
                    claim = self.store.claim()
                    self.cancel = threading.Event(); self.outcome = 'interrupted'
                if claim is None:
                    self.stop.wait(.1); continue
                if not self.jobs.begin('radio_jukebox', claim['token'], owner=f"account:{claim['user_id']}", label='Radio requested track'):
                    raise RuntimeError('Could not register source ownership')
                registered = True
                song = self.store.catalogue()[claim['song_id']]
                path = Path(song['path'])
                if not path.is_absolute() or not path.is_file(): raise RuntimeError('Catalogue source unavailable')
                owned = self.track(config, song)
                owned.on_pcm = self._publish_pcm
                with self.lock:
                    self.current_track, self.current_song = owned, song
                owned.run(self.cancel)
                status = self.outcome if self.cancel.is_set() else 'completed'
            except Exception as exc:
                self.error = str(exc)
                print(f'[radio-jukebox] Source failed: {exc}', flush=True)
                with self.lock:
                    for tap in self.taps: tap.error = 'Jukebox source failed'
                status = 'failed'
                # Stop on a failure instead of burning every queued request.
                self.stop.set()
            finally:
                with self.lock:
                    self.current_track = self.current_song = None
                if owned is not None:
                    try: owned.close()
                    except Exception as exc: self.error = str(exc); status = 'failed'; self.stop.set()
                if claim is not None:
                    try:
                        if registered and not self.jobs.end('radio_jukebox', claim['token']):
                            self.error = 'Source ownership deregistration failed'
                            status = 'failed'; self.stop.set()
                    finally:
                        self.store.finish(claim['token'], status)
