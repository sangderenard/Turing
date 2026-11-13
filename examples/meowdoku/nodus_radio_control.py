"""Account-gated PA control for the configured private Nodus Radio station.

No route accepts executable paths, ports, peers, devices or credentials. Those
are operator-owned local state. Opening a menu or GET never starts anything.
"""
import json
from pathlib import Path
import threading
import uuid
from meowdoku.nodus_radio_pcm import Subscription
from meowdoku.nodus_radio_playback import Playback


class ControlError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class Manager:
    def __init__(self, config_path, jobs, *, subscription=Subscription, playback=Playback):
        self.config_path, self.jobs = Path(config_path), jobs
        self.subscription, self.playback = subscription, playback
        self.lock = threading.Lock()
        self.worker = None
        self.cancel = threading.Event()
        self.state = {'status': 'stopped', 'owner_id': None, 'job_id': None, 'error': None}

    def _config(self):
        try:
            config = json.loads(self.config_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            raise ControlError('Radio PA is not configured', 503)
        if not isinstance(config, dict) or not isinstance(config.get('playback', {}), dict):
            raise ControlError('Invalid Radio PA configuration', 503)
        pa = config.get('playback', {})
        if config.get('enabled') is not True or config.get('source') != 'file' or pa.get('enabled') is not True:
            raise ControlError('Radio PA is disabled; an operator must select a file-source station', 503)
        for key in ('router_port', 'listen_port', 'peer_port'):
            if type(pa.get(key)) is not int or not 0 < pa[key] < 65536:
                raise ControlError('Invalid configured Radio PA port', 503)
        if type(pa.get('ssrc')) is not int or not 0 <= pa['ssrc'] <= 0xffffffff:
            raise ControlError('Invalid configured Radio PA source', 503)
        if pa.get('peer_host') != '127.0.0.1':
            raise ControlError('Radio PA currently requires the private loopback producer', 503)
        if type(pa.get('device_id')) is not int or not 0 <= pa['device_id'] <= 0xffffffff:
            raise ControlError('Invalid configured Radio PA device', 503)
        executable = pa.get('executable')
        if not isinstance(executable, str) or not Path(executable).is_absolute() or not Path(executable).is_file():
            raise ControlError('The configured native Radio PA sink is unavailable', 503)
        return pa

    def status(self, owner):
        with self.lock:
            state = dict(self.state)
        state['can_stop'] = state['owner_id'] == owner and state['status'] in ('starting', 'playing', 'stopping')
        try:
            self._config()
            state['configured'] = True
        except ControlError:
            state['configured'] = False
        return state

    def start(self, owner):
        config = self._config()
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                raise ControlError('Server PA is already owned by an active request', 409)
            job_id = uuid.uuid4().hex
            if not self.jobs.begin('radio_pa', job_id, owner=f'account:{owner}', label='Explicit server-speaker playback'):
                raise ControlError('Could not register PA ownership; nothing was started', 503)
            self.cancel = threading.Event()
            self.state = {'status': 'starting', 'owner_id': owner, 'job_id': job_id, 'error': None}
            self.worker = threading.Thread(target=self._run, args=(config, job_id), name='owned-radio-pa', daemon=True)
            try: self.worker.start()
            except Exception:
                self.jobs.end('radio_pa', job_id)
                self.state['status'] = 'failed'
                raise
        return self.status(owner)

    def stop(self, owner):
        with self.lock:
            if self.state['owner_id'] != owner:
                raise ControlError('Only the account that started this PA stream can stop it here', 403)
            if self.worker is not None and self.worker.is_alive():
                self.state['status'] = 'stopping'
                self.cancel.set()
        return self.status(owner)

    def close(self):
        self.cancel.set()
        worker = self.worker
        if worker is not None:
            worker.join(timeout=8)

    def _run(self, config, job_id):
        sub = sink = None
        error = None
        try:
            sub = self.subscription(router_port=config['router_port'], listen_port=config['listen_port'],
                peer=(config['peer_host'], config['peer_port']), ssrc=config['ssrc'])
            if not sub.ready.wait(3) or sub.error:
                raise RuntimeError('Private PCM router/source was not ready')
            if self.cancel.is_set(): return
            sink = self.playback(config['executable'], device_id=config['device_id'])
            sink.start()
            with self.lock:
                if self.cancel.is_set(): return
                self.state['status'] = 'playing'
            sink.consume(sub, self.cancel)
        except Exception as exc:
            error = f'{type(exc).__name__}: Radio PA failed; inspect the server diagnostic'
            print(f'[radio-pa] {job_id}: {exc}', flush=True)
        finally:
            for resource, release in ((sink, 'stop'), (sub, 'close')):
                if resource is not None:
                    try:
                        getattr(resource, release)()
                    except Exception as exc:
                        error = 'Radio PA cleanup failed; inspect the server diagnostic'
                        print(f'[radio-pa] {job_id} cleanup: {exc}', flush=True)
            try:
                if not self.jobs.end('radio_pa', job_id):
                    error = 'Radio PA stopped but job deregistration failed'
            finally:
                with self.lock:
                    self.state.update(status='failed' if error else 'stopped', error=error)



def http_route(handler, method, manager):
    handler._classify()
    principal = handler._principal()
    if principal.user_id is None:
        return handler._json({'error': 'Sign in to control server PA'}, 401)
    if 'account_manage' not in handler._caps():
        return handler._json({'error': 'Server-speaker control requires account_manage'}, 403)
    if method == 'GET':
        return handler._json(manager.status(principal.user_id))
    origin = handler.headers.get('Origin')
    expected = ('https://' if handler._is_secure() else 'http://') + handler.headers.get('Host', '')
    if ((origin and origin != expected) or handler.headers.get('Sec-Fetch-Site') == 'cross-site' or
            handler.headers.get('X-Meowdoku-Radio') != '1'):
        return handler._json({'error': 'Same-origin Radio control header required'}, 403)
    if handler.headers.get('Content-Type', '').split(';', 1)[0].strip().lower() != 'application/json':
        return handler._json({'error': 'JSON only'}, 415)
    if handler.headers.get('Transfer-Encoding') or len(handler.headers.get_all('Content-Length', [])) != 1:
        return handler._json({'error': 'One Content-Length is required'}, 400)
    body = handler._body_json(1024)
    if body is None: return
    if not isinstance(body, dict) or set(body) != {'action'} or body['action'] not in ('start', 'stop'):
        return handler._json({'error': 'Choose start or stop; station/device settings are operator-managed'}, 400)
    try:
        return handler._json(getattr(manager, body['action'])(principal.user_id), 202)
    except ControlError as error:
        return handler._json({'error': str(error)}, error.status)
