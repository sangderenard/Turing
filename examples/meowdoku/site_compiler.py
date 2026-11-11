"""Wire the existing signed-in divider API to its Nodus bridge and workspace ledger.

Configuration is operator-owned startup JSON, never submitted page fields.
No new queue: the single worker drains source_divider's existing lease queue.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading

from meowdoku import compile_artifacts, divider_nodus_bridge as bridge
from meowdoku import source_divider, workspace_accounting


class SiteCompiler:
    def __init__(self, state, accounts, config, *, federation_port=None):
        if not isinstance(config, dict):
            raise ValueError('site compiler configuration must be an object')
        required = {'jobs_dir', 'service_id', 'capacity_bytes', 'user_quota_bytes',
                    'free_floor_bytes', 'reservation_bytes', 'review_after_s', 'backend'}
        if required - config.keys():
            raise ValueError('missing site compiler settings: ' + ', '.join(sorted(required - config.keys())))
        if config['backend'] not in bridge.DRIVER_BACKENDS:
            raise ValueError('site Nodus backend must be c, llvm or fortran')
        port = config.get('port', federation_port)
        if type(port) is not int or not 0 < port < 65536 or type(config['service_id']) is not int or config['service_id'] <= 0:
            raise ValueError('site compiler requires a valid contract port and service participant id')
        root = compile_artifacts.external_directory(config['jobs_dir'])
        # Explicit external user-space root; never provision a checkout fallback.
        # Existing ledger root/policy bindings remain immutable, no migration.
        root.mkdir(parents=True, exist_ok=True)
        store, _ = workspace_accounting.for_state(state, accounts, workspace_root=root,
                    capacity_bytes=config['capacity_bytes'], user_quota_bytes=config['user_quota_bytes'],
                    free_floor_bytes=config['free_floor_bytes'])
        artifacts = compile_artifacts.Artifacts(state, store, workspace_root=root,
                    reservation_bytes=config['reservation_bytes'], review_after_s=config['review_after_s'])
        self.route = bridge.NodusCompileRoute('127.0.0.1', port, config['service_id'], root,
                    nodus_tools=config.get('nodus_tools'), source_root=state.jobs_dir,
                    backend=config['backend'], turing_root=config.get('turing_root'), artifacts=artifacts)
        self.state = state
        self.backends = (config['backend'],)
        self.timeout_s = float(config.get('timeout_s', bridge.DEFAULT_TIMEOUT_S))
        if not 0 < self.timeout_s <= 3600:
            raise ValueError('site compiler timeout must be in (0, 3600] seconds')
        self._stop, self._wake = threading.Event(), threading.Event()
        self.thread = None
        self.last_error = None
        # Initialize the incumbent queue before the HTTP hook and worker can
        # concurrently make their first connection to a fresh state database.
        db = source_divider.connect(self.state.accounts_db)
        db.close()

    def __call__(self, job_id=None):
        self._wake.set()  # request thread returns promptly; SQL queue owns the job

    def start(self):
        if self.thread is not None:
            raise RuntimeError('site compiler already started')
        self.thread = threading.Thread(target=self._loop, name='site-nodus-compiler', daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._wake.set()

    def _loop(self):
        db = source_divider.connect(self.state.accounts_db)
        try:
            while not self._stop.is_set():
                try:
                    result = bridge.run_pending(db, self.state, self.route, max_pieces=1,
                             lease_owner='site-nodus-compiler', timeout_s=self.timeout_s)
                    self.last_error = None
                except Exception as error:
                    self.last_error = f'{type(error).__name__}: {error}'
                    result = []
                if not result or result[-1].get('state') in ('busy', 'unfinished') or self.last_error:
                    self._wake.wait(5.0)
                    self._wake.clear()
        finally:
            db.close()


def configure(sessions, config_path, *, federation_port=None):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    worker = SiteCompiler(sessions.state, sessions.accounts, config, federation_port=federation_port)
    sessions.divider_hook = worker
    return worker
