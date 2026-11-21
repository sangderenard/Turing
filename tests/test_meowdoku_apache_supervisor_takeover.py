"""A standby piped logger takes over when the manager lock is released."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
SUP = ROOT / "examples" / "meowdoku" / "deploy" / "apache_supervisor.py"


def _module():
    spec = importlib.util.spec_from_file_location("supervisor_takeover", SUP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standby_releases_failed_lock_handle_and_can_acquire_later(tmp_path):
    module = _module()
    config = tmp_path / "instances.json"
    config.write_text(json.dumps({"instances": []}))
    manager = module.Supervisor(tmp_path / "accounts.db", config, tmp_path / "state")
    standby = module.Supervisor(tmp_path / "accounts.db", config, tmp_path / "state")
    assert manager.acquire_lock()
    try:
        assert not standby.acquire_lock()
        assert standby._lock_file is None
    finally:
        manager._lock_file.close()
    assert standby.acquire_lock()
    standby._lock_file.close()


def test_standby_takeover_reuses_the_existing_pipe_reader(monkeypatch):
    module = _module()
    started = []

    class Reader:
        def __init__(self, **kwargs):
            assert kwargs["daemon"] is True
            started.append(kwargs["target"])

        def start(self):
            pass

    class Stop:
        def wait(self, _seconds):
            return False

        def is_set(self):
            return False

    class Standby:
        stop = Stop()

        def __init__(self):
            self.logs = []
            self.promotions = []

        def tail_stdin(self):
            pass

        def log(self, line):
            self.logs.append(line)

        def acquire_lock(self):
            return True

        def run(self, **kwargs):
            self.promotions.append(kwargs)
            return 37

    monkeypatch.setattr(module.threading, "Thread", Reader)
    monkeypatch.setattr(module.signal, "signal", lambda *_args: None)
    standby = Standby()
    assert module.Supervisor.run(standby, manage=False) == 37
    assert started == [standby.tail_stdin]
    assert standby.promotions == [{"manage": True, "reader_started": True}]
    assert any("taking over" in line for line in standby.logs)
