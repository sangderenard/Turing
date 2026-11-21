"""An unreadable scheduler directory must not make the presence endpoint fail."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku import agent_presence as presence  # noqa: E402


def test_inaccessible_scheduler_directory_keeps_presence_available(tmp_path, monkeypatch):
    running = tmp_path / "running"
    original_is_dir = Path.is_dir

    def is_dir(path):
        if path == running:
            raise PermissionError("scheduler directory is not readable")
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    result = presence.snapshot(tmp_path / "missing.db", tmp_path, now=100)
    codex = next(agent for agent in result["agents"] if agent["username"] == "codex")
    assert codex["runs"] == []
    assert codex["channels"]["local"]["state"] == "unknown"
