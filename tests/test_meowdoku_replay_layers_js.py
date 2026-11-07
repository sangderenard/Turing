"""Node check of web/replay_layers.js: layer sync, identity join, divergence metric, memory cap."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_replay_layers_helpers():
    done = subprocess.run(["node", str(ROOT / "tests" / "meowdoku_replay_layers_check.mjs")], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
