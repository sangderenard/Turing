"""Node check of web/wordle_layout.js: centred rack rows, the centred entry bar, and exact hit tests."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_wordle_layout_helpers():
    done = subprocess.run(["node", str(ROOT / "tests" / "meowdoku_wordle_layout_check.mjs")], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
