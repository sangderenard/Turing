"""Node check of web/world_curve.js: the world curve slider's notches, its ease, the dolly zoom whose limit is the orthographic
camera, particle sizing and picking under it, and the control's share-link spec (tests/meowdoku_world_curve_check.mjs)."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_world_curve_helpers():
    done = subprocess.run(["node", str(ROOT / "tests" / "meowdoku_world_curve_check.mjs")], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
