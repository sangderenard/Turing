"""Node check of web/wheel_blur.js: the waterwheel motion blur's regime choice, smear weights and polar blur raster."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_wheel_blur_helpers():
    done = subprocess.run(["node", str(ROOT / "tests" / "meowdoku_wheel_blur_check.mjs")], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
