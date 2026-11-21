"""The Nodus page's input forwarding never leaves a drag or a key held on the canvas (tests/meowdoku_nodus_forwarding_check.cjs)."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not shutil.which("node"), reason="Node required for the page's isolated behavior check")
def test_every_press_forwarded_to_a_canvas_is_released():
    done = subprocess.run([shutil.which("node"), str(ROOT / "tests" / "meowdoku_nodus_forwarding_check.cjs")],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.count("ok   ") == 6
