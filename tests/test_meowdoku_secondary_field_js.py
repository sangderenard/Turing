"""CPU orchestration contract; real GPU/browser acceptance is reported separately."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_secondary_field_orchestration():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for browser module orchestration')
    script = Path(__file__).with_name('meowdoku_secondary_field_check.mjs')
    subprocess.run([node, str(script)], check=True, timeout=30, cwd=script.parent.parent)
