"""Opt-in receiver lifecycle against the real browser modules (Node CPU gate)."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_studio_light_field_receiver_and_request_contract():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the browser-module contract test')
    script = Path(__file__).with_name('meowdoku_studio_light_field_check.mjs')
    subprocess.run([node, str(script)], check=True, timeout=30, cwd=script.parent.parent)
