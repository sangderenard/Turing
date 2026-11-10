"""Offline lifecycle checks of the real optional game-layer host module."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_original_game_layer_host_async_disposal():
    result = subprocess.run(
        ["node", "--test", str(ROOT / "tests/mechanical_creature_studio_layer_host_check.mjs")],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
