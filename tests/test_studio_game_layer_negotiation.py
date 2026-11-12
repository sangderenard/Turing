"""Offline checks of the actual opt-in Studio controller and admission contract."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("script", [
    "meowdoku_studio_game_protocol_check.mjs",
    "meowdoku_studio_game_layer_check.mjs",
    "meowdoku_studio_pose_stream_check.mjs",
    "meowdoku_studio_empty_return_check.mjs",
    "meowdoku_studio_authored_revision_check.mjs",
    "world_registry_append_check.mjs",
    "mechanical_creature_studio_topology_check.mjs",
])
def test_studio_game_layer_negotiation(script):
    result = subprocess.run(["node", str(ROOT / "tests" / script)], cwd=ROOT,
                            capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
