"""The real Studio head controls, shared geometry and MRP1 builder contract."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_modular_light_heads_use_shared_scene_and_ray_payload():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for Studio module checks")
    script = Path(__file__).with_name("meowdoku_modular_light_heads_check.mjs")
    subprocess.run([node, str(script)], check=True, timeout=30, cwd=script.parent.parent)
