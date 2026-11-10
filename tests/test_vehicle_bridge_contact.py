"""Exercise the current closed-solid tire boundary without a full build."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_bridge_faces_and_underpass_reach_contact_law_inputs():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the authored worker geometry")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, str(root / "tests/vehicle_bridge_contact_check.cjs")],
                            cwd=root, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: closed-solid" in result.stdout
