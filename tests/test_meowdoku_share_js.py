"""Node check of web/share.js over the page's actual specs (tests/meowdoku_share_check.mjs)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_share_js_round_trips_the_pages_specs(tmp_path):
    from meowdoku.fountain import FountainGame
    from meowdoku.fountain_game import FountainGame as Fountain3D

    served = {"fountain": FountainGame().state()["controls_ui"], "fountain3d": Fountain3D().state()["controls_ui"]}
    path = tmp_path / "specs.json"
    path.write_text(json.dumps(served), encoding="utf-8")
    done = subprocess.run(["node", str(ROOT / "tests" / "meowdoku_share_check.mjs"), str(path)], capture_output=True, text=True, timeout=120)
    print(done.stdout)
    assert done.returncode == 0, done.stdout + done.stderr
