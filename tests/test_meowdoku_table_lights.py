"""Actual ten-emitter Studio upload reaches Spectral's fixture source reader."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from meowdoku import marble_photo as photos


def test_ten_emitter_wire_and_fixture_derived_sources():
    spectral = ROOT.parent / "spectral-analyzer"
    if not (spectral / "camera_designer" / "marble_preview_worker.py").is_file():
        pytest.skip("canonical sibling Spectral fixture reader is unavailable")
    sys.path.insert(0, str(spectral))
    from camera_designer.marble_preview_worker import _derive_lights_from_fixtures

    result = subprocess.run(
        ["node", "tests/meowdoku_marble_table_lights_check.mjs", "--json"],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=15)
    packed = json.loads(result.stdout)
    proof = np.zeros(1, dtype=[("positions", "<f4", (9,)),
                              ("normals", "<f4", (9,)), ("id", "<u4")])
    proof["positions"] = [0, 0, 0, 1, 0, 0, 0, 1, 0]
    proof["normals"] = [0, 0, 1] * 3
    saved = {"digest": "a" * 64,
             "geometry_sha256": hashlib.sha256(proof.tobytes()).hexdigest(),
             "physical": {"radius-mm": 25},
             "materials": {"tables": {"pbr": {"stride": 16, "data": [0] * 16}}}}
    *_, config, _digest = photos.decode_upload(
        base64.b64decode(packed["wire"]), SimpleNamespace(get=lambda _: saved))
    assert config["fixtures"] == packed["scene"]
    lights = _derive_lights_from_fixtures(config["fixtures"])
    assert len(lights) == 10
    assert [light["label"] for light in lights] == [f"fixture-mat-{i}" for i in range(10)]
    for index in range(10):
        assert np.allclose(lights[index]["dir"], packed["directions"][index], atol=1e-12, rtol=0)
        assert lights[index]["power"] > 0
        assert lights[index]["filter_rgb"] == [1, 1, 1]
    # The fixtures path, unlike the separate fallback point-light list, owns
    # all emitter identities and has no eight-source truncation.
    fixtures = dict(config["fixtures"])
    rows = np.array(fixtures["pbr"]).reshape(-1, 16)
    rows[:, 8:11] = 0
    fixtures["pbr"] = rows.ravel().tolist()
    assert _derive_lights_from_fixtures(fixtures) == []
