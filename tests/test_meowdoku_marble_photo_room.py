"""CPU-only authored Studio room -> MRP1 -> validated worker config proof."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from meowdoku import marble_photo as photos


def test_authored_room_survives_real_pack_and_decode_once():
    checked = subprocess.run(
        ["node", "tests/meowdoku_marble_photo_room_check.mjs", "--json"],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=15,
    )
    packed = json.loads(checked.stdout)
    wire = base64.b64decode(packed["wire"])
    proof = np.zeros(1, dtype=[("positions", "<f4", (9,)),
                              ("normals", "<f4", (9,)), ("id", "<u4")])
    proof["positions"] = [0, 0, 0, 1, 0, 0, 0, 1, 0]
    proof["normals"] = [0, 0, 1] * 3
    saved = {
        "digest": "a" * 64,
        "geometry_sha256": hashlib.sha256(proof.tobytes()).hexdigest(),
        "physical": {"radius-mm": 25},
        "materials": {"tables": {"pbr": {"stride": 16, "data": [0] * 16}}},
    }
    _, _, _, config, _ = photos.decode_upload(wire, SimpleNamespace(get=lambda digest: saved))
    assert config["fixtures"] == packed["scene"]
    assert "room" not in config
    assert len(config["fixtures"]["material_ids"]) == 777  # one fixture + six 8x8 room surfaces + two swept sheets
    assert set(config["fixtures"]["material_ids"]) == {0, packed["roomBase"], packed["roomBase"] + 1}


def test_photography_table_contacts_saved_radius_in_real_worker_decode():
    checked = subprocess.run(
        ["node", "tests/meowdoku_marble_studio_table_check.mjs", "--json"],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=15,
    )
    cases = json.loads(checked.stdout)
    assert [case["radiusMm"] for case in cases] == [10, 25]
    hashes = set()
    for case in cases:
        saved = {
            "digest": "a" * 64,
            "geometry_sha256": case["geometryHash"],
            "physical": {"radius-mm": case["radiusMm"]},
            "materials": {"tables": {"pbr": {"stride": 16, "data": [0] * 16}}},
        }
        positions, _, _, config, _ = photos.decode_upload(
            base64.b64decode(case["wire"]), SimpleNamespace(get=lambda digest: saved))
        assert float(positions[:, :, 1].min()) == -1.0
        scale = case["radiusMm"] / 1000
        assert config["scale_m"] == scale  # worker's saved physical radius owns scale
        world = positions * scale + np.asarray(config["object_center_m"])
        furniture = np.asarray(config["fixtures"]["vertices"]).reshape(-1, 3)
        assert abs(float(furniture[:, 1].max()) - float(world[:, :, 1].min())) < 1e-9
        assert abs(config["camera"]["focus_distance_m"] - 3.45 * scale) < 1e-12
        assert config["camera"]["fov_degrees"] == 360 / 7
        assert len(config["fixtures"]["material_ids"]) == 60
        hashes.add(case["geometryHash"])
    assert len(hashes) == 1  # identical hashed local sphere at both physical scales
