"""Physical scale reaches actual MRP1 admission and CPU scene preparation."""
import base64
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples.meowdoku.marble_photo import decode_upload, PhotoError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cases():
    result = subprocess.run(["node", "tests/meowdoku_marble_scale_check.mjs", "--json"],
                            cwd=ROOT, capture_output=True, text=True, check=True, timeout=20)
    return json.loads(result.stdout)["cases"]


def test_actual_wire_radius_contact_material_and_analytic_media(cases):
    spectral = ROOT.parent / "spectral-analyzer"
    if not (spectral / "camera_designer" / "studio_photo_geometry.py").is_file():
        pytest.skip("canonical Spectral CPU scene preparation is unavailable")
    sys.path.insert(0, str(spectral))
    from camera_designer.studio_photo_geometry import prepare_photo_geometry

    initial_mesh = initial_pbr = initial_camera = initial_fixtures = initial_owners = None
    for case in cases:
        saved = case["saved"]
        before = copy.deepcopy(saved)
        vertices, ids, pbr, config, _ = decode_upload(
            base64.b64decode(case["wire"]), SimpleNamespace(get=lambda _: saved))
        assert saved == before
        radius = saved["physical"]["radius-mm"] / 1000
        assert config["scale_m"] == radius  # ignores client's nominal .025
        center = np.asarray(config["object_center_m"])
        assert center[1] - radius == pytest.approx(-.01, abs=1e-15)
        assert config["analytic_spheres"] == saved["design"]["analytic_spheres"]
        assert config["boundary_media"] == saved["design"]["boundary_media"]
        expected_pbr = np.asarray(saved["materials"]["tables"]["pbr"]["data"], np.float32).reshape(-1,16)
        assert pbr.tobytes() == expected_pbr.tobytes()
        if initial_mesh is None:
            initial_mesh, initial_pbr = vertices.tobytes(), pbr.tobytes()
            initial_camera, initial_fixtures = config["camera"], config["fixtures"]
        assert vertices.tobytes() == initial_mesh
        assert pbr.tobytes() == initial_pbr
        assert config["camera"] == initial_camera
        assert config["fixtures"] == initial_fixtures

        # The existing live and still worker share this exact CPU preparation.
        # No GL context, ray job, or service request is created by this call.
        scene, spheres, stats, _ = prepare_photo_geometry(config, vertices, ids, pbr)
        assert stats["analytic_spheres"] == 3
        for record, authored in zip(spheres, config["analytic_spheres"]["spheres"]):
            expected_center = center + radius * np.asarray(authored["center"])
            np.testing.assert_allclose(record[:3], expected_center, rtol=0, atol=2e-8)
            assert record[3] == pytest.approx(radius * authored["radius"], abs=1e-8)
        assert spheres[0,1] - spheres[0,3] == pytest.approx(-.01, abs=3e-8)
        # Physical path diameters grow with saved radius; absorption/material
        # records are untouched, rather than rescaled to preserve appearance.
        assert 2 * spheres[0,3] == pytest.approx(2 * radius, abs=2e-8)
        assert config["boundary_transport"] == "studio-explicit-media-v1"
        owners = spheres[:,4].astype(int)
        assert np.all(scene[0][owners,4:7] == 0)
        assert np.all(scene[0][owners,8:11] == 0)
        owner_materials = scene[1][owners].tobytes()
        if initial_owners is None:
            initial_owners = owner_materials
        assert owner_materials == initial_owners


def test_scale_does_not_silently_admit_incompatible_exact_saved_geometry(cases):
    saved = copy.deepcopy(cases[0]["saved"])
    saved["design"].pop("analytic_spheres")
    with pytest.raises(PhotoError, match="Save this design again"):
        decode_upload(base64.b64decode(cases[0]["wire"]), SimpleNamespace(get=lambda _: saved))
    saved = copy.deepcopy(cases[0]["saved"])
    saved["design"]["analytic_spheres"]["spheres"][1]["radius"] *= .9
    with pytest.raises(PhotoError, match="metadata does not match"):
        decode_upload(base64.b64decode(cases[0]["wire"]), SimpleNamespace(get=lambda _: saved))
