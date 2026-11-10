"""Real shop meshes feed the incumbent native emissive-triangle source path."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPECTRAL = ROOT.parent / "spectral-analyzer"


def test_shop_tubes_and_leds_emit_native_surface_roots():
    if not (SPECTRAL / "camera_designer/studio_native_emitters.py").is_file():
        pytest.skip("canonical native Studio emitter adapter is unavailable")
    sys.path.insert(0, str(SPECTRAL))
    pytest.importorskip("_spectral_kernels")
    from camera_designer.studio_photo_geometry import prepare_photo_geometry
    from camera_designer.studio_native_emitters import native_emissive_sources

    generated = subprocess.run(
        ["node", "tests/meowdoku_marble_shop_lights_check.mjs", "--json"],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=15,
    )
    cases = json.loads(generated.stdout)
    subject = np.asarray([[[0., 0., 0.], [.001, 0., 0.], [0., .001, 0.]]])
    subject_material = np.asarray([[.1, .1, .1, 1., 0., 0., 1.5, 1.] + [0.] * 8])

    def prepare(fixtures):
        scene, *_ = prepare_photo_geometry(
            {"fixtures": fixtures, "transport": "fixed:3", "triangle_budget": 250000},
            subject, np.asarray([0], np.int32), subject_material,
        )
        return scene

    for kind, fixtures in cases.items():
        scene = prepare(fixtures)
        roots = native_emissive_sources(scene, [.450, .550, .650], ray_budget=512, seed=17)
        assert len(roots) > 0
        assert np.isfinite(roots).all()
        assert np.all(roots[:, 10] > 0)
        triangle_ids = roots[:, 7].astype(np.int64) - 1
        triangles = np.asarray(scene[3], np.float64)[triangle_ids]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        assert np.all(np.abs(((roots[:, :3] - triangles[:, 0]) * normals).sum(axis=1)) < 2e-6)
        assert np.all((roots[:, 4:7] * normals).sum(axis=1) >= -1e-6)
        assert len(np.unique(roots[:, :3], axis=0)) > 100  # Surface sampling, not one centroid source.
        assert len(np.unique(triangle_ids)) == (144 if kind == "tubes" else 768)
        if kind == "leds":
            assert np.all(roots[:, 5] <= 1e-6)
        else:
            assert np.any(roots[:, 5] > .1) and np.any(roots[:, 5] < -.1)

        # Turning photo emission off retains every mesh triangle but launches
        # no source roots. The adapter's own spectral energy checks also ran.
        dark = copy.deepcopy(fixtures)
        rows = np.asarray(dark["pbr"]).reshape(-1, 16)
        rows[:, 8:11] = 0
        dark["pbr"] = rows.ravel().tolist()
        dark_scene = prepare(dark)
        assert len(dark_scene[3]) == len(scene[3])
        assert native_emissive_sources(dark_scene, [.450, .550, .650], ray_budget=512, seed=17).shape == (0, 12)
