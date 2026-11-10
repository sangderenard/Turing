"""Live forward-ray budgets are validated and handed off without a restart."""
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "examples")]
from meowdoku.marble_live import live_settings, update_live
from meowdoku.marble_photo import PhotoError


def test_default_ray_batch_matches_existing_live_budget():
    assert live_settings({}, {"resolution": 128})["ray_batch_size"] == 8192


@pytest.mark.parametrize("size", [128, 129, 8192, 65536])
def test_ray_batch_accepts_integer_endpoints_and_partial_dispatch(size):
    assert live_settings({"ray_batch_size": size}, {"resolution": 128})["ray_batch_size"] == size


@pytest.mark.parametrize("size", [None, True, False, 0, -1, 127, 65537,
    10**100, 8192.0, 8192.5, "8192", [], {}, float("nan"), float("inf")])
def test_ray_batch_rejects_noninteger_or_unbounded_work(size):
    with pytest.raises(PhotoError, match="ray_batch_size"):
        live_settings({"ray_batch_size": size}, {"resolution": 128})


def test_live_ray_batch_revision_preserves_camera_checkpoint_and_display(tmp_path):
    digest = "a" * 64
    directory = tmp_path / digest
    directory.mkdir()
    camera = dict(resolution=128, eye_m=[0, 0, .07], target_m=[0, 0, .18],
                  fov_degrees=18, focus_distance_m=.11, aperture_radius_m=.002)
    live = live_settings({"checkpoint": True, "display": {
        "tint": 1, "normalization": "specular", "normalization_percentile": 94}}, camera)
    (directory / "config.json").write_text(json.dumps({"live": live}))
    path = directory / "live-control.json"
    path.write_text(json.dumps(dict(camera=camera, live=live, revision=0, expires_at=0)))
    jobs = SimpleNamespace(directory=tmp_path, lock=threading.Lock(), active_digest=digest,
        process=SimpleNamespace(poll=lambda: None), _status_locked=lambda _: {})
    update_live(jobs, digest, {"revision": 1, "live": {"ray_batch_size": 16384}})
    current = json.loads(path.read_text())
    assert current["camera"] == camera
    assert current["live"] == {**live, "ray_batch_size": 16384}
    assert current["revision"] == 1
    before = path.read_bytes()
    with pytest.raises(PhotoError, match="ray_batch_size"):
        update_live(jobs, digest, {"revision": 2, "live": {"ray_batch_size": 65537}})
    assert path.read_bytes() == before
    update_live(jobs, digest, {"revision": 1, "live": {"ray_batch_size": 128}})
    assert json.loads(path.read_text())["live"]["ray_batch_size"] == 16384
