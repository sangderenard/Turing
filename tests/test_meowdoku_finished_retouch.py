"""Completed HDR retouch must not depend on worker process exit timing."""
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku.marble_photo import PhotoJobs


def test_completed_worker_still_exiting_retouches_saved_hdr(tmp_path):
    digest = "a" * 64
    folder = tmp_path / digest
    folder.mkdir()
    np.savez(folder / "exposure-000001.npz", linear_rgb_exposure=np.ones((2, 2, 3), np.float32))
    status = {"digest": digest, "state": "complete", "hdr": "exposure-000001.npz", "frames": []}
    (folder / "status.json").write_text(json.dumps(status))
    store = PhotoJobs.__new__(PhotoJobs)
    store.directory = tmp_path
    store.lock = threading.Lock()
    store.active_digest = digest
    store.process = SimpleNamespace(poll=lambda: None)
    result = store.retouch(digest, {"red": 0})
    assert result["ok"] is True
    assert "Legacy HDR" in result["warning"]
    first = json.loads((folder / "status.json").read_text())["retouched_image"]
    first_pixels = np.asarray(Image.open(folder / first))
    assert np.all(first_pixels[..., 0] == 0)
    assert np.all(first_pixels[..., 1:] > 0)
    store.retouch(digest, {"red": 1})
    second = json.loads((folder / "status.json").read_text())["retouched_image"]
    assert first != second
    assert np.all(np.asarray(Image.open(folder / second))[..., 0] > 0)
    assert not (folder / first).exists()
