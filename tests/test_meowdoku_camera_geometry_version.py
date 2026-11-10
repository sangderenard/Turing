"""Camera geometry convention participates in both still/live photo identity."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku.marble_photo import _config


def test_camera_geometry_convention_is_server_owned_and_breaks_old_digest():
    # The worker also records this version for direct legacy configuration files.
    for transport in ("fixed:1", "fixed:3"):
        saved = {"digest": "a" * 64, "geometry_sha256": "b" * 64,
                 "physical": {"radius-mm": 25}}
        config, _ = _config({"camera_geometry_version": "old-or-client-invented",
                             "object_digest": saved["digest"], "transport": transport}, saved)
        assert config["camera_geometry_version"] == "native-housing-behind-thin-lens-v1"
        previous = {k: v for k, v in config.items() if k != "camera_geometry_version"}
        digest = lambda c: hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
        assert digest(previous) != digest(config)
