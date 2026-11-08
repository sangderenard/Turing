"""Saved mesh proof and bounded Spectral subprocess route contract."""
import hashlib
import http.client
import json
import struct
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku import marble_objects as objects  # noqa: E402
from meowdoku import marble_photo as photos  # noqa: E402
from meowdoku import server  # noqa: E402


def _saved_object(vertices, normals, material_ids):
    count = len(material_ids)
    packed = np.empty(count, dtype=[("positions", "<f4", (9,)),
                                    ("normals", "<f4", (9,)), ("id", "<u4")])
    packed["positions"] = vertices.reshape(count, 9)
    packed["normals"] = normals.reshape(count, 9)
    packed["id"] = material_ids
    pbr = np.zeros((2, 16), np.float32)
    pbr[:, 0:3] = .9
    pbr[:, 6] = 1.52
    pbr[0, 5] = 1.0
    pbr[1, 7] = 1.0
    tables = {name: {"stride": 16, "data": pbr.reshape(-1).tolist()}
              for name in ("pbr", "phong", "enamel", "texstack")}
    return {"v": 1, "name": "Photo source", "generator": "meowdoku-marble-object-v1",
            "design": {"base-shape": "marble", "kind": "cat_eye", "seed": 5},
            "materials": {"index": {"glass": 0, "eye": 1},
                          "extra": {"coated": 1, "paintA": 1, "paintB": 1},
                          "tables": tables},
            "physical": {"radius-mm": 25},
            "derived": {"massKg": .02},
            "geometry_sha256": hashlib.sha256(packed.tobytes()).hexdigest()}


def _wire(digest, vertices, normals, material_ids, **options):
    config = json.dumps({"object_digest": digest, "exposure_seconds": 30, **options}).encode()
    count = len(material_ids)
    return (b"MRP1" + struct.pack("<III", len(config), count, 0) + config +
            vertices.astype("<f4").tobytes() + normals.astype("<f4").tobytes() +
            material_ids.astype("<u4").tobytes())


@pytest.fixture()
def photo_payload(tmp_path):
    vertices = np.array([[[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                         [[0, 0, -1], [0, 1, 0], [1, 0, 0]]], np.float32)
    normals = np.tile(np.array([[[0, 0, 1]] * 3], np.float32), (2, 1, 1))
    ids = np.array([0, 1], np.uint32)
    store = objects.Store(tmp_path / "saved")
    digest = store.put(_saved_object(vertices, normals, ids))["digest"]
    return store, digest, vertices, normals, ids


def test_mrp1_matches_saved_geometry_and_pbr(photo_payload):
    store, digest, vertices, normals, ids = photo_payload
    wire = _wire(digest, vertices, normals, ids, camera={"resolution": 1024}, transport="fixed:3",
                 exposure_seconds=1800, orbit_frames=8)
    decoded_vertices, decoded_ids, pbr, config, seconds = photos.decode_upload(wire, store)
    np.testing.assert_array_equal(decoded_vertices, vertices)
    np.testing.assert_array_equal(decoded_ids, ids)
    assert pbr.shape == (2, 16) and config["scale_m"] == .025 and seconds == 1800
    assert config["camera"]["resolution"] == 1024 and config["orbit_frames"] == 8
    assert config["transport"] == "fixed:3"
    altered = bytearray(wire)
    altered[-4] = 0  # change the second face's material ID
    with pytest.raises(photos.PhotoError, match="differ from the saved"):
        photos.decode_upload(bytes(altered), store)
    with pytest.raises(photos.PhotoError, match="length"):
        photos.decode_upload(wire[:-1], store)


class _FakeProcess:
    instances = []

    def __init__(self, command, **kwargs):
        self.command = command
        self.returncode = None
        self.instances.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_photo_routes_reuse_cancel_and_guard_image(tmp_path, photo_payload, monkeypatch):
    store, saved_digest, vertices, normals, ids = photo_payload
    monkeypatch.setattr(photos.subprocess, "Popen", _FakeProcess)
    _FakeProcess.instances.clear()
    web = tmp_path / "build" / "meowdoku" / "web"
    web.mkdir(parents=True)
    (web / "boards.json").write_text(json.dumps({"boards": [], "levels": {}}))
    sessions = server.Sessions(web, None)
    sessions.marble_objects = store
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(sessions))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]

    def request(method, path, body=b"", content_type="application/octet-stream"):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request(method, path, body=body, headers={"Content-Type": content_type})
        response = conn.getresponse()
        data = response.read()
        status = response.status
        conn.close()
        return status, data

    try:
        wire = _wire(saved_digest, vertices, normals, ids)
        status, body = request("POST", "/api/marble_photo", wire)
        started = json.loads(body)
        assert status == 202 and started["state"] == "preparing"
        digest = started["digest"]
        assert len(_FakeProcess.instances) == 1
        assert "--seconds" in _FakeProcess.instances[0].command
        assert request("POST", "/api/marble_photo", wire)[0] == 202
        assert len(_FakeProcess.instances) == 1  # same saved scene reuses its job
        status, body = request("GET", f"/api/marble_photo?digest={digest}")
        assert status == 200 and json.loads(body)["state"] == "preparing"
        directory = sessions.marble_photos.directory / digest
        (directory / "preview-000120.png").write_bytes(b"\x89PNG\r\n\x1a\nframe")
        (directory / "status.json").write_text(json.dumps({
            "digest": digest, "state": "exposing", "image": "preview-000120.png"}))
        status, body = request("GET", f"/api/marble_photo/image?digest={digest}&name=preview-000120.png")
        assert status == 200 and body.startswith(b"\x89PNG")
        assert request("GET", f"/api/marble_photo/image?digest={digest}&name=../payload.npz")[0] == 400
        (directory / "exposure-000120.npz").write_bytes(b"linear-rgb-hdr")
        (directory / "orbit.avi").write_bytes(b"RIFFtestAVI ")
        (directory / "status.json").write_text(json.dumps({
            "digest": digest, "state": "complete", "image": "preview-000120.png",
            "hdr": "exposure-000120.npz", "frames": [{"index": 0, "image": "preview-000120.png",
            "hdr": "exposure-000120.npz"}], "animation": "orbit.avi", "animation_expires_at": photos.time.time() + 3600}))
        status, body = request("GET", f"/api/marble_photo/hdr?digest={digest}&name=exposure-000120.npz")
        assert status == 200 and body == b"linear-rgb-hdr"
        status, body = request("GET", f"/api/marble_photo/animation?digest={digest}")
        assert status == 200 and body.startswith(b"RIFF")
        expired = json.loads((directory / "status.json").read_text())
        expired["animation_expires_at"] = photos.time.time() - 1
        (directory / "status.json").write_text(json.dumps(expired))
        assert request("GET", f"/api/marble_photo/animation?digest={digest}")[0] == 404
        assert not (directory / "orbit.avi").exists()
        assert json.loads((directory / "status.json").read_text())["animation_expired"]
        assert request("GET", f"/api/marble_photo/hdr?digest={digest}&name=../payload.npz")[0] == 400
        status, body = request("POST", "/api/marble_photo/cancel", json.dumps({"digest": digest}).encode(),
                               "application/json")
        assert status == 200 and json.loads(body)["state"] == "canceled"
        assert _FakeProcess.instances[0].returncode == -15
        assert request("GET", f"/api/marble_photo?digest={digest}")[0] == 404
    finally:
        sessions.marble_photos.close()
        httpd.shutdown()
        httpd.server_close()
