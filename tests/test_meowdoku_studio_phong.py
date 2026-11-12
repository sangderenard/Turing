"""Captured-scene operation bounds; real GL smoke is explicitly opted in."""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from meowdoku.studio_agent import StudioAgentProducts, StudioAgentError, product_bytes, commit_studio_ray
from meowdoku.studio_phong import StudioPhongHost, validate_camera

I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
CAMERA = {"view": I, "projection": I, "width": 320, "height": 320}


class StudioPhongTests(unittest.TestCase):
    def test_camera_bounds(self):
        self.assertEqual(validate_camera(CAMERA), CAMERA)
        for patch in ({"width": True}, {"width": 2049}, {"height": 15}, {"view": [0]*16},
                      {"view": [math.nan]*16}, {"projection": [math.inf]*16}, {"url": "https://example.com"}):
            with self.assertRaises(StudioAgentError):
                validate_camera({**CAMERA, **patch})

    def test_explicit_product_seam_and_no_ray_commit(self):
        raw = b"bounded PNG product"
        product = {"content_type": "image/png", "encoding": "base64",
                   "sha256": hashlib.sha256(raw).hexdigest(), "blob": base64.b64encode(raw).decode()}
        host = Mock()
        host.render.return_value = product
        with tempfile.TemporaryDirectory() as folder:
            assets = Path(folder) / "assets.json"
            assets.write_text("{}")
            worker = StudioAgentProducts(assets, node="node", phong_host=host)
            request = {"schema": "turing.studio-agent-request.v1", "operation": "render-captured", "camera": CAMERA}
            self.assertEqual(worker.build(request), product)
            host.render.assert_called_once_with(CAMERA)
            for patch in ({"settings": {}}, {"sceneAsset": {}}, {"shader": "untrusted"}, {"schema": "wrong"}):
                with self.assertRaises(StudioAgentError):
                    worker.build({**request, **patch})
            worker.phong_host = None
            with self.assertRaisesRegex(StudioAgentError, "no trusted captured"):
                worker.build(request)
        jobs = Mock()
        with self.assertRaises(StudioAgentError):
            commit_studio_ray(product, authorize=lambda: True, photo_jobs=jobs, saved_objects=None, owners=("owner",))
        jobs.start.assert_not_called()

    def test_loopback_capability_and_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            scene = Path(folder) / "scene.json"
            scene.write_text(json.dumps({"schema": "turing.studio-scene-asset.v1", "identity": "test",
                                        "render": {"shaders": {"vertex": "trusted", "fragment": "trusted"}}}))
            host = StudioPhongHost(scene, chrome=Path(sys.executable))
            urls = []
            def browser(command, **kwargs):
                url = command[-1]
                urls.append(url)
                base = url.rsplit("/", 1)[0]
                origin = "/".join(url.split("/")[:3])
                with urlopen(base + "/standalone_studio.js", timeout=2) as response:
                    self.assertEqual(response.read(), host.sources["standalone_studio.js"])
                for req in (origin + "/index.html", base + "/../studio_agent.py",
                            Request(url, headers={"Host": "attacker.example"}),
                            Request(base + "/receipt", data=b"{}", headers={"Origin": "https://attacker.example"})):
                    with self.assertRaises(HTTPError) as error:
                        urlopen(req, timeout=2)
                    self.assertEqual(error.exception.code, 404)
                with urlopen(Request(base + "/receipt", data=b'{"ok":false,"error":"renderer refused"}',
                                     headers={"Origin": origin}), timeout=2) as response:
                    self.assertEqual(response.status, 204)
                process = Mock()
                process.pid = 123456789
                return process
            with patch("meowdoku.studio_phong.subprocess.Popen", side_effect=browser), \
                    patch("meowdoku.studio_phong.subprocess.run"), patch("meowdoku.studio_phong.os.killpg", create=True):
                with self.assertRaisesRegex(StudioAgentError, "renderer refused"):
                    host.render(CAMERA)
            self.assertFalse(host.lock.locked())
            with self.assertRaises(OSError):
                urlopen(urls[0], timeout=1)

    @unittest.skipUnless(os.environ.get("STUDIO_PHONG_GL_CHECK"), "opt-in actual Chrome software-WebGL render")
    def test_actual_shared_full_scene(self):
        scene = ROOT / "build/dot-studio-game-transfer-20261110/complete-studio.scene-asset.json"
        asset = json.loads(scene.read_bytes())
        host = StudioPhongHost(scene, chrome=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
        f = 1/math.tan((asset["render"]["referenceCamera"].get("fovDegrees") or 50)*math.pi/360)
        camera = {"view": asset["render"]["referenceCamera"]["view"], "width": 320, "height": 320,
                  "projection": [f,0,0,0,0,f,0,0,0,0,-30.01/29.99,-1,0,0,-.6/29.99,0]}
        first = product_bytes(host.render(camera), content_type="image/png")
        changed = {**camera, "view": list(camera["view"])}
        changed["view"][12] += .4
        second = product_bytes(host.render(changed), content_type="image/png")
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith(b"\x89PNG\r\n\x1a\n"))
        receipt = ROOT / "build/dot-standalone-studio-20261111/receipt.json"
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            self.assertEqual(hashlib.sha256(first).hexdigest(), previous["images"][0]["sha256"])
        print(json.dumps({**host.describe(), "png_sha256": hashlib.sha256(first).hexdigest(),
                          "changed_camera_sha256": hashlib.sha256(second).hexdigest(), "bytes": len(first)}), flush=True)


if __name__ == "__main__":
    unittest.main()
