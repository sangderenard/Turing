"""The host commits only authenticated, byte-exact existing ray payloads."""
import base64
import hashlib
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from meowdoku.studio_agent import StudioAgentError, StudioAgentProducts, product_bytes, commit_studio_ray


def product(raw=b"MRP1existing-payload", content_type="application/octet-stream"):
    return {"content_type": content_type, "encoding": "base64", "sha256": hashlib.sha256(raw).hexdigest(),
            "blob": base64.b64encode(raw).decode()}


class StudioAgentTests(unittest.TestCase):
    def test_missing_node_does_not_stop_server_startup(self):
        with tempfile.TemporaryDirectory(prefix="studio-agent-no-node-") as directory:
            assets = Path(directory) / "assets.json"
            assets.write_text("{}")
            with patch("meowdoku.studio_agent.shutil.which", return_value=None):
                worker = StudioAgentProducts(assets)
                with self.assertRaisesRegex(StudioAgentError, "Node is required"):
                    worker.build({"operation": "ray-payload"})

    def test_explicit_node_path_is_used_for_studio_request(self):
        with tempfile.TemporaryDirectory(prefix="studio-agent-node-path-") as directory:
            assets = Path(directory) / "assets.json"
            assets.write_text("{}")
            worker = StudioAgentProducts(assets, node="C:/nvm4w/nodejs/node.exe")
            result = Mock(returncode=0, stdout=json.dumps(product()).encode())
            with patch("meowdoku.studio_agent.shutil.which", return_value=worker.node) as which:
                with patch("meowdoku.studio_agent.subprocess.run", return_value=result) as run:
                    self.assertEqual(worker.build({"operation": "ray-payload"}), product())
            which.assert_called_once_with(worker.node)
            self.assertEqual(run.call_args.args[0][0], worker.node)

    def test_checked_blob(self):
        value = product()
        self.assertEqual(product_bytes(value), b"MRP1existing-payload")
        with self.assertRaises(StudioAgentError): product_bytes({**value, "sha256": "0"*64})
        with self.assertRaises(StudioAgentError): product_bytes({**value, "blob": "?invalid"})
        with self.assertRaises(StudioAgentError): product_bytes({**value, "owner": "claimed-by-payload"})

    def test_authority_is_not_payload(self):
        jobs = Mock()
        for authorize in (None, lambda: False, lambda: "photo_submit"):
            with self.assertRaises((PermissionError, StudioAgentError)):
                commit_studio_ray(product(), authorize=authorize, photo_jobs=jobs, saved_objects=object(), owners=("account:1",))
        jobs.start.assert_not_called()

    def test_existing_owner_and_exact_job_path(self):
        jobs, objects = Mock(), object()
        receipt={"digest":"a"*64,"state":"preparing"}
        jobs.start.return_value=receipt
        result=commit_studio_ray(product(),authorize=lambda:True,photo_jobs=jobs,saved_objects=objects,owners=("account:1",))
        self.assertIs(result, receipt)
        jobs.start.assert_called_once_with(b"MRP1existing-payload", objects, owners=("account:1",))

    def test_invalid_product_never_starts(self):
        jobs=Mock()
        for value, owners in ((product(content_type="application/json"),("account:1",)),
                              (product(b"not-MRP1"),("account:1",)),(product(),()),(product(),"account:1")):
            with self.assertRaises(StudioAgentError):
                commit_studio_ray(value,authorize=lambda:True,photo_jobs=jobs,saved_objects=None,owners=owners)
        jobs.start.assert_not_called()

    def test_uncertain_job_error_is_not_retried(self):
        jobs=Mock();jobs.start.side_effect=RuntimeError("uncertain outcome")
        with self.assertRaisesRegex(RuntimeError,"uncertain outcome"):
            commit_studio_ray(product(),authorize=lambda:True,photo_jobs=jobs,saved_objects=None,owners=("account:1",))
        self.assertEqual(jobs.start.call_count,1)

    @unittest.skipUnless(shutil.which("node"), "Node is required for the real Studio product")
    def test_actual_node_product_from_portable_authoring(self):
        fixture=json.loads((Path(__file__).parent / "fixtures/studio_scene_builder_authoring.json").read_text())
        request={"schema":"turing.studio-agent-request.v1","operation":"ray-payload",
            "settings":{**fixture["authoring"],"camera":{"eye_m":[0,.08,.07],"target_m":[0,.08,.18],
                "fov_degrees":18,"aperture_radius_m":.002,"focus_distance_m":.11}},
            "photo":{"resolution":128,"seconds":.1,"maxBounces":1}}
        # The regression authoring record contains only object and lighting.
        request["settings"]={key:request["settings"][key] for key in ("object","lighting","camera")}
        with tempfile.TemporaryDirectory(prefix="studio-agent-product-") as directory:
            assets=Path(directory) / "assets.json"
            assets.write_text(json.dumps(fixture["assets"]))
            worker=StudioAgentProducts(assets,timeout=30)
            value=worker.build(request)
            raw=product_bytes(value,content_type="application/octet-stream")
            self.assertEqual(raw[:4],b"MRP1")
            length=struct.unpack_from("<I",raw,4)[0]
            config=json.loads(raw[16:16+length])
            self.assertEqual(config["camera"]["resolution"],128)
            self.assertEqual(config["exposure_seconds"],.1)
            self.assertGreater(len(config["fixtures"]["material_ids"]),0)
            with self.assertRaisesRegex(StudioAgentError,"Choose scene or ray-payload"):
                worker.build({**request,"operation":"render"})


if __name__ == "__main__": unittest.main()
