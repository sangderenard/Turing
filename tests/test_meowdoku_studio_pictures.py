"""studio_pictures: what a picture request may contain, who may read its receipt, and (opt-in) the real Nodus round trip.

The first tests never reach Nodus: the connection seam refuses, so requests stay queued and only the admission rules
run. The last test starts the real canvas host on a temporary canvas when NODUS_HEADLESS_TEST_DLL names the built
library, and carries a request through the real scene builder (Node) and the real FIFO. PhotoJobs is a recorder
there: no GPU is used.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku import studio_agent, studio_pictures  # noqa: E402

NODUS = ROOT.parent / "nodus"


class Products:
    def __init__(self, defaults):
        self.defaults = defaults

    def build(self, request):
        raise AssertionError("no request reaches the builder in these tests")


class Recorder:
    def __init__(self):
        self.calls = []

    def start(self, raw, saved_objects, *, owners=()):
        self.calls.append((raw[:4], saved_objects, owners))
        return {"digest": "d" * 64, "state": "recorded"}


def refuse(role, identity):
    raise OSError("no Nodus host in this test")


@pytest.fixture
def service(tmp_path):
    made = studio_pictures.StudioPictures(port=1, prefix="studio.product", products=Products(tmp_path), store_dir=tmp_path,
                                          photo_jobs=Recorder(), saved_objects_for=lambda account: account,
                                          claim=lambda account, digest: None, connect=refuse, log=lambda text: None)
    yield made
    made.close()


def test_a_request_is_admitted_and_replays_by_its_request_id(service):
    first = service.submit(7, ["account:7"], lambda: True, {"request_id": "a", "photo": {"resolution": 256}})
    assert first["state"] == "queued" and first["schema"] == "speaktome.picture.v1" and "job" not in first
    again = service.submit(7, ["account:7"], lambda: True, {"request_id": "a", "photo": {"resolution": 256}})
    assert again["replayed"] is True and again["picture_id"] == first["picture_id"]
    with pytest.raises(studio_pictures.PictureError) as changed:
        service.submit(7, ["account:7"], lambda: True, {"request_id": "a", "photo": {"resolution": 512}})
    assert changed.value.status == 409
    other = service.submit(8, ["account:8"], lambda: True, {"request_id": "a", "photo": {"resolution": 256}})
    assert other["picture_id"] != first["picture_id"], "a request id belongs to its account"


@pytest.mark.parametrize("body", [
    {"photo": {}}, {"request_id": "bad id"}, {"request_id": "a", "photo": []}, {"request_id": "a", "settings": "x"},
    {"request_id": "a", "operation": "scene"}, {"request_id": "a", "owners": ["account:1"]},
    {"request_id": "a", "photo": {"seconds": float("nan")}}, "text"])
def test_a_malformed_request_is_refused(service, body):
    with pytest.raises(studio_pictures.PictureError) as refused:
        service.submit(7, ["account:7"], lambda: True, body)
    assert refused.value.status == 400


def test_identity_comes_from_the_host_never_the_request(service):
    with pytest.raises(studio_pictures.PictureError) as refused:
        service.submit(7, [], lambda: True, {"request_id": "a"})
    assert refused.value.status == 401
    with pytest.raises(studio_pictures.PictureError):
        service.submit(7, ["account:7"], None, {"request_id": "a"})


def test_settings_are_required_without_baked_defaults(tmp_path):
    bare = studio_pictures.StudioPictures(port=1, prefix="studio.product", products=Products(None), store_dir=tmp_path,
                                          photo_jobs=Recorder(), saved_objects_for=lambda account: account,
                                          claim=lambda account, digest: None, connect=refuse, log=lambda text: None)
    try:
        with pytest.raises(studio_pictures.PictureError) as refused:
            bare.submit(7, ["account:7"], lambda: True, {"request_id": "a"})
        assert "settings are required" in str(refused.value)
        assert bare.submit(7, ["account:7"], lambda: True, {"request_id": "a", "settings": {}})["state"] == "queued"
    finally:
        bare.close()


def test_a_receipt_is_read_only_by_its_account_and_the_queue_is_bounded(service):
    receipts = [service.submit(7, ["account:7"], lambda: True, {"request_id": f"r{n}"})
                for n in range(studio_pictures.MAX_QUEUED)]
    assert service.status(7, receipts[0]["picture_id"])["state"] == "queued"
    with pytest.raises(studio_pictures.PictureError) as hidden:
        service.status(8, receipts[0]["picture_id"])
    assert hidden.value.status == 404
    with pytest.raises(studio_pictures.PictureError) as full:
        service.submit(7, ["account:7"], lambda: True, {"request_id": "one-too-many"})
    assert full.value.status == 429
    assert service.photo_jobs.calls == [], "nothing queued ever reached the photo service"


def test_the_real_nodus_round_trip_when_the_library_is_named(tmp_path):
    library = os.environ.get("NODUS_HEADLESS_TEST_DLL")
    baked = ROOT / "build" / "dot-studio-agent-20261111"
    if not library or not (baked / "default-assets.json").is_file():
        pytest.skip("Set NODUS_HEADLESS_TEST_DLL (and PATH for its dependencies) and keep the baked Studio defaults to test")
    canvas, port = tmp_path / "studio.canvas", 47693
    blank = tmp_path / "blank.canvas"
    blank.write_text("CANVAS V4\n1 1 56\nOFFSET 0 0\n\n", encoding="utf-8")
    subprocess.run([sys.executable, str(NODUS / "tools" / "add_contract_port.py"), "--canvas", str(blank), "--out", str(canvas),
                    "--prefix", "studio.product", "--port", str(port), "--loan", "7200"], check=True, capture_output=True, timeout=60)
    host = subprocess.Popen([sys.executable, str(NODUS / "tools" / "run_canvas.py"), "--canvas", str(canvas), "--rate", "60",
                             "--acquire", "studio.product:4:7200", "--ticks", "18000"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in host.stdout:
            if "contract_ports" in line:
                break
        request = json.loads((baked / "default-1k-2x-request.json").read_text(encoding="utf-8"))
        defaults = tmp_path / "defaults.json"
        defaults.write_text(json.dumps(request["settings"]), encoding="utf-8")
        photo = {k: v for k, v in request["photo"].items() if k not in ("oversampling", "tiled")}
        photo.update(resolution=256, seconds=5)
        jobs, claims, granted = Recorder(), [], [True]
        service = studio_pictures.StudioPictures(
            port=port, prefix="studio.product", store_dir=tmp_path / "store", photo_jobs=jobs,
            products=studio_agent.StudioAgentProducts(baked / "default-assets.json", defaults=defaults),
            saved_objects_for=lambda account: {"account": account}, claim=lambda account, digest: claims.append((account, digest)),
            log=lambda text: None)
        try:
            def finish(receipt):
                end = time.time() + 200
                while receipt["state"] not in studio_pictures.TERMINAL and time.time() < end:
                    time.sleep(0.2)
                    receipt = service.status(7, receipt["picture_id"])
                return receipt
            done = finish(service.submit(7, ["account:7"], lambda: granted[0], {"request_id": "real-1", "photo": photo}))
            assert done["state"] == "committed" and done["job"]["digest"] == "d" * 64
            assert done["status_url"].endswith("d" * 64)
            assert jobs.calls == [(b"MRP1", {"account": 7}, ("account:7",))] and claims == [(7, "d" * 64)]
            granted[0] = False
            refused = finish(service.submit(7, ["account:7"], lambda: granted[0], {"request_id": "real-2", "photo": photo}))
            assert refused["state"] == "refused" and len(jobs.calls) == 1
            assert not list((tmp_path / "store" / "requests").glob("*.json")), "payloads are released after reconciliation"
            assert not list((tmp_path / "store" / "results").glob("*.json"))
        finally:
            service.close()
    finally:
        host.kill()
        host.wait(timeout=10)
