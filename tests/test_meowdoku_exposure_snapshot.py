"""Exposure snapshots reuse immutable object validation without a library write."""
import json
import pytest
from tests.test_meowdoku_marble_photo import photo_payload, _wire, _saved_object
from meowdoku import marble_photo as photos, marble_objects as objects

class NoLibrary:
    def get(self, digest):
        raise AssertionError("snapshot must not read or write the object library")

def test_inline_snapshot_same_geometry_and_materials(photo_payload):
    store, digest, vertices, normals, ids = photo_payload
    snapshot = _saved_object(vertices, normals, ids)
    before = list(store.directory.iterdir())
    result = photos.decode_upload(_wire(None, vertices, normals, ids, object_snapshot=snapshot), NoLibrary())
    assert result[3]["object_digest"] == objects.source_digest(snapshot)
    assert result[3]["exposure_snapshot"] == snapshot
    assert list(store.directory.iterdir()) == before
    snapshot["physical"]["radius-mm"] = 2
    result = photos.decode_upload(_wire(None, vertices, normals, ids, object_snapshot=snapshot), NoLibrary())
    assert result[3]["scale_m"] == .002

def test_inline_snapshot_rejects_mismatch_ambiguity_and_live(photo_payload):
    _, digest, vertices, normals, ids = photo_payload
    snapshot = _saved_object(vertices, normals, ids)
    with pytest.raises(photos.PhotoError, match="not both"):
        photos.decode_upload(_wire(digest, vertices, normals, ids, object_snapshot=snapshot), NoLibrary())
    with pytest.raises(photos.PhotoError, match="still photographs"):
        photos.decode_upload(_wire(None, vertices, normals, ids, object_snapshot=snapshot, live={}), NoLibrary())
    changed = vertices.copy(); changed[0,0,0] += .1
    with pytest.raises(photos.PhotoError, match="differ from the saved"):
        photos.decode_upload(_wire(None, changed, normals, ids, object_snapshot=snapshot), NoLibrary())

@pytest.mark.parametrize("bad", [{}, [], {"v":1}, {"v":float("nan")}])
def test_invalid_snapshots_reject(photo_payload, bad):
    _, _, vertices, normals, ids = photo_payload
    with pytest.raises(photos.PhotoError, match="invalid exposure snapshot"):
        photos.decode_upload(_wire(None, vertices, normals, ids, object_snapshot=bad), NoLibrary())


def test_job_owns_snapshot_without_library_entry(photo_payload, tmp_path, monkeypatch):
    from tests.test_meowdoku_marble_photo import _FakeProcess
    monkeypatch.setattr(photos.subprocess, "Popen", _FakeProcess)
    monkeypatch.setattr(photos.PhotoJobs, "_validate_geometry", lambda *args: {})
    _, _, vertices, normals, ids = photo_payload
    snapshot = _saved_object(vertices, normals, ids)
    jobs = photos.PhotoJobs(tmp_path / "photos", tmp_path / "spectral")
    result = jobs.start(_wire(None, vertices, normals, ids, object_snapshot=snapshot), NoLibrary())
    config = json.loads((jobs.directory / result["digest"] / "config.json").read_text())
    assert config["exposure_snapshot"] == snapshot
    assert config["object_digest"] == objects.source_digest(snapshot)
    assert not (tmp_path / "photos" / "objects").exists()
    jobs.cancel(result["digest"])
    jobs._sweep_stop.set()


def test_new_snapshot_cannot_interrupt_active_photo(photo_payload, tmp_path, monkeypatch):
    from tests.test_meowdoku_marble_photo import _FakeProcess
    monkeypatch.setattr(photos.subprocess, "Popen", _FakeProcess)
    monkeypatch.setattr(photos.PhotoJobs, "_validate_geometry", lambda *args: {})
    _, _, vertices, normals, ids = photo_payload
    snapshot = _saved_object(vertices, normals, ids)
    jobs = photos.PhotoJobs(tmp_path / "photos", tmp_path / "spectral")
    try:
        wire = _wire(None, vertices, normals, ids, object_snapshot=snapshot)
        first = jobs.start(wire, NoLibrary())
        process = jobs.process
        # An identical retry shares the existing job; a different photo cannot replace it.
        assert jobs.start(wire, NoLibrary())["digest"] == first["digest"]
        snapshot["physical"]["radius-mm"] = 2
        with pytest.raises(photos.PhotoError, match="explicitly cancel"):
            jobs.start(_wire(None, vertices, normals, ids, object_snapshot=snapshot), NoLibrary())
        assert jobs.active_digest == first["digest"]
        assert jobs.process is process and process.poll() is None
        jobs.cancel(first["digest"])
    finally:
        jobs._sweep_stop.set()


@pytest.mark.parametrize("mode", ["glass_rod", "injection_dye"])
def test_inline_authored_optics_and_exact_bubble_contract(mode):
    import base64, subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    script = (root / "tests/meowdoku_studio_optical_check.mjs").read_text().split("const originalJSON")[0]
    script = script.replace("'../examples/", "'./examples/")
    script += """
import {packMarblePhoto} from './examples/meowdoku/web/marble_photo.js';
original.design['bubble-count']=1;
original.design.optical_authoring={version:1,mode:MODE,provenance:OPTICAL_PROVENANCE,materials:records};
const prepared=await prepareObjectSnapshot(original);
const wire=packMarblePhoto(prepared.geometry,null,{objectSnapshot:prepared.value,sphereCoverage:'exact'});
console.log(Buffer.from(wire).toString('base64'));
""".replace("MODE", json.dumps(mode))
    encoded = subprocess.run(["node", "--input-type=module", "-e", script], cwd=root,
                             check=True, capture_output=True, text=True, timeout=30).stdout
    config = photos.decode_upload(base64.b64decode(encoded), NoLibrary())[3]
    assert config["exposure_snapshot"]["design"]["optical_authoring"]["mode"] == mode
    assert config["region_media"]["regions"]
    assert config["optical_media"]["materials"]
    assert config["analytic_spheres"]["spheres"]
