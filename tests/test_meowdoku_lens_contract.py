"""Actual source profile -> wire validation -> live camera, without a GPU job."""
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku.marble_lens_contract import (lens_catalog, apply_lens_selection,
    SELECTION_SCHEMA, SPECTRAL_ROOT, FOCUS_SEMANTICS)
from meowdoku.marble_photo import decode_upload, PhotoError
from meowdoku.marble_live import update_live
from tests.test_meowdoku_marble_photo import photo_payload, _wire
from tests.test_meowdoku_marble_live import fake_jobs


def selection(key=None, sha=None, f=.05, n=4):
    return dict(schema=SELECTION_SCHEMA, key=key, source_sha256=sha,
                focal_length_m=f, f_number=n, sensor_height_m=.024)


def test_current_profiles_and_unavailable_compound_inventory():
    catalog = lens_catalog()
    assert len(catalog["profiles"]) == 7 and len(catalog["references"]) == 13
    assert all(not c["selectable"] for c in catalog["references"])
    assert catalog["custom_focus_minimum_m"] == .001
    assert catalog["reference_sensor_height_m"] == .024


def test_source_identity_units_aperture_focus_and_unresolved_calibration():
    c = next(p for p in lens_catalog()["profiles"] if "Macro" in p["key"] or "macro" in p["key"])
    s = selection(c["key"], c["source_sha256"])
    out = apply_lens_selection({"focus_distance_m": .23}, s)
    assert out["aperture_radius_m"] == .00625
    assert out["fov_degrees"] == pytest.approx(math.degrees(2 * math.atan(.024/.1)))
    assert out["focus_semantics"] == FOCUS_SEMANTICS
    assert "not calibrated/applied" in out["lens_calibration"]
    for cam, spec in [({"focus_distance_m": .019}, s), ({"focus_distance_m": .23}, {**s,"source_sha256":"0"*64}),
                      ({"focus_distance_m": .23}, {**s,"focal_length_m":.06}),
                      ({"focus_distance_m": .23}, {**s,"f_number":2})]:
        with pytest.raises(ValueError): apply_lens_selection(cam, spec)
    cleared = apply_lens_selection(out, None)
    assert "lens_selection" not in cleared and "lens_calibration" not in cleared


@pytest.mark.parametrize("field,bad", [("f_number",True),("f_number",float("nan")),("sensor_height_m",.036),
                                      ("focal_length_m",.3),("focal_length_m",10**400),("key","gullstrand_eye")])
def test_malformed_or_unavailable_selection_refused(field,bad):
    with pytest.raises(ValueError): apply_lens_selection({"focus_distance_m":.005}, {**selection(),field:bad})


def test_still_wire_supports_custom_sub_two_cm_without_changing_source(photo_payload):
    store,digest,vertices,normals,ids = photo_payload
    for focus in (.001,.005,.019):
        raw = _wire(digest,vertices,normals,ids,camera={"focus_distance_m":focus,"lens_selection":selection()})
        config = decode_upload(raw,store)[3]
        assert config["camera"]["focus_distance_m"] == focus
        assert config["camera_focus_semantics"] == FOCUS_SEMANTICS
        assert config["camera"]["lens_selection"] == selection()
        assert config["camera"]["aperture_radius_m"] == .00625
    with pytest.raises(PhotoError):
        decode_upload(_wire(digest,vertices,normals,ids,camera={"focus_distance_m":.0009}),store)


def test_live_same_selection_rules_and_explicit_clear(tmp_path):
    jobs,digest,directory = fake_jobs(tmp_path); jobs.spectral_root = SPECTRAL_ROOT
    update_live(jobs,digest,{"revision":1,"camera":{"focus_distance_m":.005,"lens_selection":selection()}})
    c = json.loads((directory/"live-control.json").read_text())["camera"]
    assert c["focus_distance_m"] == .005 and c["aperture_radius_m"] == .00625
    with pytest.raises(PhotoError):
        update_live(jobs,digest,{"revision":2,"camera":{"focus_distance_m":.0009}})
    update_live(jobs,digest,{"revision":2,"camera":{"lens_selection":None,"fov_degrees":18.,"aperture_radius_m":.002}})
    c = json.loads((directory/"live-control.json").read_text())["camera"]
    assert c["fov_degrees"] == 18 and "lens_selection" not in c


def test_still_validator_uses_explicit_source_checkout(photo_payload, tmp_path, monkeypatch):
    from meowdoku import marble_photo as photos
    store,digest,vertices,normals,ids = photo_payload
    seen = []
    original = photos.apply_lens_selection
    def capture(camera, chosen, root):
        seen.append(root)
        return original(camera, chosen, root)
    monkeypatch.setattr(photos, "apply_lens_selection", capture)
    raw = _wire(digest,vertices,normals,ids,camera={"focus_distance_m":.005,"lens_selection":selection()})
    photos.decode_upload(raw,store,tmp_path)
    assert seen == [tmp_path]
    def intercept(payload, saved, root):
        assert root == tmp_path
        raise PhotoError("intercepted before process launch")
    monkeypatch.setattr(photos, "decode_upload", intercept)
    jobs = photos.PhotoJobs(tmp_path/"jobs",tmp_path)
    with pytest.raises(PhotoError,match="intercepted"):
        jobs.start(raw,store)


def test_fresh_web_export_carries_lens_module_dependency():
    import ast
    tree=ast.parse((ROOT/"examples/meowdoku/web_export.py").read_text(encoding="utf-8"))
    tuples=[n for n in ast.walk(tree) if isinstance(n,ast.Tuple)]
    assert any(any(isinstance(v,ast.Constant) and v.value=="marble_lens_controls.js" for v in n.elts) for n in tuples)
    studio=(ROOT/"examples/meowdoku/web/marble_studio.js").read_text(encoding="utf-8")
    assert 'document.addEventListener("studio-lens-change", () => liveRay?.cameraChanged())' in studio


def test_specs_only_mirror_fixed_aperture_and_extended_bounds(photo_payload, tmp_path):
    contract = next(p for p in lens_catalog()["profiles"] if p["key"] == "mirror_500mm_f8")
    assert contract["controls"]["focus_distance"]["minimum"] is None
    assert contract["effects"]["transmission"] is None
    chosen = selection(contract["key"], contract["source_sha256"], .5, 8)
    camera = apply_lens_selection({"focus_distance_m": 5}, chosen)
    assert camera["aperture_radius_m"] == .03125
    assert camera["fov_degrees"] == pytest.approx(2.74966956113881)
    for f, n in ((.499, 8), (.501, 8), (.5, 7.9), (.5, 8.1)):
        with pytest.raises(ValueError):
            apply_lens_selection({"focus_distance_m": 5}, selection(contract["key"], contract["source_sha256"], f, n))
    store, digest, vertices, normals, ids = photo_payload
    assert decode_upload(_wire(digest, vertices, normals, ids, camera=camera), store)[3]["camera"]["lens_selection"] == chosen
    jobs, digest, directory = fake_jobs(tmp_path)
    jobs.spectral_root = SPECTRAL_ROOT
    update_live(jobs, digest, {"revision": 1, "camera": camera})
    assert json.loads((directory / "live-control.json").read_text())["camera"]["aperture_radius_m"] == .03125
