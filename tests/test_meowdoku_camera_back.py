"""Physical back geometry survives the real still/live camera validation path."""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "examples")]
from meowdoku.marble_camera_back import BACK_SCHEMA, BACK_PRESETS, camera_back, sensor_gate, apply_camera_back
from meowdoku.marble_lens_contract import apply_lens_selection, SELECTION_SCHEMA, SPECTRAL_ROOT
from meowdoku.marble_photo import decode_upload, PhotoError
from meowdoku.marble_live import update_live
from tests.test_meowdoku_marble_photo import photo_payload, _wire
from tests.test_meowdoku_marble_live import fake_jobs


def back(key="full_frame"):
    width, height = BACK_PRESETS[key]
    return dict(schema=BACK_SCHEMA, key=key, width_m=width, height_m=height, fit="center_crop")


def lens(f=.05):
    return dict(schema=SELECTION_SCHEMA, key=None, source_sha256=None,
                focal_length_m=f, f_number=8., sensor_height_m=.024)


def camera(f=.05):
    return apply_lens_selection(dict(fov_degrees=18., focus_distance_m=.11, aperture_radius_m=.002), lens(f))


@pytest.mark.parametrize("key", BACK_PRESETS)
@pytest.mark.parametrize("aspect", [.5, 1., 1.5, 2., 10.])
def test_crop_fits_inside_back_without_stretch(key, aspect):
    chosen = back(key)
    gate = sensor_gate(chosen, aspect)
    assert gate["width_m"] <= chosen["width_m"] + 1e-15
    assert gate["height_m"] <= chosen["height_m"] + 1e-15
    assert gate["width_m"] / gate["height_m"] == pytest.approx(aspect)
    assert min(abs(gate["width_m"] - chosen["width_m"]), abs(gate["height_m"] - chosen["height_m"])) < 1e-15
    c = camera(.1)
    result = apply_camera_back(c, chosen, aspect)
    # Existing shader ray slopes are exactly the physical cropped gate / focal.
    slope = math.tan(math.radians(result["fov_degrees"]) / 2)
    assert slope == pytest.approx(gate["height_m"] / .2)
    assert slope * aspect == pytest.approx(gate["width_m"] / .2)
    assert result["aperture_radius_m"] == c["aperture_radius_m"]
    assert result["focus_distance_m"] == c["focus_distance_m"]


def test_default_identity_and_back_fov_are_resolution_independent():
    c = camera()
    assert apply_camera_back(c, None) == c
    first = apply_camera_back(c, back("aps_c"))
    assert first["fov_degrees"] < c["fov_degrees"]
    for edge in (128, 256, 512, 4096, 32768):
        result = apply_camera_back({**c, "resolution": edge, "tile_edge": 256}, back("aps_c"))
        assert result["sensor_gate"] == first["sensor_gate"]
        assert result["fov_degrees"] == first["fov_degrees"]
        assert result["resolution"] == edge
    assert apply_camera_back(first, first["camera_back"]) == first
    cleared = apply_camera_back(first, None)
    assert cleared["fov_degrees"] == c["fov_degrees"]
    cleared = apply_camera_back(apply_lens_selection(first, lens()), None)
    assert cleared["fov_degrees"] == c["fov_degrees"]


def test_original_lens_anchor_prevents_repeated_crop_and_clear_restores_framing():
    original = dict(fov_degrees=18., focus_distance_m=.11, aperture_radius_m=.002)
    c = apply_camera_back(original, back("aps_c"))
    assert apply_camera_back(c, back("aps_c")) == c
    assert apply_camera_back(c, None) == original


@pytest.mark.parametrize("change", [{"width_m":True}, {"width_m":float("nan")}, {"width_m":10**400},
    {"height_m":0}, {"width_m":.3}, {"key":"unknown"}, {"fit":"stretch"}, {"extra":1}, {"schema":"v0"}])
def test_malformed_or_spoofed_back_rejected(change):
    with pytest.raises(ValueError):
        camera_back({**back(), **change})


def test_custom_and_wide_field_bounds():
    custom = {**back(), "key":"custom", "width_m":.3, "height_m":.001}
    assert sensor_gate(custom)["height_m"] == .001
    with pytest.raises(ValueError, match="100 degree"):
        apply_camera_back(camera(.012), back("large_4x5"))


def test_still_wire_preserves_back_and_original_pixel_choices(photo_payload):
    store, digest, vertices, normals, ids = photo_payload
    for edge in (128, 256, 512, 768, 1024, 2048, 4096):
        c = {**camera(), "resolution":edge, "camera_back":back("aps_c"), "sensor_gate":{"height_m":9}}
        result = decode_upload(_wire(digest, vertices, normals, ids, camera=c), store)[3]["camera"]
        assert result["resolution"] == edge
        assert result["sensor_gate"]["height_m"] == .0157
        assert result["fov_degrees"] == pytest.approx(math.degrees(2 * math.atan(.0157 / .1)))
        assert result["camera_back"] == back("aps_c")
    with pytest.raises(PhotoError):
        decode_upload(_wire(digest, vertices, normals, ids, camera={**c, "camera_back":{**back(), "width_m":9}}), store)


def test_live_wire_uses_existing_shader_viewport_aspect(photo_payload):
    store, digest, vertices, normals, ids = photo_payload
    c = {**camera(), "resolution":256, "camera_back":back()}
    config = decode_upload(_wire(digest, vertices, normals, ids, camera=c, transport="fixed:3",
        live=dict(session_id="a"*32, aspect=2.01)), store)[3]
    aspect = config["live"]["aspect"]
    assert config["camera"]["sensor_gate"] == sensor_gate(back(), aspect)


def test_live_back_edits_clear_and_invalid_edit_preserve_prior_control(tmp_path):
    jobs, digest, directory = fake_jobs(tmp_path)
    jobs.spectral_root = SPECTRAL_ROOT
    update_live(jobs, digest, {"revision":1, "camera":{"camera_back":back("aps_c")}})
    path = directory / "live-control.json"
    first = path.read_bytes()
    chosen = json.loads(first)["camera"]
    assert chosen["sensor_gate"]["height_m"] == .0157
    with pytest.raises(PhotoError):
        update_live(jobs, digest, {"revision":2, "camera":{"camera_back":{**back(), "height_m":.1}}})
    assert path.read_bytes() == first
    update_live(jobs, digest, {"revision":2, "camera":{"camera_back":None, "fov_degrees":18}})
    cleared = json.loads(path.read_bytes())["camera"]
    assert cleared["fov_degrees"] == 18
    assert "sensor_gate" not in cleared and "camera_back" not in cleared


def test_fresh_export_contains_back_module():
    source = (Path(__file__).resolve().parents[1]/"examples/meowdoku/web_export.py").read_text(encoding="utf-8")
    assert '"marble_camera_back.js"' in source
