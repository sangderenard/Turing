"""Capture planning/catalog contracts, not a claim of rendered geodesic views."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

from examples.meowdoku.marble_capture_jig import geodesic_capture_plan, record_completed_view
from src.cells.softbody.geometry.geodesic import geodesic_sphere

SPECTRAL = Path(__file__).resolve().parents[2] / "spectral-analyzer"


@pytest.fixture
def config():
    return {"object_digest": "a" * 64, "geometry_sha256": "b" * 64,
            "transport": "fixed:3", "orbit_frames": 16, "scale_m": .025,
            "object_center_m": [0., 0., .18], "exposure_seconds": .1,
            "camera": {"eye_m": [0., 0., -.02], "target_m": [0., 0., .18],
                       "focus_distance_m": .2, "aperture_radius_m": .001,
                       "fov_degrees": 30., "resolution": 128},
            "lights": [{"pos": [.2, .2, .3], "power": 1., "label": "fixed"}],
            "fixtures": {"vertices": [0., 1., 2.], "material_ids": [3]},
            "film_response": {"version": 1, "centers_nm": [650., 550., 450.],
                              "sigmas_oct": [.06, .06, .06]}}


@pytest.mark.parametrize("frequency,count", [(1, 12), (2, 42), (3, 92)])
def test_uses_canonical_geodesic_and_native_camera(config, frequency, count):
    before = deepcopy(config)
    plan = geodesic_capture_plan(config, frequency=frequency, spectral_root=SPECTRAL)
    assert config == before
    assert len(plan["jobs"]) == count
    expected = geodesic_sphere(frequency)
    np.testing.assert_array_equal([v["direction"] for v in plan["jobs"]], expected["verts"])
    np.testing.assert_array_equal(plan["view_faces"], expected["faces"])
    assert not plan["suspension_verified"] and not plan["product_ready"]
    assert not plan["source_mesh_bounds_verified"] and not plan["view_occlusion_verified"]
    from camera_designer.camera_frame import sensor_camera_basis
    from camera_software.render_assets import LightFieldCondition
    assert len({v["condition_key"] for v in plan["jobs"]}) == count
    for view in plan["jobs"]:
        job = view["config"]
        assert job["orbit_frames"] == 1
        for key in ("lights", "fixtures", "scale_m", "object_digest", "geometry_sha256", "film_response"):
            assert job[key] == config[key]
        for key in ("focus_distance_m", "aperture_radius_m", "fov_degrees", "resolution"):
            assert job["camera"][key] == config["camera"][key]
        basis = sensor_camera_basis(job["camera"]["eye_m"], job["camera"]["target_m"])
        for key, value in zip(("right", "up", "forward"), basis[1:]):
            np.testing.assert_array_equal(view["basis"][key], value)
        assert view["condition_key"] == LightFieldCondition(**view["condition"]).condition_key
        assert np.linalg.norm(np.array(job["camera"]["eye_m"]) - plan["center_m"]) == pytest.approx(.2)


def test_rejects_clipped_or_unbound_views(config):
    for patch in ({"transport": "fixed:1"}, {"live": {"active": True}},
                  {"client_intersections": True}, {"object_digest": "unknown"},
                  {"scale_m": .19}):
        with pytest.raises(ValueError):
            geodesic_capture_plan({**config, **patch}, spectral_root=SPECTRAL)
    for frequency in (True, 0, 4, 1.5):
        with pytest.raises(ValueError):
            geodesic_capture_plan(config, frequency=frequency, spectral_root=SPECTRAL)
    config["camera"]["target_m"] = [0., .01, .18]
    with pytest.raises(ValueError, match="subject center"):
        geodesic_capture_plan(config, spectral_root=SPECTRAL)


def fixture_result(directory, plan):
    """A file-contract fixture only; never represented as actual ray transport."""
    config = plan["jobs"][0]["config"]
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    (directory / "config.json").write_bytes(canonical)
    np.savez_compressed(directory / "payload.npz", vertices=np.zeros((1, 9), np.float32),
                        material_ids=np.zeros(1, np.int32), pbr=np.zeros((1, 16), np.float32))
    digest = hashlib.sha256((directory / "payload.npz").read_bytes() + canonical).hexdigest()
    from camera_designer.marble_color_profile import sensor_to_linear_srgb
    from camera_designer.film_response import normalize_film_response
    response = normalize_film_response(config["film_response"])
    sensor = np.full((128, 128, 3), .1, np.float32)
    linear, profile = sensor_to_linear_srgb(sensor, response)
    np.savez_compressed(directory / "exposure-000001.npz",
                        linear_srgb_exposure=linear, linear_rgb_exposure=sensor,
                        color_profile_json=np.asarray(json.dumps(profile, sort_keys=True)),
                        film_response_version=np.asarray(response["version"], np.int32),
                        film_centers_nm=np.asarray(response["centers_nm"], np.float64),
                        film_sigmas_oct=np.asarray(response["sigmas_oct"], np.float64))
    Image.new("RGB", (128, 128), (10, 20, 30)).save(directory / "preview-000001.png")
    status = {"state": "complete", "digest": digest, "total_frames": 1, "sensor_sites": 16384,
              "color_profile": profile,
              "frames": [{"index": 0, "image": "preview-000001.png", "hdr": "exposure-000001.npz"}]}
    (directory / "status.json").write_text(json.dumps(status))
    return status


def test_existing_catalog_records_only_bound_completed_images(config, tmp_path):
    plan = geodesic_capture_plan(config, spectral_root=SPECTRAL)
    from camera_software.render_assets import RenderAssetCatalog, LightFieldCondition
    from camera_software.display_scene import DisplayProductKind
    fixture_result(tmp_path, plan)
    catalog = RenderAssetCatalog(str(tmp_path / "catalog.json"))
    before = (tmp_path / "exposure-000001.npz").read_bytes()
    record = record_completed_view(plan, 0, tmp_path, catalog, spectral_root=SPECTRAL)
    assert record.product_kind is DisplayProductKind.IMAGE  # Not invented LIGHT_FIELD.
    assert record.samples == 16384
    assert record.metadata["object_coverage"] == "not yet isolated"
    assert not record.metadata["view_dependent_product_ready"]
    assert (tmp_path / "exposure-000001.npz").read_bytes() == before
    again = RenderAssetCatalog(str(tmp_path / "catalog.json"))
    assert again.find(record.target_key, LightFieldCondition(**plan["jobs"][0]["condition"]),
                      DisplayProductKind.IMAGE).record_key == record.record_key
    with pytest.raises(ValueError, match="does not match"):
        record_completed_view(plan, 1, tmp_path, catalog, spectral_root=SPECTRAL)
    changed = deepcopy(plan); changed["radius_m"] *= 2
    with pytest.raises(ValueError, match="identity changed"):
        record_completed_view(changed, 0, tmp_path, catalog, spectral_root=SPECTRAL)


@pytest.mark.parametrize("change", ["unfinished", "foreign", "traversal", "nonfinite", "bad-preview",
                                    "wrong-profile", "jpeg", "negative-samples"])
def test_refuses_unfinished_or_unbound_artifacts(config, tmp_path, change):
    plan = geodesic_capture_plan(config, spectral_root=SPECTRAL)
    from camera_software.render_assets import RenderAssetCatalog
    status = fixture_result(tmp_path, plan)
    if change == "unfinished": status["state"] = "exposing"
    if change == "foreign": status["digest"] = "c" * 64
    if change == "traversal": status["frames"][0]["hdr"] = "../outside.npz"
    if change == "nonfinite":
        np.savez_compressed(tmp_path / "exposure-000001.npz", linear_srgb_exposure=np.full((128, 128, 3), np.nan))
    if change == "bad-preview": Image.new("RGB", (16, 16)).save(tmp_path / "preview-000001.png")
    if change == "jpeg": Image.new("RGB", (128, 128)).save(tmp_path / "preview-000001.png", format="JPEG")
    if change == "wrong-profile": status["color_profile"]["id"] = "unrelated-profile"
    if change == "negative-samples": status["sensor_sites"] = -1
    (tmp_path / "status.json").write_text(json.dumps(status))
    catalog = RenderAssetCatalog()
    with pytest.raises(ValueError): record_completed_view(plan, 0, tmp_path, catalog, spectral_root=SPECTRAL)


def test_capture_distance_is_part_of_catalog_identity(config):
    first = geodesic_capture_plan(config, spectral_root=SPECTRAL)
    other = deepcopy(config)
    other["camera"]["eye_m"] = [0., 0., -.22]
    second = geodesic_capture_plan(other, spectral_root=SPECTRAL)
    assert first["capture_distance_m"] != second["capture_distance_m"]
    assert first["rig_key"] != second["rig_key"]
    assert all(a["condition_key"] != b["condition_key"] for a, b in zip(first["jobs"], second["jobs"]))


def test_original_photo_normalizer_accepts_each_planned_view(config):
    # Real config normalization only; never starts PhotoJobs or the GPU owner.
    from examples.meowdoku.marble_photo import _config
    saved = {"digest": config["object_digest"], "geometry_sha256": config["geometry_sha256"],
             "physical": {"radius-mm": 25}}
    request = deepcopy(config)
    request.pop("fixtures")  # The smaller unit fixture above is not an admitted stage.
    request["lights"][0]["dir"] = [-.2, -.2, -.12]
    normalized, _ = _config(request, saved, SPECTRAL)
    plan = geodesic_capture_plan(normalized, spectral_root=SPECTRAL)
    for view in plan["jobs"]:
        repeated, _ = _config(view["config"], saved, SPECTRAL)
        assert repeated == view["config"]


def test_original_epoch_format_allows_more_than_six_digits(config, tmp_path):
    plan = geodesic_capture_plan(config, spectral_root=SPECTRAL)
    from camera_software.render_assets import RenderAssetCatalog
    status = fixture_result(tmp_path, plan)
    for key, prefix, extension in (("image", "preview", "png"), ("hdr", "exposure", "npz")):
        old = status["frames"][0][key]
        new = f"{prefix}-1000000.{extension}"
        (tmp_path / old).rename(tmp_path / new)
        status["frames"][0][key] = new
    (tmp_path / "status.json").write_text(json.dumps(status))
    assert record_completed_view(plan, 0, tmp_path, RenderAssetCatalog(), spectral_root=SPECTRAL)
