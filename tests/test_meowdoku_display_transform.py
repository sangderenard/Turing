"""Display-only normalization and tone controls, with preserved HDR provenance."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku import marble_photo as photos


@pytest.mark.parametrize("params", [
    {"normalization": "bad"}, {"tone_curve": "bad"},
    {"normalization_white": 0}, {"normalization_white": 1e13},
    {"normalization_white": float("inf")}, {"normalization_white": True},
    {"normalization_white": 10 ** 1000},
    {"exposure": 2000}, {"whitebalance": -2000},
    {"gamma": 0}, {"red": 4}, {"tint": -4}, {"intensity": -1},
    {"saturation": -1}, {"saturation": 2.01}, {"saturation": True},
    {"vibrance": 1.01}, {"vibrance": float("inf")},
    {"black_point": .51}, {"black_point": float("nan")},
    {"firefly_filter": 1}, {"firefly_filter": "false"},
    {"firefly_threshold": 1.9}, {"firefly_threshold": 65},
    {"firefly_strength": -.1}, {"firefly_strength": float("nan")},
    {"normalization_percentile": 0}, {"normalization_percentile": 100.1},
    {"normalization_percentile": True}, {"normalization_percentile": float("nan")},
])
def test_invalid_display_controls_rejected_before_status_or_file_access(tmp_path, params):
    jobs = photos.PhotoJobs(tmp_path, tmp_path)
    try:
        with pytest.raises(photos.PhotoError, match="retouch"):
            jobs.retouch("a" * 64, params)
        assert not (tmp_path / ("a" * 64)).exists()
    finally:
        jobs.close()


def test_firefly_toggle_reprocesses_saved_hdr_without_altering_measurements(tmp_path):
    digest = "b" * 64
    directory = tmp_path / digest
    directory.mkdir()
    rgb = np.full((9, 9, 3), .01); rgb[4, 4] = 100.
    hdr = directory / "exposure-000001.npz"
    np.savez_compressed(hdr, linear_rgb_exposure=rgb.astype(np.float32),
                        linear_srgb_exposure=rgb,
                        color_profile_json=np.asarray('{"id":"reference-test"}'))
    original = hdr.read_bytes()
    (directory / "status.json").write_text(json.dumps({
        "digest": digest, "state": "complete", "hdr": hdr.name}))
    jobs = photos.PhotoJobs(tmp_path, tmp_path)
    try:
        jobs.retouch(digest, {"firefly_filter": True})
        status = json.loads((directory / "status.json").read_text())
        enabled = np.asarray(Image.open(directory / status["retouched_image"]))
        np.testing.assert_array_equal(enabled[4, 4], enabled[4, 3])
        assert status["retouch_display_transform"]["firefly_filter"] is True
        jobs.retouch(digest, {"firefly_filter": False})
        status = json.loads((directory / "status.json").read_text())
        disabled = np.asarray(Image.open(directory / status["retouched_image"]))
        np.testing.assert_array_equal(disabled, photos._retonemap_linear(rgb, {}, colorimetric=True))
        assert disabled[4, 4, 0] > enabled[4, 4, 0]
        assert hdr.read_bytes() == original
    finally:
        jobs.close()


def test_saved_specular_mode_uses_selected_percentile_and_preserves_hdr(tmp_path):
    digest = "c" * 64
    directory = tmp_path / digest; directory.mkdir()
    rgb = np.repeat(np.array([0., 1., 2., 3., 4., 10.])[None, :, None], 3, axis=2)
    hdr = directory / "exposure-000001.npz"
    np.savez_compressed(hdr, linear_rgb_exposure=rgb.astype(np.float32),
                        linear_srgb_exposure=rgb,
                        color_profile_json=np.asarray('{"id":"reference-test"}'))
    original = hdr.read_bytes()
    (directory / "status.json").write_text(json.dumps({
        "digest": digest, "state": "complete", "hdr": hdr.name}))
    jobs = photos.PhotoJobs(tmp_path, tmp_path)
    try:
        jobs.retouch(digest, {"normalization": "specular", "normalization_percentile": 50.,
                             "tone_curve": "exponential"})
        status = json.loads((directory / "status.json").read_text())
        pixels = np.asarray(Image.open(directory / status["retouched_image"]))
        np.testing.assert_array_equal(pixels[0, :, 0], [0, 156, 213, 255, 255, 255])
        assert status["retouch_params"]["tone_curve"] == "linear"
        assert status["retouch_display_transform"]["effective_white"] == pytest.approx(3.)
        assert status["retouch_display_transform"]["effective_percentile"] == 50.
        assert hdr.read_bytes() == original
    finally:
        jobs.close()


@pytest.mark.parametrize("profiled", [False, True])
@pytest.mark.parametrize("creative", [False, True])
def test_saved_retouch_preserves_hdr_and_records_applied_display_transform(tmp_path, profiled, creative):
    digest = "a" * 64
    directory = tmp_path / digest
    directory.mkdir()
    raw = np.full((2, 3, 3), .18, np.float32)
    hdr = directory / "exposure-000001.npz"
    extra = {"linear_srgb_exposure": np.full((2, 3, 3), .18, np.float64),
             "color_profile_json": np.asarray('{"id":"reference-test"}')} if profiled else {}
    np.savez_compressed(hdr, linear_rgb_exposure=raw, **extra)
    original = hdr.read_bytes()
    (directory / "status.json").write_text(json.dumps({
        "digest": digest, "state": "complete", "hdr": hdr.name}))
    params = {"normalization": "manual", "normalization_white": 1., "tone_curve": "linear"}
    if creative:
        params.update(saturation=1.4, vibrance=.7, black_point=.1)
    jobs = photos.PhotoJobs(tmp_path, tmp_path)
    try:
        result = jobs.retouch(digest, params)
        status = json.loads((directory / "status.json").read_text())
        assert status["retouch_params"] == photos._display_params(params)
        assert hdr.read_bytes() == original
        pixels = np.array(Image.open(directory / status["retouched_image"]))
        if profiled:
            assert result == {"ok": True}
            expected = round((1.055 * ((.18 - .1) / .9) ** (1 / 2.4) - .055) * 255) if creative else 118
            np.testing.assert_array_equal(pixels, np.full((2, 3, 3), expected, np.uint8))
            assert status["retouch_display_transform"]["effective_white"] == 1.
            assert status["retouch_display_transform"]["tone_formula"] == ("clip(x,0,1)" if creative else "clip(Y/white,0,1)")
        else:
            assert "Legacy HDR" in result["warning"]
            assert status["retouch_display_transform"]["legacy"] is True
            np.testing.assert_array_equal(pixels, photos._retonemap_linear(raw.astype(float), {}))
    finally:
        jobs.close()
