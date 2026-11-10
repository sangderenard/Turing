"""Direct sensor-pattern preview stays bounded, CPU-only and isolated from jobs."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku import film_science as science
from meowdoku.marble_photo import PhotoError

RESPONSE = {"version": 1, "centers_nm": [650, 550, 450], "sigmas_oct": [.06] * 3}


@pytest.mark.parametrize("body", [None, [], {}, {"film_response": RESPONSE, "sensor_samples": [[1, 0, 0]]},
    {"film_response": RESPONSE, "params": {"gamma": 0}},
    {"film_response": RESPONSE, "params": {"red": True}},
    {"film_response": RESPONSE, "params": {"exposure": float("inf")}},
    {"film_response": RESPONSE, "params": {"exposure": 10 ** 1000}},
    { "film_response": RESPONSE, "params": {"unexpected": 1}},
    {"film_response": RESPONSE, "params": {"normalization": "none"}},
    {"film_response": RESPONSE, "params": {"tone_curve": True}},
    {"film_response": RESPONSE, "params": {"normalization_white": 0}},
    {"film_response": RESPONSE, "params": {"normalization_white": float("nan")}},
    {"film_response": RESPONSE, "params": {"saturation": 2.01}},
    {"film_response": RESPONSE, "params": {"vibrance": -1.01}},
    {"film_response": RESPONSE, "params": {"black_point": .51}},
    {"film_response": RESPONSE, "params": {"black_point": True}}])
def test_rejects_invalid_request_before_starting_subprocess(body, monkeypatch):
    monkeypatch.setattr(science.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(PhotoError):
        science.preview_film(body, ROOT)


def test_cached_preview_uses_only_fixed_cpu_module_and_no_job_paths(tmp_path, monkeypatch):
    package = tmp_path / "camera_designer"
    package.mkdir()
    for name in ("marble_color_profile.py", "film_response.py", "spectral_sensor_basis.py"):
        (package / name).write_text("# test placeholder")
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"ok": True, "profile": {"id": "test"}}))
    monkeypatch.setattr(science.subprocess, "run", run)
    first = science.preview_film({"film_response": RESPONSE}, tmp_path)
    first["profile"]["id"] = "mutated"
    assert science.preview_film({"film_response": RESPONSE}, tmp_path)["profile"]["id"] == "test"
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [sys.executable, "-B", "-m", "camera_designer.marble_color_profile"]
    assert kwargs["cwd"] == str(tmp_path.resolve()) and kwargs["timeout"] == 15
    request = json.loads(kwargs["input"])
    assert set(request) == {"film_response", "params"}
    assert request["params"]["gamma"] == 1 and request["params"]["exposure"] == 0
    assert list(tmp_path.iterdir()) == [package]


def test_real_direct_pipeline_returns_exact_eight_patches_without_job_files():
    spectral = ROOT.parent / "spectral-analyzer"
    if not (spectral / "camera_designer/marble_color_profile.py").is_file():
        pytest.skip("sibling Spectral profile module unavailable")
    result = science.preview_film({"film_response": RESPONSE}, spectral)
    assert result["ok"] is True
    assert len(result["patch_names"]) == 8
    for field in ("sensor_rgb", "input_display_rgb8", "linear_srgb", "output_display_rgb8"):
        assert len(result[field]) == 8 and all(len(row) == 3 for row in result[field])
    assert result["profile"]["id"] == "marble-fixed3-cie1931-d65-v1"
    assert result["profile"]["neutral_fallback"] is False
    assert result["display_white"] > 0
    assert result["sensor_rgb"][1] == [1, 0, 0]
    assert result["input_display_rgb8"][1] == [255, 0, 0]


def test_supported_normalization_and_tone_are_forwarded_without_conversion():
    request = science._request({"film_response": RESPONSE, "params": {
        "normalization": "manual", "normalization_white": .25, "tone_curve": "linear"}})
    assert request["params"]["normalization"] == "manual"
    assert request["params"]["normalization_white"] == .25
    assert request["params"]["tone_curve"] == "linear"
    defaults = science._request({"film_response": RESPONSE})["params"]
    assert (defaults["normalization"], defaults["normalization_white"], defaults["tone_curve"]) == ("auto99", 1., "exponential")


def test_artistic_colour_controls_validate_and_forward_without_scaling():
    params = {"saturation": 1.25, "vibrance": -.4, "black_point": .1}
    result = science._request({"film_response": RESPONSE, "params": params})["params"]
    assert {name: result[name] for name in params} == params
    defaults = science._request({"film_response": RESPONSE})["params"]
    assert (defaults["saturation"], defaults["vibrance"], defaults["black_point"]) == (1., 0., 0.)


@pytest.mark.parametrize("normalization,percentile,tone", [
    ("auto99", 99, "exponential"), ("manual", 99, "exponential"),
    ("percentile", 99, "exponential"), ("specular", 95, "linear")])
def test_normalization_mode_defaults_and_explicit_percentile(normalization, percentile, tone):
    params = {"normalization": normalization}
    clean = science._request({"film_response": RESPONSE, "params": params})["params"]
    assert (clean["normalization_percentile"], clean["tone_curve"]) == (percentile, tone)
    for value in (1, 93.8, 100):
        clean = science._request({"film_response": RESPONSE, "params": {
            **params, "normalization_percentile": value, "tone_curve": "exponential"}})["params"]
        assert clean["normalization_percentile"] == value
        assert clean["tone_curve"] == tone


@pytest.mark.parametrize("bad", [True, "95", None, .99, 100.01, float("nan"), float("inf"), 10**1000])
def test_invalid_percentile_never_launches_preview_subprocess(bad, monkeypatch):
    monkeypatch.setattr(science.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(PhotoError, match="normalization_percentile"):
        science.preview_film({"film_response": RESPONSE, "params": {
            "normalization": "specular", "normalization_percentile": bad}}, ROOT)


def test_specular_does_not_hide_invalid_requested_tone():
    with pytest.raises(PhotoError, match="tone_curve"):
        science._request({"film_response": RESPONSE, "params": {"normalization": "specular", "tone_curve": "invalid"}})


def test_real_specular_preview_reports_effective_percentile_and_preserves_sensor_input():
    spectral = ROOT.parent / "spectral-analyzer"
    if not (spectral / "camera_designer/marble_color_profile.py").is_file():
        pytest.skip("sibling Spectral profile module unavailable")
    default = science.preview_film({"film_response": RESPONSE}, spectral)
    specular = science.preview_film({"film_response": RESPONSE, "params": {
        "normalization": "specular", "normalization_percentile": 50}}, spectral)
    assert specular["sensor_rgb"] == default["sensor_rgb"]
    assert specular["linear_srgb"] == default["linear_srgb"]
    assert specular["input_display_rgb8"] == default["input_display_rgb8"]
    recipe = specular["display_transform"]
    assert (recipe["effective_percentile"], recipe["tone_curve"], recipe["specular_clip_mode"]) == (50, "linear", True)
    assert recipe["effective_white"] == specular["display_white"]
    assert specular["output_display_rgb8"] != default["output_display_rgb8"]


@pytest.mark.parametrize("name,bad", [
    ("firefly_filter", 1), ("firefly_filter", "true"), ("firefly_filter", None),
    ("firefly_threshold", True), ("firefly_threshold", "8"), ("firefly_threshold", 1.99),
    ("firefly_threshold", 64.01), ("firefly_threshold", float("nan")),
    ("firefly_strength", False), ("firefly_strength", "1"), ("firefly_strength", -.01),
    ("firefly_strength", 1.01), ("firefly_strength", float("inf")),
])
def test_isolated_highlight_validation_rejects_before_subprocess(name, bad, monkeypatch):
    monkeypatch.setattr(science.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(PhotoError, match="firefly_"):
        science.preview_film({"film_response": RESPONSE, "params": {name: bad}}, ROOT)


def test_isolated_highlight_defaults_bounds_and_unscaled_forwarding():
    defaults = science._request({"film_response": RESPONSE})["params"]
    assert defaults["firefly_filter"] is False
    assert (defaults["firefly_threshold"], defaults["firefly_strength"]) == (8., 1.)
    for threshold, strength in [(2, 0), (64, 1), (12.5, .35)]:
        params = {"firefly_filter": True, "firefly_threshold": threshold, "firefly_strength": strength}
        clean = science._request({"film_response": RESPONSE, "params": params})["params"]
        assert {key: clean[key] for key in params} == params


def test_real_flat_patch_pipeline_filter_has_no_spatial_effect_and_records_recipe():
    spectral = ROOT.parent / "spectral-analyzer"
    if not (spectral / "camera_designer/marble_color_profile.py").is_file():
        pytest.skip("sibling Spectral profile module unavailable")
    defaults = science.preview_film({"film_response": RESPONSE}, spectral)
    params = {"firefly_filter": True, "firefly_threshold": 2, "firefly_strength": 1}
    filtered = science.preview_film({"film_response": RESPONSE, "params": params}, spectral)
    for key in ("sensor_rgb", "linear_srgb", "input_display_rgb8", "output_display_rgb8", "display_white"):
        assert filtered[key] == defaults[key], key
    recipe = filtered["display_transform"]
    assert {key: recipe[key] for key in params} == params
    assert recipe["firefly_method"] and recipe["firefly_warning"]
    assert defaults["display_transform"]["firefly_filter"] is False
