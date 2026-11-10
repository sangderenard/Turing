"""Bounded photo config boundary checks, without server startup or GPU jobs."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku.marble_photo import PhotoError, _config, _film_response

DEFAULT = {"version": 1, "centers_nm": [650, 550, 450], "sigmas_oct": [.06, .06, .06]}
SAVED = {"digest": "a" * 64, "geometry_sha256": "b" * 64, "physical": {"radius-mm": 10}}


def config(**kwargs):
    return _config({"object_digest": SAVED["digest"], "transport": "fixed:3", **kwargs}, SAVED)[0]


def test_response_preserved_in_persisted_job_config_and_digest():
    default = config()
    assert default["film_response"] == DEFAULT
    custom = {"version": 1, "centers_nm": [620, 540, 465], "sigmas_oct": [.1, .2, .03]}
    changed = config(film_response=custom)
    assert changed["film_response"] == custom
    custom["centers_nm"][0] = 500
    assert changed["film_response"]["centers_nm"][0] == 620
    canonical = lambda c: json.dumps(c, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical(changed)).digest() != hashlib.sha256(canonical(default)).digest()
    assert json.loads(canonical(changed))["film_response"] == changed["film_response"]


def test_monochrome_remains_unchanged_and_rejects_unapplied_rgb_settings():
    assert "film_response" not in config(transport="fixed:1")
    with pytest.raises(PhotoError, match="requires"):
        config(transport="fixed:1", film_response=DEFAULT)


@pytest.mark.parametrize("value", [None, [], {}, {**DEFAULT, "version": True}, {**DEFAULT, "version": 1.0},
    {**DEFAULT, "version": 2}, {**DEFAULT, "extra": 0}, {**DEFAULT, "centers_nm": [650, 550]},
    {**DEFAULT, "centers_nm": [650, 550, 781]}, {**DEFAULT, "centers_nm": [379, 550, 450]},
    {**DEFAULT, "centers_nm": [650, "550", 450]}, {**DEFAULT, "centers_nm": [10 ** 1000, 550, 450]},
    {**DEFAULT, "sigmas_oct": [True, .06, .06]},
    {**DEFAULT, "sigmas_oct": [float("nan"), .06, .06]}, {**DEFAULT, "sigmas_oct": [float("inf"), .06, .06]},
    {**DEFAULT, "sigmas_oct": [.009, .06, .06]}, {**DEFAULT, "sigmas_oct": [1.01, .06, .06]}])
def test_invalid_response_rejected_before_launch(value):
    with pytest.raises(PhotoError, match="film_response"):
        config(film_response=copy.deepcopy(value))


def test_independent_centres_and_bounds_are_valid():
    value = {"version": 1, "centers_nm": [380, 780, 380], "sigmas_oct": [.01, 1, .01]}
    assert _film_response(value) == value


def test_exposure_default_is_30_seconds_but_explicit_choices_survive():
    assert config()["exposure_seconds"] == 30
    assert config(exposure_seconds=600)["exposure_seconds"] == 600


def test_color_profile_revision_is_owned_by_server_and_invalidates_legacy_digest():
    current = config(output_color_profile="caller-cannot-select-a-profile")
    assert current["output_color_profile"] == "marble-fixed3-cie1931-d65-v1"
    old = {k: v for k, v in current.items() if k != "output_color_profile"}
    encode = lambda c: json.dumps(c, sort_keys=True).encode()
    assert hashlib.sha256(encode(old)).digest() != hashlib.sha256(encode(current)).digest()
    assert "output_color_profile" not in config(transport="fixed:1")
