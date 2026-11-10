"""Bounce count admission is shared by still photographs and live rays; no GPU."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]
from meowdoku import marble_photo as photos  # noqa: E402


SAVED = {"digest": "a" * 64, "geometry_sha256": "b" * 64,
         "physical": {"radius-mm": 25}}


@pytest.mark.parametrize("live", [False, True])
def test_bounce_count_default_and_all_integer_counts(live):
    request = {"object_digest": SAVED["digest"], "transport": "fixed:3"}
    if live:
        request["live"] = {"session_id": "c" * 32, "history_samples": 1, "aspect": 1}
    assert photos._config(request, SAVED)[0]["max_bounces"] == 4
    for count in range(1, 201):
        config, _ = photos._config({**request, "max_bounces": count}, SAVED)
        assert config["max_bounces"] == count


@pytest.mark.parametrize("count", [0, 201, 1.5, True, False, "4.5", float("nan"), float("inf")])
def test_bounce_count_rejects_values_outside_supported_range(count):
    with pytest.raises(photos.PhotoError, match="max_bounces"):
        photos._config({"object_digest": SAVED["digest"], "max_bounces": count}, SAVED)
