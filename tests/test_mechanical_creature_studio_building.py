"""Narrow startup append preserves every existing model and behavior byte."""
import json
import re
from pathlib import Path

import pytest

from tools.add_mechanical_creature_studio_building import SEAM, patch_studio_building


ROOT = Path(__file__).resolve().parents[1]


def test_startup_append_is_reversible_and_model_is_not_reserialized():
    source = (ROOT / "build/meowdoku/web/mechanical_creature_play.html").read_bytes()
    definition = {"schema": "mechanical-creature-studio-building-v1", "identity": "studio:test", "boxes": []}
    output, receipt = patch_studio_building(source, definition)
    before, after = source.decode(), output.decode()
    pattern = r'<script type="application/json" id="abstract-ui-model">(.*?)</script>'
    assert re.search(pattern, before, re.S)[0] == re.search(pattern, after, re.S)[0]
    prefix, suffix = before.split(SEAM)
    assert after.startswith(prefix + SEAM)
    assert after.endswith(suffix)
    assert receipt["original_page_recovered_exactly"]
    assert receipt["embedded_model_worker_unchanged"]
    assert not receipt["published"]


def test_bad_definition_and_ambiguous_seam_rejected():
    with pytest.raises(ValueError, match="definition"):
        patch_studio_building(b"", {})
    definition = {"schema": "mechanical-creature-studio-building-v1", "identity": "test", "boxes": []}
    with pytest.raises(ValueError, match="not unique"):
        patch_studio_building((SEAM + SEAM).encode(), definition)


def test_definition_is_script_safe():
    definition = {"schema": "mechanical-creature-studio-building-v1", "identity": "</script>", "boxes": []}
    output, _ = patch_studio_building(SEAM.encode(), definition)
    assert b"</script>" not in output
