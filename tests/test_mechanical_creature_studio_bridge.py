import hashlib
import re

import pytest

from tools.add_mechanical_creature_studio_bridge import (
    ROOT, HELPER, FUNCTION, START, START_NEW, END, END_NEW, READY, READY_NEW, patch_studio_bridge,
)
from tools.add_mechanical_creature_appearance import behavior_bounds, patch_playable_page, patches_for_page, HOOK_SOURCE, AMMO_SOURCE


@pytest.fixture(params=["build/dot-playable-tires-20261110/index.html", "build/dot-living-map-runtime-20261110/matched/index.html"])
def original(request):
    return (ROOT / request.param).read_bytes()


def test_both_original_games_preserve_every_existing_byte(original):
    output, receipt = patch_studio_bridge(original)
    document = output.decode(); separator = "\r\n" if b"\r\n" in original else "\n"
    helper = HELPER.read_bytes().decode()
    recovered = document.replace(helper + "\n" + FUNCTION, FUNCTION, 1)
    for before, after in ((START, START_NEW), (END, END_NEW), (READY, READY_NEW)):
        recovered = recovered.replace(after.replace("\n", separator), before.replace("\n", separator), 1)
    assert recovered.encode() == original
    assert receipt["source_recovered_exactly"] and receipt["worker_model_controls_unchanged"]
    assert not receipt["new_clock"] and not receipt["published"]
    assert document.index("applyMechanicalCreatureStudioEnvironment();", document.index(FUNCTION)) < document.index("updateViewportControls(dt);", document.index(FUNCTION))
    assert document.index("publishMechanicalCreatureStudioSnapshot(now,dt);") > document.index("  updateShaderViewer();", document.index(FUNCTION))


def test_refuse_unknown_and_repeat_installation(original):
    with pytest.raises(ValueError, match="unverified"):
        patch_studio_bridge(original + b"changed")
    output, _ = patch_studio_bridge(original)
    with pytest.raises(ValueError, match="already installed"):
        patch_studio_bridge(output, expected_source_sha256=hashlib.sha256(output).hexdigest())


def test_appearance_and_bridge_compose_on_both_complete_versions(original):
    appearance, _ = patch_playable_page(original)
    combined, _ = patch_studio_bridge(appearance, expected_source_sha256=hashlib.sha256(appearance).hexdigest())
    pattern = rb'<script type="application/json" id="abstract-ui-model">(.*?)</script>'
    assert re.search(pattern, original, re.S)[0] == re.search(pattern, combined, re.S)[0]
    text = appearance.decode(); start, end = behavior_bounds(text)
    original_text = original.decode(); old_start, old_end = behavior_bounds(original_text)
    pairs = patches_for_page(original_text[old_start:old_end])
    helper = HOOK_SOURCE.read_bytes().decode() + "\n" + AMMO_SOURCE.read_bytes().decode()
    recovered = text[start:end].replace(helper + "\n" + pairs[0][1], pairs[0][1], 1)
    for before, after in reversed(pairs):
        recovered = recovered.replace(after, before, 1)
    assert recovered == original_text[old_start:old_end]
