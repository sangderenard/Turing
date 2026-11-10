"""The additive appearance hook must not recompile or rewrite the saved game."""

import hashlib
import re
from pathlib import Path

import pytest

from tools.add_mechanical_creature_appearance import (
    COLOR_ORIGINAL, COLOR_REPLACEMENT, DRAW_ORIGINAL, DRAW_REPLACEMENT,
    HOOK_SOURCE, AMMO_SOURCE, PATCHES, PLAYABLE_SHA256, behavior_bounds, patch_playable_page,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def playable_bytes():
    path = ROOT / "build/dot-playable-tires-20261110/index.html"
    if not path.is_file():
        pytest.skip("exact original playable page is not present in this checkout")
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == PLAYABLE_SHA256
    return data


def test_refuse_any_unverified_source():
    with pytest.raises(ValueError, match="unverified playable"):
        patch_playable_page(b"<html>not the original game</html>")


def test_only_declared_behavior_seams_change_and_all_other_bytes_survive(playable_bytes):
    output, receipt = patch_playable_page(playable_bytes)
    original, modified = playable_bytes.decode(), output.decode()
    start, end = behavior_bounds(original)
    new_start, new_end = behavior_bounds(modified)
    assert original[:start] == modified[:new_start]
    assert original[end:] == modified[new_end:]
    hook = HOOK_SOURCE.read_bytes().decode() + "\n" + AMMO_SOURCE.read_bytes().decode()
    restored = modified[new_start:new_end].replace(hook + "\n" + PATCHES[0][1], PATCHES[0][1], 1)
    for before, after in reversed(PATCHES):
        restored = restored.replace(after, before, 1)
    assert restored == original[start:end]
    assert modified.count(hook) == 1
    assert receipt["unchanged_model_worker_wasm"]
    assert receipt["original_behavior_recovered_exactly"]
    assert not receipt["compile_performed"]
    assert not receipt["published"]


def test_original_depth_shadow_picking_input_and_world_functions_are_exact(playable_bytes):
    original = playable_bytes.decode().replace("\r\n", "\n")
    modified = patch_playable_page(playable_bytes)[0].decode().replace("\r\n", "\n")
    for name in ("renderCameraDepthPass", "renderShadowPass", "rebuildPortableSceneMesh",
                 "raySceneTriangle", "registerProjectilePhysicsMembership", "updateViewportControls",
                 "bindMobileStick", "renderMobileControls"):
        expression = re.compile(r"^function " + name + r"\([\s\S]*?^\}", re.M)
        before, after = expression.search(original), expression.search(modified)
        assert before is not None, name
        assert after is not None, name
        assert before.group() == after.group(), name
    assert "  drawSceneMeshes(gl);\n  gl.bindFramebuffer(gl.FRAMEBUFFER,null);" in modified


def test_unsafe_script_is_rejected(playable_bytes):
    with pytest.raises(ValueError, match="unsafe script"):
        patch_playable_page(playable_bytes, "</script><script>bad()</script>")
