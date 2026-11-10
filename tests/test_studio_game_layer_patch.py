from pathlib import Path
import pytest
from tools.add_studio_game_layer import patch_studio_game_layer

ROOT = Path(__file__).resolve().parents[1]

def test_current_studio_narrow_layer_seams():
    source = (ROOT / 'build/meowdoku/web/marble_studio.js').read_bytes()
    output, receipt = patch_studio_game_layer(source, './studio-game-layer/')
    assert receipt['original_recovered_exactly']
    assert not receipt['enabled_by_default']
    assert not receipt['photo_camera_changed']
    assert b'subjectPose:lastStudioFrame.model' in output
    assert b'phase:"opaque"' in output and b'phase:"transparent"' in output
    with pytest.raises(ValueError, match='already installed'):
        patch_studio_game_layer(output)

def test_unknown_seam_and_prefix_refused():
    with pytest.raises(ValueError, match='not unique'):
        patch_studio_game_layer(b'not a Studio module')
    with pytest.raises(ValueError, match='safe relative'):
        patch_studio_game_layer(b'', 'https://third-party/')
