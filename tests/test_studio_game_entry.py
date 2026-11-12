"""Bounded checks of the original-player entry adapter, without a browser."""
import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest

from tools.add_mechanical_creature_studio_bridge import patch_studio_bridge
from tools.add_studio_game_entry import patch_studio_game_entry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def source():
    return patch_studio_bridge((ROOT / 'build/dot-newer-tire-dynamics-20261110/index.html').read_bytes())[0]


def test_entry_preserves_original_model_and_native_respawn(source):
    output, receipt = patch_studio_game_entry(source, expected_source_sha256=hashlib.sha256(source).hexdigest())
    assert receipt['source_recovered_exactly']
    assert receipt['entry'] == 'outside-studio-entrance'
    assert not receipt['new_controller'] and not receipt['new_clock']
    assert b'function respawnViewportActor(studioSpawn=null){' in output
    assert b'spawn=studioSpawn||mechanicalCreatureStudioRespawnPosition()||actor?.entity?.pose?.position||actor?.record?.pose?.position;' in output
    assert source.split(b'id="abstract-ui-model">')[1].split(b'</script>')[0] == output.split(b'id="abstract-ui-model">')[1].split(b'</script>')[0]
    with pytest.raises(ValueError, match='already installed'):
        patch_studio_game_entry(output, expected_source_sha256=hashlib.sha256(output).hexdigest())


@pytest.mark.parametrize('alter', [lambda s:s.replace(b'function respawnViewportActor(){',b'function missingRespawn(){'),
    lambda s:s.replace(b'spawn=actor?.entity?.pose?.position||actor?.record?.pose?.position;',b'spawn=null;')])
def test_entry_refuses_unknown_seam(source, alter):
    changed=alter(source)
    with pytest.raises(ValueError, match='not unique'):
        patch_studio_game_entry(changed,expected_source_sha256=hashlib.sha256(changed).hexdigest())


def test_entry_requires_verified_bytes_and_runtime_bridge(source):
    with pytest.raises(ValueError, match='Unverified'):
        patch_studio_game_entry(source,expected_source_sha256='0'*64)
    with pytest.raises(ValueError, match='requires'):
        patch_studio_game_entry(b'',expected_source_sha256=hashlib.sha256(b'').hexdigest())


@pytest.mark.skipif(shutil.which('node') is None,reason='node is not installed')
def test_actual_native_player_entry(source,tmp_path):
    output,_=patch_studio_game_entry(source,expected_source_sha256=hashlib.sha256(source).hexdigest())
    candidate=tmp_path/'entry.html';candidate.write_bytes(output)
    result=subprocess.run(['node',str(ROOT/'tests/mechanical_creature_studio_entry_check.mjs'),str(candidate)],
        cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.skipif(shutil.which('node') is None,reason='node is not installed')
def test_actual_view_handoff():
    result=subprocess.run(['node',str(ROOT/'tests/meowdoku_studio_view_handoff_check.mjs')],
        cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
