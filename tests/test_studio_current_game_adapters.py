"""Matched current full-game adapter guards; no compiler, stage or publication."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import pytest

from tools.add_mechanical_creature_appearance import patch_playable_page, patches_for_page, behavior_bounds, HOOK_SOURCE, AMMO_SOURCE
from tools.add_studio_registry_append import patch_studio_registry_append, canonical_append
from tools.add_mechanical_creature_studio_building import patch_studio_building
from tools.add_mechanical_creature_studio_bridge import patch_studio_bridge
from tools.add_studio_game_entry import patch_studio_game_entry

ROOT=Path(__file__).resolve().parents[1]
CURRENT=ROOT/'build/dot-tire-state-candidate-20261110/index.html'
SHA='a7b75ff09b7fcd903408623cd33ef91e12b5262df0be7ce4d676190ca71ebaf5'


@pytest.fixture(scope='module')
def current():
    if not CURRENT.exists(): pytest.skip('Frozen matched current game is not present')
    data=CURRENT.read_bytes();assert hashlib.sha256(data).hexdigest()==SHA
    return data


def test_current_appearance_reverses_every_byte_and_preserves_authored_world(current):
    output,receipt=patch_playable_page(current,expected_source_sha256=SHA)
    original=current.decode();modified=output.decode();start,end=behavior_bounds(original)
    new_start,new_end=behavior_bounds(modified);patches=patches_for_page(original[start:end])
    hook=HOOK_SOURCE.read_bytes().decode()+'\n'+AMMO_SOURCE.read_bytes().decode()
    restored=modified[new_start:new_end].replace(hook+'\n'+patches[0][1],patches[0][1],1)
    for before,after in reversed(patches): restored=restored.replace(after,before,1)
    assert (modified[:new_start]+restored+modified[new_end:]).encode()==current
    assert 'gl.drawArrays(gl.TRIANGLES, 0, shaderViewer.authoredWorldSurface.mesh.length / 9);' in modified
    assert receipt['unchanged_model_worker_wasm']
    for name in ['launchQualifiedLivingMap','finishVehicleQualification','updateVehicleQualification','installAuthoredWorldSurfacePacket']:
        pattern=r'^function '+name+r'\b[\s\S]*?^}'
        assert re.search(pattern,original,re.M)[0]==re.search(pattern,modified,re.M)[0]


def test_current_canonical_registry_is_verified_noop_and_forged_hash_refused(current):
    output,receipt=patch_studio_registry_append(current)
    assert output==current and receipt['already_canonical']
    assert receipt['inline_utility_sha256']==canonical_append()[2]
    changed=current.replace(b'this.world = world;',b'this.world = foreign;',1)
    with pytest.raises(ValueError,match='content key'): patch_studio_registry_append(changed)
    text=changed.decode();pattern=r'^// turing-runtime-utility: turing\.world\.registry \[javascript-utility:sha256:([0-9a-f]{64})\]\n'
    header=re.search(pattern,text,re.M);end=text.index('// turing-runtime-utility:',header.end())
    key=hashlib.sha256(text[header.end():end].strip().encode()).hexdigest()
    changed=text[:header.start()]+header[0].replace(header[1],key)+text[header.end():]
    with pytest.raises(ValueError,match='not the exact canonical utility'):patch_studio_registry_append(changed.encode())


def test_current_full_adapter_chain_and_qualification(current,tmp_path):
    data,_=patch_playable_page(current,expected_source_sha256=SHA)
    data,_=patch_studio_registry_append(data)
    definition=json.loads((ROOT/'build/dot-studio-game-transfer-20261110/studio-building-definition-v2.json').read_text())
    data,_=patch_studio_building(data,definition)
    data,_=patch_studio_bridge(data,expected_source_sha256=hashlib.sha256(data).hexdigest())
    data,receipt=patch_studio_game_entry(data,expected_source_sha256=hashlib.sha256(data).hexdigest())
    assert receipt['source_recovered_exactly']
    page=tmp_path/'current.html';page.write_bytes(data)
    start,end=behavior_bounds(data.decode());script=tmp_path/'behavior.js';script.write_text(data.decode()[start:end],encoding='utf8')
    for command in [['node','--check',str(script)],['node',str(ROOT/'tests/mechanical_creature_studio_entry_check.mjs'),str(page)]]:
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stdout+result.stderr
    assert hashlib.sha256(CURRENT.read_bytes()).hexdigest()==SHA


def test_oriented_qualification_snapshot_backpressure():
    candidate=ROOT/'build/dot-oriented-arena-candidate-20261110/index.html'
    if not candidate.exists(): pytest.skip('Frozen oriented full game is not present')
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()=='297cda67f0c96cc2dada8278fcd1f559b279fd14ba3c8d31b62448e1f1fa82a1'
    result=subprocess.run(['node',str(ROOT/'tests/mechanical_creature_qualification_backpressure_check.mjs'),str(candidate)],
        cwd=ROOT,capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
