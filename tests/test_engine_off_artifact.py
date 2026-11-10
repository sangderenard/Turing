"""Guarded startup patch preserves runtime data and honors explicit crank state."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import pytest
from tools.repair_mechanical_creature_engine_off import repair, REPLACEMENTS


def test_rejects_unknown_base():
    with pytest.raises(ValueError, match="Unknown"):
        repair(b"unknown")


def test_crank_seed_respects_off_and_existing_state():
    seed = REPLACEMENTS[1][1]
    source = """
const assert=require('assert');
for(const [ignition,existing,expected] of [[false,undefined,0],[true,undefined,65],[false,12,12]]){
 const body={energy:{ignitionOn:ignition},defaults:{engine_idle_angular_speed:65},wasmState:existing===undefined?{}:{engine_angular_speed:existing}};
 if(!Number.isFinite(body.wasmState?.engine_angular_speed))body.wasmState={...(body.wasmState||{}),SEED
 assert.equal(body.wasmState.engine_angular_speed,expected);
}
""".replace("SEED", seed)
    subprocess.run(["node", "-e", source], check=True, capture_output=True, text=True)


def test_guarded_artifact_changes_no_model_data_except_worker():
    path = Path("build/dot-machine-graph-hud-20261110/index.html")
    if not path.exists():
        pytest.skip("Preserved integration artifact not present")
    data = path.read_bytes()
    result, report = repair(data)
    assert report["reverse_byte_restore"]
    def model(raw):
        return json.loads(re.search(rb'<script\b[^>]*\bid="abstract-ui-model"[^>]*>(.*?)</script>', raw, re.S)[1])
    old, new = model(data), model(result)
    old_worker = old["loop_deployment"]["workers"][0]["source"]
    new_worker = new["loop_deployment"]["workers"][0]["source"]
    assert new_worker == old_worker.replace(*REPLACEMENTS[1]).replace(*REPLACEMENTS[2])
    new["loop_deployment"]["workers"][0]["source"] = old_worker
    assert old == new


def test_respawn_keeps_off_crank_zero_in_source_and_patch():
    source = Path('src/compiler/state_loop_deployment.py').read_text(encoding='utf-8')
    start = source.index('  if(body.pendingRespawn)')
    end = source.index('    body.tireDeformation', start)
    code = source[start:end].replace('{{', '{').replace('}}', '}') + '}'
    assert REPLACEMENTS[2][1] in code
    script = "const assert=require('assert');for(const ignition of [false,true]){const body={energy:{ignitionOn:ignition},pendingRespawn:{position:[1,2,3]}};" + code + "assert.equal(body.wasmState.engine_angular_speed,ignition?undefined:0);assert.equal(body.effectiveThrottle,0);}"
    subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True)
