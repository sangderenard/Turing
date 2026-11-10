"""Execute authored worker functions without compiling a vehicle."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def test_junction_loss_measurement_has_no_spawn_impulse_or_mechanical_effect():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for actual worker-function checks")
    source = (Path(__file__).parents[1] / "src/compiler/state_loop_deployment.py").read_text(encoding="utf-8")
    functions = []
    for name in ("ensureVehicleDamage", "updateVehicleDamage"):
        start = source.index("function " + name + "(")
        end = source.index("\nfunction ", start + 1)
        functions.append(source[start:end].replace("{{", "{").replace("}}", "}"))
    script = "\n".join(functions) + r'''
const assert=require('assert/strict');
function trial(scale){
 const edge={identity:'test.link',a:'a',b:'b',rest_length:1,constraint:'rigid-distance',
  joint_bushings:{a:{linear_damping_n_s_per_m:3*scale},b:{linear_damping_n_s_per_m:7*scale}}};
 const body={config:{suspension:{travel:1,stiffness:26000},wheels:{wheelbase_half_length:1,track_half_width:1},
  mechanical_graph:{edges:[edge]}},defaults:{},compressions:{},velocity:[0,0,0],
  wheelOmegas:{},mechanicalGraphPositions:new Map([['a',[0,0,0]],['b',[2,0,0]]])};
 const physical=()=>JSON.stringify([body.velocity,body.compressions,body.wheelOmegas,
  body.rollVelocity,body.pitchVelocity,body.yawVelocity]);
 const before=physical();updateVehicleDamage(body,.1);
 const first=body.damage.junctions['test.link'];
 assert.equal(first.powerW,0);assert.equal(first.energyJ,0);
 // Existing actuator-rate consumer is deliberately unchanged.
 assert.equal(first.relativeLinearSpeedMps,10);
 assert.equal(first.estimatedRelativeLinearSpeedMps,0);
 updateVehicleDamage(body,.1);assert.equal(first.powerW,0);assert.equal(first.energyJ,0);
 body.mechanicalGraphPositions.set('b',[2.1,0,0]);updateVehicleDamage(body,.1);
 assert.ok(Math.abs(first.powerW-10*scale)<1e-8);
 assert.ok(Math.abs(first.energyJ-scale)<1e-8);
 assert.equal(physical(),before);
 assert.equal(body.damage.junctionLossTelemetry.mechanicalEnergyRemoved,false);
 assert.equal(body.damage.junctionLossTelemetry.kind,'estimated-constitutive-loss');
 assert.match(body.damage.junctionLossTelemetry.angularRate,/proxy-not-relative/);
 return {physical:physical(),power:first.powerW,energy:first.energyJ};
}
const one=trial(1),hundred=trial(100);assert.equal(one.physical,hundred.physical);
assert.ok(Math.abs(hundred.power/one.power-100)<1e-10);
console.log(JSON.stringify({passed:true,one,hundred}));
'''
    result = subprocess.run([node, "-"], input=script, text=True,
                            capture_output=True, timeout=15, check=True)
    assert json.loads(result.stdout)["passed"]


def test_geometry_reset_restarts_only_the_loss_measurement_history():
    source = (Path(__file__).parents[1] / "src/compiler/state_loop_deployment.py").read_text(encoding="utf-8")
    assert 'junction.previousLength=Number(edge.rest_length)' in source
    assert 'junction.lossMeasurementInitialized=false' in source
    assert 'jointBushingLossTelemetry:body.damage?.junctionLossTelemetry||null' in source
