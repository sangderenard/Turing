"""Persistent differential state contracts in both browser solver handoffs."""
import json
import re
import shutil
import subprocess

import pytest

from src.compiler.state_loop_deployment import emit_javascript_physics_worker


def test_gpu_commit_carries_declared_differential_shaft_outputs():
    import src.compiler.abstract_ui_vehicles as v
    equations, _ = v._symbolic_vehicle_equations_authored()
    contact, _ = v._symbolic_wheel_contact_equations_authored()
    inputs = tuple(sorted({str(s) for e in equations for s in e.rhs.free_symbols}))
    outputs = tuple(str(e.lhs) for e in equations)
    ci = tuple(sorted({str(s) for e in contact for s in e.rhs.free_symbols}))
    source = v._vehicle_gpu_graph_adapters(ci, v.CONTACT_PATCH_OUTPUTS, inputs, outputs)['commit']['source']
    assert set(v.VEHICLE_STATE_FEEDBACK) == {name for name in outputs if name.endswith('_next')}
    assert set(v.VEHICLE_STATE_FEEDBACK.values()) <= set(inputs)
    for result, state in v.VEHICLE_STATE_FEEDBACK.items():
        assert f'vehicle_feed[{inputs.index(state)}u] = finite_or(vehicle_outputs[{outputs.index(result)}u], vehicle_feed[{inputs.index(state)}u]);' in source


@pytest.mark.skipif(shutil.which('node') is None, reason='node unavailable')
def test_worker_handoff_feeds_next_shaft_state_and_preserves_active_states(tmp_path):
    source = emit_javascript_physics_worker()
    # Python's regex has no JavaScript [^] idiom.
    def function(name):
        return re.search(r'^function ' + name + r'\([\s\S]*?^\}', source, re.M).group(0)
    wasm = function('applyVehicleWasmOutput')
    gpu = function('applyResidentVehicleSnapshot')
    assert "retainVehicleCompiledState(body,output);" in gpu
    gpu_handoff = "retainVehicleCompiledState(body,output);"
    step = function('residentVehicleWasmStep')
    next_input = step.split('    input=', 1)[1].split(';\n  names.forEach', 1)[0]
    from src.compiler.abstract_ui_vehicles import VEHICLE_STATE_FEEDBACK
    data = {'feedback': VEHICLE_STATE_FEEDBACK, 'retain': function('retainVehicleCompiledState'), 'wasm': wasm, 'gpu_handoff': gpu_handoff, 'next_input': next_input,
            'initialize': function('initializeResidentVehicleState')}
    (tmp_path / 'source.json').write_text(json.dumps(data))
    script = tmp_path / 'check.cjs'
    script.write_text(r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert'),s=JSON.parse(fs.readFileSync(process.argv[2]));
const ctx={VEHICLE_STATE_FEEDBACK:s.feedback,captureVehicleCompiledEnergy:()=>{},residentVehicleControlValues:()=>[],SUBSTEP_DT:1/120};
vm.createContext(ctx);vm.runInContext(s.retain+'\n'+s.wasm+'\n'+s.initialize,ctx);
for(const lane of ['wasm','gpu']){
 const body={position:[0,0,0],velocity:[0,0,0],config:{},wheelOmegas:{},compressions:{},previousSlips:{},
  wasmState:{activeSentinel:29,differential_wrench_shaft_omega_front:0,differential_wrench_shaft_omega_rear:0},defaults:{}};
 const output=Object.fromEntries(Object.keys(s.feedback).map((name,i)=>[name,i+.125]));
 const contacts=Array.from({length:4},()=>({normal_x:0,normal_y:1,normal_z:0,mu_static:1}));
 const co=contacts.map(()=>({contact_area:1,chassis_force_x:0,chassis_force_y:1,chassis_force_z:0}));
 if(lane==='wasm')ctx.applyVehicleWasmOutput(body,output,contacts,co,[0,0,0],1/120);
 else ctx.retainVehicleCompiledState(body,output);
 assert.equal(body.wasmState.activeSentinel,29);
 for(const [result,state] of Object.entries(s.feedback))assert.equal(body.wasmState[state],output[result]);
 const prior=body.wasmState.clutch_health;ctx.retainVehicleCompiledState(body,{});assert.equal(body.wasmState.clutch_health,prior);
 const input=new Function('body','transmission','driveFractions','dt','wheelForce','wheelTorque','return '+s.next_input)(body,{}, {},1/120,[0,0,0],[0,0,0]);
 if(lane==='wasm')for(const [result,state] of Object.entries(s.feedback))assert.equal(input[state],output[result]);
 const names=['differential_wrench_shaft_omega_front','differential_wrench_shaft_omega_rear'];let uploaded;
 const gpu={residentGraph:true,integration:{inputs:names},device:{queue:{writeBuffer:(target,offset,values)=>{if(target==='feed')uploaded=values;}}},vehicleFeed:'feed'};
 ctx.initializeResidentVehicleState(gpu,body);
 names.forEach((name,i)=>assert.equal(uploaded[i],Math.fround(output[name+'_next'])));
}
''')
    subprocess.run(['node', str(script), str(tmp_path / 'source.json')], check=True, capture_output=True, text=True)
