// Trace the driver's reference plane to both EXISTING native rotation readers.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {projectCircuit,observeLine} from '../examples/meowdoku/web/arena_line_driver.js';
const host=fs.readFileSync('src/compiler/abstract_ui_div_map.py','utf8');
const worker=fs.readFileSync('src/compiler/state_loop_deployment.py','utf8');
const extract=(s,name,next)=>s.slice(s.indexOf(`function ${name}`),s.indexOf(`function ${next}`,s.indexOf(`function ${name}`)));
const c=vm.createContext({});
vm.runInContext(extract(host,'rotateVehiclePresentationVector','drawVehicleWheels'),c);
vm.runInContext(extract(worker,'rotateBodyVector','c2Unit').replaceAll('{{','{').replaceAll('}}','}'),c);
for(const yaw of [0,-Math.PI/2,Math.PI/2]){
 const state={roll:0,pitch:0,yaw};const a=Array.from(c.rotateVehiclePresentationVector([1,0,0],state));
 const b=Array.from(c.rotateBodyVector(state,[1,0,0]));
 assert.deepEqual(a,b);assert.ok(Math.abs(a[0]-Math.cos(yaw))<1e-14);assert.ok(Math.abs(a[2]-Math.sin(yaw))<1e-14);
}
const state={position:[8.31,.58,-15.06],velocity:[0,0,-1],roll:0,pitch:0,yaw:-Math.PI/2,yawVelocity:0};
const path=projectCircuit({id:'qa',color:'#00ff00',segments:[{support:'world-floor',from:[8.31,-15.06],to:[8.31,-18.06]}]},()=>0);
const basis=[[1,0,0],[0,1,0],[0,0,1]].map(v=>Array.from(c.rotateVehiclePresentationVector(v,state)));
const o=observeLine(path,state,basis);
assert.ok(Math.abs(o.localPoints[1][0]-1)<1e-12);assert.ok(Math.abs(o.localPoints[1][2])<1e-12);assert.equal(o.features[12],.1);
console.log('Native host/worker basis parity + forward QA sensor regression passed');
