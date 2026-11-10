// Actual page spawn and matched worker/Wasm, with delayed snapshot recycling.
// The controlled clock changes delivery cadence, not any simulation law.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const page=readFileSync(process.argv[2],'utf8');
const model=JSON.parse(page.match(/<script type="application\/json" id="abstract-ui-model">([^]*?)<\/script>/)[1]);
const exact=name=>{
  const match=page.match(new RegExp('^function '+name+'\\([^]*?^\\}','m'));
  assert.ok(match,`native ${name}`);return match[0];
};
const names=['sampleDeclaredSurface','cameraGroundHeight','rotateVehiclePresentationVector',
  'vehicleTerrainRestPose','vehicleSpawnState','setActiveVehicle','registerActiveVehiclePhysicsBody'];
if(page.includes('function closedSolidVerticalSupport('))names.unshift('closedSolidShape','closedSolidLocalVector',
  'closedSolidLocalPoint','closedSolidWorldVector','closedSolidRay','closedSolidVerticalSupport');
const messages=[],vehicle=model.vehicle_slot.vehicles[0],noop=()=>{};
const colliders=model.document_geometry.boxes.filter(b=>b.surface).map((b,i)=>({
  identity:b.identity+'/surface:sampled-terrain',objectIdentity:b.identity,role:'contact-surface',runtimePartId:i+1,
  minimum:[b.surface.domain.minimum_x,-8,b.surface.domain.minimum_z],
  maximum:[b.surface.domain.maximum_x,b.height,b.surface.domain.maximum_z],surface:b.surface}));
const client=vm.createContext({model,console,structuredClone,
  vehicleRuntime:{active:null,parkedState:null},shaderViewer:{colliders,shaderOnly:false},
  viewportControls:{position:[...vehicle.pose.position],horizontalVelocity:[0,0],yaw:Number(vehicle.pose.yaw||0),policy:{}},
  stateLoopRuntime:{ready:false,bodies:new Map(),worker:{postMessage:m=>messages.push(structuredClone(m))}},
  reservePhysicsSnapshotSlot:()=>({slot:0,generation:1}),ensureVehiclePresentation:noop,ensureVehicleWorldMarker:()=>null,
  cancelEntityNavigation:noop,setPlacementStatus:noop,releasePhysicsSnapshotSlot:noop,removeVehicleInventoryItem:noop});
vm.runInContext(names.map(exact).join('\n'),client);
vm.runInContext('setActiveVehicle(model.vehicle_slot.vehicles[0].identity,{placeAtActor:true});'+
  'vehicleRuntime.electrical.ignitionOn=false;vehicleRuntime.electrical.starterEngaged=false;'+
  'stateLoopRuntime.ready=true;registerActiveVehiclePhysicsBody();',client);
const upsert=messages.find(m=>m.type==='upsert');assert.ok(upsert);
upsert.body.wasmState={...(upsert.body.wasmState||{}),engine_angular_speed:0};
const plugin=model.world.plugins.find(p=>p.identity===model.physics_program.plugin);
const descriptor=p=>({wasm:Buffer.from(model.world.wasm_modules.find(m=>m.content_key===p.module).binary_base64,'base64'),
  abi:p.abi,entrypoint:p.entrypoint});
const initialization={type:'init',...descriptor(plugin),parameters:Object.fromEntries(model.physics_program.parameters.map(p=>[p.name,p.default])),snapshotCapacity:128,
  vehicleWasm:descriptor(model.world.plugins.find(p=>p.capability==='vehicle-physics')),
  contactWasm:descriptor(model.world.plugins.find(p=>p.capability==='vehicle-contact-fallback')),
  worldBottom:model.contact_surfaces.world_bottom};
const workerSource=model.loop_deployment.workers.find(w=>w.target==='javascript-worker').source;
const helper=readFileSync(new URL('../examples/meowdoku/web/mechanical_creature_studio_entry.js',import.meta.url),'utf8');
const wake=helper.match(/^function wakeMechanicalCreatureStudioQualification\([^]*?^}/m)[0];
async function run(recycleEvery,repair){
  let tick=null,now=0,pending=null,lastSequence=0,lastWake=0,wakes=0,ticks=0;
  const events=[],errors=[],outbox=[];
  const qualification=vm.createContext({vehicleQualification:{active:true,complete:false},vehicleRuntime:{active:vehicle},
    stateLoopRuntime:{worker:{postMessage:m=>outbox.push(m)}}});
  vm.runInContext(wake,qualification);
  const worker=vm.createContext({WebAssembly,console,performance:{now:()=>now},
    setInterval:fn=>{tick=fn;return 1;},clearInterval:()=>{tick=null;},close:noop,
    postMessage:m=>{
      if(m.type==='snapshot-buffer')pending=m;
      if(m.type==='engine-state'){
        events.push({tick:ticks,stage:m.stage,reason:m.reason});
        if(repair&&m.sleeping){qualification.wakeMechanicalCreatureStudioQualification(m);wakes++;}
      }
      if(m.type==='vehicle-gpu-error')errors.push(m.error);
    }});
  vm.runInContext(workerSource,worker);
  await worker.onmessage({data:initialization});
  await worker.onmessage({data:{type:'colliders',colliders}});
  for(const message of messages)await worker.onmessage({data:message.type==='upsert'?upsert:message});
  for(ticks=1;ticks<=500&&tick;ticks++){
    now=ticks*1000/120;await tick();
    for(const message of outbox.splice(0))await worker.onmessage({data:message});
    if(pending&&ticks%recycleEvery===0){
      const delta=Math.max(1,pending.sequence-lastSequence);lastSequence=pending.sequence;
      // This is the original qualifier's neutral wake cadence, not a new clock.
      if(lastSequence%30<delta){lastWake=lastSequence;
        await worker.onmessage({data:{type:'vehicle-control',identity:vehicle.identity,throttle:0,steering:0,brake:0}});}
      await worker.onmessage({data:{type:'recycle',buffer:pending.buffer}});pending=null;
    }
  }
  assert.deepEqual(errors,[]);return {ticks:ticks-1,lastSequence,lastWake,wakes,events};
}
const fast=await run(1,false),stalled=await run(8,false),repaired=await run(8,true);
assert.equal(fast.ticks,500);assert.equal(fast.lastSequence,500);
assert.ok(stalled.ticks<500);assert.ok(stalled.lastSequence<30);
assert.equal(stalled.events.at(-1).reason,'integration-only-quiescence');
assert.equal(repaired.ticks,500);assert.ok(repaired.lastSequence>stalled.lastSequence);
assert.ok(repaired.wakes>0);assert.ok(repaired.events.some(e=>e.reason==='vehicle-control'));
console.log(JSON.stringify({pass:true,fast,stalled,repaired}));
