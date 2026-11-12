import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const root=new URL('../',import.meta.url),page=readFileSync(process.argv[2],'utf8');
const exact=(name,kind='function')=>{
  const match=page.match(new RegExp(`^${kind} ${name}\\b[^]*?^}`, 'm'));
  assert.ok(match,`actual native ${name} is present`);return match[0];
};
const captured=JSON.parse(readFileSync(new URL('build/dot-studio-game-transfer-20261110/captured-studio-marble.json',root),'utf8'));
function setup(){
  const model=JSON.parse(page.match(/<script type="application\/json" id="abstract-ui-model">([^]*?)<\/script>/)[1]);
  const messages=[],calls=[],actorIdentity=model.viewer.control_policy.actor;
  const actor={entity:{pose:{position:[1,.2875,2]}},worldPosition:[1,.2875,2],velocity:[0,0,0],facing:[1,0,0]};
  const style={toggle(){},remove(){}},element={classList:style,focus(){calls.push('focus');},blur(){calls.push('blur');}};
  const context=vm.createContext({model,structuredClone,console,Float32Array,Set,Map,Math,
    performance:{now:()=>20},document:{pointerLockElement:null,exitPointerLock(){calls.push('unlock');this.pointerLockElement=null;}},
    viewportControls:{policy:model.viewer.control_policy,position:[1,.2875,2],horizontalVelocity:[3,2],
      highlighted:true,keys:new Set(['KeyW']),colliderSides:new Map(),yaw:1,pitch:.4},
    physicsRuntime:{verticalVelocity:4,parameters:new Map()},
    navigationRuntime:{generations:new Map(),routes:new Map([[actorIdentity,{}]]),waypointQueues:new Map()},
    stateLoopRuntime:{ready:true,actorRegistered:true,bodies:new Map(),worker:{postMessage(message){messages.push(message);}}},
    entityState:new Map([[actorIdentity,actor]]),
    controlFocus:{mode:'game',dialogue:null},latestControlInput:null,
    primaryActionState:{down:true},secondaryActionState:{down:true},
    endViewportPrimary(reason){calls.push('primary:'+reason);context.primaryActionState.down=false;},
    endViewportSecondary(reason){calls.push('secondary:'+reason);context.secondaryActionState.down=false;},
    clearNavigationRouteOverlay(){calls.push('cancel-route');},
    reservePhysicsSnapshotSlot(identity){const body={identity,position:[],generation:9,slot:1};context.stateLoopRuntime.bodies.set(identity,body);return body;},
    releasePhysicsSnapshotSlot(identity){calls.push('release-player');context.stateLoopRuntime.bodies.delete(identity);},
    setVehicleHorn:()=>Promise.resolve(),rebuildPortableSceneMesh(){calls.push('mesh');},
    respawnActiveVehicleAtAuthoredPose(){calls.push('original-vehicle-respawn');return true;},
    setPlacementStatus(value){calls.push(value);},
  });
  vm.runInContext(readFileSync(new URL('examples/meowdoku/web/mechanical_creature_studio_building.js',root),'utf8'),context);
  const definition=context.prepareMechanicalCreatureStudioBuilding(captured);
  context.appendMechanicalCreatureStudioBuilding(model,definition);
  vm.runInContext(exact('TuringWorldRegistry','class')+'\nglobalThis.turingWorld=new TuringWorldRegistry(model.world);',context);
  const nativeNames=['colorVector','boxYawDegrees','normalized3','cross3','buildExtrudedBoxMesh','navigationOpeningPoint',
    'cameraGroundHeight','sampleContactSurface','playerColliderVerticalOverlap','cancelEntityNavigation',
    'seedViewportControlPose','setViewportControlHighlight','requestViewportControls','clearActiveVehicle',
    'cloneVehicleState','synchronizePortalBody','registerPlayerPhysicsBody','respawnViewportActor','mechanicalCreatureStudioFreeze'];
  if(page.includes('function closedSolidVerticalSupport('))nativeNames.unshift('closedSolidShape','closedSolidLocalVector',
    'closedSolidLocalPoint','closedSolidWorldVector','closedSolidRay','closedSolidVerticalSupport');
  vm.runInContext(nativeNames.map(name=>exact(name)).join('\n'),context);
  const geometry=model.document_geometry.boxes;
  context.shaderViewer={geometry,revision:1,element,mapElement:null,canvas:{requestPointerLock(){calls.push('lock-request');}},active:true};
  const apron=geometry.find(box=>box.label==='Outer yard north flat apron');
  const realization=context.buildExtrudedBoxMesh([...definition.boxes,apron],model.appearance.colors);
  context.shaderViewer.colliders=realization.colliders;
  const machine={identity:'original-vehicle',position:[10,1,-2],velocity:[1,2,3],fuel:73,battery:.8};
  context.vehicleRuntime={active:{identity:'springtail',configuration:{chassis:{camera_height:.4,half_width:.3}}},
    state:machine,wheelBoxes:[],camera:{old:true},cameraChassisYaw:1};
  model.vehicle_slot.active='springtail';
  context.mechanicalCreatureStudioBridgeState={applied:{buildingIdentity:definition.identity,assetIdentity:definition.assetIdentity,revision:1,sequence:1}};
  context.readMechanicalCreatureStudioRegisteredDefinition=()=>definition;
  const helper=readFileSync(new URL('examples/meowdoku/web/mechanical_creature_studio_entry.js',root),'utf8');
  assert.ok(page.includes(helper),'the candidate contains exactly the tested additive helper');
  vm.runInContext(helper,context);
  const receipt={state:'visible',identity:definition.assetIdentity,sourceRevision:1,sequence:1,building:{identity:definition.identity}};
  return {context,definition,actor,actorIdentity,messages,calls,machine,receipt,entry:context.mechanicalCreatureStudioEntry};
}
for(const mutate of [
  s=>{s.receipt.sequence=2;},
  s=>{s.definition.boxes[0].openings[0].width=.02;},
  s=>{s.context.model.viewer.camera.eye_height=Infinity;},
  s=>{s.definition.boxes[0].center[0]=999;},
  s=>{s.context.shaderViewer.colliders.push({identity:'blocking-object',minimum:[0,0,-99],maximum:[100,5,99]});},
]){
  const s=setup(),before=JSON.stringify(s.entry.read());mutate(s);
  assert.throws(()=>s.entry.enter(s.receipt));
  assert.equal(JSON.stringify(s.entry.read()),before,'failed preflight cannot dismount, release controls or move the original player');
  assert.equal(s.messages.length,0);assert.equal(s.calls.length,0);assert.equal(s.context.vehicleRuntime.state,s.machine);
}
const s=setup(),spawnBefore=JSON.stringify(s.actor.entity.pose),entry=s.entry.enter(s.receipt);
assert.equal(entry.location,'outside-studio-entrance');assert.equal(entry.buildingIdentity,s.definition.identity);
assert.ok(entry.positionM[2]>s.definition.entrance.worldCenter[2]);assert.equal(entry.positionM[0],s.definition.entrance.worldCenter[0]);
assert.equal(entry.positionM[1],s.context.model.viewer.camera.eye_height);assert.equal(entry.yaw,-Math.PI/2);
assert.ok(Object.isFrozen(entry));assert.ok(Object.isFrozen(entry.positionM));assert.equal(JSON.stringify(s.actor.entity.pose),spawnBefore);
assert.equal(s.context.vehicleRuntime.active,null);assert.equal(s.context.vehicleRuntime.state,null);
assert.equal(s.context.vehicleRuntime.parkedState.fuel,73);assert.equal(s.context.vehicleRuntime.parkedState.battery,.8);
assert.equal(s.context.vehicleRuntime.camera,null);assert.ok(s.messages.some(m=>m.type==='vehicle-control'&&m.throttle===0));
assert.equal(s.messages.filter(m=>m.type==='remove').length,1);assert.equal(s.messages.find(m=>m.type==='remove').identity,s.actorIdentity);
const upsert=s.messages.filter(m=>m.type==='upsert').at(-1);assert.deepEqual([...upsert.body.position],[...entry.positionM]);
assert.deepEqual([...upsert.body.velocity],[0,0,0]);assert.equal(s.context.physicsRuntime.verticalVelocity,0);
assert.deepEqual([...s.context.viewportControls.horizontalVelocity],[0,0]);assert.equal(s.context.navigationRuntime.routes.size,0);
s.entry.setActive(true);assert.equal(s.context.viewportControls.highlighted,true);assert.ok(s.calls.includes('focus'));
s.context.document.pointerLockElement=s.context.shaderViewer.canvas;s.context.viewportControls.keys.add('KeyW');
const moved=[entry.positionM[0]+1,entry.positionM[1],entry.positionM[2]-.5];s.context.viewportControls.position=moved;
const count=s.messages.length;s.entry.setActive(false);
assert.equal(s.context.viewportControls.highlighted,false);assert.equal(s.context.viewportControls.keys.size,0);
assert.equal(s.context.document.pointerLockElement,null);assert.ok(s.calls.includes('primary:studio-view'));assert.ok(s.calls.includes('secondary:studio-view'));
s.entry.setActive(true);assert.equal(s.entry.enter({...s.receipt,sequence:2}),entry);
assert.deepEqual([...s.entry.read().positionM],moved);assert.equal(s.messages.length,count,'return/reopen and later revisions do not reset any body');
s.definition.boxes[0].center[0]+=2;
s.context.respawnViewportActor();
assert.equal(s.entry.read().positionM[0],entry.positionM[0]+2,'R resolves the current imported doorway rather than old authored start');
assert.equal(JSON.stringify(s.actor.entity.pose),spawnBefore);
const beforeInvalid=s.messages.length,positionBeforeInvalid=JSON.stringify(s.entry.read().positionM);
s.definition.boxes[0].openings[0].width=.01;
assert.throws(()=>s.context.respawnViewportActor(),/capsule/);
assert.equal(s.messages.length,beforeInvalid);assert.equal(JSON.stringify(s.entry.read().positionM),positionBeforeInvalid);
s.context.vehicleRuntime.active={identity:'springtail'};s.context.respawnViewportActor();
assert.equal(s.calls.at(-1),'original-vehicle-respawn','mounted R keeps its original branch even when Studio entry is obstructed');
if(page.includes('const vehicleQualification = {')){
  {
    const q=setup(),c=q.context;
    c.vehicleQualification={active:true,complete:false};
    const sleeping={sleeping:true,reason:'integration-only-quiescence'};
    c.wakeMechanicalCreatureStudioQualification(sleeping);
    assert.equal(q.messages.length,1);
    assert.deepEqual(JSON.parse(JSON.stringify(q.messages[0])),
      {type:'vehicle-control',identity:'springtail',throttle:0,steering:0,brake:0});
    for(const message of [{sleeping:false,reason:sleeping.reason},{sleeping:true,reason:'stopped'}])
      c.wakeMechanicalCreatureStudioQualification(message);
    c.vehicleQualification.complete=true;c.wakeMechanicalCreatureStudioQualification(sleeping);
    c.vehicleQualification.complete=false;c.vehicleQualification.active=false;c.wakeMechanicalCreatureStudioQualification(sleeping);
    c.vehicleQualification.active=true;c.vehicleRuntime.active=null;c.wakeMechanicalCreatureStudioQualification(sleeping);
    delete c.vehicleQualification;c.wakeMechanicalCreatureStudioQualification(sleeping);
    assert.equal(q.messages.length,1,'qualification wake cannot affect a completed gate, unrelated sleep or parked vehicle');
  }
  for(const failed of [false,true]){
    const q=setup(),c=q.context,scheduled=[];
    c.vehicleQualification={active:true,complete:false,passed:false,animationStarted:false,simulatedSeconds:0,
      failures:new Set(failed?['measured native fault']:[]),maximumPenetrationM:0,longestContactDropoutTicks:0,
      maximumEnergyCreationJ:0,oscillations:0,maximumTiltRad:0,epsilon:.01};
    c.document.body={classList:{remove(){q.calls.push('remove-gate-class');}}};
    c.document.getElementById=()=>({remove(){q.calls.push('remove-gate');},classList:{add(){}},querySelector:()=>({})});
    c.qualificationStatus=value=>q.calls.push(value);c.setTimeout=(fn,ms)=>scheduled.push({fn,ms});
    c.requestAnimationFrame=fn=>q.calls.push('native-frame-scheduled');c.runEntityCycle=()=>{};
    vm.runInContext(exact('launchQualifiedLivingMap')+'\n'+exact('finishVehicleQualification'),c);
    const originalMachine=c.vehicleRuntime.state,wait=q.entry.whenReady();let settled=false;
    const observation=wait.then(()=>{settled=true;},error=>{settled=true;assert.match(error.message,/qualification failed/);});
    q.entry.setActive(false);assert.equal(q.calls.length,0);assert.equal(q.messages.length,0);
    assert.throws(()=>q.entry.enter(q.receipt),/qualification must pass/);
    assert.equal(c.vehicleRuntime.state,originalMachine);assert.equal(c.vehicleRuntime.active.identity,'springtail');
    c.finishVehicleQualification({});await Promise.resolve();
    if(failed){assert.equal(settled,true);assert.equal(scheduled.length,0);assert.equal(q.entry.readQualification().state,'failed');
      assert.equal(c.vehicleRuntime.state,originalMachine);assert.throws(()=>q.entry.enter(q.receipt),/qualification must pass/);
    }else{assert.equal(settled,false,'native successful qualification still waits for its 700ms launch');
      assert.equal(scheduled.length,1);assert.equal(scheduled[0].ms,700);scheduled[0].fn();await observation;
      assert.equal(q.entry.readQualification().state,'ready');q.entry.enter(q.receipt);assert.equal(c.vehicleRuntime.active,null);
    }
    await observation;
  }
}
console.log(JSON.stringify({pass:true,positionM:entry.positionM,yaw:entry.yaw,parkedOriginalVehicle:true,
  nativeRespawn:true,invalidDestinations:5,returnReopenPreservesPlayer:true,originalAuthoredSpawnUnchanged:true}));
