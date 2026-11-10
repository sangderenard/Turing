import assert from 'node:assert/strict';
import {createStudioGameProtocol,validateStudioGameProtocol,validateStudioGameReceipt} from '../examples/meowdoku/web/marble_studio_game_protocol.js';
const protocol=createStudioGameProtocol('actual-original-session');
const receipt={epoch:protocol.epoch,state:'visible',identity:'scene:one',sourceRevision:7,sequence:3,sourceIdentity:'python:actual.Manager',renderFrame:4,
  building:{identity:'room',assetIdentity:'scene:one',placement:[1,0,0,0,0,1,0,0,0,0,1,0,32,1,-18,1]},
  environment:{buildingIdentity:'room',assetIdentity:'scene:one',revision:7,sequence:3,geometryRevision:20,colliderPublication:'posted-to-original-worker'},
  colliderPublication:'posted-to-original-worker',physics:'original-game-worker',camera:'original-game-controls'};
const request={protocol,assetIdentity:'scene:one',classIdentity:'python:actual.Manager',revision:7,sequence:3};
assert(Object.isFrozen(protocol.owners));assert.deepEqual(validateStudioGameProtocol(protocol),protocol);
const admitted=validateStudioGameReceipt(receipt,request);receipt.building.placement[12]=999;
assert.equal(admitted.building.placement[12],32);assert(Object.isFrozen(admitted.environment));
for(const mutate of [p=>p.epoch='',p=>p.schema='future',p=>p.owners.environment='game',p=>p.dtAuthority='invented-solver-dt',
  p=>p.capabilities.dynamicFixtureBodies=true,p=>p.capabilities.workerAppliedBarrier=true]){
  const value=structuredClone(protocol);mutate(value);assert.throws(()=>validateStudioGameProtocol(value));
}
for(const mutate of [r=>r.epoch='other-session',r=>r.sequence=4,r=>r.identity='other-scene',r=>r.state='queued',r=>r.sourceIdentity='python:other.Manager',r=>r.renderFrame=0,
  r=>r.building.assetIdentity='other',r=>r.building.placement[0]=NaN,r=>r.environment.revision=8,r=>r.environment.buildingIdentity='other',
  r=>r.environment.geometryRevision=-1,r=>r.environment.colliderPublication='worker-applied',r=>r.physics='other-solver']){
  const value=structuredClone(admitted);mutate(value);assert.throws(()=>validateStudioGameReceipt(value,request));
}
const pending=structuredClone(admitted);pending.colliderPublication=pending.environment.colliderPublication='native-scene-ready-worker-pending';
assert.equal(validateStudioGameReceipt(pending,request).colliderPublication,'native-scene-ready-worker-pending');
console.log(JSON.stringify({passed:true,rejections:19,scope:'offline protocol/receipt ownership; no worker-applied claim'}));
