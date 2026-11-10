import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import vm from 'node:vm';
import {createStudioGameProtocol,validateStudioGameReceipt} from '../examples/meowdoku/web/marble_studio_game_protocol.js';

// Execute the actual host. Only its two asynchronous module dependencies are
// replaced so each installer boundary can be settled explicitly, without a GL
// context, browser, timer, worker or change to the production host API.
const path=new URL('../examples/meowdoku/web/mechanical_creature_studio_layer_host.js',import.meta.url);
let source=readFileSync(path,'utf8');
for(const declaration of [
  "import {installStudioCollection} from './mechanical_creature_studio_collection.js';",
  "import {installStudioRoom} from './mechanical_creature_studio_room.js';",
  "import {createStudioGameProtocol,validateStudioGameReceipt} from './marble_studio_game_protocol.js';",
  'export function createMechanicalCreatureStudioLayerHost',
]){
  assert.equal(source.split(declaration).length,2,'The real host module seam must be unique.');
  source=source.replace(declaration,declaration.startsWith('export ')?declaration.slice(7):'');
}
const deferred=()=>{let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});
  return {promise,resolve,reject};};
const node=()=>({style:{},dataset:{},children:[],textContent:'',removed:false,
  append(...children){this.children.push(...children);},replaceChildren(...children){this.children=children;},
  addEventListener(){},setAttribute(){},remove(){this.removed=true;}});
const input=(sequence=1)=>({asset:{identity:'scene:'+sequence,revision:sequence},
  source:{revision:sequence,classIdentity:'python:Studio'},marbleAsset:{identity:'chosen-marble'},sequence});
const visible=(sequence=1,publication='posted-to-original-worker')=>({state:'visible',identity:'scene:'+sequence,
  sequence,sourceRevision:sequence,sourceIdentity:'python:Studio',renderFrame:sequence,
  environment:{buildingIdentity:'owned',assetIdentity:'scene:'+sequence,revision:sequence,sequence,
    geometryRevision:sequence,colliderPublication:publication}});
const placement=[1,0,0,0,0,1,0,0,0,0,1,0,32,1.5,-18,1];
const bridgeSource=readFileSync(new URL('../examples/meowdoku/web/mechanical_creature_studio_bridge.js',import.meta.url),'utf8');

function setup({collectionInstaller,roomInstaller,actualBridge=false,qualification=null}={}){
  const calls={collection:0,room:0,queue:0,sinks:[],entries:[],active:[],roomOptions:null,collectionOptions:null,queued:null},body=node();
  const doc={body,createElement:node,createTextNode:text=>({textContent:text})};
  const collection={disposed:0,readStatus:()=>({kind:'collection'}),dispose(){this.disposed++;}};
  const ready=deferred(),room={disposed:0,ready:ready.promise,readStatus:()=>({kind:'room'}),
    update:async value=>visible(value.sequence),dispose(){this.disposed++;}};
  // The actual room also observes its own initial promise before returning its
  // handle; cancellation during installation must not create an unhandled rejection.
  ready.promise.catch(()=>{});
  const host={mechanicalCreatureStudioEntry:{
    whenReady:()=>qualification,readQualification:()=>qualification?{state:'pending',simulatedSeconds:0}:null,
    enter(receipt){calls.entries.push(receipt);return {buildingIdentity:receipt.building.identity,positionM:[32,.3315,-11.7]};},
    setActive(value){calls.active.push(value);},read:()=>({positionM:[32,.3315,-11.7],active:true}),
  },mechanicalCreatureAmmo:{items:()=>[],selected:()=>null},mechanicalCreatureStudioBridge:{
    prepareEnvironment:(_,{assetIdentity})=>({definition:{identity:'owned'},building:{identity:'owned',assetIdentity,placement:placement.slice()}}),
    queueEnvironment(value){calls.queue++;calls.queued=value;return Promise.resolve();},
    setSink(value){calls.sinks.push(value);},readSnapshot:()=>null,
  }};
  const context=vm.createContext({Error,TypeError,structuredClone,createStudioGameProtocol,validateStudioGameReceipt,
    crypto:{randomUUID:()=> 'original-epoch'},
    stateLoopRuntime:{ready:true,latestSequence:7,bodies:new Map(),slotRecords:[]},
    shaderViewer:{revision:1,cameraPosition:[1,2,3],cameraFacing:[0,0,-1]},
    projectileState:new Map(),projectilePickupState:new Map(),reportRuntimeFault:(_kind,error)=>{throw error;},
    installStudioCollection:options=>{calls.collection++;calls.collectionOptions=options;
      return collectionInstaller?.(options,collection)??Promise.resolve(collection);},
    installStudioRoom:options=>{calls.room++;calls.roomOptions=options;
      return roomInstaller?.(options,room)??Promise.resolve(room);},
  });
  if(actualBridge){vm.runInContext(bridgeSource,context);host.mechanicalCreatureStudioBridge=context.mechanicalCreatureStudioBridge;}
  vm.runInContext(source,context,{filename:path.pathname});
  const layer=context.createMechanicalCreatureStudioLayerHost({host,doc});
  return {layer,calls,collection,room,ready,panel:body.children[0],context};
}
const assertNoReceipt=layer=>{assert.equal(layer.read().lastReceipt,null);assert.equal(layer.read().lastSequence,0);};
const assertClosed=layer=>{assert.equal(layer.read().closed,true);assert.equal(layer.read().busy,false);};
const closedError=error=>/closed/i.test(error.message);

test('open host admits both installers and preserves normal receipt/reuse',async()=>{
  const s=setup();s.ready.resolve(visible());const first=await s.layer.update(input());
  assert.equal(first.state,'visible');assert.equal(first.sequence,1);
  assert.equal(s.layer.read().collection.kind,'collection');assert.equal(s.layer.read().room.kind,'room');
  assert.equal((await s.layer.update(input(2))).sequence,2);
  assert.equal(s.calls.collection,1);assert.equal(s.calls.room,1);
  assert.equal(s.collection.disposed,0);assert.equal(s.room.disposed,0);
  s.layer.dispose();assert.equal(s.collection.disposed,1);assert.equal(s.room.disposed,1);
});

test('direct player entry follows the first validated receipt only; view changes preserve it',async()=>{
  const s=setup();assert.equal(s.calls.entries.length,0);
  s.ready.resolve(visible());const first=await s.layer.update(input());
  assert.equal(s.calls.entries.length,1);assert.equal(s.calls.entries[0],first);
  s.layer.setActive(false);s.layer.setActive(true);
  assert.deepEqual(s.calls.active.slice(-2),[false,true]);
  await s.layer.update(input(2));assert.equal(s.calls.entries.length,1);
  assert.equal(s.layer.read().entry.buildingIdentity,'owned');
  assert.ok(s.panel.children.some(child=>child.id==='studio-layer-player-status'&&child.textContent.includes('32.000')));
  s.layer.dispose();assert.equal(s.calls.active.at(-1),false);
});

test('pending original qualification holds every installer and entry until native success',async()=>{
  const gate=deferred(),s=setup({qualification:gate.promise});s.ready.resolve(visible());
  const pending=s.layer.update(input());
  assert.equal(s.calls.collection,0);assert.equal(s.calls.room,0);assert.equal(s.calls.entries.length,0);
  gate.resolve();await pending;assert.equal(s.calls.collection,1);assert.equal(s.calls.entries.length,1);s.layer.dispose();
});

test('original qualification failure or disposal cannot admit Studio or dismount its vehicle',async()=>{
  const gate=deferred(),s=setup({qualification:gate.promise}),failure=Error('qualification failed');
  const pending=s.layer.update(input()),rejected=assert.rejects(pending,error=>error===failure);
  gate.reject(failure);await rejected;
  assert.equal(s.calls.collection,0);assert.equal(s.calls.room,0);assert.equal(s.calls.entries.length,0);assertNoReceipt(s.layer);
  const later=deferred(),closed=setup({qualification:later.promise});
  const update=closed.layer.update(input()),closedResult=assert.rejects(update,closedError);
  closed.layer.dispose();later.resolve();await closedResult;
  assert.equal(closed.calls.collection,0);assert.equal(closed.calls.entries.length,0);assertNoReceipt(closed.layer);
});

test('collection resolving after disposal is closed without assignment or room install',async()=>{
  const gate=deferred(),s=setup({collectionInstaller:()=>gate.promise});
  s.ready.resolve(visible());
  const pending=s.layer.update(input()),rejected=assert.rejects(pending,closedError);
  s.layer.dispose();gate.resolve(s.collection);await rejected;
  assert.equal(s.collection.disposed,1);assert.equal(s.layer.read().collection,undefined);
  assert.equal(s.calls.room,0);assert.equal(s.calls.queue,0);assertNoReceipt(s.layer);assertClosed(s.layer);
  s.layer.dispose();assert.equal(s.collection.disposed,1,'Late candidate was never retained.');
});

test('collection installer rejection after disposal retains the original thrown object',async()=>{
  const gate=deferred(),s=setup({collectionInstaller:()=>gate.promise}),failure=Error('collection verification failed');
  const pending=s.layer.update(input()),rejected=assert.rejects(pending,error=>error===failure);
  s.layer.dispose();gate.reject(failure);await rejected;
  assert.equal(s.calls.room,0);assertNoReceipt(s.layer);assertClosed(s.layer);
});

test('room resolving after disposal is closed without assignment or receipt',async()=>{
  const gate=deferred(),entered=deferred(),s=setup({roomInstaller:()=>{entered.resolve();return gate.promise;}});
  s.ready.resolve(visible());
  const pending=s.layer.update(input()),rejected=assert.rejects(pending,closedError);
  await entered.promise;s.layer.dispose();gate.resolve(s.room);await rejected;
  assert.equal(s.collection.disposed,1);assert.equal(s.room.disposed,1);
  assert.equal(s.layer.read().room,undefined);assert.equal(s.calls.queue,0);
  assertNoReceipt(s.layer);assertClosed(s.layer);s.layer.dispose();assert.equal(s.room.disposed,1);
});

test('room installer rejection after disposal retains the original thrown object',async()=>{
  const gate=deferred(),entered=deferred(),s=setup({roomInstaller:()=>{entered.resolve();return gate.promise;}}),failure=Error('room verification failed');
  const pending=s.layer.update(input()),rejected=assert.rejects(pending,error=>error===failure);
  await entered.promise;s.layer.dispose();gate.reject(failure);await rejected;
  assert.equal(s.layer.read().room,undefined);assertNoReceipt(s.layer);assertClosed(s.layer);
});

test('dispose while admitted room.ready is pending preserves its original rejection',async()=>{
  const awaited=deferred(),s=setup(),failure=Error('room closed before visible');
  Object.defineProperty(s.room,'ready',{get(){awaited.resolve();return s.ready.promise;}});
  s.room.dispose=function(){this.disposed++;s.ready.reject(failure);};
  const pending=s.layer.update(input()),rejected=assert.rejects(pending,error=>error===failure);
  await awaited.promise;s.layer.dispose();await rejected;
  assert.equal(s.room.disposed,1);assert.equal(s.collection.disposed,1);assertNoReceipt(s.layer);assertClosed(s.layer);
});

test('late successful room.ready cannot publish a receipt after disposal',async()=>{
  const awaited=deferred(),s=setup();
  Object.defineProperty(s.room,'ready',{get(){awaited.resolve();return s.ready.promise;}});
  const pending=s.layer.update(input()),rejected=assert.rejects(pending,closedError);
  await awaited.promise;s.layer.dispose();s.ready.resolve(visible());await rejected;
  assertNoReceipt(s.layer);assertClosed(s.layer);assert.equal(s.room.disposed,1);
});

test('late existing-room update cannot overwrite its last admitted receipt',async()=>{
  const s=setup();s.ready.resolve(visible());const first=await s.layer.update(input());
  const gate=deferred();s.room.update=()=>gate.promise;
  const pending=s.layer.update(input(2)),rejected=assert.rejects(pending,closedError);
  s.layer.dispose();gate.resolve(visible(2));await rejected;
  assert.equal(s.layer.read().lastReceipt,first);assert.equal(s.layer.read().lastSequence,1);assertClosed(s.layer);
  await assert.rejects(s.layer.update(input(3)),closedError);
  assert.equal(s.calls.collection,1);assert.equal(s.calls.room,1);
});

test('existing-room update rejection after disposal retains its original outcome',async()=>{
  const s=setup();s.ready.resolve(visible());const first=await s.layer.update(input());
  const gate=deferred(),failure=Error('visible pass failed');s.room.update=()=>gate.promise;
  const pending=s.layer.update(input(2)),rejected=assert.rejects(pending,error=>error===failure);
  s.layer.dispose();gate.reject(failure);await rejected;
  assert.equal(s.layer.read().lastReceipt,first);assert.equal(s.layer.read().lastSequence,1);assertClosed(s.layer);
});

test('host protocol is stable, deeply frozen and matches the actual bridge epoch',()=>{
  const s=setup({actualBridge:true}),protocol=s.layer.readProtocol();
  assert.equal(s.layer.readProtocol(),protocol);assert.ok(Object.isFrozen(protocol));
  assert.ok(Object.isFrozen(protocol.owners));assert.ok(Object.isFrozen(protocol.capabilities));
  s.layer.connect(()=>{});s.context.publishMechanicalCreatureStudioSnapshot(10,.016);
  const packet=s.layer.readSnapshot();assert.equal(packet.epoch,protocol.epoch);
  assert.equal(packet.schema,protocol.snapshotSchema);assert.equal(packet.dtAuthority,protocol.dtAuthority);
  s.layer.dispose();assert.equal(s.layer.readProtocol(),protocol);assert.equal(s.layer.readSnapshot(),null);
});

test('replacement connection survives stale disconnect and duplicate disconnect with real bridge',()=>{
  const s=setup({actualBridge:true}),first=[],second=[],third=[];
  const off1=s.layer.connect(value=>first.push(value));s.context.publishMechanicalCreatureStudioSnapshot(10,.016);
  const off2=s.layer.connect(value=>second.push(value));off1();off1();
  s.context.publishMechanicalCreatureStudioSnapshot(20,.016);
  assert.equal(first.length,1);assert.equal(second.length,1);assert.equal(s.layer.readSnapshot(),second[0]);
  off2();off2();assert.equal(s.layer.readSnapshot(),null);
  s.context.publishMechanicalCreatureStudioSnapshot(30,.016);assert.equal(second.length,1);
  const off3=s.layer.connect(value=>third.push(value));off2();
  s.context.publishMechanicalCreatureStudioSnapshot(40,.016);assert.equal(third.length,1);
  assert.equal(third[0].epoch,first[0].epoch,'Reconnect retains the original host session, not a new world.');
  s.layer.dispose();off3();s.context.publishMechanicalCreatureStudioSnapshot(50,.016);assert.equal(third.length,1);
  assert.throws(()=>s.layer.connect(()=>{}),closedError);assert.equal(s.layer.readSnapshot(),null);
});

test('nonfunction connect cannot replace an active sink; callbacks retained by old owner stay inert',()=>{
  const s=setup(),first=[],second=[];s.layer.connect(value=>first.push(value));
  const old=s.calls.sinks.at(-1);assert.throws(()=>s.layer.connect(null),/function/);
  old('one');s.layer.connect(value=>second.push(value));const latest=s.calls.sinks.at(-1);
  old('stale');latest('two');assert.deepEqual(first,['one']);assert.deepEqual(second,['two']);
  s.layer.dispose();latest('closed');assert.deepEqual(second,['two']);
});

test('dispose is idempotent after admission and cannot revive a connection',async()=>{
  const s=setup();s.ready.resolve(visible());await s.layer.update(input());const off=s.layer.connect(()=>{});
  s.layer.dispose();s.layer.dispose();off();off();
  assert.equal(s.room.disposed,1);assert.equal(s.collection.disposed,1);
  assert.equal(s.calls.sinks.filter(value=>value===null).length,1);
  await assert.rejects(s.layer.update(input(2)),closedError);assert.throws(()=>s.layer.connect(()=>{}),closedError);
});

test('pending worker publication is preserved in an immutable detached visible receipt',async()=>{
  const s=setup(),ack=visible(1,'native-scene-ready-worker-pending');s.ready.resolve(ack);
  const receipt=await s.layer.update(input());
  assert.equal(receipt.colliderPublication,'native-scene-ready-worker-pending');
  assert.equal(receipt.environment.colliderPublication,receipt.colliderPublication);
  assert.equal(receipt.epoch,s.layer.readProtocol().epoch);assert.ok(Object.isFrozen(receipt));
  assert.ok(Object.isFrozen(receipt.environment));assert.ok(Object.isFrozen(receipt.building.placement));
  ack.environment.colliderPublication='posted-to-original-worker';
  assert.equal(receipt.environment.colliderPublication,'native-scene-ready-worker-pending');s.layer.dispose();
});

test('mismatched visible receipt cannot advance host sequence or replace prior receipt',async()=>{
  const s=setup();s.ready.resolve(visible());const prior=await s.layer.update(input());
  s.room.update=async()=>visible(3);await assert.rejects(s.layer.update(input(2)),/acknowledgment/);
  assert.equal(s.layer.read().lastSequence,1);assert.equal(s.layer.read().lastReceipt,prior);
  s.room.update=async()=>({...visible(2),environment:{...visible(2).environment,colliderPublication:'worker-applied'}});
  await assert.rejects(s.layer.update(input(2)),/publication/);assert.equal(s.layer.read().lastReceipt,prior);s.layer.dispose();
});

test('queued environment retains its input identity and cancels when host closes',async()=>{
  const awaited=deferred(),s=setup(),request=input();
  Object.defineProperty(s.room,'ready',{get(){awaited.resolve();return s.ready.promise;}});
  const pending=s.layer.update(request),rejected=assert.rejects(pending,closedError);
  await awaited.promise;request.asset.identity='mutated';request.source.revision=90;
  const {beforeCommit}=s.calls.roomOptions;await beforeCommit({isCurrent:()=>true,commit:()=>true});
  assert.equal(s.calls.queued.assetIdentity,'scene:1');assert.equal(s.calls.queued.revision,1);
  assert.equal(s.calls.queued.isCurrent(),true);s.layer.dispose();assert.equal(s.calls.queued.isCurrent(),false);
  assert.throws(()=>beforeCommit({isCurrent:()=>true,commit:()=>true}),closedError);
  assert.equal(s.calls.queue,1);s.ready.resolve(visible());await rejected;
});

test('full environment can start without sphere ammunition and updates never refill collection',async()=>{
  const s=setup();s.ready.resolve(visible());await s.layer.update({...input(),marbleAsset:null});
  assert.equal(s.calls.collectionOptions.assets.length,0);assert.equal(s.calls.room,1);
  await s.layer.update(input(2));assert.equal(s.calls.collection,1,'Later environment updates cannot mint inventory.');
  s.layer.dispose();
});

test('explicit malformed ammunition is not silently treated as ineligible',async()=>{
  const failure=Error('malformed asset'),s=setup({collectionInstaller:options=>{
    assert.equal(options.assets[0].asset,false);throw failure;
  }});
  await assert.rejects(s.layer.update({...input(),marbleAsset:false}),error=>error===failure);
  assert.equal(s.calls.room,0);assertNoReceipt(s.layer);s.layer.dispose();
});

test('invalid source revisions fail before installer work',async()=>{
  const s=setup();
  for(const revision of [undefined,-1,NaN,1.5])await assert.rejects(s.layer.update({
    ...input(),asset:{identity:'scene:1',revision},source:{revision}}),/matching/);
  assert.equal(s.calls.collection,0);assert.equal(s.layer.read().busy,false);assertNoReceipt(s.layer);s.layer.dispose();
});


test('preview pose requires current session and stops with host',async()=>{
  const s=setup(),poses=[];s.room.previewPose=p=>{poses.push(p);return true;};
  const pose={epoch:s.layer.readProtocol().epoch,assetIdentity:'scene:1',revision:1,environmentSequence:1,sequence:1,model:placement};
  assert.equal(s.layer.previewPose(pose),false);
  s.ready.resolve(visible());await s.layer.update(input());
  assert.equal(s.layer.previewPose({...pose,epoch:'old-session'}),false);assert.equal(poses.length,0);
  assert.equal(s.layer.previewPose(pose),true);assert.equal(poses[0],pose);
  s.layer.dispose();assert.equal(s.layer.previewPose(pose),false);assert.equal(poses.length,1);
});
