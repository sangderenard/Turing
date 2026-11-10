import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import vm from 'node:vm';

// Execute the actual host. Only its two asynchronous module dependencies are
// replaced so each installer boundary can be settled explicitly, without a GL
// context, browser, timer, worker or change to the production host API.
const path=new URL('../examples/meowdoku/web/mechanical_creature_studio_layer_host.js',import.meta.url);
let source=readFileSync(path,'utf8');
for(const declaration of [
  "import {installStudioCollection} from './mechanical_creature_studio_collection.js';",
  "import {installStudioRoom} from './mechanical_creature_studio_room.js';",
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
  source:{revision:sequence},marbleAsset:{identity:'chosen-marble'},sequence});
const visible=(sequence=1)=>({state:'visible',identity:'scene:'+sequence,sequence,sourceRevision:sequence});

function setup({collectionInstaller,roomInstaller}={}){
  const calls={collection:0,room:0,queue:0,sinks:[],roomOptions:null},body=node();
  const doc={body,createElement:node,createTextNode:text=>({textContent:text})};
  const collection={disposed:0,readStatus:()=>({kind:'collection'}),dispose(){this.disposed++;}};
  const ready=deferred(),room={disposed:0,ready:ready.promise,readStatus:()=>({kind:'room'}),
    update:async value=>visible(value.sequence),dispose(){this.disposed++;}};
  // The actual room also observes its own initial promise before returning its
  // handle; cancellation during installation must not create an unhandled rejection.
  ready.promise.catch(()=>{});
  const host={mechanicalCreatureAmmo:{items:()=>[],selected:()=>null},mechanicalCreatureStudioBridge:{
    prepareEnvironment:()=>({definition:{identity:'owned'},building:{identity:'owned'}}),
    queueEnvironment(){calls.queue++;return Promise.resolve();},
    setSink(value){calls.sinks.push(value);},readSnapshot:()=>null,
  }};
  const context=vm.createContext({Error,crypto:{randomUUID:()=> 'original-epoch'},
    installStudioCollection:options=>{calls.collection++;return collectionInstaller?.(options,collection)??Promise.resolve(collection);},
    installStudioRoom:options=>{calls.room++;calls.roomOptions=options;
      return roomInstaller?.(options,room)??Promise.resolve(room);},
  });
  vm.runInContext(source,context,{filename:path.pathname});
  const layer=context.createMechanicalCreatureStudioLayerHost({host,doc});
  return {layer,calls,collection,room,ready,panel:body.children[0]};
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
