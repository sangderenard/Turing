import assert from 'node:assert/strict';
import {createStudioGameLayerController} from '../examples/meowdoku/web/marble_studio_game_layer.js';
const tick=()=>new Promise(resolve=>setImmediate(resolve));
let revision=1,captures=0,sink=null,updates=[],draws=0,disposed=0,receiverCreated=0,environment=null,hold=null;
const packet=()=>({schema:'turing.studio-class-world-import.v1',revision,classIdentity:'python:actual.Class',
  scene:{frame:{unitM:.3048},room:{},parts:[]}});
const host={connect(fn){sink=fn;return ()=>{sink=null;};},readSnapshot:()=>({epoch:'original-test',accepted:true}),
  async update(value){updates.push(value);return {building:{identity:'registered-room',placement:[1,0,0,0,0,1,0,0,0,0,1,0,32,1.5,-18,1]},sequence:value.sequence};}};
const make=()=>createStudioGameLayerController({capture:()=>{captures++;return hold??packet();},
  sceneAsset:async p=>({identity:'scene:'+p.revision,revision:p.revision}),gameAsset:async p=>({identity:'marble:'+p.revision}),
  createReturn:async options=>{receiverCreated++;environment=options.environment;return {receive:p=>({accepted:p.accepted}),read:()=>({bodies:[]}),
    setEnvironment:e=>{environment=e;},draw(){draws++;},dispose(){disposed++;}};}});
const c=make();c.frame({revision});await tick();assert.equal(captures,0,'off means no capture or engine action');
c.connect(host);c.frame({revision});await tick();assert.equal(c.read().sentRevision,1);assert.equal(updates.length,1);
assert.equal(environment.environmentSequence,1);assert.equal(environment.buildingIdentity,'registered-room');
assert.equal(receiverCreated,1);assert.equal(c.read().receivedFrames,1);
c.frame({revision,gl:{},view:[],projection:[]});await tick();assert.equal(captures,1,'same revision does not recapture');assert.equal(draws,1);
sink({accepted:false});assert.equal(c.read().receivedFrames,1);sink({accepted:true});assert.equal(c.read().receivedFrames,2);
revision=2;c.frame({revision});revision=3;c.frame({revision});await tick();c.frame({revision});await tick();
assert.deepEqual(updates.map(x=>x.source.revision),[1,2,3]);assert.equal(updates[1].marbleAsset,null,'environment update cannot refill ammo');
assert.equal(receiverCreated,1,'reuse the admitted returned-body renderer');assert.equal(environment.environmentRevision,3);
c.stop();assert.equal(disposed,1);assert.equal(sink,null);assert.equal(c.read().enabled,false);
let finish;hold=new Promise(resolve=>{finish=resolve;});const stopped=make();stopped.connect(host);stopped.frame({revision});stopped.stop();
const count=updates.length;finish(packet());await tick();assert.equal(updates.length,count,'stopped generation never reaches game');hold=null;
const bad=createStudioGameLayerController({capture(){throw Error('not displayed');},sceneAsset:async()=>{throw Error('unreachable');}});
bad.connect(host);bad.frame({revision});await tick();assert.match(bad.read().error,/not displayed/);bad.frame({revision});await tick();
assert.equal(bad.read().busy,false);bad.retry();assert.equal(bad.read().error,null);bad.stop();
// Stop owns its generation immediately. A rejected old capture must not block
// or poison a new game's pending capture after the user has opened it again.
const pending=[];let reopenUpdates=0;
const reopening=createStudioGameLayerController({capture:()=>new Promise((resolve,reject)=>pending.push({resolve,reject})),
  sceneAsset:async p=>({identity:'scene:'+p.revision,revision:p.revision}),gameAsset:async p=>({identity:'marble:'+p.revision}),
  createReturn:async()=>({receive:()=>({accepted:true}),read:()=>({bodies:[]}),dispose(){}})});
const newHost=()=>({connect(){return ()=>{};},readSnapshot:()=>({epoch:'new-game'}),dispose(){},
  async update(){reopenUpdates++;return {building:{identity:'room',placement:[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]}};}});
reopening.connect(newHost());reopening.frame({revision:7});assert.equal(pending.length,1);
reopening.stop();assert.equal(reopening.read().busy,false,'Stop releases its own busy generation');
reopening.connect(newHost());reopening.frame({revision:8});assert.equal(pending.length,2,'new generation can capture without the old promise');
pending[0].reject(Error('old capture failed after Stop'));await tick();
assert.equal(reopening.read().error,null,'stale rejection cannot poison current generation');
assert.equal(reopening.read().busy,true,'stale finally cannot clear current capture ownership');
reopening.frame({revision:8});assert.equal(pending.length,2,'new capture remains single flight');
pending[1].resolve({schema:'captured',revision:8,classIdentity:'original',scene:{frame:{unitM:.3048},room:{},parts:[]}});
await tick();assert.equal(reopening.read().sentRevision,8);assert.equal(reopenUpdates,1);reopening.stop();
console.log(JSON.stringify({passed:true,checks:13,scope:'controller lifecycle only; no browser or native physics claim'}));
