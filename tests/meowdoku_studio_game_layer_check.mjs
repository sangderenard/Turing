import assert from 'node:assert/strict';
import {createStudioGameProtocol} from '../examples/meowdoku/web/marble_studio_game_protocol.js';
import {createStudioGameLayerController} from '../examples/meowdoku/web/marble_studio_game_layer.js';
const tick=()=>new Promise(resolve=>setImmediate(resolve));
let revision=1,captures=0,sink=null,updates=[],draws=0,disposed=0,receiverCreated=0,environment=null,hold=null;
const packet=()=>({schema:'turing.studio-class-world-import.v1',revision,classIdentity:'python:actual.Class',
  scene:{frame:{unitM:.3048},room:{},parts:[]}});
const protocol=createStudioGameProtocol('original-test');
const receiptFor=value=>({epoch:protocol.epoch,state:'visible',identity:value.asset.identity,sourceRevision:value.source.revision,sequence:value.sequence,sourceIdentity:value.source.classIdentity,renderFrame:1,
  building:{identity:'registered-room',assetIdentity:value.asset.identity,placement:[1,0,0,0,0,1,0,0,0,0,1,0,32,1.5,-18,1]},
  environment:{buildingIdentity:'registered-room',assetIdentity:value.asset.identity,revision:value.source.revision,sequence:value.sequence,geometryRevision:1,colliderPublication:'posted-to-original-worker'},
  colliderPublication:'posted-to-original-worker',physics:'original-game-worker',camera:'original-game-controls'});
const host={readProtocol:()=>protocol,connect(fn){sink=fn;return ()=>{sink=null;};},readSnapshot:()=>({epoch:'original-test',accepted:true}),
  async update(value){updates.push(value);return receiptFor(value);}};
const make=()=>createStudioGameLayerController({capture:()=>{captures++;return hold??packet();},
  sceneAsset:async p=>({identity:'scene:'+p.revision,revision:p.revision}),ammunition:async p=>({supported:true,asset:{identity:'marble:'+p.revision},reason:null}),
  createReturn:async options=>{receiverCreated++;environment=options.environment;return {receive:p=>({accepted:p.accepted}),read:()=>({bodies:[]}),
    setEnvironment:e=>{environment=e;},draw(){draws++;},dispose(){disposed++;}};}});
const c=make();c.frame({revision});await tick();assert.equal(captures,0,'off means no capture or engine action');
c.connect(host);c.frame({revision});await tick();assert.equal(c.read().sentRevision,1);assert.equal(updates.length,1);
assert.equal(environment.environmentSequence,1);assert.equal(environment.buildingIdentity,'registered-room');
assert.equal(receiverCreated,1);assert.equal(c.read().receivedFrames,1);
const poses=[];host.previewPose=p=>poses.push(structuredClone(p));
const model=[1,0,0,0,0,1,0,0,0,0,1,0,0,2,0,1];
c.frame({revision,subjectPose:model});
assert.equal(poses.length,1);assert.equal(poses[0].epoch,protocol.epoch);assert.equal(poses[0].assetIdentity,'scene:1');
assert.equal(poses[0].environmentSequence,1);assert.deepEqual(poses[0].model,model);
assert.equal(captures,1,'pose does not recapture or hash scene');
c.frame({revision,gl:{},view:[],projection:[]});await tick();assert.equal(captures,1,'same revision does not recapture');assert.equal(draws,1);
sink({accepted:false});assert.equal(c.read().receivedFrames,1);sink({accepted:true});assert.equal(c.read().receivedFrames,2);
revision=2;c.frame({revision,subjectPose:model});assert.equal(poses.length,1,'new revision cannot animate old scene');revision=3;c.frame({revision});await tick();c.frame({revision});await tick();
assert.deepEqual(updates.map(x=>x.source.revision),[1,3],'only latest uncommitted authored revision is packed');assert.equal(updates[1].marbleAsset,null,'environment update cannot refill ammo');
assert.equal(receiverCreated,1,'reuse the admitted returned-body renderer');assert.equal(environment.environmentRevision,3);
c.stop();c.frame({revision,subjectPose:model});assert.equal(poses.length,1,'Stop blocks pose forwarding');assert.equal(disposed,1);assert.equal(sink,null);assert.equal(c.read().enabled,false);
let finish;hold=new Promise(resolve=>{finish=resolve;});const stopped=make();stopped.connect(host);stopped.frame({revision});stopped.stop();
const count=updates.length;finish(packet());await tick();assert.equal(updates.length,count,'stopped generation never reaches game');hold=null;
const bad=createStudioGameLayerController({capture(){throw Error('not displayed');},sceneAsset:async()=>{throw Error('unreachable');}});
bad.connect(host);bad.frame({revision});await tick();assert.match(bad.read().error,/not displayed/);bad.frame({revision});await tick();
assert.equal(bad.read().busy,false);bad.retry();assert.equal(bad.read().error,null);bad.stop();
// Stop owns its generation immediately. A rejected old capture must not block
// or poison a new game's pending capture after the user has opened it again.
const pending=[];let reopenUpdates=0;
const reopening=createStudioGameLayerController({capture:()=>new Promise((resolve,reject)=>pending.push({resolve,reject})),
  sceneAsset:async p=>({identity:'scene:'+p.revision,revision:p.revision}),ammunition:async p=>({supported:true,asset:{identity:'marble:'+p.revision},reason:null}),
  createReturn:async()=>({receive:()=>({accepted:true}),read:()=>({bodies:[]}),dispose(){}})});
const newHost=()=>({readProtocol:()=>protocol,connect(){return ()=>{};},readSnapshot:()=>({epoch:protocol.epoch}),dispose(){},
  async update(value){reopenUpdates++;return receiptFor(value);}});
reopening.connect(newHost());reopening.frame({revision:7});assert.equal(pending.length,1);
reopening.stop();assert.equal(reopening.read().busy,false,'Stop releases its own busy generation');
reopening.connect(newHost());reopening.frame({revision:8});assert.equal(pending.length,2,'new generation can capture without the old promise');
pending[0].reject(Error('old capture failed after Stop'));await tick();
assert.equal(reopening.read().error,null,'stale rejection cannot poison current generation');
assert.equal(reopening.read().busy,true,'stale finally cannot clear current capture ownership');
reopening.frame({revision:8});assert.equal(pending.length,2,'new capture remains single flight');
pending[1].resolve({schema:'captured',revision:8,classIdentity:'original',scene:{frame:{unitM:.3048},room:{},parts:[]}});
await tick();assert.equal(reopening.read().sentRevision,8);assert.equal(reopenUpdates,1);reopening.stop();
// A successful promise is not an acknowledgment of the requested scene.
let admittedWrong=0;
const wrong=createStudioGameLayerController({capture:packet,sceneAsset:async p=>({identity:'scene:'+p.revision,revision:p.revision}),ammunition:async()=>({supported:true,asset:{identity:'marble'},reason:null}),
  createReturn:async()=>{admittedWrong++;return {receive:()=>({accepted:true}),read:()=>null,dispose(){}};}});
wrong.connect({...host,async update(value){return {...receiptFor(value),sequence:value.sequence+1};}});wrong.frame({revision});await tick();
assert.equal(wrong.read().sentRevision,-1,'wrong acknowledgment is not admitted');assert.equal(admittedWrong,0);assert.match(wrong.read().error,/acknowledgment/);wrong.stop();
// The epoch is negotiated before snapshots. Initial worker startup may not
// have published a frame yet; this must not fail a valid visible scene.
let currentSink=null,connectionCount=0,returnOptions=null;
const noSnapshot=createStudioGameLayerController({capture:packet,sceneAsset:async p=>({identity:'scene:'+p.revision,revision:p.revision}),ammunition:async()=>({supported:true,asset:{identity:'marble'},reason:null}),
  createReturn:async options=>{returnOptions=options;return {receive:()=>({accepted:true}),read:()=>null,dispose(){}};}});
const lateHost={...host,connect(fn){connectionCount++;currentSink=fn;return ()=>{};},readSnapshot:()=>null};
noSnapshot.connect(lateHost);noSnapshot.connect(lateHost);assert.equal(connectionCount,1,'connect is idempotent for its current host');
noSnapshot.frame({revision});await tick();assert.equal(noSnapshot.read().sentRevision,revision);assert.equal(noSnapshot.read().error,null);
assert.equal(returnOptions.epoch,protocol.epoch);assert.equal(noSnapshot.read().receivedFrames,0);
currentSink({accepted:true});assert.equal(noSnapshot.read().receivedFrames,1);
const oldSink=currentSink;noSnapshot.stop();noSnapshot.connect({...lateHost});noSnapshot.frame({revision});await tick();
oldSink({accepted:true});assert.equal(noSnapshot.read().receivedFrames,0,'stopped callback cannot feed the replacement session');noSnapshot.stop();
const incompatible=make();let attemptedConnect=0;
assert.throws(()=>incompatible.connect({...host,readProtocol:()=>({...protocol,dtAuthority:'foreign-clock'}),connect(){attemptedConnect++;}}),/dtAuthority/);
assert.equal(attemptedConnect,0,'unsupported ownership rejected before sink installation');assert.equal(incompatible.read().enabled,false);
// The full Studio scene is independent of the original sphere ammunition.
let ammoChecks=0,emptyReturnAssets=null;const noAmmoUpdates=[];
const noAmmo=createStudioGameLayerController({capture:packet,sceneAsset:async p=>({identity:'scene:'+p.revision,revision:p.revision}),
  ammunition:async()=>{ammoChecks++;return {supported:false,asset:null,reason:'Existing sphere body cannot represent this subject.'};},
  createReturn:async options=>{emptyReturnAssets=options.assets;return {receive:()=>({accepted:true}),read:()=>null,setEnvironment(){},dispose(){}};}});
noAmmo.connect({...host,async update(value){noAmmoUpdates.push(value);return receiptFor(value);}});noAmmo.frame({revision});await tick();
assert.equal(noAmmo.read().sentRevision,revision);assert.deepEqual(emptyReturnAssets,[]);assert.equal(noAmmoUpdates[0].marbleAsset,null);
assert.equal(noAmmo.read().ammunition.supported,false);revision++;noAmmo.frame({revision});await tick();
assert.equal(ammoChecks,1,'later environment changes cannot refill or reselect inventory');assert.equal(noAmmoUpdates.length,2);noAmmo.stop();
// Once the original game owns an in-flight commit, bind that receipt before
// requesting the latest revision; do not leave Studio using the old mapping.
let settleCommit=null;const committed=[];
const commitHost={...host,update:value=>{committed.push(value);return new Promise(resolve=>{settleCommit=()=>resolve(receiptFor(value));});}};
const duringCommit=createStudioGameLayerController({capture:packet,sceneAsset:async p=>({identity:'scene:'+p.revision,revision:p.revision}),
  ammunition:async()=>({supported:false,asset:null,reason:'No sphere ammo'}),
  createReturn:async()=>({receive:()=>({accepted:true}),read:()=>null,setEnvironment(){},dispose(){}})});
duringCommit.connect(commitHost);duringCommit.frame({revision});await tick();assert.equal(committed.length,1);
const committingRevision=revision;revision++;duringCommit.frame({revision});settleCommit();await tick();
assert.equal(duringCommit.read().sentRevision,committingRevision,'actual committed intermediate environment is retained');
duringCommit.frame({revision});await tick();assert.equal(committed.length,2);settleCommit();await tick();
assert.equal(duringCommit.read().sentRevision,revision);duringCommit.stop();
console.log(JSON.stringify({passed:true,checks:21,scope:'controller lifecycle only; no browser or native physics claim'}));
