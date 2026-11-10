// Studio manager appearance on the original class world's existing render
// boundary. The original game still owns controls, clocks, worker and contacts.
import {hydrateStudioSceneAsset,verifyStudioSceneAssetIdentity,createStudioSceneRenderer} from './marble_studio_scene_asset.js';
import {mechanicalCreatureStudioCamera} from './mechanical_creature_studio_camera.js';

const error=(code,message)=>Object.assign(new Error(message),{code});
const assert=(ok,message)=>{if(!ok)throw error('INVALID_STUDIO_UPDATE',message);};

/** beforeCommit({identity,sequence,sourceRevision,building,isCurrent,commit})
 * queues the host's native environment update. The host checks isCurrent
 * immediately before mutation, applies geometry/contacts at its original frame
 * boundary, then synchronously calls commit(receipt). update() resolves only
 * after both visible phases run. This is not a worker execution receipt.
 * Without beforeCommit, pre-registered proxies gate the initial static path.
 * One hydration may run while one latest complete request waits. No new loop.
 */
export async function installStudioRoom({asset,building,host=globalThis,onStatus=()=>{},beforeCommit=null,
  sequence=0,sourceRevision=asset?.revision}={}) {
  const previous=host.mechanicalCreatureImportedAppearance;
  if(!previous)throw Error('The original game appearance hook must be installed first.');
  let closed=false,active=null,desired=null,hydrating=null,frame=null,lastFrame=0,drawCount=0,failure=null;
  let latestSequence=-1,latestRevision=-1;const retired=new Set();
  const report=value=>{try{onStatus(value);}catch(e){console.warn('Studio room status observer failed',e);}};
  const disposeRenderer=request=>{if(!request)return;const renderer=request.renderer;request.renderer=null;
    if(renderer)try{renderer.dispose();}catch(e){console.warn('Studio room disposal failed',e);}
    // A retired receipt must not retain the full decoded source geometry.
    request.hydrated=null;if(request!==hydrating)request.serialized=null;};
  const sourceIdentity=asset?.source?.classIdentity;
  assert(typeof sourceIdentity==='string'&&sourceIdentity.startsWith('python:'),'The Studio manager source class identity is required.');
  assert(beforeCommit===null||typeof beforeCommit==='function','Studio beforeCommit must be a callback.');
  function requestFor(input) {
    const {asset,building}=input,seq=input.sequence??latestSequence+1,revision=input.sourceRevision??asset?.revision,
      callback=input.beforeCommit===undefined?beforeCommit:input.beforeCommit;
    assert(asset?.schema==='turing.studio-scene-asset.v1'&&typeof asset.identity==='string','An actual serialized Studio scene asset is required.');
    assert(Number.isSafeInteger(seq)&&seq>=0&&Number.isSafeInteger(revision)&&revision>=0,'Manager sequence and source revision must be nonnegative integers.');
    if(seq<=latestSequence||revision<latestRevision)throw error('STALE_STUDIO_UPDATE','The Studio manager update is older than an already accepted sequence/revision.');
    assert(revision===asset.revision&&building?.assetIdentity===asset.identity,'Studio asset, source revision and native environment identity must match.');
    assert(building.sourceRevision===undefined||building.sourceRevision===revision,'The native environment has a different Studio source revision.');
    assert(asset.source?.classIdentity===sourceIdentity,'A scene update cannot change its Studio manager source class.');
    assert(callback===null||typeof callback==='function','Studio beforeCommit must be a callback.');
    let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});
    promise.catch(()=>{}); // Initial ready may be observed through onStatus.
    return {serialized:asset,building:structuredClone(building),sequence:seq,sourceRevision:revision,identity:asset.identity,
      beforeCommit:callback,stage:'queued',promise,resolve,reject,settled:false,visible:false,failed:false,
      hydrated:null,renderer:null,proxyIds:[],environment:null};
  }
  const current=request=>!closed&&desired===request&&!request.settled&&host.mechanicalCreatureImportedAppearance===adapter;
  function reject(request,reason,stage='failed') {
    if(!request||request.settled)return;
    request.settled=true;request.stage=stage;request.reject(reason);
    if(request!==active)disposeRenderer(request);
    if(stage==='failed'){failure=String(reason.message||reason);report({state:'failed',identity:request.identity,
      sequence:request.sequence,sourceRevision:request.sourceRevision,error:failure});}
  }
  function verifyHydrated(request,hydrated) {
    assert(hydrated.identity===request.identity&&hydrated.packet.revision===request.sourceRevision&&
      hydrated.packet.classIdentity===sourceIdentity,'The verified Studio asset changed identity, source or revision.');
    request.hydrated=hydrated;request.stage='waiting-for-original-render';
  }
  async function hydrate(request) {
    hydrating=request;request.stage='hydrating';
    try {
      // Same content needs no second geometry decode, but even the same caller
      // object must pass its checksum again: callers can mutate their objects.
      let hydrated;const reuse=active?.identity===request.identity?active.hydrated:null;
      if(reuse){await verifyStudioSceneAssetIdentity(request.serialized);hydrated=reuse;}
      else hydrated=await hydrateStudioSceneAsset(request.serialized);
      if(current(request)){verifyHydrated(request,hydrated);report({state:request.stage,identity:request.identity,sequence:request.sequence,sourceRevision:request.sourceRevision});}
    }catch(e){reject(request,e);}finally{
      if(hydrating===request)hydrating=null;
      if(request.settled&&request!==active){request.hydrated=null;request.serialized=null;}
      if(!closed&&desired&&!desired.settled&&!desired.hydrated&&desired!==request)void hydrate(desired);
    }
  }
  function proxyIds(request,context) {
    const registered=(context.sceneGeometry||[]).filter(box=>box.appearance?.studio_asset_identity===request.identity),
      expected=request.building.proxyIdentities??request.building.proxyIds;
    if(expected!==undefined){
      assert(Array.isArray(expected)&&expected.length>0&&expected.every(x=>typeof x==='string')&&new Set(expected).size===expected.length,'Native Studio proxy IDs must be unique.');
      const present=new Set(registered.map(x=>x.identity));return expected.every(id=>present.has(id))?expected.slice():null;
    }
    return registered.length?registered.map(box=>box.identity):null;
  }
  function publish(request,receipt) {
    if(!current(request))return false;
    assert(request.stage==='waiting-for-environment-commit','A Studio environment can commit its staged appearance only once.');
    assert(request.renderer?.ready,'A Studio candidate must be GPU-ready before native commit.');
    assert(receipt?.assetIdentity===request.identity&&receipt.revision===request.sourceRevision&&receipt.sequence===request.sequence,
      'Native environment commit does not match the staged Studio appearance.');
    if(active&&active!==request){if(active.visible)retired.add(active);else disposeRenderer(active);}
    active=request;request.environment=structuredClone(receipt);request.stage='awaiting-visible-frame';failure=null;
    report({state:request.stage,identity:request.identity,sequence:request.sequence,sourceRevision:request.sourceRevision,
      environment:request.environment});return true;
  }
  function stage(request,context) {
    if(!request.hydrated||request.settled||request.renderer)return;
    try {
      request.renderer=createStudioSceneRenderer({gl:context.gl,asset:request.hydrated,samplers:request.hydrated.samplers,
        placement:request.building.placement,entrance:request.building.renderEntrance??request.building.entrance});
      assert(request.renderer.ready,'The staged Studio renderer is not ready.');
      request.stage='waiting-for-environment-commit';
      report({state:request.stage,identity:request.identity,sequence:request.sequence,sourceRevision:request.sourceRevision});
      if(request.beforeCommit){
        const result=request.beforeCommit({identity:request.identity,sequence:request.sequence,sourceRevision:request.sourceRevision,
          building:request.building,isCurrent:()=>current(request),commit:receipt=>publish(request,receipt)});
        if(result&&typeof result.then==='function')Promise.resolve(result).catch(e=>reject(request,e));
      }
    }catch(e){reject(request,e);}
  }
  const adapter={drawWhenEmpty:true,
    identities(context) {
      // Never mutate the collection's Set or the original state Maps.
      const identities=new Set(previous.identities(context));frame=null;lastFrame++;
      if(closed)return identities;
      if(desired&&!desired.settled){
        stage(desired,context);
        if(desired.renderer&&!desired.beforeCommit&&desired!==active){
          const ids=proxyIds(desired,context);
          if(ids)publish(desired,{assetIdentity:desired.identity,revision:desired.sourceRevision,sequence:desired.sequence,
            geometryRevision:null,colliderPublication:'pre-existing-native-environment'});
        }
      }
      if(active&&!active.failed&&active.renderer?.ready){
        const ids=proxyIds(active,context);
        if(ids){active.proxyIds=ids;for(const id of ids)identities.add(id);
          frame={request:active,serial:lastFrame,phase:0,gameSnapshotSequence:context.stateLoopRuntime?.latestSequence??null};}
      }
      return identities;
    },
    draw(context) {
      const drawing=!closed?frame:null;
      try {
        if(drawing){
          assert(context.phase==='opaque'&&drawing.phase===0||context.phase==='transparent'&&drawing.phase===1,
            'Studio visibility acknowledgement requires original opaque then transparent passes.');
          assert(drawing.request.proxyIds.every(id=>context.identities?.has(id)),'The native visible pass did not admit every Studio proxy.');
          drawing.request.renderer.draw({...mechanicalCreatureStudioCamera(context),phase:context.phase});drawCount++;
        }
        const result=previous.draw(context);
        if(result===false||result&&typeof result.then==='function')throw Error('The original collection did not finish its synchronous appearance pass.');
        if(drawing){
          drawing.phase++;
          if(drawing.phase===2){
            const request=drawing.request;request.visible=true;
            if(!request.settled){
              request.settled=true;request.stage='visible';
              const receipt={state:'visible',identity:request.identity,sequence:request.sequence,sourceRevision:request.sourceRevision,
                sourceIdentity,renderFrame:drawing.serial,gameSnapshotSequence:drawing.gameSnapshotSequence,
                environment:request.environment};
              request.resolve(receipt);report({...receipt,state:'ready',visible:true,counts:request.renderer.counts,proxyCount:request.proxyIds.length});
            }
            for(const old of retired)if(old!==active)disposeRenderer(old);retired.clear();
          }
        }
        return result;
      }catch(e){
        if(drawing){drawing.request.failed=true;reject(drawing.request,e);failure=String(e.message||e);frame=null;}
        throw e; // Incumbent appearance hook restores native proxies on failure.
      }
    },
  };
  const initial=requestFor({asset,building,sequence,sourceRevision});
  verifyHydrated(initial,await hydrateStudioSceneAsset(asset));
  if(host.mechanicalCreatureImportedAppearance!==previous)throw Error('The game appearance owner changed while preparing Studio.');
  desired=initial;latestSequence=initial.sequence;latestRevision=initial.sourceRevision;
  host.mechanicalCreatureImportedAppearance=adapter;report({state:'waiting-for-original-render',identity:initial.identity,sequence:initial.sequence});
  return {
    ready:initial.promise,
    update(input) {
      try {
        if(closed)throw error('CLOSED_STUDIO_ROOM','The Studio room is closed.');
        if(host.mechanicalCreatureImportedAppearance!==adapter)throw error('STUDIO_APPEARANCE_OWNER_CHANGED','The original game appearance owner changed.');
        const request=requestFor(input);latestSequence=request.sequence;latestRevision=request.sourceRevision;
        reject(desired,error('SUPERSEDED_STUDIO_UPDATE','A newer complete Studio manager update replaced this request.'),'superseded');
        desired=request;if(!hydrating)void hydrate(request);return request.promise;
      }catch(e){return Promise.reject(e);}
    },
    readStatus:()=>({closed,ready:!!active?.visible&&!active.failed,identity:active?.identity??null,
      sourceIdentity,sourceRevision:active?.sourceRevision??null,sequence:active?.sequence??null,latestSequence,latestRevision,
      pending:desired&&!desired.settled?{identity:desired.identity,sourceRevision:desired.sourceRevision,sequence:desired.sequence,stage:desired.stage}:null,
      drawCount,proxyIds:active?.proxyIds.slice()??[],failure,bounds:active?.renderer?.bounds,counts:active?.renderer?.counts,
      environment:active?.environment??null}),
    dispose(){if(closed)return;closed=true;
      if(host.mechanicalCreatureImportedAppearance===adapter)host.mechanicalCreatureImportedAppearance=previous;
      reject(desired,error('CLOSED_STUDIO_ROOM','The Studio room was closed before this update became visible.'),'closed');
      for(const request of new Set([active,desired,hydrating,...retired]))disposeRenderer(request);
      active=null;frame=null;retired.clear();report({state:'closed'});},
  };
}
