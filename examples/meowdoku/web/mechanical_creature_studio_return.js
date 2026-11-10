// Return copied, actual MechanicalCreature frames to the existing Studio draw.
// No clock, interpolation, worker, collider, angular law or new draw mechanism.
import {hydrateStudioGameAsset} from './marble_game_asset.js';
import {createStudioSubjectRenderer,validateStudioLightTransform} from './marble_studio_subject_renderer.js';
import {studioRigToGame} from './mechanical_creature_studio_collection.js';
import {rotationMatrix} from './marble_roll.js';

const I=Object.freeze([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
const fail=(condition,message)=>{if(!condition)throw Error(message);};
const identity=(value,name)=>{fail(typeof value==='string'&&value.length>0,`${name} must be an explicit identity`);return value;};
const counter=(value,name)=>{fail(Number.isSafeInteger(value)&&value>=0,`${name} must be a nonnegative integer`);return value;};
const finite=(value,name)=>{fail(Number.isFinite(value),`${name} must be finite`);return value;};
const vector=(value,size,name)=>{fail(Array.isArray(value)&&value.length===size&&value.every(Number.isFinite),`${name} needs ${size} finite values`);return value.slice();};
const freeze=value=>{if(value&&typeof value==='object'&&!Object.isFrozen(value)){for(const child of Object.values(value))freeze(child);Object.freeze(value);}return value;};
const multiply=(a,b)=>{const m=Array(16).fill(0);for(let c=0;c<4;c++)for(let r=0;r<4;r++)
  for(let k=0;k<4;k++)m[c*4+r]+=a[k*4+r]*b[c*4+k];return m;};
const point=(m,p,w=1)=>[0,1,2].map(r=>m[r]*p[0]+m[4+r]*p[1]+m[8+r]*p[2]+m[12+r]*w);

function environment(value) {
  const {buildingIdentity,environmentAssetIdentity,environmentRevision,environmentSequence,buildingPlacement,unitM}=value??{};
  identity(buildingIdentity,'buildingIdentity');identity(environmentAssetIdentity,'environmentAssetIdentity');
  counter(environmentRevision,'environmentRevision');counter(environmentSequence,'environmentSequence');
  const placement=vector(Array.from(buildingPlacement??[]),16,'buildingPlacement');
  fail(Math.abs(validateStudioLightTransform(placement).scale-1)<1e-6,'Registered Studio placement must be rigid; unitM owns its scale');
  fail(Number.isFinite(unitM)&&unitM>0,'Studio unitM must be positive');
  const inverseUnit=I.slice();inverseUnit[0]=inverseUnit[5]=inverseUnit[10]=1/unitM;
  // Existing rig inversion: target scale * inverse registered placement.
  const worldToStudio=Array.from(studioRigToGame(placement,inverseUnit));
  fail(worldToStudio.every(Number.isFinite),'World-to-Studio mapping is nonfinite');
  return freeze({buildingIdentity,environmentAssetIdentity,environmentRevision,environmentSequence,
    buildingPlacement:placement,unitM,worldToStudio});
}

/** Two retained same-thread slots, published only after complete validation.
 * This is latest-frame overwrite, not SharedArrayBuffer/Atomics or a FIFO.
 * environmentSequence orders requests; authored environmentRevision can repeat.
 * setEnvironment stages mapping until an exact copied game receipt arrives.
 * colliderPublication is only posted-to-original-worker, not a worker-applied
 * contact acknowledgment. Missing IDs in an accepted full frame are removed.
 */
export function createMechanicalCreatureStudioReturnStore({epoch,environment:initialEnvironment,
  appearanceAssetIdentities,fixedQuaternionXYZW}={}) {
  identity(epoch,'epoch');
  fail(Array.isArray(appearanceAssetIdentities)&&appearanceAssetIdentities.length<=8,
    'Zero to eight admitted appearance identities are required');
  const appearances=new Set(appearanceAssetIdentities.map(value=>identity(value,'appearanceAssetIdentity')));
  fail(appearances.size===appearanceAssetIdentities.length,'Duplicate appearance identity');
  const q=vector(fixedQuaternionXYZW,4,'fixedQuaternionXYZW');
  fail(Math.abs(Math.hypot(...q)-1)<=8*Number.EPSILON,'Fixed authored quaternion must be unit length');
  const rotation=rotationMatrix([q[3],q[0],q[1],q[2]]);
  let desired=environment(initialEnvironment),acceptedEnvironment=null,closed=false,active=-1;
  const slots=[null,null];
  const current=()=>active<0?null:slots[active];
  function setEnvironment(next) {
    fail(!closed,'Studio return store is closed');next=environment(next);
    fail(next.buildingIdentity===desired.buildingIdentity,'Registered Studio building identity cannot change within an epoch');
    fail(next.environmentSequence>=desired.environmentSequence,'Environment request sequence regressed');
    if(next.environmentSequence===desired.environmentSequence) {
      fail(JSON.stringify(next)===JSON.stringify(desired),'An environment sequence cannot name two different placements or captures');
      return desired;
    }
    desired=next;return desired; // Keep accepted slots and their old mapping.
  }
  function receive(input) {
    fail(!closed,'Studio return store is closed');
    fail(input?.schema==='mechanical-creature-studio-state-v1','Unknown original game return packet');
    if(input.epoch!==epoch)return {accepted:false,reason:'different-epoch'};
    const packet=structuredClone(input),prior=current();
    fail(packet.dtAuthority==='original-viewport-slice','The original viewport dt slice must remain authoritative');
    finite(packet.frameTimeMs,'frameTimeMs');finite(packet.frameDtSeconds,'frameDtSeconds');
    fail(packet.frameTimeMs>=0&&packet.frameDtSeconds>=0,'Original frame time and dt cannot be negative');
    counter(packet.workerSequence,'workerSequence');counter(packet.geometryRevision,'geometryRevision');
    fail(typeof packet.workerReady==='boolean','Original workerReady flag is required');
    if(prior&&packet.frameTimeMs<=prior.frameTimeMs)return {accepted:false,reason:'stale-frame'};
    if(prior&&packet.workerSequence<prior.workerSequence)return {accepted:false,reason:'stale-worker-sequence'};
    if(prior&&packet.geometryRevision<prior.geometryRevision)return {accepted:false,reason:'stale-geometry-revision'};
    fail(Array.isArray(packet.environments),'Copied game environment receipts are required');
    const matching=packet.environments.filter(row=>row?.buildingIdentity===desired.buildingIdentity);
    fail(matching.length<=1,'Duplicate environment receipt for the registered building');
    if(!matching.length)return {accepted:false,reason:'environment-not-published'};
    const receipt=matching[0];
    identity(receipt.assetIdentity,'Receipt assetIdentity');counter(receipt.revision,'Receipt revision');
    counter(receipt.sequence,'Receipt sequence');counter(receipt.geometryRevision,'Receipt geometryRevision');
    if(receipt.colliderPublication==='native-scene-ready-worker-pending')
      return {accepted:false,reason:'environment-worker-pending'};
    fail(receipt.colliderPublication==='posted-to-original-worker','Unsupported environment publication receipt');
    fail(receipt.geometryRevision<=packet.geometryRevision,'Environment receipt is newer than the returned world geometry');
    let mapping=null;
    for(const candidate of [desired,acceptedEnvironment])if(candidate&&receipt.assetIdentity===candidate.environmentAssetIdentity&&
      receipt.revision===candidate.environmentRevision&&receipt.sequence===candidate.environmentSequence){mapping=candidate;break;}
    if(!mapping)return {accepted:false,reason:'environment-receipt-not-registered'};
    if(acceptedEnvironment&&mapping.environmentSequence<acceptedEnvironment.environmentSequence)
      return {accepted:false,reason:'stale-environment-sequence'};
    if(prior&&receipt.sequence===prior.environmentReceipt.sequence&&receipt.geometryRevision<prior.environmentReceipt.geometryRevision)
      return {accepted:false,reason:'stale-environment-geometry'};
    fail(Array.isArray(packet.bodies)&&packet.bodies.length<=4096,'A bounded complete tagged body list is required');
    const ids=new Set(),memberships=new Set(),oldBodies=new Map((prior?.bodies??[]).map(body=>[body.identity,body]));
    const bodies=packet.bodies.map(body=>{
      identity(body?.identity,'Original body identity');identity(body.appearanceAssetIdentity,'Original appearance tag');
      identity(body.ammunitionItemIdentity,'Original ammunition item tag');
      fail(appearances.has(body.appearanceAssetIdentity),'Returned body names an unregistered appearance asset');
      fail(!ids.has(body.identity),'Duplicate returned body identity');ids.add(body.identity);
      const positionM=vector(body.positionM,3,'Original body positionM'),velocityMPerS=vector(body.velocityMPerS,3,'Original body velocityMPerS');
      fail(Number.isFinite(body.radiusM)&&body.radiusM>0,'Original body radiusM must be positive');
      fail(typeof body.sleeping==='boolean'&&typeof body.pickup==='boolean','Original lifecycle flags are required');
      fail(body.status===(body.pickup?'pickup':body.sleeping?'sleeping':'active'),'Original body status and lifecycle flags disagree');
      fail(!('quaternion' in body)&&!('orientation' in body)&&body.physicsQuaternionXYZW==null,
        'Original projectile orientation contract changed; do not discard a published pose');
      if(body.membership!==null){
        fail(body.membership&&typeof body.membership==='object','Explicit original snapshot membership or null is required');
        counter(body.membership.slot,'Original slot');counter(body.membership.generation,'Original generation');
        fail(body.membership.generation>0,'Original snapshot generation must be positive');
        fail(!body.sleeping&&!body.pickup,'Sleeping/pickup bodies cannot retain active snapshot membership');
        fail(!memberships.has(body.membership.slot),'Two returned bodies claim the same original snapshot slot');memberships.add(body.membership.slot);
      }else fail(body.sleeping||body.pickup||!packet.workerReady,'Active returned body lost its original worker membership');
      const old=oldBodies.get(body.identity);
      if(old){
        fail(old.appearanceAssetIdentity===body.appearanceAssetIdentity&&old.ammunitionItemIdentity===body.ammunitionItemIdentity,
          'An existing original body cannot silently change appearance or ammunition identity');
        fail(old.radiusM===body.radiusM,'An existing original body cannot silently change radius');
        if(old.membership&&body.membership&&old.membership.slot===body.membership.slot)
          fail(body.membership.generation>=old.membership.generation,'Original snapshot slot generation regressed');
      }
      const gameModel=Array(16).fill(0);
      for(let c=0;c<3;c++)for(let r=0;r<3;r++)gameModel[c*4+r]=rotation[r][c]*body.radiusM;
      gameModel.splice(12,4,...positionM,1);
      const model=multiply(mapping.worldToStudio,gameModel),positionStudio=point(mapping.worldToStudio,positionM),
        velocityStudioPerS=point(mapping.worldToStudio,velocityMPerS,0),radiusStudio=body.radiusM/mapping.unitM;
      fail([...model,...positionStudio,...velocityStudioPerS,radiusStudio].every(Number.isFinite),'Mapped body exceeds Studio coordinate range');
      return {...body,positionM,velocityMPerS,positionStudio,velocityStudioPerS,radiusStudio,model,
        physicsQuaternionXYZW:null,fixedQuaternionXYZW:q.slice(),orientationAuthority:'explicit-fixed-authored-game-world-pose'};
    });
    if(packet.camera!==null){
      fail(packet.camera&&typeof packet.camera==='object','Original camera must be copied or explicitly null');
      vector(packet.camera.positionM,3,'Original camera positionM');
      const facing=vector(packet.camera.facing,3,'Original camera facing');fail(Math.hypot(...facing)>0,'Original camera facing is degenerate');
    }
    const next=freeze({schema:'mechanical-creature-studio-return-frame-v1',epoch,
      frameTimeMs:packet.frameTimeMs,frameDtSeconds:packet.frameDtSeconds,dtAuthority:packet.dtAuthority,
      workerSequence:packet.workerSequence,workerReady:packet.workerReady,geometryRevision:packet.geometryRevision,
      environment:mapping,environmentReceipt:receipt,bodies,camera:packet.camera,
      lighting:'captured-ammunition-Phong-rig-in-original-Studio-coordinates',
      publication:'same-thread-latest-complete-frame-double-slot',workerAppliedEnvironment:false});
    const nextSlot=active<0?0:1-active;slots[nextSlot]=next;active=nextSlot;acceptedEnvironment=mapping;
    return {accepted:true,slot:active,frameTimeMs:next.frameTimeMs,environmentSequence:mapping.environmentSequence};
  }
  return {receive,read:current,setEnvironment,
    readStatus:()=>({closed,epoch,activeSlot:active,frameTimeMs:current()?.frameTimeMs??null,
      acceptedEnvironmentSequence:acceptedEnvironment?.environmentSequence??null,desiredEnvironmentSequence:desired.environmentSequence,
      bodyCount:current()?.bodies.length??0,workerAppliedEnvironment:false}),
    dispose(){closed=true;slots[0]=slots[1]=null;active=-1;}};
}

/** Hydrate exact existing appearance assets once. Caller owns Studio's current
 * GL, framebuffer, viewport, view/projection and frame scheduling. Existing
 * createStudioSubjectRenderer owns all GPU drawing and GL state restoration.
 */
export async function createMechanicalCreatureStudioReturn({assets,...options}={}) {
  fail(Array.isArray(assets)&&assets.length<=8,'Zero to eight exact Studio appearance assets are required');
  const hydrated=[];for(const asset of assets)hydrated.push(await hydrateStudioGameAsset(asset));
  const store=createMechanicalCreatureStudioReturnStore({...options,appearanceAssetIdentities:hydrated.map(asset=>asset.identity)});
  const byIdentity=new Map(hydrated.map(asset=>[asset.identity,asset])),renderers=new Map();let ownerGL=null,closed=false;
  function draw({gl,view,projection,phase='all'}={}) {
    fail(!closed,'Studio return renderer is closed');fail(['all','opaque','transparent'].includes(phase),'Invalid Studio return draw phase');
    const frame=store.read();if(!frame||!frame.bodies.length)return {bodies:0,draws:0};
    fail(gl,'Use the existing Studio WebGL2 context');
    if(ownerGL)fail(ownerGL===gl,'Studio context changed; dispose and recreate the return renderer');else ownerGL=gl;
    const v=vector(Array.from(view??[]),16,'Studio view');vector(Array.from(projection??[]),16,'Studio projection');
    for(const body of frame.bodies)if(!renderers.has(body.appearanceAssetIdentity)){
      const asset=byIdentity.get(body.appearanceAssetIdentity);
      renderers.set(asset.identity,createStudioSubjectRenderer({gl,packet:asset.packet,samplers:asset.samplers}));
    }
    const transparent=[...frame.bodies].sort((a,b)=>point(v,a.positionStudio)[2]-point(v,b.positionStudio)[2]);
    let draws=0;
    for(const pass of phase==='all'?['opaque','transparent']:[phase])for(const body of pass==='transparent'?transparent:frame.bodies){
      renderers.get(body.appearanceAssetIdentity).draw({view,projection,model:body.model,lightTransform:I,phase:pass});draws++;
    }
    return {bodies:frame.bodies.length,draws,frameTimeMs:frame.frameTimeMs,
      lighting:'captured-ammunition-Phong-rig-in-original-Studio-coordinates'};
  }
  return {...store,draw,dispose(){if(closed)return;closed=true;store.dispose();for(const renderer of renderers.values())renderer.dispose();renderers.clear();}};
}
