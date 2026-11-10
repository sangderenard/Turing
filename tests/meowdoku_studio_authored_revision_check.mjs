// Execute current revision/invalidation hooks, not a new revision mechanism.
// Pure offline test: no GL, browser, native worker, compiler or server.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {createMechanicalCreatureStudioReturnStore} from '../examples/meowdoku/web/mechanical_creature_studio_return.js';
const source=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio.js',import.meta.url),'utf8'),
  lighting=fs.readFileSync(new URL('../examples/meowdoku/web/marble_studio_lighting.js',import.meta.url),'utf8');
const listeners=new Map(),doc={addEventListener(kind,fn,capture){assert.equal(capture,true);if(!listeners.has(kind))listeners.set(kind,[]);listeners.get(kind).push(fn);}};
const context=vm.createContext({document:doc,doc});
const between=(text,start,end)=>{const a=text.indexOf(start),b=text.indexOf(end,a);assert(a>=0&&b>a);return text.slice(a,b);};
vm.runInContext(`let lastGeometry={};let materialSignals=0;
let onChange=()=>{},onRoomChange=()=>{},onBackdropChange=()=>{},onGoboChange=()=>{};
${between(source,'let studioSceneRevision=0,lastStudioFrame=null;','let suppliedSubjectAsset=null;')}
${between(lighting,'  let renderCache=null;','  const $ = id => doc.getElementById(id);')}
function studioRenderState(){if(!renderCache)renderCache={studio:{}};return renderCache.studio;}
function displayFrame(){
${between(source,'    const lighting=studioRenderState("studio");','    const mode=$("preview-motion").value')}
lastStudioFrame={revision:studioSceneRevision,geometry:lastGeometry,lighting,model:[0,0,0]};
return studioSceneRevision;
}
const sceneEditedMaterials=new Set(),ids={actualMaterial:3},selected=()=>3,selectedSceneGroup=()=>"stage";
const studioLighting={invalidateRenderState},photo={sceneChanged:()=>materialSignals++},liveRay={sceneChanged:()=>materialSignals++};
${between(source,'function markSceneMaterialEdited() {','function refreshSceneMaterialTargets()')}
`,context);
const frame=()=>vm.runInContext('displayFrame()',context),revision=()=>vm.runInContext('studioSceneRevision',context);
const event=(kind,id,toolbox=false)=>{for(const fn of listeners.get(kind)||[])fn({type:kind,target:{id,closest:s=>s==='#camera-toolbox'&&toolbox?{}:null}});};
assert.equal(frame(),0);assert.equal(frame(),0,'Unchanged displayed frame keeps authored revision');
for(const id of ['studio-table-width','room-scale','light-strength-left','key-filter-left','backdrop-layers','scene-room-engine','albedo']) {
  const before=revision();event('input',id);assert(revision()>before,id+' authoring event');const after=frame();assert.equal(frame(),after,id+' unchanged subsequent frame');
}
for(const callback of ['onChange','onRoomChange','onBackdropChange','onGoboChange']) {
  const before=revision();vm.runInContext(callback+'()',context);assert.equal(revision(),before,'Programmatic callback waits for displayed identity');
  assert.equal(frame(),before+1,callback+' changes displayed lighting reference');
}
const beforeMaterial=revision();vm.runInContext('markSceneMaterialEdited()',context);assert.equal(frame(),beforeMaterial+1);
assert.equal(vm.runInContext('sceneEditedMaterials.has("actualMaterial")',context),true);
const beforeGeometry=revision();vm.runInContext('lastGeometry={}',context);assert.equal(frame(),beforeGeometry+1);
const beforeTool=revision();for(const kind of ['input','change','click'])event(kind,'camera-range-hit',true);
event('click','view');event('click','photo-image');assert.equal(frame(),beforeTool,'Pointer inspection is not scene authoring');
event('studio-lens-change','lens-event');assert(revision()>beforeTool,'Explicit lens-change event remains conservative');frame();
// Incoming actual game-body frames affect only the return store, not authoring.
const I=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],environment={buildingIdentity:'room',environmentAssetIdentity:'scene',
  environmentRevision:revision(),environmentSequence:1,buildingPlacement:I,unitM:.3048};
const store=createMechanicalCreatureStudioReturnStore({epoch:'game',environment,appearanceAssetIdentities:['ammo'],fixedQuaternionXYZW:[0,0,0,1]});
const beforeReturn=revision();
for(let n=0;n<4;n++) {
  assert(store.receive({schema:'mechanical-creature-studio-state-v1',epoch:'game',dtAuthority:'original-viewport-slice',
    frameTimeMs:100+n*16,frameDtSeconds:.016,workerSequence:10+n,geometryRevision:2,workerReady:true,
    environments:[{buildingIdentity:'room',assetIdentity:'scene',revision:environment.environmentRevision,sequence:1,geometryRevision:2,colliderPublication:'posted-to-original-worker'}],
    camera:null,bodies:[{identity:'body',appearanceAssetIdentity:'ammo',ammunitionItemIdentity:'item',positionM:[n,1,2],velocityMPerS:[1,0,0],
      radiusM:.11,sleeping:false,pickup:false,status:'active',physicsQuaternionXYZW:null,membership:{slot:0,generation:1}}]}).accepted);
  assert.equal(frame(),beforeReturn,'Receiving/mapping a moving game body must not re-author Studio');
}
assert.equal(store.read().bodies[0].positionM[0],3);store.dispose();
// The authored revision intentionally is not a continuous preview-pose clock.
vm.runInContext('lastStudioFrame.model[0]=123',context);assert.equal(frame(),beforeReturn);
console.log(JSON.stringify({pass:true,actualRevisionAndLightingHooks:true,roomLightMaterialChangesObserved:true,
  unchangedFramesStable:true,gameBodyReturnDoesNotReauthor:true,pointerInspectionExcluded:true,
  continuousPreviewPoseNotAnAuthoredRevision:true,browserTested:false}));
