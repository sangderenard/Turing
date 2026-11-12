// Agent short circuit: settings -> the existing Studio builders -> a scene
// blob or the existing ray upload blob. No page, renderer, shader, DOM or job
// admission lives here. Only the authorized host can turn a payload into a job.
import {buildStudioScene,studioSceneMaterials} from './studio_scene_builder.js';
import {prepareObjectSnapshot} from './marble_object_snapshot.js';
import {registerObjAsset} from './marble_imported_object.js';
import {packMarblePhoto,OBJECT_CENTER,validatePhotoFocus} from './marble_photo.js';
import {validateStudioLighting} from './marble_studio_lighting.js';
import {studioLightingFields} from './studio_lighting_fields.js';
import {STUDIO_LIGHTS} from './marble_light_strengths.js';

export const STUDIO_AGENT_SCHEMA='turing.studio-agent-request.v1';
const object=(value,label)=>{
  if(!value||typeof value!=='object'||Array.isArray(value))throw Error(label+' must be an object');
  return value;
};
function fields(value,names,label){
  object(value,label);
  if(Object.keys(value).some(key=>!names.includes(key)))throw Error('Unknown '+label+' field');
}
const finiteJSON=(value)=>JSON.stringify(value,(_key,item)=>{
  if(typeof item==='number'&&!Number.isFinite(item))throw Error('Studio scene contains a nonfinite value');
  return item;
});

export async function buildStudioAgentProduct(request,{assets,defaults}={}){
  fields(request,['schema','operation','settings','photo','objAssets'],'Studio agent request');
  if(request.schema!==STUDIO_AGENT_SCHEMA)throw Error('Unsupported Studio agent schema');
  if(!['scene','ray-payload'].includes(request.operation))throw Error('Choose scene or ray-payload');
  const settings=structuredClone(request.settings??defaults);
  fields(settings,['object','lighting','camera'],'Studio settings');
  object(settings.object,'Studio object');object(settings.lighting,'Studio lighting');
  const camera=validatePhotoFocus(object(settings.camera,'Studio camera'));
  const materialNames=Object.keys(assets.material_index);
  settings.lighting=validateStudioLighting(settings.lighting,studioLightingFields(materialNames),materialNames);
  // Supplied OBJ bytes must already be present. No arbitrary URL fetch or
  // account lookup occurs while resolving an agent's detached contract.
  if(request.objAssets!==undefined&&!Array.isArray(request.objAssets))throw Error('objAssets must be an array');
  for(const asset of request.objAssets??[])registerObjAsset(asset);
  if(settings.object.design?.['base-shape']==='obj'&&!(request.objAssets??[]).some(asset=>asset.digest===settings.object.design.source_asset))
    throw Error('Supply the referenced existing OBJ asset in objAssets');
  const prepared=await prepareObjectSnapshot(settings.object);
  const scene=buildStudioScene({lighting:settings.lighting,objectSnapshot:prepared.value,
    materials:studioSceneMaterials(assets,prepared.value,settings.lighting),camera,rayOriginM:OBJECT_CENTER,
    shopEmissionProfiles:assets.emitterProfiles,timedExposure:request.operation==='ray-payload'});
  finiteJSON(scene);
  if(request.operation==='scene'){
    if(request.photo!==undefined)throw Error('Photo options require ray-payload');
    return {contentType:'application/json',bytes:new TextEncoder().encode(finiteJSON({...scene,geometry:prepared.geometry}))};
  }
  const photo=request.photo??{};
  fields(photo,['resolution','width','height','seconds','samplesPerOutputPixel','orbitFrames','cameraPoses','transport','gamma','maxBounces','bounceMode','triangleBudget','sphereCoverage','filmResponse','tiled','oversampling','receiveLightField',
    'exposureSubsteps','exposureIso','highlightClipPercent','lightTimeline','cameraTriggerGraph','cameraTriggerTiming'],'photo');
  const seconds=photo.seconds??30,substeps=photo.exposureSubsteps??Math.ceil(seconds*60);
  if(!Number.isFinite(seconds)||seconds<=0||!Number.isInteger(substeps)||substeps<Math.ceil(seconds*10)||substeps>216000)
    throw Error('Exposure substeps must give a model tick of at most 0.1 seconds.');
  const base=scene.fixtures;
  const timedEligible=(photo.transport??'fixed:3')==='fixed:3' && (photo.orbitFrames??1)===1 &&
    !photo.tiled && !photo.samplesPerOutputPixel && !photo.width && !photo.height && !photo.cameraPoses && !photo.receiveLightField &&
    (photo.oversampling??1)===1;
  base.emissive_material_ids=[...new Set([...Array.from({length:STUDIO_LIGHTS.length},(_,i)=>i),
    ...(base.emitter_profiles?.records??[]).map(record=>record.material)])];
  const timeline=photo.lightTimeline??[];
  if(!Array.isArray(timeline)||timeline.length>4095)throw Error('lightTimeline must contain at most 4095 Studio changes');
  if(!timedEligible && (timeline.length||base.trigger_emitters?.length))
    throw Error('Timed lights require one ordinary untiled RGB Studio still');
  const states=[{tick:0,fixtures:base}];
  for(const entry of timeline){
    fields(entry,['tick','lighting'],'lightTimeline entry');
    if(!Number.isInteger(entry.tick)||entry.tick<=states[states.length-1].tick||entry.tick>=substeps)
      throw Error('lightTimeline ticks must increase within the shutter');
    const lighting=validateStudioLighting(entry.lighting,studioLightingFields(materialNames),materialNames);
    const next=buildStudioScene({lighting,objectSnapshot:prepared.value,
      materials:studioSceneMaterials(assets,prepared.value,lighting),camera,rayOriginM:OBJECT_CENTER,
      shopEmissionProfiles:assets.emitterProfiles,timedExposure:true}).fixtures;
    if(JSON.stringify([next.vertices,next.material_ids])!==JSON.stringify([base.vertices,base.material_ids]))
      throw Error('lightTimeline changed fixture geometry');
    states.push({tick:entry.tick,fixtures:next});
  }
  const exposureStateFrames=states.map((item,i)=>({t0:item.tick*seconds/substeps,
    t1:states[i+1]?states[i+1].tick*seconds/substeps:seconds,pbr:item.fixtures.pbr,
    emitter_profiles:item.fixtures.emitter_profiles??{version:1,records:[]}}));
  const {lightTimeline:ignoredTimeline,...photoOptions}=photo;
  const bytes=new Uint8Array(packMarblePhoto(prepared.geometry,null,{...photoOptions,...scene,
    ...(timedEligible?{exposureStateFrames}:{}),objectSnapshot:prepared.value}));
  return {contentType:'application/octet-stream',bytes};
}
