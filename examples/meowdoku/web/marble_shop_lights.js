import {writeStudioFixtureRows} from './marble_studio_scene.js';

// Nominal shop fixtures. Sizes are authored feet, not measured product specs.
// Tube length is the real emitting span; the housing adds 3 inches overall.
export const SHOP_LIGHT_MODELS=Object.freeze({
  '8ft-2tube':Object.freeze({lengthFeet:8,tubes:2,widthFeet:.75}),
  '4ft-4tube':Object.freeze({lengthFeet:4,tubes:4,widthFeet:1.25}),
});
export const SHOP_LIGHT_MATERIALS=Object.freeze({backingMaterial:'metal_silver',topMaterial:'black_matte_paint'});
export const MAX_SHOP_LIGHTS=8;
export const SHOP_LED_LAYOUTS=Object.freeze({legacy:'Original 32 emitters',dense:'Dense / selected profile','dense-mixed':'Dense / alternating cool and warm'});
const mixed=f=>f.source==='led-array'&&f.ledLayout==='dense-mixed';
const sourceForRow=(f,row)=>mixed(f)?{...f,emitterProfile:row.secondary?'led_warm_white':'led_cool_white'}:f;
export const SHOP_EMISSION_PROFILES=Object.freeze({fluorescent_cool_white:'Fluorescent cool white (4100 K)',fluorescent_warm_white:'Fluorescent warm white (2900 K)',led_cool_white:'LED cool white',led_warm_white:'LED warm white'});
export function shopLightProfileKey(fixture){return fixture.emitterProfile??(fixture.source==='led-array'?'led_cool_white':'fluorescent_cool_white');}
export function shopLightEmissionProfile(fixture,profiles){
  const key=shopLightProfileKey(fixture),record=(profiles?.records??[]).find(r=>r.key===key);
  if(!record||!Array.isArray(record.preview_rgb)||record.preview_rgb.length!==3||!record.preview_rgb.every(v=>Number.isFinite(v)&&v>=0)||!Array.isArray(record.filter_rgb_matrix)||record.filter_rgb_matrix.length!==3||!record.filter_rgb_matrix.every(row=>Array.isArray(row)&&row.length===3&&row.every(Number.isFinite))||!/^[a-f0-9]{64}$/i.test(record.revision??''))throw Error(`Missing canonical emission profile: ${key}`);
  return record;
}
export function shopLightEmissionRecords(value,firstRow,materialIndex,profiles,{channel='exposure'}={}){
  const state=shopState(value);
  return {version:1,records:shopLightEmitterAllocations(state,firstRow).flatMap(row=>{
    const f=state.fixtures.find(f=>f.id===row.fixtureId);
    const material=materialIndex.get(row.materialId);if(material===undefined)return [];
    const record=shopLightEmissionProfile(sourceForRow(f,row),profiles);
    return [{material,profile:record.key,revision:record.revision,gain:f.enabled&&(channel==='studio'||f.exposureMode==='continuous')?f.sourceLevel*f.dimmer:0,filter:[...f.filter]}];
  })};
}
const SIDES=12,FEET_MM=304.8;
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const index=value=>Number.isSafeInteger(value)&&value>=0;
function feetToUnits(unitMm) {
  if(!finite(unitMm)||unitMm<=0)throw new RangeError('Shop lights need a positive Studio unit in millimetres');
  return FEET_MM/unitMm;
}
export function createShopLight(id,model,{x=0,z=0,yaw=0}={}) {
  return {id,model,enabled:true,source:'tubes',ledLayout:'dense-mixed',emitterProfile:'fluorescent_cool_white',x,z,yaw,ceilingDropFeet:.5,sourceLevel:1,dimmer:1,filter:[1,1,1],exposureMode:'off'};
}
export function defaultShopLights(unitMm=FEET_MM) {
  const scale=feetToUnits(unitMm);
  return {version:1,...SHOP_LIGHT_MATERIALS,fixtures:[
    createShopLight('shop-long','8ft-2tube',{x:-6*scale}),createShopLight('shop-wide','4ft-4tube',{x:6*scale})]};
}
function shopState(value) {
  if(value===undefined)return {version:1,...SHOP_LIGHT_MATERIALS,fixtures:[]};
  if(!value||value.version!==1||!Array.isArray(value.fixtures)||value.fixtures.length>MAX_SHOP_LIGHTS||
     !['backingMaterial','topMaterial'].every(k=>typeof value[k]==='string'&&value[k].trim()))
    throw new Error('Invalid ceiling shop-light settings');
  const seen=new Set();
  const fixtures=value.fixtures.map(f=>{
    if(!f||typeof f.id!=='string'||!/^[a-zA-Z0-9-]{1,80}$/.test(f.id)||seen.has(f.id))throw new Error('Shop lights need unique stable fixture IDs');
    seen.add(f.id);
    if(!Object.hasOwn(SHOP_LIGHT_MODELS,f.model)||typeof f.enabled!=='boolean'||!['tubes','led-array'].includes(f.source)||
       !['off','burst','continuous'].includes(f.exposureMode))throw new Error(`${f.id}: invalid shop-light model, source, visibility or exposure mode`);
    if(![f.x,f.z,f.yaw,f.ceilingDropFeet,f.dimmer].every(finite)||Math.abs(f.x)>1000||Math.abs(f.z)>1000||
       Math.abs(f.yaw)>180||f.ceilingDropFeet<0||f.dimmer<0||f.dimmer>1||
       !Array.isArray(f.filter)||f.filter.length!==3||!f.filter.every(v=>finite(v)&&v>=0&&v<=1))
      throw new Error(`${f.id}: invalid shop-light position, ceiling drop, dimmer or filter`);
    const ledLayout=f.ledLayout??'legacy';
    if(!Object.hasOwn(SHOP_LED_LAYOUTS,ledLayout))throw Error(`${f.id}: unknown LED layout`);
    const sourceLevel=f.sourceLevel??1;
    if(!finite(sourceLevel)||sourceLevel<0||sourceLevel>400)throw Error(`${f.id}: source level must be between 0 and 400 Studio emission units`);
    if(f.emitterProfile!==undefined&&!Object.hasOwn(SHOP_EMISSION_PROFILES,f.emitterProfile))throw Error(`${f.id}: unknown emission profile`);
    return {ledLayout,emitterProfile:shopLightProfileKey(f),id:f.id,model:f.model,enabled:f.enabled,source:f.source,x:f.x,z:f.z,yaw:f.yaw,
      ceilingDropFeet:f.ceilingDropFeet,sourceLevel,dimmer:f.dimmer,filter:[...f.filter],exposureMode:f.exposureMode};
  });
  return {version:1,backingMaterial:value.backingMaterial,topMaterial:value.topMaterial,fixtures};
}
export function validateShopLights(value,materialNames) {
  const state=shopState(value),names=new Set(materialNames);
  for(const key of ['backingMaterial','topMaterial'])if(!names.has(state[key])&&
      (state.fixtures.some(f=>f.enabled)||state[key]!==SHOP_LIGHT_MATERIALS[key]))
    throw new Error(`Shop-light ${key} "${state[key]}" is unavailable in the shared material catalogue`);
  return state;
}
export function setShopLightSource(value,id,source) {
  const state=shopState(value),fixture=state.fixtures.find(f=>f.id===id);
  if(!fixture)throw new Error(`Unknown shop light: ${id}`);
  fixture.source=source;fixture.emitterProfile=(source==='led-array'?'led_':'fluorescent_')+(fixture.emitterProfile?.includes('warm')?'warm_white':'cool_white');return shopState(state);
}
export function shopLightEmitterAllocations(value,firstRow) {
  if(!index(firstRow))throw new Error('Shop fixtures need a valid first shared material row');
  return shopState(value).fixtures.flatMap((f,i)=>{
    const row={id:f.id,fixtureId:f.id,materialId:firstRow+i,groupId:firstRow+i,exposureMode:f.exposureMode,source:f.source};
    return mixed(f)?[row,{...row,id:f.id+':warm',secondary:true,materialId:firstRow+MAX_SHOP_LIGHTS+i,groupId:firstRow+MAX_SHOP_LIGHTS+i}]:[row];
  });
}

/** Rows append after the caller's existing catalogue/runtime rows. They use the
 * same central emitter writer as Studio fixtures; this module invents no PBR
 * coefficients or lumen calibration. Disabled fixtures retain their row index.
 * Burst exposure remains unavailable until the native timing owner supplies
 * its real packet integration; no constant averaged gain stands in for it. */
export function writeShopLightEmitterRows(data,stride,firstRow,value,{channel='studio',profiles}={}) {
  const state=shopState(value),allocations=shopLightEmitterAllocations(state,firstRow);
  if(!['studio','exposure'].includes(channel))throw new Error('Shop-light emission needs an explicit Studio/exposure channel');
  if(!index(firstRow)||!Number.isSafeInteger(stride)||stride<16||!data||!index(data.length)||
      data.length%stride||(allocations.length&&data.length<(1+Math.max(...allocations.map(r=>r.materialId)))*stride))
    throw new Error('Allocate distinct shared material rows for every shop fixture before writing emission');
  if(channel==='exposure')for(const f of state.fixtures)if(f.enabled&&f.dimmer>0&&f.exposureMode==='burst')
    throw new Error(`${f.id}: shutter burst requires the authoritative timing-packet integration; no steady flash substitute is available`);
  // An explicit continuous exposure is the same shared emissive material,
  // not a timed flash or an averaged substitute for the unavailable burst.
  const resolved=allocations.map(row=>{
    const f=state.fixtures.find(f=>f.id===row.fixtureId),profile=shopLightEmissionProfile(sourceForRow(f,row),profiles);
    const color=profile.filter_rgb_matrix.map(r=>Math.max(0,r.reduce((sum,v,i)=>sum+v*f.filter[i],0)));
    const strength=f.enabled&&(channel==='studio'||f.exposureMode==='continuous')?f.sourceLevel*f.dimmer:0;
    return {row,color,strength};
  });
  for(const {row,color,strength}of resolved)writeStudioFixtureRows(data,stride,row.materialId,[color],[strength]);
  return resolved.map(({row,strength})=>({...row,relativeIntensity:strength}));
}

const add=(a,b)=>a.map((v,i)=>v+b[i]);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
function boxFaces(low,high,side,bottom,group) {
  const p=(x,y,z)=>[x?high[0]:low[0],y?high[1]:low[1],z?high[2]:low[2]],faces=[];
  for(const [corners,n]of [
    [[[0,0,0],[0,0,1],[0,1,1],[0,1,0]],[-1,0,0]],[[[1,0,1],[1,0,0],[1,1,0],[1,1,1]],[1,0,0]],
    [[[0,0,1],[0,0,0],[1,0,0],[1,0,1]],[0,-1,0]],[[[0,1,0],[0,1,1],[1,1,1],[1,1,0]],[0,1,0]],
    [[[1,0,0],[0,0,0],[0,1,0],[1,1,0]],[0,0,-1]],[[[0,0,1],[1,0,1],[1,1,1],[0,1,1]],[0,0,1]],
  ]) {
    const [a,b,c,d]=corners.map(v=>p(...v)),material=n[1]===-1?bottom:side;
    faces.push([a,b,c,material,[n,n,n],group],[a,c,d,material,[n,n,n],group]);
  }
  return faces;
}
// Round surfaces are physical triangle strips, not decorative tube labels or
// point-light proxies. Cached ring coordinates close the seam exactly.
function tubeFaces(center,length,radius,emitter,cap,group) {
  const normals=Array.from({length:SIDES},(_,i)=>[0,Math.cos(i*2*Math.PI/SIDES),Math.sin(i*2*Math.PI/SIDES)]);
  const left=add(center,[-length/2,0,0]),right=add(center,[length/2,0,0]);
  const a=normals.map(n=>add(left,n.map(v=>v*radius))),b=normals.map(n=>add(right,n.map(v=>v*radius))),faces=[];
  for(let i=0;i<SIDES;i++) {
    const j=(i+1)%SIDES,n0=normals[i],n1=normals[j];
    faces.push([a[i],a[j],b[j],emitter,[n0,n1,n1],group],[a[i],b[j],b[i],emitter,[n0,n1,n0],group],
      [left,a[j],a[i],cap,[[-1,0,0],[-1,0,0],[-1,0,0]],group],
      [right,b[i],b[j],cap,[[1,0,0],[1,0,0],[1,0,0]],group]);
  }
  return faces;
}
function ledFaces(center,radius,height,emitter,body,group) {
  const normals=Array.from({length:SIDES},(_,i)=>[Math.cos(i*2*Math.PI/SIDES),0,-Math.sin(i*2*Math.PI/SIDES)]);
  const bottom=add(center,[0,-height,0]),a=normals.map(n=>add(bottom,n.map(v=>v*radius))),b=normals.map(n=>add(center,n.map(v=>v*radius))),faces=[];
  for(let i=0;i<SIDES;i++) {
    const j=(i+1)%SIDES,n0=normals[i],n1=normals[j];
    faces.push([a[i],a[j],b[j],body,[n0,n1,n1],group],[a[i],b[j],b[i],body,[n0,n1,n0],group],
      [bottom,a[j],a[i],emitter,[[0,-1,0],[0,-1,0],[0,-1,0]],group],
      [center,b[i],b[j],body,[[0,1,0],[0,1,0],[0,1,0]],group]);
  }
  return faces;
}
function roomBounds(bounds) {
  if(!bounds||!['leftX','rightX','backZ','frontZ','floorY','ceilingY'].every(k=>finite(bounds[k]))||
      bounds.leftX>=bounds.rightX||bounds.backZ>=bounds.frontZ||bounds.floorY>=bounds.ceilingY)
    throw new Error('Shop fixtures require the real room bounds and ceiling');
}

/** Returns one physical mesh plus exact emitter triangle ownership. Material
 * IDs are caller allocations in the shared table, not a private coefficient
 * table. Body rows may be shared; emitter rows must be distinct per fixture. */
export function shopLightVerticalPlacement(fixture,bounds,unitMm){
 roomBounds(bounds);const scale=feetToUnits(unitMm);
 const bodyTop=bounds.ceilingY-fixture.ceilingDropFeet*scale,bodyBottom=bodyTop-.15*scale;
 const below=fixture.source==='tubes'?(2/24+.04):.025;
 return {bodyTop,bodyBottom,lowestY:bodyBottom-below*scale,
  maxDropFeet:Math.max(0,(bounds.ceilingY-bounds.floorY)/scale-.15-below)};
}
export function studioShopLightGeometry(value,ids,emitterRows,{bounds,unitMm}={}) {
  const state=validateShopLights(value,Object.keys(ids));
  if(!state.fixtures.some(f=>f.enabled))return {faces:[],parts:[],emitters:[]};
  roomBounds(bounds);const scale=feetToUnits(unitMm);
  const backing=ids[state.backingMaterial],top=ids[state.topMaterial];
  if(![backing,top].every(index))throw new Error('Shop fixture backing/top need real shared catalogue IDs');
  if(!Array.isArray(emitterRows))throw new Error('Shop fixtures require explicit shared emitter row allocations');
  const rows=new Map(),usedMaterials=new Set(),usedGroups=new Set();
  for(const row of emitterRows) {
    if(!row||typeof row.id!=='string'||rows.has(row.id)||!index(row.materialId)||!index(row.groupId)||
       usedMaterials.has(row.materialId)||usedGroups.has(row.groupId)||Object.values(ids).includes(row.materialId))
      throw new Error('Shop emitter rows must be unique and separate from authored catalogue rows');
    rows.set(row.id,row);usedMaterials.add(row.materialId);usedGroups.add(row.groupId);
  }
  const faces=[],parts=[],emitters=[];
  for(const fixture of state.fixtures) {
    if(!fixture.enabled)continue;
    const row=rows.get(fixture.id);if(!row)throw new Error(`${fixture.id}: missing shared emitter row`);
    const spec=SHOP_LIGHT_MODELS[fixture.model],length=spec.lengthFeet*scale,width=spec.widthFeet*scale;
    const {bodyTop,bodyBottom}=shopLightVerticalPlacement(fixture,bounds,unitMm);
    const theta=fixture.yaw*Math.PI/180,c=Math.cos(theta),s=Math.sin(theta);
    const rotate=([x,y,z])=>[c*x+s*z,y,-s*x+c*z],point=p=>add(rotate(p),[fixture.x,0,fixture.z]);
    const partFaces=[],components=[],bodyId=`shop-${fixture.id}`;
    const append=(id,local,construction)=>{
      const firstTriangle=partFaces.length;
      partFaces.push(...local.map(([a,b,c,material,normals,group])=>[point(a),point(b),point(c),material,normals.map(rotate),group]));
      components.push({id,bodyId,firstTriangle,triangleCount:local.length,sourceConstruction:construction,contactAdmission:'unverified'});
    };
    append('housing',boxFaces([-length/2-.125*scale,bodyBottom,-width/2],[length/2+.125*scale,bodyTop,width/2],top,backing,row.groupId),'closed-shop-housing');
    if(fixture.ceilingDropFeet>0)for(const sign of [-1,1]) {
      const x=sign*length*.3,r=.02*scale;
      append(`mount-${sign}`,boxFaces([x-r,bodyTop,-r],[x+r,bounds.ceilingY,r],top,top,row.groupId),'ceiling-support');
    }
    if(fixture.source==='led-array'&&fixture.ledLayout!=='legacy'){
      // Nominal one-inch pitch, 0.4-inch circular emitting faces. Real HR672/NL660
      // panels motivate the packed pattern; these long housings are not replicas.
      const cols=spec.lengthFeet*12,lines=Math.floor(spec.widthFeet*12)-1;
      for(let i=0;i<lines;i++)for(let j=0;j<cols;j++){
        const ledRow=mixed(fixture)&&(i+j)%2?rows.get(fixture.id+':warm'):row;
        if(!ledRow)throw Error(`${fixture.id}: missing warm LED material group`);
        append(`led-${i}-${j}`,ledFaces([-length/2+(j+.5)*length/cols,bodyBottom,-width/2+(i+1)*width/(lines+1)],scale/60,scale*.025,ledRow.materialId,top,ledRow.groupId),'emissive-led-package');
      }
    }else for(let i=0;i<spec.tubes;i++) {
      const z=-width/2+(i+1)*width/(spec.tubes+1);
      if(fixture.source==='tubes')append(`tube-${i}`,tubeFaces([0,bodyBottom-(1/24+.04)*scale,z],length,scale/24,row.materialId,top,row.groupId),'emissive-capped-tube');
      else {
        const count=spec.lengthFeet*2;
        for(let j=0;j<count;j++)append(`led-${i}-${j}`,ledFaces([-length/2+(j+.5)*length/count,bodyBottom,z],scale/24,scale*.025,row.materialId,top,row.groupId),'emissive-led-package');
      }
    }
    for(const face of partFaces)for(const p of face.slice(0,3))if(p[0]<bounds.leftX||p[0]>bounds.rightX||p[1]<bounds.floorY||p[1]>bounds.ceilingY||p[2]<bounds.backZ||p[2]>bounds.frontZ)
      throw new Error(`${fixture.id}: ${fixture.model} ceiling fixture does not fit this room at ${unitMm} mm/unit. Move it, reduce its ceiling drop, or enlarge the room; saved scale and poses were not changed.`);
    const firstTriangle=faces.length;
    const fixtureRows=mixed(fixture)?[row,rows.get(fixture.id+':warm')]:[row];
    for(const emitterRow of fixtureRows){
    const emittingTriangles=[];
    let surfaceArea=0;const centroid=[0,0,0];
    partFaces.forEach((face,i)=>{if(face[3]===emitterRow.materialId){emittingTriangles.push(firstTriangle+i);const area=Math.hypot(...cross(sub(face[1],face[0]),sub(face[2],face[0])))/2;surfaceArea+=area;for(let axis=0;axis<3;axis++)centroid[axis]+=area*(face[0][axis]+face[1][axis]+face[2][axis])/3;}});
    emitters.push({id:emitterRow.id,materialId:emitterRow.materialId,groupId:emitterRow.groupId,triangles:emittingTriangles,
      position:centroid.map(v=>v/surfaceArea),surfaceAreaStudio:surfaceArea,surfaceAreaM2:surfaceArea*(unitMm/1000)**2,source:fixture.source,exposureMode:fixture.exposureMode});
    }
    faces.push(...partFaces);
    parts.push({id:fixture.id,bodyId,construction:'ceiling-shop-fixture',closed:false,convex:false,components,
      contactOwnership:'shared-body-union-required',faces:partFaces});
  }
  return {faces,parts,emitters};
}

