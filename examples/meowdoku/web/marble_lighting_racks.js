import {createShopLight,SHOP_LIGHT_MODELS,MAX_SHOP_LIGHTS,shopLightBoxFaces} from './marble_shop_lights.js';

// Rack dimensions are nominal authored feet, matching the shop fixture module.
// No emission, material tables, or transport are implemented here.
export const LIGHTING_BUSES=Object.freeze([
  {id:'upper-wall',label:'Upper wall rack'}, {id:'middle-wall',label:'Middle wall rack'},
  {id:'edge',label:'Outer perimeter rack'}, {id:'near-edge',label:'Inner perimeter rack'},
  {id:'center',label:'Sliding center crane'},
].map(Object.freeze));
export function defaultLightingRacks(){return {version:1,buses:LIGHTING_BUSES.map(({id})=>({id,enabled:false,dropFeet:null,dimmer:null})),shopDimmer:.15,houseDimmer:1,allOut:false,solo:null,craneX:0,craneZ:0};}
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const dimmer=v=>v===null||(finite(v)&&v>=0&&v<=1);
export function validateLightingRacks(value){
  if(value===undefined)return defaultLightingRacks();
  if(!value||value.version!==1||!Array.isArray(value.buses)||value.buses.length!==LIGHTING_BUSES.length||!dimmer(value.shopDimmer)||!dimmer(value.houseDimmer)||typeof value.allOut!=='boolean'||!['craneX','craneZ'].every(k=>finite(value[k]??0)&&Math.abs(value[k]??0)<=1)||!(value.solo===null||LIGHTING_BUSES.some(b=>b.id===value.solo)||value.solo==='house'||value.solo==='shop'))throw Error('Invalid lighting rack controls');
  const seen=new Set();const buses=value.buses.map(b=>{
    if(!b||!LIGHTING_BUSES.some(v=>v.id===b.id)||seen.has(b.id)||typeof b.enabled!=='boolean'||!(b.dropFeet===null||(finite(b.dropFeet)&&b.dropFeet>=0))||!dimmer(b.dimmer))throw Error('Invalid lighting rack bus');
    seen.add(b.id);return {id:b.id,enabled:b.enabled,dropFeet:b.dropFeet,dimmer:b.dimmer};
  });return {version:1,buses,shopDimmer:value.shopDimmer,houseDimmer:value.houseDimmer,allOut:value.allOut,solo:value.solo,craneX:value.craneX??0,craneZ:value.craneZ??0};
}
function dimensions(bounds,unitMm){
  if(!bounds||!['leftX','rightX','backZ','frontZ','floorY','ceilingY'].every(k=>finite(bounds[k]))||bounds.leftX>=bounds.rightX||bounds.backZ>=bounds.frontZ||bounds.floorY>=bounds.ceilingY||!finite(unitMm)||unitMm<=0)throw Error('Racks require the real room bounds and Studio unit');
  return 304.8/unitMm;
}
function segments(bus,bounds,s,racks,forRails=false){
  const inset=({'upper-wall':1,'middle-wall':2,edge:3,'near-edge':6,center:7})[bus.id]*s;
  const x0=bounds.leftX+inset,x1=bounds.rightX-inset,z0=bounds.backZ+inset,z1=bounds.frontZ-inset;
  if(x1-x0<2.5*s||z1-z0<2.5*s)throw Error(`${bus.id}: room is too small for this rack and its physical fixtures`);
  if(bus.id==='center'){
    const center=(z0+z1)/2+racks.craneZ*Math.max(0,(z1-z0)/2-1.5*s);
    const width=x1-x0,span=forRails?width:Math.min(width,8.75*s),mid=(x0+x1)/2+racks.craneX*(width-span)/2;
    return [{id:'bar-0',a:[mid-span/2,center-s],b:[mid+span/2,center-s]},{id:'bar-1',a:[mid-span/2,center+s],b:[mid+span/2,center+s]}];
  }
  // Opposing sides stop before the perpendicular fixtures' footprint. This
  // leaves real corner clearance instead of intersecting housings.
  const c=forRails?0:s;
  return [{id:'back',a:[x0+c,z0],b:[x1-c,z0]},{id:'front',a:[x0+c,z1],b:[x1-c,z1]},
    {id:'left',a:[x0,z0+c],b:[x0,z1-c]},{id:'right',a:[x1,z0+c],b:[x1,z1-c]}];
}
function fitModules(lengthFeet,bus){
  const primary=bus.id==='upper-wall'||bus.id==='middle-wall'?'8ft-2tube':'4ft-4tube';
  const options=[primary,'4ft-2tube','2ft-2tube'].filter((v,i,a)=>a.indexOf(v)===i);
  // Six modules per side leaves capacity for individually spawned fixtures;
  // larger rooms distribute the remaining span as real space between modules.
  const models=[];let remaining=lengthFeet;
  for(const model of options){const spec=SHOP_LIGHT_MODELS[model];if(!spec)throw Error(`Missing canonical shop module ${model}`);
    const span=spec.lengthFeet+.25;
    while(remaining>=span&&models.length<6){models.push(model);remaining-=span;}
  }
  if(!models.length)throw Error(`${bus.id}: track segment cannot fit a two-foot fixture`);
  return {models,gap:remaining/(models.length+1)};
}
/** Explicit layout operation. Existing rack modules retain unique source/filter
 * settings by stable ID; ordinary authored fixtures are never moved or removed.
 * The caller chooses when to apply a new layout (including room scale changes). */
export function layoutLightingRackFixtures(shopLights,racks,{bounds,unitMm}={}){
  const state=validateLightingRacks(racks),s=dimensions(bounds,unitMm),old=new Map(shopLights.fixtures.map(f=>[f.id,f]));
  const fixtures=shopLights.fixtures.filter(f=>!(f.rackBus&&f.id.startsWith('rack-'))).map(f=>structuredClone(f));
  for(const bus of state.buses)if(bus.enabled){
    const drop=bus.dropFeet??(bus.id==='middle-wall'?(bounds.ceilingY-bounds.floorY)/s/2:.5);
    if(drop+.3>(bounds.ceilingY-bounds.floorY)/s)throw Error(`${bus.id}: chain drop puts the fixture below the floor`);
    for(const line of segments(bus,bounds,s,state)){
      const dx=line.b[0]-line.a[0],dz=line.b[1]-line.a[1],length=Math.hypot(dx,dz)/s,{models,gap}=fitModules(length,bus);
      let cursor=gap;
      for(const [i,model]of models.entries()){
        const span=SHOP_LIGHT_MODELS[model].lengthFeet+.25,offset=(cursor+span/2)/length,id=`rack-${bus.id}-${line.id}-${i}`;
        const prior=old.get(id),f=prior?structuredClone(prior):{...createShopLight(id,model),sourceLevel:400,exposureMode:'continuous'};
        Object.assign(f,{id,model,rackBus:bus.id,x:line.a[0]+dx*offset,z:line.a[1]+dz*offset,yaw:dz?90:0,ceilingDropFeet:bus.dropFeet??prior?.ceilingDropFeet??drop});
        fixtures.push(f);cursor+=span+gap;
      }
    }
  }
  if(fixtures.length>MAX_SHOP_LIGHTS)throw Error(`Rack layout needs ${fixtures.length} fixtures; the shared emitter capacity is ${MAX_SHOP_LIGHTS}. Disable a bus or enlarge the supported emitter allocation before applying it.`);
  return {...structuredClone(shopLights),fixtures};
}
/** Nonemissive track support geometry, using the shop module's box builder and
 * shared catalogue material ID. No fake light/centroid source is introduced. */
export function studioLightingRackGeometry(racks,ids,{bounds,unitMm,material='black_matte_paint',topMaterial}={}){
  const state=validateLightingRacks(racks),s=dimensions(bounds,unitMm),faces=[],parts=[];
  const mat=ids[topMaterial??material];if(!Number.isSafeInteger(mat)||mat<0)throw Error('Racks need an existing shared housing material');
  const append=(id,a,b,y)=>{const r=.04*s,low=[Math.min(a[0],b[0])-r,y-r,Math.min(a[1],b[1])-r],high=[Math.max(a[0],b[0])+r,y+r,Math.max(a[1],b[1])+r];
    const mesh=shopLightBoxFaces(low,high,mat,mat,mat);faces.push(...mesh);parts.push({id,bodyId:id,construction:'lighting-track',closed:true,convex:true,faces:mesh});};
  for(const bus of state.buses)if(bus.enabled){
    const lines=segments(bus,bounds,s,state,true),y=bounds.ceilingY-.05*s;
    for(const line of lines)append(`track-${bus.id}-${line.id}`,line.a,line.b,y);
    if(bus.id==='center')for(const [index,x]of [lines[0].a[0],lines[0].b[0]].entries())append(`track-center-runner-${index}`,[x,bounds.backZ+7*s],[x,bounds.frontZ-7*s],y-.1*s);
  }
  return {faces,parts,emitters:[]};
}

