import {lightStandLayout,lightStandFaces} from './marble_light_stands.js';

// Four feet of emitting width is a transparent authoring threshold, not an
// engineering load rating. Old profiles opt out until the user selects it.
export const WIDE_LIGHT_SUPPORT_THRESHOLD_FEET=4;
export function wideLightSupportRequired(role,unitMm){
  if(!Number.isFinite(unitMm)||unitMm<=0)throw new RangeError('Wide-light supports require the real Studio millimetres per unit');
  return !role.round&&2*role.halfW*unitMm>=WIDE_LIGHT_SUPPORT_THRESHOLD_FEET*304.8;
}
export function wideLightSupportLayout(role,floorY,options={}){
  const length=Math.hypot(role.dir?.[0],role.dir?.[2]);
  const side=length>0?[role.dir[2]/length,0,-role.dir[0]/length]:[0,0,1];
  const layouts=[side,side.map(v=>-v)].map(direction=>lightStandLayout(role,floorY,{...options,supportDirections:[direction]}));
  const failed=layouts.findIndex(layout=>!layout.available);
  if(failed!==-1)return {available:false,segments:[],reason:`The ${failed?'opposite':'first'} side support does not fit. ${layouts[failed].reason}`};
  return {available:true,layouts,segments:layouts.flatMap((layout,i)=>layout.segments.map(segment=>({...segment,id:`side-${i}/${segment.id}`})))};
}
export function wideLightSupportFaces(role,floorY,material,onComponent,options={}){
  const layout=wideLightSupportLayout(role,floorY,options);
  if(!layout.available)throw new RangeError(layout.reason);
  const faces=[];
  layout.layouts.forEach((side,i)=>{
    const first=faces.length;
    faces.push(...lightStandFaces(role,floorY,material,component=>onComponent?.({...component,
      id:`side-${i}/${component.id}`,firstTriangle:first+component.firstTriangle}),{...options,supportDirections:[side.exclusionDirection]}));
  });
  return faces;
}
