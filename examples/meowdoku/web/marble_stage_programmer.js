// The programmer is real Studio furniture. Its four raised keys are also the
// hit targets in the Phong view; the same faces enter photo scene packing.
import {shopLightBoxFaces} from './marble_shop_lights.js';
import {rangefinderRay} from './marble_rangefinder.js';
import {LIGHTING_BUSES} from './marble_lighting_racks.js';

export const PROGRAMMER_KEYS=Object.freeze(['open','flash','all-out','shop']);
export const PROGRAMMER_FADERS=Object.freeze(LIGHTING_BUSES.map(bus=>bus.id));
export function stageProgrammerLayout(bounds,table) {
  const x=Math.min(bounds.rightX-1.05,Math.max(3,table.enabled?table.width/2+1.05:3));
  const z=Math.min(bounds.frontZ-.72,Math.max(2.8,table.enabled?table.length/2+1.05:2.8));
  const top=bounds.floorY+.88;
  return {x,z,top,buttonY:top+.18};
}
export function stageProgrammerFaces(bounds,table,ids,onPart) {
  const {x,z,top,buttonY}=stageProgrammerLayout(bounds,table),faces=[];
  const add=(id,low,high,material)=>{
    const part=shopLightBoxFaces(low,high,material,material,0);
    faces.push(...part);onPart?.({id,bodyId:'stage-programmer',construction:'cuboid',closed:true,convex:true,faces:part});
  };
  const frame=ids.metal_steel,deck=ids.black_matte_paint,accent=ids.metal_aluminum;
  add('desktop',[x-.85,top-.09,z-.43],[x+.85,top,z+.43],frame);
  for(const dx of [-.72,.72])for(const dz of [-.31,.31])
    add(`leg-${dx}-${dz}`,[x+dx-.055,bounds.floorY,z+dz-.055],[x+dx+.055,top-.09,z+dz+.055],frame);
  add('console',[x-.72,top,z-.34],[x+.72,top+.15,z+.34],deck);
  for(let i=0;i<PROGRAMMER_KEYS.length;i++){
    const cx=x-.51+i*.34;
    add(`key-${PROGRAMMER_KEYS[i]}`,[cx-.105,top+.15,z+.06],[cx+.105,buttonY,z+.27],i===1?accent:frame);
  }
  for(let i=0;i<PROGRAMMER_FADERS.length;i++){
    const cx=x-.55+i*.275;
    add(`fader-${i}`,[cx-.025,top+.15,z-.24],[cx+.025,top+.165,z-.02],accent);
  }
  return faces;
}

export function pickStageProgrammer(frame,bounds,table,u,v) {
  const {ray}=rangefinderRay({state:{schema:'turing.studio-visible-scene.v1',channel:'studio',
    camera:{view:frame.view,fovDegrees:frame.fovDegrees,aspect:frame.aspect},
    frame:{rayOriginM:[0,0,0],unitM:1}}},u,v);
  const {x,z,top,buttonY}=stageProgrammerLayout(bounds,table);
  const intersect=y=>{const t=(y-ray[1])/ray[5];return t>0&&Number.isFinite(t)?[ray[0]+t*ray[4],ray[2]+t*ray[6]]:null;};
  const key=intersect(buttonY);
  if(key&&key[1]>=z+.03&&key[1]<=z+.30){
    for(let i=0;i<PROGRAMMER_KEYS.length;i++)if(Math.abs(key[0]-(x-.51+i*.34))<=.14)return PROGRAMMER_KEYS[i];
  }
  const fader=intersect(top+.165);
  if(fader&&fader[1]>=z-.27&&fader[1]<=z+.01)
    for(let i=0;i<PROGRAMMER_FADERS.length;i++)if(Math.abs(fader[0]-(x-.55+i*.275))<=.09)
      return {bus:PROGRAMMER_FADERS[i],level:Math.max(0,Math.min(1,(fader[1]-(z-.24))/.22))};
  const panel=intersect(top+.15);
  return panel&&Math.abs(panel[0]-x)<=.72&&Math.abs(panel[1]-z)<=.34?'open':null;
}
