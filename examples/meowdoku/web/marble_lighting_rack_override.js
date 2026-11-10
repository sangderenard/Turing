import {validateLightingRacks} from './marble_lighting_racks.js';
import {STUDIO_LIGHTS} from './marble_light_strengths.js';

// Resolve shortcuts over a detached A/B profile. Authored unique controls never
// change; clearing an override reveals the latest individual settings again.
export function applyLightingRackOverrides(profile,value){
  const racks=validateLightingRacks(value),out=structuredClone(profile);
  const buses=new Map(racks.buses.map(b=>[b.id,b]));
  for(const f of out.shopLights.fixtures){
    const bus=buses.get(f.rackBus);
    if(racks.shopDimmer!==null)f.dimmer=racks.shopDimmer;
    if(bus?.dimmer!==null&&bus?.dimmer!==undefined)f.dimmer=bus.dimmer;
    if(bus?.dropFeet!==null&&bus?.dropFeet!==undefined)f.ceilingDropFeet=bus.dropFeet;
    if(racks.allOut||(racks.solo&&racks.solo!=='shop'&&racks.solo!==f.rackBus))f.dimmer=0;
  }
  for(const {id,strengthId} of STUDIO_LIGHTS){
    const bus=id==='fill-upper'?'center':'house';
    let gain=racks.houseDimmer??1;
    if(bus==='center'&&buses.get('center')?.dimmer!==null)gain=buses.get('center').dimmer;
    if(racks.allOut||(racks.solo&&racks.solo!==bus))gain=0;
    out.controls[strengthId]*=gain;
  }
  return out;
}
