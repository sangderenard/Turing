import {STUDIO_LIGHTS} from './marble_light_strengths.js';
import {DEFAULT_LIGHT_STAND_MATERIAL,lightStandLayout,lightStandFaces} from './marble_light_stands.js';
import {lightCeilingSupportLayout,lightCeilingSupportFaces} from './marble_light_ceiling_support.js';
import {wideLightSupportRequired,wideLightSupportLayout,wideLightSupportFaces} from './marble_wide_light_supports.js';

// A owns this saved record. B uses the normal Room leaf overrides, not another
// stand controller or a second material table. These are the only editable leaves.
export const LIGHT_STAND_PROFILE_PATHS = Object.freeze(['lightStands/enabled','lightStands/material','lightStands/upperSupport','lightStands/wideSupports','lightStands/backdropSupport']);
export function defaultLightStands() {return {version:1,enabled:true,material:DEFAULT_LIGHT_STAND_MATERIAL,upperSupport:'ceiling',wideSupports:true,backdropSupport:true};}
function standState(value) {
  if(value===undefined)return {...defaultLightStands(),enabled:false,upperSupport:'floor',wideSupports:false,backdropSupport:false};
  if(!value||value.version!==1||typeof value.enabled!=='boolean'||typeof value.material!=='string'||!value.material.trim()||
      (value.upperSupport!==undefined&&!['floor','ceiling'].includes(value.upperSupport))||
      ['wideSupports','backdropSupport'].some(k=>value[k]!==undefined&&typeof value[k]!=='boolean'))
    throw new Error('Invalid light stand settings');
  // Existing saved rigs retain their floor support until explicitly changed.
  return {version:1,enabled:value.enabled,material:value.material,upperSupport:value.upperSupport??'floor',
    wideSupports:value.wideSupports??false,backdropSupport:value.backdropSupport??false};
}
export function validateLightStands(value,materialNames) {
  const state=standState(value),names=new Set(materialNames);
  // A disabled legacy rig has no stand material dependency. Keep its explicit
  // default name so save/load is stable before that catalogue row is exported.
  // Enabling it must validate the real row; no felt/glossy fallback is allowed.
  if(!names.has(state.material)&&(state.enabled||state.material!==DEFAULT_LIGHT_STAND_MATERIAL))
    throw new Error(`Light stand material "${state.material}" is unavailable. Choose an installed catalogue material or disable the light supports.`);
  return state;
}
export function setLightStandMaterial(value,material,materialNames) {
  if(!new Set(materialNames).has(material))throw new Error(`Light stand material "${material}" is unavailable in the catalogue.`);
  return validateLightStands({...standState(value),material},materialNames);
}

function standEntries(state,roles,floorY,{bounds,fixtureVertices,unitMm}={}) {
  if(!Array.isArray(roles))throw new Error('Light stand fixtures must be an array');
  const seen=new Set();
  return roles.map((role,index)=>{
    const lightId=STUDIO_LIGHTS[role?.roleIndex]?.id;
    if(!lightId||seen.has(lightId))throw new Error('Each light stand requires one existing fixture identity');
    seen.add(lightId);
    try {
      const options={bounds,fixtureVertices:fixtureVertices?.(role,index),unitMm};
      const ceiling=lightId==='fill-upper'&&state.upperSupport==='ceiling';
      const wide=!ceiling&&state.wideSupports&&wideLightSupportRequired(role,unitMm);
      const layout=ceiling?lightCeilingSupportLayout(role,bounds?.ceilingY,options):wide?wideLightSupportLayout(role,floorY,options):lightStandLayout(role,floorY,options);
      return {role,lightId,options,layout,ceiling,wide};
    } catch(error) {return {role,lightId,error:error.message};}
  });
}
/** Advisory geometry result with exact fixture IDs. An empty list means the
 * tested placements fit the room, not that table/skirt/other collisions pass. */
export function lightStandPlacementWarnings(value,roles,floorY,options={}) {
  const state=standState(value);if(!state.enabled)return [];
  return standEntries(state,roles,floorY,options).flatMap(entry=>entry.error||!entry.layout.available?
    [{lightId:entry.lightId,message:entry.error??entry.layout.reason}]:[]);
}

/** Same catalogue-ID geometry for Phong and the existing photo room-material
 * remapper. All layouts are checked before returning faces or part callbacks. */
export function studioLightStandFaces(value,roles,ids,floorY,{onPart,...options}={}) {
  const state=validateLightStands(value,Object.keys(ids));
  if(!state.enabled)return [];
  const material=ids[state.material];
  if(!Number.isSafeInteger(material)||material<0)throw new Error(`Light stand material "${state.material}" has no valid catalogue ID.`);
  const entries=standEntries(state,roles,floorY,options);
  const errors=entries.flatMap(entry=>entry.error||!entry.layout.available?
    [`${entry.lightId}: ${entry.error??entry.layout.reason}`]:[]);
  if(errors.length)throw new Error('Light supports cannot be built. '+errors.join(' '));
  const faces=[],parts=[];
  for(const {role,lightId,options:entryOptions,ceiling,wide}of entries) {
    const bodyId=`light-${ceiling?'ceiling-support':'stand'}-${lightId}`,components=[];
    const part=(ceiling?lightCeilingSupportFaces:wide?wideLightSupportFaces:lightStandFaces)(role,ceiling?entryOptions.bounds.ceilingY:floorY,material,component=>components.push({...component,bodyId}),entryOptions);
    faces.push(...part);
    parts.push({id:lightId,bodyId,construction:ceiling?'ceiling-light-support':wide?'dual-floor-light-stand':'floor-light-stand',closed:false,convex:false,
      components,componentGroups:[],contactOwnership:'shared-body-union-required',faces:part});
  }
  parts.forEach(part=>onPart?.(part));
  return faces;
}

export function createLightStandControls(container,materialNames,{onChange=()=>{},validateChange=()=>{},initialState=defaultLightStands()}={}) {
  const doc=container.ownerDocument,names=[...new Set(materialNames)],element=doc.createElement('details');
  element.id='light-stand-controls';
  const title=doc.createElement('summary');title.textContent='A (exposure): light supports';element.append(title);
  const help=doc.createElement('p');
  help.textContent='Pipe masts and three feet support the floor lights. The upper softbox can hang from real ceiling cables. All supports share the selected catalogue material. Room B can override each setting independently. Room fitting does not certify table, skirt, subject or inter-fixture collisions, or mounting strength.';
  element.append(help);
  const enabledLabel=doc.createElement('label'),enabled=doc.createElement('input');
  enabled.id='light-stands-enabled';enabled.type='checkbox';enabledLabel.textContent='Automatic light supports ';enabledLabel.append(enabled);element.append(enabledLabel);
  const materialLabel=doc.createElement('label'),material=doc.createElement('select');
  material.id='light-stands-material';materialLabel.textContent='Shared stand material ';materialLabel.append(material);element.append(materialLabel);
  const upperLabel=doc.createElement('label'),upper=doc.createElement('select');upper.id='light-stands-upper-support';upperLabel.textContent='Upper softbox support ';
  for(const [value,text]of [['ceiling','Ceiling cables'],['floor','Floor stand']]){const option=doc.createElement('option');option.value=value;option.textContent=text;upper.append(option);}
  upperLabel.append(upper);element.append(upperLabel);
  const flags={};
  for(const [key,text]of [['wideSupports','Support both sides of floor-mounted boxes at least 4 feet wide (not a load rating)'],['backdropSupport','Backdrop holder: two tripods and crossbar']]){
    const label=doc.createElement('label'),input=doc.createElement('input');input.type='checkbox';input.id='light-stands-'+key;label.textContent=text+' ';label.append(input);element.append(label);flags[key]=input;
    input.addEventListener('change',()=>{try{change({...state,[key]:input.checked});}catch{/* Status explains the rejected edit. */}});
  }
  const status=doc.createElement('p');status.id='light-stands-status';status.setAttribute('role','status');status.setAttribute('aria-live','polite');element.append(status);
  container.append(element);
  let state=standState(initialState),lastError='',warnings=[];
  // A fresh rig can display the missing default explicitly while the user picks
  // a real material or disables it. Capture/render validation still rejects the
  // unavailable enabled state; construction never chooses another row silently.
  const sync=()=>{
    enabled.checked=state.enabled;upper.value=state.upperSupport;for(const [key,input]of Object.entries(flags))input.checked=state[key];material.replaceChildren();
    for(const name of names.includes(state.material)?names:[state.material,...names]) {
      const option=doc.createElement('option');option.value=name;
      option.textContent=names.includes(name)?name.replaceAll('_',' '):`${name} (unavailable)`;material.append(option);
    }
    material.value=state.material;
    let materialError='';try{validateLightStands(state,names);}catch(error){materialError=error.message;}
    status.textContent=[lastError,materialError,...warnings.map(w=>`${w.lightId}: ${w.message}`)].filter((v,i,a)=>v&&a.indexOf(v)===i).join(' ');
  };
  function change(next) {
    try {
      next=validateLightStands(next,names);validateChange({...next},{...state});
      state=next;lastError='';warnings=[];sync();onChange();return {...state};
    } catch(error) {lastError=error.message;sync();throw error;}
  }
  const setMaterial=name=>change(setLightStandMaterial(state,name,names));
  const setEnabled=value=>change({...state,enabled:value});
  enabled.addEventListener('change',()=>{try{setEnabled(enabled.checked);}catch{/* Status explains the rejected edit. */}});
  material.addEventListener('change',()=>{try{setMaterial(material.value);}catch(error){lastError=error.message;sync();}});
  upper.addEventListener('change',()=>{try{change({...state,upperSupport:upper.value});}catch{/* Status explains the rejected edit. */}});
  sync();
  return {element,ownsControl:id=>['light-stands-enabled','light-stands-material','light-stands-upper-support','light-stands-wideSupports','light-stands-backdropSupport'].includes(id),getState:()=>({...state}),
    setEnabled,setMaterial,
    setState(value) {const next=validateLightStands(value,names);state=next;lastError='';warnings=[];sync();},
    setPlacementWarnings(value) {
      if(!Array.isArray(value)||value.some(w=>typeof w?.lightId!=='string'||typeof w.message!=='string'))throw new Error('Invalid floor stand placement warnings');
      warnings=value.map(w=>({...w}));sync();
    },sync};
}
