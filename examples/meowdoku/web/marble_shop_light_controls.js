import {shopLightVerticalPlacement,SHOP_LED_LAYOUTS,SHOP_EMISSION_PROFILES,shopLightProfileKey,SHOP_LIGHT_MODELS,MAX_SHOP_LIGHTS,createShopLight,defaultShopLights,validateShopLights} from './marble_shop_lights.js';

const MODEL_LABELS={'8ft-2tube':'8 ft / 2 tubes','4ft-4tube':'4 ft / 4 tubes','4ft-2tube':'4 ft / 2 tubes','2ft-2tube':'2 ft / 2 tubes'};

/** Edits the shop module's canonical records, never renderer/material copies.
 * Callbacks are synchronous: validateChange(next, previous) runs before commit;
 * onChange(next, previous) runs with the new state visible through state().
 * Throwing or returning false rejects an edit and restores this editor's state.
 * Callback arguments are detached snapshots. setState validates without notifying.
 */
export function createShopLightControls(container,materialNames,{getUnitMm=()=>304.8,getRoomBounds=()=>null,onChange=()=>{},validateChange=()=>{}}={}){
 const names=[...new Set(materialNames)],valid=value=>validateShopLights(value,names);
 let current=valid(defaultShopLights(getUnitMm())),serial=1,renderedIds='';
 const doc=container.ownerDocument,root=doc.createElement('section'),views=new Map(),owned=new Set(),usedIds=new Set();
 root.id='shop-light-controls';
 const title=doc.createElement('h3');title.textContent='Ceiling shop lights';
 const help=doc.createElement('p');help.textContent='Nominal physical fixture sizes in feet. X/Z use Studio units; changing scale does not move saved poses. Source level uses the existing Studio emission-strength units, not calibrated watts or lumens. Effective emission is source level × relative dimmer × spectral filter. A is the exposure rig; inspect B (room) for preview overrides. Phong uses the same fixture registration as the Studio lights; native emission uses the full mesh.';
 const photos=doc.createElement('p');photos.id='shop-light-photo-status';
 const status=doc.createElement('p');status.id='shop-light-status';status.setAttribute('role','status');
 const list=doc.createElement('div');list.id='shop-light-fixtures';
 const additions=doc.createElement('div'),addButtons=[];
 root.append(title,help);

 const labelFor=(parent,id,text,tag='input')=>{
  const label=doc.createElement('label'),input=doc.createElement(tag);
  label.textContent=text+' ';input.id=id;label.append(input);parent.append(label);owned.add(id);return input;
 };
 const option=(select,value,text,disabled=false)=>{
  const item=doc.createElement('option');item.value=value;item.textContent=text;item.disabled=disabled;select.append(item);
 };
 const materials={};
 for(const [key,label]of [['backingMaterial','Reflective backing material'],['topMaterial','Housing and mounts material']]){
  const select=labelFor(root,'shop-light-'+key,label,'select');materials[key]=select;
  select.addEventListener('change',()=>edit(next=>{next[key]=select.value;}));
 }
 for(const model of Object.keys(SHOP_LIGHT_MODELS)){
  const button=doc.createElement('button');button.type='button';button.id='shop-light-add-'+model;
  button.textContent='Add '+MODEL_LABELS[model];owned.add(button.id);additions.append(button);addButtons.push(button);
  button.addEventListener('click',()=>edit(next=>{
   if(next.fixtures.length>=MAX_SHOP_LIGHTS)throw Error(`At most ${MAX_SHOP_LIGHTS} ceiling shop lights are supported. Remove one before adding another.`);
   while(usedIds.has('shop-'+serial))serial++;
   next.fixtures.push(createShopLight('shop-'+serial++,model));
  }));
 }
 root.append(photos,additions,list,status);container.append(root);

 function makeView(fixture){
  const panel=doc.createElement('fieldset'),legend=doc.createElement('legend'),fields=[];
  panel.id='shop-light-fixture-'+fixture.id;panel.append(legend);
  const field=(key,label,{type='number',min,max,step='.01',choices,read=f=>f[key],write=(f,value)=>{f[key]=value;}}={})=>{
   const input=labelFor(panel,`shop-light-input-${fixture.id}-${key}`,label,choices?'select':'input');
   if(choices)for(const [value,text]of choices)option(input,value,text);
   else {input.type=type;if(type==='number'){input.min=String(min);input.max=String(max);input.step=step;}}
   fields.push({input,read});
   input.addEventListener('change',()=>edit(next=>{
    const changed=next.fixtures.find(f=>f.id===fixture.id);
    if(!changed)throw Error(`Shop light ${fixture.id} was removed.`);
    const value=choices?input.value:type==='checkbox'?input.checked:input.value.trim()===''?NaN:Number(input.value);
    write(changed,value);
   }));
  };
  field('enabled','Enabled in Studio',{type:'checkbox'});
  field('model','Fixture model',{choices:Object.keys(SHOP_LIGHT_MODELS).map(model=>[model,MODEL_LABELS[model]])});
  field('rackBus','Track bus',{choices:[['','Unassigned'],...['upper-wall','middle-wall','edge','near-edge','center'].map(id=>[id,id])],read:f=>f.rackBus??'',write:(f,value)=>{if(value)f.rackBus=value;else delete f.rackBus;}});
  field('source','Light source',{choices:[['tubes','Tubes'],['led-array','LED array']],write:(f,value)=>{f.source=value;f.emitterProfile=(value==='led-array'?'led_':'fluorescent_')+(f.emitterProfile?.includes('warm')?'warm_white':'cool_white');}});
  field('ledLayout','LED packing / color groups',{choices:Object.entries(SHOP_LED_LAYOUTS)});
  field('emitterProfile','Emission profile (single-color layout)',{choices:Object.entries(SHOP_EMISSION_PROFILES),read:shopLightProfileKey});
  field('x','X (Studio units)',{min:-1000,max:1000});
  field('z','Z (Studio units)',{min:-1000,max:1000});
  field('yaw','Yaw (degrees)',{min:-180,max:180,step:'1'});
  field('ceilingDropFeet','Drop below ceiling (feet)',{min:0});
  const height=doc.createElement('output');height.id='shop-light-height-'+fixture.id;panel.append(height);
  field('sourceLevel','Source level (Studio emission units)',{min:0,max:400,step:'.1'});
  field('dimmer','Relative dimmer',{min:0,max:1});
  field('continuousExposure','Continuous light in ray exposures (override)',{type:'checkbox',
   read:f=>f.exposureMode==='continuous',write:(f,value)=>{f.exposureMode=value?'continuous':'off';}});
  for(const [index,channel]of ['R','G','B'].entries())field('filter-'+channel.toLowerCase(),'Filter '+channel,
   {min:0,max:1,read:f=>f.filter[index],write:(f,value)=>{f.filter[index]=value;}});
  const remove=doc.createElement('button');remove.type='button';remove.id='shop-light-remove-'+fixture.id;
  remove.textContent='Remove fixture';owned.add(remove.id);panel.append(remove);
  remove.addEventListener('click',()=>edit(next=>{next.fixtures=next.fixtures.filter(f=>f.id!==fixture.id);}));
  return {panel,legend,fields,remove,height};
 }

 function refreshPlacement(){
  const bounds=getRoomBounds();if(!bounds)return;
  for(const fixture of current.fixtures){const view=views.get(fixture.id);if(!view)continue;
   const placement=shopLightVerticalPlacement(fixture,bounds,getUnitMm());
   const input=view.fields.find(v=>v.input.id==='shop-light-input-'+fixture.id+'-ceilingDropFeet')?.input;
   if(input)input.max=String(placement.maxDropFeet);
   view.height.textContent=`Lowest fixture surface: Y ${placement.lowestY.toFixed(3)} Studio units; ${(placement.lowestY-bounds.floorY).toFixed(3)} above floor. Table top is Y -1. Ceiling Y ${bounds.ceilingY.toFixed(3)}. Floor-limited drop: ${placement.maxDropFeet.toFixed(2)} ft.`;
  }
 }
 function sync(){
  for(const [key,select]of Object.entries(materials)){
   select.replaceChildren();for(const name of names)option(select,name,name);
   if(!names.includes(current[key]))option(select,current[key],current[key]+' (unavailable)',true);
   select.value=current[key];
  }
  const active=new Set(current.fixtures.map(f=>f.id));
  for(const [id,view]of views)if(!active.has(id)){
   for(const {input}of view.fields)owned.delete(input.id);
   owned.delete(view.remove.id);views.delete(id);
  }
  for(const fixture of current.fixtures){
   usedIds.add(fixture.id);
   if(!views.has(fixture.id))views.set(fixture.id,makeView(fixture));
   const view=views.get(fixture.id);view.legend.textContent=fixture.id+' · '+MODEL_LABELS[fixture.model];
   for(const {input,read}of view.fields){const value=read(fixture);if(input.type==='checkbox')input.checked=value;else input.value=String(value);}
  }
  refreshPlacement();
  const nextIds=JSON.stringify(current.fixtures.map(f=>f.id));
  if(nextIds!==renderedIds){list.replaceChildren(...current.fixtures.map(f=>views.get(f.id).panel));renderedIds=nextIds;}
  for(const button of addButtons)button.disabled=current.fixtures.length>=MAX_SHOP_LIGHTS;
  photos.textContent=current.fixtures.some(f=>f.exposureMode==='burst')?
   'Saved shutter-burst request is unavailable. Photos require the original timing-packet integration; the saved request has not been changed.':
   current.fixtures.some(f=>f.exposureMode==='continuous')?
   'Photos: continuous light override selected. Enabled fixtures use their A source level, dimmer and spectral filter throughout exposure; this is not a timed flash. Shutter burst remains unavailable.':
   'Photos: off. Enable a continuous light override below to illuminate ray exposures without flash timing. Shutter burst remains unavailable.';
 }

 function check(next,before){
  if(validateChange(valid(next),valid(before))===false)throw Error('The shop-light change was rejected.');
 }
 function edit(change){
  const before=current;
  try{
   let next=valid(before);change(next);next=valid(next);check(next,before);
   current=next;
   if(onChange(valid(next),valid(before))===false)throw Error('The shop-light change was rejected.');
   sync();status.textContent='';return true;
  }catch(error){current=before;sync();status.textContent=error.message;return false;}
 }
 function setState(value){
  const before=current;
  try{const next=valid(value);check(next,before);current=next;sync();status.textContent='';}
  catch(error){current=before;sync();status.textContent=error.message;throw error;}
 }
 sync();
 return {element:root,refreshPlacement,state:()=>valid(current),setState,ownsControl:id=>owned.has(id)};
}
