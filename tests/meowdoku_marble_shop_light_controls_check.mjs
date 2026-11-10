import assert from 'node:assert/strict';
import {createShopLightControls} from '../examples/meowdoku/web/marble_shop_light_controls.js';
import {defaultShopLights,validateShopLights,MAX_SHOP_LIGHTS} from '../examples/meowdoku/web/marble_shop_lights.js';

class Element{
 constructor(tag,doc){this.tagName=tag;this.ownerDocument=doc;this.children=[];this.events={};this.value='';this.textContent='';this.checked=false;this.disabled=false;this.replacements=0;}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=nodes;this.replacements++;}
 setAttribute(key,value){this[key]=value;}
 addEventListener(kind,callback){(this.events[kind]??=[]).push(callback);}
 fire(kind){for(const callback of this.events[kind]??[])callback();}
}
const walk=node=>[node,...node.children.flatMap(walk)];
const names=['metal_silver','black_matte_paint','metal_gold'];
function setup(options={}){
 const doc={createElement:tag=>new Element(tag,doc)},container=doc.createElement('div');
 const controls=createShopLightControls(container,names,options);
 const find=id=>walk(container).find(node=>node.id===id);
 const change=(id,value)=>{const input=find(id);assert.ok(input,id);if(input.type==='checkbox')input.checked=value;else input.value=String(value);input.fire('change');};
 return {controls,container,find,change};
}
const calls=[],checks=[];
let rejectValidation=false,rejectNotification=false,returnFalse=false,controls;
const ui=setup({getUnitMm:()=>609.6,
 validateChange(next,previous){checks.push([structuredClone(next),structuredClone(previous)]);if(rejectValidation)throw Error('Fixture does not fit the room');next.fixtures.length=0;previous.fixtures.length=0;},
 onChange(next,previous){assert.deepEqual(controls.state(),next);calls.push([structuredClone(next),structuredClone(previous)]);next.fixtures.length=0;previous.fixtures.length=0;if(rejectNotification)throw Error('Scene update refused');if(returnFalse)return false;},
});
controls=ui.controls;
const initial=defaultShopLights(609.6),fixed=['shop-light-backingMaterial','shop-light-topMaterial','shop-light-add-8ft-2tube','shop-light-add-4ft-4tube','shop-light-status'];
const fixedNodes=fixed.map(ui.find);
assert.deepEqual(controls.state(),initial);assert.equal(calls.length,0);assert.equal(checks.length,0);
assert.equal(controls.element,ui.find('shop-light-controls'));assert.match(ui.find('shop-light-photo-status').textContent,/Photos: off/);
assert.equal(walk(ui.container).some(e=>/exposureMode|burst/.test(e.id??'')),false,'No dummy shutter-burst control');
const detached=controls.state();detached.fixtures[0].filter[0]=0;detached.fixtures.pop();assert.deepEqual(controls.state(),initial);
const other=setup();other.change('shop-light-input-shop-long-x',33);assert.deepEqual(controls.state(),initial,'Editors do not share defaults');
const firstInput=ui.find('shop-light-input-shop-long-x'),sibling=structuredClone(initial.fixtures[1]);
for(const [key,value]of [['x',-4.25],['z',2.5],['yaw',90],['ceilingDropFeet',1.25],['dimmer',.4],['filter-r',.1],['filter-g',.2],['filter-b',.3],['source','led-array'],['model','4ft-4tube'],['enabled',false]])ui.change('shop-light-input-shop-long-'+key,value);
assert.deepEqual(controls.state().fixtures[0],{...initial.fixtures[0],x:-4.25,z:2.5,yaw:90,ceilingDropFeet:1.25,dimmer:.4,filter:[.1,.2,.3],source:'led-array',emitterProfile:'led_cool_white',model:'4ft-4tube',enabled:false});
assert.deepEqual(controls.state().fixtures[1],sibling);assert.equal(ui.find('shop-light-input-shop-long-x'),firstInput,'Field nodes stay stable during edits');
assert.equal(ui.find('shop-light-fixtures').replacements,1,'Ordinary edits do not detach the focused fixture fieldset');
assert.equal(calls.length,11);assert.deepEqual(calls[0][1],initial);assert.equal(calls[0][0].fixtures[0].x,-4.25);
ui.change('shop-light-backingMaterial','metal_gold');ui.change('shop-light-topMaterial','metal_silver');
assert.equal(controls.state().backingMaterial,'metal_gold');assert.equal(controls.state().topMaterial,'metal_silver');
const accepted=controls.state(),count=calls.length;
for(const [key,value]of [['x',''],['x','not a number'],['z',1001],['yaw',181],['ceilingDropFeet',-1],['dimmer',1.01],['filter-r',-.1],['source','bulb'],['model','unknown']]){
 ui.change('shop-light-input-shop-long-'+key,value);assert.deepEqual(controls.state(),accepted);assert.equal(calls.length,count);assert.ok(ui.find('shop-light-status').textContent);
}
ui.change('shop-light-backingMaterial','missing');assert.deepEqual(controls.state(),accepted);assert.equal(ui.find('shop-light-backingMaterial').value,'metal_gold');
rejectValidation=true;ui.change('shop-light-input-shop-long-x',8);assert.deepEqual(controls.state(),accepted);assert.equal(calls.length,count);assert.match(ui.find('shop-light-status').textContent,/does not fit/);rejectValidation=false;
rejectNotification=true;ui.change('shop-light-input-shop-long-x',8);assert.deepEqual(controls.state(),accepted);assert.equal(firstInput.value,'-4.25');assert.match(ui.find('shop-light-status').textContent,/Scene update refused/);
ui.find('shop-light-remove-shop-long').fire('click');assert.deepEqual(controls.state(),accepted);assert.equal(ui.find('shop-light-input-shop-long-x'),firstInput,'Rejected removal retains the fixture editor');
ui.find('shop-light-add-8ft-2tube').fire('click');assert.deepEqual(controls.state(),accepted);assert.equal(ui.find('shop-light-fixtures').children.length,2);rejectNotification=false;
returnFalse=true;ui.change('shop-light-input-shop-long-x',8);assert.deepEqual(controls.state(),accepted);assert.match(ui.find('shop-light-status').textContent,/rejected/);returnFalse=false;
const saved=structuredClone(accepted);saved.fixtures[0].x=19;const beforeLoadCount=calls.length;controls.setState(saved);saved.fixtures[0].x=200;assert.equal(controls.state().fixtures[0].x,19);assert.equal(calls.length,beforeLoadCount);
const loaded=controls.state();
for(const bad of [null,{...loaded,version:2},{...loaded,backingMaterial:'missing'},{...loaded,fixtures:[loaded.fixtures[0],loaded.fixtures[0]]}]){
 assert.throws(()=>controls.setState(bad));assert.deepEqual(controls.state(),loaded);assert.equal(ui.find('shop-light-input-shop-long-x').value,'19');
}
rejectValidation=true;assert.throws(()=>controls.setState(undefined),/does not fit/);assert.deepEqual(controls.state(),loaded);rejectValidation=false;
controls.setState(undefined);assert.deepEqual(controls.state(),validateShopLights(undefined,names));assert.equal(ui.find('shop-light-fixtures').children.length,0);assert.equal(calls.length,beforeLoadCount);assert.equal(controls.ownsControl('shop-light-input-shop-long-x'),false);
for(let i=0;i<MAX_SHOP_LIGHTS;i++)ui.find('shop-light-add-'+(i%2?'4ft-4tube':'8ft-2tube')).fire('click');
const full=controls.state(),ids=full.fixtures.map(f=>f.id);assert.equal(full.fixtures.length,8);assert.equal(new Set(ids).size,8);assert.equal(full.fixtures[1].model,'4ft-4tube');
assert.equal(ui.find('shop-light-add-8ft-2tube').disabled,true);assert.equal(ui.find('shop-light-add-4ft-4tube').disabled,true);
ui.find('shop-light-add-4ft-4tube').fire('click');assert.deepEqual(controls.state(),full);assert.match(ui.find('shop-light-status').textContent,/At most 8/);
ui.find('shop-light-remove-'+ids[2]).fire('click');assert.equal(controls.state().fixtures.length,7);assert.equal(ui.find('shop-light-add-8ft-2tube').disabled,false);assert.equal(controls.ownsControl('shop-light-input-'+ids[2]+'-x'),false);
assert.equal(calls.at(-1)[1].fixtures.length,8);assert.equal(calls.at(-1)[0].fixtures.length,7,'Parent gets both snapshots to prune removed overrides');
ui.find('shop-light-add-8ft-2tube').fire('click');assert.equal(controls.state().fixtures.length,8);assert.ok(!ids.includes(controls.state().fixtures.at(-1).id),'Removed fixture ID is not reused');
const repeat=controls.state();repeat.fixtures=repeat.fixtures.slice(0,1);repeat.fixtures[0].id='shop-100';controls.setState(repeat);ui.find('shop-light-add-4ft-4tube').fire('click');assert.equal(new Set(controls.state().fixtures.map(f=>f.id)).size,2);
const burst=controls.state();burst.fixtures[0].exposureMode='burst';controls.setState(burst);assert.equal(controls.state().fixtures[0].exposureMode,'burst');assert.match(ui.find('shop-light-photo-status').textContent,/Saved shutter-burst request is unavailable/);
assert.deepEqual(fixed.map(ui.find),fixedNodes,'Fixed control nodes and IDs survive lifecycle edits');
const allIds=walk(ui.container).map(e=>e.id).filter(Boolean);assert.equal(new Set(allIds).size,allIds.length);assert.equal(controls.ownsControl('shop-light-backingMaterial'),true);
const unusual=defaultShopLights();unusual.fixtures[0].id='remove';unusual.fixtures[1].id='model';controls.setState(unusual);
const unusualIds=walk(ui.container).map(e=>e.id).filter(Boolean);assert.equal(new Set(unusualIds).size,unusualIds.length,'Every valid fixture ID has a separate control namespace');
const blocked=setup({validateChange:()=>false});const blockedBefore=blocked.controls.state();blocked.change('shop-light-input-shop-long-x',9);assert.deepEqual(blocked.controls.state(),blockedBefore);assert.throws(()=>blocked.controls.setState(undefined),/rejected/);
const override=setup(),overrideId='shop-light-input-shop-long-continuousExposure';
assert.equal(override.find(overrideId).checked,false);
override.change(overrideId,true);
assert.equal(override.controls.state().fixtures[0].exposureMode,'continuous');
assert.equal(override.controls.state().fixtures[1].exposureMode,'off');
assert.match(override.find('shop-light-photo-status').textContent,/not a timed flash/);
const savedOverride=JSON.parse(JSON.stringify(override.controls.state()));
override.controls.setState(savedOverride);assert.equal(override.find(overrideId).checked,true);
override.change(overrideId,false);assert.equal(override.controls.state().fixtures[0].exposureMode,'off');
blocked.change(overrideId,true);assert.deepEqual(blocked.controls.state(),blockedBefore);
assert.equal(blocked.find(overrideId).checked,false,'Rejected override rolls back checkbox');
console.log('Shop-light controls: defaults/physical scale, canonical fields, detached snapshots, validation/callback rollback, material choices, legacy restore, stable controls and IDs, add/remove/cap, unavailable burst and callback pruning snapshots passed');


const levelUI=setup();levelUI.change('shop-light-input-shop-long-sourceLevel',80);levelUI.change('shop-light-input-shop-long-dimmer',.25);assert.equal(levelUI.controls.state().fixtures[0].sourceLevel,80);assert.equal(levelUI.controls.state().fixtures[0].dimmer,.25);levelUI.change('shop-light-input-shop-long-sourceLevel',401);assert.equal(levelUI.controls.state().fixtures[0].sourceLevel,80);console.log('PASS source level control and invalid-value rollback');
{
 const {shopLightVerticalPlacement,studioShopLightGeometry,shopLightEmitterAllocations}=await import('../examples/meowdoku/web/marble_shop_lights.js');
 const bounds={leftX:-24,rightX:24,backZ:-18.6,frontZ:24,floorY:-5,ceilingY:23.1};
 const u=setup({getUnitMm:()=>304.8,getRoomBounds:()=>bounds,validateChange:next=>studioShopLightGeometry(next,{metal_silver:0,black_matte_paint:1,metal_gold:2},shopLightEmitterAllocations(next,3),{bounds,unitMm:304.8})});
 const id=u.controls.state().fixtures[0].id,field='shop-light-input-'+id+'-ceilingDropFeet';
 assert.ok(Number(u.find(field).max)>27);u.change(field,23.7);assert.equal(u.controls.state().fixtures[0].ceilingDropFeet,23.7);
 const state=u.controls.state(),geometry=studioShopLightGeometry(state,{metal_silver:0,black_matte_paint:1,metal_gold:2},shopLightEmitterAllocations(state,3),{bounds,unitMm:304.8});
 const part=geometry.parts[0],lowest=Math.min(...part.faces.flatMap(f=>f.slice(0,3).map(p=>p[1])));
 assert.ok(Math.abs(lowest-shopLightVerticalPlacement(state.fixtures[0],bounds,304.8).lowestY)<1e-10);
 assert.match(u.find('shop-light-height-'+id).textContent,/Y -0.873/);
 for(const face of part.faces)for(const n of face[4])assert.ok(Math.abs(Math.hypot(...n)-1)<1e-10);
 const before=u.controls.state();u.change(field,29);assert.deepEqual(u.controls.state(),before);assert.match(u.find('shop-light-status').textContent,/does not fit/);
 bounds.ceilingY=24;u.controls.refreshPlacement();assert.ok(Number(u.find(field).max)>28);
 console.log('PASS floor-owned shop drop above10ft, actual mesh/height readout equality, unit normals, floor rejection/rollback and dynamic room bound.');
}
